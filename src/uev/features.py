"""Feature construction.

Every feature is evaluated at or before the prediction origin. The convention is
fixed and is the thing the leakage tests check: for a target at time ``t + h``,
the origin is ``t``, and no feature may read any series at a time later than
``t``. The one exception is the calendar, which is known arbitrarily far ahead
and is therefore taken at the target time.

Weather is taken at the origin rather than at the target time. The dataset
carries observed weather, not a forecast, so using the value at the target time
would give the model information that would not exist when the forecast is
actually made. Prices are treated the same way, even though a published tariff
is arguably known ahead, because nothing is lost by the stricter choice.

The layout is deliberately array shaped rather than long shaped. Blocks are held
as ``(hours, zones)`` matrices and flattened only when a design matrix is
requested, which keeps the repeated construction across folds and horizons
cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG, HOLIDAYS, FeatureConfig
from .graph import neighbour_weights
from .io_load import Bundle
from .logging_utils import get_logger

LOG = get_logger("features")

# Offsets from the origin, in hours. Offset 0 is the origin itself, which is the
# most recent observation available when the forecast is made.
LAG_OFFSETS: Tuple[int, ...] = (0, 1, 2, 23, 24, 47, 167)
NEIGHBOUR_OFFSETS: Tuple[int, ...] = (0, 23)

# The longest lookback any feature uses. Origins earlier than this cannot carry
# a complete feature row and are excluded.
WARMUP = max(max(LAG_OFFSETS), max(CONFIG.features.roll_windows) - 1,
             max(NEIGHBOUR_OFFSETS)) + 1


def _shift_rows(matrix: np.ndarray, offset: int) -> np.ndarray:
    """Return ``matrix`` shifted down by ``offset`` rows, filled with nan.

    Row ``t`` of the result holds row ``t - offset`` of the input, so a positive
    offset only ever exposes the past.
    """
    out = np.full_like(matrix, np.nan, dtype=np.float32)
    if offset == 0:
        out[:] = matrix
    elif offset > 0:
        out[offset:] = matrix[:-offset]
    else:
        raise ValueError("a negative offset would read the future")
    return out


def _rolling(matrix: np.ndarray, window: int, how: str) -> np.ndarray:
    """Trailing rolling statistic ending at and including each row."""
    frame = pd.DataFrame(matrix)
    rolled = frame.rolling(window=window, min_periods=window)
    if how == "mean":
        out = rolled.mean()
    elif how == "std":
        out = rolled.std()
    elif how == "max":
        out = rolled.max()
    elif how == "min":
        out = rolled.min()
    else:
        raise ValueError(f"unknown statistic: {how}")
    return out.to_numpy(dtype=np.float32)


@dataclass
class FeatureSet:
    """Feature blocks for one target, plus the target matrix itself."""

    target: str
    index: pd.DatetimeIndex
    zone_ids: List[int]
    y: np.ndarray                                   # (hours, zones)
    origin_blocks: Dict[str, np.ndarray] = field(default_factory=dict)   # (hours, zones)
    shared_blocks: Dict[str, np.ndarray] = field(default_factory=dict)   # (hours,)
    calendar_blocks: Dict[str, np.ndarray] = field(default_factory=dict)  # (hours,)
    static_blocks: Dict[str, np.ndarray] = field(default_factory=dict)   # (zones,)

    @property
    def n_hours(self) -> int:
        return self.y.shape[0]

    @property
    def n_zones(self) -> int:
        return self.y.shape[1]

    def feature_names(self, include_static: bool = True) -> List[str]:
        names = list(self.origin_blocks) + list(self.shared_blocks) + list(self.calendar_blocks)
        if include_static:
            names += list(self.static_blocks)
        return names

    def valid_origins(self, horizon: int) -> np.ndarray:
        """Origins with a complete feature row and an observed target."""
        return np.arange(WARMUP - 1, self.n_hours - horizon, dtype=int)

    def design(self, origins: Sequence[int], horizon: int,
               zones: Sequence[int] | None = None,
               include_static: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Build the design matrix and target vector.

        Rows are ordered origin major then zone, so that a reshape recovers the
        ``(origins, zones)`` grid without another sort.
        """
        origins = np.asarray(origins, dtype=int)
        if origins.size and origins.max() + horizon >= self.n_hours:
            raise ValueError("an origin would require a target beyond the observed window")
        zone_positions = (np.arange(self.n_zones) if zones is None
                          else np.asarray(zones, dtype=int))

        n_rows = origins.size * zone_positions.size
        columns: List[np.ndarray] = []

        for name in self.origin_blocks:
            block = self.origin_blocks[name][np.ix_(origins, zone_positions)]
            columns.append(block.reshape(n_rows))
        for name in self.shared_blocks:
            block = self.shared_blocks[name][origins]
            columns.append(np.repeat(block, zone_positions.size))
        for name in self.calendar_blocks:
            block = self.calendar_blocks[name][origins + horizon]
            columns.append(np.repeat(block, zone_positions.size))
        if include_static:
            for name in self.static_blocks:
                block = self.static_blocks[name][zone_positions]
                columns.append(np.tile(block, origins.size))

        design = np.column_stack(columns).astype(np.float32) if columns else np.zeros((n_rows, 0),
                                                                                      np.float32)
        target = self.y[np.ix_(origins + horizon, zone_positions)].reshape(n_rows)
        return design, target


def build_features(bundle: Bundle, target: str,
                   settings: FeatureConfig | None = None) -> FeatureSet:
    """Build every feature block for one target."""
    settings = settings or CONFIG.features
    index = bundle.index
    zone_ids = list(bundle.zones["zone_id"])

    wide = bundle.long.pivot(index="timestamp", columns="zone_id", values=target)
    wide = wide.reindex(index=index, columns=zone_ids)
    y = wide.to_numpy(dtype=np.float32)

    features = FeatureSet(target=target, index=index, zone_ids=zone_ids, y=y)

    # Lags of the target, taken at or before the origin.
    for offset in LAG_OFFSETS:
        features.origin_blocks[f"lag_{offset}"] = _shift_rows(y, offset)

    # Trailing rolling statistics, each window ending at the origin.
    for window in settings.roll_windows:
        features.origin_blocks[f"roll_mean_{window}"] = _rolling(y, window, "mean")
        features.origin_blocks[f"roll_std_{window}"] = _rolling(y, window, "std")
    features.origin_blocks["roll_max_24"] = _rolling(y, 24, "max")
    features.origin_blocks["roll_min_24"] = _rolling(y, 24, "min")

    # Short differences, which carry the recent direction of travel.
    features.origin_blocks["diff_1"] = (features.origin_blocks["lag_0"]
                                        - features.origin_blocks["lag_1"])
    features.origin_blocks["diff_24"] = (features.origin_blocks["lag_0"]
                                         - features.origin_blocks["lag_24"])

    # Neighbour aggregates. Demand at adjacent zones substitutes, so the state
    # of the neighbourhood carries information the zone's own history does not.
    if settings.use_neighbours:
        weights = neighbour_weights(bundle.adjacency, bundle.distance).to_numpy(dtype=np.float32)
        neighbour_mean = np.nan_to_num(y) @ weights.T
        for offset in NEIGHBOUR_OFFSETS:
            features.origin_blocks[f"neighbour_mean_{offset}"] = _shift_rows(
                neighbour_mean.astype(np.float32), offset)

    # Prices at the origin. Nothing causal is read from these.
    if settings.use_prices:
        for name in ("e_price", "s_price"):
            price = bundle.long.pivot(index="timestamp", columns="zone_id", values=name)
            price = price.reindex(index=index, columns=zone_ids).to_numpy(dtype=np.float32)
            features.origin_blocks[f"{name}_0"] = _shift_rows(price, 0)
            features.origin_blocks[f"{name}_roll_24"] = _rolling(price, 24, "mean")

    # Weather at the origin, shared across zones. It is a lower resolution
    # covariate: consecutive hours repeat in roughly two thirds of the series.
    if settings.use_weather:
        weather = bundle.weather.set_index("timestamp").reindex(index)
        for column in ("air_temp_c_central", "humidity_pct_central",
                       "rain_category_central", "dewpoint_c_central"):
            values = weather[column].to_numpy(dtype=np.float32)
            features.shared_blocks[column] = values
        features.shared_blocks["air_temp_roll_24"] = pd.Series(
            weather["air_temp_c_central"].to_numpy(dtype=np.float32)
        ).rolling(24, min_periods=24).mean().to_numpy(dtype=np.float32)

    # Calendar, taken at the target time because it is known in advance.
    hours = index.hour.to_numpy()
    dows = index.dayofweek.to_numpy()
    holidays = {pd.Timestamp(day).date() for day in HOLIDAYS}
    features.calendar_blocks["hour_sin"] = np.sin(2 * np.pi * hours / 24).astype(np.float32)
    features.calendar_blocks["hour_cos"] = np.cos(2 * np.pi * hours / 24).astype(np.float32)
    features.calendar_blocks["dow_sin"] = np.sin(2 * np.pi * dows / 7).astype(np.float32)
    features.calendar_blocks["dow_cos"] = np.cos(2 * np.pi * dows / 7).astype(np.float32)
    features.calendar_blocks["hour_of_day"] = hours.astype(np.float32)
    features.calendar_blocks["is_weekend"] = (dows >= 5).astype(np.float32)
    features.calendar_blocks["is_holiday"] = np.array(
        [1.0 if day in holidays else 0.0 for day in index.date], dtype=np.float32)
    features.calendar_blocks["trend"] = (np.arange(len(index)) / len(index)).astype(np.float32)

    # Static zone attributes. These are the zone identity the pooled model sees.
    if settings.use_zone_attrs:
        zones = bundle.zones.set_index("zone_id").reindex(zone_ids)
        features.static_blocks["capacity_points"] = zones["capacity_points"].to_numpy(np.float32)
        features.static_blocks["n_stations"] = zones["n_stations"].to_numpy(np.float32)
        features.static_blocks["log_area"] = np.log1p(
            zones["area_m2"].to_numpy(np.float32)).astype(np.float32)
        features.static_blocks["points_density"] = zones["points_density"].to_numpy(np.float32)
        features.static_blocks["station_density"] = zones["station_density"].to_numpy(np.float32)
        features.static_blocks["shape_index"] = zones["shape_index"].to_numpy(np.float32)
        features.static_blocks["centroid_lon"] = zones["centroid_lon"].to_numpy(np.float32)
        features.static_blocks["centroid_lat"] = zones["centroid_lat"].to_numpy(np.float32)
        from .graph import degree
        features.static_blocks["n_neighbours"] = degree(bundle.adjacency).reindex(
            zone_ids).to_numpy(np.float32)

    LOG.info("built %d features for %s (%d origin, %d shared, %d calendar, %d static)",
             len(features.feature_names()), target, len(features.origin_blocks),
             len(features.shared_blocks), len(features.calendar_blocks),
             len(features.static_blocks))
    return features
