"""The scoring interface and its input contract.

A model that cannot say why it refused an input is not deployable, so the
contract is explicit and every violation is collected before anything is
raised. Callers get the whole list rather than discovering one problem per
attempt.

The scorer builds its features with the same function the training used. That
is deliberate: a separate serving path is the usual way a feature definition
drifts between training and production without anyone noticing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .features import WARMUP, build_features
from .io_load import Bundle
from .logging_utils import get_logger

LOG = get_logger("scoring")

REQUIRED_COLUMNS = ("zone_id", "timestamp", "occupancy", "volume", "volume_11kw",
                    "duration", "e_price", "s_price")


class ContractError(ValueError):
    """Raised when an input does not satisfy the scoring contract."""

    def __init__(self, violations: Sequence[str]):
        self.violations = list(violations)
        super().__init__("; ".join(self.violations))


@dataclass
class InputContract:
    """What the scorer requires of an input frame."""

    known_zones: Sequence[int]
    capacity: Dict[int, int]
    min_history_hours: int = WARMUP
    required_columns: Sequence[str] = REQUIRED_COLUMNS
    non_negative: Sequence[str] = ("occupancy", "volume", "volume_11kw",
                                   "duration", "e_price", "s_price")

    def validate(self, frame: pd.DataFrame) -> None:
        """Raise ``ContractError`` listing every violation, or return quietly."""
        violations: List[str] = []

        missing = [c for c in self.required_columns if c not in frame.columns]
        if missing:
            raise ContractError([f"missing required columns: {sorted(missing)}"])

        if frame.empty:
            raise ContractError(["the input frame is empty"])

        if not pd.api.types.is_datetime64_any_dtype(frame["timestamp"]):
            violations.append("timestamp is not a datetime column")
            raise ContractError(violations)

        if not pd.api.types.is_integer_dtype(frame["zone_id"]):
            violations.append("zone_id is not an integer column")

        unknown = sorted(set(frame["zone_id"]) - set(self.known_zones))
        if unknown:
            violations.append(f"unknown zone ids: {unknown[:5]}")

        duplicates = int(frame.duplicated(subset=["zone_id", "timestamp"]).sum())
        if duplicates:
            violations.append(f"{duplicates} duplicate zone and timestamp pairs")

        for column in self.non_negative:
            if column in frame.columns:
                negative = int((frame[column] < 0).sum())
                if negative:
                    violations.append(f"{column} has {negative} negative values")
                nulls = int(frame[column].isna().sum())
                if nulls:
                    violations.append(f"{column} has {nulls} missing values")

        if "occupancy" in frame.columns:
            capacity = frame["zone_id"].map(self.capacity)
            over = int((frame["occupancy"] > capacity + 1e-6).sum())
            if over:
                violations.append(
                    f"occupancy exceeds installed charging points in {over} rows")

        for zone, part in frame.groupby("zone_id"):
            stamps = pd.DatetimeIndex(part["timestamp"]).sort_values()
            if len(stamps) < self.min_history_hours:
                violations.append(
                    f"zone {zone} supplies {len(stamps)} hours of history, "
                    f"{self.min_history_hours} are required")
                continue
            expected = pd.date_range(stamps[0], stamps[-1], freq="h")
            if len(expected) != len(stamps) or not (expected == stamps).all():
                violations.append(f"zone {zone} has a gap or a non hourly timestamp")

        zone_counts = frame.groupby("zone_id").size()
        if zone_counts.nunique() > 1:
            violations.append("zones supply different numbers of hours")

        if violations:
            raise ContractError(violations)


@dataclass
class Scorer:
    """Turns a validated history into forecasts at a given horizon."""

    predict: Callable[[np.ndarray], np.ndarray]
    contract: InputContract
    reference: Bundle
    target: str
    feature_names: Sequence[str] = field(default_factory=tuple)
    clip_to_capacity: bool = True

    def score(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Forecast every zone ``horizon`` hours past the last supplied hour."""
        self.contract.validate(history)
        if horizon < 1:
            raise ContractError([f"horizon must be at least 1, received {horizon}"])

        frame = history.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)
        index = pd.DatetimeIndex(sorted(frame["timestamp"].unique()))

        window = Bundle(
            long=frame,
            zones=self.reference.zones,
            weather=self.reference.weather[
                self.reference.weather["timestamp"].isin(index)].reset_index(drop=True),
            adjacency=self.reference.adjacency,
            distance=self.reference.distance,
            index=index,
        )
        features = build_features(window, self.target)

        # The origin is the last supplied hour. A design matrix is requested for
        # that origin alone, with the target column ignored because it is in the
        # future and is not observed.
        origin = len(index) - 1
        columns = []
        for name in features.origin_blocks:
            columns.append(features.origin_blocks[name][origin])
        for name in features.shared_blocks:
            columns.append(np.repeat(features.shared_blocks[name][origin],
                                     features.n_zones))
        target_time = index[-1] + pd.Timedelta(hours=horizon)
        calendar = _calendar_at(target_time, len(self.reference.index), origin)
        for name in features.calendar_blocks:
            columns.append(np.repeat(calendar[name], features.n_zones))
        for name in features.static_blocks:
            columns.append(features.static_blocks[name])

        design = np.column_stack(columns).astype(np.float32)
        if np.isnan(design).any():
            raise ContractError(["the supplied history leaves a feature undefined; "
                                 "more history is required"])

        predictions = np.asarray(self.predict(design), dtype=float)
        predictions = np.clip(predictions, 0.0, None)
        capacity = np.array([self.contract.capacity[z] for z in features.zone_ids],
                            dtype=float)
        if self.clip_to_capacity and self.target in ("occupancy", "duration"):
            predictions = np.minimum(predictions, capacity)

        return pd.DataFrame({
            "zone_id": features.zone_ids,
            "timestamp": target_time,
            "horizon": horizon,
            "target": self.target,
            "prediction": predictions,
        })


def _calendar_at(stamp: pd.Timestamp, n_total_hours: int, position: int) -> Dict[str, float]:
    """Calendar features for one target time, matching the training definitions."""
    from .config import HOLIDAYS
    holidays = {pd.Timestamp(day).date() for day in HOLIDAYS}
    return {
        "hour_sin": float(np.sin(2 * np.pi * stamp.hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * stamp.hour / 24)),
        "dow_sin": float(np.sin(2 * np.pi * stamp.dayofweek / 7)),
        "dow_cos": float(np.cos(2 * np.pi * stamp.dayofweek / 7)),
        "hour_of_day": float(stamp.hour),
        "is_weekend": 1.0 if stamp.dayofweek >= 5 else 0.0,
        "is_holiday": 1.0 if stamp.date() in holidays else 0.0,
        "trend": float(position / max(n_total_hours, 1)),
    }
