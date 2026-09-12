"""Central configuration for the study.

Values here are the knobs that define the experiment grid, the validation
scheme, the monitoring thresholds and the capacity scenarios. They are hashed
into cache keys, so changing one invalidates the affected cached artefacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Tuple

# Dataset pinning.
COMMIT = "44f2aa0c8d89f192bce00bafb0def74a21b39c68"
BASE_URL = f"https://raw.githubusercontent.com/IntelligentSystemsLab/UrbanEV/{COMMIT}/data"

DATA_FILES: Tuple[str, ...] = (
    "occupancy.csv",
    "volume.csv",
    "volume-11kW.csv",
    "duration.csv",
    "e_price.csv",
    "s_price.csv",
    "inf.csv",
    "inf_raw.csv",
    "adj.csv",
    "distance.csv",
    "weather_central.csv",
    "weather_airport.csv",
    "weather_header.txt",
)

# Timestamps in the source tables are written as 2022/9/1 0:00.
TIME_FORMAT = "%Y/%m/%d %H:%M"
EXPECTED_START = "2022-09-01 00:00"
EXPECTED_END = "2023-02-28 23:00"
EXPECTED_HOURS = 4344
EXPECTED_ZONES = 275

# Chinese public holidays inside the coverage window, used as calendar flags.
HOLIDAYS = (
    "2022-09-10", "2022-09-11", "2022-09-12",           # Mid Autumn Festival
    "2022-10-01", "2022-10-02", "2022-10-03",           # National Day
    "2022-10-04", "2022-10-05", "2022-10-06", "2022-10-07",
    "2022-12-31", "2023-01-01", "2023-01-02",           # New Year
    "2023-01-21", "2023-01-22", "2023-01-23",           # Spring Festival
    "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27",
)

SEED = 20250901

TARGETS: Tuple[str, ...] = ("occupancy", "volume", "duration")
HORIZONS: Tuple[int, ...] = (1, 6, 24)

MODEL_FAMILIES: Tuple[str, ...] = (
    "last_observation",
    "seasonal_naive_daily",
    "seasonal_naive_weekly",
    "autoregressive",
    "global_ridge",
    "global_gbm",
    "zone_ridge",
    "zone_gbm",
)

# Volume estimate used for the headline energy numbers. The alternative is
# carried through the sensitivity analysis.
PRIMARY_VOLUME = "volume"
ALTERNATIVE_VOLUME = "volume_11kw"

# Rated to vehicle side conversion is not applied; the two source tables are
# treated as two competing estimates of the same quantity.


@dataclass(frozen=True)
class CVConfig:
    """Rolling origin validation scheme."""

    n_folds: int = 8
    test_hours: int = 336          # two weeks per fold
    min_train_hours: int = 1344    # eight weeks before the first origin
    step_hours: int = 336          # origins advance by the test length
    gap_hours: int = 0             # the horizon shift already separates the sets


@dataclass(frozen=True)
class FeatureConfig:
    """Feature construction settings.

    Every lag and window is applied to information available strictly before
    the prediction origin, so no feature can see the value it predicts.
    """

    lags: Tuple[int, ...] = (1, 2, 3, 24, 25, 48, 168)
    roll_windows: Tuple[int, ...] = (24, 168)
    use_weather: bool = True
    use_neighbours: bool = True
    use_zone_attrs: bool = True
    use_prices: bool = True
    neighbour_lags: Tuple[int, ...] = (1, 24)


@dataclass(frozen=True)
class MonitorConfig:
    """Monitoring backtest settings.

    The calibration window and the break date are not chosen. They are the
    longest segment of the daily utilisation series with no detected change
    point, and the change point that immediately follows it, both located by the
    detectors in phase 4 and written to ``results/monitoring_window.csv``. The
    monitoring phase re-reads that file and refuses to run if the dates here no
    longer match what the detectors found.
    """

    stable_start: str = "2022-11-08 00:00"
    stable_end: str = "2023-01-17 23:00"
    primary_break: str = "2023-01-18 00:00"
    window_hours: int = 168
    reference_hours: int = 672
    target_far_per_1000h: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)
    monitors: Tuple[str, ...] = (
        "input_psi",
        "input_ks",
        "residual_mean_cusum",
        "residual_abs_ewma",
    )
    monitor_target: str = "occupancy"
    monitor_horizon: int = 1


@dataclass(frozen=True)
class CapacityConfig:
    """Queueing and allocation settings."""

    budget_points: Tuple[int, ...] = (50, 100, 200, 400, 800)
    n_seeds: int = 20
    sim_hours: int = 336           # peak fortnight replayed per zone
    spill_fraction: float = 0.35   # share of blocked demand offered to neighbours
    max_neighbour_hops: int = 1
    session_energy_kwh: float = 20.0


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    cv: CVConfig = field(default_factory=CVConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)

    def digest(self, *parts: str) -> str:
        """Stable short hash of the configuration plus any extra parts."""
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        for part in parts:
            payload += "|" + str(part)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


CONFIG = Config()


def cell_key(**kwargs) -> str:
    """Stable identifier for one grid cell, used for deduplication."""
    payload = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
