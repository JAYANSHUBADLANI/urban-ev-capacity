"""Data quality suite.

The published dataset is already cleaned. The cleaning is itself part of the
evidence, so these checks are written to characterise what the cleaning left
behind rather than to pretend the series are raw. Three consequences are
measured here and carried into the write up:

* anomalies found later are residual anomalies, not raw sensor faults;
* the retained zones are a survivorship filtered sample;
* runs of identical consecutive values are candidate imputation artefacts.

Each check returns a row with a count, a denominator and a status. The status
is informational for the characterisation checks and a hard pass or fail for
the structural checks, because a broken index is a defect while a plateau is a
property of the source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import EXPECTED_HOURS, EXPECTED_ZONES
from .graph import asymmetry_count, degree
from .io_load import Bundle
from .logging_utils import get_logger

LOG = get_logger("quality")

MEASURES = ("occupancy", "volume", "volume_11kw", "duration", "e_price", "s_price")

# Physical bounds. Occupancy and duration are bounded above by the number of
# charging points in the zone: occupancy counts points in use and duration
# accumulates at most one hour per point per hour.
BOUNDS: Dict[str, tuple] = {
    "occupancy": (0.0, None),
    "volume": (0.0, None),
    "volume_11kw": (0.0, None),
    "duration": (0.0, None),
    "e_price": (0.0, 5.0),
    "s_price": (0.0, 5.0),
}

# A run of identical consecutive values this long or longer is treated as a
# candidate imputation artefact. Four hours is chosen because a genuine flat
# stretch of that length is rare in an hourly demand series while forward fill
# from a coarser source produces them routinely.
RUN_THRESHOLD = 4


@dataclass
class Check:
    check_id: str
    description: str
    scope: str
    n_failing: int
    n_total: int
    status: str
    detail: str = ""

    @property
    def fraction(self) -> float:
        return float(self.n_failing) / self.n_total if self.n_total else 0.0

    def as_row(self) -> Dict:
        return {
            "check_id": self.check_id,
            "description": self.description,
            "scope": self.scope,
            "n_failing": self.n_failing,
            "n_total": self.n_total,
            "fraction": round(self.fraction, 6),
            "status": self.status,
            "detail": self.detail,
        }


def run_lengths(values: np.ndarray) -> np.ndarray:
    """Return the length of the constant run each element belongs to."""
    if values.size == 0:
        return np.zeros(0, dtype=int)
    change = np.empty(values.size, dtype=bool)
    change[0] = True
    change[1:] = values[1:] != values[:-1]
    group = np.cumsum(change) - 1
    counts = np.bincount(group)
    return counts[group]


def suspected_imputation(long: pd.DataFrame, column: str,
                         threshold: int = RUN_THRESHOLD) -> pd.Series:
    """Flag observations inside a constant run of at least ``threshold`` hours."""
    flags = np.zeros(len(long), dtype=bool)
    for _, index in long.groupby("zone_id", sort=False).indices.items():
        ordered = np.sort(index)
        lengths = run_lengths(long[column].to_numpy()[ordered])
        flags[ordered] = lengths >= threshold
    return pd.Series(flags, index=long.index, name=f"{column}_flat_run")


def _status(fraction: float, warn: float, fail: float) -> str:
    if fraction >= fail:
        return "fail"
    if fraction >= warn:
        return "warn"
    return "pass"


def run_checks(bundle: Bundle) -> pd.DataFrame:
    """Run every check and return the report table."""
    long = bundle.long
    zones = bundle.zones.set_index("zone_id")
    checks: List[Check] = []
    n_rows = len(long)

    # Structural checks. These are hard pass or fail.
    index = pd.DatetimeIndex(sorted(long["timestamp"].unique()))
    expected = pd.date_range(index[0], index[-1], freq="h")
    missing = len(expected) - len(index)
    checks.append(Check("index_completeness",
                        "the hourly index covers every hour between the first and last observation",
                        "index", int(missing), len(expected),
                        "pass" if missing == 0 else "fail",
                        f"{len(index)} distinct hours"))

    checks.append(Check("index_length",
                        f"the index holds the documented {EXPECTED_HOURS} hours",
                        "index", int(abs(len(index) - EXPECTED_HOURS)), EXPECTED_HOURS,
                        "pass" if len(index) == EXPECTED_HOURS else "fail"))

    dupes = int(long.duplicated(subset=["zone_id", "timestamp"]).sum())
    checks.append(Check("duplicate_keys",
                        "no zone and timestamp pair appears twice",
                        "rows", dupes, n_rows, "pass" if dupes == 0 else "fail"))

    n_zones = long["zone_id"].nunique()
    checks.append(Check("zone_count",
                        f"the documented {EXPECTED_ZONES} zones are present",
                        "zones", int(abs(n_zones - EXPECTED_ZONES)), EXPECTED_ZONES,
                        "pass" if n_zones == EXPECTED_ZONES else "fail"))

    per_zone_hours = long.groupby("zone_id").size()
    ragged = int((per_zone_hours != EXPECTED_HOURS).sum())
    checks.append(Check("zone_coverage",
                        "every zone is observed for every hour in the window",
                        "zones", ragged, n_zones, "pass" if ragged == 0 else "fail"))

    nulls = int(long[list(MEASURES)].isna().sum().sum())
    checks.append(Check("no_nulls",
                        "no measured value is null after the published imputation",
                        "cells", nulls, n_rows * len(MEASURES), "pass" if nulls == 0 else "fail"))

    # Physical bounds.
    for column, (low, high) in BOUNDS.items():
        series = long[column]
        bad = int((series < low).sum())
        if high is not None:
            bad += int((series > high).sum())
        checks.append(Check(f"bounds_{column}",
                            f"{column} lies inside its physical bounds",
                            "rows", bad, n_rows, "pass" if bad == 0 else "fail",
                            f"observed range {series.min():.3f} to {series.max():.3f}"))

    # Capacity bounds. Occupancy counts points in use and duration accumulates
    # at most one hour per point, so both are capped by the zone capacity.
    capacity = long["zone_id"].map(zones["capacity_points"]).to_numpy()
    for column in ("occupancy", "duration"):
        over = int((long[column].to_numpy() > capacity + 1e-6).sum())
        checks.append(Check(f"capacity_bound_{column}",
                            f"{column} never exceeds the charging points installed in the zone",
                            "rows", over, n_rows, "pass" if over == 0 else "fail"))

    # Cross table agreement. A zone hour with points in use but no energy, or
    # energy with no points in use, is a contradiction between two tables that
    # were derived from the same sessions.
    occ_no_vol = int(((long["occupancy"] > 0) & (long["volume"] <= 0)).sum())
    checks.append(Check("cross_occupancy_without_volume",
                        "points in use are accompanied by non zero charging volume",
                        "rows", occ_no_vol, n_rows, _status(occ_no_vol / n_rows, 0.01, 0.10),
                        "counted, not repaired"))

    vol_no_occ = int(((long["occupancy"] <= 0) & (long["volume"] > 0)).sum())
    checks.append(Check("cross_volume_without_occupancy",
                        "charging volume is accompanied by at least one point in use",
                        "rows", vol_no_occ, n_rows, _status(vol_no_occ / n_rows, 0.01, 0.10),
                        "counted, not repaired"))

    dur_no_occ = int(((long["occupancy"] <= 0) & (long["duration"] > 0)).sum())
    checks.append(Check("cross_duration_without_occupancy",
                        "charging duration is accompanied by at least one point in use",
                        "rows", dur_no_occ, n_rows, _status(dur_no_occ / n_rows, 0.01, 0.10),
                        "counted, not repaired"))

    # Saturation. The share of zone hours sitting exactly at the ceiling drives
    # the asymmetry of the error metrics near the operating point.
    at_ceiling = int((long["occupancy"].to_numpy() >= capacity - 1e-6).sum())
    checks.append(Check("saturated_zone_hours",
                        "zone hours with every charging point in use",
                        "rows", at_ceiling, n_rows, "info",
                        "informational: drives the saturation aware metric"))

    # Suspected imputation. Reported, never repaired.
    for column in ("occupancy", "volume", "duration", "e_price", "s_price"):
        flat = suspected_imputation(long, column)
        checks.append(Check(f"flat_run_{column}",
                            f"{column} observations inside a constant run of "
                            f"{RUN_THRESHOLD} hours or more",
                            "rows", int(flat.sum()), n_rows, "info",
                            "candidate imputation artefact, reported not repaired"))

    # Weather resolution. Consecutive equal hours indicate a coarser source.
    weather = bundle.weather
    for column in ("air_temp_c_central", "humidity_pct_central", "air_temp_c_airport"):
        repeats = int((weather[column].diff() == 0).sum())
        checks.append(Check(f"weather_repeat_{column}",
                            f"{column} hours identical to the preceding hour",
                            "hours", repeats, len(weather) - 1, "info",
                            "weather is a lower resolution covariate"))

    # The two volume estimates.
    identical = int((long["volume"] - long["volume_11kw"]).abs().lt(1e-6).sum())
    checks.append(Check("volume_estimates_identical",
                        "rows where the rated power and 11 kW volume estimates agree",
                        "rows", identical, n_rows, "info",
                        f"aggregate ratio {long['volume'].sum() / long['volume_11kw'].sum():.4f}"))

    # Adjacency sanity. The failures here are real and are handled explicitly
    # downstream rather than being repaired in place.
    adjacency = bundle.adjacency.to_numpy()
    asym = asymmetry_count(bundle.adjacency)
    checks.append(Check("adjacency_symmetric",
                        "the zone adjacency matrix is symmetric as published",
                        "cells", asym, adjacency.size, "pass" if asym == 0 else "fail",
                        f"{asym // 2} edges recorded in one direction only; "
                        "the neighbour structure is symmetrised by union"))

    self_loops = int(np.diag(adjacency).sum())
    checks.append(Check("adjacency_self_loops",
                        "the adjacency diagonal is zero as published",
                        "zones", self_loops, adjacency.shape[0],
                        "pass" if self_loops == 0 else "fail",
                        "a zone is listed as adjacent to itself; the diagonal is removed"))

    isolated = int((degree(bundle.adjacency) == 0).sum())
    checks.append(Check("adjacency_isolated_zones",
                        "zones with no adjacent zone once the diagonal is removed",
                        "zones", isolated, adjacency.shape[0], "info",
                        "isolated zones can neither receive nor shed spillover demand"))

    distance = bundle.distance.to_numpy()
    bad_diag = int((np.abs(np.diag(distance)) > 1e-6).sum())
    checks.append(Check("distance_zero_diagonal",
                        "the zone distance matrix has a zero diagonal",
                        "cells", bad_diag, distance.shape[0],
                        "pass" if bad_diag == 0 else "fail"))

    report = pd.DataFrame([c.as_row() for c in checks])
    LOG.info("ran %d checks: %s", len(report),
             ", ".join(f"{k}={v}" for k, v in report["status"].value_counts().items()))
    return report


def quality_summary(report: pd.DataFrame) -> str:
    """One line per check, suitable for a log or a document."""
    lines = []
    for _, row in report.iterrows():
        lines.append(f"{row['status']:>4}  {row['check_id']:<34} "
                     f"{row['n_failing']:>9,} / {row['n_total']:>10,}  "
                     f"({row['fraction'] * 100:6.3f}%)  {row['description']}")
    return "\n".join(lines)
