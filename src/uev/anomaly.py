"""Anomalous zone hours, found two different ways.

The dataset has already been through anomaly removal, imputation and outlier
replacement, so anything found here is a residual anomaly in a cleaned series,
not a raw sensor fault. That distinction is kept in the write up because it
changes what the finding means.

Two methods of genuinely different type are used. The first is distributional:
a robust score against the zone's own typical value for that hour of the week,
using the median and the median absolute deviation, which does not assume any
model. The second is model based: an isolation forest over a small feature
vector per zone hour, which can flag a combination of values that is unusual
even when no single value is.

There are no labels, so no precision or recall is reported. What is reported is
how much the two methods agree, which is a real measurement, and a sample is
inspected by hand.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .io_load import Bundle
from .logging_utils import get_logger
from .seeds import derive_seed

LOG = get_logger("anomaly")

MAD_TO_SIGMA = 1.4826
ROBUST_THRESHOLD = 6.0
CONTAMINATION = 0.005


def robust_scores(bundle: Bundle, column: str = "occupancy") -> pd.DataFrame:
    """Robust deviation from the zone's typical value for that hour of the week."""
    frame = bundle.long[["zone_id", "timestamp", column]].copy()
    stamps = pd.DatetimeIndex(frame["timestamp"])
    frame["hour_of_week"] = stamps.dayofweek * 24 + stamps.hour

    grouped = frame.groupby(["zone_id", "hour_of_week"])[column]
    median = grouped.transform("median")
    deviation = (frame[column] - median).abs()
    mad = deviation.groupby([frame["zone_id"], frame["hour_of_week"]]).transform("median")
    scale = (mad * MAD_TO_SIGMA).replace(0.0, np.nan)

    frame["robust_score"] = ((frame[column] - median) / scale).abs()
    frame["robust_score"] = frame["robust_score"].fillna(0.0)
    frame["robust_flag"] = frame["robust_score"] >= ROBUST_THRESHOLD
    return frame[["zone_id", "timestamp", column, "robust_score", "robust_flag"]]


def forest_scores(bundle: Bundle, contamination: float = CONTAMINATION) -> pd.DataFrame:
    """Isolation forest over a small per zone hour feature vector."""
    capacity = bundle.zones.set_index("zone_id")["capacity_points"]
    frame = bundle.long[["zone_id", "timestamp", "occupancy", "volume", "duration"]].copy()
    stamps = pd.DatetimeIndex(frame["timestamp"])
    frame["hour_of_day"] = stamps.hour
    frame["is_weekend"] = (stamps.dayofweek >= 5).astype(int)
    frame["utilisation"] = frame["occupancy"] / frame["zone_id"].map(capacity)
    frame["energy_per_point_hour"] = frame["volume"] / frame["duration"].replace(0, np.nan)
    frame["energy_per_point_hour"] = frame["energy_per_point_hour"].fillna(0.0)
    frame["duration_per_point"] = frame["duration"] / frame["occupancy"].replace(0, np.nan)
    frame["duration_per_point"] = frame["duration_per_point"].fillna(0.0)

    columns = ["utilisation", "energy_per_point_hour", "duration_per_point",
               "hour_of_day", "is_weekend"]
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=derive_seed("isolation_forest"), n_jobs=2)
    values = frame[columns].to_numpy(dtype=np.float32)
    model.fit(values)
    frame["forest_score"] = -model.score_samples(values)
    frame["forest_flag"] = model.predict(values) == -1
    return frame[["zone_id", "timestamp", "forest_score", "forest_flag"] + columns]


def agreement(robust: pd.DataFrame, forest: pd.DataFrame) -> Dict[str, float]:
    """How much the two detectors overlap.

    With no ground truth this is the honest measurement: the size of each flag
    set, the size of the intersection, and the Jaccard index. It says how much
    the methods see the same thing, and nothing about whether either is right.
    """
    merged = robust.merge(forest[["zone_id", "timestamp", "forest_flag", "forest_score"]],
                          on=["zone_id", "timestamp"], how="inner")
    a = merged["robust_flag"].to_numpy()
    b = merged["forest_flag"].to_numpy()
    both = int((a & b).sum())
    either = int((a | b).sum())
    return {
        "n_rows": int(len(merged)),
        "n_robust_flags": int(a.sum()),
        "n_forest_flags": int(b.sum()),
        "n_both": both,
        "n_either": either,
        "jaccard": float(both / either) if either else 0.0,
        "share_of_robust_also_forest": float(both / a.sum()) if a.sum() else 0.0,
        "share_of_forest_also_robust": float(both / b.sum()) if b.sum() else 0.0,
    }


def sample_for_inspection(robust: pd.DataFrame, forest: pd.DataFrame,
                          n_per_group: int = 8) -> pd.DataFrame:
    """A reproducible sample from each agreement group, for reading by eye."""
    merged = robust.merge(forest, on=["zone_id", "timestamp"], how="inner")
    merged["group"] = np.select(
        [merged["robust_flag"] & merged["forest_flag"],
         merged["robust_flag"] & ~merged["forest_flag"],
         ~merged["robust_flag"] & merged["forest_flag"]],
        ["both", "robust_only", "forest_only"], default="neither")

    generator = np.random.default_rng(derive_seed("anomaly_sample"))
    pieces = []
    for group in ("both", "robust_only", "forest_only"):
        part = merged[merged["group"] == group]
        if part.empty:
            continue
        take = min(n_per_group, len(part))
        rows = generator.choice(len(part), size=take, replace=False)
        pieces.append(part.iloc[np.sort(rows)])
    if not pieces:
        return pd.DataFrame()
    sample = pd.concat(pieces, ignore_index=True)
    return sample[["group", "zone_id", "timestamp", "occupancy", "utilisation",
                   "energy_per_point_hour", "duration_per_point",
                   "robust_score", "forest_score"]]
