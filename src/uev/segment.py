"""Zone segmentation by load shape.

Zones are clustered on the shape of their day rather than on their level, so
that a large zone and a small zone with the same rhythm land together. The
shape is mean utilisation by hour of day, taken separately for weekdays and
weekends, and then scaled so each zone's profile has unit mean.

The cluster count is justified on more than one criterion, because the elbow of
an inertia curve on its own is a matter of opinion. Silhouette width, the
relative drop in inertia, and the stability of the partition across random
starts are all reported, and the count is chosen where they agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score

from .io_load import Bundle
from .logging_utils import get_logger
from .seeds import derive_seed

LOG = get_logger("segment")

CANDIDATE_K = tuple(range(2, 11))
STABILITY_SEEDS = 10


def load_shapes(bundle: Bundle) -> pd.DataFrame:
    """Mean utilisation by hour of day and day type, scaled to unit mean."""
    capacity = bundle.zones.set_index("zone_id")["capacity_points"]
    frame = bundle.long[["zone_id", "timestamp", "occupancy"]].copy()
    stamps = pd.DatetimeIndex(frame["timestamp"])
    frame["hour_of_day"] = stamps.hour
    frame["is_weekend"] = (stamps.dayofweek >= 5).astype(int)
    frame["utilisation"] = frame["occupancy"] / frame["zone_id"].map(capacity)

    profile = frame.groupby(["zone_id", "is_weekend", "hour_of_day"])["utilisation"].mean()
    wide = profile.unstack(["is_weekend", "hour_of_day"])
    wide.columns = [f"{'weekend' if w else 'weekday'}_h{h:02d}" for w, h in wide.columns]
    scaled = wide.div(wide.mean(axis=1).replace(0, np.nan), axis=0)
    return scaled.fillna(0.0)


def evaluate_k(shapes: pd.DataFrame, candidates=CANDIDATE_K,
               seeds: int = STABILITY_SEEDS) -> pd.DataFrame:
    """Silhouette, inertia, variance ratio and stability for each cluster count."""
    values = shapes.to_numpy(dtype=float)
    rows = []
    for k in candidates:
        base = KMeans(n_clusters=k, n_init=10,
                      random_state=derive_seed(f"kmeans:{k}")).fit(values)
        labels = base.labels_
        agreements = []
        for seed in range(seeds):
            other = KMeans(n_clusters=k, n_init=10,
                           random_state=derive_seed(f"kmeans:{k}:{seed}")).fit_predict(values)
            agreements.append(adjusted_rand_score(labels, other))
        sizes = np.bincount(labels, minlength=k)
        rows.append({
            "k": k,
            "largest_cluster_share": float(sizes.max() / sizes.sum()),
            "smallest_cluster_size": int(sizes.min()),
            "inertia": float(base.inertia_),
            "silhouette": float(silhouette_score(values, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(values, labels)),
            "stability_mean_ari": float(np.mean(agreements)),
            "stability_min_ari": float(np.min(agreements)),
        })
    frame = pd.DataFrame(rows)
    frame["inertia_drop"] = -frame["inertia"].diff() / frame["inertia"].shift(1)
    return frame


MAX_CLUSTER_SHARE = 0.60


def choose_k(diagnostics: pd.DataFrame, min_stability: float = 0.75,
             max_cluster_share: float = MAX_CLUSTER_SHARE) -> int:
    """Pick the cluster count where the criteria agree.

    Three filters are applied in order. Counts whose partition is not stable
    across random starts are dropped, because an unstable partition is not a
    finding. Counts that put more than ``max_cluster_share`` of the zones in a
    single cluster are dropped next: silhouette width on normalised shapes is
    maximised at two clusters here, but that split leaves 87 percent of zones in
    one group, which is not a segmentation a network planner can act on.
    Among what remains the highest silhouette width wins, and the Calinski
    Harabasz ratio is reported alongside so the two can be seen to agree or not.
    """
    candidates = diagnostics[diagnostics["stability_mean_ari"] >= min_stability]
    balanced = candidates[candidates["largest_cluster_share"] <= max_cluster_share]
    if balanced.empty:
        balanced = candidates if not candidates.empty else diagnostics
    return int(balanced.loc[balanced["silhouette"].idxmax(), "k"])


def criteria_table(diagnostics: pd.DataFrame,
                   min_stability: float = 0.75,
                   max_cluster_share: float = MAX_CLUSTER_SHARE) -> pd.DataFrame:
    """Which counts each criterion admits, and which count each would pick."""
    frame = diagnostics.copy()
    frame["passes_stability"] = frame["stability_mean_ari"] >= min_stability
    frame["passes_balance"] = frame["largest_cluster_share"] <= max_cluster_share
    frame["admitted"] = frame["passes_stability"] & frame["passes_balance"]
    admitted = frame[frame["admitted"]]
    frame["best_by_silhouette"] = frame["k"] == (
        admitted.loc[admitted["silhouette"].idxmax(), "k"] if not admitted.empty else -1)
    frame["best_by_calinski_harabasz"] = frame["k"] == (
        admitted.loc[admitted["calinski_harabasz"].idxmax(), "k"]
        if not admitted.empty else -1)
    return frame


def fit_segments(shapes: pd.DataFrame, k: int) -> pd.Series:
    values = shapes.to_numpy(dtype=float)
    model = KMeans(n_clusters=k, n_init=20, random_state=derive_seed("kmeans:final")).fit(values)
    return pd.Series(model.labels_, index=shapes.index, name="segment")


def describe_segments(shapes: pd.DataFrame, labels: pd.Series,
                      bundle: Bundle) -> pd.DataFrame:
    """Business readable description of each segment."""
    weekday_columns = [c for c in shapes.columns if c.startswith("weekday_")]
    weekend_columns = [c for c in shapes.columns if c.startswith("weekend_")]
    zones = bundle.zones.set_index("zone_id")
    capacity = zones["capacity_points"]
    totals = bundle.long.groupby("zone_id")["volume"].sum()
    utilisation = (bundle.long.groupby("zone_id")["occupancy"].mean()
                   / capacity)

    rows = []
    for segment, members in labels.groupby(labels):
        ids = list(members.index)
        weekday = shapes.loc[ids, weekday_columns].mean()
        weekend = shapes.loc[ids, weekend_columns].mean()
        peak_hour = int(np.argmax(weekday.to_numpy()))
        trough_hour = int(np.argmin(weekday.to_numpy()))
        peak_ratio = float(weekday.max() / max(weekday.mean(), 1e-9))
        weekend_lift = float(weekend.mean() / max(weekday.mean(), 1e-9))
        night_share = float(weekday.to_numpy()[list(range(0, 6))].sum()
                            / max(weekday.to_numpy().sum(), 1e-9))
        rows.append({
            "segment": int(segment),
            "n_zones": len(ids),
            "peak_hour_weekday": peak_hour,
            "trough_hour_weekday": trough_hour,
            "peak_to_mean_ratio": round(peak_ratio, 3),
            "weekend_to_weekday_ratio": round(weekend_lift, 3),
            "overnight_share": round(night_share, 3),
            "mean_capacity_points": round(float(capacity.reindex(ids).mean()), 1),
            "mean_utilisation": round(float(utilisation.reindex(ids).mean()), 4),
            "total_volume_kwh": float(totals.reindex(ids).sum()),
            "label": archetype_label(peak_hour, peak_ratio, weekend_lift, night_share),
        })
    return pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)


def archetype_label(peak_hour: int, peak_ratio: float,
                    weekend_lift: float, night_share: float) -> str:
    """Name a load shape in terms a network planner would use."""
    if night_share > 0.30:
        return "overnight charging, residential rhythm"
    if peak_ratio < 1.25:
        return "flat load, close to always on"
    if weekend_lift > 1.10:
        return "weekend leaning, retail and leisure rhythm"
    if 6 <= peak_hour <= 11:
        return "morning peak, commuter arrival rhythm"
    if 12 <= peak_hour <= 16:
        return "midday peak, daytime fleet rhythm"
    if 17 <= peak_hour <= 22:
        return "evening peak, return journey rhythm"
    return "late night peak, unusual rhythm"
