"""Phase 9: zone segmentation and anomaly detection.

Zones are clustered on the shape of their day and the cluster count is
justified on silhouette width, the drop in inertia and the stability of the
partition across random starts. Anomalous zone hours are then found by two
methods of different type and the two are compared against each other, because
there are no labels to compare either against.

    python scripts/run_phase09_segment.py
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from uev import progress
from uev.anomaly import agreement, forest_scores, robust_scores, sample_for_inspection
from uev.io_load import load_bundle
from uev.logging_utils import get_logger, timed
from uev.paths import ensure_dirs
from uev.resultsio import write_table
from uev.segment import (choose_k, criteria_table, describe_segments, evaluate_k,
                         fit_segments, load_shapes)

LOG = get_logger("phase09")


def main() -> int:
    parser = argparse.ArgumentParser(description="segmentation and anomaly detection")
    parser.add_argument("--k", type=int, default=None, help="override the cluster count")
    args = parser.parse_args()

    ensure_dirs()
    progress.mark_phase(9, "running")
    bundle = load_bundle()

    with timed("load shapes", LOG):
        shapes = load_shapes(bundle)
    write_table("zone_load_shapes.csv", shapes.reset_index())

    with timed("cluster count diagnostics", LOG):
        diagnostics = evaluate_k(shapes)
    criteria = criteria_table(diagnostics)
    write_table("segment_diagnostics.csv", criteria)
    for _, row in criteria.iterrows():
        LOG.info("k=%2d silhouette %.4f  calinski %.1f  stability ARI %.3f  "
                 "largest cluster %.3f  admitted %s",
                 int(row["k"]), row["silhouette"], row["calinski_harabasz"],
                 row["stability_mean_ari"], row["largest_cluster_share"],
                 row["admitted"])

    k = args.k or choose_k(diagnostics)
    agree = criteria[criteria["best_by_calinski_harabasz"]]["k"].tolist()
    LOG.info("chosen cluster count: %d (the Calinski Harabasz ratio picks %s)",
             k, agree)

    labels = fit_segments(shapes, k)
    description = describe_segments(shapes, labels, bundle)
    write_table("zone_segments.csv", labels.reset_index())
    write_table("segment_profiles.csv", description)
    for _, row in description.iterrows():
        LOG.info("segment %d: %3d zones, peak hour %2d, peak to mean %.2f, %s",
                 int(row["segment"]), int(row["n_zones"]), int(row["peak_hour_weekday"]),
                 row["peak_to_mean_ratio"], row["label"])

    with timed("robust anomaly scores", LOG):
        robust = robust_scores(bundle)
    with timed("isolation forest", LOG):
        forest = forest_scores(bundle)

    overlap = agreement(robust, forest)
    write_table("anomaly_agreement.csv", pd.DataFrame([overlap]))
    LOG.info("anomalies: robust %d, forest %d, both %d, Jaccard %.4f",
             overlap["n_robust_flags"], overlap["n_forest_flags"],
             overlap["n_both"], overlap["jaccard"])

    sample = sample_for_inspection(robust, forest)
    write_table("anomaly_sample.csv", sample)

    flagged = robust[robust["robust_flag"]].merge(
        forest[["zone_id", "timestamp", "forest_flag", "forest_score"]],
        on=["zone_id", "timestamp"], how="outer")
    by_zone = flagged.groupby("zone_id").size().rename("n_flagged").reset_index()
    write_table("anomaly_by_zone.csv", by_zone.sort_values("n_flagged", ascending=False))

    progress.mark_phase(9, "complete",
                        f"k={k}, Jaccard {overlap['jaccard']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
