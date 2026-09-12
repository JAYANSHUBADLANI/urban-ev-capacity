"""Phase 7: the full rolling origin grid.

Every cell is a target, a horizon, a fold and a model family. Results are
appended as each cell finishes and cells already recorded are skipped, so the
phase can be stopped and resumed without losing or repeating work.

    python scripts/run_phase07_grid.py
    python scripts/run_phase07_grid.py --targets occupancy --horizons 1
    python scripts/run_phase07_grid.py --status
"""

from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401

import pandas as pd

from uev import progress
from uev.config import HORIZONS, TARGETS
from uev.cv import describe, make_folds
from uev.features import build_features
from uev.grid import (FAMILIES, GRID_DOW_METRICS, GRID_HOUR_METRICS,
                      GRID_METRICS, GRID_ZONE_METRICS, GridRunner, expected_cells)
from uev.io_load import load_bundle
from uev.logging_utils import get_logger
from uev.models import ar_lag_tensor
from uev.paths import ensure_dirs
from uev.resultsio import append_rows, existing_keys, read_rows, write_table

LOG = get_logger("phase07")


def shard_names(suffix: str) -> dict:
    """Output file names for one worker.

    Workers write to their own files so that two processes appending at the same
    time cannot interleave a row. The shards are concatenated by --merge once
    every worker has finished.
    """
    if not suffix:
        return {"metrics": GRID_METRICS, "zone": GRID_ZONE_METRICS,
                "hour": GRID_HOUR_METRICS, "dow": GRID_DOW_METRICS}
    return {"metrics": GRID_METRICS.replace(".csv", f"_{suffix}.csv"),
            "zone": GRID_ZONE_METRICS.replace(".csv", f"_{suffix}.csv"),
            "hour": GRID_HOUR_METRICS.replace(".csv", f"_{suffix}.csv"),
            "dow": GRID_DOW_METRICS.replace(".csv", f"_{suffix}.csv")}


def all_shards() -> list:
    """Every shard file present, base first."""
    from uev.paths import RESULTS
    base = [GRID_METRICS, GRID_ZONE_METRICS, GRID_HOUR_METRICS, GRID_DOW_METRICS]
    groups = []
    for name in base:
        stem = name.replace(".csv", "")
        found = sorted(p.name for p in RESULTS.glob(f"{stem}_*.csv"))
        groups.append((name, found))
    return groups


def done_keys() -> set:
    """Cells already recorded, across every shard."""
    keys = set(existing_keys(GRID_METRICS))
    from uev.paths import RESULTS
    stem = GRID_METRICS.replace(".csv", "")
    for path in RESULTS.glob(f"{stem}_*.csv"):
        keys |= existing_keys(path.name)
    return keys


def merge_shards() -> int:
    """Fold every worker shard into the base file and remove the shards."""
    from uev.paths import RESULTS
    for base, shards in all_shards():
        if not shards:
            continue
        frames = [read_rows(base)] + [read_rows(name) for name in shards]
        frames = [f for f in frames if not f.empty]
        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True)
        subset = ["cell_key"] if "cell_key" in merged.columns else \
            ["target", "horizon", "fold", "family"] + \
            [c for c in ("zone_id", "hour_of_day", "day_of_week") if c in merged.columns]
        merged = merged.drop_duplicates(subset=subset, keep="last")
        write_table(base, merged)
        for name in shards:
            (RESULTS / name).unlink()
        LOG.info("merged %d shards into %s: %d rows", len(shards), base, len(merged))
    return 0


def report_status() -> int:
    total = len(expected_cells())
    done = len(done_keys())
    LOG.info("grid cells complete: %d of %d (%.1f%%)", done, total, 100.0 * done / total)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="run the forecasting grid")
    parser.add_argument("--targets", nargs="*", default=list(TARGETS))
    parser.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    parser.add_argument("--families", nargs="*", default=list(FAMILIES))
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--status", action="store_true", help="report progress and stop")
    parser.add_argument("--suffix", default="",
                        help="write to a private set of files, for running workers in parallel")
    parser.add_argument("--merge", action="store_true",
                        help="fold worker files into the base files and stop")
    args = parser.parse_args()

    ensure_dirs()
    if args.status:
        return report_status()
    if args.merge:
        merge_shards()
        return report_status()

    names = shard_names(args.suffix)
    progress.mark_phase(7, "running")
    bundle = load_bundle()
    runner = GridRunner(bundle, done_keys=done_keys())
    write_table("cv_folds.csv", pd.DataFrame(describe(runner.folds, bundle.index)))

    total_cells = len(expected_cells())
    started = time.perf_counter()
    completed = 0

    for target in args.targets:
        needed = [(horizon, fold) for horizon in args.horizons for fold in runner.folds
                  if (args.folds is None or fold.fold in args.folds)
                  and any(family in args.families for family in runner.pending(target, horizon, fold))]
        if not needed:
            LOG.info("%s: nothing pending", target)
            continue

        features = build_features(bundle, target)
        lag_tensor = ar_lag_tensor(features)

        for horizon, fold in needed:
            pending = [f for f in runner.pending(target, horizon, fold) if f in args.families]
            if not pending:
                continue
            cell_started = time.perf_counter()
            results = runner.run_group(features, lag_tensor, target, horizon, fold, pending)

            append_rows(names["metrics"], [r.pooled for r in results])
            for result in results:
                append_rows(names["zone"], result.per_zone)
                append_rows(names["hour"], result.per_hour)
                append_rows(names["dow"], result.per_dow)
            if not args.suffix:
                progress.record_cells("grid", [r.pooled["cell_key"] for r in results])

            completed += len(results)
            done_total = len(done_keys())
            elapsed = time.perf_counter() - started
            rate = elapsed / max(completed, 1)
            remaining = total_cells - done_total
            LOG.info("%s h=%-2d fold %d: %d cells in %.1fs | done %d/%d | "
                     "about %.0f min remaining",
                     target, horizon, fold.fold, len(results),
                     time.perf_counter() - cell_started, done_total, total_cells,
                     rate * remaining / 60.0)

    done_total = len(done_keys())
    if done_total >= total_cells:
        progress.mark_phase(7, "complete", f"{done_total} of {total_cells} cells")
    else:
        progress.mark_phase(7, "running", f"{done_total} of {total_cells} cells")
    LOG.info("grid cells complete: %d of %d", done_total, total_cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
