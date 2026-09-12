"""Phase 8: monitoring, threshold calibration and the retraining backtest.

A single production model is trained on the history that precedes the stable
calibration period and is then used to score forward across the rest of the
timeline, which is what a deployed model does between refits. Four monitors run
over that timeline, their thresholds are calibrated to matched false alarm
rates on the stable period, and detection delay is measured against the break
located in phase 4.

The retraining backtest then asks what each policy would have recovered, and at
what cost in refits.

    python scripts/run_phase08_monitor.py
"""

from __future__ import annotations

import argparse
from typing import Dict

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from uev import progress
from uev.config import CONFIG
from uev.features import WARMUP, build_features
from uev.grid import clip_predictions
from uev.io_load import load_bundle
from uev.logging_utils import get_logger, timed
from uev.models import _fit_ridge
from uev.monitor import (MONITORED_FEATURES, MonitorBacktest, calibrate_threshold,
                         input_monitor_series, residual_monitor_series)
from uev.paths import RESULTS, ensure_dirs
from uev.resultsio import write_table
from uev.seeds import derive_seed

LOG = get_logger("phase08")

TRIGGER_FAR = 2.0        # alerts per 1000 hours used by the triggered policies
RETRAIN_COOLDOWN = 168   # hours before a further refit may be triggered


def build_design(features, origins, horizon, zone_mean, zone_std):
    design, target = features.design(origins, horizon)
    extra = np.column_stack([np.tile(zone_mean, origins.size),
                             np.tile(zone_std, origins.size)])
    return np.hstack([design, extra]).astype(np.float32), target


def zone_stats(features, train_end):
    window = features.y[:train_end]
    return (np.nan_to_num(np.nanmean(window, axis=0)).astype(np.float32),
            np.nan_to_num(np.nanstd(window, axis=0)).astype(np.float32))


def main() -> int:
    parser = argparse.ArgumentParser(description="monitoring and retraining backtest")
    parser.add_argument("--target", default=CONFIG.monitor.monitor_target)
    parser.add_argument("--horizon", type=int, default=CONFIG.monitor.monitor_horizon)
    args = parser.parse_args()

    ensure_dirs()
    progress.mark_phase(8, "running")
    settings = CONFIG.monitor
    bundle = load_bundle()
    index = bundle.index

    # The dates were located in phase 4. If that file disagrees with the
    # configuration the run stops, rather than silently monitoring against a
    # break that the detectors no longer find.
    window_file = RESULTS / "monitoring_window.csv"
    if window_file.exists():
        windows = pd.read_csv(window_file).set_index("role")
        detected_start = windows.loc["stable_calibration_period", "start"]
        detected_break = windows.loc["primary_break", "start"]
        if (str(detected_start) != settings.stable_start
                or str(detected_break) != settings.primary_break):
            raise SystemExit(
                "the configured monitoring window no longer matches what phase 4 detected: "
                f"detected {detected_start} and {detected_break}, configured "
                f"{settings.stable_start} and {settings.primary_break}")
        LOG.info("monitoring window confirmed against the phase 4 detection")

    capacity = bundle.zones.set_index("zone_id")["capacity_points"].reindex(
        list(bundle.zones["zone_id"])).to_numpy(dtype=float)
    features = build_features(bundle, args.target)
    train_end = int(np.searchsorted(index, pd.Timestamp(settings.stable_start)))
    break_position = int(np.searchsorted(index, pd.Timestamp(settings.primary_break)))
    LOG.info("production model trains on hours 0 to %d (%s to %s)", train_end - 1,
             index[0].strftime("%Y-%m-%d"), index[train_end - 1].strftime("%Y-%m-%d"))

    # Input monitors do not depend on the forecast horizon, so they are built
    # once. The reference is the tail of the training window, which is the
    # distribution the deployed model was fitted on.
    reference = slice(max(0, train_end - settings.reference_hours), train_end)
    blocks: Dict[str, np.ndarray] = {name: features.origin_blocks[name]
                                     for name in MONITORED_FEATURES
                                     if name in features.origin_blocks}
    LOG.info("input monitors watch %d features: %s", len(blocks), ", ".join(blocks))
    with timed("input monitor statistics", LOG):
        input_statistics = input_monitor_series(blocks, reference, settings.window_hours)

    zone_mean, zone_std = zone_stats(features, train_end)
    residual_statistics: Dict[int, Dict[str, np.ndarray]] = {}
    residual_series: Dict[int, tuple] = {}
    primary_model = None

    for horizon in sorted({args.horizon, 24}):  # noqa: B007
        train_origins = np.arange(WARMUP - 1, train_end - horizon, dtype=int)
        score_origins = np.arange(train_end - horizon, features.n_hours - horizon, dtype=int)
        with timed(f"production model at horizon {horizon}", LOG):
            train_x, train_y = build_design(features, train_origins, horizon,
                                            zone_mean, zone_std)
            model, n_fits = _fit_ridge(train_x, train_y, n_origins=train_origins.size,
                                       n_zones=features.n_zones)
            score_x, actual = build_design(features, score_origins, horizon,
                                           zone_mean, zone_std)
            predicted = clip_predictions(model.predict(score_x), args.target,
                                         np.tile(capacity, score_origins.size))
        LOG.info("horizon %d: ridge alpha %g, %d fits, mean absolute residual %.4f",
                 horizon, model.alpha, n_fits, float(np.mean(np.abs(predicted - actual))))
        if horizon == args.horizon:
            primary_model = model

        grid = (predicted - actual).reshape(score_origins.size, features.n_zones)
        mean_residual = np.full(features.n_hours, np.nan)
        mean_abs_residual = np.full(features.n_hours, np.nan)
        positions = score_origins + horizon
        mean_residual[positions] = grid.mean(axis=1)
        mean_abs_residual[positions] = np.abs(grid).mean(axis=1)
        residual_series[horizon] = (mean_residual, mean_abs_residual)
        residual_reference = slice(train_end, train_end + settings.reference_hours)
        residual_statistics[horizon] = residual_monitor_series(
            mean_residual, mean_abs_residual, residual_reference, settings.window_hours)

    statistics = {**input_statistics, **residual_statistics[args.horizon]}
    mean_residual, mean_abs_residual = residual_series[args.horizon]
    write_table("monitor_statistics.csv", pd.DataFrame({
        "timestamp": index.strftime("%Y-%m-%d %H:%M"),
        "mean_residual": mean_residual,
        "mean_abs_residual": mean_abs_residual,
        **{name: values for name, values in statistics.items()},
    }))

    # The preregistered evaluation, plus two sensitivities. The stable window
    # was chosen from the utilisation change points, and it turns out to contain
    # a one day shock in December that both residual monitors react to. That
    # shock raises their calibrated thresholds, so the second variant asks what
    # happens if the shock week is excluded from calibration, and the third asks
    # whether the answer depends on the forecast horizon.
    variants = []
    full_mask = ((index >= pd.Timestamp(settings.stable_start))
                 & (index <= pd.Timestamp(settings.stable_end)))
    shock_week = ((index >= pd.Timestamp("2022-12-07"))
                  & (index <= pd.Timestamp("2022-12-14 23:00")))
    variants.append(("preregistered", args.horizon, full_mask))
    variants.append(("stable_window_excludes_december_shock", args.horizon,
                     full_mask & ~shock_week))
    variants.append(("horizon_24", 24, full_mask))

    all_detection, all_verdicts = [], []
    for label, horizon, mask in variants:
        combined = {**input_statistics, **residual_statistics[horizon]}
        backtest = MonitorBacktest(statistics=combined, index=index, settings=settings)
        evaluation = backtest.evaluate_with_mask(mask)
        evaluation.insert(0, "variant", label)
        evaluation.insert(1, "horizon", horizon)
        all_detection.append(evaluation)

        for far in settings.target_far_per_1000h:
            part = evaluation[evaluation["target_far_per_1000h"] == far]
            best = {}
            for kind in ("input", "residual"):
                side = part[(part["kind"] == kind) & part["detected"]]
                best[kind] = ((float(side["delay_hours"].min()),
                               str(side.loc[side["delay_hours"].idxmin(), "monitor"]))
                              if not side.empty else (np.inf, "none"))
            all_verdicts.append({
                "variant": label, "horizon": horizon, "target_far_per_1000h": far,
                "best_input_monitor": best["input"][1],
                "best_input_delay_hours": best["input"][0],
                "best_residual_monitor": best["residual"][1],
                "best_residual_delay_hours": best["residual"][0],
                "input_later_than_residual": bool(best["input"][0] > best["residual"][0]),
            })

    detection = pd.concat(all_detection, ignore_index=True)
    verdict = pd.DataFrame(all_verdicts)
    write_table("monitor_detection.csv", detection)
    write_table("monitor_claim2.csv", verdict)

    for _, row in verdict.iterrows():
        LOG.info("%-38s far %.1f  input %6s  residual %6s  input later: %s",
                 row["variant"], row["target_far_per_1000h"],
                 f"{row['best_input_delay_hours']:.0f}h"
                 if np.isfinite(row["best_input_delay_hours"]) else "none",
                 f"{row['best_residual_delay_hours']:.0f}h"
                 if np.isfinite(row["best_residual_delay_hours"]) else "none",
                 row["input_later_than_residual"])

    primary = verdict[verdict["variant"] == "preregistered"]
    supported = int(primary["input_later_than_residual"].sum())
    LOG.info("claim 2 as preregistered: the input monitor is later at %d of %d matched rates",
             supported, len(primary))

    # Retraining backtest.
    with timed("retraining backtest", LOG):
        retrain_rows = run_retraining(features, args.horizon, train_end, break_position,
                                      capacity, args.target, statistics, index, settings)
    write_table("retraining_backtest.csv", pd.DataFrame(retrain_rows))
    for row in retrain_rows:
        LOG.info("%-26s refits %2d  MAE %.4f  MAE after the break %.4f",
                 row["policy"], row["n_refits"], row["mae"], row["mae_after_break"])

    progress.mark_phase(8, "complete",
                        f"claim 2 supported at {supported} of {len(verdict)} matched rates")
    return 0


def run_retraining(features, horizon, train_end, break_position, capacity, target,
                   statistics, index, settings):
    """Score the evaluation period under several retraining policies."""
    n_zones = features.n_zones
    evaluation_origins = np.arange(train_end - horizon, features.n_hours - horizon, dtype=int)
    actual_grid = features.y[evaluation_origins + horizon]

    cache: Dict[int, object] = {}

    def model_for(end: int):
        if end not in cache:
            zone_mean, zone_std = zone_stats(features, end)
            origins = np.arange(WARMUP - 1, end - horizon, dtype=int)
            design, target_values = build_design(features, origins, horizon,
                                                 zone_mean, zone_std)
            fitted, _ = _fit_ridge(design, target_values,
                                   n_origins=origins.size, n_zones=n_zones)
            cache[end] = (fitted, zone_mean, zone_std)
        return cache[end]

    def schedule_predictions(schedule):
        """Predict every evaluation hour using the most recent model."""
        predictions = np.full((evaluation_origins.size, n_zones), np.nan)
        bounds = list(schedule) + [features.n_hours]
        for i, start in enumerate(schedule):
            stop = bounds[i + 1]
            mask = (evaluation_origins + horizon >= max(start, train_end)) & \
                   (evaluation_origins + horizon < stop)
            if not mask.any():
                continue
            fitted, zone_mean, zone_std = model_for(start)
            origins = evaluation_origins[mask]
            design, _ = build_design(features, origins, horizon, zone_mean, zone_std)
            values = clip_predictions(fitted.predict(design), target,
                                      np.tile(capacity, origins.size))
            predictions[mask] = values.reshape(origins.size, n_zones)
        return predictions

    policies = {"no_retraining": [train_end]}
    for period in (336, 672):
        points = [train_end] + list(range(train_end + period, features.n_hours, period))
        policies[f"periodic_{period}h"] = points

    for monitor_name in ("residual_mean_cusum", "input_psi"):
        statistic = statistics[monitor_name]
        stable = ((index >= pd.Timestamp(settings.stable_start))
                  & (index <= pd.Timestamp(settings.stable_end)))
        threshold = calibrate_threshold(statistic, stable, TRIGGER_FAR)["threshold"]
        points, last = [train_end], train_end
        for position in range(train_end, features.n_hours):
            value = statistic[position]
            if np.isfinite(value) and value > threshold and position - last >= RETRAIN_COOLDOWN:
                points.append(position)
                last = position
        policies[f"triggered_{monitor_name}"] = points

    rows = []
    after = (evaluation_origins + horizon) >= break_position
    for name, schedule in policies.items():
        predictions = schedule_predictions(schedule)
        errors = np.abs(predictions - actual_grid)
        rows.append({
            "policy": name,
            "n_refits": len(schedule) - 1,
            "mae": float(np.nanmean(errors)),
            "mae_before_break": float(np.nanmean(errors[~after])),
            "mae_after_break": float(np.nanmean(errors[after])),
            "n_hours_scored": int(evaluation_origins.size),
            "retrain_hours": ";".join(index[p].strftime("%Y-%m-%d") for p in schedule[1:]),
        })
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
