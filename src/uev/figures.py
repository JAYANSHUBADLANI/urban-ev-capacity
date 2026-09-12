"""Figure generation.

Each figure is meant to be readable on its own: axes carry units, the sample
size is stated where it matters, and the caption says what the reader is
supposed to take from it. Figures are written as PNG into ``figures/``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .paths import FIGURES, RESULTS, rel

LOG = get_logger("figures")

DPI = 140
GRID_STYLE = {"alpha": 0.25, "linewidth": 0.6}
PALETTE = ["#2c5f8a", "#c1553b", "#4f8f5b", "#8a6bab", "#c79b34",
           "#5d8aa8", "#a0522d", "#556b2f", "#7b6888"]


def _read(name: str) -> Optional[pd.DataFrame]:
    path = RESULTS / name
    if not path.exists() or path.stat().st_size == 0:
        LOG.warning("skipping a figure: %s is not present", rel(path))
        return None
    return pd.read_csv(path)


def _finish(figure, axes, path_name: str, caption: str) -> str:
    for axis in np.atleast_1d(axes).ravel():
        axis.grid(True, **GRID_STYLE)
        axis.set_axisbelow(True)
    figure.tight_layout()
    figure.text(0.005, 0.005, caption, fontsize=7, color="#444444", va="bottom")
    figure.subplots_adjust(bottom=max(0.12, figure.subplotpars.bottom))
    target = FIGURES / path_name
    figure.savefig(target, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    LOG.info("wrote %s", rel(target))
    return path_name


def figure_daily_demand_with_breaks() -> Optional[str]:
    daily = _read("aggregate_daily.csv")
    breaks = _read("structural_breaks.csv")
    significance = _read("structural_break_significance.csv")
    if daily is None:
        return None
    daily["timestamp"] = pd.to_datetime(daily["timestamp"])

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(daily["timestamp"], daily["utilisation"], color=PALETTE[0], linewidth=1.2)
    axes[0].set_ylabel("mean utilisation\n(fraction of points in use)")
    axes[0].set_title("City wide charging demand, daily, with change points located from the data")

    axes[1].plot(daily["timestamp"], daily["volume"] / 1000.0, color=PALETTE[1], linewidth=1.2)
    axes[1].set_ylabel("energy served\n(MWh per day, rated estimate)")
    axes[1].set_xlabel("date")

    if breaks is not None:
        dates = sorted(pd.to_datetime(
            breaks.loc[breaks["series"] == "utilisation", "break_date"]).unique())
        for axis in axes:
            for date in dates:
                axis.axvline(date, color="#888888", linestyle="--", linewidth=0.9)
        if dates:
            axes[0].axvline(dates[0], color="#888888", linestyle="--", linewidth=0.9,
                            label="change point found by either detector")
            axes[0].legend(loc="lower left", fontsize=8)

    n_days = len(daily)
    caption = (f"Daily aggregates over {n_days} days and 275 zones. Dashed lines are the "
               "union of the change points found by pruned dynamic programming and by "
               "binary segmentation on the day of week adjusted daily utilisation series. "
               "They were located from the series, not chosen in advance.")
    return _finish(figure, axes, "01_daily_demand_with_breaks.png", caption)


def figure_hour_of_day_profile() -> Optional[str]:
    profile = _read("hour_of_day_profile.csv")
    if profile is None:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for is_weekend, label, colour in ((0, "weekday", PALETTE[0]), (1, "weekend", PALETTE[1])):
        part = profile[profile["is_weekend"] == is_weekend].sort_values("hour_of_day")
        axes[0].plot(part["hour_of_day"], part["mean_occupancy"], label=label,
                     color=colour, marker="o", markersize=3)
        axes[1].plot(part["hour_of_day"], part["mean_volume"], label=label,
                     color=colour, marker="o", markersize=3)
    axes[0].set_xlabel("hour of day")
    axes[0].set_ylabel("mean points in use per zone")
    axes[0].set_title("Occupancy through the day")
    axes[1].set_xlabel("hour of day")
    axes[1].set_ylabel("mean energy per zone hour (kWh)")
    axes[1].set_title("Energy through the day")
    for axis in axes:
        axis.set_xticks(range(0, 24, 3))
        axis.legend(fontsize=8)
    caption = ("Averaged over 275 zones and 4344 hours. Occupancy is a count of charging "
               "points in use, not a percentage.")
    return _finish(figure, axes, "02_hour_of_day_profile.png", caption)


def figure_saturation() -> Optional[str]:
    saturation = _read("zone_saturation.csv")
    if saturation is None:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(saturation["mean_utilisation"], bins=40, color=PALETTE[0],
                 edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("mean utilisation (fraction of points in use)")
    axes[0].set_ylabel("zones")
    axes[0].set_title("How busy zones are on average")

    axes[1].scatter(saturation["mean_utilisation"], saturation["share_at_ceiling"] * 100,
                    s=18, alpha=0.7, color=PALETTE[1], edgecolor="none")
    axes[1].set_xlabel("mean utilisation (fraction of points in use)")
    axes[1].set_ylabel("share of hours with every point in use (%)")
    axes[1].set_title("Saturation is concentrated in a minority of zones")

    n_touching = int((saturation["max_utilisation"] >= 0.999).sum())
    caption = (f"One point per zone, {len(saturation)} zones. {n_touching} zones reach full "
               "occupancy at least once. Error metrics are asymmetric here, which is why a "
               "saturation weighted metric is reported alongside the symmetric ones.")
    return _finish(figure, axes, "03_saturation.png", caption)


def figure_volume_estimates() -> Optional[str]:
    comparison = _read("volume_estimate_comparison.csv")
    if comparison is None:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(comparison["implied_kw_rated"], comparison["implied_kw_11"],
                    s=18, alpha=0.7, color=PALETTE[0], edgecolor="none")
    limit = float(comparison["implied_kw_rated"].max()) * 1.05
    axes[0].plot([0, limit], [0, limit], color="#999999", linestyle="--", linewidth=0.9,
                 label="equal")
    axes[0].axhline(11.0, color=PALETTE[1], linestyle=":", linewidth=1.0,
                    label="11 kW vehicle side limit")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("implied power, rated estimate (kW per point hour)")
    axes[0].set_ylabel("implied power, 11 kW estimate\n(kW per point hour)")
    axes[0].set_title("Mostly, but not always, the rated estimate capped")
    axes[0].legend(fontsize=8)

    ordered = comparison.sort_values("volume_rated_kwh", ascending=False).head(30)
    positions = np.arange(len(ordered))
    axes[1].bar(positions - 0.2, ordered["volume_rated_kwh"] / 1e6, width=0.4,
                label="rated power estimate", color=PALETTE[0])
    axes[1].bar(positions + 0.2, ordered["volume_11kw_kwh"] / 1e6, width=0.4,
                label="11 kW estimate", color=PALETTE[1])
    axes[1].set_xlabel("the 30 zones with the most energy, ordered by the rated estimate")
    axes[1].set_ylabel("energy over six months (GWh)")
    axes[1].set_title("The gap is concentrated in the fast charging zones")
    axes[1].set_xticks([])
    axes[1].legend(fontsize=8)

    ratio = comparison["volume_rated_kwh"].sum() / comparison["volume_11kw_kwh"].sum()
    caption = (f"275 zones. Across the whole estate the rated estimate is {ratio:.2f} times the "
               "11 kW estimate. Every energy figure in this project depends on which is "
               "believed, so both are carried through.")
    return _finish(figure, axes, "04_volume_estimates.png", caption)


def figure_data_quality() -> Optional[str]:
    report = _read("data_quality_report.csv")
    if report is None:
        return None
    flat = report[report["check_id"].str.startswith("flat_run_")].copy()
    weather = report[report["check_id"].str.startswith("weather_repeat_")].copy()
    if flat.empty:
        return None
    flat["series"] = flat["check_id"].str.replace("flat_run_", "", regex=False)
    weather["series"] = weather["check_id"].str.replace("weather_repeat_", "", regex=False)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].barh(flat["series"], flat["fraction"] * 100, color=PALETTE[0])
    axes[0].set_xlabel("share of zone hours inside a constant run of 4 hours or more (%)")
    axes[0].set_ylabel("series")
    axes[0].set_title("Suspected imputation, reported not repaired")

    axes[1].barh(weather["series"], weather["fraction"] * 100, color=PALETTE[1])
    axes[1].set_xlabel("share of hours identical to the preceding hour (%)")
    axes[1].set_ylabel("weather series")
    axes[1].set_title("Weather is not hourly at source")

    caption = ("Computed over 1,194,600 zone hours and 4344 weather hours. The published "
               "series were already cleaned by forward and backward fill, so these runs are "
               "characterised rather than treated as defects.")
    return _finish(figure, axes, "05_data_quality.png", caption)


def figure_cv_scheme() -> Optional[str]:
    folds = _read("cv_folds.csv")
    if folds is None:
        return None
    figure, axis = plt.subplots(figsize=(10, 4.2))
    for _, row in folds.iterrows():
        start = pd.Timestamp(row["train_start"])
        train_end = pd.Timestamp(row["train_end"])
        test_end = pd.Timestamp(row["test_end"])
        y = int(row["fold"])
        axis.barh(y, (train_end - start).days, left=start, height=0.6,
                  color=PALETTE[0], label="training" if y == 0 else None)
        axis.barh(y, (test_end - train_end).days, left=train_end, height=0.6,
                  color=PALETTE[1], label="held out" if y == 0 else None)
    axis.set_yticks(range(len(folds)))
    axis.set_yticklabels([f"fold {i}" for i in folds["fold"]])
    axis.set_xlabel("date")
    axis.set_title("Rolling origin validation: training always precedes the held out window")
    axis.legend(fontsize=8, loc="upper left")
    caption = (f"{len(folds)} folds, each holding out 336 hours. Origins advance by 336 hours "
               "and the training window grows. No split is random.")
    return _finish(figure, axis, "06_cv_scheme.png", caption)


def figure_model_comparison() -> Optional[str]:
    metrics = _read("grid_metrics.csv")
    if metrics is None or metrics.empty:
        return None
    pooled = (metrics.groupby(["target", "horizon", "family"])["mase"]
              .mean().reset_index())
    targets = sorted(pooled["target"].unique())
    horizons = sorted(pooled["horizon"].unique())

    figure, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 4.6),
                                sharey=False)
    axes = np.atleast_1d(axes)
    families = sorted(pooled["family"].unique())
    width = 0.8 / max(len(horizons), 1)

    for index, target in enumerate(targets):
        axis = axes[index]
        part = pooled[pooled["target"] == target]
        positions = np.arange(len(families))
        for h_index, horizon in enumerate(horizons):
            values = [part[(part["family"] == family) & (part["horizon"] == horizon)]["mase"]
                      .mean() for family in families]
            axis.bar(positions + h_index * width - 0.4 + width / 2, values, width=width,
                     label=f"{horizon} h", color=PALETTE[h_index % len(PALETTE)])
        axis.set_xticks(positions)
        axis.set_xticklabels(families, rotation=60, ha="right", fontsize=7)
        axis.set_title(target)
        axis.set_ylabel("MASE (lower is better)" if index == 0 else "")
        axis.axhline(1.0, color="#999999", linestyle="--", linewidth=0.8)
        if index == 0:
            axis.legend(title="horizon", fontsize=8, title_fontsize=8)

    n_folds = metrics["fold"].nunique()
    caption = (f"Mean over {n_folds} rolling origin folds and 275 zones. MASE scales each "
               "zone by its own seasonal naive error, so zones of very different size are "
               "comparable. The dashed line is the seasonal naive benchmark.")
    return _finish(figure, axes, "07_model_comparison.png", caption)


def figure_error_by_volume_decile() -> Optional[str]:
    zone_metrics = _read("grid_zone_metrics.csv")
    deciles = _read("zone_volume_deciles.csv")
    if zone_metrics is None or zone_metrics.empty or deciles is None:
        return None
    merged = zone_metrics.merge(deciles[["zone_id", "volume_decile"]], on="zone_id")
    best = {"global": ["global_ridge", "global_gbm"], "per zone": ["zone_ridge", "zone_gbm"]}

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for group, families in best.items():
        part = merged[merged["family"].isin(families)]
        curve = part.groupby("volume_decile")["mase"].mean()
        axes[0].plot(curve.index, curve.to_numpy(), marker="o",
                     label=group, color=PALETTE[0] if group == "global" else PALETTE[1])
    axes[0].set_xlabel("zone volume decile (1 is the quietest)")
    axes[0].set_ylabel("MASE")
    axes[0].set_title("Error by zone size")
    axes[0].legend(fontsize=8)

    pooled = merged.groupby(["zone_id", "volume_decile", "family"])["mase"].mean().reset_index()
    pivot = pooled.pivot_table(index=["zone_id", "volume_decile"], columns="family",
                               values="mase")
    for column in ("global_ridge", "global_gbm", "zone_ridge", "zone_gbm"):
        if column not in pivot.columns:
            return None
    advantage = (pivot[["zone_ridge", "zone_gbm"]].min(axis=1)
                 - pivot[["global_ridge", "global_gbm"]].min(axis=1))
    frame = advantage.reset_index(name="advantage")
    grouped = [frame[frame["volume_decile"] == d]["advantage"].to_numpy()
               for d in sorted(frame["volume_decile"].unique())]
    axes[1].boxplot(grouped, labels=sorted(frame["volume_decile"].unique()), showfliers=False)
    axes[1].axhline(0.0, color="#999999", linestyle="--", linewidth=0.9)
    axes[1].set_xlabel("zone volume decile (1 is the quietest)")
    axes[1].set_ylabel("MASE advantage of the pooled model\n(positive favours pooling)")
    axes[1].set_title("Where pooling helps")

    caption = (f"{frame['zone_id'].nunique()} zones, pooled over targets, horizons and folds. "
               "A pooled mean would be dominated by the largest zones, which is why the error "
               "is shown across deciles.")
    return _finish(figure, axes, "08_error_by_volume_decile.png", caption)


def figure_error_by_hour() -> Optional[str]:
    hourly = _read("grid_hour_metrics.csv")
    if hourly is None or hourly.empty:
        return None
    figure, axis = plt.subplots(figsize=(10, 4.4))
    families = ["seasonal_naive_daily", "autoregressive", "global_ridge", "global_gbm"]
    for index, family in enumerate(families):
        part = hourly[hourly["family"] == family]
        if part.empty:
            continue
        curve = part.groupby("hour_of_day")["mae"].mean()
        axis.plot(curve.index, curve.to_numpy(), marker="o", markersize=3,
                  label=family, color=PALETTE[index % len(PALETTE)])
    axis.set_xlabel("hour of day of the forecast target")
    axis.set_ylabel("mean absolute error (charging points)")
    axis.set_title("Forecast error through the day")
    axis.set_xticks(range(0, 24, 3))
    axis.legend(fontsize=8)
    caption = ("Averaged over targets, horizons, folds and 275 zones. Errors are largest in "
               "the hours when demand moves fastest.")
    return _finish(figure, axis, "09_error_by_hour.png", caption)


def figure_monitors() -> Optional[str]:
    statistics = _read("monitor_statistics.csv")
    detection = _read("monitor_detection.csv")
    if statistics is None:
        return None
    statistics["timestamp"] = pd.to_datetime(statistics["timestamp"])

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for index, name in enumerate(("input_psi", "input_ks")):
        if name in statistics:
            axes[0].plot(statistics["timestamp"], statistics[name],
                         label=name, color=PALETTE[index], linewidth=1.0)
    axes[0].set_ylabel("input drift statistic")
    axes[0].set_title("Input drift and residual monitors over the real timeline")
    axes[0].legend(fontsize=8)

    # The first days carry a warm up transient while the trailing window fills.
    # It sits far outside the calibration period and would otherwise compress
    # the part of the axis the comparison actually happens in.
    settled = statistics["timestamp"] >= pd.Timestamp("2022-10-01")
    ceiling = max(float(statistics.loc[settled, name].max())
                  for name in ("input_psi", "input_ks") if name in statistics)
    axes[0].set_ylim(0.0, ceiling * 1.15)

    for index, name in enumerate(("residual_mean_cusum", "residual_abs_ewma")):
        if name in statistics:
            axes[1].plot(statistics["timestamp"], statistics[name],
                         label=name, color=PALETTE[index + 2], linewidth=1.0)
    axes[1].set_ylabel("residual monitor statistic")
    axes[1].set_xlabel("date")
    axes[1].legend(fontsize=8)

    break_time = pd.Timestamp("2023-01-18")
    for axis in axes:
        axis.axvline(break_time, color="#b03030", linewidth=1.2,
                     label="regime change located in phase 4")
        axis.axvspan(pd.Timestamp("2022-11-08"), pd.Timestamp("2023-01-17 23:00"),
                     color="#cccccc", alpha=0.25)
    axes[0].text(pd.Timestamp("2022-11-20"), axes[0].get_ylim()[1] * 0.92,
                 "stable period used to calibrate thresholds", fontsize=8, color="#555555")

    delay = ""
    if detection is not None and not detection.empty:
        primary = detection[detection["variant"] == "preregistered"] \
            if "variant" in detection.columns else detection
        found = primary[primary["detected"]]
        if not found.empty:
            best = found.loc[found["delay_hours"].idxmin()]
            delay = (f" The earliest alert comes from {best['monitor']} at "
                     f"{int(best['delay_hours'])} hours after the break.")
    caption = ("Shaded band is the calibration period, the vertical line is the change point. "
               "The top axis is cropped past the first month, where a warm up transient sits "
               "while the trailing window fills. "
               "Thresholds are set so every monitor has the same false alarm rate on the "
               "shaded band, because a monitor can always alert sooner by alerting more "
               "often." + delay)
    return _finish(figure, axes, "10_monitors.png", caption)


def figure_retraining() -> Optional[str]:
    retraining = _read("retraining_backtest.csv")
    if retraining is None or retraining.empty:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    order = retraining.sort_values("mae")
    positions = np.arange(len(order))
    axes[0].barh(positions, order["mae"], color=PALETTE[0])
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(order["policy"], fontsize=8)
    axes[0].set_xlabel("mean absolute error over the scored period (charging points)")
    axes[0].set_title("What each retraining policy recovers")
    axes[0].set_xlim(left=order["mae"].min() * 0.98)

    axes[1].scatter(order["n_refits"], order["mae_after_break"], s=60, color=PALETTE[1])
    for _, row in order.iterrows():
        axes[1].annotate(row["policy"], (row["n_refits"], row["mae_after_break"]),
                         fontsize=7, xytext=(4, 4), textcoords="offset points")
    axes[1].set_xlabel("number of refits")
    axes[1].set_ylabel("mean absolute error after the break\n(charging points)")
    axes[1].set_title("Error against the cost of getting it")

    n_hours = int(retraining["n_hours_scored"].iloc[0])
    caption = (f"Scored over {n_hours} hours and 275 zones with a one hour horizon. A refit is "
               "a full retrain of the pooled ridge on all history to that point.")
    return _finish(figure, axes, "11_retraining.png", caption)


def figure_segments() -> Optional[str]:
    shapes = _read("zone_load_shapes.csv")
    segments = _read("zone_segments.csv")
    profiles = _read("segment_profiles.csv")
    if shapes is None or segments is None or profiles is None:
        return None
    merged = shapes.merge(segments, on="zone_id")
    weekday = [c for c in shapes.columns if c.startswith("weekday_h")]
    weekend = [c for c in shapes.columns if c.startswith("weekend_h")]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    labels = dict(zip(profiles["segment"], profiles["label"]))
    counts = dict(zip(profiles["segment"], profiles["n_zones"]))
    for index, segment in enumerate(sorted(merged["segment"].unique())):
        part = merged[merged["segment"] == segment]
        name = f"{labels.get(segment, segment)} ({counts.get(segment, len(part))} zones)"
        axes[0].plot(range(24), part[weekday].mean().to_numpy(), marker="o", markersize=3,
                     label=name, color=PALETTE[index % len(PALETTE)])
        axes[1].plot(range(24), part[weekend].mean().to_numpy(), marker="o", markersize=3,
                     color=PALETTE[index % len(PALETTE)])
    axes[0].set_title("Weekday")
    axes[1].set_title("Weekend")
    for axis in axes:
        axis.set_xlabel("hour of day")
        axis.set_xticks(range(0, 24, 3))
    axes[0].set_ylabel("utilisation relative to the zone's own mean")
    axes[0].legend(fontsize=8)
    caption = ("Zones are clustered on shape, not level, so a large zone and a small zone "
               "with the same rhythm land together. The cluster count was chosen where "
               "silhouette width and the Calinski Harabasz ratio agree, among counts whose "
               "partition is stable across random starts and not dominated by one cluster.")
    return _finish(figure, axes, "12_segments.png", caption)


def figure_marginal_vs_utilisation() -> Optional[str]:
    marginal = _read("marginal_value.csv")
    if marginal is None or marginal.empty:
        return None
    rated = marginal[marginal["estimate"] == "rated"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].scatter(rated["mean_utilisation"], rated["marginal_energy_kwh_mean"],
                    s=20, alpha=0.75, color=PALETTE[0], edgecolor="none")
    axes[0].set_xlabel("mean observed utilisation (fraction of points in use)")
    axes[0].set_ylabel("marginal energy from one more point\n(kWh over the peak fortnight)")
    axes[0].set_title("Utilisation is not the same ranking as marginal value")

    util_rank = rated["mean_utilisation"].rank(ascending=False)
    marginal_rank = rated["marginal_energy_kwh_mean"].rank(ascending=False)
    axes[1].scatter(util_rank, marginal_rank, s=18, alpha=0.7, color=PALETTE[1],
                    edgecolor="none")
    limit = len(rated)
    axes[1].plot([1, limit], [1, limit], color="#999999", linestyle="--", linewidth=0.9)
    axes[1].set_xlabel("rank by utilisation (1 is busiest)")
    axes[1].set_ylabel("rank by marginal energy (1 is highest)")
    axes[1].set_title("Rank against rank")

    from scipy.stats import spearmanr
    rho = spearmanr(rated["mean_utilisation"], rated["marginal_energy_kwh_mean"]).statistic
    caption = (f"One point per zone, {len(rated)} zones, 20 seeds. Spearman correlation "
               f"{rho:.3f}. Marginal energy is the system wide gain, so it accounts for the "
               "spillover a relieved zone stops pushing onto its neighbours.")
    return _finish(figure, axes, "13_marginal_vs_utilisation.png", caption)


def figure_allocation_scenarios() -> Optional[str]:
    scenarios = _read("capacity_scenarios.csv")
    if scenarios is None or scenarios.empty:
        return None
    rated = scenarios[scenarios["estimate"] == "rated"]
    base = rated[rated["rule"] == "no_investment"].set_index("budget_points")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    rules = [r for r in ("by_utilisation", "by_marginal_energy", "balanced")
             if r in set(rated["rule"])]
    budgets = sorted(rated["budget_points"].unique())
    width = 0.8 / max(len(rules), 1)
    for index, rule in enumerate(rules):
        part = rated[rated["rule"] == rule].set_index("budget_points").reindex(budgets)
        gain = (part["served_energy_kwh_mean"]
                - base["served_energy_kwh_mean"].reindex(budgets)) / 1000.0
        spread = part["served_energy_kwh_std"] / 1000.0
        axes[0].bar(np.arange(len(budgets)) + index * width - 0.4 + width / 2, gain,
                    width=width, yerr=spread, capsize=3, label=rule,
                    color=PALETTE[index % len(PALETTE)])
    axes[0].set_xticks(range(len(budgets)))
    axes[0].set_xticklabels(budgets)
    axes[0].set_xlabel("budget (charging points added)")
    axes[0].set_ylabel("extra energy served over the peak fortnight\n(MWh, rated estimate)")
    axes[0].set_title("Served energy by allocation rule")
    axes[0].legend(fontsize=8)

    for index, rule in enumerate(rules):
        part = rated[rated["rule"] == rule].set_index("budget_points").reindex(budgets)
        axes[1].plot(budgets, part["unserved_sessions_mean"], marker="o",
                     label=rule, color=PALETTE[index % len(PALETTE)])
    no_investment = base.reindex(budgets)["unserved_sessions_mean"]
    axes[1].plot(budgets, no_investment, marker="s", linestyle="--", color="#777777",
                 label="no investment")
    axes[1].set_xlabel("budget (charging points added)")
    axes[1].set_ylabel("unserved sessions over the peak fortnight")
    axes[1].set_title("Demand still turned away")
    axes[1].legend(fontsize=8)

    # Whether the volume estimate changes the answer is checked here rather
    # than asserted, since the whole point of carrying both is that the choice
    # might matter.
    def ordering(estimate: str):
        part = scenarios[(scenarios["estimate"] == estimate)
                         & (scenarios["rule"] != "no_investment")]
        return {int(budget): tuple(group.sort_values("served_energy_kwh_mean",
                                                     ascending=False)["rule"])
                for budget, group in part.groupby("budget_points")}

    same_order = ordering("rated") == ordering("11kw") if "11kw" in set(
        scenarios["estimate"]) else False
    estimate_note = ("the 11 kW estimate changes the level but produces the same ordering "
                     "of the rules at every budget"
                     if same_order else
                     "the 11 kW estimate changes the ordering of the rules at some budgets")

    n_seeds = int(rated["n_seeds"].iloc[0])
    caption = (f"{n_seeds} seeds per scenario; bars show the mean and the error bars one "
               f"standard deviation across seeds. Energy uses the rated power estimate; "
               f"{estimate_note}.")
    return _finish(figure, axes, "14_allocation_scenarios.png", caption)


def figure_anomaly_agreement() -> Optional[str]:
    agreement = _read("anomaly_agreement.csv")
    sample = _read("anomaly_sample.csv")
    if agreement is None or agreement.empty:
        return None
    row = agreement.iloc[0]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    only_robust = int(row["n_robust_flags"] - row["n_both"])
    only_forest = int(row["n_forest_flags"] - row["n_both"])
    axes[0].bar(["robust score\nonly", "both", "isolation forest\nonly"],
                [only_robust, int(row["n_both"]), only_forest],
                color=[PALETTE[0], PALETTE[2], PALETTE[1]])
    axes[0].set_ylabel("zone hours flagged")
    axes[0].set_title("Two detectors, largely disjoint")
    for index, value in enumerate([only_robust, int(row["n_both"]), only_forest]):
        axes[0].text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=8)

    if sample is not None and not sample.empty:
        for index, group in enumerate(["both", "robust_only", "forest_only"]):
            part = sample[sample["group"] == group]
            if part.empty:
                continue
            axes[1].scatter(part["utilisation"], part["energy_per_point_hour"],
                            s=40, label=group, color=PALETTE[index % len(PALETTE)],
                            alpha=0.8, edgecolor="none")
        axes[1].set_xlabel("utilisation at the flagged hour (fraction of points in use)")
        axes[1].set_ylabel("energy per point hour (kWh)")
        axes[1].set_title("The inspected sample")
        axes[1].legend(fontsize=8)

    caption = (f"Out of {int(row['n_rows']):,} zone hours. Jaccard index "
               f"{row['jaccard']:.4f}. There are no labels, so no precision or recall is "
               "reported; the agreement between two different methods is what can honestly "
               "be measured.")
    return _finish(figure, axes, "15_anomaly_agreement.png", caption)


ALL_FIGURES: Dict[str, Callable[[], Optional[str]]] = {
    "daily_demand_with_breaks": figure_daily_demand_with_breaks,
    "hour_of_day_profile": figure_hour_of_day_profile,
    "saturation": figure_saturation,
    "volume_estimates": figure_volume_estimates,
    "data_quality": figure_data_quality,
    "cv_scheme": figure_cv_scheme,
    "model_comparison": figure_model_comparison,
    "error_by_volume_decile": figure_error_by_volume_decile,
    "error_by_hour": figure_error_by_hour,
    "monitors": figure_monitors,
    "retraining": figure_retraining,
    "segments": figure_segments,
    "marginal_vs_utilisation": figure_marginal_vs_utilisation,
    "allocation_scenarios": figure_allocation_scenarios,
    "anomaly_agreement": figure_anomaly_agreement,
}


def build_all() -> List[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    written = []
    for name, builder in ALL_FIGURES.items():
        try:
            result = builder()
        except Exception as error:                      # noqa: BLE001
            LOG.warning("figure %s could not be built: %s", name, error)
            continue
        if result:
            written.append(result)
    LOG.info("wrote %d of %d figures", len(written), len(ALL_FIGURES))
    return written
