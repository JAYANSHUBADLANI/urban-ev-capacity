"""Phase 4: exploratory analysis and structural break detection.

The break dates are located empirically from aggregate demand, given a
significance level by a moving block bootstrap, and only then compared against
the calendar. Results land in ``results/``.

    python scripts/run_phase04_eda.py
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from uev import progress
from uev.breaks import (agreement, deseasonalise_daily, detect, local_break_pvalue,
                        pelt, windows_around)
from uev.eda import (aggregate_series, daily_series, hour_of_day_profile,
                     price_variation, saturation_profile, summary_rows,
                     volume_estimate_comparison, zone_volume_deciles)
from uev.io_load import load_bundle
from uev.logging_utils import get_logger, timed
from uev.paths import ensure_dirs
from uev.resultsio import write_table

LOG = get_logger("phase04")

BREAK_SERIES = ("utilisation", "volume", "duration")


def main() -> int:
    parser = argparse.ArgumentParser(description="exploratory analysis and break detection")
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    args = parser.parse_args()

    ensure_dirs()
    progress.mark_phase(4, "running")
    bundle = load_bundle()

    with timed("aggregate series", LOG):
        hourly = aggregate_series(bundle)
        daily = daily_series(hourly)
    write_table("aggregate_hourly.csv", hourly.reset_index())
    write_table("aggregate_daily.csv", daily.reset_index())

    write_table("eda_summary.csv", pd.DataFrame(summary_rows(bundle)))
    write_table("zone_volume_deciles.csv", zone_volume_deciles(bundle))
    write_table("hour_of_day_profile.csv", hour_of_day_profile(bundle))
    write_table("zone_saturation.csv", saturation_profile(bundle))
    write_table("zone_price_variation.csv", price_variation(bundle))
    write_table("volume_estimate_comparison.csv", volume_estimate_comparison(bundle))

    break_rows, segment_rows, agreement_rows = [], [], []
    with timed("structural break detection", LOG):
        for name in BREAK_SERIES:
            series = daily[name]
            results = detect(series, name)
            adjusted = deseasonalise_daily(series).to_numpy(dtype=float)
            positions = {series.index[i]: i for i in range(len(series))}

            for method, result in results.items():
                segment_rows.extend(result.segments)
                for date in result.dates:
                    index = positions[date]
                    before = adjusted[:index].mean() if index else np.nan
                    after = adjusted[index:].mean()
                    break_rows.append({
                        "series": name,
                        "method": method,
                        "break_date": date.strftime("%Y-%m-%d"),
                        "level_before": round(float(before), 6),
                        "level_after": round(float(after), 6),
                        "shift": round(float(after - before), 6),
                        "relative_shift": round(float((after - before) / before), 6)
                        if before else np.nan,
                    })
            for row in agreement(results):
                row["series"] = name
                agreement_rows.append(row)

    breaks = pd.DataFrame(break_rows)
    agreed = pd.DataFrame(agreement_rows)

    # Significance is tested locally: the window around each break is bounded by
    # its neighbouring breaks and the null is fitted inside that window only.
    p_rows = []
    with timed(f"local block bootstrap, {args.bootstrap_draws} draws per break", LOG):
        for name in BREAK_SERIES:
            series = daily[name]
            adjusted = deseasonalise_daily(series).to_numpy(dtype=float)
            points = [list(series.index).index(d) for d in detect(series, name)["pelt"].dates]
            for point, low, high in windows_around(points, len(adjusted)):
                outcome = local_break_pvalue(adjusted, low, high,
                                             n_draws=args.bootstrap_draws,
                                             label=f"{name}:{point}")
                before = float(adjusted[low:point].mean())
                after = float(adjusted[point:high].mean())
                p_rows.append({
                    "series": name,
                    "break_date": series.index[point].strftime("%Y-%m-%d"),
                    "window_days": high - low,
                    "p_value": round(outcome["p_value"], 4)
                    if outcome["p_value"] == outcome["p_value"] else np.nan,
                    "level_before": round(before, 6),
                    "level_after": round(after, 6),
                    "shift": round(after - before, 6),
                    "relative_shift": round((after - before) / before, 6) if before else np.nan,
                })
                LOG.info("%-11s %s window %3dd p=%s relative shift %+.1f%%",
                         name, series.index[point].strftime("%Y-%m-%d"), high - low,
                         f"{outcome['p_value']:.4f}"
                         if outcome["p_value"] == outcome["p_value"] else "n/a",
                         100 * (after - before) / before)

    significance = pd.DataFrame(p_rows)

    # The monitoring backtest needs a period with no detected break to calibrate
    # false alarm rates on, and one break to measure detection delay against.
    # Both are taken from the utilisation segmentation rather than chosen.
    utilisation = daily["utilisation"]
    adjusted = deseasonalise_daily(utilisation).to_numpy(dtype=float)
    points = [list(utilisation.index).index(d)
              for d in detect(utilisation, "utilisation")["pelt"].dates]
    edges = [0] + points + [len(adjusted)]
    spans = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    low, high = max(spans, key=lambda pair: pair[1] - pair[0])
    stable_start = utilisation.index[low]
    stable_end = utilisation.index[high - 1] + pd.Timedelta(hours=23)
    break_index = high if high < len(adjusted) else None

    window_rows = [{
        "role": "stable_calibration_period",
        "start": stable_start.strftime("%Y-%m-%d %H:%M"),
        "end": stable_end.strftime("%Y-%m-%d %H:%M"),
        "n_days": high - low,
        "n_hours": (high - low) * 24,
        "basis": "longest segment of the utilisation series with no detected break",
    }]
    if break_index is not None:
        break_date = utilisation.index[break_index]
        window_rows.append({
            "role": "primary_break",
            "start": break_date.strftime("%Y-%m-%d %H:%M"),
            "end": break_date.strftime("%Y-%m-%d %H:%M"),
            "n_days": 0,
            "n_hours": 0,
            "basis": "the break immediately following the stable segment",
        })
        LOG.info("stable calibration period %s to %s (%d days); primary break %s",
                 stable_start.strftime("%Y-%m-%d"), stable_end.strftime("%Y-%m-%d"),
                 high - low, break_date.strftime("%Y-%m-%d"))
        note = (f"stable {stable_start:%Y-%m-%d} to {stable_end:%Y-%m-%d}, "
                f"primary break {break_date:%Y-%m-%d}")
    else:
        note = "no break follows the longest stable segment"

    write_table("structural_breaks.csv", breaks)
    write_table("structural_break_segments.csv", pd.DataFrame(segment_rows))
    write_table("structural_break_agreement.csv", agreed)
    write_table("structural_break_significance.csv", significance)
    write_table("monitoring_window.csv", pd.DataFrame(window_rows))

    progress.mark_phase(4, "complete", note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
