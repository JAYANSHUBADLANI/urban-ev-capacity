"""Monitoring, threshold calibration and the retraining backtest.

Four monitors run over the real timeline at hourly resolution. Two watch the
distribution of the model inputs and two watch the forecast residuals. All four
are windowed, so their statistics are stationary under a stable regime and a
threshold calibrated on an early period stays meaningful later. A cumulative
statistic that never resets would drift upwards on its own and would make a
late detection look like an early one.

Thresholds are set so that every monitor produces the same false alarm rate on
a period with no detected change point, because a monitor can always be made to
alert sooner by alerting more often. Comparing detection delay at anything
other than a matched false alarm rate compares nothing.

The input monitors work on a fixed binning derived from the reference window,
which lets a rolling histogram be maintained by adding the arriving hour and
removing the departing one. Without that the hourly recomputation over 275
zones would dominate the phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG, MonitorConfig
from .logging_utils import get_logger

LOG = get_logger("monitor")

PSI_BINS = 10
KS_BINS = 50
EWMA_LAMBDA = 0.1
CUSUM_K = 0.5

# Features the input monitors watch. Calendar and static attributes are
# excluded: the calendar cannot drift and the zone attributes do not move
# inside the window, so including them would only dilute the statistic.
MONITORED_FEATURES: Tuple[str, ...] = (
    "lag_0", "lag_24", "roll_mean_24", "roll_mean_168", "roll_std_24",
    "diff_24", "neighbour_mean_0", "e_price_0", "s_price_0",
)


def _bin_edges(reference: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile edges of the reference sample, with degenerate bins collapsed."""
    finite = reference[np.isfinite(reference)]
    if finite.size == 0:
        return np.array([-np.inf, np.inf])
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(finite, quantiles))
    if edges.size < 2:
        edges = np.array([finite.min() - 1.0, finite.max() + 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _hourly_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Counts per bin for every hour, shaped ``(hours, bins)``."""
    n_bins = edges.size - 1
    indices = np.clip(np.digitize(values, edges[1:-1], right=False), 0, n_bins - 1)
    out = np.zeros((values.shape[0], n_bins), dtype=np.int32)
    for hour in range(values.shape[0]):
        out[hour] = np.bincount(indices[hour], minlength=n_bins)
    return out


def _rolling_window_counts(counts: np.ndarray, window: int) -> np.ndarray:
    """Counts summed over a trailing window ending at each hour."""
    cumulative = np.vstack([np.zeros((1, counts.shape[1]), dtype=np.int64),
                            np.cumsum(counts.astype(np.int64), axis=0)])
    n_hours = counts.shape[0]
    out = np.full((n_hours, counts.shape[1]), 0, dtype=np.int64)
    for hour in range(window - 1, n_hours):
        out[hour] = cumulative[hour + 1] - cumulative[hour + 1 - window]
    return out


def psi(expected: np.ndarray, actual: np.ndarray, floor: float = 1e-6) -> float:
    """Population stability index between two binned distributions."""
    e = np.clip(expected / max(expected.sum(), 1.0), floor, None)
    a = np.clip(actual / max(actual.sum(), 1.0), floor, None)
    return float(np.sum((a - e) * np.log(a / e)))


def binned_ks(expected: np.ndarray, actual: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic evaluated on a shared binning."""
    e = np.cumsum(expected / max(expected.sum(), 1.0))
    a = np.cumsum(actual / max(actual.sum(), 1.0))
    return float(np.max(np.abs(a - e)))


def input_monitor_series(feature_blocks: Dict[str, np.ndarray],
                         reference_slice: slice, window: int) -> Dict[str, np.ndarray]:
    """Hourly PSI and KS statistics for the monitored inputs.

    PSI is averaged across features, because drift that shows in one feature
    only is usually a data plumbing problem rather than a demand regime change.
    KS is taken as the maximum across features, which makes the two input
    monitors genuinely different rather than two scalings of one statistic.
    """
    psi_columns, ks_columns = [], []
    for name, block in feature_blocks.items():
        reference = block[reference_slice]
        psi_edges = _bin_edges(reference, PSI_BINS)
        ks_edges = _bin_edges(reference, KS_BINS)

        psi_reference = _hourly_counts(reference, psi_edges).sum(axis=0)
        ks_reference = _hourly_counts(reference, ks_edges).sum(axis=0)

        psi_rolling = _rolling_window_counts(_hourly_counts(block, psi_edges), window)
        ks_rolling = _rolling_window_counts(_hourly_counts(block, ks_edges), window)

        psi_values = np.full(block.shape[0], np.nan)
        ks_values = np.full(block.shape[0], np.nan)
        for hour in range(window - 1, block.shape[0]):
            psi_values[hour] = psi(psi_reference, psi_rolling[hour])
            ks_values[hour] = binned_ks(ks_reference, ks_rolling[hour])
        psi_columns.append(psi_values)
        ks_columns.append(ks_values)

    return {
        "input_psi": np.nanmean(np.vstack(psi_columns), axis=0),
        "input_ks": np.nanmax(np.vstack(ks_columns), axis=0),
    }


def residual_monitor_series(mean_residual: np.ndarray, mean_abs_residual: np.ndarray,
                            reference_slice: slice, window: int) -> Dict[str, np.ndarray]:
    """Hourly windowed CUSUM and EWMA statistics on the forecast residuals."""
    reference = mean_residual[reference_slice]
    centre = float(np.nanmean(reference))
    spread = float(np.nanstd(reference)) or 1.0
    standardised = (mean_residual - centre) / spread

    n_hours = mean_residual.size
    cusum = np.full(n_hours, np.nan)
    for hour in range(window - 1, n_hours):
        segment = standardised[hour - window + 1:hour + 1]
        positive = negative = 0.0
        best = 0.0
        for value in segment:
            positive = max(0.0, positive + value - CUSUM_K)
            negative = max(0.0, negative - value - CUSUM_K)
            best = max(best, positive, negative)
        cusum[hour] = best

    abs_reference = mean_abs_residual[reference_slice]
    abs_centre = float(np.nanmean(abs_reference))
    abs_spread = float(np.nanstd(abs_reference)) or 1.0
    ewma = np.full(n_hours, np.nan)
    state = abs_centre
    for hour in range(n_hours):
        value = mean_abs_residual[hour]
        if np.isfinite(value):
            state = EWMA_LAMBDA * value + (1.0 - EWMA_LAMBDA) * state
        if hour >= window - 1:
            ewma[hour] = abs(state - abs_centre) / abs_spread

    return {"residual_mean_cusum": cusum, "residual_abs_ewma": ewma}


def calibrate_threshold(statistic: np.ndarray, stable_mask: np.ndarray,
                        far_per_1000h: float) -> Dict[str, float]:
    """Threshold giving a target false alarm rate on the stable period."""
    values = statistic[stable_mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"threshold": float("nan"), "achieved_far_per_1000h": float("nan"),
                "n_stable_hours": 0, "n_false_alarms": 0}
    quantile = 1.0 - far_per_1000h / 1000.0
    threshold = float(np.quantile(values, min(max(quantile, 0.0), 1.0)))
    exceedances = int((values > threshold).sum())
    return {
        "threshold": threshold,
        "achieved_far_per_1000h": 1000.0 * exceedances / values.size,
        "n_stable_hours": int(values.size),
        "n_false_alarms": exceedances,
    }


def first_alert(statistic: np.ndarray, threshold: float,
                start_position: int) -> Optional[int]:
    """First hour at or after ``start_position`` where the threshold is exceeded."""
    if not np.isfinite(threshold):
        return None
    tail = statistic[start_position:]
    exceed = np.flatnonzero(np.isfinite(tail) & (tail > threshold))
    if exceed.size == 0:
        return None
    return int(start_position + exceed[0])


@dataclass
class MonitorBacktest:
    statistics: Dict[str, np.ndarray]
    index: pd.DatetimeIndex
    settings: MonitorConfig

    def stable_mask(self) -> np.ndarray:
        return ((self.index >= pd.Timestamp(self.settings.stable_start))
                & (self.index <= pd.Timestamp(self.settings.stable_end)))

    def break_position(self) -> int:
        return int(np.searchsorted(self.index, pd.Timestamp(self.settings.primary_break)))

    def evaluate(self) -> pd.DataFrame:
        """Detection delay for every monitor at every matched false alarm rate."""
        return self.evaluate_with_mask(self.stable_mask())

    def evaluate_with_mask(self, stable: np.ndarray) -> pd.DataFrame:
        """Detection delay, calibrating on a supplied stable period."""
        position = self.break_position()
        rows = []
        for name, statistic in self.statistics.items():
            kind = "input" if name.startswith("input") else "residual"
            for far in self.settings.target_far_per_1000h:
                calibration = calibrate_threshold(statistic, stable, far)
                alert = first_alert(statistic, calibration["threshold"], position)
                rows.append({
                    "monitor": name,
                    "kind": kind,
                    "target_far_per_1000h": far,
                    "threshold": calibration["threshold"],
                    "achieved_far_per_1000h": calibration["achieved_far_per_1000h"],
                    "n_stable_hours": calibration["n_stable_hours"],
                    "n_false_alarms": calibration["n_false_alarms"],
                    "detected": alert is not None,
                    "alert_time": self.index[alert].strftime("%Y-%m-%d %H:%M")
                    if alert is not None else "",
                    "delay_hours": int(alert - position) if alert is not None else -1,
                })
        return pd.DataFrame(rows)
