"""Structural break detection on aggregate demand.

The dates are found from the series rather than assumed, because attributing a
break to a cause before locating it is how a convenient date gets chosen. Two
independent detectors are run and their agreement is reported: exact segment
neighbourhood search by pruned dynamic programming, and binary segmentation.
A moving block bootstrap gives each break a significance level that respects
the serial correlation in the series.

The cost model is a change in mean with constant variance, applied to a daily
series from which the day of week pattern has been removed. Working on daily
values rather than hourly ones keeps the daily cycle out of the cost function,
where it would otherwise dominate every segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .seeds import rng

LOG = get_logger("breaks")

MIN_SEGMENT = 7  # a regime shorter than a week is not a regime


def deseasonalise_daily(series: pd.Series) -> pd.Series:
    """Remove the additive day of week effect from a daily series."""
    frame = pd.DataFrame({"value": series.to_numpy()}, index=series.index)
    frame["dow"] = frame.index.dayofweek
    effect = frame.groupby("dow")["value"].transform("mean")
    return pd.Series(frame["value"] - effect + frame["value"].mean(),
                     index=series.index, name=series.name)


class _Cost:
    """Least squares cost of a segment, evaluated in constant time."""

    def __init__(self, values: np.ndarray):
        self.n = values.size
        self.cumsum = np.concatenate([[0.0], np.cumsum(values)])
        self.cumsq = np.concatenate([[0.0], np.cumsum(values ** 2)])

    def __call__(self, start: int, end: int) -> float:
        length = end - start
        if length <= 0:
            return 0.0
        total = self.cumsum[end] - self.cumsum[start]
        square = self.cumsq[end] - self.cumsq[start]
        return float(square - total * total / length)


def pelt(values: np.ndarray, penalty: float, min_segment: int = MIN_SEGMENT) -> List[int]:
    """Pruned exact change point search for a change in mean.

    Returns the interior break positions, each being the index of the first
    observation of a new segment.
    """
    n = values.size
    cost = _Cost(values)
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    previous = np.zeros(n + 1, dtype=int)
    candidates = [0]

    for end in range(min_segment, n + 1):
        scores = []
        for start in candidates:
            if end - start < min_segment:
                scores.append(np.inf)
                continue
            scores.append(best[start] + cost(start, end) + penalty)
        if not scores:
            continue
        array = np.asarray(scores)
        pick = int(np.argmin(array))
        best[end] = array[pick]
        previous[end] = candidates[pick]
        keep = [candidates[i] for i in range(len(candidates))
                if array[i] <= best[end] + penalty or np.isinf(array[i])]
        candidates = keep + [end - min_segment + 1] if end - min_segment + 1 > 0 else keep
        candidates = sorted(set(c for c in candidates if c <= end))

    breaks: List[int] = []
    position = n
    while position > 0:
        position = previous[position]
        if position > 0:
            breaks.append(position)
    return sorted(breaks)


def binary_segmentation(values: np.ndarray, max_breaks: int = 6,
                        min_segment: int = MIN_SEGMENT,
                        min_gain_ratio: float = 0.01) -> List[int]:
    """Greedy top down segmentation, used as an independent detector."""
    cost = _Cost(values)
    n = values.size
    total = cost(0, n)
    segments = [(0, n)]
    breaks: List[int] = []

    for _ in range(max_breaks):
        best_gain, best_point, best_segment = 0.0, None, None
        for start, end in segments:
            if end - start < 2 * min_segment:
                continue
            base = cost(start, end)
            for point in range(start + min_segment, end - min_segment + 1):
                gain = base - cost(start, point) - cost(point, end)
                if gain > best_gain:
                    best_gain, best_point, best_segment = gain, point, (start, end)
        if best_point is None or best_gain < min_gain_ratio * total:
            break
        breaks.append(best_point)
        start, end = best_segment
        segments.remove(best_segment)
        segments.extend([(start, best_point), (best_point, end)])
    return sorted(breaks)


def segment_means(values: np.ndarray, breaks: Sequence[int]) -> List[Tuple[int, int, float]]:
    """Start, end and mean of each segment implied by the breaks."""
    edges = [0] + list(breaks) + [values.size]
    return [(edges[i], edges[i + 1], float(values[edges[i]:edges[i + 1]].mean()))
            for i in range(len(edges) - 1)]


def _max_gain(values: np.ndarray, min_segment: int = MIN_SEGMENT) -> Tuple[float, int]:
    """Largest least squares gain from a single split, and where it falls."""
    cost = _Cost(values)
    n = values.size
    total = cost(0, n)
    best, position = 0.0, -1
    for candidate in range(min_segment, n - min_segment + 1):
        gain = total - cost(0, candidate) - cost(candidate, n)
        if gain > best:
            best, position = gain, candidate
    return best, position


def _moving_block_resample(residuals: np.ndarray, block: int,
                           generator: np.random.Generator) -> np.ndarray:
    """Draw a series of the same length from overlapping blocks of residuals."""
    n = residuals.size
    n_blocks = int(np.ceil(n / block))
    starts = generator.integers(0, max(1, n - block + 1), size=n_blocks)
    return np.concatenate([residuals[s:s + block] for s in starts])[:n]


def local_break_pvalue(values: np.ndarray, start: int, end: int,
                       block: int = 14, n_draws: int = 500,
                       label: str = "break") -> Dict[str, float]:
    """Significance of the strongest break inside one window.

    The null is no break in the window, so the resampling is done on the
    residuals from a constant mean fitted to that window rather than on the
    observed values. Resampling the observed values would carry the level shift
    into the null distribution and make every break look unremarkable, which is
    the opposite of what the test is for. Blocks preserve the short range serial
    correlation, so a smooth series alone does not produce a break.
    """
    window = np.asarray(values[start:end], dtype=float)
    n = window.size
    if n < 2 * MIN_SEGMENT + 1:
        return {"p_value": float("nan"), "observed_gain": float("nan"),
                "position": -1, "n_window": n}

    observed, position = _max_gain(window)
    residuals = window - window.mean()
    generator = rng(f"bootstrap:{label}")
    exceed = 0
    for _ in range(n_draws):
        drawn = window.mean() + _moving_block_resample(residuals, block, generator)
        gain, _ = _max_gain(drawn)
        if gain >= observed:
            exceed += 1
    return {
        "p_value": (exceed + 1) / (n_draws + 1),
        "observed_gain": float(observed),
        "position": int(start + position),
        "n_window": n,
    }


def windows_around(breaks: Sequence[int], n: int) -> List[Tuple[int, int, int]]:
    """For each break, the window bounded by its neighbouring breaks."""
    edges = [0] + list(breaks) + [n]
    out = []
    for i, point in enumerate(breaks):
        out.append((point, edges[i], edges[i + 2]))
    return out


@dataclass
class BreakResult:
    series_name: str
    dates: List[pd.Timestamp]
    method: str
    segments: List[Dict]

    def as_rows(self) -> List[Dict]:
        rows = []
        for date in self.dates:
            rows.append({"series": self.series_name, "method": self.method,
                         "break_date": date.strftime("%Y-%m-%d")})
        return rows


def detect(series: pd.Series, name: str, penalty_scale: float = 3.0) -> Dict[str, BreakResult]:
    """Run both detectors on a daily series and describe the segments."""
    adjusted = deseasonalise_daily(series)
    values = adjusted.to_numpy(dtype=float)
    variance = float(np.var(np.diff(values)) / 2.0) or 1.0
    penalty = penalty_scale * np.log(values.size) * variance

    out: Dict[str, BreakResult] = {}
    for method, points in (("pelt", pelt(values, penalty)),
                           ("binary_segmentation", binary_segmentation(values))):
        dates = [series.index[p] for p in points]
        segments = []
        for start, end, mean in segment_means(values, points):
            segments.append({
                "series": name, "method": method,
                "start": series.index[start].strftime("%Y-%m-%d"),
                "end": series.index[end - 1].strftime("%Y-%m-%d"),
                "n_days": end - start,
                "mean_level": round(mean, 6),
            })
        out[method] = BreakResult(name, dates, method, segments)
        LOG.info("%s via %s: %d breaks at %s", name, method, len(points),
                 ", ".join(d.strftime("%Y-%m-%d") for d in dates) or "none")
    return out


def agreement(results: Dict[str, BreakResult], tolerance_days: int = 7) -> List[Dict]:
    """Match breaks found by the two detectors within a tolerance."""
    pelt_dates = results["pelt"].dates
    bin_dates = results["binary_segmentation"].dates
    rows = []
    for date in pelt_dates:
        nearest = min(bin_dates, key=lambda d: abs((d - date).days), default=None)
        gap = abs((nearest - date).days) if nearest is not None else None
        rows.append({
            "pelt_date": date.strftime("%Y-%m-%d"),
            "nearest_binary_seg_date": nearest.strftime("%Y-%m-%d") if nearest is not None else "",
            "gap_days": gap if gap is not None else -1,
            "agreed": bool(gap is not None and gap <= tolerance_days),
        })
    return rows
