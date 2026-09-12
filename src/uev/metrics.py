"""Error metrics.

Five quantities are reported for every cell of the grid. Three are ordinary
(MAE, RMSE, MAPE), one makes zones comparable (MASE), and one is asymmetric in
the way the capacity decision is asymmetric (saturation weighted absolute
error).

Two of these need care and both are handled explicitly rather than by dropping
the awkward rows quietly:

* MAPE is undefined wherever the actual value is zero, which happens in a few
  percent of zone hours. It is computed over the defined rows only and its
  coverage is returned next to it.
* MASE needs a scaling denominator that is itself estimated. The denominator is
  the mean absolute error of a seasonal naive forecast at a daily period,
  computed on the training part of the fold. Computing it on the test part
  would leak the test period into the metric.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

SEASONAL_PERIOD = 24

# Errors are weighted by 1 + SATURATION_WEIGHT * utilisation, so an error at a
# full zone counts four times as much as the same error at an empty one.
SATURATION_WEIGHT = 3.0


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Mean absolute percentage error over the rows where it is defined."""
    defined = np.abs(actual) > 0
    if not defined.any():
        return {"mape": float("nan"), "mape_coverage": 0.0}
    values = np.abs((actual[defined] - predicted[defined]) / actual[defined])
    return {"mape": float(np.mean(values) * 100.0),
            "mape_coverage": float(defined.mean())}


def seasonal_naive_denominator(train: np.ndarray, period: int = SEASONAL_PERIOD) -> float:
    """Mean absolute seasonal naive error on a training series.

    Returns ``nan`` for a series too short to form one seasonal difference, and
    for a series that is exactly constant, where the scaled error would divide
    by zero.
    """
    if train.size <= period:
        return float("nan")
    differences = np.abs(train[period:] - train[:-period])
    if differences.size == 0:
        return float("nan")
    value = float(np.mean(differences))
    return value if value > 0 else float("nan")


def mase(actual: np.ndarray, predicted: np.ndarray,
         denominators: np.ndarray) -> Dict[str, float]:
    """Mean absolute scaled error, pooled over rows.

    ``denominators`` is per row, carrying the denominator of the zone the row
    belongs to. Rows whose zone has no usable denominator are excluded and the
    coverage is returned.
    """
    usable = np.isfinite(denominators) & (denominators > 0)
    if not usable.any():
        return {"mase": float("nan"), "mase_coverage": 0.0}
    scaled = np.abs(actual[usable] - predicted[usable]) / denominators[usable]
    return {"mase": float(np.mean(scaled)), "mase_coverage": float(usable.mean())}


def saturation_weighted_mae(actual: np.ndarray, predicted: np.ndarray,
                            utilisation: np.ndarray,
                            weight: float = SATURATION_WEIGHT) -> float:
    """Absolute error weighted by how close the zone was to its ceiling.

    A symmetric metric treats an error at a saturated zone and an error at an
    empty one as equally costly. For a capacity decision they are not, because
    the error at the saturated zone is the one that decides whether demand was
    served, so this weighting is reported next to the symmetric metrics.
    """
    weights = 1.0 + weight * np.clip(utilisation, 0.0, 1.0)
    return float(np.sum(weights * np.abs(actual - predicted)) / np.sum(weights))


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(predicted - actual))


def evaluate(actual: np.ndarray, predicted: np.ndarray,
             denominators: Optional[np.ndarray] = None,
             utilisation: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Every metric for one set of predictions."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    out: Dict[str, float] = {
        "n": int(actual.size),
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "bias": bias(actual, predicted),
    }
    out.update(mape(actual, predicted))
    if denominators is not None:
        out.update(mase(actual, predicted, np.asarray(denominators, dtype=float)))
    else:
        out.update({"mase": float("nan"), "mase_coverage": 0.0})
    if utilisation is not None:
        out["sat_wmae"] = saturation_weighted_mae(actual, predicted,
                                                  np.asarray(utilisation, dtype=float))
    else:
        out["sat_wmae"] = float("nan")
    return out
