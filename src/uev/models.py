"""The forecasting families.

Eight families are fitted, in three groups.

Baselines, which need no fitting: the last observation carried forward, the
seasonal naive forecast at a daily period, and the seasonal naive forecast at a
weekly period.

A per zone direct autoregression, fitted by least squares on the previous 24
hours. The autoregression is direct rather than recursive: a separate set of
coefficients is fitted for each horizon and the target is the value at the
horizon, so multi step errors do not compound through repeated substitution.
This is implemented here rather than taken from a library so that the project
does not depend on one being installed.

Four learned families: ridge and gradient boosting, each fitted once across all
zones with zone identity and zone attributes among the features, and each
fitted separately per zone without the static attributes, which are constant
inside a zone and carry no information there.

Hyperparameter selection respects time. The ridge penalty is chosen on an inner
split that holds out the last 336 hours of the training window, never the test
window. The gradient boosting setting is chosen the same way on the first fold
and then held fixed, which keeps the refit count manageable and still never
allows the selection to see a test window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .features import FeatureSet
from .logging_utils import get_logger
from .seeds import derive_seed

LOG = get_logger("models")

BASELINE_FAMILIES = ("last_observation", "seasonal_naive_daily", "seasonal_naive_weekly")
PER_ZONE_FAMILIES = ("autoregressive", "zone_ridge", "zone_gbm")
GLOBAL_FAMILIES = ("global_ridge", "global_gbm")

AR_ORDER = 24
RIDGE_ALPHAS: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
INNER_VALIDATION_HOURS = 336

GLOBAL_GBM_GRID: Tuple[Dict[str, object], ...] = (
    {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31},
    {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31},
    {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 63},
)

# Per zone training sets are two orders of magnitude smaller than the pooled
# one, so a smaller model is used and is not searched.
ZONE_GBM_PARAMS: Dict[str, object] = {
    "max_iter": 60, "learning_rate": 0.1, "max_depth": 4, "min_samples_leaf": 20,
}


@dataclass
class FitOutcome:
    """Predictions plus the bookkeeping the results table records."""

    predictions: np.ndarray
    n_fits: int
    chosen: Dict[str, object] = field(default_factory=dict)


def ar_lag_tensor(features: FeatureSet, order: int = AR_ORDER) -> np.ndarray:
    """Lags 0 to ``order - 1`` of the target, shaped ``(hours, zones, order)``."""
    n_hours, n_zones = features.y.shape
    tensor = np.full((n_hours, n_zones, order), np.nan, dtype=np.float32)
    for lag in range(order):
        if lag == 0:
            tensor[:, :, 0] = features.y
        else:
            tensor[lag:, :, lag] = features.y[:-lag]
    return tensor


def baseline_predictions(family: str, features: FeatureSet,
                         origins: np.ndarray, horizon: int) -> np.ndarray:
    """Predictions for the families that need no fitting.

    Each reads the series at a time no later than the origin, which is what
    makes them usable at the horizons considered here.
    """
    if family == "last_observation":
        source = origins
    elif family == "seasonal_naive_daily":
        source = origins + horizon - 24
    elif family == "seasonal_naive_weekly":
        source = origins + horizon - 168
    else:
        raise ValueError(f"not a baseline family: {family}")
    if source.min() < 0:
        raise ValueError("the baseline would need an observation before the series starts")
    if (source > origins).any():
        raise ValueError("the baseline would read beyond the origin")
    return features.y[source].reshape(-1)


def _inner_split(n_origins: int, n_zones: int,
                 validation_origins: int = INNER_VALIDATION_HOURS) -> Tuple[slice, slice]:
    """Row slices for a time ordered inner split of a design matrix.

    Rows are origin major, so the final origins are the final rows and a slice
    is enough to hold out the most recent part of the training window.
    """
    validation_origins = min(validation_origins, max(1, n_origins // 4))
    cut = (n_origins - validation_origins) * n_zones
    return slice(0, cut), slice(cut, n_origins * n_zones)


class RidgeModel:
    """A ridge with its scaler and the columns it was actually fitted on.

    Columns with no variation inside the training window are dropped before
    fitting. Within a single zone the price series are often exactly constant,
    which leaves a zero column after centring and makes the normal equations
    singular. Dropping them is equivalent and avoids fitting through an
    ill conditioned matrix.
    """

    def __init__(self, ridge: Ridge, scaler: StandardScaler, mask: np.ndarray, alpha: float):
        self.ridge = ridge
        self.scaler = scaler
        self.mask = mask
        self.alpha = alpha

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.ridge.predict(self.scaler.transform(x[:, self.mask]))


def _usable_columns(x: np.ndarray) -> np.ndarray:
    """Columns that vary and are finite everywhere in the training window."""
    finite = np.isfinite(x).all(axis=0)
    spread = np.zeros(x.shape[1], dtype=bool)
    if x.shape[0] > 1:
        spread = np.nanstd(x, axis=0) > 1e-12
    mask = finite & spread
    if not mask.any():
        mask = np.ones(x.shape[1], dtype=bool)
    return mask


def _fit_ridge(train_x: np.ndarray, train_y: np.ndarray,
               alphas: Sequence[float] = RIDGE_ALPHAS,
               n_origins: int = 0, n_zones: int = 1) -> Tuple[RidgeModel, int]:
    """Fit a ridge, choosing the penalty on a time ordered inner split."""
    finite = np.isfinite(train_y)
    train_x, train_y = train_x[finite], train_y[finite]
    mask = _usable_columns(train_x)
    reduced = train_x[:, mask]
    n_fits = 0
    best_alpha = alphas[len(alphas) // 2]

    if n_origins > 4 and reduced.shape[0] > 50:
        inner_train, inner_validation = _inner_split(n_origins, n_zones)
        x_in, y_in = reduced[inner_train], train_y[inner_train]
        x_va, y_va = reduced[inner_validation], train_y[inner_validation]
        if x_in.shape[0] > 10 and x_va.shape[0] > 10:
            inner_scaler = StandardScaler().fit(x_in)
            x_in_s, x_va_s = inner_scaler.transform(x_in), inner_scaler.transform(x_va)
            best_score = np.inf
            for alpha in alphas:
                model = Ridge(alpha=alpha).fit(x_in_s, y_in)
                n_fits += 1
                score = float(np.mean(np.abs(model.predict(x_va_s) - y_va)))
                if score < best_score:
                    best_score, best_alpha = score, alpha

    scaler = StandardScaler().fit(reduced)
    ridge = Ridge(alpha=best_alpha).fit(scaler.transform(reduced), train_y)
    n_fits += 1
    return RidgeModel(ridge, scaler, mask, best_alpha), n_fits


def _fit_gbm(train_x: np.ndarray, train_y: np.ndarray, seed: int,
             params: Dict[str, object]) -> HistGradientBoostingRegressor:
    finite = np.isfinite(train_y)
    model = HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **params)
    model.fit(train_x[finite], train_y[finite])
    return model


def select_global_gbm_params(train_x: np.ndarray, train_y: np.ndarray,
                             n_origins: int, n_zones: int, seed: int) -> Tuple[Dict, int]:
    """Choose a gradient boosting setting on a time ordered inner split."""
    inner_train, inner_validation = _inner_split(n_origins, n_zones)
    x_in, y_in = train_x[inner_train], train_y[inner_train]
    x_va, y_va = train_x[inner_validation], train_y[inner_validation]
    best, best_score, n_fits = GLOBAL_GBM_GRID[0], np.inf, 0
    for candidate in GLOBAL_GBM_GRID:
        model = _fit_gbm(x_in, y_in, seed, dict(candidate))
        n_fits += 1
        score = float(np.mean(np.abs(model.predict(x_va) - y_va)))
        if score < best_score:
            best_score, best = score, candidate
    return dict(best), n_fits


def fit_predict_global(family: str, train_x: np.ndarray, train_y: np.ndarray,
                       test_x: np.ndarray, n_train_origins: int, n_zones: int,
                       seed: int, gbm_params: Optional[Dict] = None) -> FitOutcome:
    """Fit one model across every zone and predict the test rows."""
    if family == "global_ridge":
        model, n_fits = _fit_ridge(train_x, train_y,
                                   n_origins=n_train_origins, n_zones=n_zones)
        return FitOutcome(model.predict(test_x), n_fits, {"alpha": model.alpha})
    if family == "global_gbm":
        params = dict(gbm_params or GLOBAL_GBM_GRID[1])
        model = _fit_gbm(train_x, train_y, seed, params)
        return FitOutcome(model.predict(test_x), 1, params)
    raise ValueError(f"not a global family: {family}")


def fit_predict_per_zone(family: str, train_x: np.ndarray, train_y: np.ndarray,
                         test_x: np.ndarray, n_train_origins: int, n_test_origins: int,
                         n_zones: int, seed: int) -> FitOutcome:
    """Fit one model per zone and predict that zone's test rows.

    Rows are origin major, so the rows of zone ``z`` are ``z`` then every
    ``n_zones`` after it, which is a strided view rather than a copy.
    """
    if family not in ("zone_ridge", "zone_gbm"):
        raise ValueError(f"not a per zone family: {family}")

    predictions = np.full(test_x.shape[0], np.nan, dtype=np.float64)
    n_fits = 0
    alphas: List[float] = []

    for zone in range(n_zones):
        x_tr = train_x[zone::n_zones]
        y_tr = train_y[zone::n_zones]
        x_te = test_x[zone::n_zones]
        usable = np.isfinite(y_tr)
        if usable.sum() < 30:
            predictions[zone::n_zones] = np.nanmean(y_tr) if usable.any() else 0.0
            continue

        if family == "zone_ridge":
            model, fits = _fit_ridge(x_tr[usable], y_tr[usable],
                                     n_origins=n_train_origins, n_zones=1)
            predictions[zone::n_zones] = model.predict(x_te)
            alphas.append(model.alpha)
            n_fits += fits
        else:
            model = _fit_gbm(x_tr, y_tr, seed + zone, dict(ZONE_GBM_PARAMS))
            predictions[zone::n_zones] = model.predict(x_te)
            n_fits += 1

    chosen = {"median_alpha": float(np.median(alphas))} if alphas else dict(ZONE_GBM_PARAMS)
    return FitOutcome(predictions, n_fits, chosen)


def fit_predict_autoregressive(features: FeatureSet, lag_tensor: np.ndarray,
                               train_origins: np.ndarray, test_origins: np.ndarray,
                               horizon: int, order: int = AR_ORDER) -> FitOutcome:
    """Direct autoregression of order ``order``, fitted per zone by least squares."""
    n_zones = features.n_zones
    predictions = np.full(test_origins.size * n_zones, np.nan, dtype=np.float64)
    n_fits = 0

    for zone in range(n_zones):
        x_tr = lag_tensor[train_origins, zone, :]
        y_tr = features.y[train_origins + horizon, zone]
        keep = np.isfinite(x_tr).all(axis=1) & np.isfinite(y_tr)
        x_te = lag_tensor[test_origins, zone, :]

        if keep.sum() <= order + 1:
            predictions[zone::n_zones] = float(np.nanmean(y_tr)) if keep.any() else 0.0
            continue

        design = np.column_stack([np.ones(keep.sum(), dtype=np.float64),
                                  x_tr[keep].astype(np.float64)])
        coefficients, *_ = np.linalg.lstsq(design, y_tr[keep].astype(np.float64), rcond=None)
        n_fits += 1
        test_design = np.column_stack([np.ones(x_te.shape[0], dtype=np.float64),
                                       np.nan_to_num(x_te).astype(np.float64)])
        predictions[zone::n_zones] = test_design @ coefficients

    return FitOutcome(predictions, n_fits, {"order": order})
