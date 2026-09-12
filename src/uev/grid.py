"""The rolling origin forecasting grid.

One cell is a target, a horizon, a fold and a model family. Cells are keyed by a
stable hash of that tuple, written to disk as they finish, and skipped on a
later run if the key is already present, so a run that stops part way resumes
rather than restarting.

Predictions are clipped to the physically feasible range before scoring:
nothing below zero, and for occupancy nothing above the charging points
installed in the zone. The clip uses only information that is known in advance
and is applied identically to every family, so it does not favour one over
another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG, EXPECTED_HOURS, HORIZONS, TARGETS, cell_key
from .cv import Fold, make_folds, holdout_origins, train_origins
from .features import WARMUP, FeatureSet, build_features
from .io_load import Bundle
from .logging_utils import get_logger
from .metrics import evaluate, seasonal_naive_denominator
from .models import (BASELINE_FAMILIES, GLOBAL_FAMILIES, PER_ZONE_FAMILIES,
                     ar_lag_tensor, baseline_predictions, fit_predict_autoregressive,
                     fit_predict_global, fit_predict_per_zone, select_global_gbm_params)
from .seeds import derive_seed

LOG = get_logger("grid")

FAMILIES: Tuple[str, ...] = BASELINE_FAMILIES + PER_ZONE_FAMILIES + GLOBAL_FAMILIES

GRID_METRICS = "grid_metrics.csv"
GRID_ZONE_METRICS = "grid_zone_metrics.csv"
GRID_HOUR_METRICS = "grid_hour_metrics.csv"
GRID_DOW_METRICS = "grid_dow_metrics.csv"


def make_cell_key(target: str, horizon: int, fold: int, family: str) -> str:
    return cell_key(target=target, horizon=horizon, fold=fold, family=family,
                    cv=CONFIG.cv.n_folds, test_hours=CONFIG.cv.test_hours)


def expected_cells() -> List[Dict]:
    """Every cell the full grid contains."""
    folds = make_folds(EXPECTED_HOURS)
    out = []
    for target in TARGETS:
        for horizon in HORIZONS:
            for fold in folds:
                for family in FAMILIES:
                    out.append({"target": target, "horizon": horizon,
                                "fold": fold.fold, "family": family,
                                "cell_key": make_cell_key(target, horizon, fold.fold, family)})
    return out


def zone_denominators(features: FeatureSet, fold: Fold) -> np.ndarray:
    """Seasonal naive scaling denominator per zone, from training data only."""
    train = features.y[fold.train_start:fold.train_end]
    return np.array([seasonal_naive_denominator(train[:, z])
                     for z in range(features.n_zones)], dtype=float)


def ceiling_for(target: str, capacity: np.ndarray) -> Optional[np.ndarray]:
    """Upper bound for the target, where one exists.

    Occupancy counts points in use and duration accumulates at most one hour per
    point, so both are capped by the installed points. Energy has no such cap.
    """
    if target in ("occupancy", "duration"):
        return capacity
    return None


def clip_predictions(predictions: np.ndarray, target: str,
                     capacity_rows: np.ndarray) -> np.ndarray:
    """Clip to the feasible range, using only information known in advance."""
    out = np.clip(predictions, 0.0, None)
    ceiling = ceiling_for(target, capacity_rows)
    if ceiling is not None:
        out = np.minimum(out, ceiling)
    return out


@dataclass
class CellResult:
    pooled: Dict
    per_zone: List[Dict]
    per_hour: List[Dict]
    per_dow: List[Dict]


def score_cell(target: str, horizon: int, fold: Fold, family: str,
               actual: np.ndarray, predicted: np.ndarray,
               zone_ids: np.ndarray, denominators_row: np.ndarray,
               utilisation_row: np.ndarray, hours: np.ndarray, dows: np.ndarray,
               n_fits: int, chosen: Dict, key: str) -> CellResult:
    """Pooled and disaggregated metrics for one finished cell."""
    pooled = evaluate(actual, predicted, denominators_row, utilisation_row)
    pooled.update({
        "cell_key": key, "target": target, "horizon": horizon, "fold": fold.fold,
        "family": family, "n_fits": n_fits,
        "chosen": ";".join(f"{k}={v}" for k, v in chosen.items()),
        "n_train_hours": fold.n_train_hours,
    })

    frame = pd.DataFrame({
        "zone_id": zone_ids,
        "actual": actual,
        "predicted": predicted,
        "denominator": denominators_row,
        "utilisation": utilisation_row,
        "hour_of_day": hours,
        "day_of_week": dows,
    })
    frame["abs_error"] = (frame["actual"] - frame["predicted"]).abs()
    frame["scaled_error"] = np.where(np.isfinite(frame["denominator"])
                                     & (frame["denominator"] > 0),
                                     frame["abs_error"] / frame["denominator"], np.nan)
    frame["sat_weight"] = 1.0 + 3.0 * frame["utilisation"].clip(0.0, 1.0)
    frame["weighted_error"] = frame["sat_weight"] * frame["abs_error"]

    base = {"target": target, "horizon": horizon, "fold": fold.fold, "family": family}

    grouped = frame.groupby("zone_id", sort=True)
    per_zone = []
    for zone, part in grouped:
        per_zone.append({**base, "zone_id": int(zone), "n": len(part),
                         "mae": float(part["abs_error"].mean()),
                         "mase": float(part["scaled_error"].mean()),
                         "sat_wmae": float(part["weighted_error"].sum()
                                           / part["sat_weight"].sum()),
                         "bias": float((part["predicted"] - part["actual"]).mean())})

    per_hour = []
    for hour, part in frame.groupby("hour_of_day", sort=True):
        per_hour.append({**base, "hour_of_day": int(hour), "n": len(part),
                         "mae": float(part["abs_error"].mean()),
                         "mase": float(part["scaled_error"].mean())})

    per_dow = []
    for dow, part in frame.groupby("day_of_week", sort=True):
        per_dow.append({**base, "day_of_week": int(dow), "n": len(part),
                        "mae": float(part["abs_error"].mean()),
                        "mase": float(part["scaled_error"].mean())})

    return CellResult(pooled, per_zone, per_hour, per_dow)


class GridRunner:
    """Runs the grid, one target at a time, reusing designs across families."""

    def __init__(self, bundle: Bundle, done_keys: Optional[set] = None):
        self.bundle = bundle
        self.done = set(done_keys or set())
        self.folds = make_folds(len(bundle.index))
        self.capacity = bundle.zones.set_index("zone_id")["capacity_points"] \
            .reindex(list(bundle.zones["zone_id"])).to_numpy(dtype=float)
        occupancy = bundle.long.pivot(index="timestamp", columns="zone_id", values="occupancy")
        occupancy = occupancy.reindex(index=bundle.index, columns=list(bundle.zones["zone_id"]))
        self.utilisation = (occupancy.to_numpy(dtype=np.float32)
                            / self.capacity[None, :].astype(np.float32))
        self.hours = bundle.index.hour.to_numpy()
        self.dows = bundle.index.dayofweek.to_numpy()
        self._gbm_params: Dict[Tuple[str, int], Dict] = {}

    def pending(self, target: str, horizon: int, fold: Fold) -> List[str]:
        return [family for family in FAMILIES
                if make_cell_key(target, horizon, fold.fold, family) not in self.done]

    def run_group(self, features: FeatureSet, lag_tensor: np.ndarray,
                  target: str, horizon: int, fold: Fold,
                  families: Iterable[str]) -> List[CellResult]:
        """Fit every pending family for one target, horizon and fold."""
        families = list(families)
        n_zones = features.n_zones
        origins_train = train_origins(fold, horizon, WARMUP)
        origins_test = holdout_origins(fold, horizon, WARMUP, features.n_hours)
        if origins_train.size == 0 or origins_test.size == 0:
            return []

        zone_ids = np.tile(np.asarray(features.zone_ids), origins_test.size)
        denominators = zone_denominators(features, fold)
        denominators_row = np.tile(denominators, origins_test.size)
        capacity_row = np.tile(self.capacity, origins_test.size)
        utilisation_row = self.utilisation[origins_test + horizon].reshape(-1)
        hours_row = np.repeat(self.hours[origins_test + horizon], n_zones)
        dows_row = np.repeat(self.dows[origins_test + horizon], n_zones)
        actual = features.y[origins_test + horizon].reshape(-1).astype(float)

        need_design = any(f in families for f in GLOBAL_FAMILIES + ("zone_ridge", "zone_gbm"))
        train_x = test_x = None
        train_y = None
        static_mask = None
        if need_design:
            train_x, train_y = features.design(origins_train, horizon)
            test_x, _ = features.design(origins_test, horizon)
            names = features.feature_names()
            static_names = set(features.static_blocks)
            static_mask = np.array([name not in static_names for name in names])

            # Zone identity for the pooled model: the level and spread of the
            # zone in the training window only.
            window = features.y[fold.train_start:fold.train_end]
            zone_mean = np.nan_to_num(np.nanmean(window, axis=0)).astype(np.float32)
            zone_std = np.nan_to_num(np.nanstd(window, axis=0)).astype(np.float32)
            train_extra = np.column_stack([np.tile(zone_mean, origins_train.size),
                                           np.tile(zone_std, origins_train.size)])
            test_extra = np.column_stack([np.tile(zone_mean, origins_test.size),
                                          np.tile(zone_std, origins_test.size)])
            train_x_global = np.hstack([train_x, train_extra]).astype(np.float32)
            test_x_global = np.hstack([test_x, test_extra]).astype(np.float32)

        results: List[CellResult] = []
        for family in families:
            key = make_cell_key(target, horizon, fold.fold, family)
            seed = derive_seed(f"{target}:{horizon}:{fold.fold}:{family}")

            if family in BASELINE_FAMILIES:
                predictions = baseline_predictions(family, features, origins_test, horizon)
                n_fits, chosen = 0, {}
            elif family == "autoregressive":
                outcome = fit_predict_autoregressive(features, lag_tensor, origins_train,
                                                     origins_test, horizon)
                predictions, n_fits, chosen = outcome.predictions, outcome.n_fits, outcome.chosen
            elif family in GLOBAL_FAMILIES:
                params = None
                if family == "global_gbm":
                    cached = self._gbm_params.get((target, horizon))
                    if cached is None:
                        cached, _ = select_global_gbm_params(
                            train_x_global, train_y, origins_train.size, n_zones, seed)
                        self._gbm_params[(target, horizon)] = cached
                        LOG.info("gbm setting for %s h=%d: %s", target, horizon, cached)
                    params = cached
                outcome = fit_predict_global(family, train_x_global, train_y, test_x_global,
                                             origins_train.size, n_zones, seed, params)
                predictions, n_fits, chosen = outcome.predictions, outcome.n_fits, outcome.chosen
            else:
                outcome = fit_predict_per_zone(family, train_x[:, static_mask], train_y,
                                               test_x[:, static_mask], origins_train.size,
                                               origins_test.size, n_zones, seed)
                predictions, n_fits, chosen = outcome.predictions, outcome.n_fits, outcome.chosen

            predictions = clip_predictions(np.asarray(predictions, dtype=float),
                                           target, capacity_row)
            results.append(score_cell(target, horizon, fold, family, actual, predictions,
                                      zone_ids, denominators_row, utilisation_row,
                                      hours_row, dows_row, n_fits, chosen, key))
            self.done.add(key)

        return results
