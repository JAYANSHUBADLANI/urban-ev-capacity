"""The model families, with the baselines checked against hand computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uev.features import FeatureSet
from uev.models import (AR_ORDER, GLOBAL_FAMILIES, PER_ZONE_FAMILIES, RidgeModel,
                        _fit_ridge, _inner_split, _usable_columns, ar_lag_tensor,
                        baseline_predictions, fit_predict_autoregressive,
                        fit_predict_global, fit_predict_per_zone)


@pytest.fixture
def toy_features():
    hours, zones = 400, 3
    index = pd.date_range("2022-09-01", periods=hours, freq="h")
    grid = np.arange(hours)[:, None] + np.arange(zones)[None, :] * 100.0
    return FeatureSet(target="occupancy", index=index, zone_ids=[1, 2, 3],
                      y=grid.astype(np.float32))


def test_last_observation_returns_the_value_at_the_origin(toy_features):
    origins = np.array([200, 201])
    out = baseline_predictions("last_observation", toy_features, origins, 1)
    assert out.reshape(2, 3)[0].tolist() == toy_features.y[200].tolist()


def test_seasonal_naive_daily_looks_back_one_day_from_the_target(toy_features):
    origins = np.array([200])
    out = baseline_predictions("seasonal_naive_daily", toy_features, origins, 6)
    assert out.tolist() == toy_features.y[200 + 6 - 24].tolist()


def test_seasonal_naive_weekly_looks_back_one_week_from_the_target(toy_features):
    origins = np.array([200])
    out = baseline_predictions("seasonal_naive_weekly", toy_features, origins, 1)
    assert out.tolist() == toy_features.y[200 + 1 - 168].tolist()


def test_seasonal_naive_daily_is_exact_at_a_twenty_four_hour_horizon(toy_features):
    origins = np.array([200])
    out = baseline_predictions("seasonal_naive_daily", toy_features, origins, 24)
    assert out.tolist() == toy_features.y[200].tolist()


def test_a_baseline_that_would_read_the_future_is_refused(toy_features):
    with pytest.raises(ValueError, match="beyond the origin"):
        baseline_predictions("seasonal_naive_daily", toy_features, np.array([200]), 48)


def test_a_baseline_before_the_series_start_is_refused(toy_features):
    with pytest.raises(ValueError, match="before the series starts"):
        baseline_predictions("seasonal_naive_weekly", toy_features, np.array([10]), 1)


def test_an_unknown_baseline_is_refused(toy_features):
    with pytest.raises(ValueError, match="not a baseline family"):
        baseline_predictions("nonsense", toy_features, np.array([200]), 1)


def test_ar_lag_tensor_holds_the_right_lags(toy_features):
    tensor = ar_lag_tensor(toy_features, order=4)
    assert tensor.shape == (400, 3, 4)
    assert np.allclose(tensor[100, :, 0], toy_features.y[100])
    assert np.allclose(tensor[100, :, 3], toy_features.y[97])


def test_ar_lag_tensor_leaves_the_warmup_undefined(toy_features):
    tensor = ar_lag_tensor(toy_features, order=4)
    assert np.isnan(tensor[0, :, 1]).all()


def test_autoregression_recovers_a_deterministic_linear_series(toy_features):
    """The toy series is exactly linear in time, so a direct AR should be exact."""
    tensor = ar_lag_tensor(toy_features)
    train = np.arange(AR_ORDER, 300)
    test = np.arange(300, 320)
    outcome = fit_predict_autoregressive(toy_features, tensor, train, test, horizon=1)
    expected = toy_features.y[test + 1].reshape(-1)
    assert np.allclose(outcome.predictions, expected, atol=1e-3)


def test_autoregression_fits_one_model_per_zone(toy_features):
    tensor = ar_lag_tensor(toy_features)
    outcome = fit_predict_autoregressive(toy_features, tensor, np.arange(AR_ORDER, 300),
                                         np.arange(300, 310), horizon=1)
    assert outcome.n_fits == toy_features.n_zones


def test_usable_columns_drops_a_constant_column():
    values = np.column_stack([np.arange(10.0), np.ones(10)])
    assert _usable_columns(values).tolist() == [True, False]


def test_usable_columns_keeps_everything_when_nothing_varies():
    assert _usable_columns(np.ones((5, 3))).all()


def test_inner_split_holds_out_the_most_recent_origins():
    train, validation = _inner_split(1000, 10, validation_origins=100)
    assert train == slice(0, 9000) and validation == slice(9000, 10000)


def test_inner_split_never_takes_more_than_a_quarter():
    train, validation = _inner_split(40, 1, validation_origins=336)
    assert validation.start >= 30


def test_ridge_recovers_a_linear_relationship():
    generator = np.random.default_rng(0)
    x = generator.normal(size=(500, 3))
    y = 2.0 * x[:, 0] - 1.0 * x[:, 1] + 0.5
    model, _ = _fit_ridge(x, y, alphas=(1e-6,), n_origins=0)
    assert np.allclose(model.predict(x), y, atol=1e-3)


def test_ridge_ignores_a_constant_column():
    generator = np.random.default_rng(1)
    x = np.column_stack([generator.normal(size=400), np.full(400, 7.0)])
    y = 3.0 * x[:, 0]
    model, _ = _fit_ridge(x, y, alphas=(1e-6,), n_origins=0)
    assert model.mask.tolist() == [True, False]
    assert np.allclose(model.predict(x), y, atol=1e-3)


def test_ridge_reports_the_chosen_penalty():
    generator = np.random.default_rng(2)
    x = generator.normal(size=(2000, 4))
    y = x[:, 0] + generator.normal(scale=0.1, size=2000)
    model, n_fits = _fit_ridge(x, y, n_origins=200, n_zones=10)
    assert isinstance(model, RidgeModel) and model.alpha > 0
    assert n_fits >= 2


def test_global_families_produce_one_prediction_per_test_row():
    generator = np.random.default_rng(3)
    train_x = generator.normal(size=(600, 5)).astype(np.float32)
    train_y = train_x[:, 0] * 2.0
    test_x = generator.normal(size=(90, 5)).astype(np.float32)
    for family in GLOBAL_FAMILIES:
        outcome = fit_predict_global(family, train_x, train_y, test_x, 200, 3, seed=0)
        assert outcome.predictions.shape == (90,)
        assert outcome.n_fits >= 1


def test_an_unknown_global_family_is_refused():
    with pytest.raises(ValueError, match="not a global family"):
        fit_predict_global("nope", np.zeros((10, 2)), np.zeros(10),
                           np.zeros((2, 2)), 5, 2, seed=0)


def test_per_zone_families_fit_one_model_per_zone():
    generator = np.random.default_rng(4)
    n_zones, n_origins = 3, 300
    train_x = generator.normal(size=(n_origins * n_zones, 4)).astype(np.float32)
    train_y = train_x[:, 0] * 2.0
    test_x = generator.normal(size=(10 * n_zones, 4)).astype(np.float32)
    for family in ("zone_ridge", "zone_gbm"):
        outcome = fit_predict_per_zone(family, train_x, train_y, test_x,
                                       n_origins, 10, n_zones, seed=0)
        assert outcome.predictions.shape == (30,)
        assert not np.isnan(outcome.predictions).any()
        assert outcome.n_fits >= n_zones


def test_per_zone_fitting_uses_only_that_zone_s_rows():
    """Rows are origin major, so zone z owns every n_zones-th row."""
    n_zones, n_origins = 2, 200
    train_x = np.zeros((n_origins * n_zones, 1), dtype=np.float32)
    train_x[0::2, 0] = np.arange(n_origins)          # zone 0 varies
    train_x[1::2, 0] = 0.0                           # zone 1 is constant
    train_y = np.zeros(n_origins * n_zones)
    train_y[0::2] = np.arange(n_origins) * 3.0
    train_y[1::2] = 50.0
    test_x = np.zeros((2 * n_zones, 1), dtype=np.float32)
    test_x[0::2, 0] = [10.0, 20.0]
    outcome = fit_predict_per_zone("zone_ridge", train_x, train_y, test_x,
                                   n_origins, 2, n_zones, seed=0)
    assert np.allclose(outcome.predictions[1::2], 50.0, atol=1e-6)


def test_an_unknown_per_zone_family_is_refused():
    with pytest.raises(ValueError, match="not a per zone family"):
        fit_predict_per_zone("nope", np.zeros((10, 2)), np.zeros(10),
                             np.zeros((2, 2)), 5, 1, 2, seed=0)


def test_a_zone_with_too_little_history_falls_back_to_its_mean():
    n_zones = 2
    train_x = np.zeros((10 * n_zones, 2), dtype=np.float32)
    train_y = np.full(10 * n_zones, 4.0)
    test_x = np.zeros((2 * n_zones, 2), dtype=np.float32)
    outcome = fit_predict_per_zone("zone_ridge", train_x, train_y, test_x,
                                   10, 2, n_zones, seed=0)
    assert np.allclose(outcome.predictions, 4.0)
