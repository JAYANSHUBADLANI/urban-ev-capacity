"""Error metrics, including the cases where they are undefined."""

from __future__ import annotations

import numpy as np
import pytest

from uev.metrics import (bias, evaluate, mae, mape, mase, rmse,
                         saturation_weighted_mae, seasonal_naive_denominator)


def test_mae_of_a_perfect_forecast_is_zero():
    values = np.array([1.0, 2.0, 3.0])
    assert mae(values, values) == 0.0


def test_mae_is_the_mean_absolute_gap():
    assert mae(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(1.5)


def test_rmse_penalises_the_larger_error_more_than_mae():
    actual = np.array([0.0, 0.0])
    predicted = np.array([0.0, 4.0])
    assert rmse(actual, predicted) > mae(actual, predicted)


def test_bias_is_signed():
    assert bias(np.array([1.0, 1.0]), np.array([2.0, 2.0])) == pytest.approx(1.0)
    assert bias(np.array([2.0, 2.0]), np.array([1.0, 1.0])) == pytest.approx(-1.0)


def test_mape_skips_zero_actuals_and_reports_coverage():
    actual = np.array([0.0, 10.0, 20.0])
    predicted = np.array([5.0, 11.0, 18.0])
    out = mape(actual, predicted)
    assert out["mape_coverage"] == pytest.approx(2 / 3)
    assert out["mape"] == pytest.approx(((0.1 + 0.1) / 2) * 100)


def test_mape_is_undefined_when_every_actual_is_zero():
    out = mape(np.zeros(3), np.ones(3))
    assert np.isnan(out["mape"]) and out["mape_coverage"] == 0.0


def test_seasonal_denominator_matches_the_hand_computation():
    series = np.arange(48, dtype=float)
    assert seasonal_naive_denominator(series, period=24) == pytest.approx(24.0)


def test_seasonal_denominator_is_undefined_for_a_constant_series():
    assert np.isnan(seasonal_naive_denominator(np.ones(48), period=24))


def test_seasonal_denominator_is_undefined_for_a_short_series():
    assert np.isnan(seasonal_naive_denominator(np.arange(10, dtype=float), period=24))


def test_mase_of_one_means_as_good_as_seasonal_naive():
    actual = np.array([10.0, 20.0])
    predicted = np.array([12.0, 18.0])
    out = mase(actual, predicted, np.array([2.0, 2.0]))
    assert out["mase"] == pytest.approx(1.0)


def test_mase_excludes_zones_without_a_usable_denominator():
    actual = np.array([1.0, 2.0])
    predicted = np.array([2.0, 4.0])
    out = mase(actual, predicted, np.array([np.nan, 2.0]))
    assert out["mase_coverage"] == pytest.approx(0.5)
    assert out["mase"] == pytest.approx(1.0)


def test_mase_is_undefined_when_no_denominator_is_usable():
    out = mase(np.ones(2), np.ones(2), np.array([np.nan, 0.0]))
    assert np.isnan(out["mase"]) and out["mase_coverage"] == 0.0


def test_saturation_weighting_costs_more_at_a_full_zone():
    actual = np.array([10.0, 10.0])
    predicted = np.array([11.0, 11.0])
    empty = saturation_weighted_mae(actual, predicted, np.array([0.0, 0.0]))
    full = saturation_weighted_mae(actual, predicted, np.array([1.0, 1.0]))
    assert empty == pytest.approx(1.0)
    assert full == pytest.approx(1.0)


def test_saturation_weighting_shifts_the_mix_towards_the_saturated_row():
    actual = np.array([10.0, 10.0])
    predicted = np.array([10.0, 14.0])          # the error sits at the saturated zone
    utilisation = np.array([0.0, 1.0])
    weighted = saturation_weighted_mae(actual, predicted, utilisation)
    assert weighted > mae(actual, predicted)


def test_saturation_weighting_discounts_an_error_at_an_empty_zone():
    actual = np.array([10.0, 10.0])
    predicted = np.array([14.0, 10.0])          # the error sits at the empty zone
    utilisation = np.array([0.0, 1.0])
    assert saturation_weighted_mae(actual, predicted, utilisation) < mae(actual, predicted)


def test_saturation_weighting_clips_utilisation_to_the_unit_interval():
    actual, predicted = np.array([1.0]), np.array([2.0])
    assert (saturation_weighted_mae(actual, predicted, np.array([5.0]))
            == pytest.approx(saturation_weighted_mae(actual, predicted, np.array([1.0]))))


def test_evaluate_returns_every_metric():
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 2.0, 4.0])
    out = evaluate(actual, predicted, np.full(3, 2.0), np.array([0.1, 0.2, 0.9]))
    for key in ("n", "mae", "rmse", "bias", "mape", "mape_coverage",
                "mase", "mase_coverage", "sat_wmae"):
        assert key in out
    assert out["n"] == 3


def test_evaluate_without_a_denominator_marks_mase_undefined():
    out = evaluate(np.ones(2), np.ones(2))
    assert np.isnan(out["mase"]) and np.isnan(out["sat_wmae"])
