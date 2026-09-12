"""Change point detection, checked against series with a planted break."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uev.breaks import (MIN_SEGMENT, _Cost, _max_gain, binary_segmentation,
                        deseasonalise_daily, local_break_pvalue, pelt,
                        segment_means, windows_around)


def planted(n_before=60, n_after=60, level=5.0, noise=0.2, seed=0):
    generator = np.random.default_rng(seed)
    return np.concatenate([generator.normal(0.0, noise, n_before),
                           generator.normal(level, noise, n_after)])


def test_segment_cost_of_a_constant_segment_is_zero():
    cost = _Cost(np.full(10, 3.0))
    assert cost(0, 10) == pytest.approx(0.0)


def test_segment_cost_matches_the_sum_of_squared_deviations():
    values = np.array([1.0, 2.0, 3.0, 10.0])
    cost = _Cost(values)
    expected = float(((values - values.mean()) ** 2).sum())
    assert cost(0, 4) == pytest.approx(expected)


def test_segment_cost_of_an_empty_segment_is_zero():
    assert _Cost(np.arange(5.0))(2, 2) == pytest.approx(0.0)


def test_splitting_at_the_break_removes_almost_all_the_cost():
    values = planted()
    cost = _Cost(values)
    assert cost(0, 60) + cost(60, 120) < 0.01 * cost(0, 120)


def test_max_gain_finds_the_planted_break():
    _, position = _max_gain(planted())
    assert abs(position - 60) <= 1


def test_max_gain_of_pure_noise_is_small():
    generator = np.random.default_rng(1)
    gain, _ = _max_gain(generator.normal(0.0, 1.0, 200))
    total = _Cost(generator.normal(0.0, 1.0, 200))(0, 200)
    assert gain < 0.3 * total


def test_pelt_finds_a_single_planted_break():
    breaks = pelt(planted(), penalty=5.0)
    assert len(breaks) >= 1
    assert min(abs(b - 60) for b in breaks) <= 2


def test_pelt_finds_nothing_in_a_flat_series():
    assert pelt(np.zeros(120) + 1.0, penalty=5.0) == []


def test_pelt_respects_the_minimum_segment_length():
    breaks = pelt(planted(), penalty=1.0)
    edges = [0] + breaks + [120]
    assert all(b - a >= MIN_SEGMENT for a, b in zip(edges, edges[1:]))


def test_a_larger_penalty_yields_no_more_breaks():
    values = planted(n_before=80, n_after=80, level=3.0, noise=1.0)
    assert len(pelt(values, penalty=200.0)) <= len(pelt(values, penalty=5.0))


def test_binary_segmentation_finds_the_planted_break():
    breaks = binary_segmentation(planted(), max_breaks=3)
    assert breaks and min(abs(b - 60) for b in breaks) <= 2


def test_binary_segmentation_finds_two_planted_breaks():
    generator = np.random.default_rng(2)
    values = np.concatenate([generator.normal(0, 0.2, 50),
                             generator.normal(4, 0.2, 50),
                             generator.normal(0, 0.2, 50)])
    breaks = binary_segmentation(values, max_breaks=4)
    assert len(breaks) >= 2
    assert min(abs(b - 50) for b in breaks) <= 2
    assert min(abs(b - 100) for b in breaks) <= 2


def test_binary_segmentation_stops_on_a_flat_series():
    assert binary_segmentation(np.full(100, 2.0)) == []


def test_the_two_detectors_agree_on_a_clear_break():
    values = planted()
    from_pelt = pelt(values, penalty=5.0)
    from_binary = binary_segmentation(values, max_breaks=3)
    assert min(abs(a - b) for a in from_pelt for b in from_binary) <= 2


def test_segment_means_describe_each_segment():
    values = np.concatenate([np.zeros(10), np.full(10, 5.0)])
    segments = segment_means(values, [10])
    assert segments[0][2] == pytest.approx(0.0)
    assert segments[1][2] == pytest.approx(5.0)


def test_windows_around_bounds_each_break_by_its_neighbours():
    assert windows_around([30, 60], 100) == [(30, 0, 60), (60, 30, 100)]


def test_windows_around_a_single_break_spans_the_series():
    assert windows_around([40], 100) == [(40, 0, 100)]


def test_a_planted_break_is_significant():
    outcome = local_break_pvalue(planted(), 0, 120, n_draws=200, label="planted")
    assert outcome["p_value"] < 0.05


def test_pure_noise_is_not_significant():
    generator = np.random.default_rng(5)
    outcome = local_break_pvalue(generator.normal(0, 1, 200), 0, 200,
                                 n_draws=200, label="noise")
    assert outcome["p_value"] > 0.05


def test_significance_is_undefined_for_a_window_that_is_too_short():
    outcome = local_break_pvalue(np.arange(5.0), 0, 5, n_draws=10, label="short")
    assert np.isnan(outcome["p_value"])


def test_the_bootstrap_is_reproducible():
    values = planted()
    first = local_break_pvalue(values, 0, 120, n_draws=100, label="repeat")
    second = local_break_pvalue(values, 0, 120, n_draws=100, label="repeat")
    assert first["p_value"] == second["p_value"]


def test_deseasonalising_removes_the_day_of_week_pattern():
    index = pd.date_range("2022-09-05", periods=70, freq="D")
    effect = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])[index.dayofweek]
    series = pd.Series(10.0 + effect, index=index)
    adjusted = deseasonalise_daily(series)
    assert adjusted.std() < 1e-9


def test_deseasonalising_preserves_the_mean():
    index = pd.date_range("2022-09-05", periods=70, freq="D")
    generator = np.random.default_rng(3)
    series = pd.Series(generator.normal(10, 2, 70), index=index)
    assert deseasonalise_daily(series).mean() == pytest.approx(series.mean())


def test_deseasonalising_keeps_a_level_shift():
    index = pd.date_range("2022-09-05", periods=80, freq="D")
    values = np.concatenate([np.full(40, 10.0), np.full(40, 20.0)])
    adjusted = deseasonalise_daily(pd.Series(values, index=index))
    assert adjusted.iloc[:40].mean() < adjusted.iloc[40:].mean() - 5
