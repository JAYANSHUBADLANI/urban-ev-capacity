"""Monitors, calibration, and detection of an injected shift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uev.config import MonitorConfig
from uev.monitor import (MonitorBacktest, binned_ks, calibrate_threshold,
                         first_alert, input_monitor_series, psi,
                         residual_monitor_series)


def test_psi_of_identical_distributions_is_zero():
    counts = np.array([10.0, 20.0, 30.0])
    assert psi(counts, counts) == pytest.approx(0.0, abs=1e-9)


def test_psi_is_positive_when_mass_moves():
    assert psi(np.array([50.0, 50.0]), np.array([90.0, 10.0])) > 0


def test_psi_grows_with_the_size_of_the_shift():
    base = np.array([50.0, 50.0])
    small = psi(base, np.array([60.0, 40.0]))
    large = psi(base, np.array([95.0, 5.0]))
    assert large > small


def test_psi_is_scale_invariant():
    a = np.array([10.0, 30.0])
    assert psi(a, np.array([20.0, 60.0])) == pytest.approx(0.0, abs=1e-9)


def test_ks_of_identical_distributions_is_zero():
    counts = np.array([5.0, 5.0, 5.0])
    assert binned_ks(counts, counts) == pytest.approx(0.0)


def test_ks_is_bounded_by_one():
    assert 0.0 <= binned_ks(np.array([10.0, 0.0]), np.array([0.0, 10.0])) <= 1.0


def test_ks_detects_a_complete_separation():
    assert binned_ks(np.array([10.0, 0.0]), np.array([0.0, 10.0])) == pytest.approx(1.0)


def test_input_monitor_detects_a_known_injected_shift():
    """A level shift injected into a stationary series must raise both monitors."""
    generator = np.random.default_rng(7)
    hours, zones = 1200, 20
    values = generator.normal(10.0, 1.0, size=(hours, zones)).astype(np.float32)
    values[800:] += 6.0                          # the injected shift

    statistics = input_monitor_series({"feature": values}, slice(0, 400), window=168)
    for name in ("input_psi", "input_ks"):
        before = np.nanmax(statistics[name][400:790])
        after = np.nanmax(statistics[name][968:])
        assert after > before, name


def test_input_monitor_stays_quiet_without_a_shift():
    generator = np.random.default_rng(11)
    values = generator.normal(10.0, 1.0, size=(1200, 20)).astype(np.float32)
    statistics = input_monitor_series({"feature": values}, slice(0, 400), window=168)
    assert np.nanmax(statistics["input_psi"][400:]) < 0.25


def test_input_monitor_scales_with_the_size_of_the_shift():
    generator = np.random.default_rng(3)
    base = generator.normal(10.0, 1.0, size=(1000, 20)).astype(np.float32)
    peaks = []
    for size in (1.0, 3.0, 9.0):
        values = base.copy()
        values[700:] += size
        statistics = input_monitor_series({"feature": values}, slice(0, 400), window=168)
        peaks.append(np.nanmax(statistics["input_psi"][868:]))
    assert peaks[0] < peaks[1] < peaks[2]


def test_residual_monitors_react_to_a_residual_shift():
    generator = np.random.default_rng(5)
    hours = 1200
    residual = generator.normal(0.0, 1.0, hours)
    absolute = np.abs(residual)
    residual[800:] += 4.0
    absolute[800:] += 4.0
    statistics = residual_monitor_series(residual, absolute, slice(0, 400), window=168)
    for name in ("residual_mean_cusum", "residual_abs_ewma"):
        assert np.nanmax(statistics[name][968:]) > np.nanmax(statistics[name][400:790]), name


def test_calibration_achieves_close_to_the_target_rate():
    generator = np.random.default_rng(1)
    statistic = generator.normal(size=4000)
    stable = np.zeros(4000, dtype=bool)
    stable[:3000] = True
    outcome = calibrate_threshold(statistic, stable, 5.0)
    assert outcome["achieved_far_per_1000h"] == pytest.approx(5.0, abs=1.0)


def test_a_stricter_target_gives_a_higher_threshold():
    generator = np.random.default_rng(2)
    statistic = generator.normal(size=3000)
    stable = np.ones(3000, dtype=bool)
    strict = calibrate_threshold(statistic, stable, 0.5)["threshold"]
    loose = calibrate_threshold(statistic, stable, 5.0)["threshold"]
    assert strict >= loose


def test_calibration_of_an_empty_window_is_undefined():
    outcome = calibrate_threshold(np.arange(10.0), np.zeros(10, dtype=bool), 1.0)
    assert np.isnan(outcome["threshold"])


def test_first_alert_finds_the_first_exceedance():
    statistic = np.array([0.0, 0.0, 5.0, 0.0, 9.0])
    assert first_alert(statistic, 1.0, 0) == 2


def test_first_alert_ignores_exceedances_before_the_start():
    statistic = np.array([9.0, 0.0, 5.0])
    assert first_alert(statistic, 1.0, 1) == 2


def test_first_alert_returns_nothing_when_never_exceeded():
    assert first_alert(np.zeros(5), 1.0, 0) is None


def test_first_alert_with_an_undefined_threshold_returns_nothing():
    assert first_alert(np.ones(5), float("nan"), 0) is None


def test_backtest_reports_every_monitor_at_every_rate():
    index = pd.DatetimeIndex(pd.date_range("2022-11-08", periods=2000, freq="h"))
    settings = MonitorConfig(stable_start="2022-11-08 00:00",
                             stable_end="2022-12-08 23:00",
                             primary_break="2022-12-09 00:00")
    generator = np.random.default_rng(4)
    statistics = {"input_psi": generator.normal(size=2000),
                  "residual_abs_ewma": generator.normal(size=2000)}
    frame = MonitorBacktest(statistics, index, settings).evaluate()
    assert len(frame) == 2 * len(settings.target_far_per_1000h)
    assert set(frame["kind"]) == {"input", "residual"}
