"""Feature construction, and the leakage guards on it.

The convention under test: for a target at ``t + h`` the origin is ``t``, and no
feature may read any series later than ``t``. The calendar is the documented
exception and is taken at the target time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import needs_data
from uev.features import (LAG_OFFSETS, WARMUP, _rolling, _shift_rows,
                          build_features)


def test_shift_exposes_only_the_past():
    matrix = np.arange(12, dtype=np.float32).reshape(6, 2)
    shifted = _shift_rows(matrix, 2)
    assert np.isnan(shifted[:2]).all()
    assert np.array_equal(shifted[2:], matrix[:-2])


def test_shift_of_zero_is_the_identity():
    matrix = np.arange(6, dtype=np.float32).reshape(3, 2)
    assert np.array_equal(_shift_rows(matrix, 0), matrix)


def test_a_negative_shift_is_refused():
    with pytest.raises(ValueError, match="future"):
        _shift_rows(np.zeros((3, 2), dtype=np.float32), -1)


def test_rolling_mean_ends_at_the_current_row():
    matrix = np.arange(10, dtype=np.float32).reshape(5, 2)
    rolled = _rolling(matrix, 2, "mean")
    assert np.isnan(rolled[0]).all()
    assert rolled[1, 0] == pytest.approx(1.0)
    assert rolled[4, 0] == pytest.approx(7.0)


def test_rolling_requires_a_full_window():
    matrix = np.arange(8, dtype=np.float32).reshape(4, 2)
    assert np.isnan(_rolling(matrix, 3, "mean")[:2]).all()


def test_rolling_supports_the_four_statistics():
    matrix = np.arange(8, dtype=np.float32).reshape(4, 2)
    for how in ("mean", "std", "max", "min"):
        assert _rolling(matrix, 2, how).shape == matrix.shape


def test_unknown_rolling_statistic_is_refused():
    with pytest.raises(ValueError):
        _rolling(np.zeros((3, 2), dtype=np.float32), 2, "median")


@needs_data
def test_warmup_covers_the_longest_lookback():
    assert WARMUP > max(LAG_OFFSETS)
    assert WARMUP >= 168


@needs_data
def test_design_matrix_has_no_missing_values(occupancy_features):
    origins = occupancy_features.valid_origins(1)[:200]
    design, target = occupancy_features.design(origins, 1)
    assert not np.isnan(design).any()
    assert not np.isnan(target).any()


@needs_data
def test_design_rows_are_origin_major(occupancy_features):
    origins = occupancy_features.valid_origins(1)[:5]
    _, target = occupancy_features.design(origins, 1)
    grid = target.reshape(len(origins), occupancy_features.n_zones)
    expected = occupancy_features.y[origins + 1]
    assert np.allclose(grid, expected, equal_nan=True)


@needs_data
def test_target_is_the_series_h_steps_after_the_origin(occupancy_features):
    for horizon in (1, 6, 24):
        origins = occupancy_features.valid_origins(horizon)[:3]
        _, target = occupancy_features.design(origins, horizon)
        grid = target.reshape(len(origins), occupancy_features.n_zones)
        assert np.allclose(grid, occupancy_features.y[origins + horizon])


@needs_data
def test_an_origin_without_an_observed_target_is_refused(occupancy_features):
    last = occupancy_features.n_hours - 1
    with pytest.raises(ValueError, match="beyond the observed window"):
        occupancy_features.design([last], 1)


@needs_data
@pytest.mark.parametrize("horizon", [1, 6, 24])
def test_no_feature_reads_the_future(bundle, horizon):
    """Replacing the series after the latest origin must move no feature.

    The whole target series from one hour past the latest origin onwards is
    overwritten with noise and the feature set is rebuilt through the ordinary
    code path, so a feature that reads even one hour beyond its origin changes
    and the test fails. The target vector is expected to change and is not
    compared.
    """
    import copy

    origins = np.array([2000, 2500, 3000])
    cut = int(origins.max()) + 1

    features = build_features(bundle, "occupancy")
    before, _ = features.design(origins, horizon)

    disturbed = copy.copy(bundle)
    long = bundle.long.copy()
    generator = np.random.default_rng(0)
    future = pd.DatetimeIndex(long["timestamp"]) >= bundle.index[cut]
    long.loc[future, "occupancy"] = generator.normal(
        500.0, 50.0, size=int(future.sum())).astype("float32")
    disturbed.long = long

    after, _ = build_features(disturbed, "occupancy").design(origins, horizon)
    assert np.allclose(before, after, equal_nan=True)


@needs_data
def test_the_leakage_guard_itself_detects_a_planted_leak(bundle):
    """A deliberately future reading feature must fail the same comparison.

    Without this, the guard above would pass just as happily against a builder
    that computed nothing at all.
    """
    import copy

    origins = np.array([2000, 2500, 3000])
    cut = int(origins.max()) + 1

    features = build_features(bundle, "occupancy")
    planted = np.empty_like(features.y)
    planted[:-1] = features.y[1:]
    planted[-1] = features.y[-1]
    features.origin_blocks["leak_next_hour"] = planted
    before, _ = features.design(origins, 1)

    disturbed = copy.copy(bundle)
    long = bundle.long.copy()
    generator = np.random.default_rng(0)
    future = pd.DatetimeIndex(long["timestamp"]) >= bundle.index[cut]
    long.loc[future, "occupancy"] = generator.normal(
        500.0, 50.0, size=int(future.sum())).astype("float32")
    disturbed.long = long

    leaky = build_features(disturbed, "occupancy")
    planted_after = np.empty_like(leaky.y)
    planted_after[:-1] = leaky.y[1:]
    planted_after[-1] = leaky.y[-1]
    leaky.origin_blocks["leak_next_hour"] = planted_after
    after, _ = leaky.design(origins, 1)

    assert not np.allclose(before, after, equal_nan=True)


@needs_data
def test_lag_zero_is_the_value_at_the_origin(occupancy_features):
    block = occupancy_features.origin_blocks["lag_0"]
    assert np.allclose(block[500], occupancy_features.y[500])


@needs_data
def test_lag_24_is_the_value_a_day_earlier(occupancy_features):
    block = occupancy_features.origin_blocks["lag_24"]
    assert np.allclose(block[500], occupancy_features.y[476])


@needs_data
def test_calendar_is_taken_at_the_target_time(occupancy_features):
    origins = np.array([1000])
    horizon = 6
    design, _ = occupancy_features.design(origins, horizon)
    names = occupancy_features.feature_names()
    column = names.index("hour_of_day")
    expected = occupancy_features.index[origins[0] + horizon].hour
    assert design[0, column] == pytest.approx(expected)


@needs_data
def test_static_features_can_be_dropped_for_per_zone_models(occupancy_features):
    origins = occupancy_features.valid_origins(1)[:10]
    with_static, _ = occupancy_features.design(origins, 1, include_static=True)
    without, _ = occupancy_features.design(origins, 1, include_static=False)
    assert with_static.shape[1] - without.shape[1] == len(occupancy_features.static_blocks)


@needs_data
def test_static_features_are_constant_within_a_zone(occupancy_features):
    origins = occupancy_features.valid_origins(1)[:4]
    design, _ = occupancy_features.design(origins, 1)
    names = occupancy_features.feature_names()
    column = names.index("capacity_points")
    grid = design[:, column].reshape(len(origins), occupancy_features.n_zones)
    assert np.allclose(grid[0], grid[-1])


@needs_data
def test_selecting_a_zone_subset_matches_the_full_design(occupancy_features):
    origins = occupancy_features.valid_origins(1)[:6]
    full, full_y = occupancy_features.design(origins, 1)
    subset, subset_y = occupancy_features.design(origins, 1, zones=[3])
    grid = full.reshape(len(origins), occupancy_features.n_zones, full.shape[1])
    assert np.allclose(grid[:, 3, :], subset)
    assert np.allclose(full_y.reshape(len(origins), -1)[:, 3], subset_y)


@needs_data
def test_valid_origins_start_after_the_warmup(occupancy_features):
    assert occupancy_features.valid_origins(1)[0] == WARMUP - 1


@needs_data
def test_valid_origins_leave_room_for_the_horizon(occupancy_features):
    for horizon in (1, 6, 24):
        origins = occupancy_features.valid_origins(horizon)
        assert origins[-1] + horizon == occupancy_features.n_hours - 1
