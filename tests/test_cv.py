"""Rolling origin fold boundaries.

The property that matters is that no training row can carry information from
its own test window, so the assertions here are about the target times and not
only about the origins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uev.config import CVConfig
from uev.cv import Fold, describe, make_folds, holdout_origins, train_origins

WARMUP = 168


def test_the_configured_number_of_folds_is_produced():
    assert len(make_folds(4344)) == 8


def test_folds_are_contiguous_and_non_overlapping():
    folds = make_folds(4344)
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test_end == later.test_start
        assert earlier.test_start < later.test_start


def test_the_last_fold_ends_at_the_series_end():
    assert make_folds(4344)[-1].test_end == 4344


def test_training_windows_grow():
    folds = make_folds(4344)
    assert all(later.n_train_hours > earlier.n_train_hours
               for earlier, later in zip(folds, folds[1:]))


def test_every_test_window_is_the_configured_length():
    assert all(fold.n_test_hours == 336 for fold in make_folds(4344))


def test_training_never_starts_after_the_test_window():
    assert all(fold.train_end == fold.test_start for fold in make_folds(4344))


def test_a_series_too_short_for_the_folds_is_refused():
    with pytest.raises(ValueError, match="too short"):
        make_folds(1000)


def test_a_custom_configuration_is_honoured():
    folds = make_folds(4344, CVConfig(n_folds=4, test_hours=168, min_train_hours=336,
                                      step_hours=168))
    assert len(folds) == 4 and folds[0].n_test_hours == 168


@pytest.mark.parametrize("horizon", [1, 6, 24])
def test_no_training_target_falls_inside_the_test_window(horizon):
    for fold in make_folds(4344):
        origins = train_origins(fold, horizon, WARMUP)
        assert (origins + horizon < fold.test_start).all()


@pytest.mark.parametrize("horizon", [1, 6, 24])
def test_every_test_target_falls_inside_the_test_window(horizon):
    for fold in make_folds(4344):
        origins = holdout_origins(fold, horizon, WARMUP, 4344)
        targets = origins + horizon
        assert (targets >= fold.test_start).all()
        assert (targets < fold.test_end).all()


@pytest.mark.parametrize("horizon", [1, 6, 24])
def test_the_test_window_is_covered_exactly_once(horizon):
    for fold in make_folds(4344):
        targets = holdout_origins(fold, horizon, WARMUP, 4344) + horizon
        assert len(np.unique(targets)) == len(targets)
        assert len(targets) == fold.n_test_hours


@pytest.mark.parametrize("horizon", [1, 6, 24])
def test_training_and_test_targets_never_overlap(horizon):
    for fold in make_folds(4344):
        train_targets = set((train_origins(fold, horizon, WARMUP) + horizon).tolist())
        test_targets = set((holdout_origins(fold, horizon, WARMUP, 4344) + horizon).tolist())
        assert not (train_targets & test_targets)


def test_a_test_origin_may_use_observations_after_the_training_window():
    """At prediction time for a target at t+h the series is known up to t."""
    fold = make_folds(4344)[0]
    origins = holdout_origins(fold, 1, WARMUP, 4344)
    assert (origins >= fold.train_end - 1).all()
    assert origins.max() >= fold.train_end


def test_training_origins_respect_the_warmup():
    fold = make_folds(4344)[0]
    assert train_origins(fold, 1, WARMUP).min() >= WARMUP - 1


def test_describe_renders_every_fold():
    index = pd.date_range("2022-09-01", periods=4344, freq="h")
    rows = describe(make_folds(4344), index)
    assert len(rows) == 8
    assert rows[0]["train_start"] == "2022-09-01 00:00"


def test_an_impossible_horizon_yields_no_origins():
    fold = Fold(fold=0, train_start=0, train_end=200, test_start=200, test_end=210)
    assert train_origins(fold, 500, WARMUP).size == 0
