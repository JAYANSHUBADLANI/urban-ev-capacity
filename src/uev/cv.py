"""Rolling origin cross validation.

Splits respect time everywhere. A fold is defined by the hour at which its test
window opens: everything strictly before it is available for training, and the
test window is the 336 hours from that point. Nothing is shuffled, and the same
split function is used for hyperparameter selection, so a search cannot quietly
see the future either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import CONFIG, CVConfig


@dataclass(frozen=True)
class Fold:
    """One rolling origin fold, in positional hour indices."""

    fold: int
    train_start: int
    train_end: int      # exclusive
    test_start: int     # equals train_end
    test_end: int       # exclusive

    @property
    def n_train_hours(self) -> int:
        return self.train_end - self.train_start

    @property
    def n_test_hours(self) -> int:
        return self.test_end - self.test_start


def make_folds(n_hours: int, settings: CVConfig | None = None) -> List[Fold]:
    """Build the folds, anchored so that the last one ends at the series end."""
    settings = settings or CONFIG.cv
    folds: List[Fold] = []
    last_test_start = n_hours - settings.test_hours
    first_test_start = last_test_start - (settings.n_folds - 1) * settings.step_hours
    if first_test_start < settings.min_train_hours:
        raise ValueError(
            "the series is too short for the requested folds: "
            f"{settings.n_folds} folds of {settings.test_hours} hours need at least "
            f"{settings.min_train_hours + (settings.n_folds - 1) * settings.step_hours + settings.test_hours} hours"
        )
    for k in range(settings.n_folds):
        test_start = first_test_start + k * settings.step_hours
        folds.append(Fold(fold=k,
                          train_start=0,
                          train_end=test_start,
                          test_start=test_start,
                          test_end=test_start + settings.test_hours))
    return folds


def train_origins(fold: Fold, horizon: int, warmup: int) -> np.ndarray:
    """Origins whose feature row and target both sit inside the training window.

    A training row whose target fell inside the test window would put the test
    period into the fitted parameters, so the upper bound is on the target and
    not on the origin.
    """
    first = max(warmup - 1, fold.train_start)
    last = fold.train_end - horizon - 1
    if last < first:
        return np.zeros(0, dtype=int)
    return np.arange(first, last + 1, dtype=int)


def holdout_origins(fold: Fold, horizon: int, warmup: int, n_hours: int) -> np.ndarray:
    """Origins whose target falls inside the test window.

    The features of a test row may use observations after the training window,
    because at prediction time for a target at ``t + h`` the series is known up
    to ``t``. What they may never use is anything after ``t``.
    """
    first = max(warmup - 1, fold.test_start - horizon)
    last = min(fold.test_end - horizon - 1, n_hours - horizon - 1)
    if last < first:
        return np.zeros(0, dtype=int)
    return np.arange(first, last + 1, dtype=int)


def describe(folds: List[Fold], index) -> List[dict]:
    """Human readable fold boundaries for the results tables."""
    rows = []
    for fold in folds:
        rows.append({
            "fold": fold.fold,
            "train_start": index[fold.train_start].strftime("%Y-%m-%d %H:%M"),
            "train_end": index[fold.train_end - 1].strftime("%Y-%m-%d %H:%M"),
            "test_start": index[fold.test_start].strftime("%Y-%m-%d %H:%M"),
            "test_end": index[fold.test_end - 1].strftime("%Y-%m-%d %H:%M"),
            "n_train_hours": fold.n_train_hours,
            "n_test_hours": fold.n_test_hours,
        })
    return rows
