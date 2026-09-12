"""Grid bookkeeping, prediction clipping, and the SQL reporting layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import needs_data
from uev.config import HORIZONS, TARGETS
from uev.grid import (FAMILIES, ceiling_for, clip_predictions, expected_cells,
                      make_cell_key, score_cell, zone_denominators)


def test_the_grid_covers_every_combination():
    cells = expected_cells()
    assert len(cells) == len(TARGETS) * len(HORIZONS) * 8 * len(FAMILIES)
    assert len(cells) == 576


def test_every_cell_key_is_unique():
    keys = [c["cell_key"] for c in expected_cells()]
    assert len(set(keys)) == len(keys)


def test_cell_keys_are_stable_across_calls():
    assert make_cell_key("occupancy", 1, 0, "global_ridge") == \
        make_cell_key("occupancy", 1, 0, "global_ridge")


def test_cell_keys_distinguish_every_field():
    base = make_cell_key("occupancy", 1, 0, "global_ridge")
    assert make_cell_key("volume", 1, 0, "global_ridge") != base
    assert make_cell_key("occupancy", 6, 0, "global_ridge") != base
    assert make_cell_key("occupancy", 1, 1, "global_ridge") != base
    assert make_cell_key("occupancy", 1, 0, "global_gbm") != base


def test_eight_families_are_run():
    assert len(FAMILIES) == 8


def test_occupancy_and_duration_have_a_ceiling_but_energy_does_not():
    capacity = np.array([10.0, 20.0])
    assert ceiling_for("occupancy", capacity) is not None
    assert ceiling_for("duration", capacity) is not None
    assert ceiling_for("volume", capacity) is None


def test_clipping_removes_negative_predictions():
    out = clip_predictions(np.array([-5.0, 3.0]), "volume", np.array([10.0, 10.0]))
    assert out.tolist() == [0.0, 3.0]


def test_clipping_caps_occupancy_at_installed_points():
    out = clip_predictions(np.array([99.0, 3.0]), "occupancy", np.array([10.0, 10.0]))
    assert out.tolist() == [10.0, 3.0]


def test_clipping_does_not_cap_energy():
    out = clip_predictions(np.array([1e6]), "volume", np.array([10.0]))
    assert out[0] == 1e6


def test_clipping_leaves_feasible_predictions_untouched():
    values = np.array([1.0, 5.0, 9.0])
    out = clip_predictions(values, "occupancy", np.full(3, 10.0))
    assert np.array_equal(out, values)


def test_score_cell_produces_pooled_and_disaggregated_rows():
    from uev.cv import Fold
    fold = Fold(fold=0, train_start=0, train_end=100, test_start=100, test_end=110)
    n = 12
    actual = np.arange(n, dtype=float)
    predicted = actual + 1.0
    zones = np.tile(np.array([1, 2, 3]), 4)
    result = score_cell("occupancy", 1, fold, "global_ridge", actual, predicted, zones,
                        np.full(n, 2.0), np.full(n, 0.5),
                        np.repeat(np.arange(4), 3), np.zeros(n, dtype=int),
                        n_fits=1, chosen={"alpha": 1.0}, key="abc")
    assert result.pooled["mae"] == pytest.approx(1.0)
    assert result.pooled["cell_key"] == "abc"
    assert len(result.per_zone) == 3
    assert len(result.per_hour) == 4
    assert len(result.per_dow) == 1


def test_score_cell_scaled_error_uses_the_denominator():
    from uev.cv import Fold
    fold = Fold(fold=0, train_start=0, train_end=100, test_start=100, test_end=110)
    actual = np.zeros(4)
    predicted = np.full(4, 2.0)
    result = score_cell("occupancy", 1, fold, "f", actual, predicted,
                        np.array([1, 1, 2, 2]), np.full(4, 2.0), np.zeros(4),
                        np.zeros(4, dtype=int), np.zeros(4, dtype=int),
                        0, {}, "k")
    assert result.per_zone[0]["mase"] == pytest.approx(1.0)


@needs_data
def test_denominators_come_from_the_training_window_only(occupancy_features):
    from uev.cv import Fold
    early = Fold(fold=0, train_start=0, train_end=500, test_start=500, test_end=600)
    late = Fold(fold=1, train_start=0, train_end=3000, test_start=3000, test_end=3100)
    assert not np.allclose(zone_denominators(occupancy_features, early),
                           zone_denominators(occupancy_features, late))


@needs_data
def test_denominators_are_finite_for_most_zones(occupancy_features):
    from uev.cv import Fold
    fold = Fold(fold=0, train_start=0, train_end=1656, test_start=1656, test_end=1992)
    values = zone_denominators(occupancy_features, fold)
    assert np.isfinite(values).mean() > 0.9


@needs_data
def test_sql_and_pandas_quality_counts_agree():
    counts = pd.read_csv("results/quality_cross_check.csv")
    assert counts["agree"].all()


@needs_data
def test_the_sql_layer_runs_on_sqlite_from_the_standard_library():
    from uev.paths import DB
    from uev.sqlio import query
    if not DB.exists():
        pytest.skip("the reporting database has not been built")
    frame = query("zone_ranking.sql", engine="sqlite")
    assert len(frame) == 275
    assert "mean_utilisation" in frame.columns


@needs_data
def test_the_sql_ranking_is_ordered_by_utilisation():
    from uev.paths import DB
    from uev.sqlio import query
    if not DB.exists():
        pytest.skip("the reporting database has not been built")
    frame = query("zone_ranking.sql", engine="sqlite")
    assert frame["mean_utilisation"].is_monotonic_decreasing


@needs_data
def test_peak_windows_cover_every_zone():
    from uev.paths import DB
    from uev.sqlio import query
    if not DB.exists():
        pytest.skip("the reporting database has not been built")
    frame = query("peak_windows.sql", engine="sqlite")
    assert frame["zone_id"].nunique() == 275
    assert frame["window_start"].between(0, 23).all()


def test_sql_statements_split_cleanly():
    from uev.sqlio import split_statements
    script = "-- a comment\nSELECT 1;\n\n-- another\nSELECT 2;\n"
    assert split_statements(script) == ["SELECT 1", "SELECT 2"]


def test_sql_files_are_all_present():
    from uev.sqlio import read_sql_file
    for name in ("schema.sql", "daily_profile.sql", "hourly_profile.sql",
                 "zone_ranking.sql", "peak_windows.sql", "checks.sql"):
        assert read_sql_file(name).strip()


def test_a_missing_sql_file_raises():
    from uev.sqlio import read_sql_file
    with pytest.raises(FileNotFoundError):
        read_sql_file("no_such_query.sql")
