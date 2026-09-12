"""Configuration and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from uev import config
from uev.paths import ROOT, rel


def test_root_contains_the_package():
    assert (ROOT / "src" / "uev" / "__init__.py").exists()


def test_rel_is_relative_and_posix():
    assert rel(ROOT / "results" / "x.csv") == "results/x.csv"


def test_rel_never_leaks_an_outside_path():
    assert rel(Path("/nowhere/secret/file.csv")) == "file.csv"


def test_rel_of_the_root_itself():
    assert rel(ROOT) == "."


def test_expected_window_is_six_months_of_hours():
    assert config.EXPECTED_HOURS == 4344
    assert config.EXPECTED_ZONES == 275


def test_data_files_are_pinned_to_one_commit():
    assert config.COMMIT in config.BASE_URL
    assert len(config.COMMIT) == 40
    assert len(config.DATA_FILES) == 13


def test_digest_is_stable_and_short():
    first = config.CONFIG.digest("a")
    second = config.CONFIG.digest("a")
    assert first == second and len(first) == 16
    assert config.CONFIG.digest("b") != first


def test_cell_key_is_order_independent():
    assert config.cell_key(a=1, b=2) == config.cell_key(b=2, a=1)
    assert config.cell_key(a=1) != config.cell_key(a=2)


def test_model_families_cover_the_seven_required_types():
    assert len(config.MODEL_FAMILIES) >= 7
    assert "global_ridge" in config.MODEL_FAMILIES
    assert "zone_gbm" in config.MODEL_FAMILIES


def test_cv_has_at_least_eight_folds():
    assert config.CONFIG.cv.n_folds >= 8


def test_horizons_and_targets():
    assert config.HORIZONS == (1, 6, 24)
    assert set(config.TARGETS) == {"occupancy", "volume", "duration"}


@pytest.mark.parametrize("day", config.HOLIDAYS)
def test_holidays_fall_inside_the_window(day):
    assert config.EXPECTED_START[:10] <= day <= config.EXPECTED_END[:10]
