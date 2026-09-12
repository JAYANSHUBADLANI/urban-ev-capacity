"""The data quality mechanics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import needs_data
from uev.quality import RUN_THRESHOLD, run_lengths, suspected_imputation


def test_run_lengths_on_a_constant_series():
    assert run_lengths(np.array([5, 5, 5])).tolist() == [3, 3, 3]


def test_run_lengths_on_an_alternating_series():
    assert run_lengths(np.array([1, 2, 1, 2])).tolist() == [1, 1, 1, 1]


def test_run_lengths_on_a_mixed_series():
    assert run_lengths(np.array([1, 1, 2, 3, 3, 3])).tolist() == [2, 2, 1, 3, 3, 3]


def test_run_lengths_on_an_empty_series():
    assert run_lengths(np.array([])).tolist() == []


def test_run_lengths_on_a_single_value():
    assert run_lengths(np.array([9])).tolist() == [1]


def test_suspected_imputation_flags_only_long_runs():
    frame = pd.DataFrame({
        "zone_id": [1] * 8,
        "occupancy": [1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0],
    })
    flags = suspected_imputation(frame, "occupancy", threshold=RUN_THRESHOLD)
    assert flags.tolist() == [False, False, True, True, True, True, False, False]


def test_suspected_imputation_does_not_run_across_zones():
    frame = pd.DataFrame({
        "zone_id": [1, 1, 2, 2],
        "occupancy": [7.0, 7.0, 7.0, 7.0],
    })
    flags = suspected_imputation(frame, "occupancy", threshold=3)
    assert not flags.any()


def test_suspected_imputation_respects_the_threshold():
    frame = pd.DataFrame({"zone_id": [1] * 4, "occupancy": [2.0] * 4})
    assert suspected_imputation(frame, "occupancy", threshold=5).sum() == 0
    assert suspected_imputation(frame, "occupancy", threshold=4).sum() == 4


@needs_data
def test_report_has_a_status_for_every_check(quality_report):
    assert quality_report["status"].isin({"pass", "warn", "fail", "info"}).all()


@needs_data
def test_structural_checks_all_pass(quality_report):
    structural = ["index_completeness", "index_length", "duplicate_keys",
                  "zone_count", "zone_coverage", "no_nulls"]
    subset = quality_report.set_index("check_id").loc[structural]
    assert (subset["status"] == "pass").all()


@needs_data
def test_the_recorded_adjacency_failures_are_reported_not_hidden(quality_report):
    indexed = quality_report.set_index("check_id")
    assert indexed.loc["adjacency_symmetric", "status"] == "fail"
    assert indexed.loc["adjacency_self_loops", "status"] == "fail"


@needs_data
def test_capacity_bounds_hold(quality_report):
    indexed = quality_report.set_index("check_id")
    assert indexed.loc["capacity_bound_occupancy", "n_failing"] == 0
    assert indexed.loc["capacity_bound_duration", "n_failing"] == 0


@needs_data
def test_fractions_are_between_zero_and_one(quality_report):
    assert quality_report["fraction"].between(0.0, 1.0).all()
