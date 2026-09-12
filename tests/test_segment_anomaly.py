"""Segmentation criteria and anomaly agreement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import needs_data
from uev.anomaly import agreement, sample_for_inspection
from uev.segment import MAX_CLUSTER_SHARE, archetype_label, choose_k, criteria_table


@pytest.fixture
def diagnostics():
    return pd.DataFrame({
        "k": [2, 3, 4, 5],
        "silhouette": [0.90, 0.40, 0.35, 0.30],
        "calinski_harabasz": [200.0, 240.0, 210.0, 190.0],
        "stability_mean_ari": [1.00, 0.95, 0.90, 0.50],
        "largest_cluster_share": [0.88, 0.45, 0.40, 0.35],
        "inertia": [300.0, 200.0, 160.0, 140.0],
    })


def test_a_degenerate_split_is_not_chosen(diagnostics):
    """Silhouette alone would pick two clusters, which is 88 percent of zones."""
    assert choose_k(diagnostics) == 3


def test_an_unstable_count_is_not_chosen(diagnostics):
    unstable = diagnostics.copy()
    unstable.loc[unstable["k"] == 3, "stability_mean_ari"] = 0.1
    assert choose_k(unstable) == 4


def test_the_balance_threshold_is_respected(diagnostics):
    assert (diagnostics.loc[diagnostics["k"] == choose_k(diagnostics),
                            "largest_cluster_share"].iloc[0] <= MAX_CLUSTER_SHARE)


def test_choose_k_falls_back_when_nothing_is_admitted(diagnostics):
    hopeless = diagnostics.copy()
    hopeless["largest_cluster_share"] = 0.99
    assert choose_k(hopeless) in set(diagnostics["k"])


def test_criteria_table_marks_what_each_criterion_admits(diagnostics):
    table = criteria_table(diagnostics)
    assert not table.loc[table["k"] == 2, "admitted"].iloc[0]
    assert table.loc[table["k"] == 3, "admitted"].iloc[0]
    assert table.loc[table["k"] == 3, "best_by_silhouette"].iloc[0]


def test_the_two_criteria_can_be_seen_to_agree(diagnostics):
    table = criteria_table(diagnostics)
    best_silhouette = table.loc[table["best_by_silhouette"], "k"].iloc[0]
    best_ratio = table.loc[table["best_by_calinski_harabasz"], "k"].iloc[0]
    assert best_silhouette == best_ratio == 3


@pytest.mark.parametrize("peak_hour,peak_ratio,weekend,night,expected_fragment", [
    (3, 1.5, 1.0, 0.40, "overnight"),
    (13, 1.05, 1.0, 0.10, "flat load"),
    (13, 1.5, 1.3, 0.10, "weekend"),
    (8, 1.5, 1.0, 0.10, "morning"),
    (14, 1.5, 1.0, 0.10, "midday"),
    (19, 1.5, 1.0, 0.10, "evening"),
])
def test_archetypes_are_named_in_business_terms(peak_hour, peak_ratio, weekend,
                                                night, expected_fragment):
    assert expected_fragment in archetype_label(peak_hour, peak_ratio, weekend, night)


def _flag_frames(robust_flags, forest_flags):
    n = len(robust_flags)
    keys = pd.DataFrame({
        "zone_id": np.arange(n),
        "timestamp": pd.date_range("2022-09-01", periods=n, freq="h"),
    })
    robust = keys.assign(occupancy=1.0, robust_score=1.0, robust_flag=robust_flags)
    forest = keys.assign(forest_score=1.0, forest_flag=forest_flags,
                         utilisation=0.5, energy_per_point_hour=7.0,
                         duration_per_point=1.0, hour_of_day=0, is_weekend=0)
    return robust, forest


def test_agreement_of_identical_flags_is_one():
    robust, forest = _flag_frames([True, False, True], [True, False, True])
    assert agreement(robust, forest)["jaccard"] == pytest.approx(1.0)


def test_agreement_of_disjoint_flags_is_zero():
    robust, forest = _flag_frames([True, False], [False, True])
    assert agreement(robust, forest)["jaccard"] == pytest.approx(0.0)


def test_agreement_counts_each_side():
    robust, forest = _flag_frames([True, True, False], [True, False, False])
    out = agreement(robust, forest)
    assert out["n_robust_flags"] == 2 and out["n_forest_flags"] == 1
    assert out["n_both"] == 1 and out["n_either"] == 2


def test_agreement_with_no_flags_is_zero_not_undefined():
    robust, forest = _flag_frames([False, False], [False, False])
    assert agreement(robust, forest)["jaccard"] == 0.0


def test_the_inspection_sample_covers_each_group():
    robust, forest = _flag_frames([True, True, False, False] * 5,
                                  [True, False, True, False] * 5)
    sample = sample_for_inspection(robust, forest, n_per_group=2)
    assert set(sample["group"]) == {"both", "robust_only", "forest_only"}


def test_the_inspection_sample_is_reproducible():
    robust, forest = _flag_frames([True, True, False, False] * 5,
                                  [True, False, True, False] * 5)
    first = sample_for_inspection(robust, forest, n_per_group=2)
    second = sample_for_inspection(robust, forest, n_per_group=2)
    assert first.equals(second)


@needs_data
def test_load_shapes_are_scaled_to_unit_mean(bundle):
    from uev.segment import load_shapes
    shapes = load_shapes(bundle)
    assert np.allclose(shapes.mean(axis=1), 1.0, atol=1e-6)


@needs_data
def test_load_shapes_cover_every_zone_and_hour(bundle):
    from uev.segment import load_shapes
    shapes = load_shapes(bundle)
    assert shapes.shape == (len(bundle.zones), 48)
