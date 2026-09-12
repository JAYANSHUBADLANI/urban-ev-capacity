"""Exploratory aggregates, the generated dictionary, and figure assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import needs_data
from uev.eda import N_DECILES


@needs_data
def test_aggregate_series_sums_across_zones(bundle):
    from uev.eda import aggregate_series
    frame = aggregate_series(bundle)
    assert len(frame) == len(bundle.index)
    assert frame["volume"].sum() == pytest.approx(bundle.long["volume"].sum(), rel=1e-4)


@needs_data
def test_aggregate_utilisation_is_bounded(bundle):
    from uev.eda import aggregate_series
    frame = aggregate_series(bundle)
    assert frame["utilisation"].between(0.0, 1.0).all()


@needs_data
def test_daily_series_has_one_row_per_day(bundle):
    from uev.eda import aggregate_series, daily_series
    daily = daily_series(aggregate_series(bundle))
    assert len(daily) == 181


@needs_data
def test_volume_deciles_cover_every_zone(bundle):
    from uev.eda import zone_volume_deciles
    deciles = zone_volume_deciles(bundle)
    assert len(deciles) == len(bundle.zones)
    assert set(deciles["volume_decile"]) == set(range(1, N_DECILES + 1))


@needs_data
def test_volume_deciles_are_ordered_by_volume(bundle):
    from uev.eda import zone_volume_deciles
    deciles = zone_volume_deciles(bundle)
    means = deciles.groupby("volume_decile")["total_volume"].mean()
    assert means.is_monotonic_increasing


@needs_data
def test_decile_sizes_are_balanced(bundle):
    from uev.eda import zone_volume_deciles
    counts = zone_volume_deciles(bundle).groupby("volume_decile").size()
    assert counts.max() - counts.min() <= 1


@needs_data
def test_hour_profile_covers_every_hour_and_day_type(bundle):
    from uev.eda import hour_of_day_profile
    profile = hour_of_day_profile(bundle)
    assert len(profile) == 48


@needs_data
def test_saturation_profile_is_bounded(bundle):
    from uev.eda import saturation_profile
    frame = saturation_profile(bundle)
    assert frame["max_utilisation"].le(1.0 + 1e-6).all()
    assert frame["share_at_ceiling"].between(0.0, 1.0).all()


@needs_data
def test_the_eleven_kw_estimate_is_mostly_but_not_always_a_cap(bundle):
    """The published description holds for most zones and not for all of them.

    This is asserted rather than assumed because the write up states the
    exception, and a silent change in the data would otherwise make that
    statement wrong without anything failing.
    """
    from uev.eda import volume_estimate_comparison
    frame = volume_estimate_comparison(bundle)
    expected = frame["implied_kw_rated"].clip(upper=11.0)
    matches = (frame["implied_kw_11"] - expected).abs() < 0.5
    assert matches.mean() > 0.80
    assert not matches.all()


@needs_data
def test_the_zones_that_break_the_cap_are_a_small_minority(bundle):
    from uev.eda import volume_estimate_comparison
    frame = volume_estimate_comparison(bundle)
    assert int((frame["implied_kw_11"] > 12.0).sum()) < 10


@needs_data
def test_the_two_estimates_differ_materially_in_total(bundle):
    from uev.eda import volume_estimate_comparison
    frame = volume_estimate_comparison(bundle)
    ratio = frame["volume_rated_kwh"].sum() / frame["volume_11kw_kwh"].sum()
    assert ratio > 2.0


@needs_data
def test_summary_rows_carry_units(bundle):
    from uev.eda import summary_rows
    rows = summary_rows(bundle)
    assert all(row["unit"] for row in rows)
    assert len(rows) >= 10


@needs_data
def test_price_variation_counts_changes(bundle):
    from uev.eda import price_variation
    frame = price_variation(bundle)
    assert len(frame) == len(bundle.zones)
    assert (frame["e_price_changes"] >= 0).all()


@needs_data
def test_the_dictionary_reports_the_observed_range(bundle):
    from uev.dictionary import measured_summary
    summary = measured_summary(bundle).set_index("field")
    assert summary.loc["occupancy", "max"] == pytest.approx(
        float(bundle.long["occupancy"].max()))


@needs_data
def test_the_dictionary_renders_every_field(bundle):
    from uev.dictionary import FIELDS, render
    text = render(bundle)
    for entry in FIELDS:
        assert f"`{entry['field']}`" in text


@needs_data
def test_the_dictionary_states_the_coverage(bundle):
    from uev.dictionary import render
    assert "4344 hourly periods" in render(bundle)


def test_every_figure_has_a_builder():
    from uev.figures import ALL_FIGURES
    assert len(ALL_FIGURES) >= 10
    assert all(callable(builder) for builder in ALL_FIGURES.values())


def test_a_missing_input_skips_a_figure_rather_than_raising(tmp_path, monkeypatch):
    from uev import figures
    monkeypatch.setattr(figures, "RESULTS", tmp_path)
    assert figures.figure_daily_demand_with_breaks() is None
