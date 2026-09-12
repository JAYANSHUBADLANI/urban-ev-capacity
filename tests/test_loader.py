"""The reshape from wide to long, and the zone attribute aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import needs_data
from uev.config import EXPECTED_HOURS, EXPECTED_ZONES
from uev.io_load import build_long


def test_build_long_stacks_every_table():
    index = pd.date_range("2022-09-01", periods=5, freq="h")
    wide = {
        "occupancy": pd.DataFrame({101: [1.0] * 5, 102: [2.0] * 5}, index=index),
        "volume": pd.DataFrame({101: [7.0] * 5, 102: [14.0] * 5}, index=index),
    }
    long = build_long(wide)
    assert len(long) == 10
    assert set(long.columns) == {"timestamp", "zone_id", "occupancy", "volume"}
    assert long.loc[long.zone_id == 102, "volume"].unique().tolist() == [14.0]


def test_build_long_preserves_the_value_at_each_key():
    index = pd.date_range("2022-09-01", periods=3, freq="h")
    wide = {"occupancy": pd.DataFrame({101: [1.0, 2.0, 3.0], 102: [4.0, 5.0, 6.0]}, index=index)}
    long = build_long(wide)
    picked = long[(long.zone_id == 102) & (long.timestamp == index[2])]
    assert picked["occupancy"].iloc[0] == 6.0


def test_build_long_is_sorted_by_zone_then_time():
    index = pd.date_range("2022-09-01", periods=4, freq="h")
    wide = {"occupancy": pd.DataFrame({7: [1.0] * 4, 3: [2.0] * 4}, index=index)}
    long = build_long(wide)
    assert long["zone_id"].is_monotonic_increasing
    for _, group in long.groupby("zone_id"):
        assert group["timestamp"].is_monotonic_increasing


def test_build_long_round_trips_back_to_the_wide_shape():
    index = pd.date_range("2022-09-01", periods=6, freq="h")
    values = np.arange(12, dtype=float).reshape(6, 2)
    wide = {"occupancy": pd.DataFrame(values, index=index, columns=[10, 20])}
    long = build_long(wide)
    back = long.pivot(index="timestamp", columns="zone_id", values="occupancy")
    assert np.allclose(back.to_numpy(), values)


@needs_data
def test_long_frame_has_one_row_per_zone_hour(bundle):
    assert len(bundle.long) == EXPECTED_HOURS * EXPECTED_ZONES


@needs_data
def test_every_zone_is_fully_observed(bundle):
    counts = bundle.long.groupby("zone_id").size()
    assert (counts == EXPECTED_HOURS).all()


@needs_data
def test_zone_attributes_cover_every_zone(bundle):
    assert sorted(bundle.zones["zone_id"]) == sorted(bundle.long["zone_id"].unique())


@needs_data
def test_capacity_is_a_sum_over_stations_not_a_lookup(bundle):
    assert (bundle.zones["capacity_points"] >= bundle.zones["max_points_per_station"]).all()
    multi = bundle.zones[bundle.zones["n_stations"] > 1]
    assert (multi["capacity_points"] > multi["max_points_per_station"]).any()


@needs_data
def test_published_station_and_point_totals_are_reproduced(bundle):
    assert int(bundle.zones["n_stations"].sum()) == 1362
    assert int(bundle.zones["capacity_points"].sum()) == 17532


@needs_data
def test_occupancy_never_exceeds_installed_points(bundle):
    capacity = bundle.long["zone_id"].map(bundle.zones.set_index("zone_id")["capacity_points"])
    assert (bundle.long["occupancy"] <= capacity + 1e-6).all()


@needs_data
def test_occupancy_is_a_count_not_a_percentage(bundle):
    values = bundle.long["occupancy"].to_numpy()
    assert np.isclose(values, np.round(values)).mean() > 0.99
    assert values.max() > 100


@needs_data
def test_all_tables_share_the_zone_order(bundle):
    assert list(bundle.adjacency.columns) == list(bundle.distance.columns)
    assert list(bundle.adjacency.columns) == list(bundle.zones["zone_id"])


@needs_data
def test_weather_is_on_the_shared_index(bundle):
    assert len(bundle.weather) == EXPECTED_HOURS
    assert pd.DatetimeIndex(bundle.weather["timestamp"]).equals(bundle.index)


@needs_data
def test_no_measured_value_is_missing(bundle):
    assert not bundle.long.isna().any().any()
