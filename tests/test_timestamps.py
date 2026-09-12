"""The timestamp parser and index assertions.

Both formats in the dataset are parsed explicitly. A silent automatic parser
would resolve an ambiguous day and month order without complaint, which is the
failure this guards against.
"""

from __future__ import annotations

import pandas as pd
import pytest

from uev.io_load import (DEMAND_FORMAT, WEATHER_FORMAT, DataError, assert_hourly,
                         parse_time)


def test_demand_format_parses_the_published_form():
    parsed = parse_time(pd.Series(["2022-09-01 00:00:00"]), DEMAND_FORMAT)
    assert parsed.iloc[0] == pd.Timestamp("2022-09-01 00:00")


def test_weather_format_parses_the_published_form():
    parsed = parse_time(pd.Series(["2022/9/1 0:00"]), WEATHER_FORMAT)
    assert parsed.iloc[0] == pd.Timestamp("2022-09-01 00:00")


def test_weather_format_reads_day_and_month_in_the_published_order():
    parsed = parse_time(pd.Series(["2022/9/11 13:00"]), WEATHER_FORMAT)
    assert (parsed.iloc[0].month, parsed.iloc[0].day) == (9, 11)


def test_the_wrong_format_is_rejected_rather_than_guessed():
    with pytest.raises(DataError):
        parse_time(pd.Series(["2022/9/1 0:00"]), DEMAND_FORMAT)


def test_unparseable_values_raise():
    with pytest.raises(DataError):
        parse_time(pd.Series(["not a timestamp"]), DEMAND_FORMAT)


def test_complete_index_passes():
    index = pd.date_range("2022-09-01", "2023-02-28 23:00", freq="h")
    assert_hourly(pd.DatetimeIndex(index), "complete")


def test_short_index_is_rejected():
    index = pd.date_range("2022-09-01", periods=10, freq="h")
    with pytest.raises(DataError, match="not the expected hourly index"):
        assert_hourly(pd.DatetimeIndex(index), "short")


def test_duplicate_timestamps_are_rejected():
    index = pd.DatetimeIndex(["2022-09-01 00:00", "2022-09-01 00:00"])
    with pytest.raises(DataError, match="duplicate"):
        assert_hourly(index, "duplicated")


def test_unsorted_index_is_rejected():
    index = pd.DatetimeIndex(["2022-09-01 01:00", "2022-09-01 00:00"])
    with pytest.raises(DataError, match="not sorted"):
        assert_hourly(index, "unsorted")


def test_a_gap_inside_the_window_is_rejected():
    full = pd.date_range("2022-09-01", "2023-02-28 23:00", freq="h")
    with_gap = full.delete(100)
    with pytest.raises(DataError):
        assert_hourly(pd.DatetimeIndex(with_gap), "gapped")


def test_daily_frequency_is_rejected():
    index = pd.date_range("2022-09-01", "2023-02-28", freq="D")
    with pytest.raises(DataError):
        assert_hourly(pd.DatetimeIndex(index), "daily")
