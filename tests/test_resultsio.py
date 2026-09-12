"""Incremental result writing and deduplication."""

from __future__ import annotations

import pandas as pd
import pytest

from uev import resultsio


@pytest.fixture(autouse=True)
def temporary_results(tmp_path, monkeypatch):
    monkeypatch.setattr(resultsio, "RESULTS", tmp_path)
    return tmp_path


def test_append_writes_a_header_once():
    resultsio.append_rows("x.csv", [{"cell_key": "a", "value": 1}])
    resultsio.append_rows("x.csv", [{"cell_key": "b", "value": 2}])
    assert len(resultsio.read_rows("x.csv")) == 2


def test_append_of_nothing_is_a_no_op():
    assert resultsio.append_rows("x.csv", []) == 0
    assert resultsio.read_rows("x.csv").empty


def test_duplicate_cell_keys_keep_the_last_row():
    resultsio.append_rows("x.csv", [{"cell_key": "a", "value": 1}])
    resultsio.append_rows("x.csv", [{"cell_key": "a", "value": 9}])
    frame = resultsio.read_rows("x.csv")
    assert len(frame) == 1 and frame.loc[0, "value"] == 9


def test_existing_keys_reports_what_is_done():
    resultsio.append_rows("x.csv", [{"cell_key": "a"}, {"cell_key": "b"}])
    assert resultsio.existing_keys("x.csv") == {"a", "b"}


def test_existing_keys_of_a_missing_file_is_empty():
    assert resultsio.existing_keys("missing.csv") == set()


def test_consolidate_rewrites_without_duplicates(temporary_results):
    resultsio.append_rows("x.csv", [{"cell_key": "a", "value": 1}])
    resultsio.append_rows("x.csv", [{"cell_key": "a", "value": 2}])
    resultsio.consolidate("x.csv")
    assert len(pd.read_csv(temporary_results / "x.csv")) == 1


def test_write_table_overwrites():
    resultsio.write_table("t.csv", pd.DataFrame({"a": [1, 2]}))
    resultsio.write_table("t.csv", pd.DataFrame({"a": [3]}))
    assert len(pd.read_csv(resultsio.RESULTS / "t.csv")) == 1


def test_extra_fields_are_ignored_against_the_header():
    resultsio.append_rows("x.csv", [{"cell_key": "a", "value": 1}])
    resultsio.append_rows("x.csv", [{"cell_key": "b", "value": 2, "extra": 3}])
    assert list(resultsio.read_rows("x.csv").columns) == ["cell_key", "value"]
