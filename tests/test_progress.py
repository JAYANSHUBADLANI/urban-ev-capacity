"""Run bookkeeping."""

from __future__ import annotations

import pytest

from uev import progress


@pytest.fixture(autouse=True)
def temporary_state(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "PROGRESS", tmp_path / "progress.json")
    monkeypatch.setattr(progress, "STATE", tmp_path)
    return tmp_path


def test_unknown_phase_is_pending():
    assert progress.phase_status(7) == "pending"


def test_marking_a_phase_persists_it():
    progress.mark_phase(3, "complete", "note")
    assert progress.phase_status(3) == "complete"


def test_an_invalid_status_is_rejected():
    with pytest.raises(ValueError):
        progress.mark_phase(3, "finished")


def test_cells_accumulate_without_duplicates():
    progress.record_cells("grid", ["a", "b"])
    progress.record_cells("grid", ["b", "c"])
    assert progress.completed_cells("grid") == {"a", "b", "c"}


def test_cells_of_an_unknown_group_are_empty():
    assert progress.completed_cells("nothing") == set()


def test_summary_lists_every_phase():
    text = progress.summary()
    assert text.count("phase ") == len(progress.PHASES)


def test_save_is_atomic_and_leaves_no_temporary_file(temporary_state):
    progress.mark_phase(1, "complete")
    assert not (temporary_state / "progress.json.tmp").exists()
