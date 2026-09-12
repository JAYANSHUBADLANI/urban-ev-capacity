"""Run bookkeeping: phase status and completed grid cells.

State lives in ``state/progress.json`` and is written after every change, so a
run that stops part way can be resumed from the last completed unit of work.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable

from .paths import PROGRESS, STATE

PHASES = {
    1: "repository skeleton, configuration, seeding, logging, test harness",
    2: "fetch with manifest, loader, tidy long frame, zone attributes",
    3: "data quality suite, data dictionary, SQL layer",
    4: "exploratory analysis and structural break detection",
    5: "preregistration frozen",
    6: "feature engineering with leakage tests, model families",
    7: "full rolling origin grid",
    8: "monitoring, threshold calibration, retraining backtest",
    9: "segmentation and anomaly detection",
    10: "queueing model, capacity allocation, budget scenarios",
    11: "figures and write up",
    12: "self review, determinism check, package",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load() -> Dict[str, Any]:
    if PROGRESS.exists():
        with PROGRESS.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {"phases": {}, "cells": {}, "updated": None}


def save(state: Dict[str, Any]) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    state["updated"] = _now()
    tmp = PROGRESS.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
    tmp.replace(PROGRESS)


def mark_phase(phase: int, status: str, note: str = "") -> Dict[str, Any]:
    """Record a phase status. Valid statuses: pending, running, complete."""
    if status not in {"pending", "running", "complete"}:
        raise ValueError(f"unknown status: {status}")
    state = load()
    state["phases"][str(phase)] = {
        "status": status,
        "name": PHASES.get(phase, ""),
        "note": note,
        "at": _now(),
    }
    save(state)
    return state


def phase_status(phase: int) -> str:
    return load()["phases"].get(str(phase), {}).get("status", "pending")


def completed_cells(group: str) -> set:
    return set(load()["cells"].get(group, []))


def record_cells(group: str, keys: Iterable[str]) -> None:
    state = load()
    existing = set(state["cells"].get(group, []))
    existing.update(keys)
    state["cells"][group] = sorted(existing)
    save(state)


def summary() -> str:
    state = load()
    lines = []
    for phase in sorted(PHASES):
        entry = state["phases"].get(str(phase), {})
        lines.append(f"phase {phase:2d} {entry.get('status', 'pending'):>8}  {PHASES[phase]}")
    for group, keys in sorted(state.get("cells", {}).items()):
        lines.append(f"cells {group}: {len(keys)}")
    return "\n".join(lines)
