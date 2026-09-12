"""Incremental, deduplicated result writing.

Grid results are appended row by row so that a run which stops part way leaves
a usable file behind. Rows carry a ``cell_key`` and duplicates are dropped on
read and on consolidation.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from .paths import RESULTS


def append_rows(name: str, rows: Sequence[Dict], fieldnames: Sequence[str] | None = None) -> int:
    """Append rows to ``results/<name>``, writing a header on first use."""
    if not rows:
        return 0
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    if new_file:
        fields: List[str] = list(fieldnames) if fieldnames else list(rows[0].keys())
    else:
        # The header already on disk wins, so that a later row carrying an extra
        # key cannot shift the columns of a part written file.
        with path.open("r", newline="", encoding="utf-8") as handle:
            fields = next(csv.reader(handle))
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    return len(rows)


def read_rows(name: str) -> pd.DataFrame:
    """Read a results file, dropping duplicate cell keys and keeping the last."""
    path = RESULTS / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "cell_key" in frame.columns:
        frame = frame.drop_duplicates(subset="cell_key", keep="last")
    return frame.reset_index(drop=True)


def consolidate(name: str) -> pd.DataFrame:
    """Rewrite a results file in place without duplicates."""
    frame = read_rows(name)
    if not frame.empty:
        frame.to_csv(RESULTS / name, index=False)
    return frame


def existing_keys(name: str) -> set:
    frame = read_rows(name)
    if frame.empty or "cell_key" not in frame.columns:
        return set()
    return set(frame["cell_key"].astype(str))


# Six significant digits is well beyond the precision of anything measured
# here, and keeps the result tables small enough to travel in the archive.
FLOAT_FORMAT = "%.6g"


def write_table(name: str, frame: pd.DataFrame) -> None:
    """Write a finished table, overwriting any previous version."""
    (RESULTS / name).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / name, index=False, float_format=FLOAT_FORMAT)
