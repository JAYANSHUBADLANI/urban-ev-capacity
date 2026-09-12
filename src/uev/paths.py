"""Repository relative path resolution.

Every path in the project is derived from the repository root, which is located
by walking up from this module. Nothing in this module is ever rendered into a
log line or a results file; callers that need a printable path use
:func:`rel` to obtain a repository relative string.
"""

from __future__ import annotations

from pathlib import Path

# src/uev/paths.py -> src/uev -> src -> repository root
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
STATE = ROOT / "state"
SQL = ROOT / "sql"
DOCS = ROOT / "docs"

MANIFEST = DATA / "manifest.json"
PROGRESS = STATE / "progress.json"
DB = INTERIM / "uev.sqlite"


def ensure_dirs() -> None:
    """Create the directories the pipeline writes into."""
    for path in (RAW, INTERIM, RESULTS, FIGURES, STATE):
        path.mkdir(parents=True, exist_ok=True)


def rel(path) -> str:
    """Return a path as a repository relative POSIX string.

    Paths outside the repository are returned by name only, so that no absolute
    location can leak into a log line or a written artefact.
    """
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.name
