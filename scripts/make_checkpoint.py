"""Write ``checkpoint.zip`` at the repository root.

The archive carries the whole tree except the fetched inputs and the cached
intermediates, so it holds the code, the results, the figures, the documents
and the run state. Unpacking it into an empty directory is enough to continue
the work, and re-fetching the dataset against the manifest it carries is itself
a reproducibility check.

    python scripts/make_checkpoint.py
"""

from __future__ import annotations

import argparse
import zipfile

import _bootstrap  # noqa: F401

from uev import progress
from uev.logging_utils import get_logger
from uev.paths import ROOT, rel

LOG = get_logger("checkpoint")

EXCLUDED_DIRS = {"data/raw", "data/interim", "__pycache__", ".pytest_cache",
                 ".git", ".ipynb_checkpoints", ".egg-info"}
EXCLUDED_NAMES = {"checkpoint.zip", ".DS_Store"}


def is_excluded(relative: str) -> bool:
    if any(relative == name or relative.endswith("/" + name) for name in EXCLUDED_NAMES):
        return True
    parts = relative.split("/")
    for i in range(len(parts)):
        prefix = "/".join(parts[:i + 1])
        if prefix in EXCLUDED_DIRS or parts[i] in EXCLUDED_DIRS:
            return True
    if relative.endswith(".pyc"):
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="write checkpoint.zip")
    parser.add_argument("--name", default="checkpoint.zip")
    args = parser.parse_args()

    target = ROOT / args.name
    if target.exists():
        target.unlink()

    n_files, n_bytes = 0, 0
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            if is_excluded(relative):
                continue
            archive.write(path, relative)
            n_files += 1
            n_bytes += path.stat().st_size

    size = target.stat().st_size
    LOG.info("wrote %s: %d files, %.2f MB uncompressed, %.2f MB compressed",
             rel(target), n_files, n_bytes / 1e6, size / 1e6)
    if size > 25 * 1024 * 1024:
        LOG.warning("the archive is above the 25 MB budget")
    print(progress.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
