"""Phase 2: build the tidy long frame and the zone attribute table.

Reads the pinned files, reshapes the wide tables, aggregates the station level
attributes to zones and caches the result. Running this again with the cache in
place is a no-op unless ``--rebuild`` is passed.

    python scripts/run_phase02_load.py
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from uev import progress
from uev.io_load import load_bundle, manifest_digests
from uev.logging_utils import get_logger, timed
from uev.paths import ensure_dirs, rel
from uev.resultsio import write_table

LOG = get_logger("phase02")


def main() -> int:
    parser = argparse.ArgumentParser(description="build the tidy frames")
    parser.add_argument("--rebuild", action="store_true", help="ignore the cached frames")
    args = parser.parse_args()

    ensure_dirs()
    progress.mark_phase(2, "running")
    digests = manifest_digests()
    if not digests:
        raise SystemExit("no manifest found; run scripts/fetch_data.py first")
    LOG.info("manifest covers %d files", len(digests))

    with timed("loading the tidy frames", LOG):
        bundle = load_bundle(use_cache=not args.rebuild)

    LOG.info("long frame %d rows, %d zones, %d hours",
             len(bundle.long), bundle.zones.shape[0], len(bundle.index))
    LOG.info("window %s to %s", bundle.index[0], bundle.index[-1])

    write_table("zone_attributes.csv", bundle.zones)
    LOG.info("wrote %s", rel("results/zone_attributes.csv"))

    progress.mark_phase(2, "complete",
                        f"{len(bundle.long)} rows, {bundle.zones.shape[0]} zones")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
