"""Download the pinned UrbanEV files and record a SHA-256 manifest.

The dataset is fetched from a single pinned commit. Every file is hashed on
arrival. If a manifest already exists, the digests are compared and a mismatch
stops the run, because a reproducibility claim that is not checked is only a
claim.

Usage from the repository root:

    python scripts/fetch_data.py
    python scripts/fetch_data.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uev.config import BASE_URL, COMMIT, DATA_FILES  # noqa: E402
from uev.logging_utils import get_logger  # noqa: E402
from uev.paths import MANIFEST, RAW, ensure_dirs, rel  # noqa: E402

LOG = get_logger("fetch")
TIMEOUT = 120
RETRIES = 3


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download(name: str, destination: Path) -> None:
    url = f"{BASE_URL}/{name}"
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "uev-fetch/1.0"})
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                payload = response.read()
            destination.write_bytes(payload)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            LOG.warning("attempt %d/%d failed for %s: %s", attempt, RETRIES, name, error)
    raise SystemExit(
        f"could not download {name} from the pinned commit after {RETRIES} attempts: {last_error}"
    )


def load_manifest() -> dict:
    if MANIFEST.exists():
        with MANIFEST.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {"commit": COMMIT, "files": {}}


def main() -> int:
    parser = argparse.ArgumentParser(description="fetch the pinned dataset")
    parser.add_argument("--verify-only", action="store_true",
                        help="hash the files already on disk and compare against the manifest")
    parser.add_argument("--force", action="store_true",
                        help="re-download every file even when it is already present")
    args = parser.parse_args()

    ensure_dirs()
    manifest = load_manifest()
    if manifest.get("commit") != COMMIT:
        raise SystemExit(
            "the manifest was written for a different commit than the one configured"
        )

    recorded = manifest.setdefault("files", {})
    mismatches, fetched, verified = [], 0, 0

    for name in DATA_FILES:
        target = RAW / name
        if args.verify_only and not target.exists():
            mismatches.append(f"{name}: missing from {rel(RAW)}")
            continue
        if args.force or not target.exists():
            LOG.info("downloading %s", name)
            download(name, target)
            fetched += 1
        digest = sha256_of(target)
        size = target.stat().st_size
        if name in recorded:
            if recorded[name]["sha256"] != digest:
                mismatches.append(
                    f"{name}: digest {digest[:12]} does not match the recorded "
                    f"{recorded[name]['sha256'][:12]}"
                )
            else:
                verified += 1
        else:
            recorded[name] = {"sha256": digest, "bytes": size}
        LOG.info("%-22s %9d bytes  %s", name, size, digest[:16])

    if mismatches:
        for item in mismatches:
            LOG.error("%s", item)
        raise SystemExit("dataset digests do not match the manifest; stopping")

    if not args.verify_only:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    total = sum(entry["bytes"] for entry in recorded.values())
    LOG.info("files %d, downloaded %d, digest verified %d, total %.1f MB",
             len(recorded), fetched, verified, total / 1e6)
    LOG.info("manifest at %s", rel(MANIFEST))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
