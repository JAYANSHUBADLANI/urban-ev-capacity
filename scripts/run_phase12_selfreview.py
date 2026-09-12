"""Phase 12: self review, determinism check and packaging.

Runs the release checks that a reviewer would otherwise have to run by hand:
the test suite, a determinism check on the two headline tables, presence of
every deliverable, and a scan for absolute paths in tracked files. Then writes
the archive.

    python scripts/run_phase12_selfreview.py
    python scripts/run_phase12_selfreview.py --skip-tests
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import _bootstrap  # noqa: F401

import pandas as pd

from uev import progress
from uev.grid import expected_cells
from uev.logging_utils import get_logger, timed
from uev.paths import DOCS, FIGURES, RESULTS, ROOT, ensure_dirs, rel
from uev.resultsio import existing_keys, write_table

LOG = get_logger("phase12")

REQUIRED_DOCS = ("preregistration.md", "method.md", "data_dictionary.md",
                 "results.md", "model_evidence_pack.md", "limitations.md")

REQUIRED_RESULTS = ("data_quality_report.csv", "quality_cross_check.csv",
                    "structural_breaks.csv", "structural_break_significance.csv",
                    "monitoring_window.csv", "grid_metrics.csv", "grid_zone_metrics.csv",
                    "monitor_detection.csv", "monitor_claim2.csv",
                    "retraining_backtest.csv", "segment_diagnostics.csv",
                    "segment_profiles.csv", "anomaly_agreement.csv",
                    "queue_calibration.csv", "marginal_value.csv",
                    "capacity_scenarios.csv", "capacity_claim3.csv",
                    "claim_outcomes.csv", "headline_model_table.csv",
                    "headline_capacity_table.csv")

# Directory roots that must never appear in a tracked file. They are assembled
# from their names rather than written out, so that this list does not itself
# become the one absolute path in the repository.
FORBIDDEN_ROOTS = ("Users", "home", "mnt", "content", "workspace", "tmp")
FORBIDDEN_PREFIXES = tuple(f"/{name}/" for name in FORBIDDEN_ROOTS)

TRACKED_SUFFIXES = (".py", ".md", ".sql", ".toml", ".txt", ".cfg", ".ini", ".yml", ".yaml")

SKIP_DIRS = {"data", "state", "__pycache__", ".pytest_cache", ".git"}


def tracked_files() -> List[Path]:
    out = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        parts = set(path.relative_to(ROOT).parts)
        if parts & SKIP_DIRS:
            continue
        if path.suffix in TRACKED_SUFFIXES or path.name == "LICENSE":
            out.append(path)
    return out


def check_absolute_paths() -> List[Dict]:
    findings = []
    for path in tracked_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            for prefix in FORBIDDEN_PREFIXES:
                if prefix in line:
                    findings.append({"file": rel(path), "line": number, "issue": prefix})
            if len(line) > 2 and line[1] == ":" and line[0].isalpha() and line[2] in "\\/":
                findings.append({"file": rel(path), "line": number, "issue": "drive letter"})
    return findings


def headline_digests() -> Dict[str, str]:
    from run_phase11_report import (HEADLINE_CAPACITY, HEADLINE_MODEL,
                                    build_headline_capacity_table,
                                    build_headline_model_table)
    out = {}
    for name, builder in ((HEADLINE_MODEL, build_headline_model_table),
                          (HEADLINE_CAPACITY, build_headline_capacity_table)):
        table = builder()
        if table.empty:
            out[name] = "absent"
            continue
        payload = table.to_csv(index=False, float_format="%.6g").encode("utf-8")
        out[name] = hashlib.sha256(payload).hexdigest()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="self review and packaging")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-archive", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    progress.mark_phase(12, "running")
    rows: List[Dict] = []

    def record(check: str, passed: bool, detail: str = "") -> None:
        rows.append({"check": check, "status": "pass" if passed else "fail", "detail": detail})
        LOG.info("%-34s %s  %s", check, "pass" if passed else "FAIL", detail)

    # 1. Tests.
    if not args.skip_tests:
        with timed("test suite", LOG):
            result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                                    cwd=ROOT, capture_output=True, text=True)
        tail = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        n_tests = 0
        for token in tail.replace("=", " ").split():
            if token.isdigit():
                n_tests = int(token)
                break
        record("test suite passes", result.returncode == 0, tail)
        record("at least 90 tests", n_tests >= 90, f"{n_tests} tests")

    # 2. Grid completeness.
    total = len(expected_cells())
    done = len(existing_keys("grid_metrics.csv"))
    record("grid complete", done >= total, f"{done} of {total} cells")

    # 3. Determinism of the two headline tables.
    with timed("determinism check", LOG):
        first = headline_digests()
        second = headline_digests()
    for name in first:
        match = first[name] == second[name] and first[name] != "absent"
        record(f"determinism {name}", match, f"sha256 {first[name][:32]}")

    # 4. Deliverables present.
    missing_docs = [d for d in REQUIRED_DOCS if not (DOCS / d).exists()]
    record("every document present", not missing_docs, ", ".join(missing_docs) or "all present")

    missing_results = [r for r in REQUIRED_RESULTS if not (RESULTS / r).exists()]
    record("every result table present", not missing_results,
           ", ".join(missing_results) or f"{len(REQUIRED_RESULTS)} tables")

    figures = sorted(FIGURES.glob("*.png"))
    record("at least ten figures", len(figures) >= 10, f"{len(figures)} figures")

    # 5. Claims resolved.
    claims_path = RESULTS / "claim_outcomes.csv"
    if claims_path.exists():
        claims = pd.read_csv(claims_path)
        resolved = set(claims["claim"]) == {1, 2, 3}
        verdicts = ", ".join(f"claim {int(r['claim'])} {r['verdict']}"
                             for _, r in claims.iterrows())
        record("all three claims resolved", resolved, verdicts)
    else:
        record("all three claims resolved", False, "claim_outcomes.csv is absent")

    # 6. Tree constraints.
    absolute = check_absolute_paths()
    record("no absolute paths in tracked files", not absolute,
           f"{len(absolute)} occurrences" if absolute else "none")
    if absolute:
        for finding in absolute[:10]:
            LOG.error("  %s:%d contains %s", finding["file"], finding["line"], finding["issue"])

    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    excluded = all(entry in ignored for entry in ("data/raw/", "data/interim/", "state/"))
    record("inputs and run state are excluded", excluded, "data/raw, data/interim, state")

    # 7. Manifest still matches.
    with timed("manifest verification", LOG):
        verify = subprocess.run([sys.executable, "scripts/fetch_data.py", "--verify-only"],
                                cwd=ROOT, capture_output=True, text=True)
    record("dataset digests match the manifest", verify.returncode == 0,
           "13 files" if verify.returncode == 0 else verify.stdout.strip()[-160:])

    report = pd.DataFrame(rows)
    write_table("self_review.csv", report)
    n_failed = int((report["status"] == "fail").sum())
    LOG.info("self review: %d checks, %d failed", len(report), n_failed)

    if not args.skip_archive:
        subprocess.run([sys.executable, "scripts/make_checkpoint.py"], cwd=ROOT, check=True)

    progress.mark_phase(12, "complete" if n_failed == 0 else "running",
                        f"{len(report)} checks, {n_failed} failed")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
