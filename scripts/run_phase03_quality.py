"""Phase 3: data quality suite and the SQL reporting layer.

Runs the checks in pandas, builds the reporting database, runs the same
assertions in SQL and compares the two implementations against each other. The
reports are written to ``results/``.

    python scripts/run_phase03_quality.py
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

import pandas as pd

from uev import progress, sqlio
from uev.dictionary import write as write_dictionary
from uev.io_load import load_bundle
from uev.logging_utils import get_logger, timed
from uev.paths import ensure_dirs, rel
from uev.quality import quality_summary, run_checks
from uev.resultsio import write_table

LOG = get_logger("phase03")

# Checks that are computed both ways and must agree.
CROSS_CHECKED = {
    "sql_occupancy_without_volume": "cross_occupancy_without_volume",
    "sql_volume_without_occupancy": "cross_volume_without_occupancy",
    "sql_saturated_zone_hours": "saturated_zone_hours",
    "sql_duplicate_keys": "duplicate_keys",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="run the quality suite and SQL layer")
    parser.add_argument("--skip-db", action="store_true",
                        help="reuse the existing reporting database")
    args = parser.parse_args()

    ensure_dirs()
    progress.mark_phase(3, "running")
    bundle = load_bundle()

    with timed("pandas quality checks", LOG):
        report = run_checks(bundle)
    write_table("data_quality_report.csv", report)
    print(quality_summary(report))

    if not args.skip_db:
        with timed("building the reporting database", LOG):
            sqlio.build_database(bundle)

    with timed("SQL reports", LOG):
        reports = sqlio.run_reports()

    for name, frame in reports.items():
        write_table(f"sql_{name}.csv", frame)

    sql_checks = reports["checks"].set_index("check_id")
    pandas_checks = report.set_index("check_id")
    rows = []
    for sql_id, pandas_id in CROSS_CHECKED.items():
        sql_value = int(sql_checks.loc[sql_id, "n_failing"])
        pandas_value = int(pandas_checks.loc[pandas_id, "n_failing"])
        rows.append({"sql_check": sql_id, "pandas_check": pandas_id,
                     "sql_count": sql_value, "pandas_count": pandas_value,
                     "agree": sql_value == pandas_value})
    agreement = pd.DataFrame(rows)
    write_table("quality_cross_check.csv", agreement)
    LOG.info("SQL and pandas agree on %d of %d cross checked counts",
             int(agreement["agree"].sum()), len(agreement))
    if not agreement["agree"].all():
        raise SystemExit("the SQL and pandas quality implementations disagree")

    dictionary_path = write_dictionary(bundle)
    LOG.info("wrote %s", rel(dictionary_path))

    n_fail = int((report["status"] == "fail").sum())
    progress.mark_phase(3, "complete",
                        f"{len(report)} checks, {n_fail} structural failures recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
