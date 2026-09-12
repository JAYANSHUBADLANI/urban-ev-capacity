"""Shared fixtures.

Tests that need the dataset are skipped with a clear reason when the inputs
have not been fetched, so the suite is still runnable in a fresh tree. Tests of
the mechanics build their own small frames and never need the dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uev.paths import RAW  # noqa: E402

DATA_REASON = "the pinned dataset is not present; run scripts/fetch_data.py"


def dataset_present() -> bool:
    return (RAW / "occupancy.csv").exists()


needs_data = pytest.mark.skipif(not dataset_present(), reason=DATA_REASON)


@pytest.fixture(scope="session")
def bundle():
    if not dataset_present():
        pytest.skip(DATA_REASON)
    from uev.io_load import load_bundle
    return load_bundle()


@pytest.fixture(scope="session")
def synthetic_long():
    """A small tidy frame with a known shape, used for mechanics tests."""
    index = pd.date_range("2022-09-01", periods=24 * 21, freq="h")
    rows = []
    for zone in (1, 2, 3):
        hours = np.arange(len(index))
        base = 10 + 5 * zone
        signal = base + 4 * np.sin(2 * np.pi * hours / 24) + 0.5 * (hours / 24)
        rows.append(pd.DataFrame({
            "timestamp": index,
            "zone_id": zone,
            "occupancy": signal,
            "volume": signal * 7.0,
            "volume_11kw": signal * 7.0,
            "duration": signal * 0.8,
            "e_price": 1.0,
            "s_price": 0.5,
        }))
    frame = pd.concat(rows, ignore_index=True)
    return frame.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def synthetic_zones():
    return pd.DataFrame({
        "zone_id": [1, 2, 3],
        "capacity_points": [40, 60, 80],
        "n_stations": [2, 3, 4],
        "area_m2": [1e6, 2e6, 3e6],
        "perimeter_m": [4000.0, 5000.0, 6000.0],
        "centroid_lon": [114.0, 114.1, 114.2],
        "centroid_lat": [22.5, 22.6, 22.7],
        "mean_points_per_station": [20.0, 20.0, 20.0],
        "max_points_per_station": [20, 20, 20],
        "station_density": [2.0, 1.5, 1.33],
        "points_density": [40.0, 30.0, 26.7],
        "shape_index": [4.0, 3.5, 3.4],
    })


@pytest.fixture(scope="session")
def quality_report(bundle):
    from uev.quality import run_checks
    return run_checks(bundle)


@pytest.fixture(scope="session")
def occupancy_features(bundle):
    from uev.features import build_features
    return build_features(bundle, "occupancy")
