"""Loading the pinned tables into a single tidy long frame.

The published tables are wide: a time column followed by one column per zone.
They are reshaped once, joined, and cached, so that every later phase reads one
internal representation rather than re-deriving it.

Two timestamp formats appear in the dataset and both are parsed explicitly. The
five demand and price tables use ``2022-09-01 00:00:00``; the two weather tables
use ``2022/9/1 0:00``. Automatic inference is never used, because a silent
fallback on an ambiguous day and month order would corrupt the calendar
features without raising anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import (ALTERNATIVE_VOLUME, EXPECTED_END, EXPECTED_HOURS,
                     EXPECTED_START, EXPECTED_ZONES, TIME_FORMAT)
from .logging_utils import get_logger
from .paths import INTERIM, MANIFEST, RAW, rel

LOG = get_logger("load")

DEMAND_FORMAT = "%Y-%m-%d %H:%M:%S"
WEATHER_FORMAT = TIME_FORMAT

WIDE_TABLES: Dict[str, str] = {
    "occupancy": "occupancy.csv",
    "volume": "volume.csv",
    ALTERNATIVE_VOLUME: "volume-11kW.csv",
    "duration": "duration.csv",
    "e_price": "e_price.csv",
    "s_price": "s_price.csv",
}

WEATHER_COLUMNS = {
    "T": "air_temp_c",
    "P0": "pressure_station_mmhg",
    "P": "pressure_sea_mmhg",
    "U": "humidity_pct",
    "nRAIN": "rain_category",
    "Td": "dewpoint_c",
}


class DataError(RuntimeError):
    """Raised when an input does not match the documented schema."""


@dataclass
class Bundle:
    """Everything the later phases read."""

    long: pd.DataFrame          # zone_id, timestamp, the six measured series
    zones: pd.DataFrame         # one row per zone, capacity and geometry
    weather: pd.DataFrame       # one row per timestamp, both stations
    adjacency: pd.DataFrame     # zone by zone, 0/1
    distance: pd.DataFrame      # zone by zone, metres
    index: pd.DatetimeIndex     # the shared hourly index

    @property
    def zone_ids(self) -> List[int]:
        return list(self.zones["zone_id"])


def parse_time(values: pd.Series, fmt: str) -> pd.Series:
    """Parse a timestamp column with an explicit format and no fallback."""
    parsed = pd.to_datetime(values, format=fmt, errors="coerce")
    if parsed.isna().any():
        bad = values[parsed.isna()].head(3).tolist()
        raise DataError(f"timestamps did not match the format {fmt}: {bad}")
    return parsed


def assert_hourly(index: pd.DatetimeIndex, label: str) -> None:
    """Assert a complete, unique, strictly hourly index of the expected length."""
    if index.has_duplicates:
        dupes = index[index.duplicated()][:3].tolist()
        raise DataError(f"{label} contains duplicate timestamps: {dupes}")
    if not index.is_monotonic_increasing:
        raise DataError(f"{label} is not sorted in time")
    expected = pd.date_range(EXPECTED_START, EXPECTED_END, freq="h")
    if len(index) != len(expected) or not (index == expected).all():
        raise DataError(
            f"{label} is not the expected hourly index: got {len(index)} rows "
            f"from {index[0]} to {index[-1]}, expected {len(expected)}"
        )
    if len(expected) != EXPECTED_HOURS:
        raise DataError(f"the configured window is not {EXPECTED_HOURS} hours")


def _read_wide(filename: str) -> pd.DataFrame:
    frame = pd.read_csv(RAW / filename)
    if frame.columns[0] != "time":
        raise DataError(f"{filename} does not start with a time column")
    frame["time"] = parse_time(frame["time"], DEMAND_FORMAT)
    frame = frame.set_index("time").sort_index()
    assert_hourly(pd.DatetimeIndex(frame.index), filename)
    frame.columns = [int(c) for c in frame.columns]
    if len(frame.columns) != EXPECTED_ZONES:
        raise DataError(f"{filename} has {len(frame.columns)} zones, expected {EXPECTED_ZONES}")
    return frame.astype("float32")


def load_weather() -> pd.DataFrame:
    """Load both weather stations onto the shared hourly index."""
    frames = []
    for station, filename in (("central", "weather_central.csv"),
                              ("airport", "weather_airport.csv")):
        raw = pd.read_csv(RAW / filename)
        raw["time"] = parse_time(raw["time"], WEATHER_FORMAT)
        raw = raw.set_index("time").sort_index()
        assert_hourly(pd.DatetimeIndex(raw.index), filename)
        raw = raw.rename(columns=WEATHER_COLUMNS)
        keep = [c for c in WEATHER_COLUMNS.values() if c in raw.columns]
        raw = raw[keep].add_suffix(f"_{station}")
        frames.append(raw)
    weather = pd.concat(frames, axis=1).astype("float32")
    weather.index.name = "timestamp"
    return weather.reset_index()


def load_zone_attributes(zone_ids: List[int]) -> pd.DataFrame:
    """Aggregate station level attributes to the zone level.

    Capacity is a sum of the charging points of the stations in a zone, not a
    lookup, because ``inf.csv`` is station level and joins through ``TAZID``.
    """
    inf = pd.read_csv(RAW / "inf.csv")
    required = {"station_id", "longitude", "latitude", "charge_count", "TAZID", "area", "perimeter"}
    missing = required - set(inf.columns)
    if missing:
        raise DataError(f"inf.csv is missing columns: {sorted(missing)}")

    grouped = inf.groupby("TAZID")
    zones = pd.DataFrame({
        "zone_id": grouped.size().index.astype(int),
        "n_stations": grouped.size().to_numpy(),
        "capacity_points": grouped["charge_count"].sum().to_numpy(),
        "mean_points_per_station": grouped["charge_count"].mean().to_numpy(),
        "max_points_per_station": grouped["charge_count"].max().to_numpy(),
        "centroid_lon": grouped["longitude"].mean().to_numpy(),
        "centroid_lat": grouped["latitude"].mean().to_numpy(),
        "area_m2": grouped["area"].first().to_numpy(),
        "perimeter_m": grouped["perimeter"].first().to_numpy(),
    })
    zones = zones.set_index("zone_id").reindex(zone_ids)
    if zones["capacity_points"].isna().any():
        absent = zones.index[zones["capacity_points"].isna()].tolist()[:5]
        raise DataError(f"zones present in the demand tables but absent from inf.csv: {absent}")
    zones = zones.reset_index().rename(columns={"index": "zone_id"})
    zones["zone_id"] = zones["zone_id"].astype(int)
    zones["capacity_points"] = zones["capacity_points"].astype(int)
    zones["n_stations"] = zones["n_stations"].astype(int)
    zones["station_density"] = zones["n_stations"] / (zones["area_m2"] / 1e6)
    zones["points_density"] = zones["capacity_points"] / (zones["area_m2"] / 1e6)
    zones["shape_index"] = zones["perimeter_m"] / np.sqrt(zones["area_m2"].clip(lower=1.0))
    return zones


def load_matrix(filename: str, zone_ids: List[int]) -> pd.DataFrame:
    """Load a square zone by zone matrix and align it to the zone order."""
    matrix = pd.read_csv(RAW / filename, index_col=False)
    matrix.columns = [int(c) for c in matrix.columns]
    if matrix.shape[0] != matrix.shape[1]:
        raise DataError(f"{filename} is not square: {matrix.shape}")
    if list(matrix.columns) != list(zone_ids):
        raise DataError(f"{filename} columns do not match the demand table zone order")
    matrix.index = pd.Index(zone_ids, name="zone_id")
    matrix.columns = pd.Index(zone_ids, name="zone_id")
    return matrix.astype("float32")


def build_long(wide: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack the wide tables into one long frame keyed by zone and timestamp."""
    pieces = []
    for name, frame in wide.items():
        stacked = frame.stack()
        stacked.index.names = ["timestamp", "zone_id"]
        pieces.append(stacked.rename(name))
    long = pd.concat(pieces, axis=1).reset_index()
    long["zone_id"] = long["zone_id"].astype("int32")
    long = long.sort_values(["zone_id", "timestamp"], kind="stable").reset_index(drop=True)
    return long


def cache_path(name: str) -> "object":
    return INTERIM / name


def load_bundle(use_cache: bool = True) -> Bundle:
    """Load everything, using the cached long frame when it is current."""
    INTERIM.mkdir(parents=True, exist_ok=True)
    long_cache = cache_path("long.parquet")
    zones_cache = cache_path("zones.parquet")
    weather_cache = cache_path("weather.parquet")

    if use_cache and long_cache.exists() and zones_cache.exists() and weather_cache.exists():
        LOG.info("reading cached frames from %s", rel(INTERIM))
        long = pd.read_parquet(long_cache)
        zones = pd.read_parquet(zones_cache)
        weather = pd.read_parquet(weather_cache)
        zone_ids = list(zones["zone_id"])
    else:
        LOG.info("reading the wide tables from %s", rel(RAW))
        wide = {name: _read_wide(filename) for name, filename in WIDE_TABLES.items()}
        reference = list(wide["occupancy"].columns)
        for name, frame in wide.items():
            if list(frame.columns) != reference:
                raise DataError(f"{name} does not share the zone order of occupancy")
        zone_ids = [int(z) for z in reference]
        long = build_long(wide)
        zones = load_zone_attributes(zone_ids)
        weather = load_weather()
        long.to_parquet(long_cache, index=False)
        zones.to_parquet(zones_cache, index=False)
        weather.to_parquet(weather_cache, index=False)
        LOG.info("cached the tidy frames to %s", rel(INTERIM))

    adjacency = load_matrix("adj.csv", zone_ids)
    distance = load_matrix("distance.csv", zone_ids)
    index = pd.DatetimeIndex(sorted(long["timestamp"].unique()))
    assert_hourly(index, "tidy long frame")

    expected_rows = EXPECTED_HOURS * EXPECTED_ZONES
    if len(long) != expected_rows:
        raise DataError(f"the long frame has {len(long)} rows, expected {expected_rows}")

    return Bundle(long=long, zones=zones, weather=weather,
                  adjacency=adjacency, distance=distance, index=index)


def manifest_digests() -> Dict[str, str]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {name: entry["sha256"] for name, entry in payload.get("files", {}).items()}
