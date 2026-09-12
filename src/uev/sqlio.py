"""The SQL reporting layer.

Aggregations that feed reporting are kept as real ``.sql`` files under ``sql/``
and executed by the pipeline. The default engine is ``sqlite3`` from the
standard library, so the layer always runs. If ``duckdb`` happens to be
installed it is used instead, because it is faster on the same SQL text; the
schema materialises calendar parts as columns precisely so that one text runs
unchanged on both engines.
"""

from __future__ import annotations

import sqlite3
from typing import Dict, List, Optional

import pandas as pd

from .config import HOLIDAYS
from .graph import degree, symmetrise
from .io_load import Bundle
from .logging_utils import get_logger
from .paths import DB, SQL, rel

LOG = get_logger("sql")


def duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401
    except ImportError:
        return False
    return True


def read_sql_file(name: str) -> str:
    path = SQL / name
    if not path.exists():
        raise FileNotFoundError(f"missing query: {rel(path)}")
    return path.read_text(encoding="utf-8")


def split_statements(script: str) -> List[str]:
    """Split a script into statements, dropping comment only fragments."""
    statements = []
    for chunk in script.split(";"):
        lines = [line for line in chunk.splitlines()
                 if line.strip() and not line.strip().startswith("--")]
        if lines:
            statements.append("\n".join(lines).strip())
    return statements


def observations_frame(bundle: Bundle) -> pd.DataFrame:
    """The row set loaded into the reporting table."""
    long = bundle.long.copy()
    capacity = bundle.zones.set_index("zone_id")["capacity_points"]
    timestamps = pd.DatetimeIndex(long["timestamp"])
    holidays = {pd.Timestamp(day).date() for day in HOLIDAYS}

    long["ts"] = timestamps.strftime("%Y-%m-%d %H:%M:%S")
    long["obs_date"] = timestamps.strftime("%Y-%m-%d")
    long["hour_of_day"] = timestamps.hour.astype("int16")
    long["day_of_week"] = timestamps.dayofweek.astype("int16")
    long["is_weekend"] = (timestamps.dayofweek >= 5).astype("int8")
    long["is_holiday"] = pd.Series(timestamps.date, index=long.index).isin(holidays).astype("int8")
    long["capacity_points"] = long["zone_id"].map(capacity).astype("int32")
    long["utilisation"] = (long["occupancy"] / long["capacity_points"]).astype("float32")

    columns = ["zone_id", "ts", "obs_date", "hour_of_day", "day_of_week", "is_weekend",
               "is_holiday", "occupancy", "volume", "volume_11kw", "duration",
               "e_price", "s_price", "capacity_points", "utilisation"]
    return long[columns]


def edges_frame(bundle: Bundle) -> pd.DataFrame:
    """Undirected zone edges with their published distance."""
    undirected = symmetrise(bundle.adjacency)
    stacked = undirected.stack()
    stacked = stacked[stacked > 0]
    stacked.index.names = ["zone_id", "neighbour_id"]
    edges = stacked.rename("adjacent").reset_index()
    distances = bundle.distance.stack()
    distances.index.names = ["zone_id", "neighbour_id"]
    distances = distances.rename("distance_m").reset_index()
    edges = edges.merge(distances, on=["zone_id", "neighbour_id"], how="left")
    return edges[["zone_id", "neighbour_id", "distance_m"]]


def weather_frame(bundle: Bundle) -> pd.DataFrame:
    """Weather in long form, one row per hour and station."""
    pieces = []
    for station in ("central", "airport"):
        piece = pd.DataFrame({
            "ts": pd.DatetimeIndex(bundle.weather["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            "air_temp_c": bundle.weather[f"air_temp_c_{station}"],
            "humidity_pct": bundle.weather[f"humidity_pct_{station}"],
            "rain_category": bundle.weather[f"rain_category_{station}"],
            "dewpoint_c": bundle.weather[f"dewpoint_c_{station}"],
            "station": station,
        })
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def build_database(bundle: Bundle, path=DB) -> "object":
    """Create the reporting database from the tidy frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    zones = bundle.zones.copy()
    zones["n_neighbours"] = zones["zone_id"].map(degree(bundle.adjacency)).astype(int)
    zones["max_points_per_station"] = zones["max_points_per_station"].astype(int)

    connection = sqlite3.connect(path)
    try:
        connection.executescript(read_sql_file("schema.sql"))
        observations_frame(bundle).to_sql("observations", connection,
                                          if_exists="append", index=False, chunksize=50_000)
        zones.to_sql("zones", connection, if_exists="append", index=False)
        weather_frame(bundle).to_sql("weather", connection, if_exists="append", index=False)
        edges_frame(bundle).to_sql("zone_edges", connection, if_exists="append", index=False)
        connection.commit()
    finally:
        connection.close()
    LOG.info("built the reporting database at %s", rel(path))
    return path


def query(name: str, path=DB, engine: Optional[str] = None) -> pd.DataFrame:
    """Execute a query file and return the result."""
    sql_text = read_sql_file(name)
    use_duckdb = duckdb_available() if engine is None else (engine == "duckdb")
    if use_duckdb:
        import duckdb
        connection = duckdb.connect()
        try:
            connection.execute(f"ATTACH '{path.as_posix()}' AS src (TYPE SQLITE)")
            connection.execute("USE src")
            return connection.execute(sql_text).df()
        finally:
            connection.close()
    connection = sqlite3.connect(path)
    try:
        return pd.read_sql_query(sql_text, connection)
    finally:
        connection.close()


def run_reports(path=DB) -> Dict[str, pd.DataFrame]:
    """Run every reporting query and return the results by name."""
    names = ["daily_profile.sql", "hourly_profile.sql", "zone_ranking.sql",
             "peak_windows.sql", "checks.sql"]
    out: Dict[str, pd.DataFrame] = {}
    for name in names:
        frame = query(name, path=path)
        out[name.replace(".sql", "")] = frame
        LOG.info("%-20s returned %d rows", name, len(frame))
    return out
