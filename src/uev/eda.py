"""Exploratory analysis feeding the write up and the later phases.

The aggregate series built here are also the series the structural break
detection runs on, and the zone volume deciles defined here are the grouping
used throughout the error analysis, so that one definition is shared rather
than recomputed slightly differently in three places.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .io_load import Bundle
from .logging_utils import get_logger

LOG = get_logger("eda")

N_DECILES = 10


def aggregate_series(bundle: Bundle) -> pd.DataFrame:
    """City wide hourly totals plus the capacity weighted utilisation."""
    capacity = int(bundle.zones["capacity_points"].sum())
    frame = bundle.long.groupby("timestamp").agg(
        occupancy=("occupancy", "sum"),
        volume=("volume", "sum"),
        volume_11kw=("volume_11kw", "sum"),
        duration=("duration", "sum"),
        e_price=("e_price", "mean"),
        s_price=("s_price", "mean"),
    )
    frame["utilisation"] = frame["occupancy"] / capacity
    frame["total_capacity_points"] = capacity
    return frame


def daily_series(hourly: pd.DataFrame) -> pd.DataFrame:
    """Daily aggregates used by the change point detectors."""
    return hourly.resample("D").agg({
        "utilisation": "mean",
        "occupancy": "mean",
        "volume": "sum",
        "volume_11kw": "sum",
        "duration": "sum",
        "e_price": "mean",
        "s_price": "mean",
    })


def zone_volume_deciles(bundle: Bundle, column: str = "volume") -> pd.DataFrame:
    """Assign every zone to a decile of total volume, decile 1 being the lowest.

    A pooled error metric is dominated by the high volume zones, so every error
    table in the project is also reported across these groups.
    """
    totals = bundle.long.groupby("zone_id")[column].sum().rename("total_volume")
    ranks = totals.rank(method="first")
    decile = np.ceil(ranks * N_DECILES / len(totals)).astype(int).clip(1, N_DECILES)
    mean_level = bundle.long.groupby("zone_id")[column].mean().rename("mean_hourly_volume")
    out = pd.concat([totals, mean_level, decile.rename("volume_decile")], axis=1)
    out.index.name = "zone_id"
    return out.reset_index()


def hour_of_day_profile(bundle: Bundle) -> pd.DataFrame:
    """City wide shape of the day, split by weekday and weekend."""
    frame = bundle.long.copy()
    stamps = pd.DatetimeIndex(frame["timestamp"])
    frame["hour_of_day"] = stamps.hour
    frame["is_weekend"] = (stamps.dayofweek >= 5).astype(int)
    grouped = frame.groupby(["is_weekend", "hour_of_day"]).agg(
        mean_occupancy=("occupancy", "mean"),
        mean_volume=("volume", "mean"),
        mean_duration=("duration", "mean"),
    )
    return grouped.reset_index()


def saturation_profile(bundle: Bundle) -> pd.DataFrame:
    """How often each zone sits at or near its ceiling."""
    capacity = bundle.zones.set_index("zone_id")["capacity_points"]
    frame = bundle.long[["zone_id", "occupancy"]].copy()
    frame["capacity"] = frame["zone_id"].map(capacity)
    frame["utilisation"] = frame["occupancy"] / frame["capacity"]
    grouped = frame.groupby("zone_id")["utilisation"]
    out = pd.DataFrame({
        "mean_utilisation": grouped.mean(),
        "p95_utilisation": grouped.quantile(0.95),
        "max_utilisation": grouped.max(),
        "share_at_ceiling": frame.assign(at=frame["utilisation"] >= 0.999)
                                 .groupby("zone_id")["at"].mean(),
        "share_above_90": frame.assign(hi=frame["utilisation"] >= 0.90)
                               .groupby("zone_id")["hi"].mean(),
    })
    out["capacity_points"] = capacity
    return out.reset_index()


def price_variation(bundle: Bundle) -> pd.DataFrame:
    """How much the two price series actually move, per zone.

    Price is endogenous, so nothing causal is read from these numbers. They are
    here to show how little variation there is to work with in the first place.
    """
    grouped = bundle.long.groupby("zone_id")
    out = pd.DataFrame({
        "e_price_mean": grouped["e_price"].mean(),
        "e_price_std": grouped["e_price"].std(),
        "e_price_changes": grouped["e_price"].apply(lambda s: int((s.diff() != 0).sum())),
        "s_price_mean": grouped["s_price"].mean(),
        "s_price_std": grouped["s_price"].std(),
        "s_price_changes": grouped["s_price"].apply(lambda s: int((s.diff() != 0).sum())),
    })
    out["total_price_mean"] = out["e_price_mean"] + out["s_price_mean"]
    return out.reset_index()


def volume_estimate_comparison(bundle: Bundle) -> pd.DataFrame:
    """The two published energy estimates, side by side, per zone.

    ``volume`` applies the rated power of the charging points and
    ``volume_11kw`` applies an 11 kW vehicle side limit. For most zones the
    implied power in the second is the first capped at 11 kW, but not for all of
    them: a minority imply more power under the 11 kW series than under the
    rated one. Every energy figure in the project depends on which of the two is
    believed, so both are carried through.
    """
    grouped = bundle.long.groupby("zone_id")
    out = pd.DataFrame({
        "volume_rated_kwh": grouped["volume"].sum(),
        "volume_11kw_kwh": grouped["volume_11kw"].sum(),
        "point_hours": grouped["duration"].sum(),
    })
    out["implied_kw_rated"] = out["volume_rated_kwh"] / out["point_hours"].replace(0, np.nan)
    out["implied_kw_11"] = out["volume_11kw_kwh"] / out["point_hours"].replace(0, np.nan)
    out["ratio"] = out["volume_rated_kwh"] / out["volume_11kw_kwh"].replace(0, np.nan)
    return out.reset_index()


def summary_rows(bundle: Bundle) -> List[Dict]:
    """Headline descriptive numbers, each with its sample size."""
    long = bundle.long
    capacity = bundle.zones.set_index("zone_id")["capacity_points"]
    utilisation = long["occupancy"] / long["zone_id"].map(capacity)
    rows = [
        {"quantity": "zone hours observed", "value": len(long), "unit": "rows"},
        {"quantity": "zones", "value": long["zone_id"].nunique(), "unit": "zones"},
        {"quantity": "hours", "value": long["timestamp"].nunique(), "unit": "hours"},
        {"quantity": "charging points", "value": int(capacity.sum()), "unit": "points"},
        {"quantity": "stations", "value": int(bundle.zones["n_stations"].sum()), "unit": "stations"},
        {"quantity": "total energy, rated power estimate",
         "value": float(long["volume"].sum()), "unit": "kWh"},
        {"quantity": "total energy, 11 kW estimate",
         "value": float(long["volume_11kw"].sum()), "unit": "kWh"},
        {"quantity": "ratio of the two energy estimates",
         "value": float(long["volume"].sum() / long["volume_11kw"].sum()), "unit": "ratio"},
        {"quantity": "mean utilisation", "value": float(utilisation.mean()), "unit": "fraction"},
        {"quantity": "zone hours at the ceiling",
         "value": int((utilisation >= 0.999).sum()), "unit": "rows"},
        {"quantity": "zones that reach the ceiling at least once",
         "value": int((utilisation.groupby(long["zone_id"]).max() >= 0.999).sum()),
         "unit": "zones"},
        {"quantity": "total point hours", "value": float(long["duration"].sum()),
         "unit": "point hours"},
    ]
    return rows
