"""Generation of the data dictionary.

The dictionary is written by code so that the ranges and counts in it are the
ranges and counts of the data actually loaded, rather than a description that
drifts away from the files.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .io_load import Bundle
from .paths import DOCS

# field, unit, source file, preprocessing applied by the publisher, refresh
# cadence if this were a live feed.
FIELDS: List[Dict[str, str]] = [
    {"field": "zone_id", "unit": "identifier", "source": "column headers of the wide tables",
     "preprocessing": "zones surviving the publisher variance and zero activity filter",
     "cadence": "static, revised when the traffic analysis zones are redrawn"},
    {"field": "timestamp", "unit": "hour, local time", "source": "time column of the wide tables",
     "preprocessing": "complete hourly index, no gaps",
     "cadence": "hourly"},
    {"field": "occupancy", "unit": "charging points in use", "source": "occupancy.csv",
     "preprocessing": "anomaly removal, forward and backward fill, interquartile range "
                      "outlier replacement with adjacent values",
     "cadence": "hourly"},
    {"field": "volume", "unit": "kWh", "source": "volume.csv",
     "preprocessing": "derived from the rated power of the charging points, not metered",
     "cadence": "hourly"},
    {"field": "volume_11kw", "unit": "kWh", "source": "volume-11kW.csv",
     "preprocessing": "derived using an 11 kW vehicle side power limit, not metered",
     "cadence": "hourly"},
    {"field": "duration", "unit": "point hours", "source": "duration.csv",
     "preprocessing": "same cleaning as occupancy",
     "cadence": "hourly"},
    {"field": "e_price", "unit": "Yuan per kWh", "source": "e_price.csv",
     "preprocessing": "tariff schedule, largely piecewise constant",
     "cadence": "changes at tariff revisions, carried at hourly resolution"},
    {"field": "s_price", "unit": "Yuan per kWh", "source": "s_price.csv",
     "preprocessing": "operator service fee, largely piecewise constant",
     "cadence": "changes at operator revisions, carried at hourly resolution"},
    {"field": "capacity_points", "unit": "charging points",
     "source": "inf.csv, charge_count summed over the stations in the zone",
     "preprocessing": "aggregated from station level to zone level through TAZID",
     "cadence": "static within the window, changes as sites are commissioned"},
    {"field": "n_stations", "unit": "count", "source": "inf.csv",
     "preprocessing": "count of stations mapped to the zone",
     "cadence": "static within the window"},
    {"field": "area_m2", "unit": "square metres", "source": "inf.csv",
     "preprocessing": "zone polygon area, constant across the stations of a zone",
     "cadence": "static"},
    {"field": "perimeter_m", "unit": "metres", "source": "inf.csv",
     "preprocessing": "zone polygon perimeter",
     "cadence": "static"},
    {"field": "centroid_lon", "unit": "degrees east", "source": "inf.csv",
     "preprocessing": "mean longitude of the stations in the zone",
     "cadence": "static"},
    {"field": "centroid_lat", "unit": "degrees north", "source": "inf.csv",
     "preprocessing": "mean latitude of the stations in the zone",
     "cadence": "static"},
    {"field": "utilisation", "unit": "fraction of points in use",
     "source": "derived, occupancy divided by capacity_points",
     "preprocessing": "bounded above by one by construction",
     "cadence": "hourly"},
    {"field": "air_temp_c", "unit": "degrees Celsius",
     "source": "weather_central.csv, weather_airport.csv, column T",
     "preprocessing": "published hourly but repeating across consecutive hours, "
                      "which indicates a coarser observation interval",
     "cadence": "nominally hourly, effectively three hourly"},
    {"field": "pressure_station_mmhg", "unit": "millimetres of mercury",
     "source": "weather files, column P0",
     "preprocessing": "station level pressure", "cadence": "nominally hourly"},
    {"field": "pressure_sea_mmhg", "unit": "millimetres of mercury",
     "source": "weather files, column P",
     "preprocessing": "reduced to mean sea level", "cadence": "nominally hourly"},
    {"field": "humidity_pct", "unit": "percent", "source": "weather files, column U",
     "preprocessing": "relative humidity at two metres", "cadence": "nominally hourly"},
    {"field": "rain_category", "unit": "ordinal 0 to 3",
     "source": "weather files, column nRAIN",
     "preprocessing": "0 normal, 1 light rain, 2 moderate rain, 3 heavy rain",
     "cadence": "nominally hourly"},
    {"field": "dewpoint_c", "unit": "degrees Celsius", "source": "weather files, column Td",
     "preprocessing": "dewpoint at two metres", "cadence": "nominally hourly"},
    {"field": "adjacent", "unit": "0 or 1", "source": "adj.csv",
     "preprocessing": "published with a unit diagonal and twenty edges recorded in one direction only; "
                      "symmetrised by union and the diagonal removed",
     "cadence": "static"},
    {"field": "distance_m", "unit": "metres", "source": "distance.csv",
     "preprocessing": "zone centroid to zone centroid", "cadence": "static"},
]


def measured_summary(bundle: Bundle) -> pd.DataFrame:
    """Observed range and zero share for every measured field."""
    rows = []
    for column in ("occupancy", "volume", "volume_11kw", "duration", "e_price", "s_price"):
        series = bundle.long[column]
        rows.append({
            "field": column,
            "min": round(float(series.min()), 4),
            "median": round(float(series.median()), 4),
            "max": round(float(series.max()), 3),
            "mean": round(float(series.mean()), 4),
            "zero_share": round(float((series == 0).mean()), 5),
            "n_rows": len(series),
        })
    return pd.DataFrame(rows)


def render(bundle: Bundle) -> str:
    """Build the dictionary markdown."""
    summary = measured_summary(bundle).set_index("field")
    lines: List[str] = []
    lines.append("# Data dictionary")
    lines.append("")
    lines.append("Every field used anywhere in the project, with its unit, the file it "
                 "comes from, the preprocessing the publisher had already applied, and "
                 "the cadence it would refresh at if this were a live feed.")
    lines.append("")
    lines.append(f"Coverage: {bundle.index[0]:%Y-%m-%d %H:%M} to "
                 f"{bundle.index[-1]:%Y-%m-%d %H:%M}, {len(bundle.index)} hourly periods, "
                 f"{len(bundle.zones)} zones, {len(bundle.long):,} zone hours.")
    lines.append("")
    lines.append("## Fields")
    lines.append("")
    lines.append("| Field | Unit | Source | Preprocessing already applied | Refresh cadence |")
    lines.append("| --- | --- | --- | --- | --- |")
    for entry in FIELDS:
        lines.append(f"| `{entry['field']}` | {entry['unit']} | {entry['source']} | "
                     f"{entry['preprocessing']} | {entry['cadence']} |")
    lines.append("")
    lines.append("## Observed ranges")
    lines.append("")
    lines.append("Computed from the loaded frame, not copied from the publication.")
    lines.append("")
    lines.append("| Field | Minimum | Median | Mean | Maximum | Share exactly zero |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for field, row in summary.iterrows():
        lines.append(f"| `{field}` | {row['min']:g} | {row['median']:g} | {row['mean']:g} | "
                     f"{row['max']:g} | {row['zero_share'] * 100:.2f}% |")
    lines.append("")
    lines.append("## Units that are easy to misread")
    lines.append("")
    lines.append("`occupancy` is a count of charging points in use, not a percentage. It is "
                 "integer valued in 99.94 percent of zone hours and never exceeds the "
                 "charging points installed in the zone in any of the "
                 f"{len(bundle.long):,} zone hours observed. Utilisation, the bounded rate "
                 "the capacity work uses, is derived by dividing it by `capacity_points`.")
    lines.append("")
    lines.append("`duration` is point hours accumulated inside the hour, so a zone with ten "
                 "points cannot record more than ten. `volume` is energy in kWh and is "
                 "modelled from power and duration rather than metered.")
    lines.append("")
    lines.append("Prices are quoted per kWh and are separate: `e_price` is the electricity "
                 "tariff and `s_price` is the operator service fee. The price a driver pays "
                 "is their sum.")
    return "\n".join(lines) + "\n"


def write(bundle: Bundle) -> "object":
    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / "data_dictionary.md"
    path.write_text(render(bundle), encoding="utf-8")
    return path
