# Data dictionary

Every field used anywhere in the project, with its unit, the file it comes from, the preprocessing the publisher had already applied, and the cadence it would refresh at if this were a live feed.

Coverage: 2022-09-01 00:00 to 2023-02-28 23:00, 4344 hourly periods, 275 zones, 1,194,600 zone hours.

## Fields

| Field | Unit | Source | Preprocessing already applied | Refresh cadence |
| --- | --- | --- | --- | --- |
| `zone_id` | identifier | column headers of the wide tables | zones surviving the publisher variance and zero activity filter | static, revised when the traffic analysis zones are redrawn |
| `timestamp` | hour, local time | time column of the wide tables | complete hourly index, no gaps | hourly |
| `occupancy` | charging points in use | occupancy.csv | anomaly removal, forward and backward fill, interquartile range outlier replacement with adjacent values | hourly |
| `volume` | kWh | volume.csv | derived from the rated power of the charging points, not metered | hourly |
| `volume_11kw` | kWh | volume-11kW.csv | derived using an 11 kW vehicle side power limit, not metered | hourly |
| `duration` | point hours | duration.csv | same cleaning as occupancy | hourly |
| `e_price` | Yuan per kWh | e_price.csv | tariff schedule, largely piecewise constant | changes at tariff revisions, carried at hourly resolution |
| `s_price` | Yuan per kWh | s_price.csv | operator service fee, largely piecewise constant | changes at operator revisions, carried at hourly resolution |
| `capacity_points` | charging points | inf.csv, charge_count summed over the stations in the zone | aggregated from station level to zone level through TAZID | static within the window, changes as sites are commissioned |
| `n_stations` | count | inf.csv | count of stations mapped to the zone | static within the window |
| `area_m2` | square metres | inf.csv | zone polygon area, constant across the stations of a zone | static |
| `perimeter_m` | metres | inf.csv | zone polygon perimeter | static |
| `centroid_lon` | degrees east | inf.csv | mean longitude of the stations in the zone | static |
| `centroid_lat` | degrees north | inf.csv | mean latitude of the stations in the zone | static |
| `utilisation` | fraction of points in use | derived, occupancy divided by capacity_points | bounded above by one by construction | hourly |
| `air_temp_c` | degrees Celsius | weather_central.csv, weather_airport.csv, column T | published hourly but repeating across consecutive hours, which indicates a coarser observation interval | nominally hourly, effectively three hourly |
| `pressure_station_mmhg` | millimetres of mercury | weather files, column P0 | station level pressure | nominally hourly |
| `pressure_sea_mmhg` | millimetres of mercury | weather files, column P | reduced to mean sea level | nominally hourly |
| `humidity_pct` | percent | weather files, column U | relative humidity at two metres | nominally hourly |
| `rain_category` | ordinal 0 to 3 | weather files, column nRAIN | 0 normal, 1 light rain, 2 moderate rain, 3 heavy rain | nominally hourly |
| `dewpoint_c` | degrees Celsius | weather files, column Td | dewpoint at two metres | nominally hourly |
| `adjacent` | 0 or 1 | adj.csv | published with a unit diagonal and twenty edges recorded in one direction only; symmetrised by union and the diagonal removed | static |
| `distance_m` | metres | distance.csv | zone centroid to zone centroid | static |

## Observed ranges

Computed from the loaded frame, not copied from the publication.

| Field | Minimum | Median | Mean | Maximum | Share exactly zero |
| --- | --- | --- | --- | --- | --- |
| `occupancy` | 0 | 11 | 17.8665 | 373 | 2.70% |
| `volume` | 0 | 53.6667 | 260.885 | 16732.5 | 5.47% |
| `volume_11kw` | 0 | 49 | 107.69 | 5073 | 5.46% |
| `duration` | 0 | 7 | 13.1056 | 207.583 | 5.47% |
| `e_price` | 0.2381 | 0.96 | 0.9512 | 1.8 | 0.00% |
| `s_price` | 0 | 0.76 | 0.7241 | 1.45 | 0.74% |

## Units that are easy to misread

`occupancy` is a count of charging points in use, not a percentage. It is integer valued in 99.94 percent of zone hours and never exceeds the charging points installed in the zone in any of the 1,194,600 zone hours observed. Utilisation, the bounded rate the capacity work uses, is derived by dividing it by `capacity_points`.

`duration` is point hours accumulated inside the hour, so a zone with ten points cannot record more than ten. `volume` is energy in kWh and is modelled from power and duration rather than metered.

Prices are quoted per kWh and are separate: `e_price` is the electricity tariff and `s_price` is the operator service fee. The price a driver pays is their sum.
