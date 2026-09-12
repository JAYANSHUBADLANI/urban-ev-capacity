-- Physical schema for the reporting layer.
-- Calendar parts are materialised as columns so that the same SQL text runs
-- unchanged on sqlite3 and on duckdb, whose date formatting functions differ.

DROP TABLE IF EXISTS observations;
CREATE TABLE observations (
    zone_id         INTEGER NOT NULL,
    ts              TEXT    NOT NULL,
    obs_date        TEXT    NOT NULL,
    hour_of_day     INTEGER NOT NULL,
    day_of_week     INTEGER NOT NULL,   -- 0 is Monday
    is_weekend      INTEGER NOT NULL,
    is_holiday      INTEGER NOT NULL,
    occupancy       REAL    NOT NULL,   -- charging points in use
    volume          REAL    NOT NULL,   -- kWh, rated power estimate
    volume_11kw     REAL    NOT NULL,   -- kWh, 11 kW vehicle side estimate
    duration        REAL    NOT NULL,   -- point hours of charging
    e_price         REAL    NOT NULL,   -- Yuan per kWh
    s_price         REAL    NOT NULL,   -- Yuan per kWh
    capacity_points INTEGER NOT NULL,
    utilisation     REAL    NOT NULL    -- occupancy divided by capacity_points
);

DROP TABLE IF EXISTS zones;
CREATE TABLE zones (
    zone_id                 INTEGER NOT NULL,
    n_stations              INTEGER NOT NULL,
    capacity_points         INTEGER NOT NULL,
    mean_points_per_station REAL    NOT NULL,
    max_points_per_station  INTEGER NOT NULL,
    centroid_lon            REAL    NOT NULL,
    centroid_lat            REAL    NOT NULL,
    area_m2                 REAL    NOT NULL,
    perimeter_m             REAL    NOT NULL,
    station_density         REAL    NOT NULL,
    points_density          REAL    NOT NULL,
    shape_index             REAL    NOT NULL,
    n_neighbours            INTEGER NOT NULL
);

DROP TABLE IF EXISTS weather;
CREATE TABLE weather (
    ts                  TEXT NOT NULL,
    air_temp_c          REAL,
    humidity_pct        REAL,
    rain_category       REAL,
    dewpoint_c          REAL,
    station             TEXT NOT NULL
);

DROP TABLE IF EXISTS zone_edges;
CREATE TABLE zone_edges (
    zone_id     INTEGER NOT NULL,
    neighbour_id INTEGER NOT NULL,
    distance_m  REAL NOT NULL
);

CREATE INDEX idx_obs_zone ON observations (zone_id);
CREATE INDEX idx_obs_ts ON observations (ts);
CREATE INDEX idx_obs_zone_ts ON observations (zone_id, ts);
CREATE INDEX idx_edges_zone ON zone_edges (zone_id);
