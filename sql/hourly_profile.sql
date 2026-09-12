-- Average shape of the day per zone, split by weekday and weekend.
-- This is the table the load archetypes are read from.
SELECT
    zone_id,
    hour_of_day,
    is_weekend,
    COUNT(*)            AS n_obs,
    AVG(occupancy)      AS mean_occupancy,
    AVG(utilisation)    AS mean_utilisation,
    AVG(volume)         AS mean_volume_kwh,
    AVG(duration)       AS mean_point_hours,
    MAX(utilisation)    AS max_utilisation
FROM observations
GROUP BY zone_id, hour_of_day, is_weekend
ORDER BY zone_id, is_weekend, hour_of_day;
