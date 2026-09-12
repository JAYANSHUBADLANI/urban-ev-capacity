-- Daily demand profile per zone.
-- One row per zone and calendar day, with the quantities the capacity
-- discussion refers to: energy served, mean and peak utilisation, and the
-- number of hours at the ceiling.
SELECT
    zone_id,
    obs_date,
    day_of_week,
    is_weekend,
    is_holiday,
    COUNT(*)                                        AS n_hours,
    SUM(volume)                                     AS volume_kwh,
    SUM(volume_11kw)                                AS volume_11kw_kwh,
    SUM(duration)                                   AS point_hours,
    AVG(occupancy)                                  AS mean_occupancy,
    MAX(occupancy)                                  AS peak_occupancy,
    AVG(utilisation)                                AS mean_utilisation,
    MAX(utilisation)                                AS peak_utilisation,
    SUM(CASE WHEN utilisation >= 0.999 THEN 1 ELSE 0 END) AS hours_at_ceiling,
    AVG(e_price + s_price)                          AS mean_total_price
FROM observations
GROUP BY zone_id, obs_date, day_of_week, is_weekend, is_holiday
ORDER BY zone_id, obs_date;
