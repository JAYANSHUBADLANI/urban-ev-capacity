-- Zone ranking on the measures a capacity decision would reach for by default.
-- Utilisation ranking here is the baseline that the marginal served energy
-- ranking is later compared against.
SELECT
    o.zone_id,
    z.capacity_points,
    z.n_stations,
    z.n_neighbours,
    COUNT(*)                                              AS n_hours,
    SUM(o.volume)                                         AS total_volume_kwh,
    SUM(o.volume_11kw)                                    AS total_volume_11kw_kwh,
    SUM(o.duration)                                       AS total_point_hours,
    AVG(o.utilisation)                                    AS mean_utilisation,
    MAX(o.utilisation)                                    AS peak_utilisation,
    SUM(CASE WHEN o.utilisation >= 0.999 THEN 1 ELSE 0 END) AS hours_at_ceiling,
    1.0 * SUM(CASE WHEN o.utilisation >= 0.999 THEN 1 ELSE 0 END) / COUNT(*) AS share_at_ceiling,
    SUM(CASE WHEN o.utilisation >= 0.90 THEN 1 ELSE 0 END)  AS hours_above_90pct,
    SUM(o.volume) / z.capacity_points                     AS kwh_per_point,
    AVG(o.e_price)                                        AS mean_e_price,
    AVG(o.s_price)                                        AS mean_s_price,
    AVG(o.e_price + o.s_price)                            AS mean_total_price
FROM observations o
JOIN zones z ON z.zone_id = o.zone_id
GROUP BY o.zone_id, z.capacity_points, z.n_stations, z.n_neighbours
ORDER BY mean_utilisation DESC;
