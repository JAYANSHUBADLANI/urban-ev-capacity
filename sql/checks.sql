-- Data quality assertions expressed as SQL.
-- Each branch returns one row: a check identifier, the failing count, the
-- denominator and a status. The suite is run by the pipeline and the result is
-- compared against the pandas implementation, so the two are cross checked
-- against each other rather than either being taken on trust.

SELECT 'sql_row_count' AS check_id,
       ABS(COUNT(*) - 1194600) AS n_failing,
       1194600 AS n_total,
       CASE WHEN COUNT(*) = 1194600 THEN 'pass' ELSE 'fail' END AS status
FROM observations

UNION ALL
SELECT 'sql_distinct_hours',
       ABS(COUNT(DISTINCT ts) - 4344), 4344,
       CASE WHEN COUNT(DISTINCT ts) = 4344 THEN 'pass' ELSE 'fail' END
FROM observations

UNION ALL
SELECT 'sql_distinct_zones',
       ABS(COUNT(DISTINCT zone_id) - 275), 275,
       CASE WHEN COUNT(DISTINCT zone_id) = 275 THEN 'pass' ELSE 'fail' END
FROM observations

UNION ALL
SELECT 'sql_duplicate_keys',
       COALESCE(SUM(n - 1), 0), (SELECT COUNT(*) FROM observations),
       CASE WHEN COALESCE(SUM(n - 1), 0) = 0 THEN 'pass' ELSE 'fail' END
FROM (SELECT zone_id, ts, COUNT(*) AS n
      FROM observations GROUP BY zone_id, ts HAVING COUNT(*) > 1) d

UNION ALL
SELECT 'sql_ragged_zone_coverage',
       COUNT(*), 275,
       CASE WHEN COUNT(*) = 0 THEN 'pass' ELSE 'fail' END
FROM (SELECT zone_id FROM observations GROUP BY zone_id HAVING COUNT(*) <> 4344) r

UNION ALL
SELECT 'sql_negative_values',
       SUM(CASE WHEN occupancy < 0 OR volume < 0 OR volume_11kw < 0
                  OR duration < 0 OR e_price < 0 OR s_price < 0
                THEN 1 ELSE 0 END),
       COUNT(*),
       CASE WHEN SUM(CASE WHEN occupancy < 0 OR volume < 0 OR volume_11kw < 0
                            OR duration < 0 OR e_price < 0 OR s_price < 0
                          THEN 1 ELSE 0 END) = 0 THEN 'pass' ELSE 'fail' END
FROM observations

UNION ALL
SELECT 'sql_utilisation_above_one',
       SUM(CASE WHEN utilisation > 1.000001 THEN 1 ELSE 0 END),
       COUNT(*),
       CASE WHEN SUM(CASE WHEN utilisation > 1.000001 THEN 1 ELSE 0 END) = 0
            THEN 'pass' ELSE 'fail' END
FROM observations

UNION ALL
SELECT 'sql_duration_above_capacity',
       SUM(CASE WHEN duration > capacity_points + 0.000001 THEN 1 ELSE 0 END),
       COUNT(*),
       CASE WHEN SUM(CASE WHEN duration > capacity_points + 0.000001 THEN 1 ELSE 0 END) = 0
            THEN 'pass' ELSE 'fail' END
FROM observations

UNION ALL
SELECT 'sql_occupancy_without_volume',
       SUM(CASE WHEN occupancy > 0 AND volume <= 0 THEN 1 ELSE 0 END),
       COUNT(*),
       'info'
FROM observations

UNION ALL
SELECT 'sql_volume_without_occupancy',
       SUM(CASE WHEN occupancy <= 0 AND volume > 0 THEN 1 ELSE 0 END),
       COUNT(*),
       'info'
FROM observations

UNION ALL
SELECT 'sql_saturated_zone_hours',
       SUM(CASE WHEN utilisation >= 0.999 THEN 1 ELSE 0 END),
       COUNT(*),
       'info'
FROM observations

UNION ALL
SELECT 'sql_zones_missing_attributes',
       COUNT(*), 275,
       CASE WHEN COUNT(*) = 0 THEN 'pass' ELSE 'fail' END
FROM (SELECT DISTINCT o.zone_id FROM observations o
      LEFT JOIN zones z ON z.zone_id = o.zone_id
      WHERE z.zone_id IS NULL) m

UNION ALL
SELECT 'sql_edges_are_reciprocal',
       (SELECT COUNT(*) FROM zone_edges a
        WHERE NOT EXISTS (SELECT 1 FROM zone_edges b
                          WHERE b.zone_id = a.neighbour_id
                            AND b.neighbour_id = a.zone_id)),
       (SELECT COUNT(*) FROM zone_edges),
       CASE WHEN (SELECT COUNT(*) FROM zone_edges a
                  WHERE NOT EXISTS (SELECT 1 FROM zone_edges b
                                    WHERE b.zone_id = a.neighbour_id
                                      AND b.neighbour_id = a.zone_id)) = 0
            THEN 'pass' ELSE 'fail' END

UNION ALL
SELECT 'sql_weather_hours',
       ABS((SELECT COUNT(DISTINCT ts) FROM weather) - 4344), 4344,
       CASE WHEN (SELECT COUNT(DISTINCT ts) FROM weather) = 4344
            THEN 'pass' ELSE 'fail' END

ORDER BY check_id;
