-- Peak window extraction.
-- For every zone the four consecutive hours of the day with the highest mean
-- utilisation, taken on weekdays. The window wraps across midnight, which is
-- why the hour arithmetic uses a modulo rather than a range.
WITH shape AS (
    SELECT zone_id, hour_of_day, AVG(utilisation) AS mean_utilisation
    FROM observations
    WHERE is_weekend = 0
    GROUP BY zone_id, hour_of_day
),
windows AS (
    SELECT
        s.zone_id,
        s.hour_of_day AS window_start,
        AVG(t.mean_utilisation) AS window_utilisation,
        COUNT(*) AS window_hours
    FROM shape s
    JOIN shape t
      ON t.zone_id = s.zone_id
     AND ((t.hour_of_day - s.hour_of_day + 24) % 24) < 4
    GROUP BY s.zone_id, s.hour_of_day
),
best AS (
    SELECT zone_id, MAX(window_utilisation) AS best_utilisation
    FROM windows
    WHERE window_hours = 4
    GROUP BY zone_id
)
SELECT
    w.zone_id,
    w.window_start,
    (w.window_start + 3) % 24 AS window_end,
    w.window_utilisation
FROM windows w
JOIN best b
  ON b.zone_id = w.zone_id
 AND b.best_utilisation = w.window_utilisation
WHERE w.window_hours = 4
GROUP BY w.zone_id, w.window_start, w.window_utilisation
ORDER BY w.window_utilisation DESC;
