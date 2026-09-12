# Method

How the study is built, in the order the pipeline runs it. Every number quoted
anywhere in this repository is produced by code in it; nothing is carried over
from the publication the data comes from.

## 1. Inputs

I fetch thirteen files from one pinned commit of the UrbanEV repository and
hash them on arrival. I write the digests to `data/manifest.json`, and a later
run that sees a different digest stops rather than continuing. Re-fetching into
an empty tree and checking against a manifest carried in a checkpoint is a real
reproducibility check, not a formality.

Coverage is 2022-09-01 00:00 to 2023-02-28 23:00: 4344 hourly periods, 275
zones, 1,194,600 zone hours, covering 1362 stations and 17,532 charging points.

## 2. Reshape and schema

The five demand and price tables are published wide, with a time column and one
column per zone. I reshape them once into a long frame keyed by zone and
timestamp, which every later phase reads.

Two timestamp formats appear in the dataset and I parse both with an explicit
format string. The demand and price tables use `2022-09-01 00:00:00`; the two
weather tables use `2022/9/1 0:00`. I never use automatic inference, because a
silent fallback on an ambiguous day and month order would corrupt every
calendar feature without raising anything.

Zone attributes come from `inf.csv`, which is station level. I compute capacity
as the sum of `charge_count` over the stations mapped to a zone through
`TAZID`, so it is an aggregation and not a lookup.

## 3. What the fields actually are

`occupancy` is a count of charging points in use, not a percentage. It is
integer valued in 99.94 percent of zone hours, reaches 373 at its maximum, and
never exceeds the points installed in its zone in any of the 1,194,600 zone
hours. Utilisation, the bounded quantity the capacity work uses, is that count
divided by installed points.

`volume` and `volume-11kW` are two estimates of the same energy, not two
different quantities. Dividing each by `duration` recovers the implied power
per point hour. The first returns the rated power of the points. The second is
close to that figure capped at 11 kW for 239 of the 275 zones, but the
description does not hold everywhere: 34 zones carry a higher implied power
under the 11 kW series than under the rated one, and 5 zones exceed 12 kW under
it entirely, reaching 140 kW. Those 5 zones hold 14.4 percent of energy under
the 11 kW estimate. I therefore treat the two series as two competing estimates
whose relationship is mostly but not entirely a power cap, rather than as one
being a clean transformation of the other. Across the estate the rated estimate
totals 2.423 times the other.

## 4. Data quality

I run thirty one checks in pandas and a further fourteen as SQL assertions
against the reporting database. I compute four counts both ways and compare
them; the pipeline stops if they disagree.

I write the checks to characterise the cleaning the publisher already applied
rather than to pretend the series are raw. Structural checks are pass or fail.
Characterisation checks are informational, because a plateau produced by
forward fill is a property of the source rather than a defect introduced here.

Two checks fail and both failures are real: the published adjacency matrix has
a unit diagonal, and twenty of its edges are recorded in one direction only. I
handle both explicitly downstream, by removing the diagonal and taking the
union with the transpose, and I report both rather than repair them in silence.

## 5. SQL layer

I keep the aggregations that feed reporting as real `.sql` files under `sql/`,
executed by the pipeline. The default engine is `sqlite3` from the standard
library so the layer always runs; the pipeline uses `duckdb` instead when it is
already installed, on the same SQL text. I materialise calendar parts as
columns in the schema precisely so that one text runs on both, since the two
engines disagree about the argument order of `strftime`.

## 6. Structural breaks

I locate change points from the data. Two independent detectors run on the day
of week adjusted daily aggregates: pruned dynamic programming with a Bayesian
information criterion penalty, and binary segmentation. I report their
agreement.

I test significance locally. The window around each break is bounded by its
neighbouring breaks, I fit the null of no break inside that window only, and I
resample the residuals from that null in blocks of fourteen days to preserve
short range serial correlation. Resampling the observed values instead would
carry the level shift into the null distribution and make every break look
unremarkable.

The longest segment of the daily utilisation series with no detected change
point runs from 2022-11-08 to 2023-01-17. I use that segment as the calibration
period for every monitor, and measure detection delay against the change point
that immediately follows it, 2023-01-18. I chose neither date.

## 7. Features

For a target at `t + h` the origin is `t`, and no feature may read any series
later than `t`. I build forty three features: lags of the target at offsets 0,
1, 2, 23, 24, 47 and 167 hours; trailing rolling means, standard deviations,
maxima and minima; short differences; distance weighted neighbour aggregates;
prices; weather; calendar terms; and static zone attributes.

I take weather and prices at the origin, not at the target time. The dataset
carries observed weather rather than a forecast, so using the value at the
target time would hand the model information that would not exist when the
forecast is made.

The calendar is the one exception, and I take it at the target time, because it
is known arbitrarily far ahead.

The leakage guard replaces the whole target series from one hour past the
latest origin onwards and rebuilds the feature set through the ordinary code
path. Any feature that reads even one hour beyond its origin changes, and the
test fails. I also wrote a companion test that plants a deliberately future
reading feature and confirms the same comparison catches it, so I know the
guard has teeth.

## 8. Validation

I use rolling origin cross validation with eight folds. Each fold holds out 336
consecutive hours, origins advance by 336 hours, and the training window grows.
I shuffle nothing.

A training row is admitted only if its target falls inside the training window,
so no fitted parameter can have seen its own test period. A test row may use
observations after the training window, because at prediction time for a target
at `t + h` the series really is known up to `t`. What it may never use is
anything after `t`.

Hyperparameter selection respects time as well. I choose the ridge penalty on
an inner split that holds out the last 336 hours of the training window. I
choose the gradient boosting setting the same way on the first fold and then
hold it fixed, which keeps the refit count manageable and still never lets the
selection see a test window.

## 9. Models

Eight families across three groups.

Three baselines need no fitting: last observation carried forward, seasonal
naive at a daily period, and seasonal naive at a weekly period.

One per zone direct autoregression of order 24, fitted by least squares. It is
direct rather than recursive, so a separate set of coefficients is fitted for
each horizon and multi step errors do not compound through repeated
substitution. I implemented it here rather than taking it from a library so the
project does not depend on one being installed.

Four learned families: ridge and gradient boosting, each fitted once across all
zones with zone identity and zone attributes among the features, and each
fitted separately per zone without the static attributes, which are constant
inside a zone and carry nothing there.

Zone identity for the pooled models is the zone's mean and standard deviation
of the target computed on the training window only, alongside the static
attributes. A statistic computed over the whole series would leak.

I clip predictions to the physically feasible range before scoring: never below
zero, and never above installed points for occupancy and duration. The clip
uses only information known in advance, and I apply it identically to every
family.

## 10. Metrics

MAE, RMSE and MAPE are reported, with MAPE computed only where the actual is
non zero and its coverage reported alongside.

MASE is the scaled error used for pooling across zones. The denominator is the
mean absolute seasonal naive error at a daily period computed on the training
part of the fold; computing it on the test part would leak.

A saturation weighted absolute error weights each row by `1 + 3u`, where `u` is
observed utilisation, so an error at a full zone counts four times as much as
the same error at an empty one. A symmetric metric treats those two errors as
equally costly, and for a capacity decision they are not.

I also break out every headline number by zone volume decile, because a pooled
mean is dominated by the largest zones.

## 11. Monitoring

One production model is trained on the history preceding the calibration period
and then scores forward across the rest of the timeline, which is what a
deployed model does between refits.

Four monitors run hourly. Two watch the distribution of the model inputs,
comparing a trailing 168 hour window against the training tail through a fixed
binning: a population stability index averaged across features, and a binned
Kolmogorov-Smirnov statistic taken at its maximum across features. Two watch
the forecast residuals: a windowed CUSUM on the mean residual and an
exponentially weighted moving average of the mean absolute residual.

All four are windowed. A cumulative statistic that never resets drifts upwards
on its own, which would make a late detection look like an early one.

I calibrate thresholds so that every monitor produces the same false alarm rate
on the calibration period, at four target rates. A monitor can always be made
to alert sooner by alerting more often, so comparing detection delay at
anything other than a matched false alarm rate compares nothing.

## 12. Retraining

I backtest five policies over the scored timeline: never retrain, retrain every
336 hours, retrain every 672 hours, retrain when the residual CUSUM alerts, and
retrain when the input drift monitor alerts. Triggered policies use the
threshold calibrated at two alerts per thousand hours and observe a one week
cooldown. I score each policy on mean absolute error, before and after the
break, and on the number of refits it spends.

## 13. Segmentation and anomalies

I cluster zones on the shape of their day rather than its level: mean
utilisation by hour of day, separately for weekdays and weekends, scaled to
unit mean. I filter the cluster count by stability across random starts and by
balance, then choose it on silhouette width, and report the Calinski Harabasz
ratio alongside. I added the balance filter because silhouette width alone
selects two clusters, which puts 87 percent of zones in one group and is not a
segmentation a planner can act on.

I find anomalous zone hours two ways: a robust score against the zone's own
median for that hour of the week, and an isolation forest over a small feature
vector. There are no labels, so I report no precision or recall. What I report
is how much the two methods overlap.

## 14. Capacity

I treat each zone as a loss system with as many servers as it has charging
points. I recover offered load by inverting the Erlang loss formula on the
observed occupancy. A zone hour recorded as completely full implies unbounded
offered load, so I cap recovery at the level where half of arrivals would be
turned away; without that cap the 0.7 percent of zone hours at the ceiling
carry 94 percent of the total recovered load and the calibration is
meaningless. Every unserved demand figure is therefore a lower bound.

Mean session length is an assumption, not a measurement: the dataset records
hourly aggregates and never individual sessions. I set it from the power of the
zone's points: one hour where points are rated at 22 kW or more, two hours
between 11 and 22 kW, and three and a half hours below that, and vary the
whole assumption by half in each direction in the sensitivity analysis.

I compute the marginal value of one more point two ways. The analytic route
evaluates the loss formula at one more server. The simulated route replays
arrivals and departures across seeds with neighbour substitution, drawing
arrivals once per hour and sharing them across capacity scenarios, so two
scenarios differing by a single point see exactly the same demand and their
difference is the effect of that point rather than sampling noise. I report
agreement between the two routes.

Blocked demand is partly offered to adjacent zones with a distance decayed
weight, so the value of capacity at one zone depends on the state of its
neighbours. I measure marginal value system wide, because a point added where
demand is blocked also removes the spillover that zone was pushing outwards.
