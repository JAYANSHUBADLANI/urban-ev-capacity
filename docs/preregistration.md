# Preregistration

Written after the exploratory analysis and the structural break detection, and
before any model in the forecasting grid was fitted. The purpose of fixing the
claims here is that the analysis cannot be quietly reshaped once the numbers
arrive. Each claim states the quantity, the comparison, and the condition under
which the claim is refuted. All three outcomes are reported in
`docs/results.md` and in the README, whichever way they fall.

## Context fixed before the grid ran

The following were established in phases 2 to 4 and are treated as given:

- Coverage is 2022-09-01 00:00 to 2023-02-28 23:00, 4344 hourly periods,
  275 zones, 1,194,600 zone hours.
- `occupancy` is a count of charging points in use, bounded above by the
  charging points installed in the zone. Utilisation is that count divided by
  installed points and is bounded to the unit interval.
- Change points in aggregate daily demand were located by pruned dynamic
  programming and by binary segmentation independently. The longest segment of
  the daily utilisation series containing no detected change point runs from
  2022-11-08 to 2023-01-17, which is 71 days or 1704 hours. The change point
  immediately following it falls on 2023-01-18.
- That stable segment is the calibration period for every monitor, and
  2023-01-18 00:00 is the break that detection delay is measured against. These
  are written to `results/monitoring_window.csv` and are not revised later.

## Metrics fixed before the grid ran

- **MAE**, **RMSE** and **MAPE** are reported. MAPE is undefined when the
  actual value is zero, which happens in 2.7 percent of zone hours for
  occupancy, so MAPE is computed only over rows with a non zero actual and its
  coverage is reported alongside it.
- **MASE** is the scaled error used for pooling across zones. The scaling
  denominator is the mean absolute error of the seasonal naive forecast at a
  daily period, computed on the training part of the fold only. A denominator
  computed on the test part would leak.
- **Saturation weighted absolute error** is the saturation aware metric. Errors
  are weighted by `1 + 3 * u` where `u` is the observed utilisation in that zone
  hour, so a given absolute error counts four times as much at a full zone as at
  an empty one. This is reported because a symmetric error metric understates
  the cost of being wrong exactly where capacity is binding.
- All headline comparisons are made on pooled MASE unless a claim says
  otherwise. Per zone error distributions and error by zone volume decile are
  reported for every headline number, because a pooled mean is dominated by the
  high volume zones.

## Claim 1

**Statement.** A single global model trained across all zones, with zone
identity and zone attributes as features, beats per zone models on held out
error, and the advantage is larger for zones with lower average volume.

**How it is tested.** Rolling origin cross validation with 8 folds, each fold
holding out 336 consecutive hours, origins advancing by 336 hours, and the
training set always strictly earlier than the test set. Targets are occupancy,
volume and duration; horizons are 1, 6 and 24 hours. The global families are
`global_ridge` and `global_gbm`, each fitted once per fold across all 275 zones
with zone identity and zone attributes among the features. The per zone
families are `zone_ridge` and `zone_gbm`, each fitted separately for every zone
in every fold.

The primary comparison is pooled MASE over all zones and folds, best global
family against best per zone family, reported for each of the nine target and
horizon combinations.

The secondary comparison is the advantage by zone volume decile, where the
advantage for a zone is per zone MASE of the best per zone family minus per
zone MASE of the best global family, so that a positive number favours the
global model. The trend is measured as the Spearman correlation between volume
decile and advantage across the 275 zones, and separately as the difference in
mean advantage between decile 1 and decile 10.

**Refutation condition.** The claim fails if the best per zone family beats the
best global family on pooled MASE in the majority of the nine target and
horizon combinations, or if the advantage does not increase as zone volume
decreases, meaning the Spearman correlation between volume decile and advantage
is not negative.

## Claim 2

**Statement.** A monitor watching input feature distributions detects the demand
regime change later than a monitor watching forecast residuals, measured as
hours from the true break to the first alert, at a false alarm rate calibrated
on a stable period.

**How it is tested.** Four monitors are run over the full timeline at hourly
resolution: two watching model inputs, `input_psi` and `input_ks`, and two
watching forecast residuals, `residual_mean_cusum` and `residual_abs_ewma`.
Each monitor produces a scalar statistic per hour from a trailing window of 168
hours compared against a reference window of 672 hours.

Thresholds are calibrated so that each monitor produces the same false alarm
rate on the stable period of 2022-11-08 to 2023-01-17, at target rates of 0.5,
1.0, 2.0 and 5.0 false alarms per 1000 hours. Comparison between monitors is
only made at a matched target rate, since a monitor can always be made to alert
sooner by alerting more often.

Detection delay is the number of hours from 2023-01-18 00:00 to the first hour
at which the monitor statistic exceeds its calibrated threshold. A monitor that
never exceeds its threshold before the end of the series is recorded as not
detecting, and is treated as later than any monitor that does detect.

**Refutation condition.** The claim fails if, at the majority of the four
matched false alarm rates, the best input monitor alerts no later than the best
residual monitor.

## Claim 3

**Statement.** Ranking zones by observed utilisation and ranking them by
marginal served energy from an added charging point disagree materially, and
allocating a fixed budget by the second serves more energy.

**How it is tested.** A queueing model is calibrated per zone from observed
occupancy and duration, and is used to estimate served and unserved demand,
because the counterfactual of what would have happened with more capacity is
not observable in the data. Demand blocked at a saturated zone is offered to
adjacent zones with a distance decayed weight, so the value of capacity at one
zone depends on the state of its neighbours.

Marginal served energy for a zone is the increase in served energy from adding
one charging point to that zone, holding every other zone fixed, averaged over
20 seeds.

The ranking comparison is the Spearman correlation between mean observed
utilisation and marginal served energy across the 275 zones.

The allocation comparison distributes a fixed budget of added points under
three rules: proportional to observed utilisation, greedy on marginal served
energy, and a balanced rule that splits the budget evenly between the two.
Budgets of 50, 100, 200, 400 and 800 points are each simulated over 20 seeds,
and the distribution of served energy is reported rather than a single number.

Energy is reported under both published volume estimates, and the headline
figure uses the rated power estimate, with the ratio between the two stated
next to it.

**Refutation condition.** The claim fails if the Spearman correlation between
the two rankings is above 0.9, or if the utilisation allocation serves at least
as much energy as the marginal served energy allocation at the majority of the
five budgets.

## What is not claimed

No causal claim is made about price. Electricity price and service price were
set partly in response to expected demand, so a regression of demand on price
recovers a mixture of the demand response and the pricing rule. Any price
number reported in this project is descriptive.

No claim is made about an individual forecourt. The data is zone level, and a
zone aggregates several stations.

No claim is made about zones outside the published sample. The publisher
excluded low activity zones by a variance test and a zero value filter, so the
275 retained zones are a survivorship filtered sample and the results do not
extend to quiet zones.

## Outcomes

Added after the grid finished. Everything above this heading is as it was written before any model was fitted; the digest of that frozen version is `0606492a2056934df71fe2300f9ffcc7`, recorded in `results/preregistration_digest.txt` at the time it was frozen.

**Claim 1: refuted.**

The pooled model wins 6 of 9 target and horizon combinations, which satisfies the first part. The refutation condition was met on the second part: the Spearman correlation between volume decile and the pooled advantage is +0.3343, which is positive, so the advantage does not increase as zone volume decreases. Mean advantage is -0.1780 MASE in the quietest decile and +0.0116 in the busiest.

**Claim 2: refuted.**

The refutation condition was met: at 4 of 4 matched false alarm rates the input monitor alerts no later than the residual monitor. The input monitor alerted 143 hours after the break and the residual monitor did not alert at any matched rate.

**Claim 3: supported.**

Neither refutation condition was met. The Spearman correlation between the two rankings is 0.5765, at or below the 0.9 threshold, and the marginal energy allocation serves more than the utilisation allocation at 5 of 5 budgets.
