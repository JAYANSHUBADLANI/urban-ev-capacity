# Results

Every figure below is produced by code in this repository. Sample sizes and intervals are stated next to the numbers they belong to.

## What the data covers

- 1,194,600 zone hours: 275 zones by 4344 hourly periods, 2022-09-01 00:00 to 2023-02-28 23:00.
- 1,362 stations and 17,532 charging points.
- Mean utilisation 0.271 of installed points. 8,816 zone hours (0.738 percent) have every point in use, across 104 zones.
- Total energy over the window: 3.117e+08 kWh under the rated power estimate and 1.286e+08 kWh under the 11 kW estimate, a ratio of 2.423.

## Structural breaks, located from the data

Two detectors were run independently on the day of week adjusted daily series. Their agreement, and the significance of each break under a local block bootstrap, are in `results/structural_break_agreement.csv` and `results/structural_break_significance.csv`.

| Series | Break | Window (days) | Relative shift | p value |
| --- | --- | --- | --- | --- |
| utilisation | 2022-10-25 | 61 | +8.4% | 0.0519 |
| utilisation | 2022-11-01 | 14 | -20.1% | n/a |
| utilisation | 2022-11-08 | 78 | +14.6% | 0.0240 |
| utilisation | 2023-01-18 | 89 | -15.0% | 0.0359 |
| utilisation | 2023-02-05 | 42 | +19.2% | 0.0040 |
| volume | 2022-09-08 | 31 | +11.7% | 0.0639 |
| volume | 2022-10-02 | 48 | -9.3% | 0.0599 |
| volume | 2022-10-26 | 32 | -11.4% | 0.0100 |
| volume | 2022-11-03 | 45 | +14.3% | 0.0120 |
| volume | 2022-12-10 | 47 | +8.7% | 0.0040 |
| volume | 2022-12-20 | 34 | -11.3% | 0.0479 |
| volume | 2023-01-13 | 31 | -21.0% | 0.0040 |
| volume | 2023-01-20 | 16 | -31.4% | 0.0020 |
| volume | 2023-01-29 | 20 | +56.0% | 0.0020 |
| volume | 2023-02-09 | 31 | +14.6% | 0.0040 |
| duration | 2022-09-10 | 31 | +11.9% | 0.0818 |
| duration | 2022-10-02 | 46 | -6.2% | 0.0040 |
| duration | 2022-10-26 | 32 | -42.0% | 0.0020 |
| duration | 2022-11-03 | 57 | +76.8% | 0.0080 |
| duration | 2022-12-22 | 78 | -7.1% | 0.0479 |
| duration | 2023-01-20 | 36 | -33.6% | 0.0020 |
| duration | 2023-01-27 | 20 | +30.6% | 0.0020 |
| duration | 2023-02-09 | 33 | +11.4% | 0.0020 |

The longest segment of the daily utilisation series with no detected break runs from 2022-11-08 00:00 to 2023-01-17 23:00. That segment calibrates every monitor, and the break immediately after it, 2023-01-18 00:00, is what detection delay is measured against.

## Claim 1: does a pooled model beat per zone models

**Verdict: refuted.**

The pooled model wins 6 of 9 target and horizon combinations on pooled MASE. The Spearman correlation between zone volume decile and the pooled model's advantage is +0.3343 across 275 zones; the claim required it to be negative, meaning the advantage grows as zones get quieter.

Mean advantage in the quietest decile: -0.1780 MASE. In the busiest decile: +0.0116 MASE. A positive number favours pooling.

| Target | Horizon | Best pooled | MASE | Best per zone | MASE | Advantage |
| --- | --- | --- | --- | --- | --- | --- |
| duration | 1 h | global_gbm | 0.2932 | zone_gbm | 0.3022 | +0.0090 |
| duration | 6 h | global_gbm | 0.7304 | zone_gbm | 0.7614 | +0.0310 |
| duration | 24 h | global_gbm | 0.7915 | zone_ridge | 0.8600 | +0.0685 |
| occupancy | 1 h | global_gbm | 0.3842 | zone_ridge | 0.4134 | +0.0292 |
| occupancy | 6 h | global_gbm | 0.7603 | zone_ridge | 0.8052 | +0.0450 |
| occupancy | 24 h | global_gbm | 0.8382 | zone_ridge | 0.8918 | +0.0536 |
| volume | 1 h | global_gbm | 0.3706 | zone_gbm | 0.3189 | -0.0517 |
| volume | 6 h | global_gbm | 0.8733 | zone_gbm | 0.7778 | -0.0954 |
| volume | 24 h | global_gbm | 0.9326 | zone_ridge | 0.8625 | -0.0701 |

## Claim 2: does input drift show up later than residual error

**Verdict: refuted.**

As preregistered, the earliest input monitor alert comes 143 hours after the break, and the best residual monitor alerts never, at any of the matched rates. The input monitor is later at 0 of 4 matched false alarm rates, and the claim required a majority.

| Variant | Horizon | False alarms per 1000 h | Best input | Input delay | Best residual | Residual delay |
| --- | --- | --- | --- | --- | --- | --- |
| preregistered | 1 h | 0.5 | input_psi | 143 h | none | not detected |
| preregistered | 1 h | 1.0 | input_psi | 143 h | none | not detected |
| preregistered | 1 h | 2.0 | input_psi | 143 h | none | not detected |
| preregistered | 1 h | 5.0 | input_psi | 143 h | none | not detected |
| stable_window_excludes_december_shock | 1 h | 0.5 | input_psi | 143 h | none | not detected |
| stable_window_excludes_december_shock | 1 h | 1.0 | input_psi | 143 h | none | not detected |
| stable_window_excludes_december_shock | 1 h | 2.0 | input_psi | 143 h | residual_abs_ewma | 48 h |
| stable_window_excludes_december_shock | 1 h | 5.0 | input_psi | 143 h | residual_abs_ewma | 45 h |
| horizon_24 | 24 h | 0.5 | input_psi | 143 h | residual_mean_cusum | 329 h |
| horizon_24 | 24 h | 1.0 | input_psi | 143 h | residual_mean_cusum | 329 h |
| horizon_24 | 24 h | 2.0 | input_psi | 143 h | residual_mean_cusum | 329 h |
| horizon_24 | 24 h | 5.0 | input_psi | 143 h | residual_mean_cusum | 329 h |

The two variants below the preregistered rows are sensitivities, not the registered test. The calibration window was chosen as the longest stretch with no detected change point in the utilisation series, and it turns out to contain a one day shock on 2022-12-09 that both residual monitors react to strongly. That shock raises their calibrated thresholds above anything the January break produces. Excluding the week around it from calibration reverses the answer at the two looser rates, which is reported here rather than left out.

### What retraining recovers

| Policy | Refits | MAE | MAE before the break | MAE after the break |
| --- | --- | --- | --- | --- |
| periodic_336h | 8 | 1.6965 | 1.8279 | 1.4744 |
| periodic_672h | 4 | 1.6975 | 1.8233 | 1.4847 |
| triggered_input_psi | 5 | 1.6992 | 1.8331 | 1.4729 |
| no_retraining | 0 | 1.7222 | 1.8384 | 1.5257 |
| triggered_residual_mean_cusum | 0 | 1.7222 | 1.8384 | 1.5257 |

Scored over 2712 hours and 275 zones at a one hour horizon, in charging points.

## Claim 3: does marginal value rank zones differently from utilisation

**Verdict: supported.**

The Spearman correlation between mean observed utilisation and simulated marginal served energy is 0.5765 across 275 zones and 20 seeds; the claim required it to be at or below 0.9. Only 11 of the top twenty zones are shared between the two rankings.

The marginal energy rule serves more than the utilisation rule at 5 of 5 budgets, by 268,708 kWh on average over the peak fortnight.

152 of 275 zones gain nothing at all from an added point over the simulated fortnight, because they never fill. That is the substance behind the ranking disagreement: utilisation orders every zone, while marginal value is zero for most of them.

| Budget (points) | No investment (kWh) | By utilisation (kWh) | By marginal energy (kWh) | Difference |
| --- | --- | --- | --- | --- |
| 50 | 36,269,400 | 36,307,400 | 36,512,600 | +205,160 |
| 100 | 36,260,400 | 36,305,100 | 36,578,800 | +273,685 |
| 200 | 36,277,200 | 36,343,200 | 36,654,800 | +311,545 |
| 400 | 36,252,200 | 36,377,700 | 36,672,500 | +294,838 |
| 800 | 36,276,200 | 36,486,200 | 36,744,500 | +258,313 |

Energy uses the rated power estimate over the busiest 336 hours, 20 seeds per scenario. The 11 kW estimate is reported alongside in `results/capacity_scenarios.csv` and moves the level, not the ordering of the rules.

### Sensitivity to the session length assumption

| Session length scale | Median assumed session (h) | Spearman with utilisation | Rankings disagree |
| --- | --- | --- | --- |
| 0.5 | 1.75 | 0.6666 | yes |
| 1.0 | 3.50 | 0.6666 | yes |
| 1.5 | 5.25 | 0.6666 | yes |

## Segmentation

| Segment | Zones | Weekday peak hour | Peak to mean | Mean utilisation | Archetype |
| --- | --- | --- | --- | --- | --- |
| 0 | 150 | 1 | 1.31 | 0.2890 | overnight charging, residential rhythm |
| 1 | 12 | 11 | 1.82 | 0.1808 | morning peak, commuter arrival rhythm |
| 2 | 113 | 13 | 1.08 | 0.2566 | flat load, close to always on |

## Anomalies

The robust score flags 6,431 zone hours and the isolation forest flags 5,973, out of 1,194,600. They agree on 101, a Jaccard index of 0.0082. There are no labels, so no precision or recall is reported. Two methods of different type flagging almost disjoint sets is the honest finding: neither should be treated as an oracle, and a reviewer reading the sample in `results/anomaly_sample.csv` will see they are detecting different things.

## Headline tables and determinism

- `results/headline_model_table.csv`: 72 rows, SHA-256 `b177212ebcb1ac21ce1675de04697c3c`
- `results/headline_capacity_table.csv`: 40 rows, SHA-256 `6ecae845a15abbb97d64060b4804b685`

Phase 12 regenerates both back to back and compares the digests.
