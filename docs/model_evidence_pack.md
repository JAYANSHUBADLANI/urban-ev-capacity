# Model evidence pack

A reviewer should be able to sign the checklist at the end of this document, or
refuse to, using only what is in this repository.

## 1. Intended use

**Decision supported.** Where to add public EV charging capacity across an urban
estate under a fixed budget, and when to retrain the demand model that supports
that decision.

**Users.** Network planning and operations at a fuel retail network operator.

**Scope of the forecast.** Hourly charging demand per traffic analysis zone, at
horizons of 1, 6 and 24 hours, for three targets: occupancy (charging points in
use), volume (energy in kWh) and duration (point hours).

**Out of scope.** Site level decisions, pricing decisions, zones outside the
published sample, and any horizon beyond 24 hours. The model is not a revenue
model: it forecasts demand, and energy is converted from demand using an assumed
power, not a measured one.

## 2. Data lineage

| Stage | What happens | Where |
| --- | --- | --- |
| Source | UrbanEV, public EV charging in Shenzhen, CC0 1.0 | pinned commit, `scripts/fetch_data.py` |
| Integrity | SHA-256 per file, compared on every later run | `data/manifest.json` |
| Reshape | Wide tables to one long frame, explicit timestamp formats | `src/uev/io_load.py` |
| Attributes | Station level `inf.csv` aggregated to zones through `TAZID` | `src/uev/io_load.py` |
| Quality | 31 pandas checks, 14 SQL assertions, 4 cross checked | `src/uev/quality.py`, `sql/checks.sql` |
| Reporting | SQLite database, queries as `.sql` files | `src/uev/sqlio.py`, `sql/` |
| Features | 43 features, all evaluated at or before the origin | `src/uev/features.py` |

Upstream preprocessing applied by the publisher before this project sees the
data: anomaly removal, forward and backward fill of missing values,
interquartile range outlier replacement with adjacent values, and exclusion of
low activity zones by a variance test and a zero value filter.

## 3. Assumptions and their impact

| Assumption | Why it is needed | Impact if wrong |
| --- | --- | --- |
| Vehicles draw the rated power of the point (`volume`) | Energy is not metered | Every energy figure changes by a factor of 2.42 if the 11 kW estimate is right instead. The zone ranking is robust: Spearman 0.946 between the two |
| Mean session length by power tier: 1 h at 22 kW and above, 2 h from 11 to 22 kW, 3.5 h below | Sessions are not recorded | Arrival rates scale inversely. A sensitivity at half and one and a half times the assumption is reported |
| Blocked demand diverts to adjacent zones at 0.35, decaying with distance | Diversion is unobservable | Overstates the benefit of capacity if too high, understates the value of relieving a saturated zone if too low |
| Arrivals are Poisson and service is memoryless | Needed to invert observed occupancy to offered load | Blocking is misestimated if arrivals are bursty; offered load is understated |
| Offered load capped where half of arrivals are blocked | Full occupancy implies unbounded load | Unserved demand and marginal value are lower bounds at saturated zones |
| Adjacency is symmetric and a zone is not its own neighbour | Published matrix has a unit diagonal and 20 one way edges | Neighbour features and spillover would be wrong without the repair |
| Weather is a coarse covariate | Consecutive hours repeat in 52 to 69 percent of the series | Weather features carry less than hourly information; no claim rests on them |

## 4. Validation

**Scheme.** Rolling origin, 8 folds, 336 held out hours per fold, origins
advancing by 336 hours, training window growing from 1656 to 4008 hours. No
random split anywhere, including inside hyperparameter selection.

**Grid.** 3 targets by 3 horizons by 8 folds by 8 model families, 576 cells,
each scored over 275 zones.

**Metrics.** MAE, RMSE, MAPE with coverage, MASE against a training period
seasonal naive denominator, and a saturation weighted absolute error. Results
are reported pooled, per zone, by hour of day, by day of week and by zone volume
decile.

**Leakage controls.** Feature construction is guarded by a test that replaces
the series after the latest origin and rebuilds through the ordinary code path;
a companion test plants a future reading feature and confirms the guard catches
it. Fold boundaries are guarded by tests asserting that no training target falls
inside a test window and that test targets cover the window exactly once. The
MASE denominator and the zone identity features are computed on training data
only.

Detailed results are in `docs/results.md` and the underlying tables in
`results/`.

## 5. Known failure modes

| Failure mode | Evidence | Mitigation |
| --- | --- | --- |
| Regime change in demand | Two episodes inside a six month window, located by two independent detectors | Input drift monitoring, which detects the January break 143 hours in; triggered retraining recovers most of the gap at five refits |
| Residual monitoring is blind at a one hour horizon | Mean absolute residual falls after the break rather than rising, because the model leans on the most recent observation and tracks the level shift | Do not rely on residual monitoring alone at short horizons; the input monitor is the earlier signal here |
| Saturated zones | Errors are asymmetric near the ceiling and 104 of 275 zones reach it | Saturation weighted metric reported alongside the symmetric ones; predictions clipped to installed capacity |
| Quiet zones | Excluded upstream, so unrepresented | Do not apply the model outside the published sample |
| Imputation artefacts | 41.4 percent of occupancy observations sit inside a constant run of four hours or more | Reported, not repaired; error figures should be read as measured against a smoothed series |
| Short history for a new zone | Features need 168 hours of history | The scoring contract refuses the request rather than imputing |
| Malformed input | Schema, range, capacity, continuity and history length are all checked | Every violation is collected and returned together; nothing is silently coerced |

## 6. Monitoring plan

| Monitor | Watches | Window | Calibration |
| --- | --- | --- | --- |
| `input_psi` | Population stability index, averaged over 9 model inputs | 168 h against the training tail | Matched false alarm rate on 2022-11-08 to 2023-01-17 |
| `input_ks` | Binned Kolmogorov-Smirnov, maximum over the same inputs | 168 h against the training tail | as above |
| `residual_mean_cusum` | Windowed CUSUM on the mean residual | 168 h | as above |
| `residual_abs_ewma` | Exponentially weighted mean absolute residual | 168 h | as above |

Thresholds are set at 0.5, 1, 2 and 5 alerts per thousand hours. Comparisons
between monitors are only made at a matched rate.

**Recommended operating point.** Run `input_psi` at two alerts per thousand
hours as the retraining trigger, and keep both residual monitors running as
diagnostics rather than triggers. In this backtest that policy spent five refits
and produced the lowest error after the break of any policy tested.

## 7. Acceptance checklist

A reviewer can check each of these against the repository.

| # | Criterion | How to check |
| --- | --- | --- |
| 1 | Inputs are pinned and verified | `python scripts/fetch_data.py --verify-only` exits zero |
| 2 | The schema is as documented | `docs/data_dictionary.md`, regenerated by phase 3 |
| 3 | Quality checks run and their failures are declared | `results/data_quality_report.csv`, two declared failures, both handled |
| 4 | Two implementations of the quality counts agree | `results/quality_cross_check.csv`, all rows agree |
| 5 | No feature reads the future | `pytest tests/test_features.py` |
| 6 | The leakage guard is capable of failing | `test_the_leakage_guard_itself_detects_a_planted_leak` |
| 7 | No training target falls inside a test window | `pytest tests/test_cv.py` |
| 8 | Metrics behave at their edges | `pytest tests/test_metrics.py` |
| 9 | The queueing model matches closed forms | `pytest tests/test_queueing.py` |
| 10 | The drift monitor detects an injected shift | `test_input_monitor_detects_a_known_injected_shift` |
| 11 | The scoring contract rejects bad input | `pytest tests/test_scoring.py` |
| 12 | The grid is complete, with no cell silently skipped | `python scripts/run_phase07_grid.py --status` reports 576 of 576 |
| 13 | Headline tables are reproducible | phase 12 reports matching SHA-256 digests across two runs |
| 14 | Pre-registered claims are resolved either way | `docs/results.md` and `docs/preregistration.md` |
| 15 | Limitations are stated | `docs/limitations.md` |

## 8. Sign off

| Role | Question to answer | Signature |
| --- | --- | --- |
| Data owner | Is the lineage in section 2 accurate and complete? | |
| Modeller | Are the assumptions in section 3 the ones actually implemented? | |
| Reviewer | Does section 4 support the intended use in section 1? | |
| Operations | Is the monitoring plan in section 6 deployable as written? | |
| Accountable owner | Are the limitations acceptable for the decision at hand? | |
