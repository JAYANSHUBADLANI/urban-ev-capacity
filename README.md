# Urban EV charging: demand, capacity, and what breaks when demand shifts

A fuel retail network operator runs a large estate of forecourt sites and has
to decide how much public EV charging capacity to add, and where. Capital is
limited, sites differ, and the obvious rule (add capacity where utilisation is
already highest) may not be the rule that serves the most energy.

I built this project to answer three questions in order, using six months of
hourly public charging data from Shenzhen:

1. How well can hourly charging demand be predicted per zone, and does one
   model pooled across all zones beat a model fitted per zone?
2. When demand regime shifts, how quickly does that show up, and what should
   trigger a retrain?
3. Given a fixed capacity budget, where should it go, and does that ranking
   match the utilisation ranking the business would reach for by default?

I pre-registered three claims in `docs/preregistration.md` before running the
model grid, each with a refutation condition. I report all three outcomes
below, including the one that failed.

## Headline findings

Six months of hourly data, 275 zones, 1,194,600 zone hours, 17,532 charging points across 1,362 stations. The forecasting grid is 576 cells: 3 targets by 3 horizons by 8 rolling origin folds by 8 model families, each scored over every zone.

**Claim 1, that a pooled model beats per zone models and helps quieter zones most: refuted.** The pooled model wins 6 of 9 target and horizon combinations on pooled MASE, so the first half holds. The second half does not: the Spearman correlation between zone volume decile and the pooled model's advantage is +0.334 across 275 zones, and the claim required it to be negative. The advantage runs the other way: in the quietest decile the per zone models win by 0.178 MASE on average, while the busiest decile favours pooling by +0.012. Borrowing strength across zones helps the zones that already have the most data, which is the opposite of the reasoning the claim was built on.

**Claim 2, that input drift shows up later than residual error: refuted.** Against the change point of 2023-01-18 00:00, the input monitor alerts 143 hours in and the residual monitor never alerts at any of the matched rates. The reason is visible in the data: after the break the mean absolute residual falls rather than rises, because a one hour ahead model leaning on the latest observation simply tracks the new level. A separate one day shock inside the calibration window also lifts the residual thresholds; excluding it reverses the answer at the two looser rates, which is reported in `docs/results.md` rather than left out.

**Claim 3, that marginal value ranks zones differently from utilisation: supported.** The Spearman correlation between mean utilisation and simulated marginal served energy is 0.577, below the 0.9 the claim set, and only 11 of the top twenty zones are shared. Allocating by marginal served energy beats allocating by utilisation at 5 of 5 budgets, by 268,708 kWh on average over the peak fortnight. The substance is that 152 of 275 zones gain nothing at all from an extra point, because they never fill: utilisation orders every zone, marginal value is zero for most of them.

**On retraining:** doing nothing gives a mean absolute error of 1.722 charging points over the scored period. Retraining when the input drift monitor fires costs 5 refits and gives 1.699, with the lowest error after the break of any policy tested (1.473 against 1.526). Fixed fortnightly retraining reaches a similar place for more refits.

**The number that moves everything:** energy is modelled, not metered, and the two published estimates differ by a factor of 2.42 across the estate (3.117e+08 kWh against 1.286e+08 kWh). Headline energy figures use the rated power estimate. The choice changes every energy level and none of the rankings: the allocation rules finish in the same order under both.

## Data

UrbanEV, an open benchmark of public EV charging in Shenzhen, China, released
under CC0 1.0 and published in Scientific Data in 2025, DOI
[10.1038/s41597-025-04874-4](https://doi.org/10.1038/s41597-025-04874-4).
Source repository:
[IntelligentSystemsLab/UrbanEV](https://github.com/IntelligentSystemsLab/UrbanEV).

I fetch thirteen files from one pinned commit and hash them on arrival. A later
run that sees a different digest stops rather than continuing.

Coverage is 2022-09-01 00:00 to 2023-02-28 23:00: 4344 hourly periods, 275
zones, 1,194,600 zone hours, 1362 stations, 17,532 charging points.

**The data is zone level, not site level.** A zone aggregates several stations,
a median of three. Nothing here supports a decision about an individual
forecourt, so I read every capacity conclusion as a statement about where in a
city to invest rather than which site to extend. The retained zones are also a
survivorship filtered sample: the publisher excluded low activity zones, so I
don't extend the conclusions to quiet ones.

## Repository layout

```
src/uev/          the package: loading, quality, features, models, monitoring,
                  queueing, segmentation, figures
scripts/          one runnable script per phase, plus the fetch and the archive
sql/              the reporting layer as real SQL, executed by the pipeline
tests/            the test suite
docs/             preregistration, method, data dictionary, results,
                  model evidence pack, limitations
results/          every table the write up quotes
figures/          the figures
```

`data/` and `state/` are not part of the repository. Inputs are fetched and run
bookkeeping is local to a run.

## Running it

Dependencies are numpy, scipy, pandas, scikit-learn, matplotlib and pytest.
`duckdb` is used as a faster SQL engine if it is already installed, and the SQL
layer falls back to `sqlite3` from the standard library otherwise. The scripts
add `src/` to the import path themselves, so no install step is needed.

```bash
pip install -r requirements.txt
```

Fetch the inputs and record their digests:

```bash
python scripts/fetch_data.py
```

Then run the phases in order. Each one reads the artefacts of the phases before
it and can be run on its own.

```bash
python scripts/run_phase02_load.py
```

```bash
python scripts/run_phase03_quality.py
```

```bash
python scripts/run_phase04_eda.py
```

```bash
python scripts/run_phase07_grid.py
```

```bash
python scripts/run_phase08_monitor.py
```

```bash
python scripts/run_phase09_segment.py
```

```bash
python scripts/run_phase10_capacity.py
```

```bash
python scripts/run_phase11_report.py
```

```bash
python scripts/run_phase12_selfreview.py
```

The grid is the long part. It writes each cell as it finishes and skips cells
already recorded, so it can be stopped and resumed. Progress at any time:

```bash
python scripts/run_phase07_grid.py --status
```

The test suite:

```bash
python -m pytest tests/ -q
```

Re-verify the inputs against the manifest without downloading again:

```bash
python scripts/fetch_data.py --verify-only
```

Write an archive of the code, results, figures and documents:

```bash
python scripts/make_checkpoint.py
```

## What is in the data that had to be handled

Seven things about this dataset change the analysis, and I handle each one in
code rather than noting it and moving on.

**Occupancy is a count, not a percentage.** It counts charging points in use,
is integer valued in 99.94 percent of zone hours, and never exceeds the points
installed in its zone across all 1,194,600 zone hours. I derive utilisation by
dividing by installed points.

**Energy is modelled, not metered.** `volume` applies the rated power of the
points and `volume-11kW` applies an 11 kW vehicle side limit. Dividing each by
`duration` recovers the implied power, and for 239 of 275 zones the second is
the first capped at 11 kW. It is not a clean cap everywhere: 34 zones imply
more power under the 11 kW series than under the rated one, and 5 zones exceed
12 kW under it. Across the estate the rated estimate totals 2.42 times the
other, and 41 zones with points above 11 kW hold 72.8 percent of rated energy.
I carry both series through every energy result and state in the headline
figures which one is used.

**The series were already cleaned.** Anomaly removal, forward and backward fill
and outlier replacement were applied upstream, so anomalies I find here are
residual anomalies in a cleaned series. Runs of identical consecutive values
are common (41.4 percent of occupancy observations sit inside a run of four
hours or more), and I report them rather than repair them.

**Weather is not hourly at source.** Consecutive hours repeat exactly in 52 to
69 percent of the series, so I treat it as a coarse covariate.

**Price is endogenous.** Prices were set partly in response to expected demand,
so I report no elasticity, and price features go in as predictors, not levers.

**Occupancy is bounded, and the bound binds.** 104 of 275 zones reach full
occupancy at least once. A symmetric error metric misleads exactly at the
operating point the capacity decision cares about, so I report a saturation
weighted metric alongside the standard ones.

**Adjacent zones substitute.** Blocked demand at a saturated zone diverts to
neighbours, so the marginal value of capacity depends on the neighbourhood. The
published adjacency matrix needed repair first: it has a unit diagonal and
twenty edges recorded in one direction only. I report both as failures in the
data quality report and handle them explicitly.

## Method in one paragraph

I reshape the wide published tables once into a long frame. I build forty three
features, every one evaluated at or before the prediction origin, with a
leakage guard that replaces the series after the latest origin and rebuilds
through the ordinary code path, plus a companion test that plants a future
reading feature to prove the guard can fail. I fit eight model families across
three targets, three horizons and eight rolling origin folds: three baselines,
a per zone direct autoregression, and ridge and gradient boosting each fitted
both pooled and per zone. I locate change points from the data using two
independent detectors and give them significance with a local block bootstrap.
I calibrate four monitors to matched false alarm rates on the longest stretch
with no detected break, and measure detection delay against the break that
follows it. A queueing model recovers offered load by inverting the Erlang loss
formula on observed occupancy, and I use a simulation with neighbour
substitution to supply the counterfactual the data does not contain.

I give full detail in `docs/method.md`. What the study cannot support is in
`docs/limitations.md`.

## Reading the outputs

| File | What it holds |
| --- | --- |
| `docs/results.md` | Every headline number with its sample size |
| `docs/preregistration.md` | The three claims, frozen before the grid ran |
| `docs/model_evidence_pack.md` | Intended use, lineage, assumptions, failure modes, monitoring plan, acceptance checklist |
| `docs/limitations.md` | What this study cannot support |
| `docs/data_dictionary.md` | Every field, unit, source and prior preprocessing |
| `results/data_quality_report.csv` | All checks with pass, fail or informational status |
| `results/headline_model_table.csv` | Pooled error per target, horizon and family |
| `results/headline_capacity_table.csv` | Served energy per estimate, budget and rule |
| `results/claim_outcomes.csv` | The three verdicts |
| `results/self_review.csv` | The release checks and their results |

## Licence and citation

Code is MIT licensed; see `LICENSE`. The dataset is CC0 1.0. If the data is
used, cite the publication:

> UrbanEV: an open benchmark dataset of public electric vehicle charging in
> Shenzhen. *Scientific Data* (2025). DOI 10.1038/s41597-025-04874-4
>
> Dataset repository: https://github.com/IntelligentSystemsLab/UrbanEV
> Pinned commit: 44f2aa0c8d89f192bce00bafb0def74a21b39c68

I've deliberately not reproduced the author list here, because I didn't read it
from the data files this project fetches. The DOI resolves to the authoritative
citation.
