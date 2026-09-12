# Limitations

What this study cannot support, listed so that a reader does not have to infer
it from what is missing.

## The data is zone level, so no claim is made about a site

Every conclusion is at the granularity of a traffic analysis zone, and a zone
aggregates several stations: 1362 stations across 275 zones, a median of three
per zone. A zone that looks saturated may contain one full site and two empty
ones. Nothing here supports a decision about an individual forecourt, and I
mean the capacity allocation to be read as a decision about where in a city to
invest rather than which site to extend.

## The retained zones are a survivorship filtered sample

The publisher excluded low activity zones with a variance test and a zero value
filter. The 275 zones that remain are the ones that already had meaningful
charging demand during the window. I don't extend the conclusions to quiet
zones, and in particular the finding that marginal value and utilisation rank
zones differently is a statement about busy zones only. A quiet zone is absent
from the sample entirely, not represented as quiet.

## Energy is modelled, not metered

Neither volume series is measured energy. Both are products of duration and an
assumed power. The rated series assumes vehicles draw the full rated power of
the point; the 11 kW series assumes a vehicle side limit. Across the estate the
first is 2.42 times the second. I use the rated estimate for the headline
figures, and every energy number should be read with that factor in mind. Forty
one of 275 zones have points rated above 11 kW, and those zones hold 72.8
percent of energy under the rated estimate.

## Price numbers are descriptive, never causal

Electricity price and service price vary across zones and over time, and demand
responds to them. Prices were also set in response to expected demand. A
regression of demand on price therefore recovers a mixture of the demand
response and the pricing rule, and no design here separates the two. There is
also very little variation to work with: both price series sit inside a
constant run of four hours or more for more than 84 percent of zone hours. I
report no elasticity, and the price features in the models go in as predictors
rather than levers.

## The cleaning is upstream and cannot be undone

Anomaly removal, forward and backward fill, and interquartile range outlier
replacement were applied before publication. Three consequences follow.
Anomalies I find in this study are residual anomalies in a cleaned series, not
raw sensor faults. Runs of identical consecutive values are candidate
imputation artefacts and are common: 41.4 percent of occupancy observations sit
inside a constant run of four hours or more. And the error metrics are measured
against a smoothed series, so I expect them to be optimistic relative to raw
telemetry.

## Weather is a lower resolution covariate

The weather files are published hourly but consecutive hours repeat exactly in
52 to 69 percent of the series depending on the station and field, which
indicates interpolation or forward fill from a coarser observation interval of
roughly three hours. I treat weather as a coarse covariate and don't credit it
with hourly explanatory power.

## Mean session length is assumed

The dataset records hourly aggregates, never individual sessions, so mean
session length is not identifiable from it. I assume it from the power of the
zone's points and vary it in a sensitivity. An estimate from the hour to hour
persistence of occupancy returns roughly seven hours, which is implausible for
a charging session and reflects that the method conflates long sessions with
autocorrelated demand; I report it as a diagnostic only.

## Unserved demand is a lower bound

Offered load at a zone hour recorded as completely full is unbounded in the
loss model, so its recovery is capped. Every figure for unserved demand, and
every marginal value derived from it, is therefore understated at exactly the
zones where capacity binds hardest.

## Substitution is modelled with one assumed parameter

I set the share of blocked demand that diverts to an adjacent zone at 0.35,
decaying with distance. The dataset cannot identify it, because a driver who
diverts is not recorded as having been turned away anywhere. The adjacency
matrix that defines who can absorb whom also required repair before use: it was
published with a unit diagonal and with twenty edges recorded in one direction
only.

## The window contains several regimes, and six months is short

Six months spanning late 2022 into early 2023 covers at least two demand
episodes. Both detectors agree on a decline in late October with a recovery in
early November, and on a decline in mid January with a recovery in early
February. A model validated on this window has been validated across regime
change, which is informative, but six months does not establish an annual
seasonal pattern, and I make no claim about one.

## One break, so the monitoring comparison rests on a single event

I measure detection delay against one change point. The monitoring result is a
measurement on that event, not an estimate of average detection delay, and its
uncertainty isn't quantified by repetition. A sensitivity shows the comparison
is affected by whether an unrelated one day shock in December sits inside the
calibration window, which I state in the results rather than smooth over.

## No causal claim about capacity

The queueing model supplies a counterfactual the data does not contain. It is a
model, with a loss discipline, memoryless service and Poisson arrivals assumed.
I state those assumptions because the marginal values depend on them, and I
haven't validated them against observed behaviour at a zone that actually
gained capacity, because the window contains no such event that this data
records.
