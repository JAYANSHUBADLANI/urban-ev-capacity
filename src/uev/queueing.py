"""Queueing model, capacity simulation and budget allocation.

The question the capacity decision needs answered is what would have happened
with more charging points, and that is not observable: the data records demand
that was served, not demand that arrived and left. A queueing model supplies
the missing counterfactual, which is why simulation appears here rather than a
regression on observed throughput.

Each zone is treated as a loss system with as many servers as it has charging
points. A driver who finds every point busy does not queue indefinitely; most
divert or give up, and the adjacency matrix exists because some of them divert
to a neighbouring zone. Offered load is recovered by inverting the Erlang loss
formula on the observed carried load, so the calibration uses the occupancy the
data actually records.

Two routes to the marginal value of a point are computed and compared. The
analytic route evaluates the loss formula at one more server, which is instant
and is what the greedy allocation uses at every step. The simulated route
replays arrivals and departures with neighbour substitution across seeds, which
is what the preregistered ranking is measured on. Agreement between the two is
reported rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG, CapacityConfig
from .graph import neighbour_weights
from .io_load import Bundle
from .logging_utils import get_logger
from .seeds import rng

LOG = get_logger("queueing")

# Mean session length is an assumption, not a measurement. The dataset records
# hourly aggregates and never individual sessions, so session length is not
# identifiable from it. It is set from the power of the points in the zone,
# which the data does reveal, and the write up carries a sensitivity across the
# whole assumption. The persistence based estimate is retained as a diagnostic
# only: it conflates long sessions with autocorrelated demand and returns
# implausible values of around seven hours on this series.
MIN_SERVICE_HOURS = 0.5
MAX_SERVICE_HOURS = 8.0
SERVICE_HOURS_FAST = 1.0        # points rated at 22 kW or more
SERVICE_HOURS_MEDIUM = 2.0      # points rated between 11 and 22 kW
SERVICE_HOURS_SLOW = 3.5        # points rated below 11 kW
FAST_KW = 22.0
MEDIUM_KW = 11.0

# Offered load is unbounded at a zone hour recorded as completely full, because
# no finite arrival rate fills every point with certainty. Recovered offered
# load is therefore capped at the level where this share of arrivals would be
# turned away. Without the cap the 0.7 percent of zone hours at the ceiling
# carry 94 percent of the total recovered load and the calibration is
# meaningless. The cap makes unserved demand a lower bound, which the write up
# states.
MAX_BLOCKING = 0.5
GRID_POINTS = 1500


def erlang_b(servers: int, offered: np.ndarray) -> np.ndarray:
    """Blocking probability of an ``M/M/c/c`` loss system, by the stable recursion."""
    offered = np.asarray(offered, dtype=float)
    blocking = np.ones_like(offered)
    for n in range(1, servers + 1):
        blocking = (offered * blocking) / (n + offered * blocking)
    return blocking


def carried_load(servers: int, offered: np.ndarray) -> np.ndarray:
    """Mean number of busy servers at a given offered load."""
    return offered * (1.0 - erlang_b(servers, offered))


def offered_grid(servers: int, grid_points: int = GRID_POINTS,
                 max_blocking: float = MAX_BLOCKING) -> np.ndarray:
    """Offered load values to evaluate the loss formula on.

    The grid runs from zero to the offered load at which the blocking
    probability reaches ``max_blocking``. The upper end is found by doubling
    and then bisecting, so it adapts to the number of servers rather than being
    a fixed multiple of it.
    """
    ceiling = max(float(servers), 1.0)
    high = 2.0 * ceiling
    for _ in range(60):
        if erlang_b(servers, np.array([high]))[0] >= max_blocking:
            break
        high *= 2.0
    low = 0.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if erlang_b(servers, np.array([middle]))[0] < max_blocking:
            low = middle
        else:
            high = middle
    return np.linspace(0.0, high, grid_points)


def invert_carried_load(servers: int, carried: np.ndarray,
                        grid_points: int = GRID_POINTS) -> np.ndarray:
    """Offered load implied by an observed carried load.

    Carried load rises monotonically with offered load and saturates at the
    number of servers, so the mapping is inverted once on a grid and then read
    off by interpolation, which is far cheaper than solving per observation.

    A zone hour at full occupancy is pinned to the top of the grid, which is the
    load at which half of arrivals would be turned away. The recovered offered
    load is a lower bound there, and so is every quantity derived from it.
    """
    carried = np.asarray(carried, dtype=float)
    grid = offered_grid(servers, grid_points)
    values = np.maximum.accumulate(carried_load(servers, grid))
    clipped = np.clip(carried, 0.0, values[-1])
    return np.interp(clipped, values, grid)


def service_hours_from_power(kw_per_point_hour: np.ndarray) -> np.ndarray:
    """Assumed mean session length, set from the power of the zone's points.

    A fast charging zone turns a point over in around an hour and a slow AC zone
    holds one for several. This is an assumption about driver behaviour that the
    data cannot check, so it is applied transparently and varied in the
    sensitivity analysis rather than presented as an estimate.
    """
    power = np.asarray(kw_per_point_hour, dtype=float)
    hours = np.full(power.shape, SERVICE_HOURS_SLOW)
    hours[power >= MEDIUM_KW] = SERVICE_HOURS_MEDIUM
    hours[power >= FAST_KW] = SERVICE_HOURS_FAST
    return hours


def service_hours_from_persistence(occupancy: np.ndarray) -> float:
    """Mean session length implied by the hour to hour persistence of occupancy.

    In a loss system with memoryless service the number of busy servers decays
    towards its mean at a rate set by the service rate, so the lag one
    autocorrelation of the deseasonalised series carries that rate. The estimate
    is clamped to a physically sensible range.
    """
    series = np.asarray(occupancy, dtype=float)
    if series.size < 48 or np.allclose(series, series[0]):
        return 2.0
    hours = np.arange(series.size) % 24
    seasonal = np.array([series[hours == h].mean() for h in range(24)])
    residual = series - seasonal[hours]
    if np.std(residual) < 1e-9:
        return 2.0
    correlation = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
    correlation = min(max(correlation, 1e-3), 0.999)
    return float(np.clip(-1.0 / np.log(correlation), MIN_SERVICE_HOURS, MAX_SERVICE_HOURS))


@dataclass
class ZoneCalibration:
    """Per zone parameters the simulation runs on."""

    zone_ids: List[int]
    capacity: np.ndarray            # (zones,) installed charging points
    service_hours: np.ndarray       # (zones,) mean session length
    kw_rated: np.ndarray            # (zones,) kWh per point hour, rated estimate
    kw_11: np.ndarray               # (zones,) kWh per point hour, 11 kW estimate
    offered: np.ndarray             # (hours, zones) offered load in erlangs
    carried: np.ndarray             # (hours, zones) observed busy points
    index: pd.DatetimeIndex
    persistence_hours: Optional[np.ndarray] = None   # diagnostic only

    @property
    def arrival_rate(self) -> np.ndarray:
        """Sessions arriving per hour, implied by offered load and session length."""
        return self.offered / self.service_hours[None, :]

    def session_energy(self, estimate: str = "rated") -> np.ndarray:
        power = self.kw_rated if estimate == "rated" else self.kw_11
        return self.service_hours * power


def calibrate(bundle: Bundle, service_scale: float = 1.0) -> ZoneCalibration:
    """Recover offered load and session parameters for every zone.

    ``service_scale`` multiplies the assumed mean session length and exists so
    that the sensitivity analysis can vary the one parameter the data cannot
    identify.
    """
    zone_ids = list(bundle.zones["zone_id"])
    capacity = bundle.zones.set_index("zone_id")["capacity_points"] \
        .reindex(zone_ids).to_numpy(dtype=int)

    occupancy = bundle.long.pivot(index="timestamp", columns="zone_id", values="occupancy")
    occupancy = occupancy.reindex(index=bundle.index, columns=zone_ids).to_numpy(dtype=float)

    totals = bundle.long.groupby("zone_id")[["volume", "volume_11kw", "duration"]].sum()
    totals = totals.reindex(zone_ids)
    point_hours = totals["duration"].replace(0, np.nan)
    kw_rated = (totals["volume"] / point_hours).fillna(0.0).to_numpy(dtype=float)
    kw_11 = (totals["volume_11kw"] / point_hours).fillna(0.0).to_numpy(dtype=float)

    service = service_hours_from_power(kw_rated) * service_scale
    persistence = np.array([service_hours_from_persistence(occupancy[:, z])
                            for z in range(len(zone_ids))])

    offered = np.empty_like(occupancy)
    for z, servers in enumerate(capacity):
        offered[:, z] = invert_carried_load(int(servers), occupancy[:, z])

    LOG.info("calibrated %d zones: assumed session hours median %.2f "
             "(persistence diagnostic %.2f), offered load exceeds carried by %.1f%%",
             len(zone_ids), float(np.median(service)), float(np.median(persistence)),
             100.0 * (offered.sum() / max(occupancy.sum(), 1e-9) - 1.0))

    return ZoneCalibration(zone_ids=zone_ids, capacity=capacity, service_hours=service,
                           kw_rated=kw_rated, kw_11=kw_11, offered=offered,
                           carried=occupancy, index=bundle.index,
                           persistence_hours=persistence)


def analytic_marginal(calibration: ZoneCalibration, hours: slice,
                      estimate: str = "rated") -> np.ndarray:
    """Extra point hours served by one more point, per zone, ignoring spillover."""
    offered = calibration.offered[hours]
    out = np.zeros(len(calibration.zone_ids))
    for z, servers in enumerate(calibration.capacity):
        current = carried_load(int(servers), offered[:, z])
        improved = carried_load(int(servers) + 1, offered[:, z])
        out[z] = float(np.sum(improved - current))
    power = calibration.kw_rated if estimate == "rated" else calibration.kw_11
    return out * power


def peak_window(calibration: ZoneCalibration, n_hours: int) -> slice:
    """The busiest contiguous stretch of the timeline, used for every scenario."""
    total = calibration.carried.sum(axis=1)
    rolling = pd.Series(total).rolling(n_hours).sum()
    end = int(rolling.idxmax())
    return slice(end - n_hours + 1, end + 1)


def simulate(calibration: ZoneCalibration, capacities: np.ndarray, hours: slice,
             weights: np.ndarray, seed_label: str,
             settings: CapacityConfig | None = None,
             estimate: str = "rated") -> Dict[str, np.ndarray]:
    """Replay arrivals and departures for one or more capacity scenarios.

    ``capacities`` is ``(worlds, zones)``. Arrivals are drawn once per hour and
    shared across worlds, so two worlds differing by a single point see exactly
    the same demand and their difference is the effect of that point rather
    than sampling noise.
    """
    settings = settings or CONFIG.capacity
    generator = rng(seed_label)
    rates = calibration.arrival_rate[hours]
    n_hours, n_zones = rates.shape
    n_worlds = capacities.shape[0]

    depart = np.clip(1.0 / calibration.service_hours, 0.0, 1.0)
    max_slots = int(capacities.max())
    slot_index = np.arange(n_zones)[None, :]

    busy = np.zeros((n_worlds, n_zones), dtype=np.int64)
    served = np.zeros((n_worlds, n_zones), dtype=np.int64)
    spilled_served = np.zeros((n_worlds, n_zones), dtype=np.int64)
    blocked_at_origin = np.zeros((n_worlds, n_zones), dtype=np.int64)
    offered_total = np.zeros((n_worlds, n_zones), dtype=np.int64)

    for t in range(n_hours):
        # Departures are coupled across scenarios by giving every charging point
        # its own random draw, shared by all scenarios, rather than drawing a
        # binomial per scenario. Two scenarios that differ by one point then see
        # the same departures from the points they share, and the difference
        # between them is the effect of that point instead of sampling noise.
        slot_draw = generator.random((n_zones, max_slots)) < depart[:, None]
        cumulative = np.concatenate(
            [np.zeros((n_zones, 1), dtype=np.int64),
             np.cumsum(slot_draw, axis=1, dtype=np.int64)], axis=1)
        busy = busy - cumulative[slot_index, busy]

        arrivals = generator.poisson(rates[t])[None, :]

        free = np.maximum(capacities - busy, 0)
        direct = np.minimum(arrivals, free)
        blocked = arrivals - direct
        busy = busy + direct
        served += direct
        blocked_at_origin += blocked
        offered_total += arrivals

        # Blocked demand partly diverts to adjacent zones, weighted by distance.
        offered_out = blocked * settings.spill_fraction
        received = offered_out.astype(float) @ weights
        whole = np.floor(received)
        rounding = generator.random(n_zones)[None, :]
        received_int = (whole + (rounding < (received - whole))).astype(np.int64)

        free_after = np.maximum(capacities - busy, 0)
        diverted = np.minimum(received_int, free_after)
        busy = busy + diverted
        spilled_served += diverted

    # A session blocked at its own zone and then served at a neighbour is not
    # lost, so unserved demand is a system level quantity: everything blocked
    # where it arrived, less everything picked up elsewhere.
    total_sessions = served + spilled_served
    energy = total_sessions * calibration.session_energy(estimate)[None, :]
    unserved = (blocked_at_origin.sum(axis=1) - spilled_served.sum(axis=1))
    return {
        "served_sessions": total_sessions,
        "served_energy": energy,
        "blocked_at_origin": blocked_at_origin,
        "spill_served": spilled_served,
        "unserved_sessions": unserved,
        "offered_sessions": offered_total,
        "busy_final": busy,
    }


def simulated_marginal(calibration: ZoneCalibration, hours: slice, weights: np.ndarray,
                       n_seeds: int, estimate: str = "rated") -> pd.DataFrame:
    """Extra system wide energy served by one more point at each zone.

    The total is taken across every zone, not only the zone that gained the
    point, because a point added where demand is blocked also removes the
    spillover that zone was pushing onto its neighbours.
    """
    n_zones = len(calibration.zone_ids)
    base = calibration.capacity[None, :].repeat(n_zones + 1, axis=0)
    for z in range(n_zones):
        base[z + 1, z] += 1

    totals = np.zeros((n_seeds, n_zones + 1))
    for seed in range(n_seeds):
        outcome = simulate(calibration, base, hours, weights,
                           f"marginal:{estimate}:{seed}", estimate=estimate)
        totals[seed] = outcome["served_energy"].sum(axis=1)

    baseline = totals[:, 0][:, None]
    marginal = totals[:, 1:] - baseline
    return pd.DataFrame({
        "zone_id": calibration.zone_ids,
        "marginal_energy_kwh_mean": marginal.mean(axis=0),
        "marginal_energy_kwh_std": marginal.std(axis=0),
        "marginal_energy_kwh_p10": np.quantile(marginal, 0.10, axis=0),
        "marginal_energy_kwh_p90": np.quantile(marginal, 0.90, axis=0),
        "baseline_energy_kwh_mean": float(baseline.mean()),
        "n_seeds": n_seeds,
    })


def allocate_by_weight(weights: np.ndarray, budget: int) -> np.ndarray:
    """Distribute an integer budget in proportion to a weight, largest remainder."""
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if weights.sum() <= 0:
        allocation = np.zeros(weights.size, dtype=int)
        allocation[:budget] = 1
        return allocation
    exact = weights / weights.sum() * budget
    allocation = np.floor(exact).astype(int)
    remainder = budget - allocation.sum()
    if remainder > 0:
        order = np.argsort(-(exact - allocation))
        allocation[order[:remainder]] += 1
    return allocation


def greedy_marginal_allocation(calibration: ZoneCalibration, hours: slice,
                               budget: int, estimate: str = "rated") -> np.ndarray:
    """Assign points one at a time to whichever zone gains most from the next one.

    The gain is re-evaluated after every point, so a zone that has been relieved
    stops attracting further capacity. The analytic loss formula is used here
    because a greedy pass of this length cannot afford a simulation per step.
    """
    offered = calibration.offered[hours]
    power = calibration.kw_rated if estimate == "rated" else calibration.kw_11
    n_zones = len(calibration.zone_ids)
    added = np.zeros(n_zones, dtype=int)

    current = np.array([carried_load(int(calibration.capacity[z]), offered[:, z]).sum()
                        for z in range(n_zones)])
    gains = np.array([
        carried_load(int(calibration.capacity[z]) + 1, offered[:, z]).sum() - current[z]
        for z in range(n_zones)]) * power

    for _ in range(budget):
        pick = int(np.argmax(gains))
        added[pick] += 1
        servers = int(calibration.capacity[pick]) + added[pick]
        current[pick] = carried_load(servers, offered[:, pick]).sum()
        gains[pick] = (carried_load(servers + 1, offered[:, pick]).sum()
                       - current[pick]) * power[pick]
    return added
