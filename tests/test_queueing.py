"""The queueing model, checked against cases with a known answer.

The Erlang loss formula has closed forms at small server counts, so the
implementation is compared against those rather than against itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uev.queueing import (allocate_by_weight, carried_load, erlang_b,
                          invert_carried_load, offered_grid,
                          service_hours_from_persistence)


def test_single_server_blocking_matches_the_closed_form():
    """With one server, blocking is A / (1 + A)."""
    for offered in (0.5, 1.0, 2.0, 10.0):
        assert erlang_b(1, np.array([offered]))[0] == pytest.approx(offered / (1 + offered))


def test_two_server_blocking_matches_the_closed_form():
    """With two servers, blocking is A squared over (2 + 2A + A squared)."""
    for offered in (0.5, 1.0, 3.0):
        expected = offered ** 2 / (2 + 2 * offered + offered ** 2)
        assert erlang_b(2, np.array([offered]))[0] == pytest.approx(expected)


@pytest.mark.parametrize("servers", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("offered", [0.25, 1.0, 4.0])
def test_blocking_matches_the_factorial_definition(servers, offered):
    terms = [offered ** n / float(np.prod(range(1, n + 1)) or 1) for n in range(servers + 1)]
    expected = terms[-1] / sum(terms)
    assert erlang_b(servers, np.array([offered]))[0] == pytest.approx(expected, rel=1e-9)


def test_zero_servers_block_everything():
    assert erlang_b(0, np.array([3.0]))[0] == pytest.approx(1.0)


def test_zero_offered_load_blocks_nothing():
    assert erlang_b(5, np.array([0.0]))[0] == pytest.approx(0.0)


def test_blocking_rises_with_offered_load():
    values = erlang_b(4, np.array([0.5, 1.0, 2.0, 4.0, 8.0]))
    assert np.all(np.diff(values) > 0)


def test_blocking_falls_as_servers_are_added():
    values = [erlang_b(c, np.array([5.0]))[0] for c in range(1, 12)]
    assert np.all(np.diff(values) < 0)


def test_carried_load_never_exceeds_the_servers():
    assert carried_load(5, np.array([1e6]))[0] < 5.0


def test_carried_load_approaches_offered_load_when_lightly_loaded():
    assert carried_load(50, np.array([1.0]))[0] == pytest.approx(1.0, rel=1e-6)


def test_carried_load_rises_with_offered_load():
    values = carried_load(6, np.array([1.0, 2.0, 4.0, 8.0, 16.0]))
    assert np.all(np.diff(values) > 0)


@pytest.mark.parametrize("servers", [1, 4, 20])
@pytest.mark.parametrize("fraction", [0.1, 0.5, 0.9])
def test_inversion_round_trips_inside_the_representable_range(servers, fraction):
    """Round trip for offered loads below the blocking cap the grid stops at."""
    top = offered_grid(servers)[-1]
    offered = fraction * top
    carried = carried_load(servers, np.array([offered]))
    assert invert_carried_load(servers, carried)[0] == pytest.approx(offered, rel=2e-2)


@pytest.mark.parametrize("servers", [1, 3, 12, 40])
def test_the_offered_grid_stops_at_the_stated_blocking_level(servers):
    """Offered load is deliberately capped where half of arrivals are turned away."""
    from uev.queueing import MAX_BLOCKING
    top = offered_grid(servers)[-1]
    assert erlang_b(servers, np.array([top]))[0] == pytest.approx(MAX_BLOCKING, abs=1e-3)


def test_the_grid_starts_at_zero():
    assert offered_grid(5)[0] == pytest.approx(0.0)


def test_the_grid_is_increasing():
    assert np.all(np.diff(offered_grid(7)) > 0)


def test_offered_load_beyond_the_cap_is_pinned_not_extrapolated():
    """Carried load above what the capped grid reaches returns the cap."""
    top = offered_grid(4)[-1]
    beyond = carried_load(4, np.array([top * 50]))
    assert invert_carried_load(4, beyond)[0] == pytest.approx(top, rel=1e-6)


def test_a_zone_hour_at_full_occupancy_is_pinned_to_the_grid_top():
    """Full occupancy implies unbounded offered load, so the result is a bound."""
    recovered = invert_carried_load(4, np.array([4.0]))[0]
    assert recovered == pytest.approx(offered_grid(4)[-1])


def test_service_hours_follow_the_power_of_the_points():
    from uev.queueing import (SERVICE_HOURS_FAST, SERVICE_HOURS_MEDIUM,
                              SERVICE_HOURS_SLOW, service_hours_from_power)
    hours = service_hours_from_power(np.array([7.0, 15.0, 120.0]))
    assert hours.tolist() == [SERVICE_HOURS_SLOW, SERVICE_HOURS_MEDIUM, SERVICE_HOURS_FAST]


def test_a_faster_charger_is_assumed_to_free_its_point_sooner():
    from uev.queueing import service_hours_from_power
    hours = service_hours_from_power(np.array([3.5, 7.0, 11.0, 22.0, 150.0]))
    assert np.all(np.diff(hours) <= 0)


def test_inversion_of_zero_carried_load_is_zero():
    assert invert_carried_load(5, np.array([0.0]))[0] == pytest.approx(0.0)


def test_inversion_returns_at_least_the_carried_load():
    """Offered load can never be less than what was actually carried."""
    carried = np.array([0.5, 2.0, 4.5])
    assert np.all(invert_carried_load(5, carried) >= carried - 1e-6)


def test_service_hours_of_a_persistent_series_are_long():
    hours = np.arange(500)
    slow = 10 + np.cumsum(np.random.default_rng(0).normal(0, 0.05, 500))
    fast = 10 + np.random.default_rng(0).normal(0, 1.0, 500)
    assert service_hours_from_persistence(slow) > service_hours_from_persistence(fast)


def test_service_hours_are_clamped_to_a_plausible_range():
    from uev.queueing import MAX_SERVICE_HOURS, MIN_SERVICE_HOURS
    value = service_hours_from_persistence(np.cumsum(np.ones(500)))
    assert MIN_SERVICE_HOURS <= value <= MAX_SERVICE_HOURS


def test_service_hours_of_a_constant_series_fall_back():
    assert service_hours_from_persistence(np.full(100, 7.0)) == pytest.approx(2.0)


def test_allocation_spends_the_whole_budget():
    allocation = allocate_by_weight(np.array([1.0, 2.0, 3.0]), 10)
    assert allocation.sum() == 10


def test_allocation_follows_the_weights():
    allocation = allocate_by_weight(np.array([1.0, 9.0]), 10)
    assert allocation[1] > allocation[0]


def test_allocation_of_a_zero_weight_still_spends_the_budget():
    assert allocate_by_weight(np.zeros(5), 3).sum() == 3


def test_allocation_never_goes_negative():
    assert (allocate_by_weight(np.array([-1.0, 2.0]), 4) >= 0).all()


def test_allocation_of_an_empty_budget_is_empty():
    assert allocate_by_weight(np.array([1.0, 2.0]), 0).sum() == 0
