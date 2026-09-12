"""The neighbourhood structure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import needs_data
from uev.graph import (asymmetry_count, degree, nearest_neighbours,
                       neighbour_lists, neighbour_weights, symmetrise)


@pytest.fixture
def directed():
    values = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=float)
    ids = pd.Index([10, 20, 30], name="zone_id")
    return pd.DataFrame(values, index=ids, columns=ids)


@pytest.fixture
def distances():
    values = np.array([[0.0, 1000.0, 4000.0],
                       [1000.0, 0.0, 2000.0],
                       [4000.0, 2000.0, 0.0]])
    ids = pd.Index([10, 20, 30], name="zone_id")
    return pd.DataFrame(values, index=ids, columns=ids)


def test_symmetrise_takes_the_union(directed):
    out = symmetrise(directed).to_numpy()
    assert out[0, 1] == 1 and out[1, 0] == 1
    assert out[1, 2] == 1 and out[2, 1] == 1


def test_symmetrise_clears_the_diagonal(directed):
    assert np.diag(symmetrise(directed).to_numpy()).sum() == 0


def test_symmetrise_is_idempotent(directed):
    once = symmetrise(directed)
    assert (symmetrise(once).to_numpy() == once.to_numpy()).all()


def test_asymmetry_count_finds_the_one_way_edges(directed):
    assert asymmetry_count(directed) == 4


def test_degree_excludes_the_self_loop(directed):
    assert degree(directed).tolist() == [1, 2, 1]


def test_neighbour_lists_match_the_matrix(directed):
    assert neighbour_lists(directed) == {10: [20], 20: [10, 30], 30: [20]}


def test_nearest_neighbours_orders_by_distance(distances):
    assert nearest_neighbours(distances, k=2)[10] == [20, 30]


def test_nearest_neighbours_never_returns_the_zone_itself(distances):
    for zone, others in nearest_neighbours(distances, k=2).items():
        assert zone not in others


def test_neighbour_weights_rows_sum_to_one_or_zero(directed, distances):
    rows = neighbour_weights(directed, distances).to_numpy().sum(axis=1)
    assert np.allclose(rows, 1.0)


def test_neighbour_weights_favour_the_closer_neighbour(directed, distances):
    weights = neighbour_weights(directed, distances)
    assert weights.loc[20, 10] > weights.loc[20, 30]


def test_isolated_zone_gets_zero_weight(distances):
    ids = pd.Index([10, 20, 30], name="zone_id")
    isolated = pd.DataFrame(np.zeros((3, 3)), index=ids, columns=ids)
    assert neighbour_weights(isolated, distances).to_numpy().sum() == 0.0


@needs_data
def test_published_adjacency_is_asymmetric_as_recorded(bundle):
    assert asymmetry_count(bundle.adjacency) == 40


@needs_data
def test_published_adjacency_has_a_unit_diagonal(bundle):
    assert np.diag(bundle.adjacency.to_numpy()).sum() == 275


@needs_data
def test_symmetrised_graph_is_undirected(bundle):
    values = symmetrise(bundle.adjacency).to_numpy()
    assert (values == values.T).all()
