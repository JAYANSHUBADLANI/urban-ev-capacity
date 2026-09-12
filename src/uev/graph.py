"""Zone neighbourhood structure.

The published adjacency matrix has two properties that have to be handled
before it can be used. The diagonal is one, so a zone is listed as adjacent to
itself; and twenty of the five hundred and ninety nine undirected edges are
recorded in one direction only. Spatial adjacency is symmetric by definition,
so the matrix is symmetrised by union and the diagonal is removed. Both
decisions are recorded in the data quality report rather than applied silently.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def symmetrise(adjacency: pd.DataFrame) -> pd.DataFrame:
    """Return the undirected adjacency with a zero diagonal."""
    values = adjacency.to_numpy().astype(np.int8)
    undirected = np.maximum(values, values.T)
    np.fill_diagonal(undirected, 0)
    return pd.DataFrame(undirected, index=adjacency.index, columns=adjacency.columns)


def asymmetry_count(adjacency: pd.DataFrame) -> int:
    """Number of ordered cells that disagree with their transpose."""
    values = adjacency.to_numpy()
    return int((values != values.T).sum())


def neighbour_lists(adjacency: pd.DataFrame) -> Dict[int, List[int]]:
    """Map each zone to its neighbouring zones."""
    undirected = symmetrise(adjacency)
    ids = list(undirected.columns)
    values = undirected.to_numpy()
    return {int(ids[i]): [int(ids[j]) for j in np.flatnonzero(values[i])]
            for i in range(len(ids))}


def degree(adjacency: pd.DataFrame) -> pd.Series:
    """Number of neighbours per zone."""
    undirected = symmetrise(adjacency)
    return pd.Series(undirected.to_numpy().sum(axis=1), index=undirected.index,
                     name="n_neighbours")


def nearest_neighbours(distance: pd.DataFrame, k: int = 5) -> Dict[int, List[int]]:
    """Map each zone to its k nearest other zones by the published distance."""
    ids = list(distance.columns)
    values = distance.to_numpy().astype(float).copy()
    np.fill_diagonal(values, np.inf)
    order = np.argsort(values, axis=1)[:, :k]
    return {int(ids[i]): [int(ids[j]) for j in order[i]] for i in range(len(ids))}


def neighbour_weights(adjacency: pd.DataFrame, distance: pd.DataFrame,
                      decay_m: float = 2000.0) -> pd.DataFrame:
    """Row normalised substitution weights between adjacent zones.

    Weight falls with distance because a driver denied a point is more likely
    to divert to a close neighbour than a far one. Rows for zones with no
    neighbour sum to zero, which means their blocked demand is simply lost.
    """
    undirected = symmetrise(adjacency).to_numpy().astype(float)
    metres = distance.to_numpy().astype(float)
    weights = undirected * np.exp(-metres / decay_m)
    totals = weights.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        weights = np.where(totals > 0, weights / totals, 0.0)
    return pd.DataFrame(weights, index=adjacency.index, columns=adjacency.columns)
