"""Manifold and MDS visualization through shared affinity/distance conversion."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS

from forestgap.manifold import embed_from_forest, transform_from_forest
from forestgap.manifold.proximity import prepare_proximity_affinity, proximity_to_distance
from forestgap.proximity import Proximity


def embed(proximity: Proximity, method: str = "phate", **kwargs: Any) -> np.ndarray:
    """Fit a manifold representation from the declared relation."""

    return embed_from_forest(proximity, method=method, **kwargs)


def embed_transform(
    proximity: Proximity, X_new: Any, method: str = "phate", **kwargs: Any,
) -> np.ndarray:
    """Project new rows through a stored manifold model when supported."""

    return transform_from_forest(proximity, X_new, method=method, **kwargs)


def plot_mds(
    proximity: Proximity,
    *,
    X: Any | None = None,
    y: Any | None = None,
    symmetrize: str | None = "auto",
    random_state: int | None = 42,
    ax: Any | None = None,
) -> Any:
    """Plot metric MDS, mapping implicit sparse zero affinity to max distance."""

    relation = proximity.get_proximities() if X is None else proximity.prox_extend(X)
    if relation.shape[0] != relation.shape[1]:
        raise ValueError("MDS visualization requires a square within-set relation")
    affinity = prepare_proximity_affinity(
        relation, symmetrize=symmetrize, diagonal="one",
    )
    distance = proximity_to_distance(affinity, dense=True)
    coords = MDS(
        n_components=2, dissimilarity="precomputed", random_state=random_state,
    ).fit_transform(distance)
    if ax is None:
        _, ax = plt.subplots()
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=y)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(f"{proximity.relation} proximity MDS")
    return ax.figure
