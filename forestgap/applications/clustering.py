"""Clustering from declared symmetric forest affinities."""

from __future__ import annotations

from typing import Literal

from sklearn.cluster import AgglomerativeClustering, SpectralClustering

from forestgap.manifold.proximity import prepare_proximity_affinity, proximity_to_distance
from forestgap.proximity import Proximity


def proximity_clustering(
    proximity: Proximity,
    n_clusters: int,
    *,
    method: Literal["spectral", "agglomerative"] = "spectral",
    random_state: int | None = None,
):
    """Cluster indexed rows after explicit symmetrization."""

    affinity = prepare_proximity_affinity(
        proximity.get_proximities(), symmetrize="auto", diagonal="one",
    )
    if method == "spectral":
        return SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=random_state,
        ).fit_predict(affinity)
    if method == "agglomerative":
        distance = proximity_to_distance(affinity, dense=True)
        return AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average",
        ).fit_predict(distance)
    raise ValueError("method must be 'spectral' or 'agglomerative'")
