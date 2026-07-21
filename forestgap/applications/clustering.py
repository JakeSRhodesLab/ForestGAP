"""Clustering from declared forest-proximity affinities."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import KernelPCA

from forestgap.manifold.proximity import (
    SymmetrizeMode,
    prepare_proximity_affinity,
    proximity_to_distance,
)
from forestgap.proximity import Proximity


def proximity_clustering(
    proximity: Proximity,
    n_clusters: int | None = None,
    *,
    method: Literal["spectral", "hdbscan", "kmeans", "agglomerative"] = "spectral",
    random_state: int | None = None,
    symmetrize: SymmetrizeMode = "auto",
    **kwargs: Any,
) -> np.ndarray:
    """Cluster indexed rows from a declared forest-proximity relation.

    ``"spectral"`` consumes the prepared affinity directly. ``"hdbscan"``
    uses a dense precomputed distance obtained from that affinity, and
    ``"kmeans"`` applies :class:`~sklearn.cluster.KMeans` to a precomputed-
    kernel PCA embedding. Any additional keyword arguments are passed to the
    selected scikit-learn clustering estimator. ``n_clusters`` is required for
    spectral clustering and K-means; HDBSCAN determines its cluster count from
    its density settings (for example, ``min_cluster_size``).

    ``"agglomerative"`` remains available for backwards compatibility and
    receives a precomputed proximity-derived distance.
    """

    affinity = prepare_proximity_affinity(
        proximity.get_proximities(), symmetrize=symmetrize, diagonal="one",
    )
    if method == "spectral":
        _require_n_clusters(n_clusters, method)
        return SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=random_state,
            **kwargs,
        ).fit_predict(affinity)
    if method == "hdbscan":
        distance = proximity_to_distance(affinity, dense=True)
        return HDBSCAN(metric="precomputed", **kwargs).fit_predict(distance)
    if method == "kmeans":
        _require_n_clusters(n_clusters, method)
        kernel = affinity.toarray() if sparse.issparse(affinity) else affinity
        embedding = KernelPCA(
            n_components=n_clusters,
            kernel="precomputed",
        ).fit_transform(kernel)
        return KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            **kwargs,
        ).fit_predict(embedding)
    if method == "agglomerative":
        _require_n_clusters(n_clusters, method)
        distance = proximity_to_distance(affinity, dense=True)
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
            **kwargs,
        ).fit_predict(distance)
    raise ValueError(
        "method must be one of {'spectral', 'hdbscan', 'kmeans', 'agglomerative'}"
    )


def _require_n_clusters(n_clusters: int | None, method: str) -> None:
    if n_clusters is None:
        raise ValueError(f"n_clusters is required when method={method!r}")
