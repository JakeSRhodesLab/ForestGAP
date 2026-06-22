from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import normalize

from forestgap.manifold.proximity import prepare_proximity_affinity


def _require_phate() -> Any:
    try:
        import phate
    except ImportError as exc:
        raise ImportError(
            "PHATE support requires optional dependencies. Install with "
            "`pip install .[phate]` or `pip install .[manifold]`."
        ) from exc
    return phate


def _require_graphtools() -> Any:
    try:
        import graphtools
    except ImportError as exc:
        raise ImportError(
            "PHATE transform support requires graphtools. Install with "
            "`pip install .[phate]` or `pip install .[manifold]`."
        ) from exc
    return graphtools


def _make_pagerank_phate_class() -> type:
    phate = _require_phate()
    graphtools = _require_graphtools()

    class PageRankPHATE(phate.PHATE):
        """PHATE variant that adds PageRank-style teleportation."""

        def __init__(self, beta: float = 0.9, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.beta = beta

        @property
        def diff_op(self) -> np.ndarray:
            if self.graph is None:
                raise NotFittedError(
                    "This PageRankPHATE instance is not fitted yet. Call "
                    "`fit` before accessing the diffusion operator."
                )

            if isinstance(self.graph, graphtools.graphs.LandmarkGraph):
                diff_op = self.graph.landmark_op
            else:
                diff_op = self.graph.diff_op

            if sparse.issparse(diff_op):
                diff_op = diff_op.toarray()

            dim = diff_op.shape[0]
            teleport = np.full((dim, dim), 1.0 / dim, dtype=np.float64)
            return self.beta * diff_op + (1.0 - self.beta) * teleport

    return PageRankPHATE


def _ensure_manifold_storage(forest: Any) -> None:
    if not hasattr(forest, "manifold_models_"):
        forest.manifold_models_ = {}
    if not hasattr(forest, "manifold_embeddings_"):
        forest.manifold_embeddings_ = {}
    if not hasattr(forest, "manifold_metadata_"):
        forest.manifold_metadata_ = {}


def _default_random_state(forest: Any) -> Any:
    return getattr(forest, "random_state", None)


def fit_phate_from_forest(
    forest: Any,
    *,
    store_key: str = "phate",
    symmetrize: Optional[str] = None,
    symm_mode: Optional[str] = None,
    diagonal: Optional[str] = "one",
    pagerank_beta: Optional[float] = None,
    kernel_symm: Optional[str] = None,
    store: bool = True,
    phate_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Fit PHATE from proximities cached on an already fitted forest.

    The forest must implement ``get_proximities()`` and have already been fit
    so proximity structures are available.
    """

    if not hasattr(forest, "get_proximities"):
        raise TypeError("forest must provide get_proximities().")

    if symmetrize is None and symm_mode is not None:
        symmetrize = symm_mode

    P = forest.get_proximities()
    affinity = prepare_proximity_affinity(
        P,
        symmetrize=symmetrize,
        diagonal=diagonal,
    )

    operator_kwargs: Dict[str, Any] = {}
    if phate_kwargs is not None:
        operator_kwargs.update(phate_kwargs)
    operator_kwargs.update(kwargs)
    operator_kwargs.setdefault("n_components", 2)
    operator_kwargs.setdefault("knn_dist", "precomputed_affinity")
    operator_kwargs.setdefault("random_state", _default_random_state(forest))
    if kernel_symm is not None:
        operator_kwargs["kernel_symm"] = kernel_symm

    if pagerank_beta is None:
        phate = _require_phate()
        operator = phate.PHATE(**operator_kwargs)
    else:
        if not (0.0 < pagerank_beta <= 1.0):
            raise ValueError("pagerank_beta must be in the interval (0, 1].")
        PageRankPHATE = _make_pagerank_phate_class()
        operator = PageRankPHATE(beta=float(pagerank_beta), **operator_kwargs)

    embedding = operator.fit_transform(affinity)
    embedding = np.asarray(embedding)

    if store:
        _ensure_manifold_storage(forest)
        forest.manifold_models_[store_key] = operator
        forest.manifold_embeddings_[store_key] = embedding
        forest.manifold_metadata_[store_key] = {
            "method": "phate",
            "symmetrize": symmetrize,
            "diagonal": diagonal,
            "pagerank_beta": pagerank_beta,
            "kernel_symm": kernel_symm,
        }

    return embedding


def transform_phate_from_forest(
    forest: Any,
    X_new: Any,
    *,
    model_key: str = "phate",
    diagonal: Optional[str] = "preserve",
) -> np.ndarray:
    """Project new observations through a PHATE model fitted by proximity."""

    if not hasattr(forest, "manifold_models_") or model_key not in forest.manifold_models_:
        raise NotFittedError(
            "No fitted PHATE manifold model is stored on this forest. "
            "Call forest.embed(method='phate') first."
        )
    if not hasattr(forest, "prox_extend"):
        raise TypeError("forest must provide prox_extend(X_new).")

    operator = forest.manifold_models_[model_key]
    graph = getattr(operator, "graph", None)
    embedding = getattr(operator, "embedding", None)
    if graph is None or embedding is None:
        raise NotFittedError("The stored PHATE operator is missing graph or embedding state.")

    kernel = forest.prox_extend(X_new)
    kernel = prepare_proximity_affinity(
        kernel,
        symmetrize=None,
        diagonal=diagonal,
    )

    graphtools = _require_graphtools()
    if isinstance(graph, graphtools.graphs.LandmarkGraph):
        clusters = graph.clusters
        n_train_points = kernel.shape[1]
        unique_clusters, remapped_clusters = np.unique(clusters, return_inverse=True)
        row_idx = np.arange(n_train_points)
        cluster_map = sparse.csc_matrix(
            (
                np.ones(n_train_points, dtype=np.float64),
                (row_idx, remapped_clusters),
            ),
            shape=(n_train_points, len(unique_clusters)),
        )
        pnm = kernel @ cluster_map
        pnm = normalize(pnm, norm="l1", axis=1)
    else:
        pnm = normalize(kernel, norm="l1", axis=1)

    return np.asarray(graph.interpolate(embedding, pnm))
