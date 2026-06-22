from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.exceptions import NotFittedError

from forestgap.manifold.phate import fit_phate_from_forest, transform_phate_from_forest
from forestgap.manifold.proximity import (
    prepare_proximity_affinity,
    proximity_to_distance,
)


_SUPPORTED_METHODS = {"phate", "tsne", "t-sne", "isomap", "umap"}


def _normalize_method(method: str) -> str:
    normalized = method.lower().replace("_", "-")
    if normalized == "t-sne":
        normalized = "tsne"
    if normalized not in _SUPPORTED_METHODS:
        raise ValueError(
            "method must be one of {'phate', 'tsne', 'isomap', 'umap'}."
        )
    return normalized


def _ensure_manifold_storage(forest: Any) -> None:
    if not hasattr(forest, "manifold_models_"):
        forest.manifold_models_ = {}
    if not hasattr(forest, "manifold_embeddings_"):
        forest.manifold_embeddings_ = {}
    if not hasattr(forest, "manifold_metadata_"):
        forest.manifold_metadata_ = {}


def _default_random_state(forest: Any) -> Any:
    return getattr(forest, "random_state", None)


def embed_from_forest(
    forest: Any,
    *,
    method: str = "phate",
    store_key: Optional[str] = None,
    symmetrize: Optional[str] = None,
    symm_mode: Optional[str] = None,
    diagonal: Optional[str] = "one",
    distance_method: str = "sqrt_one_minus",
    dense_distance: bool = True,
    sparse_mode: str = "raise",
    store: bool = True,
    estimator_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Create a manifold embedding from a fitted proximity-enabled forest."""

    method_name = _normalize_method(method)
    key = store_key or method_name

    if method_name == "phate":
        return fit_phate_from_forest(
            forest,
            store_key=key,
            symmetrize=symmetrize,
            symm_mode=symm_mode,
            diagonal=diagonal,
            store=store,
            **kwargs,
        )

    if not hasattr(forest, "get_proximities"):
        raise TypeError("forest must provide get_proximities().")

    if symmetrize is None:
        symmetrize = symm_mode if symm_mode is not None else "auto"

    P = forest.get_proximities()
    affinity = prepare_proximity_affinity(
        P,
        symmetrize=symmetrize,
        diagonal=diagonal,
    )
    distances = proximity_to_distance(
        affinity,
        method=distance_method,
        dense=dense_distance,
        sparse_mode=sparse_mode,
    )

    params: Dict[str, Any] = {}
    if estimator_kwargs is not None:
        params.update(estimator_kwargs)
    params.update(kwargs)

    estimator = _make_precomputed_estimator(method_name, forest, params)
    embedding = np.asarray(estimator.fit_transform(distances))

    if store:
        _ensure_manifold_storage(forest)
        forest.manifold_models_[key] = estimator
        forest.manifold_embeddings_[key] = embedding
        forest.manifold_metadata_[key] = {
            "method": method_name,
            "symmetrize": symmetrize,
            "diagonal": diagonal,
            "distance_method": distance_method,
            "dense_distance": dense_distance,
            "sparse_mode": sparse_mode,
        }

    return embedding


def transform_from_forest(
    forest: Any,
    X_new: Any,
    *,
    method: str = "phate",
    model_key: Optional[str] = None,
    distance_method: str = "sqrt_one_minus",
    dense_distance: bool = True,
    sparse_mode: str = "raise",
    **kwargs: Any,
) -> np.ndarray:
    """Transform new observations through a stored manifold model when possible."""

    method_name = _normalize_method(method)
    key = model_key or method_name

    if method_name == "phate":
        return transform_phate_from_forest(forest, X_new, model_key=key, **kwargs)

    if method_name == "tsne":
        raise NotImplementedError("sklearn TSNE does not provide out-of-sample transform.")
    if method_name == "umap":
        raise NotImplementedError(
            "UMAP transform from precomputed forest distances is not enabled yet."
        )
    if method_name != "isomap":
        raise ValueError("Only PHATE and Isomap have supported transform paths.")

    if not hasattr(forest, "manifold_models_") or key not in forest.manifold_models_:
        raise NotFittedError(
            "No fitted Isomap manifold model is stored on this forest. "
            "Call forest.embed(method='isomap') first."
        )
    if not hasattr(forest, "prox_extend"):
        raise TypeError("forest must provide prox_extend(X_new).")

    metadata = getattr(forest, "manifold_metadata_", {}).get(key, {})
    diagonal = metadata.get("diagonal", "preserve")

    affinity = prepare_proximity_affinity(
        forest.prox_extend(X_new),
        symmetrize=None,
        diagonal=diagonal if diagonal in (None, "preserve") else "preserve",
    )
    distances = proximity_to_distance(
        affinity,
        method=distance_method,
        dense=dense_distance,
        sparse_mode=sparse_mode,
    )
    return np.asarray(forest.manifold_models_[key].transform(distances))


def _make_precomputed_estimator(method: str, forest: Any, params: Dict[str, Any]) -> Any:
    if method == "tsne":
        from sklearn.manifold import TSNE

        params.setdefault("n_components", 2)
        params.setdefault("metric", "precomputed")
        params.setdefault("init", "random")
        params.setdefault("random_state", _default_random_state(forest))
        return TSNE(**params)

    if method == "isomap":
        from sklearn.manifold import Isomap

        params.setdefault("n_components", 2)
        params.setdefault("metric", "precomputed")
        return Isomap(**params)

    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "UMAP support requires optional dependencies. Install with "
                "`pip install .[umap]` or `pip install .[manifold]`."
            ) from exc

        params.setdefault("n_components", 2)
        params.setdefault("metric", "precomputed")
        params.setdefault("random_state", _default_random_state(forest))
        return umap.UMAP(**params)

    raise ValueError(f"Unsupported manifold method: {method}.")
