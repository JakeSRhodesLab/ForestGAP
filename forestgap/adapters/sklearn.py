"""scikit-learn forest capability helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def sklearn_leaf_ids(estimator: Any, X: Any) -> np.ndarray:
    """Return the standard sklearn ``(sample, tree)`` leaf matrix."""

    apply = getattr(estimator, "apply", None)
    if not callable(apply):
        raise TypeError("estimator does not expose apply(X)")
    leaves = np.asarray(apply(X))
    if leaves.ndim != 2:
        raise ValueError("apply(X) must return a two-dimensional leaf matrix")
    return leaves


def isolation_leaf_ids(estimator: Any, X: Any) -> np.ndarray:
    """Apply each isolation tree to its estimator-specific feature subset."""

    features_per_tree = getattr(estimator, "estimators_features_", None)
    trees = getattr(estimator, "estimators_", None)
    if features_per_tree is None or trees is None:
        raise TypeError("fitted isolation estimator lacks tree/feature inventory")
    return np.column_stack([
        tree.apply(np.asarray(X)[:, features])
        for tree, features in zip(trees, features_per_tree)
    ])


def sklearn_in_bag_counts(estimator: Any, n_samples: int) -> np.ndarray:
    """Build per-tree in-bag multiplicities from public sample inventories."""

    inventories = getattr(estimator, "estimators_samples_", None)
    if inventories is None:
        raise TypeError("estimator does not expose estimators_samples_")
    counts = np.zeros((n_samples, len(inventories)), dtype=np.float32)
    for tree, sampled in enumerate(inventories):
        indices, multiplicity = np.unique(np.asarray(sampled), return_counts=True)
        keep = (indices >= 0) & (indices < n_samples)
        counts[indices[keep], tree] = multiplicity[keep]
    return counts
