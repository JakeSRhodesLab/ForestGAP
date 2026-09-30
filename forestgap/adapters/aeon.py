"""RotationForest transformation adapters.

aeon deliberately does not persist ``estimators_samples_`` for RotationForest.
The OOB/RF-GAP adapter in :mod:`forestgap.proximity` reconstructs its private
training-estimate resampling protocol; this module owns the public-facing
rotation, leaf, and path transformations shared by that adapter.  Forestgap's
native implementation exposes a compiled transformation fast path while
retaining this same relation contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse


def rotation_forest_transforms(estimator: Any, X: Any) -> list[np.ndarray]:
    """Return each fitted RotationForest tree's rotated feature matrix."""

    transform_all = getattr(estimator, "_transform_all", None)
    if callable(transform_all):
        return [np.asarray(values) for values in transform_all(X)]

    X_proc = estimator._check_X(X)
    validate = getattr(estimator, "_validate_data", None)
    if callable(validate):
        X_proc = validate(X=X_proc, reset=False, accept_sparse=False)
    X_proc = X_proc[:, estimator._useful_atts]
    X_proc = (X_proc - estimator._min) / estimator._ptp

    transformed: list[np.ndarray] = []
    for pcas, groups in zip(estimator._pcas, estimator._groups):
        X_tree = np.concatenate(
            [pcas[index].transform(X_proc[:, group]) for index, group in enumerate(groups)],
            axis=1,
        ).astype(np.float32)
        transformed.append(np.nan_to_num(X_tree))
    return transformed


def rotation_forest_leaf_ids(estimator: Any, X: Any) -> np.ndarray:
    """Reapply stored rotations and return terminal nodes per tree."""

    transformed = rotation_forest_transforms(estimator, X)
    leaves = [tree.apply(X_tree) for tree, X_tree in zip(estimator.estimators_, transformed)]
    return np.column_stack(leaves)


def rotation_forest_decision_path(
    estimator: Any, X: Any,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return concatenated base-tree paths and sklearn-style node offsets."""

    transformed = rotation_forest_transforms(estimator, X)
    paths = []
    for tree, X_tree in zip(estimator.estimators_, transformed):
        decision_path = getattr(tree, "decision_path", None)
        if not callable(decision_path):
            raise TypeError("RotationForest base tree does not expose decision_path(X)")
        paths.append(sparse.csr_matrix(decision_path(X_tree)))
    offsets = np.cumsum([0] + [path.shape[1] for path in paths], dtype=np.int64)
    return sparse.hstack(paths, format="csr"), offsets


def rotation_forest_node_depths(estimator: Any) -> list[np.ndarray]:
    """Return per-tree node depths for fitted RotationForest base trees."""

    depths: list[np.ndarray] = []
    for tree in estimator.estimators_:
        tree_state = getattr(tree, "tree_", None)
        if tree_state is None:
            raise TypeError("RotationForest base tree does not expose sklearn tree_ state")
        children_left = np.asarray(tree_state.children_left)
        children_right = np.asarray(tree_state.children_right)
        tree_depths = np.zeros(tree_state.node_count, dtype=np.int32)
        stack = [0]
        while stack:
            node = stack.pop()
            for child in (children_left[node], children_right[node]):
                if child >= 0:
                    tree_depths[child] = tree_depths[node] + 1
                    stack.append(int(child))
        depths.append(tree_depths)
    return depths
