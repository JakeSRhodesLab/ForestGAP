"""aeon RotationForest leaf adapter."""

from __future__ import annotations

from typing import Any

import numpy as np


def rotation_forest_leaf_ids(estimator: Any, X: Any) -> np.ndarray:
    """Reapply stored rotations and return terminal nodes per tree."""

    X_proc = estimator._check_X(X)
    validate = getattr(estimator, "_validate_data", None)
    if callable(validate):
        X_proc = validate(X=X_proc, reset=False, accept_sparse=False)
    X_proc = X_proc[:, estimator._useful_atts]
    X_proc = (X_proc - estimator._min) / estimator._ptp
    leaves = []
    for tree, pcas, groups in zip(estimator.estimators_, estimator._pcas, estimator._groups):
        transformed = np.concatenate(
            [pcas[index].transform(X_proc[:, group]) for index, group in enumerate(groups)],
            axis=1,
        ).astype(np.float32)
        transformed = np.nan_to_num(transformed)
        leaves.append(tree.apply(transformed))
    return np.column_stack(leaves)
