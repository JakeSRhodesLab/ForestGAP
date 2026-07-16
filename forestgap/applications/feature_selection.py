"""Proximity-neighborhood feature selection primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from forestgap.proximity import Proximity


def neighborhood_feature_scores(proximity: Proximity, X: Any) -> np.ndarray:
    """Score features by global variance explained by local neighbor agreement."""

    features = np.asarray(X, dtype=float)
    relation = proximity.get_proximities()
    weights = sparse.csr_matrix(relation, dtype=float)
    weights.setdiag(0.0)
    row_sum = np.asarray(weights.sum(axis=1)).ravel()
    scale = np.zeros_like(row_sum)
    scale[row_sum > 0] = 1.0 / row_sum[row_sum > 0]
    local_mean = (sparse.diags(scale) @ weights) @ features
    residual_var = np.nanvar(features - local_mean, axis=0)
    total_var = np.nanvar(features, axis=0)
    return 1.0 - np.divide(residual_var, total_var, out=np.zeros_like(total_var), where=total_var > 0)


def select_features(proximity: Proximity, X: Any, *, k: int) -> np.ndarray:
    """Return indices of the top-k proximity-stable features."""

    scores = neighborhood_feature_scores(proximity, X)
    k = min(max(1, int(k)), scores.size)
    return np.argsort(scores)[::-1][:k]
