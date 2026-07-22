"""Label-free feature selection from a fitted proximity neighborhood graph."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from forestgap.proximity import Proximity


def neighborhood_feature_scores(proximity: Proximity, X: Any) -> np.ndarray:
    """Score columns by how well proximity-weighted neighbors reconstruct them.

    After excluding self-relations, each row is reconstructed from its
    row-normalized relation weights. The returned score for feature ``r`` is
    ``1 - Var(X[:, r] - local_mean[:, r]) / Var(X[:, r])``: high values denote
    locally stable features, while negative values denote poor reconstruction.
    ``X`` must contain the training rows in the same order as ``proximity``;
    directed relations use their outgoing row weights. Constant columns score
    zero.
    """

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
    """Return indices of the ``k`` highest-scoring proximity-stable features.

    This is a label-free ranking convenience over
    :func:`neighborhood_feature_scores`; ``k`` is clipped to ``[1, n_features]``.
    Perform selection within each training fold when evaluating a downstream
    model.
    """

    scores = neighborhood_feature_scores(proximity, X)
    k = min(max(1, int(k)), scores.size)
    return np.argsort(scores)[::-1][:k]
