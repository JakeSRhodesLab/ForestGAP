"""Conditional-distribution weights and quantile predictions."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import sparse

from forestgap.proximity import Proximity


def conditional_distribution_weights(
    proximity: Proximity, X: Any | None = None,
) -> sparse.csr_matrix:
    """Return row-normalized query-to-training conditional weights."""

    # A native quantile adapter may be attached with relation="conditional";
    # other declared leaf relations are also valid empirical donor weights.
    matrix = proximity.get_proximities() if X is None else proximity.prox_extend(X)
    weights = sparse.csr_matrix(matrix, dtype=float)
    row_sums = np.asarray(weights.sum(axis=1)).ravel()
    scale = np.zeros_like(row_sums)
    nonzero = row_sums > 0
    scale[nonzero] = 1.0 / row_sums[nonzero]
    return sparse.diags(scale) @ weights


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    keep = weights > 0
    if not np.any(keep):
        return np.full(quantiles.shape, np.nan)
    order = np.argsort(values[keep])
    sorted_values = values[keep][order]
    sorted_weights = weights[keep][order]
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, sorted_values)


def conditional_quantiles(
    proximity: Proximity,
    y: Any,
    quantiles: float | Sequence[float],
    *,
    X: Any | None = None,
) -> np.ndarray:
    """Evaluate the weighted empirical conditional distribution."""

    weights = conditional_distribution_weights(proximity, X)
    targets = np.asarray(y, dtype=float)
    qs = np.atleast_1d(np.asarray(quantiles, dtype=float))
    if np.any((qs <= 0) | (qs >= 1)):
        raise ValueError("quantiles must lie strictly between zero and one")
    result = np.vstack([
        _weighted_quantile(targets, weights.getrow(row).toarray().ravel(), qs)
        for row in range(weights.shape[0])
    ])
    return result[:, 0] if np.ndim(quantiles) == 0 else result


def native_quantile_predict(proximity: Proximity, X: Any, quantiles: Any) -> np.ndarray:
    """Call the optional quantile forest's native quantile API."""

    return np.asarray(proximity.estimator.predict(X, quantiles=quantiles))
