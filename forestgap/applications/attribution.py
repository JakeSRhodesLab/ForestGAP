"""Training-instance attribution from directed predictive weights."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from forestgap.proximity import Proximity


def instance_attributions(
    proximity: Proximity,
    X: Any,
    *,
    target_values: Any | None = None,
    top_k: int | None = None,
) -> dict[str, np.ndarray]:
    """Return predictive training weights and optional signed contributions."""

    if proximity.relation not in {"rfgap", "honest_gap", "conditional"}:
        raise ValueError("instance attribution requires a directed predictive relation")
    matrix = proximity.prox_extend(X)
    weights = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    contributions = weights if target_values is None else weights * np.asarray(target_values)[None, :]
    order = np.argsort(np.abs(contributions), axis=1)[:, ::-1]
    if top_k is not None:
        order = order[:, :max(1, int(top_k))]
    return {
        "indices": order,
        "weights": np.take_along_axis(weights, order, axis=1),
        "contributions": np.take_along_axis(contributions, order, axis=1),
    }
