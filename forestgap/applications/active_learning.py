"""Leakage-safe batch acquisition primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse


def select_active_batch(
    uncertainty: Any,
    pool_affinity: Any,
    batch_size: int,
    *,
    diversity_weight: float = 0.5,
) -> np.ndarray:
    """Greedily combine label-free uncertainty with within-pool diversity."""

    uncertainty = np.asarray(uncertainty, dtype=float).ravel()
    affinity = pool_affinity.toarray() if sparse.issparse(pool_affinity) else np.asarray(pool_affinity, dtype=float)
    if affinity.shape != (uncertainty.size, uncertainty.size):
        raise ValueError("pool_affinity must be square and align with uncertainty")
    batch_size = min(max(1, int(batch_size)), uncertainty.size)
    selected = [int(np.argmax(uncertainty))]
    while len(selected) < batch_size:
        redundancy = affinity[:, selected].max(axis=1)
        score = uncertainty + diversity_weight * (1.0 - redundancy)
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))
    return np.asarray(selected, dtype=int)
