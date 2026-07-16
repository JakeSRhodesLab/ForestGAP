"""Query-to-index retrieval and generic relevance metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from forestgap.proximity import Proximity


def retrieve_neighbors(proximity: Proximity, X: Any, *, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return strongest training indices and relation scores per query."""

    matrix = proximity.prox_extend(X)
    scores = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    k = min(max(1, int(k)), scores.shape[1])
    indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    local_scores = np.take_along_axis(scores, indices, axis=1)
    order = np.argsort(local_scores, axis=1)[:, ::-1]
    return np.take_along_axis(indices, order, axis=1), np.take_along_axis(local_scores, order, axis=1)


def retrieval_metrics(retrieved: Any, relevant: list[set[int]]) -> dict[str, float]:
    """Compute macro Recall@k, average precision, and binary nDCG@k."""

    retrieved = np.asarray(retrieved, dtype=int)
    recalls, aps, ndcgs = [], [], []
    for row, truth in zip(retrieved, relevant):
        hits = np.asarray([int(index in truth) for index in row], dtype=float)
        recalls.append(float(hits.sum() / len(truth)) if truth else 0.0)
        precision = np.cumsum(hits) / np.arange(1, hits.size + 1)
        aps.append(float((precision * hits).sum() / min(len(truth), hits.size)) if truth else 0.0)
        discounts = 1.0 / np.log2(np.arange(2, hits.size + 2))
        ideal = np.sort(hits)[::-1]
        denom = float(np.sum(ideal * discounts))
        ndcgs.append(float(np.sum(hits * discounts) / denom) if denom else 0.0)
    return {"recall_at_k": float(np.mean(recalls)), "map_at_k": float(np.mean(aps)), "ndcg_at_k": float(np.mean(ndcgs))}
