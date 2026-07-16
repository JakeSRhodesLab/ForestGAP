"""Prediction identity, precomputed-kernel prediction, and proximity kNN."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import sparse

from forestgap.manifold.proximity import prepare_proximity_affinity
from forestgap.proximity import Proximity


def _dense(matrix: Any) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def prediction_identity(proximity: Proximity, X: Any | None = None) -> np.ndarray:
    """Reconstruct native predictions for RF-GAP or Honest-GAP."""

    return proximity.prox_predict(X)


class ProximityKernelPredictor:
    """Small ridge predictor operating on a symmetric forest kernel."""

    def __init__(self, proximity: Proximity, *, alpha: float = 1e-6) -> None:
        self.proximity = proximity
        self.alpha = alpha

    def fit(self, y: Any) -> "ProximityKernelPredictor":
        kernel = _dense(prepare_proximity_affinity(
            self.proximity.get_proximities(), symmetrize="auto", diagonal="one",
        ))
        self.classes_ = None
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if self.proximity.capabilities_ and self.proximity.capabilities_.task_type == "classification":
            self.classes_, indices = np.unique(y_arr, return_inverse=True)
            targets = np.eye(len(self.classes_))[indices]
        else:
            targets = y_arr.astype(float)
        self.dual_coef_ = np.linalg.solve(
            kernel + self.alpha * np.eye(kernel.shape[0]), targets,
        )
        return self

    def predict(self, X: Any | None = None) -> np.ndarray:
        if not hasattr(self, "dual_coef_"):
            raise RuntimeError("Call fit(y) before predict().")
        kernel = self.proximity.get_proximities() if X is None else self.proximity.prox_extend(X)
        values = _dense(kernel) @ self.dual_coef_
        if self.classes_ is not None:
            return self.classes_[np.argmax(values, axis=1)]
        return np.asarray(values).ravel()


def knn_predict(
    proximity: Proximity,
    y: Any,
    *,
    X: Any | None = None,
    n_neighbors: int = 5,
    task: Literal["classification", "regression"] | None = None,
) -> np.ndarray:
    """Predict from the strongest query-to-training relation entries."""

    weights = _dense(proximity.get_proximities() if X is None else proximity.prox_extend(X))
    y_arr = np.asarray(y)
    if task is None:
        task = "classification" if proximity.capabilities_ and proximity.capabilities_.task_type == "classification" else "regression"
    k = min(max(1, int(n_neighbors)), weights.shape[1])
    top = np.argpartition(weights, -k, axis=1)[:, -k:]
    output: list[Any] = []
    for row, indices in enumerate(top):
        row_weights = weights[row, indices]
        if task == "regression":
            denom = row_weights.sum()
            output.append(float(np.average(y_arr[indices], weights=row_weights)) if denom > 0 else float(np.mean(y_arr[indices])))
        else:
            labels, inverse = np.unique(y_arr[indices], return_inverse=True)
            votes = np.bincount(inverse, weights=row_weights, minlength=len(labels))
            output.append(labels[int(np.argmax(votes))])
    return np.asarray(output)
