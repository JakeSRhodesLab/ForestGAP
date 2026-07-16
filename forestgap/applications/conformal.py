"""Split conformal prediction with strict calibration inputs."""

from __future__ import annotations

from typing import Any

import numpy as np


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    n = scores.size
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def split_conformal_regression(
    y_calibration: Any,
    calibration_predictions: Any,
    test_predictions: Any,
    *,
    alpha: float = 0.1,
) -> np.ndarray:
    """Return marginal split-conformal regression intervals."""

    scores = np.abs(np.asarray(y_calibration) - np.asarray(calibration_predictions))
    radius = _conformal_quantile(scores.ravel(), alpha)
    predictions = np.asarray(test_predictions, dtype=float).ravel()
    return np.column_stack([predictions - radius, predictions + radius])


def split_conformal_classification(
    y_calibration: Any,
    calibration_probabilities: Any,
    test_probabilities: Any,
    *,
    classes: Any | None = None,
    alpha: float = 0.1,
) -> list[np.ndarray]:
    """Return label sets from inverse-probability nonconformity scores."""

    y = np.asarray(y_calibration)
    probabilities = np.asarray(calibration_probabilities, dtype=float)
    classes = np.asarray(np.unique(y) if classes is None else classes)
    lookup = {label: idx for idx, label in enumerate(classes)}
    indices = np.asarray([lookup[label] for label in y], dtype=int)
    threshold = _conformal_quantile(1.0 - probabilities[np.arange(y.size), indices], alpha)
    test = np.asarray(test_probabilities, dtype=float)
    return [classes[(1.0 - row) <= threshold] for row in test]
