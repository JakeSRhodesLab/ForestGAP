"""Split conformal prediction with strict calibration inputs."""

from __future__ import annotations

from typing import Any

import numpy as np


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    n = scores.size
    if n == 0:
        raise ValueError("calibration data must contain at least one sample")
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

    y = np.asarray(y_calibration, dtype=float).ravel()
    calibration = np.asarray(calibration_predictions, dtype=float).ravel()
    if y.shape != calibration.shape:
        raise ValueError("y_calibration and calibration_predictions must align")
    scores = np.abs(y - calibration)
    radius = _conformal_quantile(scores, alpha)
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

    y = np.asarray(y_calibration).ravel()
    probabilities = np.asarray(calibration_probabilities, dtype=float)
    test = np.asarray(test_probabilities, dtype=float)
    if probabilities.ndim != 2 or test.ndim != 2:
        raise ValueError("calibration_probabilities and test_probabilities must be two-dimensional")
    if probabilities.shape[0] != y.size:
        raise ValueError("y_calibration and calibration_probabilities must align")
    if probabilities.shape[1] != test.shape[1]:
        raise ValueError("calibration and test probabilities must have the same class columns")
    classes = np.asarray(np.unique(y) if classes is None else classes)
    if classes.ndim != 1 or classes.size != probabilities.shape[1]:
        raise ValueError("classes must contain one label for each probability column")
    lookup = {label: idx for idx, label in enumerate(classes)}
    try:
        indices = np.asarray([lookup[label] for label in y], dtype=int)
    except KeyError as exc:
        raise ValueError(f"calibration label {exc.args[0]!r} is absent from classes") from exc
    threshold = _conformal_quantile(1.0 - probabilities[np.arange(y.size), indices], alpha)
    return [classes[(1.0 - row) <= threshold] for row in test]
