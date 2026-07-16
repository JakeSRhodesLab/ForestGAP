"""Native and declared-relation anomaly scores."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from forestgap.proximity import Proximity


def native_anomaly_scores(estimator: Any, X: Any) -> np.ndarray:
    """Return anomaly-oriented native scores (larger means more anomalous)."""

    estimator = estimator.estimator if isinstance(estimator, Proximity) else estimator
    score_samples = getattr(estimator, "score_samples", None)
    if not callable(score_samples):
        raise TypeError("estimator must expose native score_samples(X)")
    return -np.asarray(score_samples(X), dtype=float)


def outlier_scores(
    proximity: Proximity,
    y: Any = None,
    *,
    scaling: Literal["normalize", "log"] | None = None,
    method: Literal["breiman", "ene"] = "breiman",
) -> Any:
    """Score indexed rows from an explicitly named proximity relation."""

    return proximity._outlier_scores(y=y, scaling=scaling, method=method)
