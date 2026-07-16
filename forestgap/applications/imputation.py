"""Proximity-weighted missing-value imputation."""

from __future__ import annotations

from typing import Any, Sequence

from forestgap.proximity import Proximity


def impute(
    proximity: Proximity,
    X: Any,
    categorical_features: Sequence[int] | None = None,
    X_test: Any | None = None,
    initialization: str = "median",
) -> Any:
    """Impute training and optional test features using indexed donor weights."""

    return proximity._impute(
        X,
        categorical_features=categorical_features,
        X_test=X_test,
        initialization=initialization,
    )
