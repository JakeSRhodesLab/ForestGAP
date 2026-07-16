"""Local feature importance from proximity neighborhoods."""

from __future__ import annotations

from typing import Any

from forestgap.proximity import Proximity


def local_feature_importance(proximity: Proximity, *args: Any, **kwargs: Any) -> Any:
    """Compute the legacy validated local perturbation statistic standalone."""

    return proximity._local_feature_importance(*args, **kwargs)
