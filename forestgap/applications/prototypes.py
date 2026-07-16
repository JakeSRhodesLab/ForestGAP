"""Class-constrained forest-space prototype selection."""

from __future__ import annotations

from typing import Any

from forestgap.proximity import Proximity


def get_prototypes(proximity: Proximity, *args: Any, **kwargs: Any) -> Any:
    """Select representative indexed observations and feature summaries."""

    return proximity._get_prototypes(*args, **kwargs)
