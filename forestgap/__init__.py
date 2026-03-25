"""
forestgap: Sparse proximity computations for random forest-like ensembles.

This package provides a mixin and factory pattern for adding sparse proximity
computations to forest-based estimators from scikit-learn and aeon.

Main exports:
    - SparseProximityMixin: Mixin class for sparse proximity computation
    - with_sparse_proximity: Decorator/factory for creating proximity-enabled estimators
    - _RotationForestApplyAdapter: Adapter for aeon's RotationForest models
"""

from forestgap.core import (
    SparseProximityMixin,
    with_sparse_proximity,
    _RotationForestApplyAdapter,
)
from forestgap.datasets import (
    OpenMLBenchmarkSuite,
    OpenMLDataset,
    load_openml_dataset,
    load_openml_suite,
)

__version__ = "0.1.0"
__all__ = [
    "SparseProximityMixin",
    "with_sparse_proximity",
    "_RotationForestApplyAdapter",
    "OpenMLBenchmarkSuite",
    "OpenMLDataset",
    "load_openml_dataset",
    "load_openml_suite",
]
