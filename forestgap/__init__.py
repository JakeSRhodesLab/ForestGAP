"""
forestgap: Sparse proximity computations for random forest-like ensembles.

This package provides a mixin and factory pattern for adding sparse proximity
computations to forest-based estimators from scikit-learn and aeon.

Main exports:
    - SparseProximityMixin: Mixin class for sparse proximity computation
    - with_sparse_proximity: Decorator/factory for creating proximity-enabled estimators
    - _RotationForestApplyAdapter: Adapter for aeon's RotationForest models

Manifold applications are available from fitted proximity-enabled forests via
``forest.embed(...)`` and from the ``forestgap.manifold`` namespace.
"""

from forestgap.core import (
    SparseProximityMixin,
    with_sparse_proximity,
    _RotationForestApplyAdapter,
)
from forestgap.datasets import (
    OpenMLBenchmarkSuite,
    OpenMLDataset,
    SIMULATION_DATASET_GENERATORS,
    SyntheticDataset,
    load_openml_dataset,
    load_openml_suite,
    load_simulated_dataset,
    make_additive_interaction_correlated,
    make_diagonal_band,
    make_dual_projection_xor,
    make_oblique_gaussian_clusters,
    make_oblique_slab,
    make_rotated_linear_boundary,
    make_rotated_subspace_signal,
    make_rotated_xor,
    make_sparse_oblique_hyperplane,
)

__version__ = "0.1.0"
__all__ = [
    "SparseProximityMixin",
    "with_sparse_proximity",
    "_RotationForestApplyAdapter",
    "OpenMLBenchmarkSuite",
    "OpenMLDataset",
    "SyntheticDataset",
    "SIMULATION_DATASET_GENERATORS",
    "load_openml_dataset",
    "load_openml_suite",
    "load_simulated_dataset",
    "make_rotated_linear_boundary",
    "make_rotated_xor",
    "make_oblique_gaussian_clusters",
    "make_rotated_subspace_signal",
    "make_additive_interaction_correlated",
    "make_diagonal_band",
    "make_sparse_oblique_hyperplane",
    "make_oblique_slab",
    "make_dual_projection_xor",
]
