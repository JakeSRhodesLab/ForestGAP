"""
Manifold-learning applications for forest proximity matrices.

The primary user-facing API is available from fitted proximity-enabled forests:

    forest.embed(method="phate")
    forest.embed(method="tsne")
    forest.embed(method="isomap")
    forest.embed(method="umap")
"""

from forestgap.manifold.embedding import embed_from_forest, transform_from_forest
from forestgap.manifold.phate import fit_phate_from_forest, transform_phate_from_forest
from forestgap.manifold.proximity import (
    is_symmetric,
    prepare_proximity_affinity,
    proximity_to_distance,
    symmetrize_affinity,
)

__all__ = [
    "embed_from_forest",
    "transform_from_forest",
    "fit_phate_from_forest",
    "transform_phate_from_forest",
    "is_symmetric",
    "prepare_proximity_affinity",
    "proximity_to_distance",
    "symmetrize_affinity",
]
