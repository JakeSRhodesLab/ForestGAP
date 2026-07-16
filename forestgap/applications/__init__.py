"""Standalone consumers of :class:`forestgap.proximity.Proximity` relations."""

from .active_learning import select_active_batch
from .anomaly import native_anomaly_scores, outlier_scores
from .attribution import instance_attributions
from .clustering import proximity_clustering
from .conformal import split_conformal_classification, split_conformal_regression
from .feature_selection import neighborhood_feature_scores, select_features
from .imputation import impute
from .importance import local_feature_importance
from .prediction import ProximityKernelPredictor, knn_predict, prediction_identity
from .prototypes import get_prototypes
from .quantiles import (
    conditional_distribution_weights,
    conditional_quantiles,
    native_quantile_predict,
)
from .retrieval import retrieve_neighbors, retrieval_metrics
from .visualization import embed, embed_transform, plot_mds

__all__ = [
    "ProximityKernelPredictor",
    "conditional_distribution_weights",
    "conditional_quantiles",
    "embed",
    "embed_transform",
    "get_prototypes",
    "impute",
    "instance_attributions",
    "knn_predict",
    "local_feature_importance",
    "native_anomaly_scores",
    "native_quantile_predict",
    "neighborhood_feature_scores",
    "outlier_scores",
    "plot_mds",
    "prediction_identity",
    "proximity_clustering",
    "retrieve_neighbors",
    "retrieval_metrics",
    "select_active_batch",
    "select_features",
    "split_conformal_classification",
    "split_conformal_regression",
]
