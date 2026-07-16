"""Estimator capability discovery and optional forest adapters.

The adapter layer is deliberately declarative.  A forest family is allowed to
construct a relation only when its :class:`EstimatorCapabilities` say that the
required estimator contract is present.  This prevents an estimator that just
happens to expose ``estimators_`` from accidentally receiving RF-GAP semantics.
"""

from .registry import (
    METHOD_REGISTRY,
    EstimatorCapabilities,
    MethodCapability,
    UnsupportedRelationError,
    capabilities_for,
    method_capabilities,
    method_registry_rows,
    validate_relation,
)
from .synthetic_null import SyntheticNullForest
from .ccf import (
    CCFConformance,
    CanonicalCorrelationForestClassifier,
    CanonicalCorrelationForestRegressor,
    load_ccf_oracle_fixture,
    validate_ccf_oracle,
)
from .quantile import native_quantiles, proximity_counts
from .survival import censoring_safe_survival_metrics, native_survival_outputs
from .treeple import patch_treeple_regressor_tags, patch_treeple_sample_weight_signature

patch_treeple_regressor_tags()
patch_treeple_sample_weight_signature()

__all__ = [
    "METHOD_REGISTRY",
    "EstimatorCapabilities",
    "CCFConformance",
    "CanonicalCorrelationForestClassifier",
    "CanonicalCorrelationForestRegressor",
    "MethodCapability",
    "SyntheticNullForest",
    "UnsupportedRelationError",
    "capabilities_for",
    "censoring_safe_survival_metrics",
    "method_capabilities",
    "method_registry_rows",
    "load_ccf_oracle_fixture",
    "native_quantiles",
    "native_survival_outputs",
    "proximity_counts",
    "patch_treeple_regressor_tags",
    "patch_treeple_sample_weight_signature",
    "validate_ccf_oracle",
    "validate_relation",
]
