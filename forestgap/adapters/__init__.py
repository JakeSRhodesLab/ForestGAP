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
from .boosting import (
    BOOSTED_FAMILIES,
    BoostedTreeAdapter,
    CatBoostAdapter,
    LightGBMAdapter,
    SklearnGradientBoostingAdapter,
    XGBoostAdapter,
    boosted_family,
    make_boosted_adapter,
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
from forestgap.rotation import RotationForestClassifier, RotationForestRegressor

patch_treeple_regressor_tags()
patch_treeple_sample_weight_signature()

__all__ = [
    "METHOD_REGISTRY",
    "BOOSTED_FAMILIES",
    "BoostedTreeAdapter",
    "CatBoostAdapter",
    "EstimatorCapabilities",
    "CCFConformance",
    "CanonicalCorrelationForestClassifier",
    "CanonicalCorrelationForestRegressor",
    "RotationForestClassifier",
    "RotationForestRegressor",
    "MethodCapability",
    "LightGBMAdapter",
    "SklearnGradientBoostingAdapter",
    "SyntheticNullForest",
    "UnsupportedRelationError",
    "XGBoostAdapter",
    "boosted_family",
    "capabilities_for",
    "censoring_safe_survival_metrics",
    "method_capabilities",
    "method_registry_rows",
    "make_boosted_adapter",
    "load_ccf_oracle_fixture",
    "native_quantiles",
    "native_survival_outputs",
    "proximity_counts",
    "patch_treeple_regressor_tags",
    "patch_treeple_sample_weight_signature",
    "validate_ccf_oracle",
    "validate_relation",
]
