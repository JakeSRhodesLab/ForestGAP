"""Capability registry for supported, optional, and scientifically gated methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from typing import Any, Mapping

from .boosting import BOOSTED_FAMILIES, boosted_family, make_boosted_adapter


class UnsupportedRelationError(ValueError):
    """Raised when an estimator does not satisfy a relation's validity contract."""

    def __init__(self, family: str, relation: str, reason: str) -> None:
        self.family = family
        self.relation = relation
        self.reason = reason
        super().__init__(f"{family!r} does not support relation {relation!r}: {reason}")


@dataclass(frozen=True)
class MethodCapability:
    """Static roadmap row for an estimator family."""

    family: str
    status: str
    task_types: tuple[str, ...]
    valid_relations: tuple[str, ...] = ()
    optional_dependency: str | None = None
    reason: str = ""

    @property
    def available(self) -> bool:
        if self.optional_dependency is None:
            return True
        import_name = {
            "scikit-survival": "sksurv",
        }.get(self.optional_dependency, self.optional_dependency.replace("-", "_"))
        return find_spec(import_name) is not None


@dataclass(frozen=True)
class EstimatorCapabilities:
    """Runtime estimator contract used by relation dispatch."""

    family: str
    task_type: str
    leaf_ids: bool
    tree_inventory: bool
    decision_paths: bool
    node_depths: bool
    bootstrap: bool
    in_bag: bool
    oob: bool
    honest_roles: bool
    contribution_weights: bool
    valid_relations: tuple[str, ...]
    status: str = "available"
    optional_dependency: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["valid_relations"] = list(self.valid_relations)
        return result


# Every non-reference estimator family in notes/application_roadmap.md has a
# visible row, including work that is intentionally gated.
METHOD_REGISTRY: Mapping[str, MethodCapability] = {
    "random_forest": MethodCapability(
        "random_forest", "available", ("classification", "regression"),
        ("original", "oob", "rfgap", "kerf", "path", "englund", "random_height"),
    ),
    "extra_trees": MethodCapability(
        "extra_trees", "available", ("classification", "regression"),
        ("original", "oob", "rfgap", "kerf", "path", "englund", "random_height"),
    ),
    "rotation_forest": MethodCapability(
        "rotation_forest", "optional", ("classification", "regression"),
        ("original", "oob", "rfgap", "kerf", "path", "englund", "random_height"), "aeon",
        "OOB/RF-GAP require aeon's synchronized training-estimate contract.",
    ),
    "treeple_oblique_forest": MethodCapability(
        "treeple_oblique_forest", "optional", ("classification", "regression"),
        ("original", "oob", "rfgap", "kerf", "path", "englund", "random_height"), "treeple",
    ),
    "honest_forest": MethodCapability(
        "honest_forest", "optional", ("classification",),
        ("original", "honest_gap", "kerf", "path", "englund", "random_height"), "treeple",
        "Prediction identity is valid only for honest_gap.",
    ),
    "isolation_forest": MethodCapability(
        "isolation_forest", "native_only", ("anomaly",), ("collision",), None,
        "Use score_samples for anomaly scoring; collision is a distinct partition relation.",
    ),
    "extended_isolation_forest": MethodCapability(
        "extended_isolation_forest", "optional_native_only", ("anomaly",),
        ("collision",), "treeple",
        "Use the native path score; collision is not RF-GAP.",
    ),
    "canonical_correlation_forest": MethodCapability(
        "canonical_correlation_forest", "available", ("classification", "regression"),
        ("original", "oob", "rfgap", "kerf", "path", "englund", "random_height"), None,
        "OOB and RF-GAP are available only for CCF-Bag (bootstrap=True).",
    ),
    "quantile_forest": MethodCapability(
        "quantile_forest", "optional", ("quantile_regression",),
        ("conditional", "original", "kerf", "path", "englund", "random_height"),
        "quantile_forest",
    ),
    "random_survival_forest": MethodCapability(
        "random_survival_forest", "optional", ("survival",),
        ("survival_leaf",), "scikit-survival",
    ),
    "synthetic_null_urf": MethodCapability(
        "synthetic_null_urf", "available", ("anomaly", "clustering"),
        ("original", "oob", "kerf", "path", "englund", "random_height"),
    ),
    "gradient_boosting": MethodCapability(
        "gradient_boosting", "available", ("classification", "regression"),
        ("original", "kerf", "boosted", "path", "englund", "random_height"), None,
        "OOB and RF-GAP are invalid because boosting lacks exchangeable bootstrap roles.",
    ),
    "xgboost": MethodCapability(
        "xgboost", "optional", ("classification", "regression"),
        ("original", "kerf", "boosted", "path", "englund", "random_height"), "xgboost",
        "OOB and RF-GAP are invalid because boosting lacks exchangeable bootstrap roles.",
    ),
    "lightgbm": MethodCapability(
        "lightgbm", "optional", ("classification", "regression"),
        ("original", "kerf", "boosted", "path", "englund", "random_height"), "lightgbm",
        "OOB and RF-GAP are invalid because boosting lacks exchangeable bootstrap roles.",
    ),
    "catboost": MethodCapability(
        "catboost", "optional", ("classification", "regression"),
        ("original", "kerf", "boosted", "path", "englund", "random_height"), "catboost",
        "Path-derived relations additionally require grow_policy='SymmetricTree'; "
        "OOB and RF-GAP are invalid for boosting.",
    ),
    "boosted_trees": MethodCapability(
        "boosted_trees", "gated", ("classification", "regression"), (), None,
        "This boosted family has no implemented fitted-contribution adapter; ordinary RF-GAP is invalid.",
    ),
    "causal_forest": MethodCapability(
        "causal_forest", "gated", ("causal",), (), None,
        "Requires declared treatment/outcome roles, honesty masks, and an estimand protocol.",
    ),
    "generalized_random_forest": MethodCapability(
        "generalized_random_forest", "gated", ("causal", "local_moments"), (), None,
        "Requires honest local-moment weights and leakage protocol.",
    ),
    "bayesian_trees": MethodCapability(
        "bayesian_trees", "gated", ("classification", "regression"), (), None,
        "Posterior co-clustering semantics have not been specified.",
    ),
    "structured_pf_gap": MethodCapability(
        "structured_pf_gap", "gated", ("structured", "time_series"), (), None,
        "Requires an object metric and structure-preserving evaluation.",
    ),
    "multiview_forest": MethodCapability(
        "multiview_forest", "gated", ("multiview",), (), None,
        "View pairing and missing-view relation semantics are not implemented.",
    ),
    "graph_forest": MethodCapability(
        "graph_forest", "gated", ("graph",), (), None,
        "Inductive/transductive graph construction protocol is not implemented.",
    ),
    "autoencoding_forest": MethodCapability(
        "autoencoding_forest", "gated", ("representation",), (), None,
        "Out-of-sample encoder and reconstruction relation semantics are not implemented.",
    ),
}


def method_capabilities(family: str) -> MethodCapability:
    """Return a static roadmap capability row."""

    try:
        return METHOD_REGISTRY[family]
    except KeyError as exc:
        raise KeyError(f"Unknown forest method family: {family!r}") from exc


def method_registry_rows() -> list[dict[str, Any]]:
    """Return serializable rows, retaining unavailable and gated methods."""

    rows: list[dict[str, Any]] = []
    for item in METHOD_REGISTRY.values():
        row = asdict(item)
        row["available"] = item.available
        row["valid_relations"] = list(item.valid_relations)
        rows.append(row)
    return rows


def _mro_names(estimator: Any) -> set[str]:
    return {cls.__name__ for cls in type(estimator).mro()}


def _task_type(estimator: Any) -> str:
    names = _mro_names(estimator)
    if getattr(estimator, "_forestgap_synthetic_null", False):
        return "anomaly"
    if any("Survival" in name for name in names):
        return "survival"
    if any("Isolation" in name for name in names):
        return "anomaly"
    if any("Quantile" in name for name in names):
        return "quantile_regression"
    estimator_type = getattr(estimator, "_estimator_type", None)
    if estimator_type == "classifier" or hasattr(estimator, "predict_proba"):
        return "classification"
    if estimator_type == "regressor":
        return "regression"
    return "unknown"


def _family_for(estimator: Any) -> str:
    names = _mro_names(estimator)
    module = type(estimator).__module__
    supported_boosted = boosted_family(estimator)
    if supported_boosted is not None:
        return supported_boosted
    if getattr(estimator, "_forestgap_synthetic_null", False):
        return "synthetic_null_urf"
    if any("HonestForest" in name for name in names):
        return "honest_forest"
    if any("ExtendedIsolation" in name for name in names):
        return "extended_isolation_forest"
    if any("IsolationForest" in name for name in names):
        return "isolation_forest"
    if any("RotationForest" in name for name in names):
        return "rotation_forest"
    if any("RandomSurvivalForest" in name for name in names):
        return "random_survival_forest"
    if any("Quantile" in name and "Forest" in name for name in names):
        return "quantile_forest"
    if any("CanonicalCorrelationForest" in name or name.startswith("CCF") for name in names):
        return "canonical_correlation_forest"
    if "treeple" in module and any("Oblique" in name or "Patch" in name for name in names):
        return "treeple_oblique_forest"
    if any("ExtraTrees" in name for name in names):
        return "extra_trees"
    if any(
        token in name
        for name in names
        for token in ("GradientBoost", "XGB", "LGBM", "CatBoost", "AdaBoost")
    ):
        return "boosted_trees"
    if any("RandomForest" in name or "ForestClassifier" in name or "ForestRegressor" in name for name in names):
        return "random_forest"
    return "unknown"


def capabilities_for(estimator: Any) -> EstimatorCapabilities:
    """Inspect *estimator* and return the relation contract it actually exposes."""

    family = _family_for(estimator)
    task = _task_type(estimator)
    static = METHOD_REGISTRY.get(family)
    if static is None:
        return EstimatorCapabilities(
            family=family,
            task_type=task,
            leaf_ids=callable(getattr(estimator, "apply", None)),
            tree_inventory=hasattr(estimator, "estimators_"),
            decision_paths=callable(getattr(estimator, "decision_path", None)),
            node_depths=False,
            bootstrap=False,
            in_bag=False,
            oob=False,
            honest_roles=False,
            contribution_weights=False,
            valid_relations=(),
            status="unsupported",
            reason="No registered estimator adapter matches this estimator.",
        )

    boosted_adapter = None
    if family in BOOSTED_FAMILIES:
        try:
            boosted_adapter = make_boosted_adapter(estimator)
            inventory = boosted_adapter.tree_count > 0
        except (AttributeError, RuntimeError, ValueError):
            inventory = False
        estimators: tuple[Any, ...] = ()
        leaf_ids = inventory
        paths = bool(inventory and boosted_adapter is not None and boosted_adapter.path_capable)
        node_depths = paths
        rotation_path_contract = False
    else:
        leaf_ids = callable(getattr(estimator, "apply", None)) or family in {
            "rotation_forest", "isolation_forest", "extended_isolation_forest"
        }
        inventory = hasattr(estimator, "estimators_")
        estimators = tuple(getattr(estimator, "estimators_", ()))
        rotation_path_contract = (
            family == "rotation_forest"
            and inventory
            and all(hasattr(estimator, attr) for attr in ("_pcas", "_groups", "_check_X"))
            and len(getattr(estimator, "_pcas", ())) == len(estimators)
            and len(getattr(estimator, "_groups", ())) == len(estimators)
            and all(callable(getattr(tree, "decision_path", None)) for tree in estimators)
        )
        paths = (
            rotation_path_contract
            or (
                callable(getattr(estimator, "decision_path", None))
                and family not in {"isolation_forest", "extended_isolation_forest"}
            )
        )
        node_depths = paths and (
            (
                family == "rotation_forest"
                and all(hasattr(tree, "tree_") for tree in estimators)
            )
            or all(
                hasattr(tree, "node_depths_") or hasattr(tree, "tree_")
                for tree in estimators
            )
        )
    rotation_oob_contract = (
        family == "rotation_forest"
        and inventory
        and len(estimators) > 0
        and all(
            hasattr(estimator, attr)
            for attr in ("_pcas", "_groups", "_check_X", "_base_estimator", "n_cases_", "_n_estimators")
        )
        and len(getattr(estimator, "_pcas", ())) == len(estimators)
        and len(getattr(estimator, "_groups", ())) == len(estimators)
        and int(getattr(estimator, "_n_estimators", 0)) == len(estimators)
        and (
            callable(getattr(estimator, "_train_probas_for_estimator", None))
            or callable(getattr(estimator, "_train_preds_for_estimator", None))
        )
    )

    bootstrap = (
        False
        if family in BOOSTED_FAMILIES
        else bool(getattr(estimator, "bootstrap", False)) or rotation_oob_contract
    )
    samples = hasattr(estimator, "estimators_samples_")
    recoverable_sklearn_bootstrap = (
        bootstrap and inventory and all(hasattr(tree, "random_state") for tree in estimators)
    )
    in_bag = (
        False
        if family in BOOSTED_FAMILIES
        else samples or recoverable_sklearn_bootstrap or rotation_oob_contract
    )
    oob = bootstrap and in_bag
    honest = all(
        hasattr(estimator, attr)
        for attr in ("honest_indices_", "structure_indices_", "estimators_")
    )
    contribution = inventory if family in BOOSTED_FAMILIES else (
        bootstrap and in_bag and family not in {
            "isolation_forest", "extended_isolation_forest", "quantile_forest",
            "random_survival_forest", "synthetic_null_urf",
        }
    )

    valid = set(static.valid_relations)
    if not leaf_ids or not inventory:
        valid -= {
            "original", "collision", "conditional", "survival_leaf", "kerf",
            "boosted", "path", "englund", "random_height",
        }
    if not contribution:
        valid.discard("boosted")
    if not paths:
        valid.discard("path")
        valid.discard("englund")
    if not node_depths:
        valid.discard("random_height")
    if not (bootstrap and in_bag and oob):
        valid.discard("oob")
    if not (bootstrap and in_bag and oob and contribution):
        valid.discard("rfgap")
    if not honest:
        valid.discard("honest_gap")

    reason = static.reason
    if static.status == "gated":
        valid.clear()

    return EstimatorCapabilities(
        family=family,
        task_type=task,
        leaf_ids=leaf_ids,
        tree_inventory=inventory,
        decision_paths=paths,
        node_depths=node_depths,
        bootstrap=bootstrap,
        in_bag=in_bag,
        oob=oob,
        honest_roles=honest,
        contribution_weights=contribution,
        valid_relations=tuple(sorted(valid)),
        status=static.status,
        optional_dependency=static.optional_dependency,
        reason=reason,
    )


def validate_relation(estimator: Any, relation: str) -> EstimatorCapabilities:
    """Return capabilities or raise :class:`UnsupportedRelationError`."""

    capabilities = capabilities_for(estimator)
    if relation not in capabilities.valid_relations:
        if capabilities.family in BOOSTED_FAMILIES and relation in {"oob", "rfgap"}:
            reason = (
                "boosting does not provide the exchangeable bootstrap membership "
                "and prediction-weight contract required by forest OOB/RF-GAP relations"
            )
        elif (
            capabilities.family == "catboost"
            and relation in {"path", "englund", "random_height"}
        ):
            try:
                policy = make_boosted_adapter(estimator).grow_policy  # type: ignore[attr-defined]
            except (AttributeError, RuntimeError, ValueError):
                policy = "unknown"
            reason = (
                "CatBoost path-derived relations require grow_policy='SymmetricTree'; "
                f"the fitted model uses grow_policy={policy!r}"
            )
        else:
            reason = capabilities.reason or (
                "the required leaf/path/bootstrap/honesty capability contract is absent"
            )
        raise UnsupportedRelationError(capabilities.family, relation, reason)
    return capabilities
