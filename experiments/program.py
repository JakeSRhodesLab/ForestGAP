"""Protocol primitives for capability-driven, leakage-safe experiment cells.

This module deliberately contains no dataset downloads or long-running search.
It freezes the declarations that a runner persists beside its scores: runtime
capabilities, trial manifests, inner/calibration partitions, relation-use
policy, and training-only alignment anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
from scipy.stats import qmc
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from forestgap import capabilities_for
from forestgap.adapters import METHOD_REGISTRY, method_capabilities

from experiments.harness import ExperimentCell


DIRECTED_RELATIONS = frozenset({"rfgap", "honest_gap"})
GEOMETRY_APPLICATIONS = frozenset({"embedding", "geometry", "clustering"})
PREDICTIVE_APPLICATIONS = frozenset({
    "prediction", "kernel_prediction", "knn_prediction", "attribution",
    "imputation", "retrieval", "active_learning",
})
IDENTITY_RELATIONS = frozenset({"rfgap", "honest_gap"})
DEFERRED_FAMILIES = frozenset({"random_survival_forest", "autoencoding_forest"})

# This is the executable counterpart of the application matrix in
# ``notes/experimental_designs.md``.  ``all`` always means the *fitted*
# capability snapshot, never a hand-maintained global relation list.
APPLICATION_RELATION_SCOPES: dict[str, frozenset[str] | None] = {
    "prediction": None,
    "kernel_prediction": None,
    "knn_prediction": None,
    "imputation": None,
    "local_feature_importance": None,
    "feature_selection": None,
    "prototypes": None,
    "retrieval": None,
    "active_learning": None,
    "embedding": None,
    "geometry": None,
    "clustering": None,
    # RF-GAP and Honest-GAP are consumed directly for their capable supervised
    # anomaly-comparator families; no implicit geometry symmetrization occurs.
    "anomaly": frozenset({
        "original", "oob", "rfgap", "honest_gap", "kerf", "boosted",
        "path", "englund", "random_height", "collision",
    }),
    # Attribution is intentionally restricted to predictive directed weights.
    "attribution": DIRECTED_RELATIONS,
    "prediction_identity": IDENTITY_RELATIONS,
}


# Values are discrete candidates or inclusive numeric ranges.  They are kept in
# source control and sampled through ``frozen_trial_manifest`` before a study.
PARENT_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "forest": {
        "n_estimators": (256, 1024),
        "max_depth": [None, 8, 16, 32],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "max_features": (0.1, 1.0),
        "max_samples": (0.5, 1.0),
    },
    "boosted": {
        "n_estimators": (100, 1000),
        "learning_rate": (0.01, 0.3),
        "max_depth": [3, 4, 6, 8, 10],
        "num_leaves": [7, 15, 31, 63],
        "min_child_samples": [1, 5, 10, 20, 50],
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
    },
}
RELATION_APPLICATION_SEARCH_SPACE: dict[str, Any] = {
    "englund_weight": [0.25, 0.5, 1.0, 2.0, 4.0],
    "n_neighbors": [3, 5, 10, 20, 40],
    "kernel_regularization": [1e-8, 1e-6, 1e-4, 1e-2],
    "imputation_iterations": [1, 3, 5, 10],
}


@dataclass(frozen=True)
class TrialManifest:
    """One deterministic point in a declared parent or relation search."""

    study: str
    kind: Literal["parent", "relation_application"]
    trial_id: int
    seed: int
    sampler: Literal["sobol", "latin_hypercube"]
    parameters: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "study": self.study,
            "kind": self.kind,
            "trial_id": self.trial_id,
            "seed": self.seed,
            "sampler": self.sampler,
            "parameters": dict(self.parameters),
        }


def _sample_parameter(value: Any, coordinate: float) -> Any:
    if isinstance(value, list):
        if not value:
            raise ValueError("Search-space candidate lists cannot be empty.")
        index = min(int(coordinate * len(value)), len(value) - 1)
        return value[index]
    if isinstance(value, tuple) and len(value) == 2:
        low, high = value
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise TypeError("Numeric search ranges must contain two numbers.")
        sampled = float(low) + (float(high) - float(low)) * coordinate
        if isinstance(low, int) and isinstance(high, int):
            return int(round(sampled))
        return sampled
    raise TypeError(
        "Search-space values must be candidate lists or two-number ranges; "
        f"received {value!r}."
    )


def frozen_trial_manifest(
    *,
    study: str,
    kind: Literal["parent", "relation_application"],
    space: Mapping[str, Any],
    n_trials: int,
    seed: int,
    sampler: Literal["sobol", "latin_hypercube"] = "sobol",
) -> list[TrialManifest]:
    """Freeze deterministic Sobol/LHS trials before evaluating any outer fold."""

    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    names = list(space)
    if not names:
        raise ValueError("A trial manifest requires at least one search parameter.")
    engine: qmc.QMCEngine
    if sampler == "sobol":
        engine = qmc.Sobol(d=len(names), scramble=True, seed=seed)
        # Sobol balance is defined for powers of two.  Generate the smallest
        # balanced design that covers the requested manifest and freeze its
        # deterministic prefix rather than accepting a warning-prone draw.
        design = engine.random_base2(int(np.ceil(np.log2(n_trials))))[:n_trials]
    elif sampler == "latin_hypercube":
        engine = qmc.LatinHypercube(d=len(names), seed=seed)
        design = engine.random(n_trials)
    else:
        raise ValueError(f"Unknown sampler {sampler!r}.")
    return [
        TrialManifest(
            study=study,
            kind=kind,
            trial_id=trial_id,
            seed=seed,
            sampler=sampler,
            parameters={
                name: _sample_parameter(space[name], float(coordinate))
                for name, coordinate in zip(names, coordinates)
            },
        )
        for trial_id, coordinates in enumerate(design)
    ]


def default_trial_manifests(
    *,
    study: str,
    parent_family: Literal["forest", "boosted"],
    seed: int,
) -> tuple[list[TrialManifest], list[TrialManifest]]:
    """Return the protocol's 50 parent and 25 relation/application trials."""

    return (
        frozen_trial_manifest(
            study=study,
            kind="parent",
            space=PARENT_SEARCH_SPACES[parent_family],
            n_trials=50,
            seed=seed,
        ),
        frozen_trial_manifest(
            study=study,
            kind="relation_application",
            space=RELATION_APPLICATION_SEARCH_SPACE,
            n_trials=25,
            seed=seed + 1,
            sampler="latin_hypercube",
        ),
    )


def inner_fold_indices(
    outer_train_indices: Sequence[int],
    y: Sequence[Any],
    *,
    task: Literal["classification", "regression"],
    n_splits: int = 5,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create five inner folds expressed in original-row indices only."""

    outer = np.asarray(outer_train_indices, dtype=np.int64)
    targets = np.asarray(y)
    if outer.ndim != 1 or targets.shape[0] != outer.size:
        raise ValueError("outer_train_indices and y must align one-to-one.")
    if np.unique(outer).size != outer.size:
        raise ValueError("outer_train_indices must be unique.")
    if outer.size < n_splits:
        raise ValueError("The outer training partition is smaller than n_splits.")
    if task == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(outer.size), targets)
    elif task == "regression":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(outer.size))
    else:
        raise ValueError(f"Unknown task {task!r}.")
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for train_position, validate_position in iterator:
        train = outer[np.asarray(train_position, dtype=np.int64)]
        validate = outer[np.asarray(validate_position, dtype=np.int64)]
        if np.intersect1d(train, validate).size:
            raise AssertionError("Inner train/validation partitions overlap.")
        output.append((train, validate))
    return output


def calibration_split_indices(
    outer_train_indices: Sequence[int],
    y: Sequence[Any],
    *,
    task: Literal["classification", "regression"],
    calibration_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split only an outer training partition into fit and calibration roles."""

    outer = np.asarray(outer_train_indices, dtype=np.int64)
    targets = np.asarray(y)
    if outer.ndim != 1 or targets.shape[0] != outer.size:
        raise ValueError("outer_train_indices and y must align one-to-one.")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie strictly between zero and one.")
    stratify = targets if task == "classification" else None
    try:
        fit, calibration = train_test_split(
            outer,
            test_size=calibration_fraction,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        # Tiny smoke datasets can lack sufficient class support; the split still
        # remains strictly inside the outer-training partition.
        fit, calibration = train_test_split(
            outer, test_size=calibration_fraction, random_state=seed, stratify=None,
        )
    if np.intersect1d(fit, calibration).size:
        raise AssertionError("Fit and calibration partitions overlap.")
    return np.sort(fit), np.sort(calibration)


def training_only_anchors(
    train_indices: Sequence[int],
    *,
    fraction: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """Choose deterministic paired-view anchors only from outer training rows."""

    train = np.asarray(train_indices, dtype=np.int64)
    if train.ndim != 1 or train.size == 0 or np.unique(train).size != train.size:
        raise ValueError("train_indices must be a non-empty unique one-dimensional sequence.")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1].")
    n_anchors = max(1, int(round(fraction * train.size)))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(train, size=n_anchors, replace=False).astype(np.int64))


def relation_use_policy(relation: str, application: str) -> dict[str, Any]:
    """Declare whether a relation is consumed directly or symmetrized for geometry."""

    directed = relation in DIRECTED_RELATIONS
    if application == "prediction_identity" and relation not in IDENTITY_RELATIONS:
        raise ValueError("Prediction identity is limited to RF-GAP and Honest-GAP.")
    if directed and application in GEOMETRY_APPLICATIONS:
        return {
            "relation": relation,
            "application": application,
            "directed": True,
            "use": "symmetrize_for_geometry",
            "label_free": True,
        }
    return {
        "relation": relation,
        "application": application,
        "directed": directed,
        "use": "direct_predictive_weights" if directed else "symmetric_relation",
        "label_free": application in GEOMETRY_APPLICATIONS,
    }


def application_relations(
    application: str,
    valid_relations: Iterable[str],
) -> list[str]:
    """Filter a fitted capability snapshot by the protocol application scope.

    An unknown application is rejected rather than inheriting all relations;
    this keeps new runners from accidentally widening the normative protocol.
    """

    if application not in APPLICATION_RELATION_SCOPES:
        raise ValueError(
            f"No application relation scope is declared for {application!r}."
        )
    allowed = APPLICATION_RELATION_SCOPES[application]
    relations = list(valid_relations)
    return relations if allowed is None else [
        relation for relation in relations if relation in allowed
    ]


def _excluded_cell(
    *,
    estimator: str,
    relation: str,
    application: str,
    data_regime: str,
    estimator_factory: Callable[[], Any],
    application_fn: Callable[[Any, ExperimentCell], Mapping[str, Any]],
    X_train: Any,
    y_train: Any,
    reason: str,
    cell_kwargs: Mapping[str, Any],
    status: Literal["unsupported", "excluded"],
    capability_snapshot: Mapping[str, Any] | None = None,
) -> ExperimentCell:
    payload = dict(cell_kwargs)
    if capability_snapshot is None:
        capability_snapshot = payload.pop("capability_snapshot", None)
    else:
        payload.pop("capability_snapshot", None)
    return ExperimentCell(
        estimator=estimator,
        relation=relation,
        application=application,
        data_regime=data_regime,
        estimator_factory=estimator_factory,
        application_fn=application_fn,
        X_train=X_train,
        y_train=y_train,
        capability_snapshot=capability_snapshot,
        declared_status=status,
        declared_reason=reason,
        **payload,
    )


def capability_aware_cells(
    *,
    family: str,
    estimator: str,
    estimator_factory: Callable[[], Any],
    application: str,
    application_fn: Callable[[Any, ExperimentCell], Mapping[str, Any]],
    X_train: Any,
    y_train: Any,
    data_regime: str,
    requested_relations: Iterable[str] | None = None,
    fit_mode: Literal["xy", "x"] = "xy",
    **cell_kwargs: Any,
) -> list[ExperimentCell]:
    """Declare cells from a fitted runtime capability snapshot.

    Explicitly requested invalid relations are retained as declared
    ``unsupported`` rows.  Gated, unregistered, and protocol-deferred families
    are retained as ``excluded`` rows without invoking their factory.
    """

    requested = list(requested_relations or [])
    if family not in METHOD_REGISTRY:
        return [_excluded_cell(
            estimator=estimator,
            relation=requested[0] if requested else "none",
            application=application,
            data_regime=data_regime,
            estimator_factory=estimator_factory,
            application_fn=application_fn,
            X_train=X_train,
            y_train=y_train,
            reason=f"{family!r} is not a registered forestgap family.",
            cell_kwargs={"fit_mode": fit_mode, **cell_kwargs},
            status="excluded",
        )]
    static = method_capabilities(family)
    if static.status == "gated" or family in DEFERRED_FAMILIES:
        reason = static.reason or f"{family!r} is deferred by the study protocol."
        return [_excluded_cell(
            estimator=estimator,
            relation=requested[0] if requested else "none",
            application=application,
            data_regime=data_regime,
            estimator_factory=estimator_factory,
            application_fn=application_fn,
            X_train=X_train,
            y_train=y_train,
            reason=reason,
            cell_kwargs={"fit_mode": fit_mode, **cell_kwargs},
            status="excluded",
        )]

    try:
        probe = estimator_factory()
        if fit_mode == "x":
            probe.fit(X_train)
        else:
            probe.fit(X_train, y_train)
        snapshot = capabilities_for(probe).to_dict()
    except (ImportError, ModuleNotFoundError) as exc:
        return [_excluded_cell(
            estimator=estimator,
            relation=requested[0] if requested else "none",
            application=application,
            data_regime=data_regime,
            estimator_factory=estimator_factory,
            application_fn=application_fn,
            X_train=X_train,
            y_train=y_train,
            reason=f"Optional dependency is unavailable: {exc}",
            cell_kwargs={"fit_mode": fit_mode, **cell_kwargs},
            status="unsupported",
            capability_snapshot={"family": family, "status": "optional_missing"},
        )]
    except Exception as exc:
        return [_excluded_cell(
            estimator=estimator,
            relation=requested[0] if requested else "none",
            application=application,
            data_regime=data_regime,
            estimator_factory=estimator_factory,
            application_fn=application_fn,
            X_train=X_train,
            y_train=y_train,
            reason=f"Capability preflight failed: {exc}",
            cell_kwargs={"fit_mode": fit_mode, **cell_kwargs},
            status="unsupported",
        )]

    valid_relations = list(snapshot["valid_relations"])
    relations = requested or application_relations(application, valid_relations)
    if not relations:
        return [_excluded_cell(
            estimator=estimator,
            relation="none",
            application=application,
            data_regime=data_regime,
            estimator_factory=estimator_factory,
            application_fn=application_fn,
            X_train=X_train,
            y_train=y_train,
            reason=(
                f"No fitted relation for {family!r} is applicable to the "
                f"{application!r} protocol scope."
            ),
            cell_kwargs={"fit_mode": fit_mode, **cell_kwargs},
            status="excluded",
            capability_snapshot=snapshot,
        )]
    cells: list[ExperimentCell] = []
    for relation in relations:
        policy = relation_use_policy(relation, application) if relation != "none" else {}
        relation_settings = dict(cell_kwargs.get("relation_settings") or {})
        preprocessing = {
            **dict(cell_kwargs.get("preprocessing") or {}),
            "relation_use": policy,
        }
        kwargs = {
            **cell_kwargs,
            "fit_mode": fit_mode,
            "relation_settings": relation_settings,
            "preprocessing": preprocessing,
            "capability_snapshot": snapshot,
        }
        if relation not in valid_relations:
            cells.append(_excluded_cell(
                estimator=estimator,
                relation=relation,
                application=application,
                data_regime=data_regime,
                estimator_factory=estimator_factory,
                application_fn=application_fn,
                X_train=X_train,
                y_train=y_train,
                reason=(
                    f"Runtime capabilities for {family!r} do not include "
                    f"relation {relation!r}."
                ),
                cell_kwargs=kwargs,
                status="unsupported",
                capability_snapshot=snapshot,
            ))
        else:
            cells.append(ExperimentCell(
                estimator=estimator,
                relation=relation,
                application=application,
                data_regime=data_regime,
                estimator_factory=estimator_factory,
                application_fn=application_fn,
                X_train=X_train,
                y_train=y_train,
                **kwargs,
            ))
    return cells


__all__ = [
    "DEFERRED_FAMILIES",
    "DIRECTED_RELATIONS",
    "GEOMETRY_APPLICATIONS",
    "IDENTITY_RELATIONS",
    "PARENT_SEARCH_SPACES",
    "PREDICTIVE_APPLICATIONS",
    "RELATION_APPLICATION_SEARCH_SPACE",
    "TrialManifest",
    "calibration_split_indices",
    "capability_aware_cells",
    "default_trial_manifests",
    "frozen_trial_manifest",
    "inner_fold_indices",
    "relation_use_policy",
    "training_only_anchors",
]
