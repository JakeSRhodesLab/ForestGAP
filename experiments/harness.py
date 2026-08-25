"""Shared estimator × relation × application × data-regime experiment harness."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import time
import tracemalloc
from typing import Any, Callable, Iterable, Literal, Mapping

import numpy as np

from forestgap import Proximity, UnsupportedRelationError, capabilities_for
from experiments.common import log_progress


Application = Callable[[Proximity, "ExperimentCell"], Mapping[str, Any]]


@dataclass(frozen=True)
class ExperimentCell:
    """One scientifically declared cross-method study cell.

    ``configuration`` must include every setting that changes the fitted parent;
    it participates in the shared-parent cache identity and result provenance.
    """

    estimator: str
    relation: str
    application: str
    data_regime: str
    estimator_factory: Callable[[], Any]
    application_fn: Application
    X_train: Any
    y_train: Any
    X_evaluate: Any | None = None
    y_evaluate: Any | None = None
    seed: int = 0
    split_id: str | None = None
    fit_mode: Literal["xy", "x"] = "xy"
    configuration: Mapping[str, Any] | None = None
    suite_id: int | None = None
    task_id: int | None = None
    fold_id: str | None = None
    repeat: int | None = None
    fold: int | None = None
    sample: int | None = None
    relation_settings: Mapping[str, Any] | None = None
    parent_search_trial: Mapping[str, Any] | None = None
    relation_search_trial: Mapping[str, Any] | None = None
    preprocessing: Mapping[str, Any] | None = None
    artifact_hashes: Mapping[str, str] | None = None
    capability_snapshot: Mapping[str, Any] | None = None
    protocol_version: str = "forestgap-experiment-program/1.1"
    declared_status: Literal["unsupported", "excluded", "error"] | None = None
    declared_reason: str | None = None
    declared_error_type: str | None = None


def _array_digest(value: Any) -> bytes:
    array = np.asarray(value)
    return b"|".join([
        str(array.shape).encode(), str(array.dtype).encode(), array.tobytes(),
    ])


def split_hash(cell: ExperimentCell) -> str:
    """Hash the actual train/evaluation arrays unless a stable split ID is supplied."""

    if cell.split_id is not None:
        return sha256(cell.split_id.encode()).hexdigest()
    digest = sha256()
    digest.update(_array_digest(cell.X_train))
    digest.update(_array_digest(cell.y_train))
    if cell.X_evaluate is not None:
        digest.update(_array_digest(cell.X_evaluate))
    if cell.y_evaluate is not None:
        digest.update(_array_digest(cell.y_evaluate))
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=repr)


def code_revision() -> str:
    """Return the checked-out revision plus a dirty marker when available."""

    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        return revision + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_versions() -> dict[str, str | None]:
    """Capture versions used by all supported and optional adapter families."""

    packages = [
        "forestgap", "numpy", "scipy", "scikit-learn", "pandas", "openml",
        "aeon", "treeple", "quantile-forest", "scikit-survival", "xgboost",
        "lightgbm", "catboost", "umap-learn", "phate", "hdbscan", "shap",
        "lime",
    ]
    output: dict[str, str | None] = {}
    for package in packages:
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = None
    return output


_PROXIMITY_SETTING_NAMES = frozenset({
    "matrix_type", "non_zero_diagonal", "symm_mode", "max_normalize",
    "englund_weight",
})


def _digest_json(value: Any) -> str:
    return sha256(_json(value).encode()).hexdigest()


def _manifest_fields(
    cell: ExperimentCell,
    *,
    digest: str,
    revision: str,
    versions: Mapping[str, str | None],
) -> dict[str, Any]:
    """Build the invariant portion of a result row before fitting anything."""

    relation_settings = dict(cell.relation_settings or {})
    unknown_settings = sorted(set(relation_settings) - _PROXIMITY_SETTING_NAMES)
    artifact_hashes = {
        "split": digest,
        "configuration": _digest_json(cell.configuration or {}),
        **dict(cell.artifact_hashes or {}),
    }
    manifest = {
        "protocol_version": cell.protocol_version,
        "estimator": cell.estimator,
        "relation": cell.relation,
        "application": cell.application,
        "data_regime": cell.data_regime,
        "seed": cell.seed,
        "suite_id": cell.suite_id,
        "task_id": cell.task_id,
        "fold_id": cell.fold_id,
        "repeat": cell.repeat,
        "fold": cell.fold,
        "sample": cell.sample,
        "split_id": cell.split_id,
        "split_hash": digest,
        "configuration": dict(cell.configuration or {}),
        "relation_settings": relation_settings,
        "invalid_relation_settings": unknown_settings,
        "parent_search_trial": dict(cell.parent_search_trial or {}),
        "relation_search_trial": dict(cell.relation_search_trial or {}),
        "preprocessing": dict(cell.preprocessing or {}),
        "artifact_hashes": artifact_hashes,
        "revision": revision,
        "versions": dict(versions),
    }
    return {
        "estimator": cell.estimator,
        "relation": cell.relation,
        "application": cell.application,
        "data_regime": cell.data_regime,
        "seed": cell.seed,
        "split_hash": digest,
        "split_id": cell.split_id,
        "suite_id": cell.suite_id,
        "task_id": cell.task_id,
        "fold_id": cell.fold_id,
        "repeat": cell.repeat,
        "fold": cell.fold,
        "sample": cell.sample,
        "protocol_version": cell.protocol_version,
        "revision": revision,
        "versions": _json(versions),
        "relation_settings": relation_settings,
        "invalid_relation_settings": unknown_settings,
        "parent_search_trial": dict(cell.parent_search_trial or {}),
        "relation_search_trial": dict(cell.relation_search_trial or {}),
        "search_trial": {
            "parent": dict(cell.parent_search_trial or {}),
            "relation": dict(cell.relation_search_trial or {}),
        },
        "preprocessing": dict(cell.preprocessing or {}),
        "artifact_hashes": artifact_hashes,
        "manifest": manifest,
        "manifest_hash": _digest_json(manifest),
    }


def _result_row(
    base: Mapping[str, Any],
    *,
    status: str,
    configuration: Mapping[str, Any],
    capability_snapshot: Mapping[str, Any] | None,
    parent_score: float | None,
    fit_runtime: float | None,
    relation_runtime: float | None,
    application_runtime: float | None,
    cell_runtime: float,
    peak_memory: int,
    relation_diagnostics: str | None,
    fit_reused: bool,
    metrics: Mapping[str, Any] | None = None,
    reason: str | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    """Create a schema-complete row for every terminal cell status."""

    attributed_runtime = (
        (0.0 if fit_reused else (fit_runtime or 0.0))
        + (relation_runtime or 0.0)
        + (application_runtime or 0.0)
    )
    metrics_dict = dict(metrics or {})
    capability_dict = dict(capability_snapshot or {})
    manifest = {
        **dict(base["manifest"]),
        "capability_snapshot": capability_dict,
        "status": status,
        "reason": reason,
        "metrics": metrics_dict,
    }
    row = {
        **base,
        "status": status,
        "reason": reason,
        "error_type": error_type,
        "configuration": _json(configuration),
        "capability_snapshot": capability_dict,
        "capabilities": capability_dict,
        "parent_score": parent_score,
        "fit_runtime_seconds": fit_runtime,
        "relation_runtime_seconds": relation_runtime,
        "application_runtime_seconds": application_runtime,
        "runtime_seconds": cell_runtime,
        "cell_runtime_seconds": cell_runtime,
        "attributed_runtime_seconds": attributed_runtime,
        "fit_reused": fit_reused,
        "peak_memory_bytes": peak_memory,
        "relation_diagnostics": relation_diagnostics,
        "metrics": metrics_dict,
        "manifest": manifest,
        "manifest_hash": _digest_json(manifest),
    }
    if metrics:
        row.update(metrics)
    return row


def run_cells(cells: Iterable[ExperimentCell]) -> list[dict[str, Any]]:
    """Run declared cells while reusing a selected parent within an outer fold.

    Cache scope includes the application because the protocol permits relation
    comparisons to share a refit, but not a parent selected for a different
    application.
    """

    cell_list = list(cells)
    total_cells = len(cell_list)
    revision = code_revision()
    versions = package_versions()
    fitted: dict[
        tuple[str, str, str, int, str, str],
        tuple[Any, float, int, dict[str, Any], dict[str, Any]],
    ] = {}
    rows: list[dict[str, Any]] = []
    for index, cell in enumerate(cell_list, start=1):
        cell_label = (
            f"estimator={cell.estimator} relation={cell.relation} "
            f"application={cell.application} regime={cell.data_regime} seed={cell.seed}"
        )
        log_progress(f"[{index}/{total_cells}] starting cell {cell_label}")
        digest = split_hash(cell)
        base = _manifest_fields(
            cell, digest=digest, revision=revision, versions=versions,
        )
        configuration: dict[str, Any] = {
            "cell": dict(cell.configuration or {}),
            "fit_mode": cell.fit_mode,
        }
        capability_snapshot: dict[str, Any] = dict(cell.capability_snapshot or {})
        cell_start = time.perf_counter()
        if cell.declared_status is not None:
            rows.append(_result_row(
                base,
                status=cell.declared_status,
                configuration=configuration,
                capability_snapshot=capability_snapshot,
                parent_score=None,
                fit_runtime=None,
                relation_runtime=None,
                application_runtime=None,
                cell_runtime=time.perf_counter() - cell_start,
                peak_memory=0,
                relation_diagnostics=None,
                fit_reused=False,
                reason=cell.declared_reason or "cell excluded by the declared study policy",
                error_type=cell.declared_error_type,
            ))
            log_progress(
                f"[{index}/{total_cells}] finished cell {cell_label} "
                f"status={cell.declared_status}"
            )
            continue
        if base["invalid_relation_settings"]:
            reason = "Unsupported relation setting(s): " + ", ".join(
                base["invalid_relation_settings"],
            )
            rows.append(_result_row(
                base,
                status="error",
                configuration=configuration,
                capability_snapshot=capability_snapshot,
                parent_score=None,
                fit_runtime=None,
                relation_runtime=None,
                application_runtime=None,
                cell_runtime=time.perf_counter() - cell_start,
                peak_memory=0,
                relation_diagnostics=None,
                fit_reused=False,
                reason=reason,
                error_type="ValueError",
            ))
            log_progress(
                f"[{index}/{total_cells}] finished cell {cell_label} status=error "
                f"reason=unsupported relation settings"
            )
            continue
        parent_key = (
            cell.estimator,
            cell.application,
            digest,
            cell.seed,
            cell.fit_mode,
            _json(cell.configuration or {}),
        )
        phase: str | None = None
        phase_start: float | None = None
        fit_runtime: float | None = None
        relation_runtime: float | None = None
        application_runtime: float | None = None
        peak_memory = 0
        parent_score: float | None = None
        relation_diagnostics: str | None = None
        fit_reused = False
        try:
            fit_reused = parent_key in fitted
            if parent_key not in fitted:
                log_progress(f"[{index}/{total_cells}] fitting parent for {cell_label}")
                estimator = cell.estimator_factory()
                estimator_config = (
                    estimator.get_params(deep=False)
                    if callable(getattr(estimator, "get_params", None))
                    else {}
                )
                declared_config = dict(cell.configuration or {})
                configuration = {
                    "estimator": estimator_config,
                    "cell": declared_config,
                    "fit_mode": cell.fit_mode,
                }
                tracemalloc.start()
                phase = "fit"
                fit_start = time.perf_counter()
                phase_start = fit_start
                if cell.fit_mode == "x":
                    estimator.fit(cell.X_train)
                else:
                    estimator.fit(cell.X_train, cell.y_train)
                fit_runtime = time.perf_counter() - fit_start
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_memory = max(peak_memory, peak)
                phase = None
                phase_start = None
                capability_snapshot = capabilities_for(estimator).to_dict()
                fitted[parent_key] = (
                    estimator, fit_runtime, peak, configuration, capability_snapshot,
                )
            estimator, fit_runtime, fit_peak, configuration, capability_snapshot = fitted[parent_key]
            peak_memory = max(peak_memory, fit_peak)
            if cell.X_evaluate is not None and cell.y_evaluate is not None:
                score = getattr(estimator, "score", None)
                if callable(score):
                    parent_score = float(score(cell.X_evaluate, cell.y_evaluate))
            log_progress(f"[{index}/{total_cells}] building relation for {cell_label}")
            relation_start = time.perf_counter()
            phase = "relation"
            phase_start = relation_start
            tracemalloc.start()
            proximity = Proximity(
                estimator,
                relation=cell.relation,
                random_state=cell.seed,
                **dict(cell.relation_settings or {}),
            ).attach(cell.X_train, cell.y_train)
            relation_matrix = proximity.get_proximities()
            relation_runtime = time.perf_counter() - relation_start
            _, relation_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memory = max(peak_memory, relation_peak)
            phase = None
            phase_start = None
            relation_diagnostics = _json(proximity.diagnostics(relation_matrix))
            log_progress(f"[{index}/{total_cells}] running application for {cell_label}")
            application_start = time.perf_counter()
            phase = "application"
            phase_start = application_start
            tracemalloc.start()
            metrics = dict(cell.application_fn(proximity, cell))
            application_runtime = time.perf_counter() - application_start
            _, application_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memory = max(peak_memory, application_peak)
            phase = None
            phase_start = None
            rows.append(_result_row(
                base,
                status="ok",
                configuration=configuration,
                capability_snapshot=capability_snapshot,
                parent_score=parent_score,
                fit_runtime=fit_runtime,
                relation_runtime=relation_runtime,
                application_runtime=application_runtime,
                cell_runtime=time.perf_counter() - cell_start,
                peak_memory=peak_memory,
                relation_diagnostics=relation_diagnostics,
                fit_reused=fit_reused,
                metrics=metrics,
            ))
            cell_runtime = time.perf_counter() - cell_start
            log_progress(
                f"[{index}/{total_cells}] finished cell {cell_label} "
                f"status=ok reused={fit_reused} runtime={cell_runtime:.2f}s"
            )
        except UnsupportedRelationError as exc:
            if tracemalloc.is_tracing():
                _, phase_peak = tracemalloc.get_traced_memory()
                peak_memory = max(peak_memory, phase_peak)
                tracemalloc.stop()
            if phase_start is not None:
                elapsed = time.perf_counter() - phase_start
                if phase == "fit":
                    fit_runtime = elapsed
                elif phase == "relation":
                    relation_runtime = elapsed
                elif phase == "application":
                    application_runtime = elapsed
            rows.append(_result_row(
                base,
                status="unsupported",
                configuration=configuration,
                capability_snapshot=capability_snapshot,
                parent_score=parent_score,
                fit_runtime=fit_runtime,
                relation_runtime=relation_runtime,
                application_runtime=application_runtime,
                cell_runtime=time.perf_counter() - cell_start,
                peak_memory=peak_memory,
                relation_diagnostics=relation_diagnostics,
                fit_reused=fit_reused,
                reason=str(exc),
                error_type=type(exc).__name__,
            ))
            cell_runtime = time.perf_counter() - cell_start
            log_progress(
                f"[{index}/{total_cells}] finished cell {cell_label} "
                f"status=unsupported runtime={cell_runtime:.2f}s"
            )
        except Exception as exc:  # experiment failures are data, not dropped cells
            if tracemalloc.is_tracing():
                _, phase_peak = tracemalloc.get_traced_memory()
                peak_memory = max(peak_memory, phase_peak)
                tracemalloc.stop()
            if phase_start is not None:
                elapsed = time.perf_counter() - phase_start
                if phase == "fit":
                    fit_runtime = elapsed
                elif phase == "relation":
                    relation_runtime = elapsed
                elif phase == "application":
                    application_runtime = elapsed
            rows.append(_result_row(
                base,
                status="error",
                configuration=configuration,
                capability_snapshot=capability_snapshot,
                parent_score=parent_score,
                fit_runtime=fit_runtime,
                relation_runtime=relation_runtime,
                application_runtime=application_runtime,
                cell_runtime=time.perf_counter() - cell_start,
                peak_memory=peak_memory,
                relation_diagnostics=relation_diagnostics,
                fit_reused=fit_reused,
                reason=str(exc),
                error_type=type(exc).__name__,
            ))
            cell_runtime = time.perf_counter() - cell_start
            log_progress(
                f"[{index}/{total_cells}] finished cell {cell_label} "
                f"status=error runtime={cell_runtime:.2f}s type={type(exc).__name__}"
            )
    return rows


__all__ = [
    "ExperimentCell", "code_revision", "package_versions", "run_cells", "split_hash",
]
