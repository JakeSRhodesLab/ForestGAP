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

from forestgap import Proximity, UnsupportedRelationError


Application = Callable[[Proximity, "ExperimentCell"], Mapping[str, Any]]


@dataclass(frozen=True)
class ExperimentCell:
    """One scientifically declared cross-method study cell."""

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
        "forestgap", "numpy", "scipy", "scikit-learn", "pandas", "aeon",
        "treeple", "quantile-forest", "scikit-survival",
    ]
    output: dict[str, str | None] = {}
    for package in packages:
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = None
    return output


def run_cells(cells: Iterable[ExperimentCell]) -> list[dict[str, Any]]:
    """Run declared cells while reusing each fitted parent model and split."""

    revision = code_revision()
    versions = package_versions()
    fitted: dict[
        tuple[str, str, int], tuple[Any, float, int, dict[str, Any]]
    ] = {}
    rows: list[dict[str, Any]] = []
    for cell in cells:
        configuration: Mapping[str, Any] = cell.configuration or {}
        digest = split_hash(cell)
        parent_key = (cell.estimator, digest, cell.seed)
        base = {
            "estimator": cell.estimator,
            "relation": cell.relation,
            "application": cell.application,
            "data_regime": cell.data_regime,
            "seed": cell.seed,
            "split_hash": digest,
            "split_id": cell.split_id,
            "revision": revision,
            "versions": _json(versions),
        }
        try:
            if parent_key not in fitted:
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
                fit_start = time.perf_counter()
                if cell.fit_mode == "x":
                    estimator.fit(cell.X_train)
                else:
                    estimator.fit(cell.X_train, cell.y_train)
                fit_runtime = time.perf_counter() - fit_start
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                fitted[parent_key] = (estimator, fit_runtime, peak, configuration)
            estimator, fit_runtime, fit_peak, configuration = fitted[parent_key]
            parent_score: float | None = None
            if cell.X_evaluate is not None and cell.y_evaluate is not None:
                score = getattr(estimator, "score", None)
                if callable(score):
                    parent_score = float(score(cell.X_evaluate, cell.y_evaluate))
            relation_start = time.perf_counter()
            tracemalloc.start()
            proximity = Proximity(
                estimator, relation=cell.relation, random_state=cell.seed,
            ).attach(cell.X_train, cell.y_train)
            relation_matrix = proximity.get_proximities()
            relation_runtime = time.perf_counter() - relation_start
            _, relation_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            application_start = time.perf_counter()
            tracemalloc.start()
            metrics = dict(cell.application_fn(proximity, cell))
            application_runtime = time.perf_counter() - application_start
            _, application_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append({
                **base,
                "status": "ok",
                "configuration": _json(configuration),
                "parent_score": parent_score,
                "fit_runtime_seconds": fit_runtime,
                "relation_runtime_seconds": relation_runtime,
                "application_runtime_seconds": application_runtime,
                "runtime_seconds": fit_runtime + relation_runtime + application_runtime,
                "peak_memory_bytes": max(fit_peak, relation_peak, application_peak),
                "relation_diagnostics": _json(proximity.diagnostics(relation_matrix)),
                **metrics,
            })
        except UnsupportedRelationError as exc:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            rows.append({
                **base,
                "status": "unsupported",
                "configuration": _json(configuration),
                "reason": str(exc),
                "error_type": type(exc).__name__,
            })
        except Exception as exc:  # experiment failures are data, not dropped cells
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            rows.append({
                **base,
                "status": "error",
                "configuration": _json(configuration),
                "reason": str(exc),
                "error_type": type(exc).__name__,
            })
    return rows


__all__ = [
    "ExperimentCell", "code_revision", "package_versions", "run_cells", "split_hash",
]
