"""Run the official-fold supervised baseline for the 1.1 study protocol.

This runner is intentionally a capability-aware baseline, not a replacement
for application-specific nested searches.  It uses every requested official
OpenML outer fold, records the protocol manifest fields through the shared
harness, and is suitable for deterministic smoke cells with ``--smoke``.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from forestgap import knn_predict

from experiments.common import (
    iter_openml_suite_folds,
    prepare_dataset,
    rough_fill_train_test,
    requested_tasks,
    log_progress,
    suite_metadata,
)
from experiments.harness import ExperimentCell, run_cells
from experiments.program import capability_aware_cells, default_trial_manifests


ModelFactory = Callable[[str, int, int], Any]


def _random_forest(task: str, n_estimators: int, seed: int) -> Any:
    estimator = RandomForestClassifier if task == "classification" else RandomForestRegressor
    return estimator(
        n_estimators=n_estimators, bootstrap=True, random_state=seed, n_jobs=1,
    )


def _extra_trees(task: str, n_estimators: int, seed: int) -> Any:
    estimator = ExtraTreesClassifier if task == "classification" else ExtraTreesRegressor
    return estimator(
        n_estimators=n_estimators, bootstrap=True, random_state=seed, n_jobs=1,
    )


def _gradient_boosting(task: str, n_estimators: int, seed: int) -> Any:
    estimator = GradientBoostingClassifier if task == "classification" else GradientBoostingRegressor
    return estimator(n_estimators=n_estimators, max_depth=4, random_state=seed)


MODELS: dict[str, tuple[str, ModelFactory, str]] = {
    "random_forest": ("random_forest", _random_forest, "forest"),
    "extra_trees": ("extra_trees", _extra_trees, "forest"),
    "gradient_boosting": ("gradient_boosting", _gradient_boosting, "boosted"),
}


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _classification_probabilities(proximity: Any, X: Any, y_train: Any) -> tuple[np.ndarray, np.ndarray]:
    weights = proximity.prox_extend(X)
    dense = weights.toarray() if hasattr(weights, "toarray") else np.asarray(weights)
    y = np.asarray(y_train)
    classes = np.unique(y)
    probabilities = np.column_stack([
        dense[:, y == label].sum(axis=1) for label in classes
    ])
    denominator = probabilities.sum(axis=1, keepdims=True)
    zero_rows = denominator.ravel() <= 0
    probabilities[~zero_rows] /= denominator[~zero_rows]
    probabilities[zero_rows] = 1.0 / len(classes)
    return classes, probabilities


def _prediction_metrics(proximity: Any, cell: ExperimentCell) -> dict[str, float]:
    task_kind = str((cell.configuration or {}).get("task_kind"))
    predicted = knn_predict(
        proximity,
        cell.y_train,
        X=cell.X_evaluate,
        n_neighbors=7,
        task="classification" if task_kind == "classification" else "regression",
    )
    y_test = np.asarray(cell.y_evaluate)
    if task_kind == "classification":
        classes, probabilities = _classification_probabilities(
            proximity, cell.X_evaluate, cell.y_train,
        )
        return {
            "proximity_balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
            "proximity_log_loss": float(log_loss(y_test, probabilities, labels=classes)),
        }
    return {
        "proximity_rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "proximity_mae": float(mean_absolute_error(y_test, predicted)),
        "proximity_r2": float(r2_score(y_test, predicted)),
    }


def _preparation_error_cell(
    *,
    task_kind: str,
    official_fold: Any,
    model_name: str,
    seed: int,
    n_estimators: int,
    error: Exception,
) -> ExperimentCell:
    """Represent an official-fold preparation failure without dropping the cell."""

    task_id = int(getattr(official_fold, "task_id", -1))
    fold_id = str(getattr(official_fold, "fold_id", "unavailable"))
    suite_id, _ = suite_metadata(task_kind)  # type: ignore[arg-type]
    return ExperimentCell(
        estimator=model_name,
        relation="setup",
        application="prediction",
        data_regime="openml_official_fold",
        estimator_factory=lambda: None,
        application_fn=lambda _proximity, _cell: {},
        X_train=np.empty((0, 0), dtype=float),
        y_train=np.empty(0, dtype=float),
        seed=seed,
        split_id=f"suite-{suite_id}:task-{task_id}:{fold_id}",
        suite_id=suite_id,
        task_id=task_id,
        fold_id=fold_id,
        repeat=getattr(official_fold, "repeat", None),
        fold=getattr(official_fold, "fold", None),
        sample=getattr(official_fold, "sample", None),
        configuration={
            "task_kind": task_kind,
            "n_estimators": n_estimators,
            "stage": "dataset_preparation",
        },
        declared_status="error",
        declared_reason=f"{type(error).__name__}: {error}",
        declared_error_type=type(error).__name__,
    )


def build_cells(
    *,
    task_kinds: Iterable[str],
    models: Iterable[str],
    relations: Iterable[str] | None,
    seeds: Iterable[int],
    dataset_limit: int | None,
    fold_limit: int | None,
    n_estimators: int,
) -> list[ExperimentCell]:
    """Build official-fold prediction cells and retain declared skipped cells."""

    cells: list[ExperimentCell] = []
    model_names = tuple(models)
    seed_values = tuple(seeds)
    requested_relations = None if relations is None else tuple(relations)
    for task_kind in task_kinds:
        log_progress(f"[protocol_matrix] task_kind={task_kind}")
        folds = iter_openml_suite_folds(task_kind, dataset_limit)
        for fold_number, official_fold in enumerate(folds):
            if fold_limit is not None and fold_number >= fold_limit:
                break
            log_progress(
                f"[protocol_matrix] task_kind={task_kind} task_id={official_fold.task_id} "
                f"fold_id={official_fold.fold_id}"
            )
            try:
                prepared = prepare_dataset(
                    task_kind,  # type: ignore[arg-type]
                    task_id=official_fold.task_id,
                    task=None,
                    random_state=0,
                    max_rows=None,
                    test_size=0.3,
                    official_fold=official_fold,
                )
                prepared.X_train, prepared.X_test = rough_fill_train_test(
                    prepared.X_train,
                    prepared.X_test,
                    prepared.categorical_indices,
                )
            except Exception as exc:
                log_progress(
                    f"[protocol_matrix] preparation failed for task_id={official_fold.task_id} "
                    f"fold_id={official_fold.fold_id}: {type(exc).__name__}: {exc}"
                )
                cells.extend(
                    _preparation_error_cell(
                        task_kind=task_kind,
                        official_fold=official_fold,
                        model_name=model_name,
                        seed=seed,
                        n_estimators=n_estimators,
                        error=exc,
                    )
                    for seed in seed_values
                    for model_name in model_names
                )
                continue
            for seed in seed_values:
                for model_name in model_names:
                    log_progress(
                        f"[protocol_matrix] task_kind={task_kind} task_id={prepared.task_id} "
                        f"fold_id={prepared.fold_id} model={model_name} seed={seed}"
                    )
                    try:
                        family, factory, search_family = MODELS[model_name]
                    except KeyError as exc:
                        raise ValueError(f"Unknown protocol-matrix model {model_name!r}") from exc
                    parent_trials, relation_trials = default_trial_manifests(
                        study=f"{task_kind}:{prepared.task_id}:{prepared.fold_id}:{model_name}",
                        parent_family=search_family,  # type: ignore[arg-type]
                        seed=seed,
                    )
                    parent_manifest_hash = sha256(json.dumps(
                        [trial.to_dict() for trial in parent_trials], sort_keys=True,
                    ).encode()).hexdigest()
                    relation_manifest_hash = sha256(json.dumps(
                        [trial.to_dict() for trial in relation_trials], sort_keys=True,
                    ).encode()).hexdigest()
                    cells.extend(capability_aware_cells(
                        family=family,
                        estimator=model_name,
                        estimator_factory=lambda f=factory, t=task_kind, s=seed: f(t, n_estimators, s),
                        application="prediction",
                        application_fn=_prediction_metrics,
                        X_train=prepared.X_train,
                        y_train=prepared.y_train,
                        X_evaluate=prepared.X_test,
                        y_evaluate=prepared.y_test,
                        data_regime="openml_official_fold",
                        requested_relations=requested_relations,
                        seed=seed,
                        split_id=(
                            f"suite-{prepared.suite_id}:task-{prepared.task_id}:"
                            f"{prepared.fold_id}"
                        ),
                        suite_id=prepared.suite_id,
                        task_id=prepared.task_id,
                        fold_id=prepared.fold_id,
                        repeat=prepared.repeat,
                        fold=prepared.fold,
                        sample=prepared.sample,
                        relation_settings={"matrix_type": "sparse"},
                        parent_search_trial={
                            "status": "not_tuned_baseline",
                            "n_trials": len(parent_trials),
                            "manifest_hash": parent_manifest_hash,
                        },
                        relation_search_trial={
                            "status": "not_tuned_baseline",
                            "n_trials": len(relation_trials),
                            "manifest_hash": relation_manifest_hash,
                        },
                        preprocessing={
                            "outer_split": "openml_official",
                            "feature_encoding": "fit_on_outer_train_only",
                            "missing_value_fill": "train_median_or_mode",
                            "task_kind": task_kind,
                        },
                        artifact_hashes={
                            "dataset": str(prepared.dataset_id),
                            "official_fold": str(prepared.fold_id),
                            "parent_trial_manifest": parent_manifest_hash,
                            "relation_trial_manifest": relation_manifest_hash,
                        },
                        configuration={
                            "task_kind": task_kind,
                            "n_estimators": n_estimators,
                            "trial_manifest": "frozen_sobol_lhs",
                        },
                    ))
    return cells


def _write_rows(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.casefold() == ".csv":
        pd.DataFrame(rows).to_csv(output, index=False)
        return
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, default=repr) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("classification", "regression", "both"), default="both")
    parser.add_argument("--models", default="random_forest,extra_trees,gradient_boosting")
    parser.add_argument("--relations", default="auto")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=512)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/raw/protocol_matrix.jsonl"))
    return parser


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = build_parser().parse_args(argv)
    smoke_dataset_limit = 1 if args.smoke and args.dataset_limit is None else args.dataset_limit
    smoke_fold_limit = 1 if args.smoke and args.fold_limit is None else args.fold_limit
    n_estimators = 8 if args.smoke else args.n_estimators
    log_progress("[protocol_matrix] building cells")
    cells = build_cells(
        task_kinds=requested_tasks(args.task),
        models=_csv(args.models),
        relations=None if args.relations.strip().casefold() == "auto" else _csv(args.relations),
        seeds=[int(seed) for seed in _csv(args.seeds)],
        dataset_limit=smoke_dataset_limit,
        fold_limit=smoke_fold_limit,
        n_estimators=n_estimators,
    )
    log_progress(f"[protocol_matrix] running {len(cells)} cells")
    rows = run_cells(cells)
    _write_rows(rows, args.output)
    statuses = ("ok", "unsupported", "excluded", "error")
    counts = {status: sum(row["status"] == status for row in rows) for status in statuses}
    log_progress(f"[protocol_matrix] wrote {args.output}")
    print(json.dumps({"cells": len(rows), "output": str(args.output), "status": counts}, sort_keys=True))
    return rows


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
