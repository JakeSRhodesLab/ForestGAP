"""Run the shared estimator × relation × application × data-regime matrix.

The default study is local and deterministic.  It is intentionally small
enough for a CLI smoke run; increase ``--n-samples`` and ``--n-estimators``
for full experiments.  Every requested cell is emitted, including unsupported
and failed combinations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.model_selection import train_test_split

from forestgap import (
    CanonicalCorrelationForestClassifier,
    knn_predict,
    proximity_clustering,
    retrieve_neighbors,
    retrieval_metrics,
)
from forestgap.manifold.proximity import prepare_proximity_affinity

try:  # module execution and direct script execution
    from experiments.harness import ExperimentCell, run_cells
    from experiments.program import capability_aware_cells
except ModuleNotFoundError:  # pragma: no cover - direct script path
    from harness import ExperimentCell, run_cells
    from program import capability_aware_cells


EstimatorFactory = Callable[[int, int], Any]


def _xgboost_factory(n_estimators: int, seed: int) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=n_estimators, max_depth=4, random_state=seed, n_jobs=1,
    )


def _lightgbm_factory(n_estimators: int, seed: int) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=n_estimators, max_depth=4, random_state=seed, n_jobs=1,
        verbosity=-1,
    )


def _catboost_factory(n_estimators: int, seed: int) -> Any:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=n_estimators, depth=4, random_seed=seed, verbose=False,
        thread_count=1, allow_writing_files=False, grow_policy="SymmetricTree",
    )


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _dataset(regime: str, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, 8))
    noise = rng.normal(scale=0.35, size=n_samples)
    if regime == "axis_aligned":
        margin = X[:, 0] + noise
    elif regime == "rotated_oblique":
        margin = X[:, :4].sum(axis=1) / 2.0 + noise
    elif regime == "correlated_oblique":
        X[:, 1] = 0.85 * X[:, 0] + np.sqrt(1 - 0.85**2) * X[:, 1]
        margin = X[:, 0] - X[:, 1] + 0.7 * X[:, 2] + noise
    else:
        raise ValueError(f"unknown data regime: {regime!r}")
    return X, (margin > np.median(margin)).astype(int)


ESTIMATORS: dict[str, EstimatorFactory] = {
    "random_forest": lambda n, seed: RandomForestClassifier(
        n_estimators=n, bootstrap=True, random_state=seed, n_jobs=1,
    ),
    "extra_trees": lambda n, seed: ExtraTreesClassifier(
        n_estimators=n, bootstrap=True, random_state=seed, n_jobs=1,
    ),
    "ccf": lambda n, seed: CanonicalCorrelationForestClassifier(
        n_estimators=n, max_depth=10, random_state=seed,
    ),
    "ccf_bag": lambda n, seed: CanonicalCorrelationForestClassifier(
        n_estimators=n, max_depth=10, bootstrap=True, oob_score=True,
        random_state=seed,
    ),
    "gradient_boosting": lambda n, seed: GradientBoostingClassifier(
        n_estimators=n, max_depth=4, random_state=seed,
    ),
    "xgboost": _xgboost_factory,
    "lightgbm": _lightgbm_factory,
    "catboost": _catboost_factory,
}

ESTIMATOR_FAMILIES: dict[str, str] = {
    "random_forest": "random_forest",
    "extra_trees": "extra_trees",
    "ccf": "canonical_correlation_forest",
    "ccf_bag": "canonical_correlation_forest",
    "gradient_boosting": "gradient_boosting",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
}


def _prediction(proximity: Any, cell: ExperimentCell) -> dict[str, float]:
    predicted = knn_predict(
        proximity, cell.y_train, X=cell.X_evaluate, n_neighbors=7,
        task="classification",
    )
    return {"downstream_accuracy": float(accuracy_score(cell.y_evaluate, predicted))}


def _clustering(proximity: Any, cell: ExperimentCell) -> dict[str, float]:
    affinity = prepare_proximity_affinity(
        proximity.get_proximities(), symmetrize="auto", diagonal="one",
    )
    dense = affinity.toarray() if hasattr(affinity, "toarray") else np.asarray(affinity)
    degrees = np.clip(np.asarray(dense.sum(axis=1), dtype=float), 1e-12, None)
    normalized = dense / np.sqrt(np.outer(degrees, degrees))
    eigenvalues = np.linalg.eigvalsh(np.eye(len(normalized)) - normalized)
    max_clusters = min(8, max(2, len(eigenvalues) - 1))
    gaps = np.diff(eigenvalues[:max_clusters])
    n_clusters = int(np.argmax(gaps) + 1) if gaps.size else 2
    n_clusters = max(2, min(n_clusters, max_clusters))
    labels = proximity_clustering(
        proximity, n_clusters=n_clusters,
        method="agglomerative", random_state=cell.seed,
    )
    return {
        "downstream_adjusted_rand": float(adjusted_rand_score(cell.y_train, labels)),
        "selected_n_clusters": float(n_clusters),
    }


def _retrieval(proximity: Any, cell: ExperimentCell) -> dict[str, float]:
    indices, _ = retrieve_neighbors(proximity, cell.X_evaluate, k=10)
    relevant = [
        set(np.flatnonzero(np.asarray(cell.y_train) == label).tolist())
        for label in np.asarray(cell.y_evaluate)
    ]
    return {f"downstream_{key}": value for key, value in retrieval_metrics(indices, relevant).items()}


APPLICATIONS: dict[str, Callable[[Any, ExperimentCell], dict[str, float]]] = {
    "prediction": _prediction,
    "clustering": _clustering,
    "retrieval": _retrieval,
}


def build_cells(
    *,
    estimators: Iterable[str],
    relations: Iterable[str] | None,
    applications: Iterable[str],
    data_regimes: Iterable[str],
    seeds: Iterable[int],
    n_samples: int,
    n_estimators: int,
) -> list[ExperimentCell]:
    """Declare cells from fitted runtime capabilities.

    Passing ``relations=None`` selects each fitted parent's valid relations.
    An explicit relation list retains invalid requests as declared unsupported
    rows, which is useful for audit tables and backwards-compatible smoke
    comparisons.
    """

    estimator_names = tuple(estimators)
    application_names = tuple(applications)
    requested_relations = None if relations is None else tuple(relations)
    cells: list[ExperimentCell] = []
    for seed in seeds:
        for regime in data_regimes:
            X, y = _dataset(regime, n_samples, seed)
            train_ids, test_ids = train_test_split(
                np.arange(n_samples), test_size=0.3, stratify=y, random_state=seed,
            )
            X_train, X_test = X[train_ids], X[test_ids]
            y_train, y_test = y[train_ids], y[test_ids]
            split_id = json.dumps(
                {"regime": regime, "seed": seed, "train": train_ids.tolist(), "test": test_ids.tolist()},
                separators=(",", ":"),
            )
            for estimator_name in estimator_names:
                if estimator_name not in ESTIMATORS:
                    raise ValueError(f"unknown estimator {estimator_name!r}")
                factory = ESTIMATORS[estimator_name]
                for application in application_names:
                    if application not in APPLICATIONS:
                        raise ValueError(f"unknown application {application!r}")
                    cells.extend(capability_aware_cells(
                        family=ESTIMATOR_FAMILIES[estimator_name],
                        estimator=estimator_name,
                        estimator_factory=lambda f=factory, s=seed: f(n_estimators, s),
                        application=application,
                        application_fn=APPLICATIONS[application],
                        X_train=X_train,
                        y_train=y_train,
                        data_regime=regime,
                        requested_relations=requested_relations,
                        X_evaluate=X_test,
                        y_evaluate=y_test,
                        seed=seed,
                        split_id=split_id,
                        relation_settings={"matrix_type": "sparse"},
                        configuration={
                            "n_samples": n_samples,
                            "n_estimators": n_estimators,
                            "test_fraction": 0.3,
                            "capability_generation": "fitted_runtime_snapshot",
                        },
                    ))
    return cells


def write_rows(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        pd.DataFrame(rows).to_csv(output, index=False)
    else:
        with output.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, default=repr) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--estimators", default="all",
        help="Comma-separated estimator keys, or 'all' for every installed matrix family.",
    )
    parser.add_argument(
        "--relations", default="auto",
        help="Comma-separated relations, or 'auto' to use each fitted runtime capability snapshot.",
    )
    parser.add_argument("--applications", default="prediction,clustering,retrieval")
    parser.add_argument("--data-regimes", default="axis_aligned,rotated_oblique,correlated_oblique")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--n-estimators", type=int, default=128)
    parser.add_argument("--smoke", action="store_true", help="Use a fast deterministic local configuration.")
    parser.add_argument("--output", type=Path, default=Path("results/raw/method_matrix.jsonl"))
    return parser


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = build_parser().parse_args(argv)
    n_samples = 90 if args.smoke else args.n_samples
    n_estimators = 8 if args.smoke else args.n_estimators
    estimator_names = (
        list(ESTIMATORS)
        if args.estimators.strip().casefold() == "all"
        else _csv(args.estimators)
    )
    relation_names = (
        None
        if args.relations.strip().casefold() == "auto"
        else _csv(args.relations)
    )
    cells = build_cells(
        estimators=estimator_names,
        relations=relation_names,
        applications=_csv(args.applications),
        data_regimes=_csv(args.data_regimes),
        seeds=[int(seed) for seed in _csv(args.seeds)],
        n_samples=n_samples,
        n_estimators=n_estimators,
    )
    rows = run_cells(cells)
    write_rows(rows, args.output)
    status_order = ["ok", "unsupported", "error"]
    if any(row["status"] == "excluded" for row in rows):
        status_order.append("excluded")
    counts = {status: sum(row["status"] == status for row in rows) for status in status_order}
    print(json.dumps({"output": str(args.output), "cells": len(rows), "status": counts}, sort_keys=True))
    return rows


if __name__ == "__main__":
    main()
