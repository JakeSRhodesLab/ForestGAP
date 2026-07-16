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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.model_selection import train_test_split

from forestgap import (
    CanonicalCorrelationForestClassifier,
    knn_predict,
    proximity_clustering,
    retrieve_neighbors,
    retrieval_metrics,
)

try:  # module execution and direct script execution
    from experiments.harness import ExperimentCell, run_cells
except ModuleNotFoundError:  # pragma: no cover - direct script path
    from harness import ExperimentCell, run_cells


EstimatorFactory = Callable[[int, int], Any]


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
}


def _prediction(proximity: Any, cell: ExperimentCell) -> dict[str, float]:
    predicted = knn_predict(
        proximity, cell.y_train, X=cell.X_evaluate, n_neighbors=7,
        task="classification",
    )
    return {"downstream_accuracy": float(accuracy_score(cell.y_evaluate, predicted))}


def _clustering(proximity: Any, cell: ExperimentCell) -> dict[str, float]:
    labels = proximity_clustering(
        proximity, n_clusters=len(np.unique(cell.y_train)),
        method="agglomerative", random_state=cell.seed,
    )
    return {
        "downstream_adjusted_rand": float(adjusted_rand_score(cell.y_train, labels))
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
    relations: Iterable[str],
    applications: Iterable[str],
    data_regimes: Iterable[str],
    seeds: Iterable[int],
    n_samples: int,
    n_estimators: int,
) -> list[ExperimentCell]:
    """Declare the full matrix while preserving shared parent/split identity."""

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
            for estimator_name in estimators:
                if estimator_name not in ESTIMATORS:
                    raise ValueError(f"unknown estimator {estimator_name!r}")
                factory = ESTIMATORS[estimator_name]
                for relation in relations:
                    for application in applications:
                        if application not in APPLICATIONS:
                            raise ValueError(f"unknown application {application!r}")
                        cells.append(ExperimentCell(
                            estimator=estimator_name,
                            relation=relation,
                            application=application,
                            data_regime=regime,
                            estimator_factory=lambda f=factory, s=seed: f(n_estimators, s),
                            application_fn=APPLICATIONS[application],
                            X_train=X_train,
                            y_train=y_train,
                            X_evaluate=X_test,
                            y_evaluate=y_test,
                            seed=seed,
                            split_id=split_id,
                            configuration={
                                "n_samples": n_samples,
                                "n_estimators": n_estimators,
                                "test_fraction": 0.3,
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
    parser.add_argument("--estimators", default="random_forest,extra_trees,ccf,ccf_bag")
    parser.add_argument("--relations", default="original,oob,rfgap,kerf,path,englund,random_height")
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
    cells = build_cells(
        estimators=_csv(args.estimators),
        relations=_csv(args.relations),
        applications=_csv(args.applications),
        data_regimes=_csv(args.data_regimes),
        seeds=[int(seed) for seed in _csv(args.seeds)],
        n_samples=n_samples,
        n_estimators=n_estimators,
    )
    rows = run_cells(cells)
    write_rows(rows, args.output)
    counts = {status: sum(row["status"] == status for row in rows) for status in ("ok", "unsupported", "error")}
    print(json.dumps({"output": str(args.output), "cells": len(rows), "status": counts}, sort_keys=True))
    return rows


if __name__ == "__main__":
    main()
