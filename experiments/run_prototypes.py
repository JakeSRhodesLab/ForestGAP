from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    add_common_arguments,
    available_model_names,
    aggregate_success_rows,
    build_proximity_model,
    compiled_output_dir,
    iter_openml_suite,
    parse_random_states,
    prepare_dataset,
    raw_output_path,
    record_exception,
    requested_model_names,
    rough_fill_train_test,
    save_dataframe,
    save_plot_dataframe,
    unsupported_model_row,
)


EXPERIMENT_NAME = "prototypes"


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for task_kind in ("classification", "regression"):
        for task_id, task in iter_openml_suite(task_kind, args.dataset_limit):
            dataset_name = str(getattr(task, "dataset_name", f"task_{task_id}"))
            dataset_id = int(getattr(task, "dataset_id", -1))
            for random_state in parse_random_states(args.random_states):
                for model_name in requested_model_names():
                    if task_kind == "regression":
                        rows.append(
                            unsupported_model_row(
                                experiment_name=EXPERIMENT_NAME,
                                task_kind=task_kind,
                                model_name=model_name,
                                dataset_name=dataset_name,
                                dataset_id=dataset_id,
                                task_id=task_id,
                                random_state=random_state,
                                reason="get_prototypes is only implemented for classifiers in forestgap.core",
                            )
                        )
                        continue
                    if model_name not in available_model_names(task_kind):
                        rows.append(
                            unsupported_model_row(
                                experiment_name=EXPERIMENT_NAME,
                                task_kind=task_kind,
                                model_name=model_name,
                                dataset_name=dataset_name,
                                dataset_id=dataset_id,
                                task_id=task_id,
                                random_state=random_state,
                                reason=f"{model_name} is not available for {task_kind}.",
                            )
                        )
                        continue

                    try:
                        prepared = prepare_dataset(
                            task_kind,
                            task_id=task_id,
                            task=task,
                            random_state=random_state,
                            max_rows=args.max_rows,
                            test_size=args.test_size,
                        )
                        X_train_filled, X_test_filled = rough_fill_train_test(
                            prepared.X_train,
                            prepared.X_test,
                            prepared.categorical_indices,
                        )
                        model = build_proximity_model(
                            prepared.task_kind,
                            model_name=model_name,
                            prox_method="original",
                            random_state=random_state,
                            n_estimators=args.n_estimators,
                            bootstrap=True,
                            oob_score=False,
                        )
                        fit_start = time.perf_counter()
                        model.fit(X_train_filled, prepared.y_train)
                        fit_runtime_seconds = time.perf_counter() - fit_start
                        prototype_start = time.perf_counter()
                        prototypes = model.get_prototypes(
                            k=args.k,
                            n_prototypes=args.n_prototypes,
                            categorical_features=prepared.categorical_indices,
                        )
                        prototype_runtime_seconds = time.perf_counter() - prototype_start

                        centers: list[int] = []
                        center_labels: list[int] = []
                        same_class_fractions: list[float] = []
                        covered_neighbors: set[int] = set()
                        for cls, records in prototypes.items():
                            for record in records:
                                center_pos = int(record["center_position"])
                                centers.append(center_pos)
                                center_labels.append(int(cls))
                                same_class_fractions.append(float(record["same_class_neighbor_fraction"]))
                                covered_neighbors.update(int(pos) for pos in record["neighbor_positions"])
                                detail_rows.append(
                                    {
                                        "task_kind": task_kind,
                                        "model_name": model_name,
                                        "dataset_name": prepared.dataset_name,
                                        "dataset_id": prepared.dataset_id,
                                        "random_state": random_state,
                                        "class_label": int(cls),
                                        "center_position": center_pos,
                                        "same_class_neighbor_fraction": float(record["same_class_neighbor_fraction"]),
                                        "same_class_neighbor_count": int(record["same_class_neighbor_count"]),
                                        "neighbor_count": int(len(record["neighbor_positions"])),
                                    }
                                )

                        if not centers:
                            raise ValueError("Prototype extraction returned no centers.")

                        projection_start = time.perf_counter()
                        prox_test = model.prox_extend(X_test_filled)
                        prox_test_dense = prox_test.toarray() if sparse.issparse(prox_test) else np.asarray(prox_test)
                        classes = np.array(sorted(set(center_labels)), dtype=np.int64)
                        scores = np.zeros((prepared.X_test.shape[0], classes.size), dtype=np.float64)
                        for class_idx, cls in enumerate(classes):
                            class_centers = [center for center, label in zip(centers, center_labels) if label == int(cls)]
                            scores[:, class_idx] = prox_test_dense[:, class_centers].sum(axis=1)
                        pred = classes[np.argmax(scores, axis=1)]
                        prototype_accuracy = float(accuracy_score(prepared.y_test, pred))
                        projection_runtime_seconds = time.perf_counter() - projection_start

                        rows.append(
                            {
                                "experiment": EXPERIMENT_NAME,
                                "status": "ok",
                                "task_kind": task_kind,
                                "model_name": model_name,
                                "random_state": random_state,
                                "task_id": prepared.task_id,
                                "dataset_id": prepared.dataset_id,
                                "dataset_name": prepared.dataset_name,
                                "n_centers": len(centers),
                                "mean_same_class_neighbor_fraction": float(np.mean(same_class_fractions)),
                                "coverage_fraction": float(len(covered_neighbors) / max(prepared.train_size, 1)),
                                "prototype_accuracy": prototype_accuracy,
                                "fit_runtime_seconds": float(fit_runtime_seconds),
                                "prototype_runtime_seconds": float(prototype_runtime_seconds),
                                "projection_runtime_seconds": float(projection_runtime_seconds),
                                "total_runtime_seconds": float(
                                    fit_runtime_seconds
                                    + prototype_runtime_seconds
                                    + projection_runtime_seconds
                                ),
                            }
                        )
                    except Exception as exc:
                        error_row = record_exception(
                            experiment_name=EXPERIMENT_NAME,
                            task_kind=task_kind,
                            random_state=random_state,
                            dataset_name=dataset_name,
                            dataset_id=dataset_id,
                            task_id=task_id,
                            error=exc,
                        )
                        error_row["model_name"] = model_name
                        rows.append(error_row)

    frame = pd.DataFrame(rows)
    if detail_rows:
        detail_df = pd.DataFrame(detail_rows)
        detail_path = raw_output_path(EXPERIMENT_NAME, "classification").with_name("classification_details.csv")
        save_dataframe(detail_df, detail_path)
    return frame


def compile_outputs(df: pd.DataFrame) -> None:
    for task_kind in sorted(df["task_kind"].dropna().unique()):
        task_df = df[df["task_kind"] == task_kind].copy()
        save_dataframe(task_df, raw_output_path(EXPERIMENT_NAME, task_kind))
        out_dir = compiled_output_dir(EXPERIMENT_NAME, task_kind)
        metric_cols = (
            "prototype_accuracy",
            "mean_same_class_neighbor_fraction",
            "coverage_fraction",
            "fit_runtime_seconds",
            "prototype_runtime_seconds",
            "projection_runtime_seconds",
            "total_runtime_seconds",
        )
        summary = aggregate_success_rows(
            task_df,
            group_cols=("model_name",),
            metric_cols=metric_cols,
        )
        save_dataframe(summary, out_dir / "summary.csv")
        save_dataframe(summary, out_dir / "model_comparison.csv")
        if not summary.empty:
            save_plot_dataframe(
                summary.sort_values("prototype_accuracy", ascending=False),
                x="model_name",
                y="prototype_accuracy",
                title="Prototype-Induced Classification Accuracy",
                ylabel="Accuracy",
                path=out_dir / "prototype_accuracy.png",
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--n-estimators", type=int, default=128)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--n-prototypes", type=int, default=3)
    return parser


if __name__ == "__main__":
    frame = run(build_parser().parse_args())
    compile_outputs(frame)
