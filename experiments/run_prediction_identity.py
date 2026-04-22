from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    add_common_arguments,
    available_model_names,
    aggregate_success_rows,
    build_proximity_model,
    classification_probability_metrics,
    compiled_output_dir,
    requested_model_names,
    prepare_dataset,
    raw_output_path,
    record_exception,
    regression_prediction_metrics,
    save_dataframe,
    requested_tasks,
    save_plot_dataframe,
    iter_openml_suite,
    parse_random_states,
    rough_fill_train_test,
    unsupported_model_row,
)


EXPERIMENT_NAME = "prediction_identity"


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for task_kind in requested_tasks(args.task):
        for task_id, task in iter_openml_suite(task_kind, args.dataset_limit):
            dataset_name = str(getattr(task, "dataset_name", f"task_{task_id}"))
            dataset_id = int(getattr(task, "dataset_id", -1))
            for random_state in parse_random_states(args.random_states):
                for model_name in requested_model_names():
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
                            task_kind,
                            model_name=model_name,
                            prox_method="rfgap",
                            random_state=random_state,
                            n_estimators=args.n_estimators,
                            bootstrap=True,
                            oob_score=True,
                        )
                        fit_start = time.perf_counter()
                        model.fit(X_train_filled, prepared.y_train)
                        fit_runtime_seconds = time.perf_counter() - fit_start

                        if task_kind == "classification":
                            prox_test_start = time.perf_counter()
                            prox_test = np.asarray(model.prox_predict(X_test_filled), dtype=np.float64)
                            prox_test_runtime_seconds = time.perf_counter() - prox_test_start
                            forest_test_start = time.perf_counter()
                            forest_test = np.asarray(model.predict_proba(X_test_filled), dtype=np.float64)
                            forest_test_runtime_seconds = time.perf_counter() - forest_test_start
                            prox_oob_start = time.perf_counter()
                            prox_oob = np.asarray(model.prox_predict(), dtype=np.float64)
                            prox_oob_runtime_seconds = time.perf_counter() - prox_oob_start
                            forest_oob = np.asarray(model.oob_decision_function_, dtype=np.float64)
                            test_metrics = classification_probability_metrics(
                                prox_test,
                                forest_test,
                                prepared.y_test,
                            )
                            oob_metrics = classification_probability_metrics(
                                prox_oob,
                                forest_oob,
                                prepared.y_train,
                            )
                        else:
                            prox_test_start = time.perf_counter()
                            prox_test = np.asarray(model.prox_predict(X_test_filled), dtype=np.float64)
                            prox_test_runtime_seconds = time.perf_counter() - prox_test_start
                            forest_test_start = time.perf_counter()
                            forest_test = np.asarray(model.predict(X_test_filled), dtype=np.float64)
                            forest_test_runtime_seconds = time.perf_counter() - forest_test_start
                            prox_oob_start = time.perf_counter()
                            prox_oob = np.asarray(model.prox_predict(), dtype=np.float64)
                            prox_oob_runtime_seconds = time.perf_counter() - prox_oob_start
                            forest_oob = np.asarray(model.oob_prediction_, dtype=np.float64)
                            test_metrics = regression_prediction_metrics(
                                prox_test,
                                forest_test,
                                prepared.y_test,
                            )
                            oob_metrics = regression_prediction_metrics(
                                prox_oob,
                                forest_oob,
                                prepared.y_train,
                            )

                        rows.append(
                            {
                                "experiment": EXPERIMENT_NAME,
                                "status": "ok",
                                "task_kind": task_kind,
                                "suite_name": prepared.suite_name,
                                "model_name": model_name,
                                "random_state": random_state,
                                "task_id": prepared.task_id,
                                "dataset_id": prepared.dataset_id,
                                "dataset_name": prepared.dataset_name,
                                "train_size": prepared.train_size,
                                "test_size": prepared.test_size,
                                "n_features": len(prepared.feature_names),
                                "n_categorical_features": len(prepared.categorical_indices),
                                "test_mean_abs_diff": test_metrics["mean_abs_diff"],
                                "test_max_abs_diff": test_metrics["max_abs_diff"],
                                "oob_mean_abs_diff": oob_metrics["mean_abs_diff"],
                                "oob_max_abs_diff": oob_metrics["max_abs_diff"],
                                "performance": test_metrics["accuracy" if task_kind == "classification" else "r2"],
                                "fit_runtime_seconds": float(fit_runtime_seconds),
                                "prox_test_runtime_seconds": float(prox_test_runtime_seconds),
                                "forest_test_runtime_seconds": float(forest_test_runtime_seconds),
                                "prox_oob_runtime_seconds": float(prox_oob_runtime_seconds),
                                "total_runtime_seconds": float(
                                    fit_runtime_seconds
                                    + prox_test_runtime_seconds
                                    + forest_test_runtime_seconds
                                    + prox_oob_runtime_seconds
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

    return pd.DataFrame(rows)


def compile_outputs(df: pd.DataFrame) -> None:
    for task_kind in sorted(df["task_kind"].dropna().unique()):
        task_df = df[df["task_kind"] == task_kind].copy()
        raw_path = raw_output_path(EXPERIMENT_NAME, task_kind)
        save_dataframe(task_df, raw_path)

        summary = aggregate_success_rows(
            task_df,
            group_cols=("model_name",),
            metric_cols=(
                "test_mean_abs_diff",
                "test_max_abs_diff",
                "oob_mean_abs_diff",
                "oob_max_abs_diff",
                "performance",
                "fit_runtime_seconds",
                "prox_test_runtime_seconds",
                "forest_test_runtime_seconds",
                "prox_oob_runtime_seconds",
                "total_runtime_seconds",
            ),
        )
        out_dir = compiled_output_dir(EXPERIMENT_NAME, task_kind)
        save_dataframe(summary, out_dir / "summary.csv")
        save_dataframe(summary, out_dir / "model_comparison.csv")
        if not summary.empty:
            save_plot_dataframe(
                summary.sort_values("test_max_abs_diff"),
                x="model_name",
                y="test_max_abs_diff",
                title=f"{task_kind.title()} RF-GAP Test Prediction Identity",
                ylabel="Mean max absolute difference",
                path=out_dir / "test_max_abs_diff.png",
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--n-estimators", type=int, default=128)
    return parser


if __name__ == "__main__":
    frame = run(build_parser().parse_args())
    compile_outputs(frame)
