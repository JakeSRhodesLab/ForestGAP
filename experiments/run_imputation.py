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
    apply_missing_mask,
    build_proximity_model,
    compiled_output_dir,
    imputation_metrics,
    induced_missing_masks,
    iterative_imputation,
    iter_openml_suite,
    parse_random_states,
    prepare_dataset,
    raw_output_path,
    record_exception,
    requested_model_names,
    requested_tasks,
    rough_fill_train_test,
    save_dataframe,
    save_plot_dataframe,
    unsupported_model_row,
)


EXPERIMENT_NAME = "imputation"


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    configurations = [
        ("original", "single_pass"),
        ("oob", "single_pass"),
        ("rfgap", "single_pass"),
        ("rfgap", "iterative"),
    ]

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
                        train_mask, test_mask = induced_missing_masks(
                            prepared.X_train,
                            prepared.X_test,
                            missing_rate=args.missing_rate,
                            random_state=random_state,
                        )
                        X_train_masked = apply_missing_mask(prepared.X_train, train_mask)
                        X_test_masked = apply_missing_mask(prepared.X_test, test_mask)

                        for prox_method, variant in configurations:
                            if variant == "single_pass":
                                X_fit_train, _ = rough_fill_train_test(
                                    X_train_masked,
                                    X_test_masked,
                                    prepared.categorical_indices,
                                )
                                model = build_proximity_model(
                                    task_kind,
                                    model_name=model_name,
                                    prox_method=prox_method,
                                    random_state=random_state,
                                    n_estimators=args.n_estimators,
                                    bootstrap=True,
                                    oob_score=False,
                                )
                                fit_start = time.perf_counter()
                                model.fit(X_fit_train, prepared.y_train)
                                fit_runtime_seconds = time.perf_counter() - fit_start
                                impute_start = time.perf_counter()
                                X_imp_train, X_imp_test = model.impute(
                                    X_train_masked,
                                    categorical_features=prepared.categorical_indices,
                                    X_test=X_test_masked,
                                    initialization="median",
                                )
                                impute_runtime_seconds = time.perf_counter() - impute_start
                            else:
                                (
                                    X_imp_train,
                                    X_imp_test,
                                    timing_metrics,
                                ) = iterative_imputation(
                                    prepared,
                                    random_state=random_state,
                                    n_estimators=args.n_estimators,
                                    n_iterations=args.n_iterations,
                                    X_train_masked=X_train_masked,
                                    X_test_masked=X_test_masked,
                                    model_name=model_name,
                                )
                                fit_runtime_seconds = float(timing_metrics["fit_runtime_seconds"])
                                impute_runtime_seconds = float(timing_metrics["impute_runtime_seconds"])

                            train_metrics = imputation_metrics(
                                prepared.X_train,
                                np.asarray(X_imp_train, dtype=np.float64),
                                train_mask,
                                prepared.categorical_indices,
                            )
                            test_metrics = imputation_metrics(
                                prepared.X_test,
                                np.asarray(X_imp_test, dtype=np.float64),
                                test_mask,
                                prepared.categorical_indices,
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
                                    "prox_method": prox_method,
                                    "variant": variant,
                                    "missing_rate": args.missing_rate,
                                    "train_numeric_rmse": train_metrics["numeric_rmse"],
                                    "test_numeric_rmse": test_metrics["numeric_rmse"],
                                    "train_categorical_accuracy": train_metrics["categorical_accuracy"],
                                    "test_categorical_accuracy": test_metrics["categorical_accuracy"],
                                    "n_train_numeric_masked": train_metrics["n_numeric_masked"],
                                    "n_test_numeric_masked": test_metrics["n_numeric_masked"],
                                    "n_train_categorical_masked": train_metrics["n_categorical_masked"],
                                    "n_test_categorical_masked": test_metrics["n_categorical_masked"],
                                    "fit_runtime_seconds": float(fit_runtime_seconds),
                                    "impute_runtime_seconds": float(impute_runtime_seconds),
                                    "total_runtime_seconds": float(
                                        fit_runtime_seconds + impute_runtime_seconds
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
        save_dataframe(task_df, raw_output_path(EXPERIMENT_NAME, task_kind))

        summary = aggregate_success_rows(
            task_df,
            group_cols=("model_name", "prox_method", "variant"),
            metric_cols=(
                "test_numeric_rmse",
                "test_categorical_accuracy",
                "train_numeric_rmse",
                "train_categorical_accuracy",
                "fit_runtime_seconds",
                "impute_runtime_seconds",
                "total_runtime_seconds",
            ),
        )
        out_dir = compiled_output_dir(EXPERIMENT_NAME, task_kind)
        save_dataframe(summary, out_dir / "summary.csv")
        model_comparison = aggregate_success_rows(
            task_df,
            group_cols=("model_name",),
            metric_cols=(
                "test_numeric_rmse",
                "test_categorical_accuracy",
                "train_numeric_rmse",
                "train_categorical_accuracy",
                "fit_runtime_seconds",
                "impute_runtime_seconds",
                "total_runtime_seconds",
            ),
        )
        save_dataframe(model_comparison, out_dir / "model_comparison.csv")
        if not summary.empty:
            save_plot_dataframe(
                model_comparison.sort_values("test_numeric_rmse"),
                x="model_name",
                y="test_numeric_rmse",
                title=f"{task_kind.title()} Imputation RMSE",
                ylabel="Test numeric RMSE",
                path=out_dir / "test_numeric_rmse.png",
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--n-estimators", type=int, default=128)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--n-iterations", type=int, default=3)
    return parser


if __name__ == "__main__":
    frame = run(build_parser().parse_args())
    compile_outputs(frame)
