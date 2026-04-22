from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    add_common_arguments,
    available_model_names,
    aggregate_success_rows,
    build_proximity_model,
    compiled_output_dir,
    corrupt_dataset,
    iter_openml_suite,
    outlier_metrics,
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


EXPERIMENT_NAME = "outlier_detection"


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for task_kind in requested_tasks(args.task):
        modes = ["unsupervised"] if task_kind == "regression" else ["supervised", "unsupervised"]
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
                        X_train_filled, _ = rough_fill_train_test(
                            prepared.X_train,
                            prepared.X_test,
                            prepared.categorical_indices,
                        )
                        X_corrupt, corruption_labels = corrupt_dataset(
                            X_train_filled,
                            prepared.categorical_indices,
                            random_state=random_state,
                            fraction=args.outlier_fraction,
                        )
                        for prox_method in ("original", "oob", "rfgap"):
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
                            model.fit(X_corrupt, prepared.y_train)
                            fit_runtime_seconds = time.perf_counter() - fit_start
                            for scoring_method in ("breiman", "ene"):
                                for mode in modes:
                                    score_labels = prepared.y_train if mode == "supervised" else None
                                    score_start = time.perf_counter()
                                    scores = model.outlier_scores(score_labels, method=scoring_method)
                                    score_runtime_seconds = time.perf_counter() - score_start
                                    metrics = outlier_metrics(scores, corruption_labels, args.outlier_fraction)
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
                                            "score_method": scoring_method,
                                            "label_mode": mode,
                                            "outlier_fraction": args.outlier_fraction,
                                            "roc_auc": metrics["roc_auc"],
                                            "average_precision": metrics["average_precision"],
                                            "topk_precision": metrics["topk_precision"],
                                            "fit_runtime_seconds": float(fit_runtime_seconds),
                                            "score_runtime_seconds": float(score_runtime_seconds),
                                            "total_runtime_seconds": float(
                                                fit_runtime_seconds + score_runtime_seconds
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
            group_cols=("model_name", "prox_method", "score_method", "label_mode"),
            metric_cols=(
                "roc_auc",
                "average_precision",
                "topk_precision",
                "fit_runtime_seconds",
                "score_runtime_seconds",
                "total_runtime_seconds",
            ),
        )
        out_dir = compiled_output_dir(EXPERIMENT_NAME, task_kind)
        save_dataframe(summary, out_dir / "summary.csv")
        model_comparison = aggregate_success_rows(
            task_df,
            group_cols=("model_name",),
            metric_cols=(
                "roc_auc",
                "average_precision",
                "topk_precision",
                "fit_runtime_seconds",
                "score_runtime_seconds",
                "total_runtime_seconds",
            ),
        )
        save_dataframe(model_comparison, out_dir / "model_comparison.csv")
        if not model_comparison.empty:
            plot_df = model_comparison.sort_values("average_precision", ascending=False).copy()
            plot_df["config"] = (
                plot_df["model_name"]
            )
            save_plot_dataframe(
                plot_df,
                x="config",
                y="average_precision",
                title=f"{task_kind.title()} Outlier Detection Average Precision",
                ylabel="Average precision",
                path=out_dir / "average_precision.png",
                rotation=50,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--n-estimators", type=int, default=128)
    parser.add_argument("--outlier-fraction", type=float, default=0.05)
    return parser


if __name__ == "__main__":
    frame = run(build_parser().parse_args())
    compile_outputs(frame)
