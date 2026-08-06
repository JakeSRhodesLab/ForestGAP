from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from forestgap import plot_mds
from experiments.common import (
    add_common_arguments,
    available_model_names,
    aggregate_success_rows,
    build_proximity_model,
    compiled_output_dir,
    iter_openml_suite,
    parse_random_states,
    prepare_dataset,
    proximity_methods_for_model,
    log_progress,
    raw_output_path,
    record_exception,
    requested_model_names,
    requested_tasks,
    run_module_parallel,
    selected_model_names,
    selected_task_ids,
    setup_selected,
    rough_fill_train_test,
    save_dataframe,
    unsupported_model_row,
)


EXPERIMENT_NAME = "mds_visualization"


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for task_kind in requested_tasks(args.task):
        log_progress(f"[{EXPERIMENT_NAME}] task_kind={task_kind}")
        for task_id, task in iter_openml_suite(
            task_kind,
            args.dataset_limit,
            task_ids=selected_task_ids(args, task_kind),
        ):
            dataset_name = str(getattr(task, "dataset_name", f"task_{task_id}"))
            dataset_id = int(getattr(task, "dataset_id", -1))
            for random_state in parse_random_states(args.random_states):
                if not setup_selected(args, task_kind, task_id, random_state):
                    continue
                log_progress(
                    f"[{EXPERIMENT_NAME}] task_kind={task_kind} dataset={dataset_name} "
                    f"task_id={task_id} random_state={random_state}"
                )
                for model_name in selected_model_names(args, requested_model_names()):
                    log_progress(
                        f"[{EXPERIMENT_NAME}] dataset={dataset_name} "
                        f"model={model_name} random_state={random_state}"
                    )
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
                            task_kind, task_id=task_id, task=task,
                            random_state=random_state, max_rows=args.max_rows,
                            test_size=args.test_size,
                        )
                        rng = np.random.RandomState(random_state)
                        n_points = min(args.n_points, prepared.train_size)
                        sample_idx = np.sort(rng.choice(
                            np.arange(prepared.train_size), size=n_points, replace=False,
                        ))
                        X_train = prepared.X_train[sample_idx]
                        y_train = prepared.y_train[sample_idx]
                        X_train_filled, _ = rough_fill_train_test(
                            X_train, prepared.X_test[:1],
                            [idx for idx in prepared.categorical_indices if idx < X_train.shape[1]],
                        )
                    except Exception as exc:
                        error_row = record_exception(
                            experiment_name=EXPERIMENT_NAME, task_kind=task_kind,
                            random_state=random_state, dataset_name=dataset_name,
                            dataset_id=dataset_id, task_id=task_id, error=exc,
                        )
                        error_row["model_name"] = model_name
                        rows.append(error_row)
                        continue

                    for prox_method in proximity_methods_for_model(model_name):
                        log_progress(
                            f"[{EXPERIMENT_NAME}] model={model_name} prox_method={prox_method} "
                            f"dataset={dataset_name}"
                        )
                        try:
                            model = build_proximity_model(
                                prepared.task_kind, model_name=model_name,
                                prox_method=prox_method, random_state=random_state,
                                n_estimators=args.n_estimators, bootstrap=True,
                                oob_score=False,
                                extra_kwargs={"symm_mode": "arithmetic", "non_zero_diagonal": True},
                            )
                            fit_start = time.perf_counter()
                            model.fit(X_train_filled, y_train)
                            fit_runtime_seconds = time.perf_counter() - fit_start
                            plot_start = time.perf_counter()
                            fig = plot_mds(model, y=y_train, random_state=random_state)
                            out_dir = compiled_output_dir(EXPERIMENT_NAME, task_kind)
                            filename = (
                                f"{_safe_slug(model_name)}__{prox_method}__"
                                f"{_safe_slug(prepared.dataset_name)}__seed{random_state}.png"
                            )
                            path = out_dir / filename
                            fig.savefig(path, dpi=180, bbox_inches="tight")
                            plt.close(fig)
                            plot_runtime_seconds = time.perf_counter() - plot_start
                            rows.append({
                                "experiment": EXPERIMENT_NAME, "status": "ok",
                                "task_kind": task_kind, "model_name": model_name,
                                "prox_method": prox_method, "random_state": random_state,
                                "task_id": prepared.task_id, "dataset_id": prepared.dataset_id,
                                "dataset_name": prepared.dataset_name, "n_points": int(n_points),
                                "plot_path": str(path.relative_to(Path.cwd())),
                                "fit_runtime_seconds": float(fit_runtime_seconds),
                                "plot_runtime_seconds": float(plot_runtime_seconds),
                                "total_runtime_seconds": float(fit_runtime_seconds + plot_runtime_seconds),
                            })
                        except Exception as exc:
                            error_row = record_exception(
                                experiment_name=EXPERIMENT_NAME, task_kind=task_kind,
                                random_state=random_state, dataset_name=dataset_name,
                                dataset_id=dataset_id, task_id=task_id, error=exc,
                            )
                            error_row.update({"model_name": model_name, "prox_method": prox_method})
                            rows.append(error_row)

    return pd.DataFrame(rows)


def compile_outputs(df: pd.DataFrame) -> None:
    for task_kind in sorted(df["task_kind"].dropna().unique()):
        task_df = df[df["task_kind"] == task_kind].copy()
        save_dataframe(task_df, raw_output_path(EXPERIMENT_NAME, task_kind))
        out_dir = compiled_output_dir(EXPERIMENT_NAME, task_kind)
        summary = aggregate_success_rows(
            task_df,
            group_cols=("model_name", "prox_method"),
            metric_cols=(
                "n_points",
                "fit_runtime_seconds",
                "plot_runtime_seconds",
                "total_runtime_seconds",
            ),
        )
        save_dataframe(summary, out_dir / "summary.csv")
        save_dataframe(summary, out_dir / "model_comparison.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--n-estimators", type=int, default=96)
    parser.add_argument("--n-points", type=int, default=250)
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    frame = run_module_parallel(
        "experiments.run_mds_visualization",
        parsed_args,
        model_names=requested_model_names(),
    )
    compile_outputs(frame)
