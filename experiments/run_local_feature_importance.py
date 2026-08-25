from __future__ import annotations

import argparse
import os
from pathlib import Path
import site
import sys
import time
from typing import Any


def _remove_user_site_from_sys_path() -> None:
    # Prevent ABI-incompatible user-site wheels from shadowing the conda env.
    candidate_paths: set[str] = set()
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            candidate_paths.add(user_site)
        else:
            candidate_paths.update(str(path) for path in user_site)
    except Exception:
        pass
    candidate_paths.add(str(Path.home() / ".local" / "lib"))

    normalized_candidates = tuple(os.path.normpath(path) for path in candidate_paths)
    sys.path = [
        path
        for path in sys.path
        if path
        and not any(os.path.normpath(path).startswith(candidate) for candidate in normalized_candidates)
    ]

if not any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
    _remove_user_site_from_sys_path()

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from forestgap import local_feature_importance
from experiments.common import (
    add_common_arguments,
    available_model_names,
    aggregate_success_rows,
    build_checkpoint_manifest,
    build_proximity_model,
    checkpoint_store,
    compiled_output_dir,
    compute_fill_values,
    iter_openml_suite,
    parse_random_states,
    prepare_dataset,
    prepared_dataset_hash,
    proximity_methods_for_model,
    log_progress,
    raw_output_path,
    record_exception,
    requested_tasks,
    run_module_parallel,
    selected_model_names,
    selected_task_ids,
    setup_selected,
    requested_model_names,
    rough_fill_train_test,
    save_dataframe,
    save_plot_dataframe,
    seed_checkpoint_unit,
    unsupported_model_row,
    unavailable_data_hash,
)


EXPERIMENT_NAME = "local_feature_importance"


def _checkpoint_unit(
    *,
    task_kind: str,
    task_id: int,
    dataset_id: int,
    random_state: int,
    model_name: str,
    prox_method: str,
) -> dict[str, Any]:
    return {
        "task_kind": task_kind,
        "task_id": int(task_id),
        "dataset_id": int(dataset_id),
        "random_state": int(random_state),
        "model_name": model_name,
        "prox_method": prox_method,
    }


def _checkpoint_config(args: argparse.Namespace, prox_method: str) -> dict[str, Any]:
    return {
        "max_rows": getattr(args, "max_rows", None),
        "test_size": getattr(args, "test_size", 0.3),
        "feature_cap": getattr(args, "feature_cap", 32),
        "n_estimators": getattr(args, "n_estimators", 96),
        "n_repeats": getattr(args, "n_repeats", 5),
        "n_queries": getattr(args, "n_queries", 16),
        "perturbation": "auto",
        "bootstrap": prox_method in {"oob", "rfgap"},
        "oob_score": prox_method == "rfgap",
    }


def _cached_rows(store: Any, manifest: Any, args: argparse.Namespace, prox_method: str) -> list[dict[str, Any]] | None:
    if not getattr(args, "resume", True):
        return None
    cached = store.load(manifest)
    if cached is None or not cached.summary_rows:
        return None
    # Older local-importance exports omitted the relation on some terminal rows.
    # Such rows cannot prove which model/proximity unit produced them.
    if any(row.get("prox_method") != prox_method for row in cached.summary_rows):
        return None
    return cached.summary_rows


def _save_row(store: Any, manifest: Any, row: dict[str, Any]) -> list[dict[str, Any]]:
    return store.save(manifest, [row]).summary_rows


def _probability_drop(
    model: Any,
    X_base: np.ndarray,
    predicted_classes: np.ndarray,
    feature_indices: np.ndarray,
    fill_values: np.ndarray,
) -> np.ndarray:
    perturbed = np.array(X_base, copy=True)
    rows = np.arange(X_base.shape[0])
    perturbed[rows, feature_indices] = fill_values[feature_indices]
    base_prob = np.asarray(model.predict_proba(X_base), dtype=np.float64)
    perturbed_prob = np.asarray(model.predict_proba(perturbed), dtype=np.float64)
    return base_prob[rows, predicted_classes] - perturbed_prob[rows, predicted_classes]


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prox_method = getattr(args, "_prox_method", "original")
    store = checkpoint_store(
        EXPERIMENT_NAME,
        root=getattr(args, "_checkpoint_root", None),
    )

    for task_kind in requested_tasks(args.task):
        log_progress(f"[{EXPERIMENT_NAME}] prox_method={prox_method} task_kind={task_kind}")
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
                    f"[{EXPERIMENT_NAME}] dataset={dataset_name} task_id={task_id} "
                    f"random_state={random_state} prox_method={prox_method}"
                )
                for model_name in selected_model_names(args, requested_model_names()):
                    log_progress(
                        f"[{EXPERIMENT_NAME}] model={model_name} "
                        f"prox_method={prox_method} dataset={dataset_name}"
                    )
                    if prox_method not in proximity_methods_for_model(model_name):
                        continue
                    unit = _checkpoint_unit(
                        task_kind=task_kind,
                        task_id=task_id,
                        dataset_id=dataset_id,
                        random_state=random_state,
                        model_name=model_name,
                        prox_method=prox_method,
                    )
                    scientific_config = _checkpoint_config(args, prox_method)

                    unsupported_reason: str | None = None
                    if task_kind == "regression":
                        unsupported_reason = (
                            "the standalone local_feature_importance application is "
                            "classification-only"
                        )
                    elif model_name == "HonestForest":
                        unsupported_reason = (
                            "HonestForest is unstable for local feature importance on this "
                            "benchmark in the current treeple build (native segfault during fit)."
                        )
                    elif model_name not in available_model_names(task_kind):
                        unsupported_reason = f"{model_name} is not available for {task_kind}."

                    if unsupported_reason is not None:
                        manifest = build_checkpoint_manifest(
                            experiment=EXPERIMENT_NAME,
                            runner_file=__file__,
                            unit=unit,
                            scientific_config=scientific_config,
                            data_hash=unavailable_data_hash(
                                task_kind=task_kind,
                                task_id=task_id,
                                dataset_id=dataset_id,
                            ),
                        )
                        cached_rows = _cached_rows(store, manifest, args, prox_method)
                        if cached_rows is not None:
                            rows.extend(cached_rows)
                            continue
                        seed_checkpoint_unit(manifest)
                        unsupported_row = unsupported_model_row(
                            experiment_name=EXPERIMENT_NAME,
                            task_kind=task_kind,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            dataset_id=dataset_id,
                            task_id=task_id,
                            random_state=random_state,
                            reason=unsupported_reason,
                            extra_fields={"prox_method": prox_method},
                        )
                        rows.extend(_save_row(store, manifest, unsupported_row))
                        continue

                    manifest: Any = None
                    prepared: Any = None
                    try:
                        prepared = prepare_dataset(
                            task_kind,
                            task_id=task_id,
                            task=task,
                            random_state=random_state,
                            max_rows=args.max_rows,
                            test_size=args.test_size,
                            feature_cap=args.feature_cap,
                        )
                        manifest = build_checkpoint_manifest(
                            experiment=EXPERIMENT_NAME,
                            runner_file=__file__,
                            unit=unit,
                            scientific_config=scientific_config,
                            data_hash=prepared_dataset_hash(prepared),
                        )
                        cached_rows = _cached_rows(store, manifest, args, prox_method)
                        if cached_rows is not None:
                            rows.extend(cached_rows)
                            continue
                        seed_checkpoint_unit(manifest)
                        X_train_filled, X_test_filled = rough_fill_train_test(
                            prepared.X_train,
                            prepared.X_test,
                            prepared.categorical_indices,
                        )
                        fill_values = compute_fill_values(prepared.X_train, prepared.categorical_indices)
                        model = build_proximity_model(
                            prepared.task_kind,
                            model_name=model_name,
                            prox_method=prox_method,
                            random_state=random_state,
                            n_estimators=args.n_estimators,
                            bootstrap=prox_method in {"oob", "rfgap"},
                            oob_score=prox_method == "rfgap",
                        )
                        fit_start = time.perf_counter()
                        model.fit(X_train_filled, prepared.y_train)
                        fit_runtime_seconds = time.perf_counter() - fit_start

                        rng = np.random.RandomState(random_state)
                        n_queries = min(args.n_queries, X_test_filled.shape[0])
                        query_idx = np.sort(rng.choice(np.arange(X_test_filled.shape[0]), size=n_queries, replace=False))
                        X_query = X_test_filled[query_idx]
                        y_query = prepared.y_test[query_idx]

                        importance_start = time.perf_counter()
                        importance_pred = np.asarray(
                            local_feature_importance(
                                model,
                                X_query,
                                perturbation="auto",
                                n_repeats=args.n_repeats,
                                random_state=random_state,
                            ),
                            dtype=np.float64,
                        )
                        importance_true = np.asarray(
                            local_feature_importance(
                                model,
                                X_query,
                                labels=y_query,
                                perturbation="auto",
                                n_repeats=args.n_repeats,
                                random_state=random_state,
                            ),
                            dtype=np.float64,
                        )
                        importance_runtime_seconds = time.perf_counter() - importance_start

                        evaluation_start = time.perf_counter()
                        pred_proba = np.asarray(model.predict_proba(X_query), dtype=np.float64)
                        pred_class = np.argmax(pred_proba, axis=1)
                        top_features = np.argmax(np.abs(importance_pred), axis=1)
                        random_features = rng.randint(0, X_query.shape[1], size=X_query.shape[0])
                        top_drop = _probability_drop(model, X_query, pred_class, top_features, fill_values)
                        random_drop = _probability_drop(model, X_query, pred_class, random_features, fill_values)
                        evaluation_runtime_seconds = time.perf_counter() - evaluation_start

                        misclassified = pred_class != y_query
                        result_row = {
                            "experiment": EXPERIMENT_NAME,
                            "status": "ok",
                            "task_kind": task_kind,
                            "model_name": model_name,
                            "prox_method": prox_method,
                            "random_state": random_state,
                            "task_id": prepared.task_id,
                            "dataset_id": prepared.dataset_id,
                            "dataset_name": prepared.dataset_name,
                            "n_queries": int(n_queries),
                            "n_features": int(X_query.shape[1]),
                            "mean_abs_importance_pred": float(np.mean(np.abs(importance_pred))),
                            "mean_abs_importance_true": float(np.mean(np.abs(importance_true))),
                            "true_vs_pred_mean_abs_delta": float(
                                np.mean(np.abs(importance_true - importance_pred))
                            ),
                            "top_feature_drop_mean": float(np.mean(top_drop)),
                            "random_feature_drop_mean": float(np.mean(random_drop)),
                            "top_feature_win_rate": float(np.mean(top_drop > random_drop)),
                            "misclassified_fraction": float(np.mean(misclassified)),
                            "fit_runtime_seconds": float(fit_runtime_seconds),
                            "importance_runtime_seconds": float(importance_runtime_seconds),
                            "evaluation_runtime_seconds": float(evaluation_runtime_seconds),
                            "total_runtime_seconds": float(
                                fit_runtime_seconds
                                + importance_runtime_seconds
                                + evaluation_runtime_seconds
                            ),
                        }
                        rows.extend(_save_row(store, manifest, result_row))
                    except Exception as exc:
                        if manifest is None:
                            manifest = build_checkpoint_manifest(
                                experiment=EXPERIMENT_NAME,
                                runner_file=__file__,
                                unit=unit,
                                scientific_config=scientific_config,
                                data_hash=unavailable_data_hash(
                                    task_kind=task_kind,
                                    task_id=task_id,
                                    dataset_id=dataset_id,
                                ),
                            )
                            seed_checkpoint_unit(manifest)
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
                        error_row["prox_method"] = prox_method
                        rows.extend(_save_row(store, manifest, error_row))

    return pd.DataFrame(rows)


def run_all_relations(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for relation in (
        "original", "oob", "rfgap", "honest_gap", "kerf", "path",
        "englund", "random_height",
    ):
        log_progress(f"[{EXPERIMENT_NAME}] starting relation={relation}")
        args._prox_method = relation
        frames.append(run(args))
    return pd.concat(frames, ignore_index=True)


def compile_outputs(df: pd.DataFrame) -> None:
    if df.empty or "task_kind" not in df.columns:
        return

    for task_kind in sorted(df["task_kind"].dropna().unique()):
        task_df = df[df["task_kind"] == task_kind].copy()
        save_dataframe(task_df, raw_output_path(EXPERIMENT_NAME, task_kind))
        out_dir = compiled_output_dir(EXPERIMENT_NAME, task_kind)
        metric_cols = (
            "true_vs_pred_mean_abs_delta",
            "top_feature_drop_mean",
            "random_feature_drop_mean",
            "top_feature_win_rate",
            "misclassified_fraction",
            "fit_runtime_seconds",
            "importance_runtime_seconds",
            "evaluation_runtime_seconds",
            "total_runtime_seconds",
        )
        summary = aggregate_success_rows(
            task_df,
            group_cols=("model_name", "prox_method"),
            metric_cols=metric_cols,
        )
        save_dataframe(summary, out_dir / "summary.csv")
        save_dataframe(summary, out_dir / "model_comparison.csv")
        if not summary.empty:
            save_plot_dataframe(
                summary.sort_values("top_feature_win_rate", ascending=False),
                x="model_name",
                y="top_feature_win_rate",
                title="Local Importance Top-Feature Win Rate",
                ylabel="Win rate",
                path=out_dir / "top_feature_win_rate.png",
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--n-estimators", type=int, default=96)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=16)
    parser.add_argument("--feature-cap", type=int, default=32)
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    frame = run_module_parallel(
        "experiments.run_local_feature_importance",
        parsed_args,
        runner_name="run_all_relations",
        model_names=requested_model_names(),
    )
    compile_outputs(frame)
