from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import site
import sys
import time
from typing import Any
import warnings


def _remove_user_site_from_sys_path() -> None:
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

from experiments.common import (
    add_common_arguments,
    available_model_names,
    aggregate_success_rows,
    build_checkpoint_manifest,
    build_proximity_model,
    checkpoint_store,
    compiled_output_dir,
    proximity_methods_for_model,
    requested_model_names,
    prepare_dataset,
    prepared_dataset_hash,
    raw_output_path,
    record_exception,
    run_module_parallel,
    save_dataframe,
    requested_tasks,
    save_plot_dataframe,
    seed_checkpoint_unit,
    selected_model_names,
    selected_task_ids,
    setup_selected,
    suite_metadata,
    iter_openml_suite,
    parse_random_states,
    rough_fill_train_test,
    log_progress,
    unsupported_model_row,
    unavailable_data_hash,
)


EXPERIMENT_NAME = "prediction_identity"
PROX_METHOD = "rfgap"
RAW_COLUMN_ORDER = (
    "experiment",
    "status",
    "task_kind",
    "suite_name",
    "dataset_id",
    "dataset_name",
    "task_id",
    "random_state",
    "model_name",
    "prox_method",
    "train_size",
    "test_size",
    "n_features",
    "n_categorical_features",
    "test_mean_abs_diff",
    "test_max_abs_diff",
    "oob_mean_abs_diff",
    "oob_max_abs_diff",
    "test_argmax_agreement",
    "oob_argmax_agreement",
    "test_accuracy",
    "test_rmse_diff",
    "test_r2",
    "oob_valid_count",
    "oob_total_count",
    "oob_valid_fraction",
    "oob_missing_count",
    "oob_missing_fraction",
    "oob_comparison_status",
    "fit_runtime_seconds",
    "prox_test_runtime_seconds",
    "forest_test_runtime_seconds",
    "prox_oob_runtime_seconds",
    "total_runtime_seconds",
    "reason",
    "error_type",
    "error_message",
    "traceback",
)


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


def _checkpoint_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_rows": getattr(args, "max_rows", None),
        "test_size": getattr(args, "test_size", 0.3),
        "n_estimators": getattr(args, "n_estimators", 128),
        "bootstrap": True,
        "oob_score": True,
    }


def _cached_rows(store: Any, manifest: Any, args: argparse.Namespace) -> list[dict[str, Any]] | None:
    if not getattr(args, "resume", True):
        return None
    cached = store.load(manifest)
    if cached is None or not cached.summary_rows:
        return None
    return cached.summary_rows


def _save_row(store: Any, manifest: Any, row: dict[str, Any]) -> list[dict[str, Any]]:
    return store.save(manifest, [row]).summary_rows


def _ordered_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    extra_cols = [column for column in df.columns if column not in RAW_COLUMN_ORDER]
    return df.reindex(columns=list(RAW_COLUMN_ORDER) + extra_cols)


def _prepared_fields(prepared: Any, prox_method: str = PROX_METHOD) -> dict[str, Any]:
    return {
        "suite_name": prepared.suite_name,
        "prox_method": prox_method,
        "train_size": prepared.train_size,
        "test_size": prepared.test_size,
        "n_features": len(prepared.feature_names),
        "n_categorical_features": len(prepared.categorical_indices),
    }


def _runtime_fields(
    *,
    fit_runtime_seconds: float,
    prox_test_runtime_seconds: float,
    forest_test_runtime_seconds: float,
    prox_oob_runtime_seconds: float,
) -> dict[str, float]:
    total_runtime_seconds = (
        fit_runtime_seconds
        + prox_test_runtime_seconds
        + forest_test_runtime_seconds
        + prox_oob_runtime_seconds
    )
    return {
        "fit_runtime_seconds": float(fit_runtime_seconds),
        "prox_test_runtime_seconds": float(prox_test_runtime_seconds),
        "forest_test_runtime_seconds": float(forest_test_runtime_seconds),
        "prox_oob_runtime_seconds": float(prox_oob_runtime_seconds),
        "total_runtime_seconds": float(total_runtime_seconds),
    }


@contextmanager
def _suppress_regression_oob_target_warning(task_kind: str):
    with warnings.catch_warnings():
        if task_kind == "regression":
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"The number of unique classes is greater than 50% of the number "
                    r"of samples\. `y` could represent a regression problem, not a "
                    r"classification problem\."
                ),
                category=UserWarning,
                module=r"sklearn\.ensemble\._forest",
            )
        yield


def _oob_support_mask(model: Any, n_rows: int) -> np.ndarray:
    oob_mask = getattr(model, "oob_mask_", None)
    if oob_mask is None:
        return np.ones(n_rows, dtype=bool)
    oob_arr = np.asarray(oob_mask)
    if oob_arr.ndim != 2 or oob_arr.shape[0] != n_rows:
        return np.ones(n_rows, dtype=bool)
    return np.asarray(oob_arr.sum(axis=1) > 0, dtype=bool)


def _classification_oob_valid_mask(
    model: Any,
    prox_oob: np.ndarray,
    forest_oob: np.ndarray,
) -> np.ndarray:
    if prox_oob.shape[0] != forest_oob.shape[0]:
        raise ValueError(
            "prox_predict() and oob_decision_function_ returned different row counts: "
            f"{prox_oob.shape[0]} vs {forest_oob.shape[0]}."
        )
    support_mask = _oob_support_mask(model, int(forest_oob.shape[0]))
    finite_mask = np.all(np.isfinite(forest_oob), axis=1) & np.all(np.isfinite(prox_oob), axis=1)
    return support_mask & finite_mask


def _regression_oob_valid_mask(
    model: Any,
    prox_oob: np.ndarray,
    forest_oob: np.ndarray,
) -> np.ndarray:
    if prox_oob.shape[0] != forest_oob.shape[0]:
        raise ValueError(
            "prox_predict() and oob_prediction_ returned different row counts: "
            f"{prox_oob.shape[0]} vs {forest_oob.shape[0]}."
        )
    support_mask = _oob_support_mask(model, int(forest_oob.shape[0]))
    finite_mask = np.isfinite(forest_oob) & np.isfinite(prox_oob)
    return support_mask & finite_mask


def _classification_identity_metrics(
    prox_prob: np.ndarray,
    forest_prob: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    diff = np.asarray(prox_prob, dtype=np.float64) - np.asarray(forest_prob, dtype=np.float64)
    forest_argmax = np.argmax(forest_prob, axis=1)
    prox_argmax = np.argmax(prox_prob, axis=1)
    return {
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "argmax_agreement": float(np.mean(prox_argmax == forest_argmax)),
        "accuracy": float(np.mean(forest_argmax == np.asarray(y_true))),
    }


def _regression_identity_metrics(
    prox_pred: np.ndarray,
    forest_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    diff = np.asarray(prox_pred, dtype=np.float64) - np.asarray(forest_pred, dtype=np.float64)
    y_arr = np.asarray(y_true, dtype=np.float64)
    forest_arr = np.asarray(forest_pred, dtype=np.float64)
    residual = y_arr - forest_arr
    denominator = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r2 = float("nan")
    if denominator > 0:
        r2 = float(1.0 - (np.sum(residual**2) / denominator))
    return {
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "rmse_diff": float(np.sqrt(np.mean(diff**2))),
        "r2": r2,
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    store = checkpoint_store(
        EXPERIMENT_NAME,
        root=getattr(args, "_checkpoint_root", None),
    )
    scientific_config = _checkpoint_config(args)

    for task_kind in requested_tasks(args.task):
        log_progress(f"[{EXPERIMENT_NAME}] task_kind={task_kind}")
        _, suite_name = suite_metadata(task_kind)
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
                    prox_method = "honest_gap" if model_name == "HonestForest" else PROX_METHOD
                    log_progress(
                        f"[{EXPERIMENT_NAME}] dataset={dataset_name} "
                        f"model={model_name} prox_method={prox_method}"
                    )
                    unit = _checkpoint_unit(
                        task_kind=task_kind,
                        task_id=task_id,
                        dataset_id=dataset_id,
                        random_state=random_state,
                        model_name=model_name,
                        prox_method=prox_method,
                    )

                    unsupported_reason: str | None = None
                    if model_name not in available_model_names(task_kind):
                        unsupported_reason = f"{model_name} is not available for {task_kind}."
                    elif prox_method not in proximity_methods_for_model(model_name):
                        unsupported_reason = (
                            f"{prox_method} is not a valid prediction-identity "
                            f"relation for {model_name}."
                        )

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
                        cached_rows = _cached_rows(store, manifest, args)
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
                            extra_fields={
                                "suite_name": suite_name,
                                "prox_method": prox_method,
                            },
                        )
                        rows.extend(_save_row(store, manifest, unsupported_row))
                        continue
                    prepared: Any = None
                    manifest: Any = None
                    try:
                        prepared = prepare_dataset(
                            task_kind,
                            task_id=task_id,
                            task=task,
                            random_state=random_state,
                            max_rows=args.max_rows,
                            test_size=args.test_size,
                        )
                        manifest = build_checkpoint_manifest(
                            experiment=EXPERIMENT_NAME,
                            runner_file=__file__,
                            unit=unit,
                            scientific_config=scientific_config,
                            data_hash=prepared_dataset_hash(prepared),
                        )
                        cached_rows = _cached_rows(store, manifest, args)
                        if cached_rows is not None:
                            rows.extend(cached_rows)
                            continue
                        seed_checkpoint_unit(manifest)
                        X_train_filled, X_test_filled = rough_fill_train_test(
                            prepared.X_train,
                            prepared.X_test,
                            prepared.categorical_indices,
                        )
                        model = build_proximity_model(
                            task_kind,
                            model_name=model_name,
                            prox_method=prox_method,
                            random_state=random_state,
                            n_estimators=args.n_estimators,
                            bootstrap=True,
                            oob_score=True,
                        )
                        fit_start = time.perf_counter()
                        with _suppress_regression_oob_target_warning(task_kind):
                            model.fit(X_train_filled, prepared.y_train)
                        fit_runtime_seconds = time.perf_counter() - fit_start

                        if task_kind == "classification":
                            prox_test_start = time.perf_counter()
                            prox_test = np.asarray(model.prox_predict(X_test_filled), dtype=np.float64)
                            prox_test_runtime_seconds = time.perf_counter() - prox_test_start
                            forest_test_start = time.perf_counter()
                            forest_test = np.asarray(model.predict_proba(X_test_filled), dtype=np.float64)
                            forest_test_runtime_seconds = time.perf_counter() - forest_test_start
                            test_metrics = _classification_identity_metrics(
                                prox_test,
                                forest_test,
                                prepared.y_test,
                            )
                            task_metrics = {
                                "test_argmax_agreement": test_metrics["argmax_agreement"],
                                "test_accuracy": test_metrics["accuracy"],
                            }
                            prox_oob_start = time.perf_counter()
                            prox_oob = np.asarray(model.prox_predict(), dtype=np.float64)
                            prox_oob_runtime_seconds = time.perf_counter() - prox_oob_start
                            forest_oob_raw = getattr(model, "oob_decision_function_", None)
                            runtime_fields = _runtime_fields(
                                fit_runtime_seconds=fit_runtime_seconds,
                                prox_test_runtime_seconds=prox_test_runtime_seconds,
                                forest_test_runtime_seconds=forest_test_runtime_seconds,
                                prox_oob_runtime_seconds=prox_oob_runtime_seconds,
                            )
                            if forest_oob_raw is None:
                                unsupported_row = unsupported_model_row(
                                    experiment_name=EXPERIMENT_NAME,
                                    task_kind=task_kind,
                                    model_name=model_name,
                                    dataset_name=prepared.dataset_name,
                                    dataset_id=prepared.dataset_id,
                                    task_id=prepared.task_id,
                                    random_state=random_state,
                                    reason=(
                                        "Missing oob_decision_function_ after fitting "
                                        "with oob_score=True."
                                    ),
                                    extra_fields={
                                        **_prepared_fields(prepared, prox_method),
                                        "test_mean_abs_diff": test_metrics["mean_abs_diff"],
                                        "test_max_abs_diff": test_metrics["max_abs_diff"],
                                        "oob_valid_count": 0,
                                        "oob_total_count": int(prepared.train_size),
                                        "oob_valid_fraction": 0.0,
                                        "oob_missing_count": int(prepared.train_size),
                                        "oob_missing_fraction": 1.0,
                                        "oob_comparison_status": "missing_native_oob",
                                        **task_metrics,
                                        **runtime_fields,
                                    },
                                )
                                rows.extend(_save_row(store, manifest, unsupported_row))
                                continue
                            forest_oob = np.asarray(forest_oob_raw, dtype=np.float64)
                            oob_valid_mask = _classification_oob_valid_mask(
                                model,
                                prox_oob,
                                forest_oob,
                            )
                            oob_valid_count = int(np.sum(oob_valid_mask))
                            oob_missing_count = int(prepared.train_size - oob_valid_count)
                            oob_missing_fraction = float(oob_missing_count / max(1, prepared.train_size))
                            if oob_valid_count == 0:
                                unsupported_row = unsupported_model_row(
                                    experiment_name=EXPERIMENT_NAME,
                                    task_kind=task_kind,
                                    model_name=model_name,
                                    dataset_name=prepared.dataset_name,
                                    dataset_id=prepared.dataset_id,
                                    task_id=prepared.task_id,
                                    random_state=random_state,
                                    reason=(
                                        "No supported finite overlap between prox_predict() "
                                        "and oob_decision_function_."
                                    ),
                                    extra_fields={
                                        **_prepared_fields(prepared, prox_method),
                                        "test_mean_abs_diff": test_metrics["mean_abs_diff"],
                                        "test_max_abs_diff": test_metrics["max_abs_diff"],
                                        "oob_valid_count": 0,
                                        "oob_total_count": int(prepared.train_size),
                                        "oob_valid_fraction": 0.0,
                                        "oob_missing_count": int(prepared.train_size),
                                        "oob_missing_fraction": 1.0,
                                        "oob_comparison_status": "no_valid_oob_rows",
                                        **task_metrics,
                                        **runtime_fields,
                                    },
                                )
                                rows.extend(_save_row(store, manifest, unsupported_row))
                                continue
                            oob_metrics = _classification_identity_metrics(
                                prox_oob[oob_valid_mask],
                                forest_oob[oob_valid_mask],
                                np.asarray(prepared.y_train)[oob_valid_mask],
                            )
                            task_metrics = {
                                **task_metrics,
                                "oob_argmax_agreement": oob_metrics["argmax_agreement"],
                            }
                            oob_comparison_status = (
                                "all_oob_rows" if oob_valid_count == prepared.train_size else "partial_oob_rows"
                            )
                        else:
                            prox_test_start = time.perf_counter()
                            prox_test = np.asarray(model.prox_predict(X_test_filled), dtype=np.float64).ravel()
                            prox_test_runtime_seconds = time.perf_counter() - prox_test_start
                            forest_test_start = time.perf_counter()
                            forest_test = np.asarray(model.predict(X_test_filled), dtype=np.float64).ravel()
                            forest_test_runtime_seconds = time.perf_counter() - forest_test_start
                            test_metrics = _regression_identity_metrics(
                                prox_test,
                                forest_test,
                                prepared.y_test,
                            )
                            task_metrics = {
                                "test_rmse_diff": test_metrics["rmse_diff"],
                                "test_r2": test_metrics["r2"],
                            }
                            prox_oob_start = time.perf_counter()
                            prox_oob = np.asarray(model.prox_predict(), dtype=np.float64).ravel()
                            prox_oob_runtime_seconds = time.perf_counter() - prox_oob_start
                            forest_oob_raw = getattr(model, "oob_prediction_", None)
                            runtime_fields = _runtime_fields(
                                fit_runtime_seconds=fit_runtime_seconds,
                                prox_test_runtime_seconds=prox_test_runtime_seconds,
                                forest_test_runtime_seconds=forest_test_runtime_seconds,
                                prox_oob_runtime_seconds=prox_oob_runtime_seconds,
                            )
                            if forest_oob_raw is None:
                                unsupported_row = unsupported_model_row(
                                    experiment_name=EXPERIMENT_NAME,
                                    task_kind=task_kind,
                                    model_name=model_name,
                                    dataset_name=prepared.dataset_name,
                                    dataset_id=prepared.dataset_id,
                                    task_id=prepared.task_id,
                                    random_state=random_state,
                                    reason=(
                                        "Missing oob_prediction_ after fitting with "
                                        "oob_score=True."
                                    ),
                                    extra_fields={
                                        **_prepared_fields(prepared, prox_method),
                                        "test_mean_abs_diff": test_metrics["mean_abs_diff"],
                                        "test_max_abs_diff": test_metrics["max_abs_diff"],
                                        "oob_valid_count": 0,
                                        "oob_total_count": int(prepared.train_size),
                                        "oob_valid_fraction": 0.0,
                                        "oob_missing_count": int(prepared.train_size),
                                        "oob_missing_fraction": 1.0,
                                        "oob_comparison_status": "missing_native_oob",
                                        **task_metrics,
                                        **runtime_fields,
                                    },
                                )
                                rows.extend(_save_row(store, manifest, unsupported_row))
                                continue
                            forest_oob = np.asarray(forest_oob_raw, dtype=np.float64).ravel()
                            oob_valid_mask = _regression_oob_valid_mask(
                                model,
                                prox_oob,
                                forest_oob,
                            )
                            oob_valid_count = int(np.sum(oob_valid_mask))
                            oob_missing_count = int(prepared.train_size - oob_valid_count)
                            oob_missing_fraction = float(oob_missing_count / max(1, prepared.train_size))
                            if oob_valid_count == 0:
                                unsupported_row = unsupported_model_row(
                                    experiment_name=EXPERIMENT_NAME,
                                    task_kind=task_kind,
                                    model_name=model_name,
                                    dataset_name=prepared.dataset_name,
                                    dataset_id=prepared.dataset_id,
                                    task_id=prepared.task_id,
                                    random_state=random_state,
                                    reason=(
                                        "No supported finite overlap between prox_predict() "
                                        "and oob_prediction_."
                                    ),
                                    extra_fields={
                                        **_prepared_fields(prepared, prox_method),
                                        "test_mean_abs_diff": test_metrics["mean_abs_diff"],
                                        "test_max_abs_diff": test_metrics["max_abs_diff"],
                                        "oob_valid_count": 0,
                                        "oob_total_count": int(prepared.train_size),
                                        "oob_valid_fraction": 0.0,
                                        "oob_missing_count": int(prepared.train_size),
                                        "oob_missing_fraction": 1.0,
                                        "oob_comparison_status": "no_valid_oob_rows",
                                        **task_metrics,
                                        **runtime_fields,
                                    },
                                )
                                rows.extend(_save_row(store, manifest, unsupported_row))
                                continue
                            oob_metrics = _regression_identity_metrics(
                                prox_oob[oob_valid_mask],
                                forest_oob[oob_valid_mask],
                                np.asarray(prepared.y_train)[oob_valid_mask],
                            )
                            oob_comparison_status = (
                                "all_oob_rows" if oob_valid_count == prepared.train_size else "partial_oob_rows"
                            )

                        result_row = {
                            "experiment": EXPERIMENT_NAME,
                            "status": "ok",
                            "task_kind": task_kind,
                            "suite_name": prepared.suite_name,
                            "prox_method": prox_method,
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
                            "oob_valid_count": oob_valid_count,
                            "oob_total_count": int(prepared.train_size),
                            "oob_valid_fraction": float(
                                oob_valid_count / max(1, prepared.train_size)
                            ),
                            "oob_missing_count": oob_missing_count,
                            "oob_missing_fraction": oob_missing_fraction,
                            "oob_comparison_status": oob_comparison_status,
                            **runtime_fields,
                            **task_metrics,
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
                        extra_fields: dict[str, Any] = {
                            "model_name": model_name,
                            "suite_name": suite_name,
                            "prox_method": prox_method,
                        }
                        if prepared is not None:
                            extra_fields.update(_prepared_fields(prepared, prox_method))
                        error_row = record_exception(
                            experiment_name=EXPERIMENT_NAME,
                            task_kind=task_kind,
                            random_state=random_state,
                            dataset_name=dataset_name,
                            dataset_id=dataset_id,
                            task_id=task_id,
                            error=exc,
                            extra_fields=extra_fields,
                        )
                        rows.extend(_save_row(store, manifest, error_row))

    return pd.DataFrame(rows)


def compile_outputs(df: pd.DataFrame) -> None:
    if df.empty or "task_kind" not in df.columns:
        return

    df = _ordered_raw_frame(df)
    for task_kind in sorted(df["task_kind"].dropna().unique()):
        task_df = df[df["task_kind"] == task_kind].copy()
        raw_path = raw_output_path(EXPERIMENT_NAME, task_kind)
        save_dataframe(task_df, raw_path)

        if task_kind == "classification":
            metric_cols = (
                "test_mean_abs_diff",
                "test_max_abs_diff",
                "oob_mean_abs_diff",
                "oob_max_abs_diff",
                "test_argmax_agreement",
                "oob_argmax_agreement",
                "test_accuracy",
                "oob_valid_fraction",
                "oob_missing_fraction",
                "fit_runtime_seconds",
                "prox_test_runtime_seconds",
                "forest_test_runtime_seconds",
                "prox_oob_runtime_seconds",
                "total_runtime_seconds",
            )
        else:
            metric_cols = (
                "test_mean_abs_diff",
                "test_max_abs_diff",
                "oob_mean_abs_diff",
                "oob_max_abs_diff",
                "test_rmse_diff",
                "test_r2",
                "oob_valid_fraction",
                "oob_missing_fraction",
                "fit_runtime_seconds",
                "prox_test_runtime_seconds",
                "forest_test_runtime_seconds",
                "prox_oob_runtime_seconds",
                "total_runtime_seconds",
            )

        summary = aggregate_success_rows(
            task_df,
            group_cols=("model_name",),
            metric_cols=metric_cols,
        )
        out_dir = compiled_output_dir(EXPERIMENT_NAME, task_kind)
        save_dataframe(summary, out_dir / "summary.csv")
        save_dataframe(summary.sort_values("oob_max_abs_diff"), out_dir / "model_comparison.csv")
        if not summary.empty:
            save_plot_dataframe(
                summary.sort_values("oob_max_abs_diff"),
                x="model_name",
                y="oob_max_abs_diff",
                title=f"{task_kind.title()} RF-GAP OOB Prediction Identity",
                ylabel="Mean OOB max absolute difference",
                path=out_dir / "oob_max_abs_diff.png",
            )
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
    parsed_args = build_parser().parse_args()
    frame = run_module_parallel(
        "experiments.run_prediction_identity",
        parsed_args,
        model_names=requested_model_names(),
    )
    compile_outputs(frame)
