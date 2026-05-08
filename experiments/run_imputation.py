from __future__ import annotations

import argparse
import os
from pathlib import Path
import site
import sys
import time
from typing import Any


def _remove_user_site_from_sys_path() -> None:
    # Keep environment-local packages ahead of user-site installs to avoid ABI mismatches.
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


_remove_user_site_from_sys_path()

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    DEFAULT_MISSINGNESS_MECHANISMS,
    DEFAULT_MISSING_RATES,
    add_common_arguments,
    aggregate_success_rows,
    apply_missing_mask,
    available_model_names,
    build_proximity_model,
    compiled_output_dir,
    downstream_model_names,
    evaluate_downstream_model,
    imputation_metrics,
    induced_missing_masks,
    iterative_imputation,
    iter_openml_suite,
    json_dumps,
    parse_missing_rates,
    parse_missingness_mechanisms,
    parse_random_states,
    prepare_dataset,
    raw_output_path,
    record_exception,
    requested_model_names,
    requested_tasks,
    rough_fill_train_test,
    save_dataframe,
    save_lineplot_dataframe,
    unsupported_model_row,
)


EXPERIMENT_NAME = "imputation"
PROXIMITY_CONFIGURATIONS = (
    ("original", "single_pass"),
    ("oob", "single_pass"),
    ("rfgap", "single_pass"),
    ("rfgap", "iterative"),
)


def _mask_random_state(random_state: int, mechanism: str, missing_rate: float) -> int:
    encoded_rate = int(round(missing_rate * 1000))
    encoded_mechanism = sum(ord(char) for char in mechanism.upper())
    return int((random_state * 1_000_003 + encoded_rate * 97 + encoded_mechanism * 53) % (2**31 - 1))


def _result_context(
    *,
    prepared: Any,
    model_name: str,
    prox_method: str,
    variant: str,
    missingness_mechanism: str,
    missing_rate: float,
    downstream_model: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "suite_name": prepared.suite_name,
        "model_name": model_name,
        "task_id": prepared.task_id,
        "dataset_id": prepared.dataset_id,
        "dataset_name": prepared.dataset_name,
        "prox_method": prox_method,
        "variant": variant,
        "missingness_mechanism": missingness_mechanism,
        "missing_rate": float(missing_rate),
        "downstream_model": downstream_model,
        "missingness_fallback_count": int(metadata.get("fallback_count", 0)),
        "missingness_fallback_summary_json": json_dumps(metadata.get("fallback_counts", {})),
        "missingness_metadata_json": json_dumps(metadata),
    }


def _oracle_context(
    *,
    task_kind: str,
    prepared: Any,
    downstream_model: str,
    random_state: int,
) -> dict[str, Any]:
    oracle_start = time.perf_counter()
    metrics = evaluate_downstream_model(
        task_kind,
        downstream_model=downstream_model,
        X_train=prepared.X_train,
        X_test=prepared.X_test,
        y_train=prepared.y_train,
        y_test=prepared.y_test,
        categorical_indices=prepared.categorical_indices,
        random_state=random_state,
    )
    runtime_seconds = time.perf_counter() - oracle_start
    return {
        "status": "ok",
        "metrics": metrics,
        "runtime_seconds": float(runtime_seconds),
    }


def _empty_downstream_metric_fields() -> dict[str, float]:
    return {
        "test_accuracy": float("nan"),
        "test_roc_auc": float("nan"),
        "test_mse": float("nan"),
        "test_r2": float("nan"),
        "oracle_test_accuracy": float("nan"),
        "oracle_test_roc_auc": float("nan"),
        "oracle_test_mse": float("nan"),
        "oracle_test_r2": float("nan"),
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    random_states = parse_random_states(args.random_states)
    missingness_mechanisms = parse_missingness_mechanisms(args.missingness_mechanisms)
    missing_rates = [float(args.missing_rate)] if args.missing_rate is not None else parse_missing_rates(args.missing_rates)

    for task_kind in requested_tasks(args.task):
        downstream_models = downstream_model_names(task_kind)
        for task_id, task in iter_openml_suite(task_kind, args.dataset_limit):
            dataset_name = str(getattr(task, "dataset_name", f"task_{task_id}"))
            dataset_id = int(getattr(task, "dataset_id", -1))
            for random_state in random_states:
                try:
                    prepared = prepare_dataset(
                        task_kind,
                        task_id=task_id,
                        task=task,
                        random_state=random_state,
                        max_rows=args.max_rows,
                        test_size=args.test_size,
                    )
                except Exception as exc:
                    for missingness_mechanism in missingness_mechanisms:
                        for missing_rate in missing_rates:
                            for model_name in requested_model_names():
                                for prox_method, variant in PROXIMITY_CONFIGURATIONS:
                                    for downstream_model in downstream_models:
                                        rows.append(
                                            record_exception(
                                                experiment_name=EXPERIMENT_NAME,
                                                task_kind=task_kind,
                                                random_state=random_state,
                                                dataset_name=dataset_name,
                                                dataset_id=dataset_id,
                                                task_id=task_id,
                                                error=exc,
                                                extra_fields={
                                                    "model_name": model_name,
                                                    "prox_method": prox_method,
                                                    "variant": variant,
                                                    "missingness_mechanism": missingness_mechanism,
                                                    "missing_rate": float(missing_rate),
                                                    "downstream_model": downstream_model,
                                                },
                                            )
                                        )
                    continue

                oracle_cache: dict[str, dict[str, Any]] = {}
                for downstream_model in downstream_models:
                    try:
                        oracle_cache[downstream_model] = _oracle_context(
                            task_kind=task_kind,
                            prepared=prepared,
                            downstream_model=downstream_model,
                            random_state=random_state,
                        )
                    except Exception as exc:
                        oracle_cache[downstream_model] = {
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }

                for missingness_mechanism in missingness_mechanisms:
                    for missing_rate in missing_rates:
                        mask_result = induced_missing_masks(
                            prepared.X_train,
                            prepared.X_test,
                            categorical_indices=prepared.categorical_indices,
                            missingness_mechanism=missingness_mechanism,
                            missing_rate=missing_rate,
                            random_state=_mask_random_state(
                                random_state,
                                missingness_mechanism,
                                missing_rate,
                            ),
                        )
                        X_train_masked = apply_missing_mask(prepared.X_train, mask_result.train_mask)
                        X_test_masked = apply_missing_mask(prepared.X_test, mask_result.test_mask)

                        for model_name in requested_model_names():
                            if model_name not in available_model_names(task_kind):
                                for prox_method, variant in PROXIMITY_CONFIGURATIONS:
                                    for downstream_model in downstream_models:
                                        rows.append(
                                            unsupported_model_row(
                                                experiment_name=EXPERIMENT_NAME,
                                                task_kind=task_kind,
                                                model_name=model_name,
                                                dataset_name=prepared.dataset_name,
                                                dataset_id=prepared.dataset_id,
                                                task_id=prepared.task_id,
                                                random_state=random_state,
                                                reason=f"{model_name} is not available for {task_kind}.",
                                                extra_fields=_result_context(
                                                    prepared=prepared,
                                                    model_name=model_name,
                                                    prox_method=prox_method,
                                                    variant=variant,
                                                    missingness_mechanism=missingness_mechanism,
                                                    missing_rate=missing_rate,
                                                    downstream_model=downstream_model,
                                                    metadata=mask_result.metadata,
                                                ),
                                            )
                                        )
                                continue

                            for prox_method, variant in PROXIMITY_CONFIGURATIONS:
                                base_context = {
                                    "experiment": EXPERIMENT_NAME,
                                    "status": "ok",
                                    "task_kind": task_kind,
                                    "random_state": random_state,
                                }
                                try:
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
                                        X_imp_train,
                                        mask_result.train_mask,
                                        prepared.categorical_indices,
                                    )
                                    test_metrics = imputation_metrics(
                                        prepared.X_test,
                                        X_imp_test,
                                        mask_result.test_mask,
                                        prepared.categorical_indices,
                                    )
                                except Exception as exc:
                                    for downstream_model in downstream_models:
                                        rows.append(
                                            record_exception(
                                                experiment_name=EXPERIMENT_NAME,
                                                task_kind=task_kind,
                                                random_state=random_state,
                                                dataset_name=prepared.dataset_name,
                                                dataset_id=prepared.dataset_id,
                                                task_id=prepared.task_id,
                                                error=exc,
                                                extra_fields={
                                                    "model_name": model_name,
                                                    **_result_context(
                                                        prepared=prepared,
                                                        model_name=model_name,
                                                        prox_method=prox_method,
                                                        variant=variant,
                                                        missingness_mechanism=missingness_mechanism,
                                                        missing_rate=missing_rate,
                                                        downstream_model=downstream_model,
                                                        metadata=mask_result.metadata,
                                                    ),
                                                },
                                            )
                                        )
                                    continue

                                for downstream_model in downstream_models:
                                    context = _result_context(
                                        prepared=prepared,
                                        model_name=model_name,
                                        prox_method=prox_method,
                                        variant=variant,
                                        missingness_mechanism=missingness_mechanism,
                                        missing_rate=missing_rate,
                                        downstream_model=downstream_model,
                                        metadata=mask_result.metadata,
                                    )
                                    oracle_result = oracle_cache[downstream_model]
                                    if oracle_result["status"] != "ok":
                                        rows.append(
                                            {
                                                **base_context,
                                                **context,
                                                "status": "error",
                                                "error_type": oracle_result["error_type"],
                                                "error_message": oracle_result["error_message"],
                                                **_empty_downstream_metric_fields(),
                                            }
                                        )
                                        continue
                                    try:
                                        downstream_start = time.perf_counter()
                                        downstream_metrics = evaluate_downstream_model(
                                            task_kind,
                                            downstream_model=downstream_model,
                                            X_train=X_imp_train,
                                            X_test=X_imp_test,
                                            y_train=prepared.y_train,
                                            y_test=prepared.y_test,
                                            categorical_indices=prepared.categorical_indices,
                                            random_state=random_state,
                                        )
                                        downstream_runtime_seconds = time.perf_counter() - downstream_start
                                        rows.append(
                                            {
                                                **base_context,
                                                **context,
                                                "train_numeric_mse": train_metrics["numeric_mse"],
                                                "test_numeric_mse": test_metrics["numeric_mse"],
                                                "train_numeric_rmse": train_metrics["numeric_rmse"],
                                                "test_numeric_rmse": test_metrics["numeric_rmse"],
                                                "train_categorical_accuracy": train_metrics["categorical_accuracy"],
                                                "test_categorical_accuracy": test_metrics["categorical_accuracy"],
                                                "n_train_numeric_masked": train_metrics["n_numeric_masked"],
                                                "n_test_numeric_masked": test_metrics["n_numeric_masked"],
                                                "n_train_categorical_masked": train_metrics["n_categorical_masked"],
                                                "n_test_categorical_masked": test_metrics["n_categorical_masked"],
                                                "test_accuracy": downstream_metrics["test_accuracy"],
                                                "test_roc_auc": downstream_metrics["test_roc_auc"],
                                                "test_mse": downstream_metrics["test_mse"],
                                                "test_r2": downstream_metrics["test_r2"],
                                                "oracle_test_accuracy": oracle_result["metrics"]["test_accuracy"],
                                                "oracle_test_roc_auc": oracle_result["metrics"]["test_roc_auc"],
                                                "oracle_test_mse": oracle_result["metrics"]["test_mse"],
                                                "oracle_test_r2": oracle_result["metrics"]["test_r2"],
                                                "fit_runtime_seconds": float(fit_runtime_seconds),
                                                "impute_runtime_seconds": float(impute_runtime_seconds),
                                                "downstream_runtime_seconds": float(downstream_runtime_seconds),
                                                "total_runtime_seconds": float(
                                                    fit_runtime_seconds
                                                    + impute_runtime_seconds
                                                    + downstream_runtime_seconds
                                                ),
                                            }
                                        )
                                    except Exception as exc:
                                        rows.append(
                                            record_exception(
                                                experiment_name=EXPERIMENT_NAME,
                                                task_kind=task_kind,
                                                random_state=random_state,
                                                dataset_name=prepared.dataset_name,
                                                dataset_id=prepared.dataset_id,
                                                task_id=prepared.task_id,
                                                error=exc,
                                                extra_fields={
                                                    "model_name": model_name,
                                                    **context,
                                                },
                                            )
                                        )

    return pd.DataFrame(rows)


def _summary_metric_columns(task_kind: str) -> tuple[str, ...]:
    common_metrics = (
        "train_numeric_mse",
        "test_numeric_mse",
        "train_numeric_rmse",
        "test_numeric_rmse",
        "train_categorical_accuracy",
        "test_categorical_accuracy",
        "n_train_numeric_masked",
        "n_test_numeric_masked",
        "n_train_categorical_masked",
        "n_test_categorical_masked",
        "fit_runtime_seconds",
        "impute_runtime_seconds",
        "downstream_runtime_seconds",
        "total_runtime_seconds",
        "missingness_fallback_count",
    )
    if task_kind == "classification":
        return common_metrics + (
            "test_accuracy",
            "test_roc_auc",
            "oracle_test_accuracy",
            "oracle_test_roc_auc",
        )
    return common_metrics + (
        "test_mse",
        "test_r2",
        "oracle_test_mse",
        "oracle_test_r2",
    )


def _comparison_label(df: pd.DataFrame, include_downstream: bool) -> pd.Series:
    label = (
        df["model_name"].astype(str)
        + " | "
        + df["prox_method"].astype(str)
        + " | "
        + df["variant"].astype(str)
    )
    if include_downstream:
        label = label + " | " + df["downstream_model"].astype(str)
    return label


def compile_outputs(df: pd.DataFrame) -> None:
    for task_kind in sorted(df["task_kind"].dropna().unique()):
        task_df = df[df["task_kind"] == task_kind].copy()
        save_dataframe(task_df, raw_output_path(EXPERIMENT_NAME, task_kind))

        metric_cols = _summary_metric_columns(task_kind)
        out_dir = compiled_output_dir(EXPERIMENT_NAME, task_kind)
        summary = aggregate_success_rows(
            task_df,
            group_cols=(
                "model_name",
                "prox_method",
                "variant",
                "missingness_mechanism",
                "missing_rate",
                "downstream_model",
            ),
            metric_cols=metric_cols,
        )
        save_dataframe(summary, out_dir / "summary.csv")

        imputation_comparison = aggregate_success_rows(
            task_df,
            group_cols=(
                "model_name",
                "prox_method",
                "variant",
                "missingness_mechanism",
                "missing_rate",
            ),
            metric_cols=(
                "test_numeric_mse",
                "test_numeric_rmse",
                "test_categorical_accuracy",
                "train_numeric_mse",
                "train_numeric_rmse",
                "train_categorical_accuracy",
            ),
        )
        save_dataframe(imputation_comparison, out_dir / "imputation_rate_comparison.csv")

        downstream_comparison = aggregate_success_rows(
            task_df,
            group_cols=(
                "model_name",
                "prox_method",
                "variant",
                "missingness_mechanism",
                "missing_rate",
                "downstream_model",
            ),
            metric_cols=(
                "test_accuracy",
                "test_roc_auc",
                "test_mse",
                "test_r2",
                "oracle_test_accuracy",
                "oracle_test_roc_auc",
                "oracle_test_mse",
                "oracle_test_r2",
            ),
        )
        save_dataframe(downstream_comparison, out_dir / "downstream_rate_comparison.csv")

        if not imputation_comparison.empty:
            imputation_plot_df = imputation_comparison.copy()
            imputation_plot_df["comparison_key"] = _comparison_label(imputation_plot_df, include_downstream=False)
            primary_imputation_metric = (
                "test_categorical_accuracy" if task_kind == "classification" else "test_numeric_mse"
            )
            primary_imputation_ylabel = (
                "Test categorical accuracy" if task_kind == "classification" else "Test numeric MSE"
            )
            save_lineplot_dataframe(
                imputation_plot_df.sort_values(["missingness_mechanism", "missing_rate", "comparison_key"]),
                x="missing_rate",
                y=primary_imputation_metric,
                group_col="comparison_key",
                facet_col="missingness_mechanism",
                title=f"{task_kind.title()} Imputation vs Missingness Rate",
                ylabel=primary_imputation_ylabel,
                path=out_dir / f"{primary_imputation_metric}_vs_missing_rate.png",
            )

        if not downstream_comparison.empty:
            primary_downstream_metric = "test_accuracy" if task_kind == "classification" else "test_mse"
            primary_downstream_ylabel = "Test accuracy" if task_kind == "classification" else "Test MSE"
            for downstream_model, downstream_model_df in downstream_comparison.groupby("downstream_model", dropna=False):
                plot_df = downstream_model_df.copy()
                plot_df["comparison_key"] = _comparison_label(plot_df, include_downstream=False)
                save_lineplot_dataframe(
                    plot_df.sort_values(["missingness_mechanism", "missing_rate", "comparison_key"]),
                    x="missing_rate",
                    y=primary_downstream_metric,
                    group_col="comparison_key",
                    facet_col="missingness_mechanism",
                    title=f"{task_kind.title()} {downstream_model} vs Missingness Rate",
                    ylabel=primary_downstream_ylabel,
                    path=out_dir / f"{primary_downstream_metric}__{downstream_model}.png",
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--n-estimators", type=int, default=128)
    parser.add_argument("--missing-rate", type=float, default=None, help="Deprecated single-rate override.")
    parser.add_argument(
        "--missing-rates",
        default=",".join(str(rate) for rate in DEFAULT_MISSING_RATES),
        help="Comma-separated missingness fractions.",
    )
    parser.add_argument(
        "--missingness-mechanisms",
        default=",".join(DEFAULT_MISSINGNESS_MECHANISMS),
        help="Comma-separated missingness mechanisms.",
    )
    parser.add_argument("--n-iterations", type=int, default=3)
    return parser


if __name__ == "__main__":
    frame = run(build_parser().parse_args())
    compile_outputs(frame)
