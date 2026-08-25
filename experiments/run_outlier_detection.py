from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from forestgap import outlier_scores
from experiments.common import (
    OUTLIER_SCORE_METHODS,
    add_common_arguments,
    aggregate_success_rows,
    build_checkpoint_manifest,
    build_outlier_model,
    checkpoint_store,
    compiled_output_dir,
    corrupt_dataset,
    fit_outlier_model,
    iter_openml_suite,
    native_outlier_scores,
    native_proximity_agreement_metrics,
    outlier_label_modes,
    outlier_metrics,
    outlier_model_spec,
    parse_csv_floats,
    parse_random_states,
    prepare_dataset,
    prepared_dataset_hash,
    proximity_methods_for_model,
    log_progress,
    raw_output_path,
    record_exception,
    requested_outlier_model_names,
    requested_tasks,
    run_module_parallel,
    selected_model_names,
    selected_task_ids,
    setup_selected,
    rough_fill_train_test,
    save_dataframe,
    save_plot_dataframe,
    seed_checkpoint_unit,
    unavailable_data_hash,
    unsupported_model_row,
)


EXPERIMENT_NAME = "outlier_detection"
NATIVE_PROX_METHOD = "not_applicable"
NATIVE_LABEL_MODE = "not_applicable"
NATIVE_SCORE_METHOD = "score_samples"
UNSUPERVISED_OUTLIER_MODELS = {
    "IsolationForest",
    "ExtendedIsolationForest",
    "UnsupervisedRandomForest",
    "UnsupervisedObliqueRandomForest",
    "SyntheticNullForest",
}


def requested_outlier_fractions(args: argparse.Namespace) -> list[float]:
    fractions = parse_csv_floats(args.outlier_fractions) if args.outlier_fractions else [args.outlier_fraction]
    invalid = [fraction for fraction in fractions if fraction <= 0.0 or fraction >= 1.0]
    if invalid:
        raise ValueError(
            "Outlier fractions must lie strictly inside (0, 1), received "
            + ", ".join(f"{value:.4f}" for value in invalid)
            + "."
        )
    return list(dict.fromkeys(float(value) for value in fractions))


def _checkpoint_manifest(
    *,
    args: argparse.Namespace,
    task_kind: str,
    task_id: int,
    dataset_id: int,
    random_state: int,
    model_name: str,
    prox_method: str,
    outlier_fraction: float,
    data_hash: str,
):
    return build_checkpoint_manifest(
        experiment=EXPERIMENT_NAME,
        runner_file=__file__,
        unit={
            "task_kind": task_kind,
            "task_id": int(task_id),
            "dataset_id": int(dataset_id),
            "random_state": int(random_state),
            "model_name": model_name,
            "prox_method": prox_method,
            "outlier_fraction": float(outlier_fraction),
        },
        scientific_config={
            "max_rows": args.max_rows,
            "test_size": float(args.test_size),
            "n_estimators": int(args.n_estimators),
            "isolation_bootstrap": bool(args.isolation_bootstrap),
            "score_methods": list(OUTLIER_SCORE_METHODS),
        },
        data_hash=data_hash,
    )


def config_label(row: pd.Series) -> str:
    parts = [
        str(row["model_name"]),
        str(row["score_source"]),
        str(row["prox_method"]),
        str(row["score_method"]),
        str(row["label_mode"]),
        f"frac={float(row['outlier_fraction']):.3f}",
    ]
    return " | ".join(parts)


def append_native_row(
    *,
    rows: list[dict[str, Any]],
    prepared: Any,
    spec: Any,
    model_name: str,
    random_state: int,
    outlier_fraction: float,
    outlier_count: int,
    fit_prox_method: str,
    fit_runtime_seconds: float,
    score_runtime_seconds: float,
    scores: Any,
    labels: Any,
    fit_metadata: dict[str, Any],
) -> None:
    metrics = outlier_metrics(scores, labels, outlier_count)
    rows.append(
        {
            "experiment": EXPERIMENT_NAME,
            "status": "ok",
            "task_kind": prepared.task_kind,
            "suite_name": prepared.suite_name,
            "dataset_id": prepared.dataset_id,
            "dataset_name": prepared.dataset_name,
            "task_id": prepared.task_id,
            "random_state": random_state,
            "model_name": model_name,
            "model_family": spec.model_family,
            "score_source": "native",
            "prox_method": NATIVE_PROX_METHOD,
            "fit_prox_method": fit_prox_method,
            "score_method": NATIVE_SCORE_METHOD,
            "label_mode": NATIVE_LABEL_MODE,
            "outlier_fraction": outlier_fraction,
            "outlier_count": outlier_count,
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "precision_at_k": metrics["precision_at_k"],
            "recall_at_k": metrics["recall_at_k"],
            "f1_at_k": metrics["f1_at_k"],
            "spearman_rank_corr": float("nan"),
            "topk_overlap_at_k": float("nan"),
            "fit_runtime_seconds": float(fit_runtime_seconds),
            "score_runtime_seconds": float(score_runtime_seconds),
            "total_runtime_seconds": float(fit_runtime_seconds + score_runtime_seconds),
            **fit_metadata,
        }
    )


def append_proximity_row(
    *,
    rows: list[dict[str, Any]],
    prepared: Any,
    spec: Any,
    model_name: str,
    random_state: int,
    outlier_fraction: float,
    outlier_count: int,
    prox_method: str,
    score_method: str,
    label_mode: str,
    fit_runtime_seconds: float,
    score_runtime_seconds: float,
    scores: Any,
    labels: Any,
    fit_metadata: dict[str, Any],
    native_scores: Any,
) -> None:
    metrics = outlier_metrics(scores, labels, outlier_count)
    comparison_metrics = (
        native_proximity_agreement_metrics(native_scores, scores, outlier_count)
        if native_scores is not None and label_mode == "unsupervised"
        else {
            "spearman_rank_corr": float("nan"),
            "topk_overlap_at_k": float("nan"),
        }
    )
    rows.append(
        {
            "experiment": EXPERIMENT_NAME,
            "status": "ok",
            "task_kind": prepared.task_kind,
            "suite_name": prepared.suite_name,
            "dataset_id": prepared.dataset_id,
            "dataset_name": prepared.dataset_name,
            "task_id": prepared.task_id,
            "random_state": random_state,
            "model_name": model_name,
            "model_family": spec.model_family,
            "score_source": "proximity",
            "prox_method": prox_method,
            "fit_prox_method": prox_method,
            "score_method": score_method,
            "label_mode": label_mode,
            "outlier_fraction": outlier_fraction,
            "outlier_count": outlier_count,
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "precision_at_k": metrics["precision_at_k"],
            "recall_at_k": metrics["recall_at_k"],
            "f1_at_k": metrics["f1_at_k"],
            "spearman_rank_corr": comparison_metrics["spearman_rank_corr"],
            "topk_overlap_at_k": comparison_metrics["topk_overlap_at_k"],
            "fit_runtime_seconds": float(fit_runtime_seconds),
            "score_runtime_seconds": float(score_runtime_seconds),
            "total_runtime_seconds": float(fit_runtime_seconds + score_runtime_seconds),
            **fit_metadata,
        }
    )


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    outlier_fractions = requested_outlier_fractions(args)
    random_states = list(dict.fromkeys(parse_random_states(args.random_states)))
    store = checkpoint_store(
        EXPERIMENT_NAME,
        root=getattr(args, "_checkpoint_root", None),
    )
    resume = bool(getattr(args, "resume", True))

    for task_kind in requested_tasks(args.task):
        log_progress(f"[{EXPERIMENT_NAME}] task_kind={task_kind}")
        for task_id, task in iter_openml_suite(
            task_kind,
            args.dataset_limit,
            task_ids=selected_task_ids(args, task_kind),
        ):
            dataset_name = str(getattr(task, "dataset_name", f"task_{task_id}"))
            dataset_id = int(getattr(task, "dataset_id", -1))
            for random_state in random_states:
                if not setup_selected(args, task_kind, task_id, random_state):
                    continue
                log_progress(
                    f"[{EXPERIMENT_NAME}] task_kind={task_kind} dataset={dataset_name} "
                    f"task_id={task_id} random_state={random_state}"
                )
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
                except Exception as exc:
                    data_hash = unavailable_data_hash(
                        task_kind=task_kind,
                        task_id=task_id,
                        dataset_id=dataset_id,
                    )
                    for outlier_fraction in outlier_fractions:
                        for model_name in selected_model_names(args, requested_outlier_model_names()):
                            prox_methods = (
                                ("collision",)
                                if model_name in {"IsolationForest", "ExtendedIsolationForest"}
                                else proximity_methods_for_model(model_name)
                            )
                            for prox_method in prox_methods:
                                manifest = _checkpoint_manifest(
                                    args=args,
                                    task_kind=task_kind,
                                    task_id=task_id,
                                    dataset_id=dataset_id,
                                    random_state=random_state,
                                    model_name=model_name,
                                    prox_method=prox_method,
                                    outlier_fraction=outlier_fraction,
                                    data_hash=data_hash,
                                )
                                cached = store.load(manifest) if resume else None
                                if cached is not None:
                                    rows.extend(cached.summary_rows)
                                    continue
                                error_row = record_exception(
                                    experiment_name=EXPERIMENT_NAME,
                                    task_kind=task_kind,
                                    random_state=random_state,
                                    dataset_name=dataset_name,
                                    dataset_id=dataset_id,
                                    task_id=task_id,
                                    error=exc,
                                    extra_fields={
                                        "suite_name": None,
                                        "model_name": model_name,
                                        "prox_method": prox_method,
                                        "fit_prox_method": prox_method,
                                        "outlier_fraction": float(outlier_fraction),
                                    },
                                )
                                rows.extend(store.save(manifest, [error_row]).summary_rows)
                    continue

                for outlier_fraction in outlier_fractions:
                    log_progress(
                        f"[{EXPERIMENT_NAME}] dataset={prepared.dataset_name} "
                        f"outlier_fraction={float(outlier_fraction):.3f}"
                    )
                    try:
                        X_corrupt, corruption_labels = corrupt_dataset(
                            X_train_filled,
                            prepared.categorical_indices,
                            random_state=random_state,
                            fraction=outlier_fraction,
                        )
                    except Exception as exc:
                        data_hash = prepared_dataset_hash(prepared)
                        for model_name in selected_model_names(args, requested_outlier_model_names()):
                            prox_methods = (
                                ("collision",)
                                if model_name in {"IsolationForest", "ExtendedIsolationForest"}
                                else proximity_methods_for_model(model_name)
                            )
                            for prox_method in prox_methods:
                                manifest = _checkpoint_manifest(
                                    args=args,
                                    task_kind=task_kind,
                                    task_id=prepared.task_id,
                                    dataset_id=prepared.dataset_id,
                                    random_state=random_state,
                                    model_name=model_name,
                                    prox_method=prox_method,
                                    outlier_fraction=outlier_fraction,
                                    data_hash=data_hash,
                                )
                                error_row = record_exception(
                                    experiment_name=EXPERIMENT_NAME,
                                    task_kind=task_kind,
                                    random_state=random_state,
                                    dataset_name=prepared.dataset_name,
                                    dataset_id=prepared.dataset_id,
                                    task_id=prepared.task_id,
                                    error=exc,
                                    extra_fields={
                                        "suite_name": prepared.suite_name,
                                        "model_name": model_name,
                                        "prox_method": prox_method,
                                        "fit_prox_method": prox_method,
                                        "outlier_fraction": float(outlier_fraction),
                                    },
                                )
                                rows.extend(store.save(manifest, [error_row]).summary_rows)
                        continue
                    outlier_count = int(corruption_labels.sum())
                    data_hash = prepared_dataset_hash(
                        prepared,
                        extra_artifacts={
                            "outlier_fraction": float(outlier_fraction),
                            "corruption_labels": corruption_labels,
                        },
                    )

                    for model_name in selected_model_names(args, requested_outlier_model_names()):
                        log_progress(
                            f"[{EXPERIMENT_NAME}] model={model_name} "
                            f"outlier_fraction={float(outlier_fraction):.3f}"
                        )
                        spec = outlier_model_spec(task_kind, model_name)
                        prox_methods = (
                            ("collision",)
                            if model_name in {"IsolationForest", "ExtendedIsolationForest"}
                            else proximity_methods_for_model(model_name)
                        )
                        if spec is None:
                            for prox_method in prox_methods:
                                manifest = _checkpoint_manifest(
                                    args=args,
                                    task_kind=task_kind,
                                    task_id=prepared.task_id,
                                    dataset_id=prepared.dataset_id,
                                    random_state=random_state,
                                    model_name=model_name,
                                    prox_method=prox_method,
                                    outlier_fraction=outlier_fraction,
                                    data_hash=data_hash,
                                )
                                cached = store.load(manifest) if resume else None
                                if cached is not None:
                                    rows.extend(cached.summary_rows)
                                    continue
                                unsupported_row = unsupported_model_row(
                                    experiment_name=EXPERIMENT_NAME,
                                    task_kind=task_kind,
                                    model_name=model_name,
                                    dataset_name=prepared.dataset_name,
                                    dataset_id=prepared.dataset_id,
                                    task_id=prepared.task_id,
                                    random_state=random_state,
                                    reason=f"{model_name} is not available for {task_kind}.",
                                    extra_fields={
                                        "suite_name": prepared.suite_name,
                                        "model_family": (
                                            "unsupervised_model"
                                            if model_name in UNSUPERVISED_OUTLIER_MODELS
                                            else "supervised_forest"
                                        ),
                                        "score_source": "not_applicable",
                                        "prox_method": prox_method,
                                        "fit_prox_method": prox_method,
                                        "score_method": "not_applicable",
                                        "label_mode": "not_applicable",
                                        "outlier_fraction": outlier_fraction,
                                        "outlier_count": outlier_count,
                                    },
                                )
                                rows.extend(
                                    store.save(manifest, [unsupported_row]).summary_rows
                                )
                            continue

                        native_scores = None
                        for prox_index, prox_method in enumerate(prox_methods):
                            manifest = _checkpoint_manifest(
                                args=args,
                                task_kind=task_kind,
                                task_id=prepared.task_id,
                                dataset_id=prepared.dataset_id,
                                random_state=random_state,
                                model_name=model_name,
                                prox_method=prox_method,
                                outlier_fraction=outlier_fraction,
                                data_hash=data_hash,
                            )
                            cached = store.load(manifest) if resume else None
                            canonical_native_unit = prox_index == 0 and spec.supports_native_scores
                            if (
                                cached is not None
                                and (not canonical_native_unit or "native_scores" in cached.artifacts)
                            ):
                                rows.extend(cached.summary_rows)
                                if canonical_native_unit:
                                    native_scores = cached.artifacts["native_scores"]
                                log_progress(
                                    f"[{EXPERIMENT_NAME}] resumed model={model_name} "
                                    f"prox_method={prox_method} fraction={float(outlier_fraction):.3f}"
                                )
                                continue

                            log_progress(
                                f"[{EXPERIMENT_NAME}] model={model_name} prox_method={prox_method} "
                                f"outlier_fraction={float(outlier_fraction):.3f}"
                            )
                            seed_checkpoint_unit(manifest)
                            unit_rows: list[dict[str, Any]] = []
                            unit_artifacts: dict[str, Any] = {}
                            try:
                                model, spec, fit_metadata = build_outlier_model(
                                    task_kind,
                                    model_name=model_name,
                                    prox_method=prox_method,
                                    random_state=random_state,
                                    n_estimators=args.n_estimators,
                                    isolation_bootstrap=args.isolation_bootstrap,
                                )

                                fit_start = time.perf_counter()
                                fit_outlier_model(model, spec, X_corrupt, prepared.y_train)
                                fit_runtime_seconds = time.perf_counter() - fit_start

                                if spec.supports_native_scores and native_scores is None:
                                    native_score_start = time.perf_counter()
                                    native_scores = native_outlier_scores(model, X_corrupt)
                                    native_score_runtime_seconds = time.perf_counter() - native_score_start
                                    if canonical_native_unit:
                                        append_native_row(
                                            rows=unit_rows,
                                            prepared=prepared,
                                            spec=spec,
                                            model_name=model_name,
                                            random_state=random_state,
                                            outlier_fraction=outlier_fraction,
                                            outlier_count=outlier_count,
                                            fit_prox_method=prox_method,
                                            fit_runtime_seconds=fit_runtime_seconds,
                                            score_runtime_seconds=native_score_runtime_seconds,
                                            scores=native_scores,
                                            labels=corruption_labels,
                                            fit_metadata=fit_metadata,
                                        )
                                        unit_artifacts["native_scores"] = native_scores

                                for score_method in OUTLIER_SCORE_METHODS:
                                    for label_mode in outlier_label_modes(task_kind, model_name):
                                        score_labels = prepared.y_train if label_mode == "supervised" else None
                                        score_start = time.perf_counter()
                                        scores = outlier_scores(model, score_labels, method=score_method)
                                        score_runtime_seconds = time.perf_counter() - score_start
                                        append_proximity_row(
                                            rows=unit_rows,
                                            prepared=prepared,
                                            spec=spec,
                                            model_name=model_name,
                                            random_state=random_state,
                                            outlier_fraction=outlier_fraction,
                                            outlier_count=outlier_count,
                                            prox_method=prox_method,
                                            score_method=score_method,
                                            label_mode=label_mode,
                                            fit_runtime_seconds=fit_runtime_seconds,
                                            score_runtime_seconds=score_runtime_seconds,
                                            scores=scores,
                                            labels=corruption_labels,
                                            fit_metadata=fit_metadata,
                                            native_scores=native_scores,
                                        )
                            except Exception as exc:
                                unit_rows = [record_exception(
                                    experiment_name=EXPERIMENT_NAME,
                                    task_kind=task_kind,
                                    random_state=random_state,
                                    dataset_name=prepared.dataset_name,
                                    dataset_id=prepared.dataset_id,
                                    task_id=prepared.task_id,
                                    error=exc,
                                    extra_fields={
                                        "suite_name": prepared.suite_name,
                                        "model_name": model_name,
                                        "model_family": spec.model_family,
                                        "score_source": "not_applicable",
                                        "prox_method": prox_method,
                                        "fit_prox_method": prox_method,
                                        "score_method": "not_applicable",
                                        "label_mode": "not_applicable",
                                        "outlier_fraction": outlier_fraction,
                                        "outlier_count": outlier_count,
                                    },
                                )]
                                unit_artifacts = {}
                            bundle = store.save(
                                manifest,
                                unit_rows,
                                artifacts=unit_artifacts,
                            )
                            rows.extend(bundle.summary_rows)

    return pd.DataFrame(rows)


def save_metric_plot(df: pd.DataFrame, *, metric: str, title: str, path: Path) -> None:
    if df.empty:
        return
    plot_df = df.copy()
    plot_df["config"] = plot_df.apply(config_label, axis=1)
    plot_df = plot_df.sort_values(metric, ascending=False)
    save_plot_dataframe(
        plot_df,
        x="config",
        y=metric,
        title=title,
        ylabel=metric.replace("_", " ").title(),
        path=path,
        rotation=80,
    )


def compile_outputs(df: pd.DataFrame) -> None:
    if df.empty or "task_kind" not in df.columns:
        return

    metric_cols = (
        "roc_auc",
        "average_precision",
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
        "spearman_rank_corr",
        "topk_overlap_at_k",
        "fit_runtime_seconds",
        "score_runtime_seconds",
        "total_runtime_seconds",
    )
    summary_group_cols = (
        "model_family",
        "model_name",
        "score_source",
        "prox_method",
        "score_method",
        "label_mode",
        "outlier_fraction",
    )

    for task_kind in sorted(df["task_kind"].dropna().unique()):
        task_df = df[df["task_kind"] == task_kind].copy()
        save_dataframe(task_df, raw_output_path(EXPERIMENT_NAME, task_kind))

        out_dir = compiled_output_dir(EXPERIMENT_NAME, task_kind)
        for legacy_name in ("summary.csv", "model_comparison.csv", "average_precision.png"):
            legacy_path = out_dir / legacy_name
            if legacy_path.exists():
                legacy_path.unlink()
        summary = aggregate_success_rows(
            task_df,
            group_cols=summary_group_cols,
            metric_cols=metric_cols,
        )
        save_dataframe(summary, out_dir / "summary_by_configuration.csv")

        supervised_summary = summary[summary["model_family"] == "supervised_forest"].copy()
        unsupervised_summary = summary[summary["model_family"] == "unsupervised_model"].copy()
        save_dataframe(supervised_summary, out_dir / "supervised_forest_summary.csv")
        save_dataframe(unsupervised_summary, out_dir / "unsupervised_model_summary.csv")

        save_metric_plot(
            summary,
            metric="average_precision",
            title=f"{task_kind.title()} Outlier Detection Average Precision",
            path=out_dir / "average_precision_by_configuration.png",
        )
        save_metric_plot(
            summary,
            metric="roc_auc",
            title=f"{task_kind.title()} Outlier Detection ROC AUC",
            path=out_dir / "roc_auc_by_configuration.png",
        )
        save_metric_plot(
            supervised_summary,
            metric="average_precision",
            title=f"{task_kind.title()} Supervised Forest Outlier Detection Average Precision",
            path=out_dir / "supervised_average_precision.png",
        )
        save_metric_plot(
            unsupervised_summary,
            metric="average_precision",
            title=f"{task_kind.title()} Unsupervised Outlier Detection Average Precision",
            path=out_dir / "unsupervised_average_precision.png",
        )

        for model_name in ("IsolationForest", "ExtendedIsolationForest"):
            model_summary = summary[summary["model_name"] == model_name].copy()
            if model_summary.empty:
                continue
            slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in model_name).strip("_")
            save_dataframe(model_summary, out_dir / f"{slug}_native_vs_proximity_summary.csv")
            save_metric_plot(
                model_summary,
                metric="average_precision",
                title=f"{task_kind.title()} {model_name} Native vs Proximity Average Precision",
                path=out_dir / f"{slug}_average_precision.png",
            )
            save_metric_plot(
                model_summary,
                metric="roc_auc",
                title=f"{task_kind.title()} {model_name} Native vs Proximity ROC AUC",
                path=out_dir / f"{slug}_roc_auc.png",
            )

            agreement_summary = model_summary[
                (model_summary["score_source"] == "proximity")
                & (
                    model_summary["spearman_rank_corr"].notna()
                    | model_summary["topk_overlap_at_k"].notna()
                )
            ].copy()
            save_dataframe(agreement_summary, out_dir / f"{slug}_agreement_summary.csv")
            save_metric_plot(
                agreement_summary,
                metric="spearman_rank_corr",
                title=f"{task_kind.title()} {model_name} Native vs Proximity Spearman Agreement",
                path=out_dir / f"{slug}_spearman_rank_corr.png",
            )
            save_metric_plot(
                agreement_summary,
                metric="topk_overlap_at_k",
                title=f"{task_kind.title()} {model_name} Native vs Proximity Top-k Overlap",
                path=out_dir / f"{slug}_topk_overlap_at_k.png",
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--n-estimators", type=int, default=128)
    parser.add_argument("--outlier-fraction", type=float, default=0.05)
    parser.add_argument(
        "--outlier-fractions",
        default=None,
        help="Optional comma-separated sweep of outlier fractions. Overrides --outlier-fraction when set.",
    )
    parser.add_argument(
        "--isolation-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bootstrap toggle for IsolationForest and ExtendedIsolationForest, mirroring the notebooks.",
    )
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    frame = run_module_parallel(
        "experiments.run_outlier_detection",
        parsed_args,
        model_names=requested_outlier_model_names(),
    )
    compile_outputs(frame)
