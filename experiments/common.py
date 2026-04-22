from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
import time
import traceback
from typing import Any, Iterable, Iterator, Literal, Optional, Sequence

for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(env_name, str(os.cpu_count() or 1))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from forestgap import load_openml_dataset, with_sparse_proximity


TaskKind = Literal["classification", "regression"]
ModelKey = Literal[
    "RandomForest",
    "ExtraTrees",
    "RotationForest",
    "HonestForest",
    "ObliqueRandomForest",
    "PatchObliqueRandomForest",
]

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
CLASSIFICATION_SUITE_ID = 99
REGRESSION_SUITE_ID = 353
REQUESTED_MODEL_ORDER: tuple[ModelKey, ...] = (
    "RandomForest",
    "RotationForest",
    "HonestForest",
    "ObliqueRandomForest",
    "PatchObliqueRandomForest",
    "ExtraTrees",
)


@dataclass(frozen=True)
class ModelSpec:
    key: ModelKey
    estimator_cls: type[Any]
    task_kind: TaskKind


def _build_model_registry() -> dict[tuple[TaskKind, ModelKey], ModelSpec]:
    registry: dict[tuple[TaskKind, ModelKey], ModelSpec] = {
        ("classification", "RandomForest"): ModelSpec(
            key="RandomForest",
            estimator_cls=RandomForestClassifier,
            task_kind="classification",
        ),
        ("regression", "RandomForest"): ModelSpec(
            key="RandomForest",
            estimator_cls=RandomForestRegressor,
            task_kind="regression",
        ),
        ("classification", "ExtraTrees"): ModelSpec(
            key="ExtraTrees",
            estimator_cls=ExtraTreesClassifier,
            task_kind="classification",
        ),
        ("regression", "ExtraTrees"): ModelSpec(
            key="ExtraTrees",
            estimator_cls=ExtraTreesRegressor,
            task_kind="regression",
        ),
    }

    try:
        from aeon.classification.sklearn import RotationForestClassifier
        from aeon.regression.sklearn import RotationForestRegressor

        registry[("classification", "RotationForest")] = ModelSpec(
            key="RotationForest",
            estimator_cls=RotationForestClassifier,
            task_kind="classification",
        )
        registry[("regression", "RotationForest")] = ModelSpec(
            key="RotationForest",
            estimator_cls=RotationForestRegressor,
            task_kind="regression",
        )
    except Exception:
        pass

    try:
        from treeple import (
            HonestForestClassifier,
            ObliqueRandomForestClassifier,
            ObliqueRandomForestRegressor,
            PatchObliqueRandomForestClassifier,
            PatchObliqueRandomForestRegressor,
        )

        registry[("classification", "HonestForest")] = ModelSpec(
            key="HonestForest",
            estimator_cls=HonestForestClassifier,
            task_kind="classification",
        )
        registry[("classification", "ObliqueRandomForest")] = ModelSpec(
            key="ObliqueRandomForest",
            estimator_cls=ObliqueRandomForestClassifier,
            task_kind="classification",
        )
        registry[("regression", "ObliqueRandomForest")] = ModelSpec(
            key="ObliqueRandomForest",
            estimator_cls=ObliqueRandomForestRegressor,
            task_kind="regression",
        )
        registry[("classification", "PatchObliqueRandomForest")] = ModelSpec(
            key="PatchObliqueRandomForest",
            estimator_cls=PatchObliqueRandomForestClassifier,
            task_kind="classification",
        )
        registry[("regression", "PatchObliqueRandomForest")] = ModelSpec(
            key="PatchObliqueRandomForest",
            estimator_cls=PatchObliqueRandomForestRegressor,
            task_kind="regression",
        )
    except Exception:
        pass

    return registry


MODEL_REGISTRY = _build_model_registry()


@dataclass
class PreparedDataset:
    task_kind: TaskKind
    suite_name: str
    task_id: int
    dataset_id: int
    dataset_name: str
    feature_names: list[str]
    categorical_indices: list[int]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    original_n_samples: int
    original_n_features: int
    train_size: int
    test_size: int


def parse_random_states(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task",
        choices=("classification", "regression", "both"),
        default="both",
        help="Which benchmark family to run.",
    )
    parser.add_argument(
        "--random-states",
        default="0,1,2",
        help="Comma-separated random states.",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Optional cap on datasets per suite for debugging.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional per-dataset row cap applied before train/test split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Held-out test fraction.",
    )


def requested_tasks(task: str) -> list[TaskKind]:
    if task == "both":
        return ["classification", "regression"]
    return [task]  # type: ignore[list-item]


def suite_metadata(task_kind: TaskKind) -> tuple[int, str]:
    if task_kind == "classification":
        return CLASSIFICATION_SUITE_ID, "OpenML-CC18"
    return REGRESSION_SUITE_ID, "OpenML-CTR23"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_output_path(experiment_name: str, task_kind: TaskKind) -> Path:
    return ensure_dir(RAW_RESULTS_DIR / experiment_name) / f"{task_kind}_raw.csv"


def compiled_output_dir(experiment_name: str, task_kind: TaskKind) -> Path:
    return ensure_dir(RESULTS_DIR / experiment_name / task_kind)


def requested_model_names() -> tuple[ModelKey, ...]:
    return REQUESTED_MODEL_ORDER


def available_model_names(task_kind: TaskKind) -> list[ModelKey]:
    return [
        model_key
        for model_key in REQUESTED_MODEL_ORDER
        if (task_kind, model_key) in MODEL_REGISTRY
    ]


def model_spec(task_kind: TaskKind, model_name: ModelKey) -> Optional[ModelSpec]:
    return MODEL_REGISTRY.get((task_kind, model_name))


def unsupported_model_row(
    *,
    experiment_name: str,
    task_kind: TaskKind,
    model_name: ModelKey,
    dataset_name: str,
    dataset_id: int,
    task_id: int,
    random_state: int,
    reason: str,
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    row = {
        "experiment": experiment_name,
        "status": "unsupported",
        "task_kind": task_kind,
        "model_name": model_name,
        "random_state": random_state,
        "task_id": task_id,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "reason": reason,
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_plot_dataframe(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    path: Path,
    rotation: int = 35,
) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind="bar", x=x, y=y, ax=ax, legend=False, color="#2c7fb8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=rotation)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def iter_openml_suite(task_kind: TaskKind, limit: Optional[int] = None) -> Iterator[tuple[int, Any]]:
    suite_id, suite_name = suite_metadata(task_kind)
    suite = openml.study.get_suite(suite_id)
    task_ids = list(getattr(suite, "tasks", []))
    if limit is not None:
        task_ids = task_ids[:limit]
    for task_id in task_ids:
        yield int(task_id), openml.tasks.get_task(int(task_id))


def _coerce_frame(X: Any, feature_names: Sequence[str]) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    columns = list(feature_names) if feature_names else [f"x{i + 1}" for i in range(np.asarray(X).shape[1])]
    return pd.DataFrame(X, columns=columns)


def _coerce_target(y: Any) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Multi-output targets are not supported in these experiments.")
        return y.iloc[:, 0].copy()
    if isinstance(y, pd.Series):
        return y.copy()
    return pd.Series(y)


def _categorical_columns(X: pd.DataFrame, categorical_indicator: Sequence[bool]) -> list[str]:
    columns: list[str] = []
    if categorical_indicator:
        for column, is_categorical in zip(X.columns, categorical_indicator):
            if bool(is_categorical):
                columns.append(str(column))
    inferred = [
        str(column)
        for column in X.columns
        if str(column) not in columns
        and (
            pd.api.types.is_object_dtype(X[column])
            or pd.api.types.is_categorical_dtype(X[column])
            or pd.api.types.is_bool_dtype(X[column])
        )
    ]
    return columns + [column for column in inferred if column not in columns]


def _encode_split_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    feature_names = [str(column) for column in X_train.columns]
    cat_set = set(str(column) for column in categorical_columns)
    X_train_num = np.empty((len(X_train), len(feature_names)), dtype=np.float64)
    X_test_num = np.empty((len(X_test), len(feature_names)), dtype=np.float64)
    categorical_indices: list[int] = []

    for col_idx, column in enumerate(feature_names):
        train_col = X_train[column]
        test_col = X_test[column]
        if column in cat_set:
            categories = pd.Index(pd.Series(train_col, dtype="object").dropna().unique())
            train_codes = pd.Categorical(train_col, categories=categories).codes.astype(np.float64)
            test_codes = pd.Categorical(test_col, categories=categories).codes.astype(np.float64)
            train_codes[train_codes < 0] = np.nan
            test_codes[test_codes < 0] = np.nan
            X_train_num[:, col_idx] = train_codes
            X_test_num[:, col_idx] = test_codes
            categorical_indices.append(col_idx)
        else:
            X_train_num[:, col_idx] = pd.to_numeric(train_col, errors="coerce").to_numpy(dtype=np.float64)
            X_test_num[:, col_idx] = pd.to_numeric(test_col, errors="coerce").to_numpy(dtype=np.float64)

    return X_train_num, X_test_num, feature_names, categorical_indices


def _apply_feature_cap(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    categorical_indices: list[int],
    feature_cap: Optional[int],
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    if feature_cap is None or X_train.shape[1] <= feature_cap:
        return X_train, X_test, feature_names, categorical_indices

    train_for_var, _ = rough_fill_train_test(X_train, X_test, categorical_indices)
    variances = np.nanvar(train_for_var, axis=0)
    ranking = np.argsort(np.nan_to_num(variances, nan=-1.0))[::-1][:feature_cap]
    ranking = np.sort(ranking)
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(ranking.tolist())}
    new_categorical = [index_map[idx] for idx in categorical_indices if idx in index_map]
    return (
        X_train[:, ranking],
        X_test[:, ranking],
        [feature_names[idx] for idx in ranking],
        new_categorical,
    )


def prepare_dataset(
    task_kind: TaskKind,
    *,
    task_id: int,
    task: Any,
    random_state: int,
    max_rows: Optional[int],
    test_size: float,
    feature_cap: Optional[int] = None,
) -> PreparedDataset:
    suite_id, suite_name = suite_metadata(task_kind)
    dataset = load_openml_dataset(
        dataset_id=int(task.dataset_id),
        target_name=getattr(task, "target_name", None),
        as_frame=True,
    )
    X_df = _coerce_frame(dataset.X, dataset.feature_names)
    y_series = _coerce_target(dataset.y)

    if task_kind == "classification":
        y_series = y_series.astype("category")
        valid_mask = ~y_series.isna()
        X_df = X_df.loc[valid_mask].reset_index(drop=True)
        y_series = y_series.loc[valid_mask].reset_index(drop=True)
        if y_series.nunique() < 2:
            raise ValueError("Need at least two classes after dropping missing targets.")
        y_values = y_series.cat.codes.to_numpy(dtype=np.int64)
    else:
        y_numeric = pd.to_numeric(y_series, errors="coerce")
        valid_mask = ~y_numeric.isna()
        X_df = X_df.loc[valid_mask].reset_index(drop=True)
        y_numeric = y_numeric.loc[valid_mask].reset_index(drop=True)
        y_values = y_numeric.to_numpy(dtype=np.float64)

    original_n_samples = len(X_df)
    original_n_features = X_df.shape[1]
    if max_rows is not None and original_n_samples > max_rows:
        rng = np.random.RandomState(random_state)
        if task_kind == "classification":
            sampled_idx = []
            for cls in np.unique(y_values):
                cls_idx = np.flatnonzero(y_values == cls)
                n_take = max(1, int(round(max_rows * (cls_idx.size / original_n_samples))))
                sampled_idx.extend(rng.choice(cls_idx, size=min(n_take, cls_idx.size), replace=False).tolist())
            sampled_idx = np.array(sorted(set(sampled_idx)), dtype=np.int64)
            if sampled_idx.size > max_rows:
                sampled_idx = np.sort(rng.choice(sampled_idx, size=max_rows, replace=False))
        else:
            sampled_idx = np.sort(rng.choice(np.arange(original_n_samples), size=max_rows, replace=False))
        X_df = X_df.iloc[sampled_idx].reset_index(drop=True)
        y_values = y_values[sampled_idx]

    categorical_columns = _categorical_columns(X_df, dataset.categorical_indicator)
    stratify = y_values if task_kind == "classification" and np.min(np.bincount(y_values)) >= 2 else None
    if stratify is not None:
        n_classes = int(np.unique(y_values).size)
        n_test = int(np.ceil(test_size * len(y_values)))
        if n_test < n_classes:
            stratify = None
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y_values,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    X_train_df = X_train_df.reset_index(drop=True)
    X_test_df = X_test_df.reset_index(drop=True)

    X_train, X_test, feature_names, categorical_indices = _encode_split_features(
        X_train_df,
        X_test_df,
        categorical_columns,
    )
    X_train, X_test, feature_names, categorical_indices = _apply_feature_cap(
        X_train,
        X_test,
        feature_names,
        categorical_indices,
        feature_cap,
    )

    return PreparedDataset(
        task_kind=task_kind,
        suite_name=suite_name,
        task_id=int(task_id),
        dataset_id=int(dataset.dataset_id),
        dataset_name=str(dataset.name),
        feature_names=feature_names,
        categorical_indices=categorical_indices,
        X_train=X_train,
        X_test=X_test,
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
        original_n_samples=int(original_n_samples),
        original_n_features=int(original_n_features),
        train_size=int(X_train.shape[0]),
        test_size=int(X_test.shape[0]),
    )


def rough_fill_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    categorical_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    fill_values = compute_fill_values(X_train, categorical_indices)
    X_train_filled = np.array(X_train, copy=True)
    X_test_filled = np.array(X_test, copy=True)

    for col_idx, fill_value in enumerate(fill_values):
        train_nan = np.isnan(X_train_filled[:, col_idx])
        test_nan = np.isnan(X_test_filled[:, col_idx])
        X_train_filled[train_nan, col_idx] = fill_value
        X_test_filled[test_nan, col_idx] = fill_value

    return X_train_filled, X_test_filled


def compute_fill_values(
    X_train: np.ndarray,
    categorical_indices: Sequence[int],
) -> np.ndarray:
    cat_set = set(int(idx) for idx in categorical_indices)
    fill_values = np.zeros(X_train.shape[1], dtype=np.float64)

    for col_idx in range(X_train.shape[1]):
        train_col = X_train[:, col_idx]
        observed = train_col[~np.isnan(train_col)]
        if observed.size == 0:
            fill_value = 0.0
        elif col_idx in cat_set:
            values, counts = np.unique(observed, return_counts=True)
            fill_value = float(values[np.argmax(counts)])
        else:
            fill_value = float(np.nanmedian(observed))
        fill_values[col_idx] = fill_value

    return fill_values


def build_proximity_model(
    task_kind: TaskKind,
    *,
    model_name: ModelKey,
    prox_method: str,
    random_state: int,
    n_estimators: int,
    bootstrap: bool = True,
    oob_score: bool = False,
    extra_kwargs: Optional[dict[str, Any]] = None,
) -> Any:
    spec = model_spec(task_kind, model_name)
    if spec is None:
        raise ValueError(f"Model {model_name!r} is not available for task_kind={task_kind!r}.")

    base_cls = spec.estimator_cls
    prox_cls = with_sparse_proximity(base_cls)
    kwargs = {
        "n_estimators": n_estimators,
        "random_state": random_state,
        "prox_method": prox_method,
        "matrix_type": "sparse",
        "n_jobs": int(os.environ.get("FORESTGAP_N_JOBS", "-1")),
    }
    signature = inspect.signature(base_cls.__init__)
    if "bootstrap" in signature.parameters:
        kwargs["bootstrap"] = bootstrap
    if "oob_score" in signature.parameters:
        kwargs["oob_score"] = oob_score
    elif oob_score:
        kwargs["oob_score"] = oob_score
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return prox_cls(**kwargs)


def classification_probability_metrics(
    prox_prob: np.ndarray,
    forest_prob: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    diff = np.asarray(prox_prob, dtype=np.float64) - np.asarray(forest_prob, dtype=np.float64)
    return {
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "accuracy": float(accuracy_score(y_true, np.argmax(forest_prob, axis=1))),
    }


def regression_prediction_metrics(
    prox_pred: np.ndarray,
    forest_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    diff = np.asarray(prox_pred, dtype=np.float64) - np.asarray(forest_pred, dtype=np.float64)
    return {
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "r2": float(r2_score(y_true, forest_pred)),
    }


def imputation_metrics(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    mask: np.ndarray,
    categorical_indices: Sequence[int],
) -> dict[str, float]:
    cat_set = set(int(idx) for idx in categorical_indices)
    numeric_scores: list[float] = []
    categorical_match = 0
    categorical_total = 0

    for col_idx in range(X_true.shape[1]):
        col_mask = mask[:, col_idx]
        if not np.any(col_mask):
            continue
        truth = X_true[col_mask, col_idx]
        pred = X_imputed[col_mask, col_idx]
        if col_idx in cat_set:
            categorical_total += int(col_mask.sum())
            categorical_match += int(np.sum(truth == pred))
        else:
            numeric_scores.extend(np.square(truth - pred).tolist())

    numeric_rmse = float(np.sqrt(np.mean(numeric_scores))) if numeric_scores else float("nan")
    categorical_accuracy = (
        float(categorical_match / categorical_total) if categorical_total else float("nan")
    )
    return {
        "numeric_rmse": numeric_rmse,
        "categorical_accuracy": categorical_accuracy,
        "n_numeric_masked": int(len(numeric_scores)),
        "n_categorical_masked": int(categorical_total),
    }


def induced_missing_masks(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    missing_rate: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    train_mask = np.zeros_like(X_train, dtype=bool)
    test_mask = np.zeros_like(X_test, dtype=bool)

    for col_idx in range(X_train.shape[1]):
        train_candidates = np.flatnonzero(~np.isnan(X_train[:, col_idx]))
        test_candidates = np.flatnonzero(~np.isnan(X_test[:, col_idx]))
        n_train_missing = min(train_candidates.size, int(np.floor(missing_rate * train_candidates.size)))
        n_test_missing = min(test_candidates.size, int(np.floor(missing_rate * test_candidates.size)))
        if n_train_missing > 0:
            train_selected = rng.choice(train_candidates, size=n_train_missing, replace=False)
            train_mask[train_selected, col_idx] = True
        if n_test_missing > 0:
            test_selected = rng.choice(test_candidates, size=n_test_missing, replace=False)
            test_mask[test_selected, col_idx] = True

    return train_mask, test_mask


def apply_missing_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.array(X, copy=True)
    out[mask] = np.nan
    return out


def iterative_imputation(
    prepared: PreparedDataset,
    *,
    model_name: ModelKey,
    random_state: int,
    n_estimators: int,
    n_iterations: int,
    X_train_masked: np.ndarray,
    X_test_masked: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    current_train, current_test = rough_fill_train_test(
        X_train_masked,
        X_test_masked,
        prepared.categorical_indices,
    )
    fit_runtime_seconds = 0.0
    impute_runtime_seconds = 0.0
    for _ in range(n_iterations):
        model = build_proximity_model(
            prepared.task_kind,
            model_name=model_name,
            prox_method="rfgap",
            random_state=random_state,
            n_estimators=n_estimators,
            bootstrap=True,
            oob_score=False,
        )
        fit_start = time.perf_counter()
        model.fit(current_train, prepared.y_train)
        fit_runtime_seconds += time.perf_counter() - fit_start
        impute_start = time.perf_counter()
        current_train, current_test = model.impute(
            X_train_masked,
            categorical_features=prepared.categorical_indices,
            X_test=X_test_masked,
            initialization="median",
        )
        impute_runtime_seconds += time.perf_counter() - impute_start
    return (
        np.asarray(current_train, dtype=np.float64),
        np.asarray(current_test, dtype=np.float64),
        {
            "fit_runtime_seconds": float(fit_runtime_seconds),
            "impute_runtime_seconds": float(impute_runtime_seconds),
            "total_runtime_seconds": float(fit_runtime_seconds + impute_runtime_seconds),
        },
    )


def corrupt_dataset(
    X: np.ndarray,
    categorical_indices: Sequence[int],
    *,
    random_state: int,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    X_corrupt = np.array(X, copy=True)
    n_samples = X.shape[0]
    n_corrupt = max(1, int(np.floor(fraction * n_samples)))
    corrupt_rows = np.sort(rng.choice(np.arange(n_samples), size=n_corrupt, replace=False))
    labels = np.zeros(n_samples, dtype=np.int64)
    labels[corrupt_rows] = 1

    cat_set = set(int(idx) for idx in categorical_indices)
    for col_idx in range(X.shape[1]):
        observed = X[:, col_idx][~np.isnan(X[:, col_idx])]
        if observed.size == 0:
            continue
        if col_idx in cat_set:
            values = np.unique(observed)
            X_corrupt[corrupt_rows, col_idx] = rng.choice(values, size=n_corrupt, replace=True)
        else:
            scale = float(np.nanstd(observed))
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            X_corrupt[corrupt_rows, col_idx] += rng.normal(
                loc=0.0,
                scale=6.0 * scale,
                size=n_corrupt,
            )

    return X_corrupt, labels


def outlier_metrics(scores: np.ndarray, labels: np.ndarray, top_fraction: float) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    metrics: dict[str, float] = {}
    metrics["roc_auc"] = float(roc_auc_score(labels, scores))
    metrics["average_precision"] = float(average_precision_score(labels, scores))
    n_top = max(1, int(np.ceil(top_fraction * labels.size)))
    ranked = np.argsort(scores)[::-1][:n_top]
    metrics["topk_precision"] = float(labels[ranked].mean())
    return metrics


def record_exception(
    *,
    experiment_name: str,
    task_kind: TaskKind,
    random_state: int,
    dataset_name: str,
    dataset_id: int,
    task_id: int,
    error: Exception,
) -> dict[str, Any]:
    return {
        "experiment": experiment_name,
        "task_kind": task_kind,
        "random_state": random_state,
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "task_id": task_id,
        "status": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(limit=8),
    }


def aggregate_success_rows(df: pd.DataFrame, group_cols: Sequence[str], metric_cols: Sequence[str]) -> pd.DataFrame:
    success = df[df["status"] == "ok"].copy()
    if success.empty:
        return pd.DataFrame(columns=list(group_cols) + list(metric_cols))
    grouped = success.groupby(list(group_cols), dropna=False)[list(metric_cols)].mean().reset_index()
    return grouped.sort_values(list(group_cols)).reset_index(drop=True)
