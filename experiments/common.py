from __future__ import annotations

import argparse
import inspect
from importlib import import_module
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
import time
import traceback
from typing import Any, Iterator, Literal, Optional, Sequence

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
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from forestgap import (
    OpenMLTaskFold,
    Proximity,
    impute,
    iter_openml_task_folds,
    load_openml_dataset,
)


TaskKind = Literal["classification", "regression"]
ModelKey = Literal[
    "RandomForest",
    "ExtraTrees",
    "RotationForest",
    "HonestForest",
    "ObliqueRandomForest",
    "PatchObliqueRandomForest",
    "ExtraObliqueRandomForest",
]
OutlierModelFamily = Literal["supervised_forest", "unsupervised_model"]
OutlierLabelMode = Literal["supervised", "unsupervised", "not_applicable"]

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
CLASSIFICATION_SUITE_ID = 99
REGRESSION_SUITE_ID = 455
REGRESSION_SUITE_NAME = "New_OpenML_Suite_2025_regression"
REQUESTED_MODEL_ORDER: tuple[ModelKey, ...] = (
    "RandomForest",
    "RotationForest",
    "HonestForest",
    "ObliqueRandomForest",
    "PatchObliqueRandomForest",
    "ExtraObliqueRandomForest",
    "ExtraTrees",
)
REQUESTED_OUTLIER_MODEL_ORDER: tuple[str, ...] = (
    *REQUESTED_MODEL_ORDER,
    "IsolationForest",
    "ExtendedIsolationForest",
    "UnsupervisedRandomForest",
    "UnsupervisedObliqueRandomForest",
)
# Legacy application runners use these declarations only to construct their
# requested cells.  The capability-aware runners in ``program.py`` remain the
# authoritative runtime filter.  Keep directed RF-GAP/Honest-GAP out of the
# anomaly runner: it is neither a predictive-weight application nor a declared
# geometry symmetrisation study.
OUTLIER_PROX_METHODS: tuple[str, ...] = (
    "original", "oob", "rfgap", "kerf", "path", "englund", "random_height",
)
OUTLIER_SCORE_METHODS: tuple[str, ...] = ("breiman", "ene")
DEFAULT_MISSINGNESS_MECHANISMS: tuple[str, ...] = ("MCAR", "MAR", "MNAR")
DEFAULT_MISSING_RATES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75)
DOWNSTREAM_MODEL_ORDER: dict[TaskKind, tuple[str, ...]] = {
    "classification": ("LogisticRegression", "KNeighborsClassifier"),
    "regression": ("Lasso", "KNeighborsRegressor"),
}
LOW_SUPPORT_CATEGORY_MIN_COUNT = 5

_BAGGED_PROXIMITY_METHODS: tuple[str, ...] = (
    "original", "oob", "rfgap", "kerf", "path", "englund", "random_height",
)
_HONEST_PROXIMITY_METHODS: tuple[str, ...] = (
    "original", "honest_gap", "kerf", "path", "englund", "random_height",
)


def proximity_methods_for_model(model_name: str) -> tuple[str, ...]:
    """Return legacy-runner relation requests for an installed model family.

    The function intentionally declares the full protocol candidate set; an
    individual fit can still emit an unsupported status when its bootstrap/OOB
    or optional-backend runtime contract is absent.
    """

    if model_name == "HonestForest":
        return _HONEST_PROXIMITY_METHODS
    return _BAGGED_PROXIMITY_METHODS


@dataclass(frozen=True)
class ModelSpec:
    key: ModelKey
    estimator_cls: type[Any]
    task_kind: TaskKind


@dataclass(frozen=True)
class OutlierModelSpec:
    key: str
    estimator_cls: type[Any]
    task_kind: TaskKind
    model_family: OutlierModelFamily
    fit_mode: Literal["fit_x", "fit_xy"]
    supports_label_aware_scores: bool
    supports_native_scores: bool
    native_score_method: Optional[str] = None
    native_scores_share_fit_across_prox_methods: bool = False


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

    def register_optional(
        task_kind: TaskKind,
        key: ModelKey,
        module_name: str,
        class_name: str,
    ) -> None:
        try:
            module = __import__(module_name, fromlist=[class_name])
            estimator_cls = getattr(module, class_name)
        except Exception:
            return
        registry[(task_kind, key)] = ModelSpec(
            key=key,
            estimator_cls=estimator_cls,
            task_kind=task_kind,
        )

    register_optional(
        "classification",
        "RotationForest",
        "aeon.classification.sklearn",
        "RotationForestClassifier",
    )
    register_optional(
        "regression",
        "RotationForest",
        "aeon.regression.sklearn",
        "RotationForestRegressor",
    )
    register_optional(
        "classification",
        "HonestForest",
        "treeple",
        "HonestForestClassifier",
    )
    register_optional(
        "classification",
        "ObliqueRandomForest",
        "treeple",
        "ObliqueRandomForestClassifier",
    )
    register_optional(
        "regression",
        "ObliqueRandomForest",
        "treeple",
        "ObliqueRandomForestRegressor",
    )
    register_optional(
        "classification",
        "PatchObliqueRandomForest",
        "treeple",
        "PatchObliqueRandomForestClassifier",
    )
    register_optional(
        "regression",
        "PatchObliqueRandomForest",
        "treeple",
        "PatchObliqueRandomForestRegressor",
    )
    register_optional(
        "classification",
        "ExtraObliqueRandomForest",
        "treeple",
        "ExtraObliqueRandomForestClassifier",
    )
    register_optional(
        "regression",
        "ExtraObliqueRandomForest",
        "treeple",
        "ExtraObliqueRandomForestRegressor",
    )

    return registry


MODEL_REGISTRY = _build_model_registry()


def _build_outlier_model_registry() -> dict[tuple[TaskKind, str], OutlierModelSpec]:
    registry: dict[tuple[TaskKind, str], OutlierModelSpec] = {}

    for task_kind in ("classification", "regression"):
        typed_task_kind = task_kind  # keep a concrete literal-like value for readability
        for model_name in REQUESTED_MODEL_ORDER:
            spec = MODEL_REGISTRY.get((typed_task_kind, model_name))
            if spec is None:
                continue
            registry[(typed_task_kind, model_name)] = OutlierModelSpec(
                key=model_name,
                estimator_cls=spec.estimator_cls,
                task_kind=typed_task_kind,
                model_family="supervised_forest",
                fit_mode="fit_xy",
                supports_label_aware_scores=(typed_task_kind == "classification"),
                supports_native_scores=False,
            )

        registry[(typed_task_kind, "IsolationForest")] = OutlierModelSpec(
            key="IsolationForest",
            estimator_cls=IsolationForest,
            task_kind=typed_task_kind,
            model_family="unsupervised_model",
            fit_mode="fit_x",
            supports_label_aware_scores=False,
            supports_native_scores=True,
            native_score_method="score_samples",
            native_scores_share_fit_across_prox_methods=True,
        )

    try:
        from treeple import (
            ExtendedIsolationForest,
            UnsupervisedObliqueRandomForest,
            UnsupervisedRandomForest,
        )

        for task_kind in ("classification", "regression"):
            typed_task_kind = task_kind
            registry[(typed_task_kind, "ExtendedIsolationForest")] = OutlierModelSpec(
                key="ExtendedIsolationForest",
                estimator_cls=ExtendedIsolationForest,
                task_kind=typed_task_kind,
                model_family="unsupervised_model",
                fit_mode="fit_x",
                supports_label_aware_scores=False,
                supports_native_scores=True,
                native_score_method="score_samples",
                native_scores_share_fit_across_prox_methods=True,
            )
            registry[(typed_task_kind, "UnsupervisedRandomForest")] = OutlierModelSpec(
                key="UnsupervisedRandomForest",
                estimator_cls=UnsupervisedRandomForest,
                task_kind=typed_task_kind,
                model_family="unsupervised_model",
                fit_mode="fit_x",
                supports_label_aware_scores=False,
                supports_native_scores=False,
            )
            registry[(typed_task_kind, "UnsupervisedObliqueRandomForest")] = OutlierModelSpec(
                key="UnsupervisedObliqueRandomForest",
                estimator_cls=UnsupervisedObliqueRandomForest,
                task_kind=typed_task_kind,
                model_family="unsupervised_model",
                fit_mode="fit_x",
                supports_label_aware_scores=False,
                supports_native_scores=False,
            )
    except Exception:
        pass

    return registry


OUTLIER_MODEL_REGISTRY = _build_outlier_model_registry()


@dataclass
class PreparedDataset:
    task_kind: TaskKind
    suite_id: int
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
    fold_id: str | None = None
    repeat: int | None = None
    fold: int | None = None
    sample: int | None = None
    train_indices: np.ndarray | None = None
    test_indices: np.ndarray | None = None


@dataclass(frozen=True)
class MissingMaskResult:
    mechanism: str
    missing_rate: float
    train_mask: np.ndarray
    test_mask: np.ndarray
    metadata: dict[str, Any]


def parse_random_states(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_csv_strings(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(part) for part in parse_csv_strings(text)]


def parse_missingness_mechanisms(text: str) -> list[str]:
    mechanisms = [mechanism.upper() for mechanism in parse_csv_strings(text)]
    invalid = sorted(set(mechanisms) - set(DEFAULT_MISSINGNESS_MECHANISMS))
    if invalid:
        raise ValueError(
            "Unknown missingness mechanisms: "
            + ", ".join(invalid)
            + f". Expected a subset of {DEFAULT_MISSINGNESS_MECHANISMS!r}."
        )
    return mechanisms


def parse_missing_rates(text: str) -> list[float]:
    rates = parse_csv_floats(text)
    for rate in rates:
        if rate < 0.0 or rate > 1.0:
            raise ValueError(f"Missing rates must lie in [0, 1], received {rate}.")
    return rates


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
    return REGRESSION_SUITE_ID, REGRESSION_SUITE_NAME


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_output_path(experiment_name: str, task_kind: TaskKind) -> Path:
    return ensure_dir(RAW_RESULTS_DIR / experiment_name) / f"{task_kind}_raw.csv"


def compiled_output_dir(experiment_name: str, task_kind: TaskKind) -> Path:
    return ensure_dir(RESULTS_DIR / experiment_name / task_kind)


def requested_model_names() -> tuple[ModelKey, ...]:
    return REQUESTED_MODEL_ORDER


def requested_outlier_model_names() -> tuple[str, ...]:
    return REQUESTED_OUTLIER_MODEL_ORDER


def available_model_names(task_kind: TaskKind) -> list[ModelKey]:
    return [
        model_key
        for model_key in REQUESTED_MODEL_ORDER
        if (task_kind, model_key) in MODEL_REGISTRY
    ]


def available_outlier_model_names(task_kind: TaskKind) -> list[str]:
    return [
        model_name
        for model_name in REQUESTED_OUTLIER_MODEL_ORDER
        if (task_kind, model_name) in OUTLIER_MODEL_REGISTRY
    ]


def downstream_model_names(task_kind: TaskKind) -> tuple[str, ...]:
    return DOWNSTREAM_MODEL_ORDER[task_kind]


def model_spec(task_kind: TaskKind, model_name: ModelKey) -> Optional[ModelSpec]:
    return MODEL_REGISTRY.get((task_kind, model_name))


def outlier_model_spec(task_kind: TaskKind, model_name: str) -> Optional[OutlierModelSpec]:
    return OUTLIER_MODEL_REGISTRY.get((task_kind, model_name))


def estimator_supports_parameter(
    task_kind: TaskKind,
    model_name: ModelKey,
    parameter_name: str,
) -> bool:
    spec = model_spec(task_kind, model_name)
    if spec is None:
        return False
    return parameter_name in inspect.signature(spec.estimator_cls.__init__).parameters


def outlier_label_modes(task_kind: TaskKind, model_name: str) -> tuple[OutlierLabelMode, ...]:
    spec = outlier_model_spec(task_kind, model_name)
    if spec is None:
        return tuple()
    if spec.model_family == "unsupervised_model":
        return ("unsupervised",)
    if task_kind == "classification" and spec.supports_label_aware_scores:
        return ("supervised", "unsupervised")
    return ("unsupervised",)


def unsupported_model_row(
    *,
    experiment_name: str,
    task_kind: TaskKind,
    model_name: str,
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
    if df.empty:
        return
    ensure_dir(path.parent)
    width = max(10.0, min(24.0, 0.4 * max(len(df), 1)))
    fig, ax = plt.subplots(figsize=(width, 5.5))
    df.plot(kind="bar", x=x, y=y, ax=ax, legend=False, color="#2c7fb8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=rotation)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=r".*layout.*")
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_lineplot_dataframe(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    group_col: str,
    facet_col: Optional[str],
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    if df.empty:
        return
    ensure_dir(path.parent)
    facet_values = [None] if facet_col is None else sorted(df[facet_col].dropna().unique())
    fig, axes = plt.subplots(
        1,
        len(facet_values),
        figsize=(6 * len(facet_values), 5),
        squeeze=False,
        sharey=True,
    )
    for axis, facet_value in zip(axes[0], facet_values):
        facet_df = df if facet_col is None else df[df[facet_col] == facet_value]
        for group_value, group_df in facet_df.groupby(group_col, dropna=False):
            ordered = group_df.sort_values(x)
            axis.plot(
                ordered[x].to_numpy(dtype=np.float64),
                ordered[y].to_numpy(dtype=np.float64),
                marker="o",
                linewidth=1.5,
                label=str(group_value),
            )
        if facet_col is not None:
            axis.set_title(f"{facet_col}={facet_value}")
        axis.set_xlabel(x.replace("_", " ").title())
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, linestyle=":")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def iter_openml_suite(task_kind: TaskKind, limit: Optional[int] = None) -> Iterator[tuple[int, Any]]:
    try:
        openml = import_module("openml")
    except ImportError as exc:
        raise ImportError(
            "OpenML experiment suites require the optional 'openml' extra; "
            "install forestgap[openml] before running full-data experiments."
        ) from exc
    suite_id, _ = suite_metadata(task_kind)
    suite = openml.study.get_suite(suite_id)
    task_ids = list(getattr(suite, "tasks", []))
    if limit is not None:
        task_ids = task_ids[:limit]
    for task_id in task_ids:
        yield int(task_id), openml.tasks.get_task(int(task_id))


def iter_openml_suite_folds(
    task_kind: TaskKind,
    limit: Optional[int] = None,
) -> Iterator[OpenMLTaskFold]:
    """Yield every official OpenML outer fold for a declared benchmark suite.

    This is the study-program iterator.  The older :func:`iter_openml_suite`
    remains available for backwards-compatible exploratory scripts, while new
    benchmark runners must consume this iterator and pass its values to
    :func:`prepare_dataset`.
    """

    try:
        openml = import_module("openml")
    except ImportError as exc:
        raise ImportError(
            "OpenML experiment suites require the optional 'openml' extra; "
            "install forestgap[openml] before running full-data experiments."
        ) from exc
    suite_id, _ = suite_metadata(task_kind)
    suite = openml.study.get_suite(suite_id)
    task_ids = list(getattr(suite, "tasks", []))
    if limit is not None:
        task_ids = task_ids[:limit]
    for task_id in task_ids:
        yield from iter_openml_task_folds(task_id=int(task_id), as_frame=True)


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
            or isinstance(X[column].dtype, pd.CategoricalDtype)
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


def _prepare_official_fold_dataset(
    task_kind: TaskKind,
    *,
    suite_id: int,
    suite_name: str,
    task_id: int,
    official_fold: OpenMLTaskFold,
    X_df: pd.DataFrame,
    y_series: pd.Series,
    categorical_indicator: Sequence[bool],
    feature_cap: Optional[int],
    max_rows: Optional[int],
) -> PreparedDataset:
    """Prepare an OpenML outer fold without changing its train/test boundary."""

    if max_rows is not None:
        raise ValueError(
            "max_rows cannot be used with official OpenML folds because it would "
            "change the declared outer evaluation protocol."
        )
    if int(task_id) != official_fold.task_id:
        raise ValueError(
            f"Official fold task_id={official_fold.task_id} does not match task_id={task_id}."
        )

    train_indices = np.asarray(official_fold.train_indices, dtype=np.int64)
    test_indices = np.asarray(official_fold.test_indices, dtype=np.int64)
    # The task/fold loader validates bounds and disjointness.  Slice first so
    # target filtering and feature processing cannot leak test-row statistics.
    X_train_df = X_df.iloc[train_indices].reset_index(drop=True)
    X_test_df = X_df.iloc[test_indices].reset_index(drop=True)
    y_train_raw = y_series.iloc[train_indices].reset_index(drop=True)
    y_test_raw = y_series.iloc[test_indices].reset_index(drop=True)

    if task_kind == "classification":
        categories = pd.Index(pd.Series(y_train_raw, dtype="object").dropna().unique())
        if categories.size < 2:
            raise ValueError("Need at least two training classes in an official outer fold.")
        y_train = pd.Categorical(y_train_raw, categories=categories).codes.astype(np.int64)
        y_test = pd.Categorical(y_test_raw, categories=categories).codes.astype(np.int64)
        if np.any(y_train < 0) or np.any(y_test < 0):
            raise ValueError(
                "Official OpenML classification folds must have observed target values "
                "and no test-only classes."
            )
    else:
        y_train = pd.to_numeric(y_train_raw, errors="coerce").to_numpy(dtype=np.float64)
        y_test = pd.to_numeric(y_test_raw, errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
            raise ValueError("Official OpenML regression folds must have finite targets.")

    categorical_columns = _categorical_columns(X_train_df, categorical_indicator)
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
    dataset = official_fold.dataset
    return PreparedDataset(
        task_kind=task_kind,
        suite_id=suite_id,
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
        original_n_samples=int(len(X_df)),
        original_n_features=int(X_df.shape[1]),
        train_size=int(X_train.shape[0]),
        test_size=int(X_test.shape[0]),
        fold_id=official_fold.fold_id,
        repeat=official_fold.repeat,
        fold=official_fold.fold,
        sample=official_fold.sample,
        train_indices=train_indices.copy(),
        test_indices=test_indices.copy(),
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
    official_fold: OpenMLTaskFold | None = None,
) -> PreparedDataset:
    suite_id, suite_name = suite_metadata(task_kind)
    if official_fold is not None:
        dataset = official_fold.dataset
    else:
        dataset = load_openml_dataset(
            dataset_id=int(task.dataset_id),
            target_name=getattr(task, "target_name", None),
            as_frame=True,
        )
    X_df = _coerce_frame(dataset.X, dataset.feature_names)
    y_series = _coerce_target(dataset.y)

    if official_fold is not None:
        return _prepare_official_fold_dataset(
            task_kind,
            suite_id=suite_id,
            suite_name=suite_name,
            task_id=task_id,
            official_fold=official_fold,
            X_df=X_df,
            y_series=y_series,
            categorical_indicator=dataset.categorical_indicator,
            feature_cap=feature_cap,
            max_rows=max_rows,
        )

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
        suite_id=suite_id,
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
    kwargs = {
        "n_estimators": n_estimators,
        "random_state": random_state,
    }
    signature = inspect.signature(base_cls.__init__)
    if "n_jobs" in signature.parameters:
        kwargs["n_jobs"] = int(os.environ.get("FORESTGAP_N_JOBS", "-1"))
    if "bootstrap" in signature.parameters:
        kwargs["bootstrap"] = bootstrap
    if "oob_score" in signature.parameters:
        kwargs["oob_score"] = oob_score
    proximity_kwargs: dict[str, Any] = {}
    if extra_kwargs:
        for key, value in extra_kwargs.items():
            if key in {"matrix_type", "non_zero_diagonal", "symm_mode", "max_normalize"}:
                proximity_kwargs[key] = value
            else:
                kwargs[key] = value
    estimator = base_cls(**kwargs)
    return Proximity(estimator, relation=prox_method, matrix_type="sparse", **proximity_kwargs)


def build_outlier_model(
    task_kind: TaskKind,
    *,
    model_name: str,
    prox_method: str,
    random_state: int,
    n_estimators: int,
    isolation_bootstrap: bool,
    extra_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[Any, OutlierModelSpec, dict[str, Any]]:
    spec = outlier_model_spec(task_kind, model_name)
    if spec is None:
        raise ValueError(f"Outlier model {model_name!r} is not available for task_kind={task_kind!r}.")

    base_cls = spec.estimator_cls
    signature = inspect.signature(base_cls.__init__)
    kwargs: dict[str, Any] = {
        "n_estimators": n_estimators,
        "random_state": random_state,
    }
    if "n_jobs" in signature.parameters:
        kwargs["n_jobs"] = int(os.environ.get("FORESTGAP_N_JOBS", "-1"))

    if spec.model_family == "supervised_forest":
        if "bootstrap" in signature.parameters:
            kwargs["bootstrap"] = True
        if "oob_score" in signature.parameters:
            kwargs["oob_score"] = prox_method == "rfgap"
    elif model_name in {"IsolationForest", "ExtendedIsolationForest"}:
        if "bootstrap" in signature.parameters:
            kwargs["bootstrap"] = bool(isolation_bootstrap)
    else:
        if "bootstrap" in signature.parameters:
            kwargs["bootstrap"] = prox_method in ("oob", "rfgap")
        if "oob_score" in signature.parameters:
            kwargs["oob_score"] = False

    if extra_kwargs:
        kwargs.update(extra_kwargs)

    fit_metadata = {
        "fit_mode": spec.fit_mode,
        "fit_bootstrap": bool(kwargs.get("bootstrap", False)),
        "native_score_shared_across_prox_methods": (
            spec.supports_native_scores and spec.native_scores_share_fit_across_prox_methods
        ),
    }
    return Proximity(base_cls(**kwargs), relation=prox_method, matrix_type="sparse"), spec, fit_metadata


def fit_outlier_model(model: Any, spec: OutlierModelSpec, X: np.ndarray, y: np.ndarray) -> Any:
    if spec.fit_mode == "fit_xy":
        model.fit(X, y)
    else:
        model.fit(X)
    return model


def native_outlier_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if not hasattr(model, "score_samples"):
        raise AttributeError(f"{type(model).__name__} does not expose score_samples().")
    return -np.asarray(model.score_samples(X), dtype=np.float64)


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


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _logit(value: float) -> float:
    clipped = float(np.clip(value, 1e-6, 1.0 - 1e-6))
    return math.log(clipped / (1.0 - clipped))


def _calibrate_logistic_intercept(scores: np.ndarray, target_rate: float) -> float:
    if target_rate <= 0.0:
        return -30.0
    if target_rate >= 1.0:
        return 30.0
    clean_scores = np.asarray(scores, dtype=np.float64)
    if clean_scores.size == 0 or np.allclose(clean_scores, clean_scores[0]):
        return _logit(target_rate) - float(clean_scores[0] if clean_scores.size else 0.0)
    low = -30.0
    high = 30.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        mean_prob = float(np.mean(_sigmoid(clean_scores + mid)))
        if mean_prob > target_rate:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def _weighted_sample_mask(
    candidate_indices: np.ndarray,
    n_select: int,
    weights: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    mask = np.zeros(candidate_indices.shape[0], dtype=bool)
    if n_select <= 0 or candidate_indices.size == 0:
        return mask
    if n_select >= candidate_indices.size:
        mask[:] = True
        return mask
    clean_weights = np.asarray(weights, dtype=np.float64)
    clean_weights = np.where(np.isfinite(clean_weights) & (clean_weights > 0.0), clean_weights, 0.0)
    if float(clean_weights.sum()) <= 0.0:
        clean_weights = np.ones(candidate_indices.size, dtype=np.float64)
    probabilities = clean_weights / clean_weights.sum()
    selected_local = rng.choice(candidate_indices.size, size=n_select, replace=False, p=probabilities)
    mask[selected_local] = True
    return mask


def _standardize_with_train(train_column: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, bool]:
    observed = train_column[~np.isnan(train_column)]
    if observed.size == 0:
        return np.zeros(values.shape[0], dtype=np.float64), False
    scale = float(np.nanstd(observed))
    if not np.isfinite(scale) or scale == 0.0:
        return np.zeros(values.shape[0], dtype=np.float64), False
    centered = np.asarray(values, dtype=np.float64) - float(np.nanmean(observed))
    standardized = centered / scale
    standardized[np.isnan(standardized)] = 0.0
    return standardized, True


def _quantile_rank_scores(train_observed: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, bool]:
    clean_train = np.sort(np.asarray(train_observed, dtype=np.float64))
    if clean_train.size == 0 or np.unique(clean_train).size < 2:
        return np.zeros(values.shape[0], dtype=np.float64), False
    ranks = np.searchsorted(clean_train, np.asarray(values, dtype=np.float64), side="right") + 1
    probs = np.clip(ranks / (clean_train.size + 1.0), 1e-4, 1.0 - 1e-4)
    scores = np.log(probs / (1.0 - probs))
    scores[np.isnan(scores)] = 0.0
    return scores, True


def _feature_label(feature_names: Optional[Sequence[str]], col_idx: int) -> str:
    if feature_names is None or col_idx >= len(feature_names):
        return f"x{col_idx + 1}"
    return str(feature_names[col_idx])


def _mar_conditioning_candidates(
    X_train: np.ndarray,
    target_idx: int,
) -> list[dict[str, Any]]:
    target_column = X_train[:, target_idx]
    target_observed = target_column[~np.isnan(target_column)]
    target_has_variation = target_observed.size >= 8 and np.unique(target_observed).size >= 2
    candidates: list[dict[str, Any]] = []
    for col_idx in range(X_train.shape[1]):
        if col_idx == target_idx:
            continue
        column = X_train[:, col_idx]
        observed = column[~np.isnan(column)]
        if observed.size < 8:
            continue
        scale = float(np.nanstd(observed))
        if not np.isfinite(scale) or scale == 0.0:
            continue
        overlap = (~np.isnan(target_column)) & (~np.isnan(column))
        overlap_count = int(overlap.sum())
        if overlap_count < 8:
            continue
        correlation = float("nan")
        if target_has_variation:
            corr = np.corrcoef(target_column[overlap], column[overlap])[0, 1]
            if np.isfinite(corr):
                correlation = float(corr)
        candidates.append(
            {
                "col_idx": col_idx,
                "abs_correlation": abs(correlation) if np.isfinite(correlation) else -1.0,
                "correlation": correlation,
                "overlap_count": overlap_count,
            }
        )
    return sorted(
        candidates,
        key=lambda item: (item["abs_correlation"], item["overlap_count"]),
        reverse=True,
    )


def _mar_sampling_weights(
    X_train: np.ndarray,
    X_split: np.ndarray,
    *,
    target_idx: int,
    target_rate: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidates = _mar_conditioning_candidates(X_train, target_idx)
    if not candidates:
        return np.ones(X_split.shape[0], dtype=np.float64), {
            "strategy": "mcar_uniform",
            "fallback_reason": "mcar_no_usable_conditioning_column",
            "conditioning_feature_indices": [],
        }
    selected = candidates[: min(2, len(candidates))]
    train_score = np.zeros(X_train.shape[0], dtype=np.float64)
    split_score = np.zeros(X_split.shape[0], dtype=np.float64)
    selected_indices: list[int] = []
    for candidate in selected:
        train_standardized, train_usable = _standardize_with_train(
            X_train[:, candidate["col_idx"]],
            X_train[:, candidate["col_idx"]],
        )
        split_standardized, split_usable = _standardize_with_train(
            X_train[:, candidate["col_idx"]],
            X_split[:, candidate["col_idx"]],
        )
        if not train_usable or not split_usable:
            continue
        sign = -1.0 if np.isfinite(candidate["correlation"]) and candidate["correlation"] < 0 else 1.0
        train_score += sign * train_standardized
        split_score += sign * split_standardized
        selected_indices.append(int(candidate["col_idx"]))
    if not selected_indices:
        return np.ones(X_split.shape[0], dtype=np.float64), {
            "strategy": "mcar_uniform",
            "fallback_reason": "mcar_no_usable_conditioning_column",
            "conditioning_feature_indices": [],
        }
    train_score /= float(len(selected_indices))
    split_score /= float(len(selected_indices))
    observed_target = ~np.isnan(X_train[:, target_idx])
    intercept = _calibrate_logistic_intercept(train_score[observed_target], target_rate)
    probabilities = _sigmoid(split_score + intercept)
    return probabilities, {
        "strategy": "mar_logistic",
        "conditioning_feature_indices": selected_indices,
    }


def _mnar_numeric_sampling_weights(
    train_column: np.ndarray,
    split_column: np.ndarray,
    *,
    target_rate: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    observed_train = train_column[~np.isnan(train_column)]
    train_score, usable = _quantile_rank_scores(observed_train, train_column)
    if not usable:
        return np.ones(split_column.shape[0], dtype=np.float64), {
            "strategy": "mcar_uniform",
            "fallback_reason": "mcar_constant_numeric_feature",
        }
    split_score, _ = _quantile_rank_scores(observed_train, split_column)
    intercept = _calibrate_logistic_intercept(train_score[~np.isnan(train_column)], target_rate)
    return _sigmoid(split_score + intercept), {"strategy": "mnar_numeric_quantile"}


def _mnar_categorical_sampling_weights(
    train_column: np.ndarray,
    split_column: np.ndarray,
    *,
    target_rate: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    observed_train = train_column[~np.isnan(train_column)]
    if observed_train.size == 0:
        return np.ones(split_column.shape[0], dtype=np.float64), {
            "strategy": "mcar_uniform",
            "fallback_reason": "mcar_all_missing_training_feature",
        }
    values, counts = np.unique(observed_train, return_counts=True)
    if values.size < 2:
        return np.ones(split_column.shape[0], dtype=np.float64), {
            "strategy": "mcar_uniform",
            "fallback_reason": "mcar_constant_categorical_feature",
        }
    min_support = min(
        LOW_SUPPORT_CATEGORY_MIN_COUNT,
        max(2, int(math.ceil(0.05 * observed_train.size))),
    )
    supported = counts >= min_support
    if int(np.sum(supported)) < 2:
        return np.ones(split_column.shape[0], dtype=np.float64), {
            "strategy": "mcar_uniform",
            "fallback_reason": "mcar_low_support_categorical_levels",
        }
    rarity = -np.log(counts.astype(np.float64))
    supported_center = float(np.mean(rarity[supported]))
    supported_scale = float(np.std(rarity[supported]))
    if not np.isfinite(supported_scale) or supported_scale == 0.0:
        supported_scale = 1.0
    score_map: dict[float, float] = {}
    pooled_any = False
    for value, raw_score, is_supported in zip(values, rarity, supported):
        if bool(is_supported):
            score_map[float(value)] = float((raw_score - supported_center) / supported_scale)
        else:
            score_map[float(value)] = 0.0
            pooled_any = True
    train_score = np.array(
        [score_map.get(float(value), 0.0) if not np.isnan(value) else 0.0 for value in train_column]
    )
    split_score = np.array(
        [score_map.get(float(value), 0.0) if not np.isnan(value) else 0.0 for value in split_column]
    )
    intercept = _calibrate_logistic_intercept(train_score[~np.isnan(train_column)], target_rate)
    metadata: dict[str, Any] = {"strategy": "mnar_categorical_level"}
    if pooled_any:
        metadata["rare_level_pooling"] = True
    return _sigmoid(split_score + intercept), metadata


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

    numeric_mse = float(np.mean(numeric_scores)) if numeric_scores else float("nan")
    numeric_rmse = float(np.sqrt(numeric_mse)) if numeric_scores else float("nan")
    categorical_accuracy = (
        float(categorical_match / categorical_total) if categorical_total else float("nan")
    )
    return {
        "numeric_mse": numeric_mse,
        "numeric_rmse": numeric_rmse,
        "categorical_accuracy": categorical_accuracy,
        "n_numeric_masked": int(len(numeric_scores)),
        "n_categorical_masked": int(categorical_total),
    }


def induced_missing_masks(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    categorical_indices: Sequence[int],
    missingness_mechanism: str,
    missing_rate: float,
    random_state: int,
) -> MissingMaskResult:
    """Generate deterministic train/test masks for MCAR, MAR, or MNAR.

    The mechanism parameters are fit from the training split only. Each feature is
    then masked with an exact per-split count of ``floor(rate * n_observed)`` so the
    realized masking rate is reproducible and shared across all compared methods.
    """
    mechanism = str(missingness_mechanism).upper()
    if mechanism not in DEFAULT_MISSINGNESS_MECHANISMS:
        raise ValueError(
            f"Unknown missingness mechanism {missingness_mechanism!r}; "
            f"expected one of {DEFAULT_MISSINGNESS_MECHANISMS!r}."
        )
    rng = np.random.RandomState(random_state)
    train_mask = np.zeros_like(X_train, dtype=bool)
    test_mask = np.zeros_like(X_test, dtype=bool)
    cat_set = set(int(idx) for idx in categorical_indices)
    feature_records: list[dict[str, Any]] = []
    for col_idx in range(X_train.shape[1]):
        train_candidates = np.flatnonzero(~np.isnan(X_train[:, col_idx]))
        test_candidates = np.flatnonzero(~np.isnan(X_test[:, col_idx]))
        n_train_missing = min(train_candidates.size, int(np.floor(missing_rate * train_candidates.size)))
        n_test_missing = min(test_candidates.size, int(np.floor(missing_rate * test_candidates.size)))
        record: dict[str, Any] = {
            "feature_index": int(col_idx),
            "feature_type": "categorical" if col_idx in cat_set else "numeric",
        }
        if train_candidates.size == 0 and test_candidates.size == 0:
            record["strategy"] = "no_observed_candidates"
            feature_records.append(record)
            continue
        if mechanism == "MCAR":
            train_weights = np.ones(X_train.shape[0], dtype=np.float64)
            test_weights = np.ones(X_test.shape[0], dtype=np.float64)
            record["strategy"] = "mcar_uniform"
        elif mechanism == "MAR":
            train_weights, train_metadata = _mar_sampling_weights(
                X_train,
                X_train,
                target_idx=col_idx,
                target_rate=missing_rate,
            )
            test_weights, _ = _mar_sampling_weights(
                X_train,
                X_test,
                target_idx=col_idx,
                target_rate=missing_rate,
            )
            record.update(train_metadata)
        else:
            if col_idx in cat_set:
                train_weights, train_metadata = _mnar_categorical_sampling_weights(
                    X_train[:, col_idx],
                    X_train[:, col_idx],
                    target_rate=missing_rate,
                )
                test_weights, _ = _mnar_categorical_sampling_weights(
                    X_train[:, col_idx],
                    X_test[:, col_idx],
                    target_rate=missing_rate,
                )
            else:
                train_weights, train_metadata = _mnar_numeric_sampling_weights(
                    X_train[:, col_idx],
                    X_train[:, col_idx],
                    target_rate=missing_rate,
                )
                test_weights, _ = _mnar_numeric_sampling_weights(
                    X_train[:, col_idx],
                    X_test[:, col_idx],
                    target_rate=missing_rate,
                )
            record.update(train_metadata)

        if n_train_missing > 0:
            selected_local = _weighted_sample_mask(
                train_candidates,
                n_train_missing,
                train_weights[train_candidates],
                rng,
            )
            train_mask[train_candidates[selected_local], col_idx] = True
        if n_test_missing > 0:
            selected_local = _weighted_sample_mask(
                test_candidates,
                n_test_missing,
                test_weights[test_candidates],
                rng,
            )
            test_mask[test_candidates[selected_local], col_idx] = True
        feature_records.append(record)

    strategy_counts: dict[str, int] = {}
    fallback_counts: dict[str, int] = {}
    fallback_examples: list[dict[str, Any]] = []
    for record in feature_records:
        strategy = str(record.get("strategy", "unknown"))
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        fallback_reason = record.get("fallback_reason")
        if fallback_reason:
            fallback_key = str(fallback_reason)
            fallback_counts[fallback_key] = fallback_counts.get(fallback_key, 0) + 1
            if len(fallback_examples) < 5:
                fallback_examples.append(
                    {
                        "feature_index": int(record["feature_index"]),
                        "feature_type": record.get("feature_type"),
                        "fallback_reason": fallback_key,
                    }
                )

    return MissingMaskResult(
        mechanism=mechanism,
        missing_rate=float(missing_rate),
        train_mask=train_mask,
        test_mask=test_mask,
        metadata={
            "mechanism": mechanism,
            "missing_rate": float(missing_rate),
            "strategy_counts": strategy_counts,
            "fallback_count": int(sum(fallback_counts.values())),
            "fallback_counts": fallback_counts,
            "fallback_examples": fallback_examples,
        },
    )


def apply_missing_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.array(X, copy=True)
    out[mask] = np.nan
    return out


def build_downstream_pipeline(
    task_kind: TaskKind,
    *,
    downstream_model: str,
    random_state: int,
) -> Pipeline:
    if task_kind == "classification":
        if downstream_model == "LogisticRegression":
            estimator: Any = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                random_state=random_state,
            )
        elif downstream_model == "KNeighborsClassifier":
            estimator = KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError(f"Unknown downstream_model={downstream_model!r} for classification.")
    else:
        if downstream_model == "Lasso":
            estimator = Lasso(alpha=0.01, max_iter=5000)
        elif downstream_model == "KNeighborsRegressor":
            estimator = KNeighborsRegressor(n_neighbors=5)
        else:
            raise ValueError(f"Unknown downstream_model={downstream_model!r} for regression.")
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


def _classification_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    classes = np.unique(y_true)
    if classes.size < 2:
        return float("nan")
    if probabilities.ndim == 1 or probabilities.shape[1] == 1:
        return float("nan")
    try:
        if probabilities.shape[1] == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        return float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted"))
    except Exception:
        return float("nan")


def evaluate_downstream_model(
    task_kind: TaskKind,
    *,
    downstream_model: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    categorical_indices: Sequence[int],
    random_state: int,
) -> dict[str, float]:
    """Evaluate a simple downstream model on a completed train/test feature pair."""
    fit_train, fit_test = rough_fill_train_test(X_train, X_test, categorical_indices)
    pipeline = build_downstream_pipeline(
        task_kind,
        downstream_model=downstream_model,
        random_state=random_state,
    )
    pipeline.fit(fit_train, y_train)
    predictions = pipeline.predict(fit_test)
    if task_kind == "classification":
        probabilities = pipeline.predict_proba(fit_test)
        return {
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_roc_auc": _classification_roc_auc(np.asarray(y_test), np.asarray(probabilities)),
            "test_mse": float("nan"),
            "test_r2": float("nan"),
        }
    return {
        "test_accuracy": float("nan"),
        "test_roc_auc": float("nan"),
        "test_mse": float(mean_squared_error(y_test, predictions)),
        "test_r2": float(r2_score(y_test, predictions)),
    }


def iterative_imputation(
    prepared: PreparedDataset,
    *,
    model_name: ModelKey,
    prox_method: str,
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
            prox_method=prox_method,
            random_state=random_state,
            n_estimators=n_estimators,
            bootstrap=True,
            oob_score=False,
        )
        fit_start = time.perf_counter()
        model.fit(current_train, prepared.y_train)
        fit_runtime_seconds += time.perf_counter() - fit_start
        impute_start = time.perf_counter()
        current_train, current_test = impute(
            model,
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


def outlier_metrics(scores: np.ndarray, labels: np.ndarray, top_k: int) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    metrics: dict[str, float] = {}
    metrics["roc_auc"] = float(roc_auc_score(labels, scores))
    metrics["average_precision"] = float(average_precision_score(labels, scores))
    n_top = max(1, min(int(top_k), labels.size))
    ranked = np.argsort(scores)[::-1][:n_top]
    y_pred = np.zeros_like(labels)
    y_pred[ranked] = 1
    true_positives = int(labels[ranked].sum())
    positives = int(labels.sum())
    precision = float(true_positives / n_top)
    recall = float(true_positives / positives) if positives > 0 else float("nan")
    if precision + recall > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = 0.0
    metrics["precision_at_k"] = precision
    metrics["recall_at_k"] = recall
    metrics["f1_at_k"] = f1
    return metrics


def topk_overlap_at_k(reference_scores: np.ndarray, candidate_scores: np.ndarray, top_k: int) -> float:
    n_top = max(1, min(int(top_k), len(reference_scores)))
    ref_top = set(np.argsort(np.asarray(reference_scores, dtype=np.float64))[::-1][:n_top].tolist())
    cand_top = set(np.argsort(np.asarray(candidate_scores, dtype=np.float64))[::-1][:n_top].tolist())
    return float(len(ref_top & cand_top) / n_top)


def native_proximity_agreement_metrics(
    native_scores: np.ndarray,
    proximity_scores: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    native = np.asarray(native_scores, dtype=np.float64)
    prox = np.asarray(proximity_scores, dtype=np.float64)
    spearman = spearmanr(native, prox).statistic
    return {
        "spearman_rank_corr": float(spearman) if spearman is not None else float("nan"),
        "topk_overlap_at_k": topk_overlap_at_k(native, prox, top_k),
    }


def record_exception(
    *,
    experiment_name: str,
    task_kind: TaskKind,
    random_state: int,
    dataset_name: str,
    dataset_id: int,
    task_id: int,
    error: Exception,
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    row = {
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
    if extra_fields:
        row.update(extra_fields)
    return row


def aggregate_success_rows(df: pd.DataFrame, group_cols: Sequence[str], metric_cols: Sequence[str]) -> pd.DataFrame:
    success = df[df["status"] == "ok"].copy()
    if success.empty:
        return pd.DataFrame(columns=list(group_cols) + list(metric_cols))
    grouped = success.groupby(list(group_cols), dropna=False)[list(metric_cols)].mean().reset_index()
    return grouped.sort_values(list(group_cols)).reset_index(drop=True)
