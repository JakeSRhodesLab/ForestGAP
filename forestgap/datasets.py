from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np


TargetName = Optional[Union[str, Sequence[str]]]


@dataclass
class OpenMLDataset:
    """Normalized representation of a single dataset loaded from OpenML."""

    name: str
    dataset_id: int
    version: Optional[int]
    target_name: TargetName
    X: Any
    y: Any
    feature_names: List[str]
    categorical_indicator: List[bool]
    url: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenMLBenchmarkSuite:
    """Collection of normalized OpenML datasets from a benchmark suite."""

    name: str
    suite_id: int
    task_ids: List[int]
    datasets: List[OpenMLDataset]
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenMLTask:
    """A typed OpenML predictive task and its normalized dataset payload.

    Unlike an :class:`OpenMLDataset`, an OpenML task also fixes the target and
    publishes the server-provided resampling dimensions.  The dimensions use
    OpenML's zero-based ``(repeat, fold, sample)`` indexing and should be used
    to construct outer evaluation folds rather than re-splitting the data.
    """

    task_id: int
    dataset: OpenMLDataset
    target_name: TargetName
    repeats: int
    folds: int
    samples: int
    task_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dataset_id(self) -> int:
        """Convenience identifier for manifest rows."""

        return self.dataset.dataset_id


@dataclass
class OpenMLTaskFold:
    """One official train/test split belonging to an :class:`OpenMLTask`."""

    task: OpenMLTask
    repeat: int
    fold: int
    sample: int
    train_indices: np.ndarray
    test_indices: np.ndarray

    @property
    def task_id(self) -> int:
        return self.task.task_id

    @property
    def dataset(self) -> OpenMLDataset:
        return self.task.dataset

    @property
    def dataset_id(self) -> int:
        return self.task.dataset_id

    @property
    def target_name(self) -> TargetName:
        return self.task.target_name

    @property
    def fold_id(self) -> str:
        """Stable, human-readable identifier suitable for result manifests."""

        return f"r{self.repeat}-f{self.fold}-s{self.sample}"


@dataclass
class SyntheticDataset:
    """Normalized representation of a simulated dataset."""

    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _feature_names(n_features: int) -> List[str]:
    return [f"x{i + 1}" for i in range(n_features)]


def _rng(random_state: Optional[int]) -> np.random.RandomState:
    return np.random.RandomState(random_state)


def _append_noise_features(
    rng: np.random.RandomState,
    signal: np.ndarray,
    n_features: int,
    noise_scale: float = 1.0,
) -> np.ndarray:
    if n_features < signal.shape[1]:
        raise ValueError(
            f"n_features={n_features} must be at least the signal dimension {signal.shape[1]}."
        )
    if n_features == signal.shape[1]:
        return signal
    noise = rng.normal(scale=noise_scale, size=(signal.shape[0], n_features - signal.shape[1]))
    return np.column_stack([signal, noise])


def _rotation_matrix(angle_degrees: float) -> np.ndarray:
    theta = np.deg2rad(angle_degrees)
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )


def _finalize_synthetic_dataset(
    *,
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    description: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> SyntheticDataset:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    return SyntheticDataset(
        name=name,
        X=X,
        y=y,
        feature_names=_feature_names(X.shape[1]),
        description=description,
        metadata=metadata or {},
    )


def _random_projection(
    rng: np.random.RandomState,
    n_features: int,
    n_active: Optional[int] = None,
) -> np.ndarray:
    if n_active is None or n_active >= n_features:
        weights = rng.normal(size=n_features)
    else:
        weights = np.zeros(n_features, dtype=np.float64)
        active = rng.choice(n_features, size=n_active, replace=False)
        weights[active] = rng.normal(size=n_active)
    norm = np.linalg.norm(weights)
    if norm == 0:
        weights[0] = 1.0
        norm = 1.0
    return weights / norm


def make_rotated_linear_boundary(
    n_samples: int = 400,
    n_features: int = 2,
    angle_degrees: float = 35.0,
    noise_std: float = 0.25,
    random_state: Optional[int] = None,
) -> SyntheticDataset:
    rng = _rng(random_state)
    latent = rng.normal(size=(n_samples, 2))
    y = (latent[:, 0] + latent[:, 1] + rng.normal(scale=noise_std, size=n_samples) > 0).astype(int)
    observed = latent @ _rotation_matrix(angle_degrees).T
    X = _append_noise_features(rng, observed, n_features)
    return _finalize_synthetic_dataset(
        name="rotated_linear_boundary",
        X=X,
        y=y,
        description="Binary classification with a diagonal linear boundary in rotated coordinates.",
        metadata={
            "angle_degrees": angle_degrees,
            "noise_std": noise_std,
            "n_samples": n_samples,
            "n_features": n_features,
        },
    )


def make_rotated_xor(
    n_samples: int = 400,
    n_features: int = 2,
    angle_degrees: float = 35.0,
    feature_noise_std: float = 0.1,
    random_state: Optional[int] = None,
) -> SyntheticDataset:
    rng = _rng(random_state)
    latent = rng.normal(size=(n_samples, 2))
    y = np.logical_xor(latent[:, 0] > 0, latent[:, 1] > 0).astype(int)
    observed = latent @ _rotation_matrix(angle_degrees).T
    observed += rng.normal(scale=feature_noise_std, size=observed.shape)
    X = _append_noise_features(rng, observed, n_features)
    return _finalize_synthetic_dataset(
        name="rotated_xor",
        X=X,
        y=y,
        description="XOR classification after rotating the 2D feature space.",
        metadata={
            "angle_degrees": angle_degrees,
            "feature_noise_std": feature_noise_std,
            "n_samples": n_samples,
            "n_features": n_features,
        },
    )


def make_oblique_gaussian_clusters(
    n_samples: int = 400,
    n_features: int = 2,
    correlation: float = 0.85,
    class_sep: float = 2.5,
    angle_degrees: float = 25.0,
    random_state: Optional[int] = None,
) -> SyntheticDataset:
    rng = _rng(random_state)
    n0 = n_samples // 2
    n1 = n_samples - n0
    cov = np.array([[1.0, correlation], [correlation, 1.0]])
    rotation = _rotation_matrix(angle_degrees)
    cov = rotation @ cov @ rotation.T
    mean0 = rotation @ np.array([-class_sep / 2.0, 0.0])
    mean1 = rotation @ np.array([class_sep / 2.0, 0.0])
    X0 = rng.multivariate_normal(mean=mean0, cov=cov, size=n0)
    X1 = rng.multivariate_normal(mean=mean1, cov=cov, size=n1)
    signal = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    X = _append_noise_features(rng, signal, n_features, noise_scale=0.75)
    return _finalize_synthetic_dataset(
        name="oblique_gaussian_clusters",
        X=X,
        y=y,
        description="Correlated Gaussian class clusters with oblique covariance structure.",
        metadata={
            "correlation": correlation,
            "class_sep": class_sep,
            "angle_degrees": angle_degrees,
            "n_samples": n_samples,
            "n_features": n_features,
        },
    )


def make_rotated_subspace_signal(
    n_samples: int = 400,
    n_features: int = 10,
    signal_noise_std: float = 0.25,
    feature_noise_std: float = 0.5,
    random_state: Optional[int] = None,
) -> SyntheticDataset:
    if n_features < 2:
        raise ValueError("n_features must be at least 2 for the rotated subspace simulation.")
    rng = _rng(random_state)
    latent = rng.normal(size=(n_samples, 2))
    y = (latent[:, 0] + 0.5 * latent[:, 1] + rng.normal(scale=signal_noise_std, size=n_samples) > 0).astype(int)
    mixing = rng.normal(size=(n_features, 2))
    q, _ = np.linalg.qr(mixing)
    X = latent @ q[:, :2].T + rng.normal(scale=feature_noise_std, size=(n_samples, n_features))
    return _finalize_synthetic_dataset(
        name="rotated_subspace_signal",
        X=X,
        y=y,
        description="Signal lives in a rotated 2D latent subspace embedded in higher dimensions.",
        metadata={
            "signal_noise_std": signal_noise_std,
            "feature_noise_std": feature_noise_std,
            "n_samples": n_samples,
            "n_features": n_features,
        },
    )


def make_additive_interaction_correlated(
    n_samples: int = 400,
    n_features: int = 6,
    correlation: float = 0.7,
    noise_std: float = 0.2,
    random_state: Optional[int] = None,
) -> SyntheticDataset:
    if n_features < 3:
        raise ValueError("n_features must be at least 3 for the additive interaction simulation.")
    rng = _rng(random_state)
    cov = np.full((n_features, n_features), correlation, dtype=np.float64)
    np.fill_diagonal(cov, 1.0)
    X = rng.multivariate_normal(mean=np.zeros(n_features), cov=cov, size=n_samples)
    signal = X[:, 0] * X[:, 1] + 0.3 * X[:, 2] + rng.normal(scale=noise_std, size=n_samples)
    y = (signal > 0).astype(int)
    return _finalize_synthetic_dataset(
        name="additive_interaction_correlated",
        X=X,
        y=y,
        description="Interaction-driven signal with correlated predictors.",
        metadata={
            "correlation": correlation,
            "noise_std": noise_std,
            "n_samples": n_samples,
            "n_features": n_features,
        },
    )


def make_diagonal_band(
    n_samples: int = 400,
    n_features: int = 2,
    band_width: float = 0.45,
    noise_std: float = 0.05,
    random_state: Optional[int] = None,
) -> SyntheticDataset:
    rng = _rng(random_state)
    signal = rng.uniform(-2.0, 2.0, size=(n_samples, 2))
    y = (np.abs(signal[:, 0] - signal[:, 1] + rng.normal(scale=noise_std, size=n_samples)) < band_width).astype(int)
    X = _append_noise_features(rng, signal, n_features, noise_scale=0.75)
    return _finalize_synthetic_dataset(
        name="diagonal_band",
        X=X,
        y=y,
        description="Thin band classification aligned with x1 - x2 rather than the coordinate axes.",
        metadata={
            "band_width": band_width,
            "noise_std": noise_std,
            "n_samples": n_samples,
            "n_features": n_features,
        },
    )


def make_sparse_oblique_hyperplane(
    n_samples: int = 400,
    n_features: int = 12,
    n_informative: int = 4,
    margin_noise_std: float = 0.2,
    random_state: Optional[int] = None,
) -> SyntheticDataset:
    if n_informative > n_features:
        raise ValueError("n_informative must be less than or equal to n_features.")
    rng = _rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    weights = _random_projection(rng, n_features, n_active=n_informative)
    margins = X @ weights + rng.normal(scale=margin_noise_std, size=n_samples)
    y = (margins > 0).astype(int)
    return _finalize_synthetic_dataset(
        name="sparse_oblique_hyperplane",
        X=X,
        y=y,
        description="High-dimensional sparse linear boundary defined by a projection across a few active features.",
        metadata={
            "n_samples": n_samples,
            "n_features": n_features,
            "n_informative": n_informative,
            "margin_noise_std": margin_noise_std,
        },
    )


def make_oblique_slab(
    n_samples: int = 400,
    n_features: int = 8,
    slab_half_width: float = 0.4,
    margin_noise_std: float = 0.05,
    random_state: Optional[int] = None,
) -> SyntheticDataset:
    rng = _rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    weights = _random_projection(rng, n_features, n_active=min(4, n_features))
    signed_distance = X @ weights + rng.normal(scale=margin_noise_std, size=n_samples)
    y = (np.abs(signed_distance) < slab_half_width).astype(int)
    return _finalize_synthetic_dataset(
        name="oblique_slab",
        X=X,
        y=y,
        description="Thin positive slab around an oblique hyperplane, requiring multiple axis-aligned boxes to approximate.",
        metadata={
            "n_samples": n_samples,
            "n_features": n_features,
            "slab_half_width": slab_half_width,
            "margin_noise_std": margin_noise_std,
        },
    )


def make_dual_projection_xor(
    n_samples: int = 400,
    n_features: int = 8,
    projection_noise_std: float = 0.1,
    random_state: Optional[int] = None,
) -> SyntheticDataset:
    rng = _rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    w1 = _random_projection(rng, n_features, n_active=min(4, n_features))
    w2 = _random_projection(rng, n_features, n_active=min(4, n_features))
    proj1 = X @ w1 + rng.normal(scale=projection_noise_std, size=n_samples)
    proj2 = X @ w2 + rng.normal(scale=projection_noise_std, size=n_samples)
    y = np.logical_xor(proj1 > 0, proj2 > 0).astype(int)
    return _finalize_synthetic_dataset(
        name="dual_projection_xor",
        X=X,
        y=y,
        description="XOR classification defined on two different oblique projections in higher dimensions.",
        metadata={
            "n_samples": n_samples,
            "n_features": n_features,
            "projection_noise_std": projection_noise_std,
        },
    )


SIMULATION_DATASET_GENERATORS: Dict[str, Callable[..., SyntheticDataset]] = {
    "rotated_linear_boundary": make_rotated_linear_boundary,
    "rotated_xor": make_rotated_xor,
    "oblique_gaussian_clusters": make_oblique_gaussian_clusters,
    "rotated_subspace_signal": make_rotated_subspace_signal,
    "additive_interaction_correlated": make_additive_interaction_correlated,
    "diagonal_band": make_diagonal_band,
    "sparse_oblique_hyperplane": make_sparse_oblique_hyperplane,
    "oblique_slab": make_oblique_slab,
    "dual_projection_xor": make_dual_projection_xor,
}


def load_simulated_dataset(name: str, **kwargs: Any) -> SyntheticDataset:
    if name not in SIMULATION_DATASET_GENERATORS:
        raise ValueError(
            f"Unknown simulated dataset {name!r}. "
            f"Available datasets: {sorted(SIMULATION_DATASET_GENERATORS)}"
        )
    return SIMULATION_DATASET_GENERATORS[name](**kwargs)


def _require_openml() -> Any:
    try:
        return importlib.import_module("openml")
    except ImportError as exc:
        raise ImportError(
            "OpenML support requires the optional 'openml' package. "
            "Install it with `pip install openml` to use forestgap.datasets."
        ) from exc


def _normalize_target_name(target_name: TargetName) -> TargetName:
    if target_name is None:
        return None
    if isinstance(target_name, str):
        names = [part.strip() for part in target_name.split(",") if part.strip()]
        if not names:
            return None
        return names[0] if len(names) == 1 else names
    names = [str(part).strip() for part in target_name if str(part).strip()]
    if not names:
        return None
    return names[0] if len(names) == 1 else names


def _dataset_identifier(name: Optional[str], dataset_id: Optional[int]) -> str:
    provided = int(name is not None) + int(dataset_id is not None)
    if provided != 1:
        raise ValueError("Provide exactly one of `name` or `dataset_id`.")
    return f"name={name!r}" if name is not None else f"dataset_id={dataset_id!r}"


def _coerce_feature_names(attribute_names: Optional[Sequence[Any]]) -> List[str]:
    if attribute_names is None:
        return []
    return [str(name) for name in attribute_names]


def _split_frame_target(
    frame: Any,
    categorical_indicator: Sequence[bool],
    attribute_names: Sequence[Any],
    target_name: Union[str, Sequence[str]],
    as_frame: bool,
) -> tuple[Any, Any, List[bool], List[str]]:
    target_columns = [target_name] if isinstance(target_name, str) else list(target_name)
    target_columns = [str(column) for column in target_columns]
    missing_targets = [column for column in target_columns if column not in frame.columns]
    if missing_targets:
        raise ValueError(
            "Requested target column(s) not found in dataset frame: "
            f"{missing_targets}. Available columns: {list(frame.columns)}"
        )
    feature_columns = [column for column in frame.columns if column not in target_columns]

    X = frame.loc[:, feature_columns]
    y = frame.loc[:, target_columns]
    if len(target_columns) == 1:
        y = y.iloc[:, 0]

    categorical_by_name = {
        str(column_name): bool(is_categorical)
        for column_name, is_categorical in zip(attribute_names, categorical_indicator)
    }
    X_categorical_indicator = [categorical_by_name.get(column, False) for column in feature_columns]

    if not as_frame:
        X = X.to_numpy()
        y = y.to_numpy()
        if getattr(y, "ndim", 1) == 2 and y.shape[1] == 1:
            y = np.ravel(y)

    return X, y, X_categorical_indicator, feature_columns


def _load_dataset_via_frame_split(
    dataset: Any,
    resolved_target: Union[str, Sequence[str]],
    as_frame: bool,
) -> tuple[Any, Any, List[bool], List[str]]:
    frame, _, categorical_indicator, attribute_names = _get_openml_data(
        dataset,
        target=None,
        dataset_format="dataframe",
    )
    return _split_frame_target(
        frame=frame,
        categorical_indicator=categorical_indicator,
        attribute_names=attribute_names,
        target_name=resolved_target,
        as_frame=as_frame,
    )


def _is_pyarrow_extension_registration_error(exc: BaseException) -> bool:
    """Return whether ``exc`` is the Pandas/PyArrow registration conflict.

    OpenML wraps the original PyArrow exception while reading a Parquet cache,
    so inspect the full exception chain instead of depending on PyArrow being
    importable here.
    """

    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current)
        if (
            "arrow.py_extension_type" in message
            and "No type extension with name" in message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _get_openml_data(
    dataset: Any,
    *,
    target: TargetName,
    dataset_format: str,
) -> tuple[Any, Any, Sequence[bool], Sequence[Any]]:
    """Load OpenML data, retrying its Parquet cache through ARFF if needed."""

    try:
        return dataset.get_data(target=target, dataset_format=dataset_format)
    except Exception as exc:
        if not _is_pyarrow_extension_registration_error(exc):
            raise

        # Pandas can fail before it reads an otherwise valid Parquet file when
        # PyArrow's legacy extension type is not registered. OpenML exposes
        # this escape hatch; retrying via ARFF avoids the optional PyArrow path
        # without constraining the user's installed versions.
        previous_skip_parquet = os.environ.get("OPENML_SKIP_PARQUET")
        os.environ["OPENML_SKIP_PARQUET"] = "true"
        try:
            dataset.parquet_file = None
            return dataset.get_data(target=target, dataset_format=dataset_format)
        finally:
            if previous_skip_parquet is None:
                os.environ.pop("OPENML_SKIP_PARQUET", None)
            else:
                os.environ["OPENML_SKIP_PARQUET"] = previous_skip_parquet


def _resolve_dataset_id_by_name(openml: Any, name: str, version: Optional[int]) -> int:
    datasets = openml.datasets.list_datasets(
        output_format="dataframe",
        data_name=name,
    )
    matches = []
    if hasattr(datasets, "iterrows"):
        for _, dataset_info in datasets.iterrows():
            if dataset_info.get("name") != name:
                continue
            dataset_version = dataset_info.get("version")
            if version is not None and int(dataset_version) != version:
                continue
            matches.append((int(dataset_info["did"]), dataset_info))
    else:
        for dataset_id, dataset_info in datasets.items():
            if dataset_info.get("name") != name:
                continue
            if version is not None and int(dataset_info.get("version", -1)) != version:
                continue
            matches.append((int(dataset_id), dataset_info))

    if not matches:
        version_suffix = f" with version {version}" if version is not None else ""
        raise ValueError(f"Could not find an OpenML dataset named {name!r}{version_suffix}.")

    if version is None:
        matches.sort(key=lambda item: int(item[1].get("version", 0)), reverse=True)

    return matches[0][0]


def load_openml_dataset(
    *,
    name: Optional[str] = None,
    dataset_id: Optional[int] = None,
    version: Optional[int] = None,
    target_name: TargetName = None,
    as_frame: bool = True,
    download_data: bool = True,
) -> OpenMLDataset:
    """
    Load a single OpenML dataset by dataset name or numeric id.

    Parameters
    ----------
    name : str, optional
        OpenML dataset name.
    dataset_id : int, optional
        OpenML dataset id.
    version : int, optional
        Dataset version to resolve when loading by name.
    target_name : str or sequence of str, optional
        Target column override. When omitted, the dataset default target is used.
    as_frame : bool, default=True
        Request a pandas DataFrame/Series payload when possible.
    download_data : bool, default=True
        Forwarded to OpenML to control whether data are downloaded immediately.
    """
    _dataset_identifier(name=name, dataset_id=dataset_id)
    openml = _require_openml()

    resolved_dataset_id = int(dataset_id) if dataset_id is not None else _resolve_dataset_id_by_name(openml, name or "", version)
    dataset = openml.datasets.get_dataset(
        dataset_id=resolved_dataset_id,
        version=version,
        download_data=download_data,
    )

    resolved_target = _normalize_target_name(
        target_name if target_name is not None else getattr(dataset, "default_target_attribute", None)
    )
    if isinstance(resolved_target, list):
        X, y, categorical_indicator, attribute_names = _load_dataset_via_frame_split(
            dataset=dataset,
            resolved_target=resolved_target,
            as_frame=as_frame,
        )
    else:
        dataset_format = "dataframe" if as_frame else "array"
        try:
            X, y, categorical_indicator, attribute_names = _get_openml_data(
                dataset,
                target=resolved_target,
                dataset_format=dataset_format,
            )
        except IndexError:
            if resolved_target is None:
                raise
            X, y, categorical_indicator, attribute_names = _load_dataset_via_frame_split(
                dataset=dataset,
                resolved_target=resolved_target,
                as_frame=as_frame,
            )

    metadata = {
        "dataset_id": getattr(dataset, "dataset_id", resolved_dataset_id),
        "name": getattr(dataset, "name", name),
        "version": getattr(dataset, "version", version),
        "default_target_attribute": getattr(dataset, "default_target_attribute", None),
        "row_id_attribute": getattr(dataset, "row_id_attribute", None),
        "ignore_attribute": getattr(dataset, "ignore_attribute", None),
        "citation": getattr(dataset, "citation", None),
        "creator": getattr(dataset, "creator", None),
        "collection_date": getattr(dataset, "collection_date", None),
        "language": getattr(dataset, "language", None),
        "licence": getattr(dataset, "licence", None),
        "url": getattr(dataset, "url", None),
    }

    return OpenMLDataset(
        name=str(getattr(dataset, "name", name)),
        dataset_id=int(getattr(dataset, "dataset_id", resolved_dataset_id)),
        version=getattr(dataset, "version", version),
        target_name=resolved_target,
        X=X,
        y=y,
        feature_names=_coerce_feature_names(attribute_names),
        categorical_indicator=[bool(flag) for flag in categorical_indicator],
        url=getattr(dataset, "url", None),
        description=getattr(dataset, "description", None),
        metadata=metadata,
    )


def _get_openml_task(openml: Any, task_id: int, *, download_splits: bool) -> Any:
    """Load a task while remaining compatible with lightweight OpenML mocks."""

    get_task = openml.tasks.get_task
    try:
        return get_task(int(task_id), download_splits=download_splits)
    except TypeError:
        # Older bindings and test doubles did not expose ``download_splits``.
        return get_task(int(task_id))


def _task_split_dimensions(task: Any) -> tuple[int, int, int]:
    """Return OpenML ``(repeats, folds, samples)`` dimensions conservatively."""

    get_dimensions = getattr(task, "get_split_dimensions", None)
    if not callable(get_dimensions):
        return (1, 1, 1)
    dimensions = tuple(int(value) for value in get_dimensions())
    if len(dimensions) != 3 or any(value < 1 for value in dimensions):
        raise ValueError(
            "OpenML task split dimensions must be three positive integers; "
            f"received {dimensions!r}."
        )
    return dimensions


def _task_metadata(task: Any, *, task_id: int) -> Dict[str, Any]:
    """Capture stable task fields without depending on a binding version."""

    fields = (
        "dataset_id",
        "target_name",
        "task_type_id",
        "task_type",
        "estimation_procedure",
        "evaluation_measure",
        "class_labels",
    )
    metadata: Dict[str, Any] = {"task_id": int(task_id)}
    for field_name in fields:
        value = getattr(task, field_name, None)
        if value is not None:
            metadata[field_name] = value
    return metadata


def _load_openml_task_from_raw(
    *,
    task_id: int,
    task: Any,
    as_frame: bool,
    download_data: bool,
) -> OpenMLTask:
    dataset_id = getattr(task, "dataset_id", None)
    if dataset_id is None:
        raise ValueError(f"OpenML task {task_id} does not expose a dataset_id.")
    target_name = _normalize_target_name(getattr(task, "target_name", None))
    dataset = load_openml_dataset(
        dataset_id=int(dataset_id),
        target_name=target_name,
        as_frame=as_frame,
        download_data=download_data,
    )
    repeats, folds, samples = _task_split_dimensions(task)
    task_type = getattr(task, "task_type", None)
    return OpenMLTask(
        task_id=int(task_id),
        dataset=dataset,
        target_name=target_name,
        repeats=repeats,
        folds=folds,
        samples=samples,
        task_type=None if task_type is None else str(task_type),
        metadata=_task_metadata(task, task_id=task_id),
    )


def load_openml_task(
    *,
    task_id: int,
    as_frame: bool = True,
    download_data: bool = True,
    download_splits: bool = False,
) -> OpenMLTask:
    """Load one typed OpenML task, including its official split dimensions.

    The data payload is normalized through :func:`load_openml_dataset`, so
    existing suite loaders and task-based studies use exactly the same feature
    and target semantics.  Set ``download_splits=True`` when a subsequent
    caller will immediately materialize a fold.
    """

    openml = _require_openml()
    resolved_task_id = int(task_id)
    task = _get_openml_task(
        openml, resolved_task_id, download_splits=download_splits,
    )
    return _load_openml_task_from_raw(
        task_id=resolved_task_id,
        task=task,
        as_frame=as_frame,
        download_data=download_data,
    )


def _validate_fold_indices(
    *,
    task: OpenMLTask,
    repeat: int,
    fold: int,
    sample: int,
    train_indices: Any,
    test_indices: Any,
) -> tuple[np.ndarray, np.ndarray]:
    repeats, folds, samples = task.repeats, task.folds, task.samples
    values = (repeat, fold, sample)
    limits = (repeats, folds, samples)
    labels = ("repeat", "fold", "sample")
    for label, value, limit in zip(labels, values, limits):
        if value < 0 or value >= limit:
            raise ValueError(
                f"OpenML task {task.task_id} has {label} range [0, {limit}); "
                f"received {value}."
            )

    train = np.asarray(train_indices, dtype=np.int64).ravel()
    test = np.asarray(test_indices, dtype=np.int64).ravel()
    if train.size == 0 or test.size == 0:
        raise ValueError(
            f"OpenML task {task.task_id} fold r{repeat}-f{fold}-s{sample} "
            "must contain non-empty train and test indices."
        )
    if np.unique(train).size != train.size or np.unique(test).size != test.size:
        raise ValueError("OpenML train/test indices must be unique within each partition.")
    if np.intersect1d(train, test).size:
        raise ValueError("OpenML train and test indices must be disjoint.")
    n_samples = len(task.dataset.X)
    if train.min() < 0 or test.min() < 0 or train.max() >= n_samples or test.max() >= n_samples:
        raise ValueError(
            f"OpenML task {task.task_id} fold indices must lie in [0, {n_samples})."
        )
    return train, test


def _load_openml_task_fold_from_raw(
    *,
    task_id: int,
    raw_task: Any,
    task: OpenMLTask,
    repeat: int,
    fold: int,
    sample: int,
) -> OpenMLTaskFold:
    get_indices = getattr(raw_task, "get_train_test_split_indices", None)
    if not callable(get_indices):
        raise ValueError(
            f"OpenML task {task_id} does not expose official train/test split indices."
        )
    train_indices, test_indices = get_indices(
        repeat=int(repeat), fold=int(fold), sample=int(sample),
    )
    train, test = _validate_fold_indices(
        task=task,
        repeat=int(repeat),
        fold=int(fold),
        sample=int(sample),
        train_indices=train_indices,
        test_indices=test_indices,
    )
    return OpenMLTaskFold(
        task=task,
        repeat=int(repeat),
        fold=int(fold),
        sample=int(sample),
        train_indices=train,
        test_indices=test,
    )


def load_openml_task_fold(
    *,
    task_id: int,
    fold: int = 0,
    repeat: int = 0,
    sample: int = 0,
    as_frame: bool = True,
    download_data: bool = True,
) -> OpenMLTaskFold:
    """Load one official, zero-based OpenML outer fold for a predictive task."""

    openml = _require_openml()
    resolved_task_id = int(task_id)
    raw_task = _get_openml_task(openml, resolved_task_id, download_splits=True)
    task = _load_openml_task_from_raw(
        task_id=resolved_task_id,
        task=raw_task,
        as_frame=as_frame,
        download_data=download_data,
    )
    return _load_openml_task_fold_from_raw(
        task_id=resolved_task_id,
        raw_task=raw_task,
        task=task,
        repeat=repeat,
        fold=fold,
        sample=sample,
    )


def iter_openml_task_folds(
    *,
    task_id: int,
    as_frame: bool = True,
    download_data: bool = True,
) -> Iterator[OpenMLTaskFold]:
    """Yield all official folds of a task in repeat/fold/sample order."""

    openml = _require_openml()
    resolved_task_id = int(task_id)
    raw_task = _get_openml_task(openml, resolved_task_id, download_splits=True)
    task = _load_openml_task_from_raw(
        task_id=resolved_task_id,
        task=raw_task,
        as_frame=as_frame,
        download_data=download_data,
    )
    for repeat in range(task.repeats):
        for fold in range(task.folds):
            for sample in range(task.samples):
                yield _load_openml_task_fold_from_raw(
                    task_id=resolved_task_id,
                    raw_task=raw_task,
                    task=task,
                    repeat=repeat,
                    fold=fold,
                    sample=sample,
                )


# ``split`` is a useful synonym in callers that mirror the OpenML binding's
# ``get_train_test_split_indices`` method.  Keep both public spellings.
OpenMLTaskSplit = OpenMLTaskFold
load_openml_task_split = load_openml_task_fold
iter_openml_task_splits = iter_openml_task_folds


def _resolve_suite_id_by_name(openml: Any, suite_name: str) -> int:
    query = str(suite_name).strip().casefold()
    suites = openml.study.list_suites(output_format="dataframe")
    candidates: list[tuple[int, str, str]] = []

    if hasattr(suites, "iterrows"):
        for _, suite_info in suites.iterrows():
            suite_id = int(suite_info.get("id"))
            name = str(suite_info.get("name", "") or "")
            alias = str(suite_info.get("alias", "") or "")
            candidates.append((suite_id, name, alias))
    else:
        for suite_id, suite_info in suites.items():
            name = str(suite_info.get("name", "") or "")
            alias = str(suite_info.get("alias", "") or "")
            candidates.append((int(suite_id), name, alias))

    for suite_id, name, alias in candidates:
        if query in {name.strip().casefold(), alias.strip().casefold()}:
            return suite_id

    prefix_matches = [
        suite_id
        for suite_id, name, alias in candidates
        if name.strip().casefold().startswith(query)
        or alias.strip().casefold().startswith(query)
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    raise ValueError(f"Could not find an OpenML benchmark suite named {suite_name!r}.")


def load_openml_suite(
    *,
    suite_id: Optional[int] = None,
    suite_name: Optional[str] = None,
    as_frame: bool = True,
    download_data: bool = True,
) -> OpenMLBenchmarkSuite:
    """
    Load every task-backed dataset in an OpenML benchmark suite.

    Parameters
    ----------
    suite_id : int, optional
        OpenML study/suite id.
    suite_name : str, optional
        OpenML suite name. Used only when `suite_id` is not supplied.
    as_frame : bool, default=True
        Request DataFrame/Series payloads when possible.
    download_data : bool, default=True
        Forwarded to OpenML dataset loading.
    """
    provided = int(suite_id is not None) + int(suite_name is not None)
    if provided != 1:
        raise ValueError("Provide exactly one of `suite_id` or `suite_name`.")

    openml = _require_openml()
    resolved_suite_id = int(suite_id) if suite_id is not None else _resolve_suite_id_by_name(openml, suite_name or "")
    suite = openml.study.get_suite(resolved_suite_id)

    datasets: List[OpenMLDataset] = []
    task_ids = [int(task_id) for task_id in getattr(suite, "tasks", [])]
    for task_id in task_ids:
        task = openml.tasks.get_task(task_id)
        datasets.append(
            load_openml_dataset(
                dataset_id=int(task.dataset_id),
                target_name=getattr(task, "target_name", None),
                as_frame=as_frame,
                download_data=download_data,
            )
        )

    metadata = {
        "suite_id": resolved_suite_id,
        "name": getattr(suite, "name", suite_name),
        "alias": getattr(suite, "alias", None),
        "main_entity_type": getattr(suite, "main_entity_type", None),
    }

    return OpenMLBenchmarkSuite(
        name=str(getattr(suite, "name", suite_name)),
        suite_id=resolved_suite_id,
        task_ids=task_ids,
        datasets=datasets,
        description=getattr(suite, "description", None),
        metadata=metadata,
    )


__all__ = [
    "OpenMLBenchmarkSuite",
    "OpenMLDataset",
    "OpenMLTask",
    "OpenMLTaskFold",
    "OpenMLTaskSplit",
    "SyntheticDataset",
    "SIMULATION_DATASET_GENERATORS",
    "load_openml_dataset",
    "load_openml_suite",
    "load_openml_task",
    "load_openml_task_fold",
    "load_openml_task_split",
    "iter_openml_task_folds",
    "iter_openml_task_splits",
    "load_simulated_dataset",
    "make_rotated_linear_boundary",
    "make_rotated_xor",
    "make_oblique_gaussian_clusters",
    "make_rotated_subspace_signal",
    "make_additive_interaction_correlated",
    "make_diagonal_band",
    "make_sparse_oblique_hyperplane",
    "make_oblique_slab",
    "make_dual_projection_xor",
]
