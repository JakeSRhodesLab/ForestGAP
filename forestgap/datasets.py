from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Any, Dict, List, Optional, Sequence, Union

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
        frame, _, categorical_indicator, attribute_names = dataset.get_data(
            target=None,
            dataset_format="dataframe",
        )
        X, y, categorical_indicator, attribute_names = _split_frame_target(
            frame=frame,
            categorical_indicator=categorical_indicator,
            attribute_names=attribute_names,
            target_name=resolved_target,
            as_frame=as_frame,
        )
    else:
        dataset_format = "dataframe" if as_frame else "array"
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=resolved_target,
            dataset_format=dataset_format,
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


def _resolve_suite_id_by_name(openml: Any, suite_name: str) -> int:
    suites = openml.study.list_suites(output_format="dict")
    for suite_id, suite_info in suites.items():
        if suite_info.get("name") == suite_name:
            return int(suite_id)
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
    "load_openml_dataset",
    "load_openml_suite",
]
