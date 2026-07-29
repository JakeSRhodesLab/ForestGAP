from __future__ import annotations

import sys as _sys
import site as _site

if not any(argument in {"-h", "--help"} for argument in _sys.argv[1:]):
    _user_site = _site.getusersitepackages()
    _sys.path = [path for path in _sys.path if not path.startswith(_user_site)]
    del _user_site
del _sys, _site

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re
import sys
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from forestgap import get_prototypes
from experiments.common import (
    PreparedDataset,
    add_common_arguments,
    aggregate_success_rows,
    available_model_names,
    build_proximity_model,
    compiled_output_dir,
    estimator_supports_parameter,
    iter_openml_suite,
    parse_random_states,
    prepare_dataset,
    proximity_methods_for_model,
    log_progress,
    raw_output_path,
    record_exception,
    requested_model_names,
    requested_tasks,
    rough_fill_train_test,
    save_dataframe,
    save_plot_dataframe,
    suite_metadata,
    unsupported_model_row,
)


EXPERIMENT_NAME = "prototypes"
SYNTHETIC_SUITE_NAME = "synthetic"


@dataclass(frozen=True)
class SyntheticScenario:
    prepared: PreparedDataset
    train_mode_labels: np.ndarray
    test_mode_labels: np.ndarray
    mode_centers: dict[str, np.ndarray]
    mode_to_class: dict[str, int]


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def _stable_seed(*parts: Any) -> int:
    value = 17
    for part in parts:
        for byte in str(part).encode("utf-8"):
            value = (value * 131 + int(byte)) % (2**32 - 5)
    return int(value)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _percentile_rank(background: np.ndarray, value: float) -> float:
    if background.size == 0 or not np.isfinite(value):
        return float("nan")
    clean = background[np.isfinite(background)]
    if clean.size == 0:
        return float("nan")
    return float(np.mean(clean <= value))


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return float("nan")
    return float(len(left & right) / len(union))


def _matrix_scalar(matrix: Any, row_idx: int, col_idx: int) -> float:
    value = matrix[row_idx, col_idx]
    if sparse.issparse(value):
        return float(value.toarray()[0, 0])
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return float(array.reshape(-1)[0])


def _symmetric_proximity(matrix: Any, row_idx: int, col_idx: int) -> float:
    if row_idx == col_idx:
        return _matrix_scalar(matrix, row_idx, col_idx)
    return float(
        0.5
        * (
            _matrix_scalar(matrix, row_idx, col_idx)
            + _matrix_scalar(matrix, col_idx, row_idx)
        )
    )


def _subset_to_dense(matrix: Any, columns: list[int]) -> np.ndarray:
    if not columns:
        return np.zeros((matrix.shape[0], 0), dtype=np.float64)
    subset = matrix[:, columns]
    if sparse.issparse(subset):
        return subset.toarray()
    return np.asarray(subset, dtype=np.float64)


def _labelled_plot_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model_prox"] = out["model_name"].astype(str) + "\n" + out["prox_method"].astype(str)
    return out


def _save_suite_metric_plot(
    df: pd.DataFrame,
    *,
    suite_name: str,
    metric: str,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    suite_df = df[df["suite_name"] == suite_name].copy()
    if suite_df.empty or metric not in suite_df.columns:
        return
    suite_df = _labelled_plot_frame(suite_df)
    save_plot_dataframe(
        suite_df.sort_values(metric, ascending=False),
        x="model_prox",
        y=metric,
        title=title,
        ylabel=ylabel,
        path=path,
        rotation=0,
    )


def _mode_specs() -> list[dict[str, Any]]:
    return [
        {
            "dataset_id": -101,
            "task_id": -101,
            "dataset_name": "synthetic_multimodal_balanced",
            "clusters": [
                {"class_label": 0, "mode_label": "c0_m0", "center": (-4.0, -4.0, 0.0, 0.0), "count": 90, "scale": 0.65},
                {"class_label": 0, "mode_label": "c0_m1", "center": (-4.0, 4.0, 0.0, 0.0), "count": 90, "scale": 0.65},
                {"class_label": 1, "mode_label": "c1_m0", "center": (4.0, -4.0, 0.0, 0.0), "count": 90, "scale": 0.65},
                {"class_label": 1, "mode_label": "c1_m1", "center": (4.0, 4.0, 0.0, 0.0), "count": 90, "scale": 0.65},
                {"class_label": 2, "mode_label": "c2_m0", "center": (0.0, 0.0, 4.0, 4.0), "count": 140, "scale": 0.75},
            ],
        },
        {
            "dataset_id": -102,
            "task_id": -102,
            "dataset_name": "synthetic_overlap_imbalance",
            "clusters": [
                {"class_label": 0, "mode_label": "c0_m0", "center": (-2.7, -1.3, 0.0, 0.0), "count": 130, "scale": 0.95},
                {"class_label": 0, "mode_label": "c0_m1", "center": (-0.8, 2.2, 0.0, 0.0), "count": 70, "scale": 0.85},
                {"class_label": 1, "mode_label": "c1_m0", "center": (1.0, -0.2, 0.4, 0.0), "count": 80, "scale": 1.05},
                {"class_label": 1, "mode_label": "c1_m1", "center": (3.0, 2.0, 0.4, 0.0), "count": 55, "scale": 0.95},
                {"class_label": 2, "mode_label": "c2_m0", "center": (0.2, 4.4, 3.2, 0.0), "count": 36, "scale": 0.70},
            ],
        },
    ]


def _build_synthetic_scenarios(
    *,
    random_state: int,
    test_size: float,
) -> list[SyntheticScenario]:
    feature_names = [f"x{i + 1}" for i in range(4)]
    scenarios: list[SyntheticScenario] = []

    for spec in _mode_specs():
        rng = np.random.RandomState(_stable_seed(spec["dataset_name"], random_state))
        blocks: list[np.ndarray] = []
        labels: list[int] = []
        modes: list[str] = []
        mode_centers: dict[str, np.ndarray] = {}
        mode_to_class: dict[str, int] = {}

        for cluster in spec["clusters"]:
            center = np.asarray(cluster["center"], dtype=np.float64)
            mode_centers[str(cluster["mode_label"])] = center
            mode_to_class[str(cluster["mode_label"])] = int(cluster["class_label"])
            block = rng.normal(
                loc=center,
                scale=float(cluster["scale"]),
                size=(int(cluster["count"]), center.shape[0]),
            )
            blocks.append(block)
            labels.extend([int(cluster["class_label"])] * block.shape[0])
            modes.extend([str(cluster["mode_label"])] * block.shape[0])

        X = np.vstack(blocks)
        y = np.asarray(labels, dtype=np.int64)
        mode_labels = np.asarray(modes, dtype=object)
        stratify_labels = np.asarray([f"{cls}|{mode}" for cls, mode in zip(y, mode_labels)], dtype=object)

        (
            X_train,
            X_test,
            y_train,
            y_test,
            train_modes,
            test_modes,
        ) = train_test_split(
            X,
            y,
            mode_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )

        prepared = PreparedDataset(
            task_kind="classification",
            suite_id=-1,
            suite_name=SYNTHETIC_SUITE_NAME,
            task_id=int(spec["task_id"]),
            dataset_id=int(spec["dataset_id"]),
            dataset_name=str(spec["dataset_name"]),
            feature_names=feature_names,
            categorical_indices=[],
            X_train=np.asarray(X_train, dtype=np.float64),
            X_test=np.asarray(X_test, dtype=np.float64),
            y_train=np.asarray(y_train, dtype=np.int64),
            y_test=np.asarray(y_test, dtype=np.int64),
            original_n_samples=int(X.shape[0]),
            original_n_features=int(X.shape[1]),
            train_size=int(X_train.shape[0]),
            test_size=int(X_test.shape[0]),
        )
        scenarios.append(
            SyntheticScenario(
                prepared=prepared,
                train_mode_labels=np.asarray(train_modes, dtype=object),
                test_mode_labels=np.asarray(test_modes, dtype=object),
                mode_centers=mode_centers,
                mode_to_class=mode_to_class,
            )
        )

    return scenarios


def _random_same_class_baseline(
    *,
    model: Any,
    proximity_matrix: Any,
    y_train: np.ndarray,
    k: int,
    n_prototypes: int,
    rng: np.random.RandomState,
) -> dict[int, list[dict[str, Any]]]:
    baseline: dict[int, list[dict[str, Any]]] = {}
    labels = np.asarray(y_train)
    for cls in np.unique(labels):
        available_mask = np.ones(labels.shape[0], dtype=bool)
        records: list[dict[str, Any]] = []
        for prototype_number in range(1, int(n_prototypes) + 1):
            candidates = np.flatnonzero((labels == cls) & available_mask)
            if candidates.size == 0:
                break
            center_position = int(rng.choice(candidates))
            neighbor_positions, neighbor_values = model._prototype_neighbor_set(  # type: ignore[attr-defined]
                proximity_matrix,
                center_position,
                available_mask,
                int(k),
            )
            if neighbor_positions.size == 0:
                break
            same_class_neighbor_count = int((labels[neighbor_positions] == cls).sum())
            records.append(
                {
                    "prototype_number": prototype_number,
                    "class_label": int(cls),
                    "center_position": center_position,
                    "center_index": center_position,
                    "neighbor_positions": np.asarray(neighbor_positions, dtype=np.int64),
                    "neighbor_index": np.asarray(neighbor_positions, dtype=np.int64),
                    "neighbor_proximities": np.asarray(neighbor_values, dtype=np.float64),
                    "same_class_neighbor_count": same_class_neighbor_count,
                    "same_class_neighbor_fraction": float(
                        same_class_neighbor_count / max(neighbor_positions.size, 1)
                    ),
                }
            )
            available_mask[np.asarray(neighbor_positions, dtype=np.int64)] = False
        baseline[int(cls)] = records
    return baseline


def _prototype_center_metrics(
    *,
    prox_test: Any,
    y_test: np.ndarray,
    center_positions: list[int],
    center_labels: list[int],
    classes: np.ndarray,
) -> dict[str, float]:
    if not center_positions:
        raise ValueError("Prototype extraction returned no centers.")

    center_scores = _subset_to_dense(prox_test, center_positions)
    score_sum = np.full((center_scores.shape[0], classes.size), -np.inf, dtype=np.float64)
    score_max = np.full((center_scores.shape[0], classes.size), -np.inf, dtype=np.float64)

    for class_idx, cls in enumerate(classes):
        class_centers = [idx for idx, label in enumerate(center_labels) if label == int(cls)]
        if not class_centers:
            continue
        class_values = center_scores[:, class_centers]
        score_sum[:, class_idx] = class_values.sum(axis=1)
        score_max[:, class_idx] = class_values.max(axis=1)

    pred_sum = classes[np.argmax(score_sum, axis=1)]
    pred_max = classes[np.argmax(score_max, axis=1)]
    nearest_center_labels = np.asarray(center_labels, dtype=np.int64)[np.argmax(center_scores, axis=1)]

    test_nearest_center_accuracy = float(accuracy_score(y_test, nearest_center_labels))
    class_balanced_test_nearest_center_accuracy = _safe_mean(
        [
            float(accuracy_score(y_test[y_test == cls], nearest_center_labels[y_test == cls]))
            for cls in classes
            if np.any(y_test == cls)
        ]
    )
    return {
        "prototype_accuracy": float(accuracy_score(y_test, pred_sum)),
        "prototype_accuracy_max": float(accuracy_score(y_test, pred_max)),
        "test_nearest_center_accuracy": test_nearest_center_accuracy,
        "class_balanced_test_nearest_center_accuracy": class_balanced_test_nearest_center_accuracy,
    }


def _summarize_prototype_run(
    *,
    prepared: PreparedDataset,
    model: Any,
    model_name: str,
    prox_method: str,
    random_state: int,
    k: int,
    n_prototypes: int,
    prototypes: dict[int, list[dict[str, Any]]],
    prototype_source: str,
    baseline_repeat: Optional[int],
    forest_accuracy: float,
    fit_runtime_seconds: float,
    prototype_runtime_seconds: float,
    projection_runtime_seconds: float,
    X_train_filled: np.ndarray,
    prox_train: Any,
    prox_test: Any,
    synthetic: Optional[SyntheticScenario],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluation_start = time.perf_counter()
    y_train = np.asarray(prepared.y_train, dtype=np.int64)
    y_test = np.asarray(prepared.y_test, dtype=np.int64)
    classes = np.asarray(sorted(np.unique(y_train)), dtype=np.int64)
    predict_proba = np.asarray(model.predict_proba(X_train_filled), dtype=np.float64)
    oob_prob = getattr(model, "oob_decision_function_", None)
    oob_status_default = "unavailable"
    if oob_prob is not None:
        oob_prob = np.asarray(oob_prob, dtype=np.float64)
        if oob_prob.shape != predict_proba.shape:
            oob_prob = None
            oob_status_default = "shape_mismatch"
        elif not np.isfinite(oob_prob).all():
            oob_prob = None
            oob_status_default = "non_finite"
        elif np.nanmin(oob_prob) < -1e-8 or np.nanmax(oob_prob) > 1.0 + 1e-8:
            oob_prob = None
            oob_status_default = "invalid_probability_range"
        else:
            oob_status_default = "available"

    clean_prototypes = {
        int(cls): [record for record in records]
        for cls, records in prototypes.items()
    }
    center_positions_by_class = {
        int(cls): [int(record["center_position"]) for record in records]
        for cls, records in clean_prototypes.items()
    }

    detail_rows: list[dict[str, Any]] = []
    all_centers: list[int] = []
    all_center_labels: list[int] = []
    same_class_fractions: list[float] = []
    same_class_counts: list[float] = []
    class_mean_purity: list[float] = []
    class_mean_counts: list[float] = []
    covered_neighbors: set[int] = set()
    total_neighbor_assignments = 0
    class_coverage_values: list[float] = []
    pairwise_overlaps: list[float] = []
    pairwise_center_proximities: list[float] = []
    predict_values: list[float] = []
    oob_values: list[float] = []
    predict_percentiles_within_class: list[float] = []
    predict_percentiles_vs_oob: list[float] = []
    oob_percentiles_vs_oob: list[float] = []
    oob_available_count = 0
    mode_coverage_values: list[float] = []
    same_mode_neighbor_fractions: list[float] = []
    center_to_mode_distances: list[float] = []
    distinct_center_mode_fractions: list[float] = []

    mode_labels = synthetic.train_mode_labels if synthetic is not None else None

    for cls in classes:
        cls = int(cls)
        records = clean_prototypes.get(cls, [])
        class_centers = center_positions_by_class.get(cls, [])
        if not records:
            class_coverage_values.append(0.0)
            continue

        class_positions = np.flatnonzero(y_train == cls)
        same_class_probs = predict_proba[class_positions, cls]
        nonprototype_idx = np.asarray(
            [idx for idx in class_positions.tolist() if idx not in set(class_centers)],
            dtype=np.int64,
        )
        nonprototype_oob = (
            None
            if oob_prob is None
            else np.asarray(oob_prob[nonprototype_idx, cls], dtype=np.float64)
        )

        class_neighbor_sets = [
            set(np.asarray(record["neighbor_positions"], dtype=np.int64).tolist())
            for record in records
        ]
        class_pairwise_overlaps: list[float] = []
        class_pairwise_center_proximities: list[float] = []
        per_record_overlap_stats: list[tuple[float, float, float, float]] = []

        for record_idx, record in enumerate(records):
            overlaps_for_record: list[float] = []
            center_prox_for_record: list[float] = []
            for other_idx, other_record in enumerate(records):
                if record_idx == other_idx:
                    continue
                overlap = _jaccard(class_neighbor_sets[record_idx], class_neighbor_sets[other_idx])
                if np.isfinite(overlap):
                    overlaps_for_record.append(overlap)
                    class_pairwise_overlaps.append(overlap)
                center_proximity = _symmetric_proximity(
                    prox_train,
                    int(record["center_position"]),
                    int(other_record["center_position"]),
                )
                if np.isfinite(center_proximity):
                    center_prox_for_record.append(center_proximity)
                    class_pairwise_center_proximities.append(center_proximity)
            per_record_overlap_stats.append(
                (
                    _safe_mean(overlaps_for_record),
                    float(np.max(overlaps_for_record)) if overlaps_for_record else float("nan"),
                    _safe_mean(center_prox_for_record),
                    float(np.max(center_prox_for_record)) if center_prox_for_record else float("nan"),
                )
            )

        pairwise_overlaps.extend(class_pairwise_overlaps)
        pairwise_center_proximities.extend(class_pairwise_center_proximities)

        class_covered_same_class = set()
        class_purities: list[float] = []
        class_counts_local: list[float] = []
        class_selected_modes: set[str] = set()

        for record, overlap_stats in zip(records, per_record_overlap_stats):
            center_position = int(record["center_position"])
            neighbor_positions = np.asarray(record["neighbor_positions"], dtype=np.int64)
            neighbor_count = int(neighbor_positions.size)
            same_class_neighbor_count = int(record["same_class_neighbor_count"])
            same_class_neighbor_fraction = float(record["same_class_neighbor_fraction"])
            class_purities.append(same_class_neighbor_fraction)
            class_counts_local.append(float(same_class_neighbor_count))
            all_centers.append(center_position)
            all_center_labels.append(cls)
            same_class_fractions.append(same_class_neighbor_fraction)
            same_class_counts.append(float(same_class_neighbor_count))
            covered_neighbors.update(neighbor_positions.tolist())
            total_neighbor_assignments += neighbor_count
            class_covered_same_class.update(int(pos) for pos in neighbor_positions if y_train[pos] == cls)

            prototype_predict = float(predict_proba[center_position, cls])
            same_class_background = np.asarray(
                [
                    same_class_probs[pos_idx]
                    for pos_idx, pos in enumerate(class_positions.tolist())
                    if pos != center_position
                ],
                dtype=np.float64,
            )
            prototype_oob = float("nan")
            predict_pct_vs_nonprototype_oob = float("nan")
            oob_pct_vs_nonprototype_oob = float("nan")
            oob_status = oob_status_default
            if oob_prob is not None:
                prototype_oob = float(oob_prob[center_position, cls])
                oob_status = "available"
                oob_values.append(prototype_oob)
                oob_available_count += 1
                if nonprototype_oob is not None:
                    predict_pct_vs_nonprototype_oob = _percentile_rank(nonprototype_oob, prototype_predict)
                    oob_pct_vs_nonprototype_oob = _percentile_rank(nonprototype_oob, prototype_oob)
                    if np.isfinite(predict_pct_vs_nonprototype_oob):
                        predict_percentiles_vs_oob.append(predict_pct_vs_nonprototype_oob)
                    if np.isfinite(oob_pct_vs_nonprototype_oob):
                        oob_percentiles_vs_oob.append(oob_pct_vs_nonprototype_oob)

            predict_values.append(prototype_predict)
            predict_pct_within_class = _percentile_rank(same_class_background, prototype_predict)
            if np.isfinite(predict_pct_within_class):
                predict_percentiles_within_class.append(predict_pct_within_class)

            detail_row = {
                "experiment": EXPERIMENT_NAME,
                "task_kind": prepared.task_kind,
                "suite_name": prepared.suite_name,
                "model_name": model_name,
                "prox_method": prox_method,
                "prototype_source": prototype_source,
                "baseline_repeat": baseline_repeat,
                "dataset_name": prepared.dataset_name,
                "dataset_id": prepared.dataset_id,
                "task_id": prepared.task_id,
                "random_state": random_state,
                "k": int(k),
                "n_prototypes": int(n_prototypes),
                "class_label": cls,
                "prototype_number": int(record["prototype_number"]),
                "center_position": center_position,
                "center_index": int(record.get("center_index", center_position)),
                "neighbor_count": neighbor_count,
                "same_class_neighbor_count": same_class_neighbor_count,
                "same_class_neighbor_fraction": same_class_neighbor_fraction,
                "mean_neighbor_overlap_within_class": overlap_stats[0],
                "max_neighbor_overlap_within_class": overlap_stats[1],
                "mean_center_proximity_within_class": overlap_stats[2],
                "max_center_proximity_within_class": overlap_stats[3],
                "prototype_center_predict_proba": prototype_predict,
                "prototype_center_oob_proba": prototype_oob,
                "predict_percentile_within_class": predict_pct_within_class,
                "predict_percentile_vs_nonprototype_oob": predict_pct_vs_nonprototype_oob,
                "oob_percentile_vs_nonprototype_oob": oob_pct_vs_nonprototype_oob,
                "oob_probability_status": oob_status,
            }

            if synthetic is not None and mode_labels is not None:
                center_mode = str(mode_labels[center_position])
                class_selected_modes.add(center_mode)
                same_mode_fraction = float(np.mean(mode_labels[neighbor_positions] == center_mode))
                center_to_mode_distance = float(
                    np.linalg.norm(
                        np.asarray(prepared.X_train[center_position], dtype=np.float64)
                        - np.asarray(synthetic.mode_centers[center_mode], dtype=np.float64)
                    )
                )
                same_mode_neighbor_fractions.append(same_mode_fraction)
                center_to_mode_distances.append(center_to_mode_distance)
                detail_row.update(
                    {
                        "center_mode_label": center_mode,
                        "same_mode_neighbor_fraction": same_mode_fraction,
                        "center_to_mode_distance": center_to_mode_distance,
                    }
                )

            detail_rows.append(detail_row)

        class_mean_purity.append(_safe_mean(class_purities))
        class_mean_counts.append(_safe_mean(class_counts_local))
        class_coverage_values.append(float(len(class_covered_same_class) / max(class_positions.size, 1)))

        if synthetic is not None:
            known_modes = [
                mode_label
                for mode_label, mode_class in synthetic.mode_to_class.items()
                if int(mode_class) == cls
            ]
            mode_coverage_values.append(float(len(class_selected_modes) / max(len(known_modes), 1)))
            distinct_center_mode_fractions.append(float(len(class_selected_modes) / max(len(records), 1)))

    predictive_metrics = _prototype_center_metrics(
        prox_test=prox_test,
        y_test=y_test,
        center_positions=all_centers,
        center_labels=all_center_labels,
        classes=classes,
    )
    evaluation_runtime_seconds = time.perf_counter() - evaluation_start

    summary_row = {
        "experiment": EXPERIMENT_NAME,
        "status": "ok",
        "task_kind": prepared.task_kind,
        "suite_name": prepared.suite_name,
        "dataset_id": prepared.dataset_id,
        "dataset_name": prepared.dataset_name,
        "task_id": prepared.task_id,
        "random_state": random_state,
        "model_name": model_name,
        "prox_method": prox_method,
        "prototype_source": prototype_source,
        "baseline_repeat": baseline_repeat,
        "k": int(k),
        "n_prototypes": int(n_prototypes),
        "n_centers": int(len(all_centers)),
        "train_size": int(prepared.train_size),
        "test_size": int(prepared.test_size),
        "forest_accuracy": float(forest_accuracy),
        "mean_same_class_neighbor_fraction": _safe_mean(same_class_fractions),
        "mean_same_class_neighbor_count": _safe_mean(same_class_counts),
        "class_balanced_same_class_neighbor_fraction": _safe_mean(class_mean_purity),
        "class_balanced_same_class_neighbor_count": _safe_mean(class_mean_counts),
        "coverage_fraction": float(len(covered_neighbors) / max(prepared.train_size, 1)),
        "class_balanced_coverage_fraction": _safe_mean(class_coverage_values),
        "mean_within_class_neighbor_overlap_jaccard": _safe_mean(pairwise_overlaps),
        "mean_within_class_center_proximity": _safe_mean(pairwise_center_proximities),
        "unique_covered_neighbor_fraction": float(
            len(covered_neighbors) / max(total_neighbor_assignments, 1)
        ),
        "prototype_accuracy": predictive_metrics["prototype_accuracy"],
        "prototype_accuracy_max": predictive_metrics["prototype_accuracy_max"],
        "prototype_accuracy_gap_vs_forest": float(
            forest_accuracy - predictive_metrics["prototype_accuracy"]
        ),
        "prototype_accuracy_max_gap_vs_forest": float(
            forest_accuracy - predictive_metrics["prototype_accuracy_max"]
        ),
        "test_nearest_center_accuracy": predictive_metrics["test_nearest_center_accuracy"],
        "class_balanced_test_nearest_center_accuracy": predictive_metrics[
            "class_balanced_test_nearest_center_accuracy"
        ],
        "mean_focal_class_predict_proba": _safe_mean(predict_values),
        "mean_focal_class_oob_proba": _safe_mean(oob_values),
        "mean_predict_percentile_within_class": _safe_mean(predict_percentiles_within_class),
        "mean_predict_percentile_vs_nonprototype_oob": _safe_mean(predict_percentiles_vs_oob),
        "mean_oob_percentile_vs_nonprototype_oob": _safe_mean(oob_percentiles_vs_oob),
        "oob_probability_available_fraction": float(
            oob_available_count / max(len(all_centers), 1)
        ),
        "mode_coverage_fraction": _safe_mean(mode_coverage_values),
        "mean_same_mode_neighbor_fraction": _safe_mean(same_mode_neighbor_fractions),
        "mean_center_to_mode_distance": _safe_mean(center_to_mode_distances),
        "distinct_center_mode_fraction": _safe_mean(distinct_center_mode_fractions),
        "fit_runtime_seconds": float(fit_runtime_seconds),
        "prototype_runtime_seconds": float(prototype_runtime_seconds),
        "projection_runtime_seconds": float(projection_runtime_seconds),
        "evaluation_runtime_seconds": float(evaluation_runtime_seconds),
        "total_runtime_seconds": float(
            fit_runtime_seconds
            + prototype_runtime_seconds
            + projection_runtime_seconds
            + evaluation_runtime_seconds
        ),
    }
    return summary_row, detail_rows


def _merge_stability_metrics(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    group_cols = [
        "experiment",
        "task_kind",
        "suite_name",
        "dataset_id",
        "dataset_name",
        "task_id",
        "model_name",
        "prox_method",
        "prototype_source",
        "k",
        "n_prototypes",
    ]
    success = summary_df[summary_df["status"] == "ok"].copy()
    if success.empty:
        return summary_df

    per_seed = (
        success.groupby(group_cols + ["random_state"], dropna=False)[
            [
                "mean_same_class_neighbor_fraction",
                "coverage_fraction",
                "prototype_accuracy",
                "mean_within_class_center_proximity",
                "mode_coverage_fraction",
            ]
        ]
        .mean()
        .reset_index()
    )
    stability = (
        per_seed.groupby(group_cols, dropna=False)
        .agg(
            stability_n_random_states=("random_state", "nunique"),
            stability_mean_same_class_neighbor_fraction_std=(
                "mean_same_class_neighbor_fraction",
                "std",
            ),
            stability_coverage_fraction_std=("coverage_fraction", "std"),
            stability_prototype_accuracy_std=("prototype_accuracy", "std"),
            stability_center_proximity_std=("mean_within_class_center_proximity", "std"),
            stability_mode_coverage_fraction_std=("mode_coverage_fraction", "std"),
        )
        .reset_index()
    )

    if not detail_df.empty and "center_mode_label" in detail_df.columns:
        synthetic_detail = detail_df[
            detail_df["suite_name"] == SYNTHETIC_SUITE_NAME
        ].copy()
        if not synthetic_detail.empty:
            synthetic_group_cols = [
                "experiment",
                "task_kind",
                "suite_name",
                "dataset_id",
                "dataset_name",
                "task_id",
                "model_name",
                "prox_method",
                "prototype_source",
                "k",
                "n_prototypes",
            ]
            pair_rows: list[dict[str, Any]] = []
            for group_key, group_df in synthetic_detail.groupby(synthetic_group_cols, dropna=False):
                runs: list[tuple[str, pd.DataFrame]] = []
                for run_key, run_df in group_df.groupby(["random_state", "baseline_repeat"], dropna=False):
                    run_id = f"{run_key[0]}::{run_key[1]}"
                    runs.append((run_id, run_df))
                for (left_id, left_df), (right_id, right_df) in combinations(runs, 2):
                    for cls in sorted(
                        set(left_df["class_label"].dropna().tolist())
                        | set(right_df["class_label"].dropna().tolist())
                    ):
                        left_modes = set(
                            left_df[left_df["class_label"] == cls]["center_mode_label"].dropna().astype(str).tolist()
                        )
                        right_modes = set(
                            right_df[right_df["class_label"] == cls]["center_mode_label"].dropna().astype(str).tolist()
                        )
                        score = _jaccard(left_modes, right_modes)
                        if np.isfinite(score):
                            pair_rows.append(
                                {
                                    **dict(zip(synthetic_group_cols, group_key)),
                                    "stability_center_mode_jaccard": score,
                                }
                            )
            if pair_rows:
                stability_modes = (
                    pd.DataFrame(pair_rows)
                    .groupby(synthetic_group_cols, dropna=False)["stability_center_mode_jaccard"]
                    .mean()
                    .reset_index()
                )
                stability = stability.merge(
                    stability_modes,
                    on=synthetic_group_cols,
                    how="left",
                )

    return summary_df.merge(stability, on=group_cols, how="left")


def _evaluate_case(
    *,
    prepared: PreparedDataset,
    synthetic: Optional[SyntheticScenario],
    model_name: str,
    prox_method: str,
    random_state: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    X_train_filled, X_test_filled = rough_fill_train_test(
        prepared.X_train,
        prepared.X_test,
        prepared.categorical_indices,
    )

    fit_start = time.perf_counter()
    model = build_proximity_model(
        prepared.task_kind,
        model_name=model_name,
        prox_method=prox_method,
        random_state=random_state,
        n_estimators=args.n_estimators,
        bootstrap=True,
        oob_score=estimator_supports_parameter(prepared.task_kind, model_name, "oob_score"),
    )
    model.fit(X_train_filled, prepared.y_train)
    fit_runtime_seconds = time.perf_counter() - fit_start

    forest_pred = np.asarray(model.predict(X_test_filled), dtype=np.int64)
    forest_accuracy = float(accuracy_score(prepared.y_test, forest_pred))

    prototype_start = time.perf_counter()
    prototypes = get_prototypes(
        model,
        k=args.k,
        n_prototypes=args.n_prototypes,
        categorical_features=prepared.categorical_indices,
    )
    learned_prototype_runtime = time.perf_counter() - prototype_start

    projection_start = time.perf_counter()
    prox_train = model.get_proximities()
    prox_test = model.prox_extend(X_test_filled)
    projection_runtime_seconds = time.perf_counter() - projection_start

    learned_row, learned_details = _summarize_prototype_run(
        prepared=prepared,
        model=model,
        model_name=model_name,
        prox_method=prox_method,
        random_state=random_state,
        k=args.k,
        n_prototypes=args.n_prototypes,
        prototypes=prototypes,
        prototype_source="learned",
        baseline_repeat=None,
        forest_accuracy=forest_accuracy,
        fit_runtime_seconds=fit_runtime_seconds,
        prototype_runtime_seconds=learned_prototype_runtime,
        projection_runtime_seconds=projection_runtime_seconds,
        X_train_filled=X_train_filled,
        prox_train=prox_train,
        prox_test=prox_test,
        synthetic=synthetic,
    )
    rows.append(learned_row)
    detail_rows.extend(learned_details)

    for baseline_repeat in range(int(args.baseline_repeats)):
        rng = np.random.RandomState(
            _stable_seed(
                prepared.dataset_name,
                model_name,
                prox_method,
                random_state,
                baseline_repeat,
                "random_same_class_center",
            )
        )
        baseline_start = time.perf_counter()
        baseline = _random_same_class_baseline(
            model=model,
            proximity_matrix=prox_train,
            y_train=np.asarray(prepared.y_train, dtype=np.int64),
            k=args.k,
            n_prototypes=args.n_prototypes,
            rng=rng,
        )
        baseline_prototype_runtime = time.perf_counter() - baseline_start
        baseline_row, baseline_details = _summarize_prototype_run(
            prepared=prepared,
            model=model,
            model_name=model_name,
            prox_method=prox_method,
            random_state=random_state,
            k=args.k,
            n_prototypes=args.n_prototypes,
            prototypes=baseline,
            prototype_source="random_same_class_center",
            baseline_repeat=baseline_repeat,
            forest_accuracy=forest_accuracy,
            fit_runtime_seconds=fit_runtime_seconds,
            prototype_runtime_seconds=baseline_prototype_runtime,
            projection_runtime_seconds=projection_runtime_seconds,
            X_train_filled=X_train_filled,
            prox_train=prox_train,
            prox_test=prox_test,
            synthetic=synthetic,
        )
        rows.append(baseline_row)
        detail_rows.extend(baseline_details)

    return rows, detail_rows


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    random_states = parse_random_states(args.random_states)

    for task_kind in requested_tasks(args.task):
        log_progress(f"[{EXPERIMENT_NAME}] task_kind={task_kind}")
        if task_kind == "classification" and not args.skip_synthetic:
            log_progress(f"[{EXPERIMENT_NAME}] starting synthetic track")
            for random_state in random_states:
                for synthetic in _build_synthetic_scenarios(
                    random_state=random_state,
                    test_size=args.test_size,
                ):
                    prepared = synthetic.prepared
                    log_progress(
                        f"[{EXPERIMENT_NAME}] synthetic dataset={prepared.dataset_name} "
                        f"random_state={random_state}"
                    )
                    for model_name in requested_model_names():
                        if model_name not in available_model_names(task_kind):
                            for prox_method in proximity_methods_for_model(model_name):
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
                                        extra_fields={
                                            "suite_name": prepared.suite_name,
                                            "prox_method": prox_method,
                                            "prototype_source": "learned",
                                            "k": int(args.k),
                                            "n_prototypes": int(args.n_prototypes),
                                        },
                                    )
                                )
                            continue

                        for prox_method in proximity_methods_for_model(model_name):
                            log_progress(
                                f"[{EXPERIMENT_NAME}] synthetic dataset={prepared.dataset_name} "
                                f"model={model_name} prox_method={prox_method}"
                            )
                            try:
                                case_rows, case_details = _evaluate_case(
                                    prepared=prepared,
                                    synthetic=synthetic,
                                    model_name=model_name,
                                    prox_method=prox_method,
                                    random_state=random_state,
                                    args=args,
                                )
                                rows.extend(case_rows)
                                detail_rows.extend(case_details)
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
                                            "suite_name": prepared.suite_name,
                                            "model_name": model_name,
                                            "prox_method": prox_method,
                                            "prototype_source": "learned",
                                            "k": int(args.k),
                                            "n_prototypes": int(args.n_prototypes),
                                        },
                                    )
                                )

        if args.skip_openml:
            continue

        log_progress(f"[{EXPERIMENT_NAME}] starting OpenML track")
        for task_id, task in iter_openml_suite(task_kind, args.dataset_limit):
            dataset_name = str(getattr(task, "dataset_name", f"task_{task_id}"))
            dataset_id = int(getattr(task, "dataset_id", -1))
            for random_state in random_states:
                log_progress(
                    f"[{EXPERIMENT_NAME}] task_kind={task_kind} dataset={dataset_name} "
                    f"task_id={task_id} random_state={random_state}"
                )
                if task_kind == "regression":
                    for model_name in requested_model_names():
                        for prox_method in proximity_methods_for_model(model_name):
                            rows.append(
                                unsupported_model_row(
                                    experiment_name=EXPERIMENT_NAME,
                                    task_kind=task_kind,
                                    model_name=model_name,
                                    dataset_name=dataset_name,
                                    dataset_id=dataset_id,
                                    task_id=task_id,
                                    random_state=random_state,
                                    reason="the standalone get_prototypes application is classification-only",
                                    extra_fields={
                                        "suite_name": suite_metadata(task_kind)[1],
                                        "prox_method": prox_method,
                                        "prototype_source": "learned",
                                        "k": int(args.k),
                                        "n_prototypes": int(args.n_prototypes),
                                    },
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
                except Exception as exc:
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
                                "suite_name": "OpenML-CC18",
                            },
                        )
                    )
                    continue

                for model_name in requested_model_names():
                    if model_name not in available_model_names(task_kind):
                        for prox_method in proximity_methods_for_model(model_name):
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
                                    extra_fields={
                                        "suite_name": prepared.suite_name,
                                        "prox_method": prox_method,
                                        "prototype_source": "learned",
                                        "k": int(args.k),
                                        "n_prototypes": int(args.n_prototypes),
                                    },
                                )
                            )
                        continue

                    for prox_method in proximity_methods_for_model(model_name):
                        log_progress(
                            f"[{EXPERIMENT_NAME}] dataset={prepared.dataset_name} "
                            f"model={model_name} prox_method={prox_method}"
                        )
                        try:
                            case_rows, case_details = _evaluate_case(
                                prepared=prepared,
                                synthetic=None,
                                model_name=model_name,
                                prox_method=prox_method,
                                random_state=random_state,
                                args=args,
                            )
                            rows.extend(case_rows)
                            detail_rows.extend(case_details)
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
                                        "suite_name": prepared.suite_name,
                                        "model_name": model_name,
                                        "prox_method": prox_method,
                                        "prototype_source": "learned",
                                        "k": int(args.k),
                                        "n_prototypes": int(args.n_prototypes),
                                    },
                                )
                            )

    summary_df = pd.DataFrame(rows)
    detail_df = pd.DataFrame(detail_rows)
    summary_df = _merge_stability_metrics(summary_df, detail_df)
    return summary_df, detail_df


def compile_outputs(summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> None:
    if summary_df.empty or "task_kind" not in summary_df.columns:
        return

    for task_kind in sorted(summary_df["task_kind"].dropna().unique()):
        task_df = summary_df[summary_df["task_kind"] == task_kind].copy()
        save_dataframe(task_df, raw_output_path(EXPERIMENT_NAME, task_kind))

        task_detail_df = (
            detail_df[detail_df["task_kind"] == task_kind].copy()
            if not detail_df.empty and "task_kind" in detail_df.columns
            else pd.DataFrame()
        )
        if not task_detail_df.empty:
            detail_path = raw_output_path(EXPERIMENT_NAME, task_kind).with_name(f"{task_kind}_details.csv")
            save_dataframe(task_detail_df, detail_path)

        out_dir = compiled_output_dir(EXPERIMENT_NAME, task_kind)
        base_group_cols = ("suite_name", "prototype_source", "model_name", "prox_method", "k", "n_prototypes")
        summary_metric_cols = (
            "forest_accuracy",
            "prototype_accuracy",
            "prototype_accuracy_max",
            "prototype_accuracy_gap_vs_forest",
            "prototype_accuracy_max_gap_vs_forest",
            "test_nearest_center_accuracy",
            "class_balanced_test_nearest_center_accuracy",
            "mean_same_class_neighbor_fraction",
            "mean_same_class_neighbor_count",
            "class_balanced_same_class_neighbor_fraction",
            "coverage_fraction",
            "class_balanced_coverage_fraction",
            "mean_within_class_neighbor_overlap_jaccard",
            "mean_within_class_center_proximity",
            "unique_covered_neighbor_fraction",
            "mean_focal_class_predict_proba",
            "mean_focal_class_oob_proba",
            "mean_predict_percentile_within_class",
            "mean_predict_percentile_vs_nonprototype_oob",
            "mean_oob_percentile_vs_nonprototype_oob",
            "oob_probability_available_fraction",
            "mode_coverage_fraction",
            "mean_same_mode_neighbor_fraction",
            "mean_center_to_mode_distance",
            "distinct_center_mode_fraction",
            "stability_mean_same_class_neighbor_fraction_std",
            "stability_coverage_fraction_std",
            "stability_prototype_accuracy_std",
            "stability_center_proximity_std",
            "stability_mode_coverage_fraction_std",
            "stability_center_mode_jaccard",
            "fit_runtime_seconds",
            "prototype_runtime_seconds",
            "projection_runtime_seconds",
            "evaluation_runtime_seconds",
            "total_runtime_seconds",
        )
        summary = aggregate_success_rows(
            task_df,
            group_cols=base_group_cols,
            metric_cols=summary_metric_cols,
        )
        save_dataframe(summary, out_dir / "summary.csv")

        learned_df = task_df[task_df["prototype_source"] == "learned"].copy()
        model_comparison = aggregate_success_rows(
            learned_df,
            group_cols=("suite_name", "model_name", "prox_method", "k", "n_prototypes"),
            metric_cols=summary_metric_cols,
        )
        save_dataframe(model_comparison, out_dir / "model_comparison.csv")

        purity_summary = aggregate_success_rows(
            task_df,
            group_cols=base_group_cols,
            metric_cols=(
                "mean_same_class_neighbor_fraction",
                "mean_same_class_neighbor_count",
                "class_balanced_same_class_neighbor_fraction",
                "stability_mean_same_class_neighbor_fraction_std",
            ),
        )
        save_dataframe(purity_summary, out_dir / "purity_summary.csv")

        coverage_summary = aggregate_success_rows(
            task_df,
            group_cols=base_group_cols,
            metric_cols=(
                "coverage_fraction",
                "class_balanced_coverage_fraction",
                "test_nearest_center_accuracy",
                "class_balanced_test_nearest_center_accuracy",
                "stability_coverage_fraction_std",
            ),
        )
        save_dataframe(coverage_summary, out_dir / "coverage_summary.csv")

        redundancy_summary = aggregate_success_rows(
            task_df,
            group_cols=base_group_cols,
            metric_cols=(
                "mean_within_class_neighbor_overlap_jaccard",
                "mean_within_class_center_proximity",
                "unique_covered_neighbor_fraction",
                "stability_center_proximity_std",
            ),
        )
        save_dataframe(redundancy_summary, out_dir / "redundancy_summary.csv")

        predictive_summary = aggregate_success_rows(
            task_df,
            group_cols=base_group_cols,
            metric_cols=(
                "forest_accuracy",
                "prototype_accuracy",
                "prototype_accuracy_max",
                "prototype_accuracy_gap_vs_forest",
                "prototype_accuracy_max_gap_vs_forest",
                "stability_prototype_accuracy_std",
            ),
        )
        save_dataframe(predictive_summary, out_dir / "predictive_utility_summary.csv")

        representativeness_summary = aggregate_success_rows(
            task_df,
            group_cols=base_group_cols,
            metric_cols=(
                "mean_focal_class_predict_proba",
                "mean_focal_class_oob_proba",
                "mean_predict_percentile_within_class",
                "mean_predict_percentile_vs_nonprototype_oob",
                "mean_oob_percentile_vs_nonprototype_oob",
                "oob_probability_available_fraction",
            ),
        )
        save_dataframe(representativeness_summary, out_dir / "representativeness_summary.csv")

        if "mode_coverage_fraction" in task_df.columns:
            synthetic_mode_summary = aggregate_success_rows(
                task_df[task_df["suite_name"] == SYNTHETIC_SUITE_NAME].copy(),
                group_cols=base_group_cols,
                metric_cols=(
                    "mode_coverage_fraction",
                    "mean_same_mode_neighbor_fraction",
                    "mean_center_to_mode_distance",
                    "distinct_center_mode_fraction",
                    "stability_mode_coverage_fraction_std",
                    "stability_center_mode_jaccard",
                ),
            )
            save_dataframe(synthetic_mode_summary, out_dir / "synthetic_mode_summary.csv")

        success = task_df[task_df["status"] == "ok"].copy()
        if success.empty:
            continue

        baseline_group_cols = [
            "suite_name",
            "dataset_id",
            "dataset_name",
            "task_id",
            "random_state",
            "model_name",
            "prox_method",
            "k",
            "n_prototypes",
        ]
        learned = success[success["prototype_source"] == "learned"].copy()
        random_baseline = (
            success[success["prototype_source"] == "random_same_class_center"]
            .groupby(baseline_group_cols, dropna=False)[
                [
                    "prototype_accuracy",
                    "coverage_fraction",
                    "mean_same_class_neighbor_fraction",
                    "mean_within_class_center_proximity",
                    "mode_coverage_fraction",
                ]
            ]
            .mean()
            .reset_index()
        )
        baseline_comparison = learned.merge(
            random_baseline,
            on=baseline_group_cols,
            suffixes=("", "_random"),
            how="inner",
        )
        if not baseline_comparison.empty:
            baseline_comparison["prototype_accuracy_improvement_vs_random"] = (
                baseline_comparison["prototype_accuracy"] - baseline_comparison["prototype_accuracy_random"]
            )
            baseline_comparison["coverage_fraction_improvement_vs_random"] = (
                baseline_comparison["coverage_fraction"] - baseline_comparison["coverage_fraction_random"]
            )
            baseline_comparison["purity_improvement_vs_random"] = (
                baseline_comparison["mean_same_class_neighbor_fraction"]
                - baseline_comparison["mean_same_class_neighbor_fraction_random"]
            )
            baseline_comparison["center_proximity_reduction_vs_random"] = (
                baseline_comparison["mean_within_class_center_proximity_random"]
                - baseline_comparison["mean_within_class_center_proximity"]
            )
            baseline_comparison["mode_coverage_improvement_vs_random"] = (
                baseline_comparison["mode_coverage_fraction"] - baseline_comparison["mode_coverage_fraction_random"]
            )
            baseline_summary = aggregate_success_rows(
                baseline_comparison.assign(status="ok"),
                group_cols=("suite_name", "model_name", "prox_method", "k", "n_prototypes"),
                metric_cols=(
                    "prototype_accuracy_improvement_vs_random",
                    "coverage_fraction_improvement_vs_random",
                    "purity_improvement_vs_random",
                    "center_proximity_reduction_vs_random",
                    "mode_coverage_improvement_vs_random",
                ),
            )
        else:
            baseline_summary = pd.DataFrame()
        save_dataframe(baseline_summary, out_dir / "baseline_comparison.csv")

        for suite_name in sorted(model_comparison["suite_name"].dropna().unique()):
            suite_slug = _safe_slug(str(suite_name))
            _save_suite_metric_plot(
                model_comparison,
                suite_name=str(suite_name),
                metric="prototype_accuracy",
                title=f"{suite_name}: Prototype Accuracy",
                ylabel="Accuracy",
                path=out_dir / f"{suite_slug}_prototype_accuracy.png",
            )
            _save_suite_metric_plot(
                model_comparison,
                suite_name=str(suite_name),
                metric="coverage_fraction",
                title=f"{suite_name}: Prototype Coverage",
                ylabel="Coverage Fraction",
                path=out_dir / f"{suite_slug}_coverage.png",
            )
            _save_suite_metric_plot(
                model_comparison,
                suite_name=str(suite_name),
                metric="mean_same_class_neighbor_fraction",
                title=f"{suite_name}: Prototype Purity",
                ylabel="Mean Same-Class Neighbor Fraction",
                path=out_dir / f"{suite_slug}_mean_same_class_neighbor_fraction.png",
            )
            _save_suite_metric_plot(
                baseline_summary,
                suite_name=str(suite_name),
                metric="prototype_accuracy_improvement_vs_random",
                title=f"{suite_name}: Accuracy Improvement vs Random Centers",
                ylabel="Accuracy Improvement",
                path=out_dir / f"{suite_slug}_baseline_improvement_over_random.png",
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--n-estimators", type=int, default=128)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--n-prototypes", type=int, default=3)
    parser.add_argument("--baseline-repeats", type=int, default=8)
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip the synthetic validation track.",
    )
    parser.add_argument(
        "--skip-openml",
        action="store_true",
        help="Skip the OpenML benchmark track.",
    )
    return parser


if __name__ == "__main__":
    summary_frame, detail_frame = run(build_parser().parse_args())
    compile_outputs(summary_frame, detail_frame)
