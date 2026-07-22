"""Fold-aware aggregation and paired inference for experiment manifests."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare


DEFAULT_DATASET_COLUMNS = ("suite_id", "task_id", "dataset_id")


def aggregate_folds_within_dataset(
    rows: pd.DataFrame,
    *,
    metric: str,
    dataset_columns: Sequence[str] = DEFAULT_DATASET_COLUMNS,
    comparison_columns: Sequence[str] = ("estimator", "relation", "application"),
) -> pd.DataFrame:
    """Average only successful outer folds before comparing datasets.

    The returned frame contains one row per dataset and comparison.  It never
    pools raw folds across datasets, preventing large datasets or ten-fold
    tasks from dominating a cross-dataset comparison.
    """

    required = [*dataset_columns, *comparison_columns, metric]
    missing = [column for column in required if column not in rows]
    if missing:
        raise KeyError(f"Rows lack required aggregation columns: {missing}")
    frame = rows.copy()
    if "status" in frame:
        frame = frame[frame["status"] == "ok"].copy()
    frame = frame[np.isfinite(pd.to_numeric(frame[metric], errors="coerce"))].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    grouping = [*dataset_columns, *comparison_columns]
    return (
        frame.groupby(grouping, dropna=False, as_index=False)[metric]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": metric, "count": "n_outer_folds"})
    )


def paired_effects(
    aggregated: pd.DataFrame,
    *,
    metric: str,
    baseline: str,
    comparison: str,
    method_column: str = "method",
    dataset_columns: Sequence[str] = DEFAULT_DATASET_COLUMNS,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Return within-dataset paired improvements over a named baseline."""

    required = [*dataset_columns, method_column, metric]
    missing = [column for column in required if column not in aggregated]
    if missing:
        raise KeyError(f"Aggregated rows lack required paired columns: {missing}")
    baseline_rows = aggregated[aggregated[method_column] == baseline]
    comparison_rows = aggregated[aggregated[method_column] == comparison]
    paired = comparison_rows.merge(
        baseline_rows,
        on=list(dataset_columns),
        suffixes=("_comparison", "_baseline"),
        how="inner",
        validate="one_to_one",
    )
    difference = paired[f"{metric}_comparison"] - paired[f"{metric}_baseline"]
    paired["effect"] = difference if higher_is_better else -difference
    return paired


def bootstrap_confidence_interval(
    values: Iterable[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return mean and a deterministic percentile bootstrap interval."""

    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("At least one finite paired effect is required.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, data.size, size=(n_resamples, data.size))
    estimates = data[draw_indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(data.mean()),
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def average_ranks(
    aggregated: pd.DataFrame,
    *,
    metric: str,
    method_column: str = "method",
    dataset_columns: Sequence[str] = DEFAULT_DATASET_COLUMNS,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Compute dataset-wise ranks, then average them across paired datasets."""

    required = [*dataset_columns, method_column, metric]
    missing = [column for column in required if column not in aggregated]
    if missing:
        raise KeyError(f"Aggregated rows lack required rank columns: {missing}")
    frame = aggregated.copy()
    frame["rank"] = frame.groupby(list(dataset_columns), dropna=False)[metric].rank(
        ascending=not higher_is_better,
        method="average",
    )
    return (
        frame.groupby(method_column, as_index=False)["rank"]
        .mean()
        .rename(columns={"rank": "average_rank"})
        .sort_values("average_rank", kind="stable")
        .reset_index(drop=True)
    )


def friedman_omnibus(
    aggregated: pd.DataFrame,
    *,
    metric: str,
    method_column: str = "method",
    dataset_columns: Sequence[str] = DEFAULT_DATASET_COLUMNS,
) -> dict[str, float | int]:
    """Run a paired Friedman omnibus test on complete dataset/method blocks."""

    index_columns = list(dataset_columns)
    matrix = aggregated.pivot_table(
        index=index_columns,
        columns=method_column,
        values=metric,
        aggfunc="first",
    ).dropna(axis=0, how="any")
    if matrix.shape[0] < 2 or matrix.shape[1] < 3:
        raise ValueError("Friedman testing requires at least two datasets and three complete methods.")
    statistic, p_value = friedmanchisquare(*[matrix[column] for column in matrix.columns])
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_datasets": int(matrix.shape[0]),
        "n_methods": int(matrix.shape[1]),
    }


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Return Holm-Bonferroni adjusted p-values in original input order."""

    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence.")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite values in [0, 1].")
    order = np.argsort(values)
    sorted_values = values[order]
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, sorted_values * (values.size - np.arange(values.size))),
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


__all__ = [
    "DEFAULT_DATASET_COLUMNS",
    "aggregate_folds_within_dataset",
    "average_ranks",
    "bootstrap_confidence_interval",
    "friedman_omnibus",
    "holm_adjust",
    "paired_effects",
]
