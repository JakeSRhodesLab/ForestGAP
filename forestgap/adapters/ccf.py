"""Native, sklearn-compatible Canonical Correlation Forest estimators.

The implementation follows the defining CCF construction: every internal
node forms supervised canonical directions between a random subset of the
features and the (expanded) response, then chooses the best impurity-reducing
threshold in that projected space.  Trees are not bagged by default; local
projection bootstrapping supplies the diversity described by Rainforth and
Wood.  ``bootstrap=True`` is the CCF-Bag variant. It persists each tree's
sampled-row multiplicities so OOB and RF-GAP relations have a concrete,
ordinary bootstrap contract, while the default CCF remains non-bagged.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json
from typing import Any, Literal

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted

try:
    from sklearn.utils.validation import validate_data
except ImportError:
    # sklearn < 1.6: validate_data is not a standalone function;
    # fall back to the estimator's _validate_data method.
    def validate_data(estimator, **kwargs):
        return estimator._validate_data(**kwargs)


@dataclass(frozen=True)
class CCFConformance:
    """Result of comparing a fitted estimator with an oracle fixture."""

    passed: bool
    max_absolute_error: float
    tolerance: float
    n_predictions: int


def load_ccf_oracle_fixture(name: str = "two_point_v1") -> dict[str, Any]:
    """Load a packaged deterministic fixture derived from the MATLAB oracle."""

    resource = files("forestgap.adapters").joinpath("fixtures", f"ccf_{name}.json")
    with resource.open("r", encoding="utf-8") as stream:
        return json.load(stream)


@dataclass
class _Node:
    feature_indices: np.ndarray | None
    coefficients: np.ndarray | None
    threshold: float
    left: int
    right: int
    value: np.ndarray
    impurity_gain: float
    n_samples: int

    @property
    def is_leaf(self) -> bool:
        return self.left < 0


class _CanonicalCorrelationTree:
    """Small internal tree with local canonical-correlation splits."""

    def __init__(
        self,
        *,
        task: Literal["classification", "regression"],
        n_outputs: int,
        max_features: str | int | float | None,
        max_depth: int | None,
        min_samples_split: int,
        min_samples_leaf: int,
        projection_bootstrap: bool,
        regularization: float,
        random_state: int,
    ) -> None:
        self.task = task
        self.n_outputs = n_outputs
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.projection_bootstrap = projection_bootstrap
        self.regularization = regularization
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_CanonicalCorrelationTree":
        self.n_features_in_ = X.shape[1]
        self.nodes_: list[_Node] = []
        self._rng = check_random_state(self.random_state)
        self._feature_gain = np.zeros(self.n_features_in_, dtype=float)
        self._grow(X, y, np.arange(X.shape[0]), depth=0)
        total_gain = self._feature_gain.sum()
        self.feature_importances_ = (
            self._feature_gain / total_gain if total_gain > 0 else self._feature_gain
        )
        return self

    def _leaf_value(self, y: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            counts = np.bincount(y.astype(int), minlength=self.n_outputs).astype(float)
            return counts / counts.sum()
        return np.mean(y, axis=0, dtype=float)

    def _resolved_max_features(self) -> int:
        value = self.max_features
        p = self.n_features_in_
        if value is None or value == "all":
            return p
        if value == "sqrt":
            return max(1, int(np.ceil(np.sqrt(p))))
        if value == "log2":
            return max(1, int(np.ceil(np.log2(p) + 1)))
        if isinstance(value, (int, np.integer)):
            return min(p, max(1, int(value)))
        if isinstance(value, (float, np.floating)):
            if not 0 < float(value) <= 1:
                raise ValueError("float max_features must be in (0, 1]")
            return min(p, max(1, int(np.ceil(float(value) * p))))
        raise ValueError("max_features must be None, 'all', 'sqrt', 'log2', int, or float")

    def _response_matrix(self, y: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            return np.eye(self.n_outputs, dtype=float)[y.astype(int)]
        return np.asarray(y, dtype=float).reshape(len(y), -1)

    def _canonical_directions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return regularized CCA directions in feature coordinates."""

        Y = self._response_matrix(y)
        Xc = X - X.mean(axis=0, keepdims=True)
        Yc = Y - Y.mean(axis=0, keepdims=True)
        if self.projection_bootstrap and len(Xc) > 2:
            rows = self._rng.randint(0, len(Xc), size=len(Xc))
            Xc_fit, Yc_fit = Xc[rows], Yc[rows]
        else:
            Xc_fit, Yc_fit = Xc, Yc

        if not np.any(np.ptp(Xc_fit, axis=0) > 1e-12) or not np.any(
            np.ptp(Yc_fit, axis=0) > 1e-12
        ):
            return np.empty((X.shape[1], 0), dtype=float)

        denom = max(1, len(Xc_fit) - 1)
        sxx = Xc_fit.T @ Xc_fit / denom
        syy = Yc_fit.T @ Yc_fit / denom
        sxy = Xc_fit.T @ Yc_fit / denom

        def inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
            values, vectors = np.linalg.eigh(matrix)
            floor = self.regularization * max(1.0, float(np.max(np.abs(values))))
            keep = values > floor
            if not np.any(keep):
                return np.zeros_like(matrix)
            return (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].T

        wx = inverse_sqrt(sxx)
        wy = inverse_sqrt(syy)
        whitened = wx @ sxy @ wy
        if not np.all(np.isfinite(whitened)) or not np.any(np.abs(whitened) > 1e-14):
            return np.empty((X.shape[1], 0), dtype=float)
        left, singular_values, _ = np.linalg.svd(whitened, full_matrices=False)
        rank = int(np.sum(singular_values > self.regularization))
        directions = wx @ left[:, :rank]
        norms = np.linalg.norm(directions, axis=0)
        return directions[:, norms > 0] / norms[norms > 0]

    def _best_classification_cut(
        self, z: np.ndarray, y: np.ndarray
    ) -> tuple[float, float] | None:
        order = np.argsort(z, kind="mergesort")
        z_sorted, y_sorted = z[order], y[order].astype(int)
        candidates = np.flatnonzero(np.diff(z_sorted) > 1e-12)
        candidates = candidates[
            (candidates + 1 >= self.min_samples_leaf)
            & (len(y) - candidates - 1 >= self.min_samples_leaf)
        ]
        if candidates.size == 0:
            return None
        one_hot = np.eye(self.n_outputs, dtype=float)[y_sorted]
        cumulative = np.cumsum(one_hot, axis=0)
        total = cumulative[-1]
        parent = 1.0 - np.sum((total / len(y)) ** 2)
        best_gain, best_index = -np.inf, -1
        for index in candidates:
            n_left = index + 1
            n_right = len(y) - n_left
            left = cumulative[index]
            right = total - left
            impurity = (
                n_left * (1.0 - np.sum((left / n_left) ** 2))
                + n_right * (1.0 - np.sum((right / n_right) ** 2))
            ) / len(y)
            gain = parent - impurity
            if gain > best_gain:
                best_gain, best_index = float(gain), int(index)
        threshold = float((z_sorted[best_index] + z_sorted[best_index + 1]) / 2.0)
        return threshold, best_gain

    def _best_regression_cut(
        self, z: np.ndarray, y: np.ndarray
    ) -> tuple[float, float] | None:
        order = np.argsort(z, kind="mergesort")
        z_sorted = z[order]
        y_sorted = np.asarray(y[order], dtype=float).reshape(len(y), -1)
        candidates = np.flatnonzero(np.diff(z_sorted) > 1e-12)
        candidates = candidates[
            (candidates + 1 >= self.min_samples_leaf)
            & (len(y) - candidates - 1 >= self.min_samples_leaf)
        ]
        if candidates.size == 0:
            return None
        cumulative = np.cumsum(y_sorted, axis=0)
        cumulative_sq = np.cumsum(y_sorted**2, axis=0)
        total, total_sq = cumulative[-1], cumulative_sq[-1]
        parent_sse = float(np.sum(total_sq - total**2 / len(y)))
        best_gain, best_index = -np.inf, -1
        for index in candidates:
            n_left = index + 1
            n_right = len(y) - n_left
            left, left_sq = cumulative[index], cumulative_sq[index]
            right, right_sq = total - left, total_sq - left_sq
            sse = np.sum(left_sq - left**2 / n_left) + np.sum(
                right_sq - right**2 / n_right
            )
            gain = (parent_sse - float(sse)) / max(1.0, parent_sse)
            if gain > best_gain:
                best_gain, best_index = gain, int(index)
        threshold = float((z_sorted[best_index] + z_sorted[best_index + 1]) / 2.0)
        return threshold, float(best_gain)

    def _grow(
        self, X: np.ndarray, y: np.ndarray, sample_ids: np.ndarray, depth: int
    ) -> int:
        node_id = len(self.nodes_)
        values = self._leaf_value(y)
        self.nodes_.append(
            _Node(None, None, np.nan, -1, -1, values, 0.0, len(y))
        )
        pure = (
            np.unique(y).size == 1
            if self.task == "classification"
            else bool(np.all(np.ptp(np.asarray(y).reshape(len(y), -1), axis=0) <= 1e-12))
        )
        stop = (
            pure
            or len(y) < self.min_samples_split
            or len(y) < 2 * self.min_samples_leaf
            or (self.max_depth is not None and depth >= self.max_depth)
        )
        if stop:
            return node_id

        n_select = self._resolved_max_features()
        features = np.sort(
            self._rng.choice(self.n_features_in_, size=n_select, replace=False)
        )
        X_selected = X[:, features]
        directions = self._canonical_directions(X_selected, y)
        if directions.shape[1] == 0:
            variable = np.flatnonzero(np.ptp(X_selected, axis=0) > 1e-12)
            if variable.size == 0:
                return node_id
            directions = np.eye(n_select, dtype=float)[:, variable]

        best: tuple[float, float, np.ndarray] | None = None
        for direction in directions.T:
            z = X_selected @ direction
            cut = (
                self._best_classification_cut(z, y)
                if self.task == "classification"
                else self._best_regression_cut(z, y)
            )
            if cut is not None and (best is None or cut[1] > best[1]):
                best = (cut[0], cut[1], direction.copy())
        if best is None or best[1] <= 1e-14:
            return node_id

        threshold, gain, direction = best
        left_mask = X_selected @ direction <= threshold
        if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
            return node_id
        left = self._grow(X[left_mask], y[left_mask], sample_ids[left_mask], depth + 1)
        right = self._grow(X[~left_mask], y[~left_mask], sample_ids[~left_mask], depth + 1)
        self.nodes_[node_id] = _Node(
            features, direction, threshold, left, right, values, gain, len(y)
        )
        self._feature_gain[features] += (
            gain * len(y) * np.abs(direction) / max(np.abs(direction).sum(), 1e-15)
        )
        return node_id

    def _traverse(self, row: np.ndarray, *, path: bool = False) -> int | list[int]:
        node_id = 0
        visited = [0]
        while not self.nodes_[node_id].is_leaf:
            node = self.nodes_[node_id]
            projection = float(row[node.feature_indices] @ node.coefficients)
            node_id = node.left if projection <= node.threshold else node.right
            visited.append(node_id)
        return visited if path else node_id

    def apply(self, X: Any) -> np.ndarray:
        X_checked = check_array(X, ensure_2d=True, dtype=float)
        return np.asarray([self._traverse(row) for row in X_checked], dtype=np.int64)

    def decision_path(self, X: Any) -> sparse.csr_matrix:
        X_checked = check_array(X, ensure_2d=True, dtype=float)
        rows: list[int] = []
        cols: list[int] = []
        for row_id, row in enumerate(X_checked):
            visited = self._traverse(row, path=True)
            rows.extend([row_id] * len(visited))
            cols.extend(visited)
        return sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(X_checked), len(self.nodes_)),
        )

    @property
    def node_depths_(self) -> np.ndarray:
        """Depth of every node in this tree's stable, integer node layout."""

        depths = np.zeros(len(self.nodes_), dtype=np.int32)
        stack = [0]
        while stack:
            node_id = stack.pop()
            node = self.nodes_[node_id]
            if not node.is_leaf:
                depths[node.left] = depths[node_id] + 1
                depths[node.right] = depths[node_id] + 1
                stack.extend((node.left, node.right))
        return depths

    def predict_values(self, X: Any) -> np.ndarray:
        leaves = self.apply(X)
        return np.vstack([self.nodes_[int(leaf)].value for leaf in leaves])


class _CanonicalCorrelationForestBase(BaseEstimator):
    """Shared estimator implementation; public classes add sklearn task mixins."""

    _task: Literal["classification", "regression"]

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        max_features: str | int | float | None = "log2",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        projection_bootstrap: bool = True,
        bootstrap: bool = False,
        oob_score: bool = False,
        regularization: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.projection_bootstrap = projection_bootstrap
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.regularization = regularization
        self.random_state = random_state

    def _validate_parameters(self) -> None:
        if not isinstance(self.n_estimators, (int, np.integer)) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be None or a positive integer")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        if not isinstance(self.bootstrap, (bool, np.bool_)):
            raise ValueError("bootstrap must be a boolean")
        if not isinstance(self.oob_score, (bool, np.bool_)):
            raise ValueError("oob_score must be a boolean")
        if self.oob_score and not self.bootstrap:
            raise ValueError("oob_score=True requires bootstrap=True for CCF-Bag")
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")

    def _fit_encoded(self, X: np.ndarray, y: np.ndarray, n_outputs: int) -> None:
        self._validate_parameters()
        rng = check_random_state(self.random_state)
        self.estimators_ = []
        self.estimators_samples_ = []
        for _ in range(int(self.n_estimators)):
            seed = int(rng.randint(np.iinfo(np.int32).max))
            indices = (
                rng.randint(0, len(X), size=len(X))
                if self.bootstrap
                else np.arange(len(X), dtype=np.int64)
            )
            tree = _CanonicalCorrelationTree(
                task=self._task,
                n_outputs=n_outputs,
                max_features=self.max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                projection_bootstrap=self.projection_bootstrap,
                regularization=self.regularization,
                random_state=seed,
            ).fit(X[indices], y[indices])
            self.estimators_.append(tree)
            self.estimators_samples_.append(indices)
        importances = np.vstack([tree.feature_importances_ for tree in self.estimators_])
        self.feature_importances_ = importances.mean(axis=0)

    def _set_oob_predictions(self, X: np.ndarray) -> None:
        """Calculate CCF-Bag OOB outputs from persisted sampled-row inventories."""

        if not (self.bootstrap and self.oob_score):
            return

        n_samples = X.shape[0]
        counts = np.zeros(n_samples, dtype=np.int32)
        if self._task == "classification":
            totals = np.zeros((n_samples, self.n_classes_), dtype=float)
        else:
            totals = np.zeros((n_samples, self.n_outputs_), dtype=float)

        for tree, sampled in zip(self.estimators_, self.estimators_samples_):
            in_bag = np.zeros(n_samples, dtype=bool)
            in_bag[np.unique(np.asarray(sampled, dtype=np.int64))] = True
            oob_rows = np.flatnonzero(~in_bag)
            if oob_rows.size == 0:
                continue
            totals[oob_rows] += tree.predict_values(X[oob_rows])
            counts[oob_rows] += 1

        valid = counts > 0
        divisors = counts.astype(float)
        divisors[~valid] = 1.0
        self.oob_counts_ = counts
        self.oob_valid_mask_ = valid

        if self._task == "classification":
            self.oob_decision_function_ = totals / divisors[:, None]
            self.oob_score_ = float(
                np.mean(
                    self.classes_[np.argmax(self.oob_decision_function_, axis=1)]
                    == self._y_original_
                )
            )
            return

        predictions = totals / divisors[:, None]
        self.oob_prediction_ = predictions[:, 0] if self._single_output else predictions
        self.oob_score_ = float(r2_score(self._y_original_, self.oob_prediction_))

    def apply(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X_checked = validate_data(self, X, reset=False, ensure_2d=True, dtype=float)
        return np.column_stack([tree.apply(X_checked) for tree in self.estimators_])

    def decision_path(self, X: Any) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Return concatenated node indicators and sklearn-style offsets."""

        check_is_fitted(self, "estimators_")
        X_checked = validate_data(self, X, reset=False, ensure_2d=True, dtype=float)
        paths = [tree.decision_path(X_checked) for tree in self.estimators_]
        offsets = np.cumsum([0] + [path.shape[1] for path in paths], dtype=np.int64)
        return sparse.hstack(paths, format="csr"), offsets


class CanonicalCorrelationForestClassifier(
    ClassifierMixin, _CanonicalCorrelationForestBase
):
    """Canonical Correlation Forest classifier with local oblique CCA splits."""

    _task = "classification"

    def fit(self, X: Any, y: Any) -> "CanonicalCorrelationForestClassifier":
        X_checked, y_checked = validate_data(
            self, X, y, reset=True, ensure_2d=True, dtype=float,
        )
        check_classification_targets(y_checked)
        if y_checked.ndim != 1:
            raise ValueError("multi-output classification is not yet supported")
        self.n_features_in_ = X_checked.shape[1]
        self.classes_, encoded = np.unique(y_checked, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self._y_original_ = np.asarray(y_checked).copy()
        self._fit_encoded(X_checked, encoded, self.n_classes_)
        self._set_oob_predictions(X_checked)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "classes_")
        X_checked = validate_data(self, X, reset=False, ensure_2d=True, dtype=float)
        probabilities = np.stack(
            [tree.predict_values(X_checked) for tree in self.estimators_], axis=0
        ).mean(axis=0)
        row_sum = probabilities.sum(axis=1, keepdims=True)
        return np.divide(
            probabilities,
            row_sum,
            out=np.full_like(probabilities, 1.0 / self.n_classes_),
            where=row_sum > 0,
        )

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "classes_")
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class CanonicalCorrelationForestRegressor(
    RegressorMixin, _CanonicalCorrelationForestBase
):
    """Canonical Correlation Forest regressor with local oblique CCA splits."""

    _task = "regression"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags

    def fit(self, X: Any, y: Any) -> "CanonicalCorrelationForestRegressor":
        X_checked, y_checked = validate_data(
            self, X, y, reset=True, ensure_2d=True, dtype=float, multi_output=True,
        )
        self.n_features_in_ = X_checked.shape[1]
        self._single_output = y_checked.ndim == 1
        y_matrix = np.asarray(y_checked, dtype=float).reshape(len(y_checked), -1)
        self.n_outputs_ = y_matrix.shape[1]
        self._y_original_ = np.asarray(y_checked).copy()
        self._fit_encoded(X_checked, y_matrix, self.n_outputs_)
        self._set_oob_predictions(X_checked)
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X_checked = validate_data(self, X, reset=False, ensure_2d=True, dtype=float)
        predictions = np.stack(
            [tree.predict_values(X_checked) for tree in self.estimators_], axis=0
        ).mean(axis=0)
        return predictions[:, 0] if self._single_output else predictions


def validate_ccf_oracle(
    estimator: Any,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    oracle_predictions: Any,
    *,
    tolerance: float = 1e-8,
) -> CCFConformance:
    """Fit a CCF and compare predictions with a deterministic MATLAB fixture."""

    estimator.fit(X_train, y_train)
    predicted = np.asarray(estimator.predict(X_test))
    expected = np.asarray(oracle_predictions)
    if predicted.shape != expected.shape:
        return CCFConformance(False, float("inf"), tolerance, int(expected.size))
    try:
        error = float(np.max(np.abs(predicted.astype(float) - expected.astype(float))))
    except (TypeError, ValueError):
        error = 0.0 if np.array_equal(predicted, expected) else float("inf")
    return CCFConformance(error <= tolerance, error, tolerance, int(expected.size))


__all__ = [
    "CCFConformance",
    "CanonicalCorrelationForestClassifier",
    "CanonicalCorrelationForestRegressor",
    "load_ccf_oracle_fixture",
    "validate_ccf_oracle",
]
