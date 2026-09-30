"""Fast, standalone rotation-forest estimators.

The estimators in this module follow the construction used by aeon's
``RotationForestClassifier`` and ``RotationForestRegressor``: every tree
partitions the attributes into random groups, learns a PCA basis from a random
subset of rows for each group, and fits a decision tree after rotating the
complete training set.  Classification PCA samples additionally use a random
non-empty subset of classes.

The implementation is independent of aeon and is optimized for the tabular
workloads used by :mod:`forestgap` and ProximityDiffusion.  In particular, a
tree's group projections are compiled into one dense or sparse linear
operator, serial execution bypasses joblib, predictions are accumulated
without a tree-by-sample prediction tensor, and regression supports native
multi-output targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
import time
from typing import Any, Literal

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted

try:
    from sklearn.utils.validation import validate_data
except ImportError:  # pragma: no cover - compatibility with sklearn < 1.6

    def validate_data(estimator: Any, **kwargs: Any) -> Any:
        return estimator._validate_data(**kwargs)


_MAX_SEED = np.iinfo(np.int32).max


@dataclass(frozen=True)
class _PCAProjection:
    """Small immutable subset of sklearn's fitted PCA transform contract."""

    mean_: np.ndarray
    components_: np.ndarray

    def transform(self, X: Any) -> np.ndarray:
        values = check_array(X, ensure_2d=True, dtype=np.float64)
        return (values - self.mean_) @ self.components_.T


@dataclass(frozen=True)
class _CompiledRotation:
    """A complete tree rotation represented by one linear operation."""

    matrix: np.ndarray | sparse.csc_matrix
    offset: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        transformed = np.asarray(X @ self.matrix, dtype=np.float32)
        transformed += self.offset
        np.nan_to_num(
            transformed,
            copy=False,
            nan=0.0,
            posinf=np.finfo(np.float32).max,
            neginf=np.finfo(np.float32).min,
        )
        return np.ascontiguousarray(transformed, dtype=np.float32)


@dataclass(frozen=True)
class _FittedRotationTree:
    estimator: Any
    projections: tuple[_PCAProjection, ...]
    groups: tuple[np.ndarray, ...]
    rotation: _CompiledRotation


class _BaseRotationForest(BaseEstimator):
    """Shared implementation for sklearn-compatible rotation forests."""

    _forestgap_rotation_forest = True
    _task: Literal["classification", "regression"]

    def __init__(
        self,
        n_estimators: int = 200,
        min_group: int = 3,
        max_group: int = 3,
        remove_proportion: float = 0.5,
        base_estimator: BaseEstimator | None = None,
        pca_solver: str = "auto",
        time_limit_in_minutes: float = 0.0,
        contract_max_n_estimators: int = 500,
        n_jobs: int | None = 1,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.min_group = min_group
        self.max_group = max_group
        self.remove_proportion = remove_proportion
        self.base_estimator = base_estimator
        self.pca_solver = pca_solver
        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_estimators = contract_max_n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _validate_hyperparameters(self) -> None:
        for name in ("n_estimators", "min_group", "max_group"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if int(self.min_group) > int(self.max_group):
            raise ValueError("min_group must be less than or equal to max_group")
        if not isinstance(self.remove_proportion, Real):
            raise TypeError("remove_proportion must be a real number")
        if not 0 < float(self.remove_proportion) <= 1:
            raise ValueError("remove_proportion must lie in (0, 1]")
        if self.pca_solver not in {
            "auto",
            "full",
            "covariance_eigh",
            "arpack",
            "randomized",
        }:
            raise ValueError(
                "pca_solver must be 'auto', 'full', 'covariance_eigh', "
                "'arpack', or 'randomized'"
            )
        if not isinstance(self.time_limit_in_minutes, Real):
            raise TypeError("time_limit_in_minutes must be a real number")
        if float(self.time_limit_in_minutes) < 0:
            raise ValueError("time_limit_in_minutes must be non-negative")
        if (
            not isinstance(self.contract_max_n_estimators, Integral)
            or isinstance(self.contract_max_n_estimators, (bool, np.bool_))
            or int(self.contract_max_n_estimators) < 1
        ):
            raise ValueError("contract_max_n_estimators must be a positive integer")
        if self.n_jobs == 0:
            raise ValueError("n_jobs may not be zero")
        # Validate RandomState inputs before fitting any trees.
        check_random_state(self.random_state)

    def _default_estimator(self) -> BaseEstimator:
        raise NotImplementedError

    def _fit_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        class_rows: tuple[np.ndarray, ...] | None,
    ) -> None:
        self._validate_hyperparameters()
        self.n_cases_, self.n_atts_ = X.shape
        self._useful_atts = np.ptp(X, axis=0) > 0
        if not np.any(self._useful_atts):
            raise ValueError("All attributes in X contain the same value.")

        useful = np.asarray(X[:, self._useful_atts], dtype=np.float32)
        self._min = useful.min(axis=0)
        self._ptp = useful.max(axis=0) - self._min
        normalized = np.ascontiguousarray(
            (useful - self._min) / self._ptp,
            dtype=np.float32,
        )
        self._n_atts = normalized.shape[1]
        self._n_jobs = effective_n_jobs(self.n_jobs)
        self._base_estimator = clone(
            self._default_estimator()
            if self.base_estimator is None
            else self.base_estimator
        )
        self.estimator_ = self._base_estimator
        if not callable(getattr(self._base_estimator, "fit", None)):
            raise TypeError("base_estimator must implement fit(X, y)")
        if not callable(getattr(self._base_estimator, "predict", None)):
            raise TypeError("base_estimator must implement predict(X)")

        parent_rng = check_random_state(self.random_state)
        fitted: list[_FittedRotationTree] = []
        seeds: list[int] = []
        time_limit = float(self.time_limit_in_minutes) * 60.0
        started = time.perf_counter()

        if time_limit == 0:
            seeds = [
                int(value)
                for value in parent_rng.randint(_MAX_SEED, size=int(self.n_estimators))
            ]
            fitted = self._fit_seed_batch(seeds, normalized, y, class_rows=class_rows)
        else:
            maximum = int(self.contract_max_n_estimators)
            batch_size = max(1, self._n_jobs)
            while len(fitted) < maximum and (
                not fitted or time.perf_counter() - started < time_limit
            ):
                size = min(batch_size, maximum - len(fitted))
                batch_seeds = [
                    int(value) for value in parent_rng.randint(_MAX_SEED, size=size)
                ]
                fitted.extend(
                    self._fit_seed_batch(
                        batch_seeds, normalized, y, class_rows=class_rows
                    )
                )
                seeds.extend(batch_seeds)

        self.estimators_ = [item.estimator for item in fitted]
        self.pcas_ = [item.projections for item in fitted]
        self.groups_ = [item.groups for item in fitted]
        self.rotation_matrices_ = [item.rotation.matrix for item in fitted]
        self.rotation_offsets_ = [item.rotation.offset for item in fitted]
        self._rotations = [item.rotation for item in fitted]
        # Compatibility attributes consumed by forestgap's legacy aeon adapter.
        self._pcas = self.pcas_
        self._groups = self.groups_
        self._n_estimators = len(self.estimators_)
        self.estimator_seeds_ = np.asarray(seeds, dtype=np.int64)
        self.feature_importances_ = self._compute_feature_importances()
        self._is_fitted = True

    def _fit_seed_batch(
        self,
        seeds: list[int],
        X: np.ndarray,
        y: np.ndarray,
        *,
        class_rows: tuple[np.ndarray, ...] | None,
    ) -> list[_FittedRotationTree]:
        if self._n_jobs == 1 or len(seeds) == 1:
            return [
                self._fit_one_tree(seed, X, y, class_rows=class_rows) for seed in seeds
            ]
        return Parallel(n_jobs=self._n_jobs, prefer="threads")(
            delayed(self._fit_one_tree)(seed, X, y, class_rows=class_rows)
            for seed in seeds
        )

    def _fit_one_tree(
        self,
        seed: int,
        X: np.ndarray,
        y: np.ndarray,
        *,
        class_rows: tuple[np.ndarray, ...] | None,
    ) -> _FittedRotationTree:
        rng = check_random_state(seed)
        groups = tuple(self._generate_groups(rng))
        projections: list[_PCAProjection] = []
        for group in groups:
            candidate_rows = self._projection_candidate_rows(rng, len(X), class_rows)
            sample_size = max(
                1, int(candidate_rows.size * float(self.remove_proportion))
            )
            selected = rng.choice(candidate_rows, size=sample_size, replace=False)
            projection = self._fit_projection(X[selected][:, group], rng)
            projections.append(projection)

        rotation = self._compile_rotation(groups, tuple(projections), X.shape[1])
        transformed = rotation.transform(X)
        tree = clone(self._base_estimator)
        parameters = tree.get_params(deep=False)
        if "random_state" in parameters:
            tree.set_params(random_state=int(rng.randint(_MAX_SEED)))
        tree.fit(transformed, y)
        if not callable(getattr(tree, "apply", None)):
            raise TypeError(
                "base_estimator must expose apply(X) after fitting so forest "
                "relations can recover terminal nodes"
            )
        return _FittedRotationTree(tree, tuple(projections), groups, rotation)

    def _projection_candidate_rows(
        self,
        rng: np.random.RandomState,
        n_samples: int,
        class_rows: tuple[np.ndarray, ...] | None,
    ) -> np.ndarray:
        if class_rows is None:
            return np.arange(n_samples, dtype=np.int64)
        selected_classes = rng.choice(
            len(class_rows),
            size=rng.randint(1, len(class_rows) + 1),
            replace=False,
        )
        return np.concatenate([class_rows[int(index)] for index in selected_classes])

    def _fit_projection(
        self, values: np.ndarray, rng: np.random.RandomState
    ) -> _PCAProjection:
        values = np.asarray(values, dtype=np.float64)
        # aeon retries a degenerate PCA after adding ten random rows.  Do the
        # same before the decomposition rather than constructing repeated PCA
        # estimator objects.
        centered = values - values.mean(axis=0, keepdims=True)
        if not np.any(np.var(centered, axis=0) > 0):
            values = np.vstack([values, rng.random_sample((10, values.shape[1]))])

        if self.pca_solver in {"auto", "full"}:
            mean = values.mean(axis=0)
            centered = values - mean
            left, _, components = np.linalg.svd(centered, full_matrices=False)
            _, components = svd_flip(left, components, u_based_decision=False)
        else:
            pca = PCA(
                random_state=int(rng.randint(_MAX_SEED)),
                svd_solver=self.pca_solver,
            ).fit(values)
            mean = pca.mean_
            components = pca.components_
        return _PCAProjection(
            np.asarray(mean, dtype=np.float32),
            np.asarray(components, dtype=np.float32),
        )

    @staticmethod
    def _compile_rotation(
        groups: tuple[np.ndarray, ...],
        projections: tuple[_PCAProjection, ...],
        n_features: int,
    ) -> _CompiledRotation:
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        data: list[np.ndarray] = []
        offsets: list[np.ndarray] = []
        output_start = 0
        for group, projection in zip(groups, projections):
            components = projection.components_.T
            n_outputs = components.shape[1]
            rows.append(np.repeat(group, n_outputs))
            cols.append(
                np.tile(np.arange(output_start, output_start + n_outputs), len(group))
            )
            data.append(components.reshape(-1))
            offsets.append(-projection.mean_ @ components)
            output_start += n_outputs

        matrix = sparse.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_features, output_start),
            dtype=np.float32,
        ).tocsc()
        matrix.sum_duplicates()
        density = matrix.nnz / max(1, matrix.shape[0] * matrix.shape[1])
        operator: np.ndarray | sparse.csc_matrix
        operator = matrix.toarray() if density >= 0.2 else matrix
        return _CompiledRotation(
            operator,
            np.ascontiguousarray(np.concatenate(offsets), dtype=np.float32),
        )

    def _generate_groups(self, rng: np.random.RandomState) -> list[np.ndarray]:
        permutation = rng.permutation(self._n_atts)
        group_size_count = np.zeros(
            int(self.max_group) - int(self.min_group) + 1, dtype=np.int64
        )
        n_attributes = 0
        n_groups = 0
        while n_attributes < self._n_atts:
            offset = int(rng.randint(group_size_count.size))
            group_size_count[offset] += 1
            n_attributes += int(self.min_group) + offset
            n_groups += 1

        groups: list[np.ndarray] = []
        current_attribute = 0
        current_size = 0
        for _ in range(n_groups):
            while group_size_count[current_size] == 0:
                current_size += 1
            group_size_count[current_size] -= 1
            size = int(self.min_group) + current_size
            group = np.empty(size, dtype=np.int64)
            for index in range(size):
                if current_attribute < len(permutation):
                    group[index] = permutation[current_attribute]
                else:
                    group[index] = permutation[int(rng.randint(len(permutation)))]
                current_attribute += 1
            groups.append(group)
        return groups

    def _prepare_X(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        checked = validate_data(
            self,
            X=X,
            reset=False,
            accept_sparse=False,
            ensure_2d=True,
            dtype=np.float32,
        )
        useful = np.asarray(checked[:, self._useful_atts], dtype=np.float32)
        return np.ascontiguousarray((useful - self._min) / self._ptp, dtype=np.float32)

    def _check_X(self, X: Any) -> np.ndarray:
        """Compatibility hook used by the legacy aeon rotation adapter."""

        return check_array(X, ensure_2d=True, accept_sparse=False, dtype=np.float32)

    def _transform_all(self, X: Any) -> list[np.ndarray]:
        """Return the rotated feature matrix used by every fitted tree."""

        prepared = self._prepare_X(X)
        if self._n_jobs == 1 or len(self._rotations) == 1:
            return [rotation.transform(prepared) for rotation in self._rotations]
        return Parallel(n_jobs=self._n_jobs, prefer="threads")(
            delayed(rotation.transform)(prepared) for rotation in self._rotations
        )

    def _transform_one(self, X: Any, tree_index: int) -> np.ndarray:
        """Return the rotated feature matrix for a single fitted tree."""

        prepared = self._prepare_X(X)
        return self._rotations[int(tree_index)].transform(prepared)

    def apply(self, X: Any) -> np.ndarray:
        """Return terminal-node identifiers with one column per tree."""

        transformed = self._transform_all(X)
        leaves = [
            tree.apply(values) for tree, values in zip(self.estimators_, transformed)
        ]
        return np.column_stack(leaves)

    def decision_path(self, X: Any) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Return concatenated tree paths and sklearn-style node offsets."""

        transformed = self._transform_all(X)
        paths: list[sparse.csr_matrix] = []
        for tree, values in zip(self.estimators_, transformed):
            decision_path = getattr(tree, "decision_path", None)
            if not callable(decision_path):
                raise TypeError("base_estimator must expose decision_path(X)")
            paths.append(sparse.csr_matrix(decision_path(values)))
        offsets = np.cumsum([0] + [path.shape[1] for path in paths], dtype=np.int64)
        return sparse.hstack(paths, format="csr"), offsets

    def _compute_feature_importances(self) -> np.ndarray:
        useful_importance = np.zeros(self._n_atts, dtype=np.float64)
        for tree, rotation in zip(self.estimators_, self._rotations):
            rotated = getattr(tree, "feature_importances_", None)
            if rotated is None:
                continue
            mapped = abs(rotation.matrix) @ np.asarray(rotated, dtype=np.float64)
            useful_importance += np.asarray(mapped).ravel()
        total = useful_importance.sum()
        if total > 0:
            useful_importance /= total
        result = np.zeros(self.n_features_in_, dtype=np.float64)
        result[self._useful_atts] = useful_importance
        return result

    @staticmethod
    def _set_tree_random_state(tree: Any, rng: np.random.RandomState) -> None:
        if "random_state" in tree.get_params(deep=False):
            # Passing the advanced RandomState matches aeon's OOB refit flow.
            tree.set_params(random_state=rng)

    def _train_preds_for_estimator(
        self,
        transformed: list[np.ndarray] | tuple[np.ndarray, ...],
        y: Any,
        index: int,
        rng: np.random.RandomState,
    ) -> list[Any]:
        """Fast equivalent of aeon's private regression OOB helper."""

        y_array = np.asarray(y)
        sampled = rng.choice(self.n_cases_, size=self.n_cases_, replace=True)
        counts = np.bincount(sampled, minlength=self.n_cases_)
        oob = np.flatnonzero(counts == 0)
        result = np.zeros_like(y_array, dtype=np.float64)
        if oob.size:
            tree = clone(self._base_estimator)
            self._set_tree_random_state(tree, rng)
            tree.fit(transformed[index][sampled], y_array[sampled])
            result[oob] = tree.predict(transformed[index][oob])
        return [result, oob.tolist()]

    def _train_probas_for_estimator(
        self,
        transformed: list[np.ndarray] | tuple[np.ndarray, ...],
        y: Any,
        index: int,
        rng: np.random.RandomState,
    ) -> list[Any]:
        """Fast equivalent of aeon's private classification OOB helper."""

        y_array = np.asarray(y)
        sampled = rng.choice(self.n_cases_, size=self.n_cases_, replace=True)
        counts = np.bincount(sampled, minlength=self.n_cases_)
        oob = np.flatnonzero(counts == 0)
        result = np.zeros((self.n_cases_, self.n_classes_), dtype=np.float64)
        if oob.size:
            tree = clone(self._base_estimator)
            self._set_tree_random_state(tree, rng)
            tree.fit(transformed[index][sampled], y_array[sampled])
            probabilities = np.asarray(
                tree.predict_proba(transformed[index][oob]), dtype=np.float64
            )
            class_to_column = {label: col for col, label in enumerate(self.classes_)}
            for source, label in enumerate(tree.classes_):
                result[oob, class_to_column[label]] = probabilities[:, source]
        return [result, oob.tolist()]

    def _compute_oob_outputs(
        self, X: Any, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        transformed = self._transform_all(X)
        parent_rng = check_random_state(self.random_state)
        seeds = [
            int(value)
            for value in parent_rng.randint(_MAX_SEED, size=self._n_estimators)
        ]

        def one(tree_index: int, seed: int) -> list[Any]:
            function = (
                self._train_probas_for_estimator
                if self._task == "classification"
                else self._train_preds_for_estimator
            )
            return function(transformed, y, tree_index, check_random_state(seed))

        if self._n_jobs == 1 or self._n_estimators == 1:
            outputs = [one(index, seed) for index, seed in enumerate(seeds)]
        else:
            outputs = Parallel(n_jobs=self._n_jobs, prefer="threads")(
                delayed(one)(index, seed) for index, seed in enumerate(seeds)
            )
        totals = np.sum([item[0] for item in outputs], axis=0)
        counts = np.zeros(self.n_cases_, dtype=np.int32)
        for _, oob in outputs:
            counts[np.asarray(oob, dtype=np.int64)] += 1
        valid = counts > 0
        divisors = np.where(valid, counts, 1)
        reshape = (self.n_cases_,) + (1,) * (totals.ndim - 1)
        return totals / divisors.reshape(reshape), valid, counts

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "estimators_")


class RotationForestClassifier(ClassifierMixin, _BaseRotationForest):
    """Rotation forest classifier for continuous tabular predictors.

    Each base tree receives its own random grouping of features and a PCA
    rotation learned group-by-group from sampled rows and classes.  Class
    probabilities are the unweighted mean of the base-tree probabilities.
    """

    _task = "classification"

    def _default_estimator(self) -> BaseEstimator:
        return DecisionTreeClassifier(criterion="entropy")

    def fit(self, X: Any, y: Any) -> RotationForestClassifier:
        """Fit the rotation forest and return ``self``."""

        X_checked, y_checked = validate_data(
            self,
            X=X,
            y=y,
            reset=True,
            accept_sparse=False,
            ensure_min_samples=2,
            dtype=np.float32,
        )
        check_classification_targets(y_checked)
        self.classes_, encoded = np.unique(y_checked, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self._class_dictionary = {
            label: index for index, label in enumerate(self.classes_)
        }
        rows = tuple(
            np.flatnonzero(encoded == index) for index in range(self.n_classes_)
        )
        self._fit_forest(
            np.asarray(X_checked, dtype=np.float32),
            np.asarray(encoded, dtype=np.int64),
            class_rows=rows,
        )
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities by averaging the rotated trees."""

        transformed = self._transform_all(X)

        def one(tree: Any, values: np.ndarray) -> np.ndarray:
            probabilities = np.asarray(tree.predict_proba(values), dtype=np.float64)
            if probabilities.shape[1] == self.n_classes_:
                return probabilities
            aligned = np.zeros((len(values), self.n_classes_), dtype=np.float64)
            aligned[:, np.asarray(tree.classes_, dtype=np.int64)] = probabilities
            return aligned

        if self._n_jobs == 1 or self._n_estimators == 1:
            total = np.zeros((len(transformed[0]), self.n_classes_), dtype=np.float64)
            for tree, values in zip(self.estimators_, transformed):
                total += one(tree, values)
        else:
            predictions = Parallel(n_jobs=self._n_jobs, prefer="threads")(
                delayed(one)(tree, values)
                for tree, values in zip(self.estimators_, transformed)
            )
            total = np.sum(predictions, axis=0)
        return total / self._n_estimators

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for ``X``."""

        check_is_fitted(self, "estimators_")
        return self.classes_.take(np.argmax(self.predict_proba(X), axis=1))

    def fit_predict_proba(self, X: Any, y: Any) -> np.ndarray:
        """Fit the forest and return aeon-compatible OOB probabilities."""

        self.fit(X, y)
        probabilities, valid, counts = self._compute_oob_outputs(X, np.asarray(y))
        probabilities[~valid] = 1.0 / self.n_classes_
        self.oob_decision_function_ = probabilities
        self.oob_valid_mask_ = valid
        self.oob_counts_ = counts
        if valid.any():
            self.oob_score_ = float(
                accuracy_score(
                    np.asarray(y)[valid],
                    self.classes_.take(np.argmax(probabilities[valid], axis=1)),
                )
            )
        else:
            self.oob_score_ = np.nan
        return probabilities

    def fit_predict(self, X: Any, y: Any) -> np.ndarray:
        """Fit the forest and return labels from OOB probabilities."""

        return self.classes_.take(np.argmax(self.fit_predict_proba(X, y), axis=1))


class RotationForestRegressor(RegressorMixin, _BaseRotationForest):
    """Rotation forest regressor with native multi-output support."""

    _task = "regression"

    def _default_estimator(self) -> BaseEstimator:
        return DecisionTreeRegressor(criterion="squared_error")

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags

    def fit(self, X: Any, y: Any) -> RotationForestRegressor:
        """Fit the rotation forest and return ``self``."""

        X_checked, y_checked = validate_data(
            self,
            X=X,
            y=y,
            reset=True,
            accept_sparse=False,
            ensure_min_samples=2,
            dtype=np.float32,
            multi_output=True,
            y_numeric=True,
        )
        y_array = np.asarray(y_checked, dtype=np.float64)
        self.n_outputs_ = 1 if y_array.ndim == 1 else y_array.shape[1]
        self._label_average = np.mean(y_array, axis=0)
        self._fit_forest(
            np.asarray(X_checked, dtype=np.float32),
            y_array,
            class_rows=None,
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict targets by averaging the rotated trees."""

        transformed = self._transform_all(X)
        if self._n_jobs == 1 or self._n_estimators == 1:
            total: np.ndarray | None = None
            for tree, values in zip(self.estimators_, transformed):
                prediction = np.asarray(tree.predict(values), dtype=np.float64)
                total = prediction.copy() if total is None else total + prediction
            if total is None:  # pragma: no cover - n_estimators is validated
                raise RuntimeError("No fitted estimators are available")
        else:
            predictions = Parallel(n_jobs=self._n_jobs, prefer="threads")(
                delayed(tree.predict)(values)
                for tree, values in zip(self.estimators_, transformed)
            )
            total = np.sum(predictions, axis=0, dtype=np.float64)
        return total / self._n_estimators

    def fit_predict(self, X: Any, y: Any) -> np.ndarray:
        """Fit the forest and return aeon-compatible OOB predictions."""

        self.fit(X, y)
        predictions, valid, counts = self._compute_oob_outputs(X, np.asarray(y))
        predictions[~valid] = self._label_average
        self.oob_prediction_ = predictions
        self.oob_valid_mask_ = valid
        self.oob_counts_ = counts
        self.oob_score_ = (
            float(r2_score(np.asarray(y)[valid], predictions[valid]))
            if valid.any()
            else np.nan
        )
        return predictions


__all__ = ["RotationForestClassifier", "RotationForestRegressor"]
