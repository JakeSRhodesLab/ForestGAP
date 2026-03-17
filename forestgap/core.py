from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
from inspect import Parameter, Signature
from typing import Any, List, Literal, Optional, Protocol, Sequence, Tuple, Type, TypeVar, Union, cast, runtime_checkable
import warnings
from warnings import warn

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

try:
    from sklearn.utils.validation import validate_data
except ImportError:
    # sklearn < 1.6: validate_data is not a standalone function;
    # fall back to the estimator's _validate_data method.
    def validate_data(estimator, **kwargs):
        return estimator._validate_data(**kwargs)

# aeon RotationForest still uses sklearn's deprecated _validate_data path on
# current releases; suppress this single known deprecation at library boundary.
warnings.filterwarnings(
    "ignore",
    message=r".*BaseEstimator\._validate_data.*",
    category=FutureWarning,
)

try:
    from sklearn.ensemble import _forest as _sk_forest
except Exception:
    _sk_forest = None

_generate_sample_indices = getattr(_sk_forest, "_generate_sample_indices", None)
_generate_unsampled_indices = getattr(_sk_forest, "_generate_unsampled_indices", None)

ArrayLike = Union[np.ndarray, Sequence[float]]
MatrixLike = Union[np.ndarray, sparse.csr_matrix]
ProxMethod = Literal["original", "oob", "rfgap"]
MatrixType = Literal["sparse", "dense"]
SymmMode = Literal["arithmetic", "geometric", "none", None]


@contextmanager
def _suppress_base_validate_data_futurewarning() -> Any:
    """Suppress aeon/sklearn deprecation warning for BaseEstimator._validate_data."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*BaseEstimator\._validate_data.*",
            category=FutureWarning,
        )
        yield


def _is_rotation_forest_instance(obj: Any) -> bool:
    if any("RotationForest" in cls.__name__ for cls in type(obj).mro()):
        return True
    return all(hasattr(obj, attr) for attr in ("_pcas", "_groups", "_useful_atts"))


def _is_isolation_forest_instance(obj: Any) -> bool:
    if any("IsolationForest" in cls.__name__ for cls in type(obj).mro()):
        return True
    return hasattr(obj, "estimators_features_") and hasattr(obj, "score_samples")


def _is_rotation_forest_class(base_cls: Type[Any]) -> bool:
    if any("RotationForest" in cls.__name__ for cls in base_cls.mro()):
        return True
    return all(hasattr(base_cls, attr) for attr in ("fit_predict", "fit_predict_proba"))


def _is_isolation_forest_class(base_cls: Type[Any]) -> bool:
    if any("IsolationForest" in cls.__name__ for cls in base_cls.mro()):
        return True
    return hasattr(base_cls, "score_samples") and hasattr(base_cls, "decision_function")


@runtime_checkable
class ForestLike(Protocol):
    estimators_: Sequence[Any]
    n_estimators: int

    def apply(self, X: Any) -> np.ndarray:
        ...


TForest = TypeVar("TForest")


@dataclass
class _LeafEncoding:
    """Compact leaf-to-column encoding for sparse proximity construction."""

    offsets: np.ndarray
    values_per_tree: List[np.ndarray]
    total_features: int


class SparseProximityMixin:
    """
    Sparse random-forest proximity mixin based on a `Q @ W.T` factorization.

    This mixin is designed for forest-like estimators that expose:
    - `fit(X, y, ...)`
    - `apply(X)` returning leaf indices of shape `(n_samples, n_estimators)`
    - `estimators_` and `n_estimators` after fitting

    Supported proximity methods:
    - `"original"`: classical RF proximity
    - `"oob"`: OOB-only approximation in separable sparse form
    - `"rfgap"`: RF-GAP asymmetric proximity

    Notes
    -----
    - For `"oob"` and `"rfgap"`, bootstrap sample indices are required.
      By default this mixin uses scikit-learn internals when available.
    - For non-sklearn estimators (e.g. aeon rotation forests), override
      `_compute_bootstrap_stats` if bootstrap internals differ.
    - The implementation stores only sparse factors and computes proximities via
      sparse matrix multiplication for scalability.
    """

    prox_method: ProxMethod
    matrix_type: MatrixType
    non_zero_diagonal: bool
    symm_mode: SymmMode
    max_normalize: bool

    def _init_sparse_proximity(
        self,
        prox_method: ProxMethod = "rfgap",
        matrix_type: MatrixType = "sparse",
        non_zero_diagonal: bool = False,
        symm_mode: SymmMode = None,
        max_normalize: bool = False,
    ) -> None:
        if prox_method not in ("original", "oob", "rfgap"):
            raise ValueError("prox_method must be one of {'original', 'oob', 'rfgap'}")
        if matrix_type not in ("sparse", "dense"):
            raise ValueError("matrix_type must be 'sparse' or 'dense'")
        if symm_mode == "none":
            symm_mode = None
        if symm_mode not in (None, "arithmetic", "geometric"):
            raise ValueError("symm_mode must be one of {None, 'arithmetic', 'geometric', 'none'}")

        self.prox_method = prox_method
        self.matrix_type = matrix_type
        self.non_zero_diagonal = non_zero_diagonal
        self.symm_mode = symm_mode
        self.max_normalize = max_normalize

        self._leaf_encoding: Optional[_LeafEncoding] = None
        self.leaf_matrix_: Optional[np.ndarray] = None
        self.W_mat: Optional[sparse.csr_matrix] = None
        self.oob_mask_: Optional[np.ndarray] = None
        self.in_bag_counts_: Optional[np.ndarray] = None
        self._diag_offset: Optional[int] = None
        self._n_train_samples: Optional[int] = None
        self._n_total_features: Optional[int] = None
        self._oob_surrogate_estimators_: Optional[List[Any]] = None
        self._oob_surrogate_leaf_matrix_: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Optional[Any] = None, **fit_kwargs: Any) -> "SparseProximityMixin":
        """
        Fit the base estimator, then build sparse proximity factors on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like, optional
            Training labels/targets, passed through to the wrapped estimator.
        **fit_kwargs : dict
            Additional fit arguments forwarded to the wrapped estimator.

        Returns
        -------
        SparseProximityMixin
            Fitted estimator with sparse proximity factors cached.
        """
        base_fit = getattr(super(), "fit")
        if _is_rotation_forest_instance(self):
            with _suppress_base_validate_data_futurewarning():
                if y is None:
                    base_fit(X, **fit_kwargs)
                else:
                    base_fit(X, y, **fit_kwargs)
        else:
            if y is None:
                base_fit(X, **fit_kwargs)
            else:
                base_fit(X, y, **fit_kwargs)

        self.prox_fit(X, y)
        self._set_rotation_forest_oob_attributes(X, y)
        return self

    def _set_rotation_forest_oob_attributes(self, X: Any, y: Optional[Any]) -> None:
        super_hook = getattr(super(), "_set_rotation_forest_oob_attributes", None)
        if callable(super_hook):
            super_hook(X, y)

    def predict(self, X: Any) -> Any:
        base_predict = getattr(super(), "predict")
        if _is_rotation_forest_instance(self):
            with _suppress_base_validate_data_futurewarning():
                return base_predict(X)
        return base_predict(X)

    def prox_fit(self, X: Any, y: Optional[Any] = None) -> None:
        """
        Build sparse factors required for proximity computations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data used during model fitting.
        """
        check_is_fitted(cast(Any, self))

        self._y_train = np.asarray(y) if y is not None else None
        self._oob_surrogate_estimators_ = None
        self._oob_surrogate_leaf_matrix_ = None
        self._rotation_original_leaf_encoding_: Optional[_LeafEncoding] = None
        self._rotation_original_W_mat_: Optional[sparse.csr_matrix] = None

        apply_fn = getattr(self, "apply", None)
        if apply_fn is None:
            raise RuntimeError("Estimator must implement apply(X) to use proximity methods.")

        leaves: Optional[np.ndarray] = None

        if self.prox_method in ("oob", "rfgap"):
            n_samples = int(len(X))
            oob_mask, in_bag_counts = self._compute_bootstrap_stats(n_samples, X=X, y=y)
            self.oob_mask_ = oob_mask.astype(np.int8)
            self.in_bag_counts_ = in_bag_counts.astype(np.float32)

            surrogate_leaves = getattr(self, "_oob_surrogate_leaf_matrix_", None)
            if surrogate_leaves is not None and surrogate_leaves.shape[0] == n_samples:
                # RotationForest: cache original-tree proximity structures for
                # prox_extend so that P_ext @ y reproduces predict(X_test).
                if _is_rotation_forest_instance(self) and self.prox_method == "rfgap":
                    orig_leaves = np.asarray(apply_fn(X))
                    orig_encoding = self._build_leaf_encoding(orig_leaves)
                    self._rotation_original_leaf_encoding_ = orig_encoding
                    self._rotation_original_W_mat_ = self._build_W_uniform(
                        orig_leaves, orig_encoding,
                    )
                leaves = np.asarray(surrogate_leaves)

        if leaves is None:
            leaves = np.asarray(apply_fn(X))

        if leaves.ndim != 2:
            raise ValueError("apply(X) must return a 2D array of shape (n_samples, n_estimators)")

        n_samples, n_trees = leaves.shape
        self._n_train_samples = n_samples
        self.leaf_matrix_ = leaves
        self._leaf_encoding = self._build_leaf_encoding(leaves)

        self._build_W_matrix()

    def get_proximities(self) -> MatrixLike:
        """
        Compute proximities among fitted/training samples.

        Returns
        -------
        scipy.sparse.csr_matrix or numpy.ndarray
            `(n_train, n_train)` proximity matrix in sparse or dense form.
        """
        self._check_proximity_ready()

        W = self.W_mat
        if W is None:
            raise RuntimeError("W matrix is not initialized.")

        leaves = self.leaf_matrix_
        if leaves is None:
            raise RuntimeError("Leaf matrix is not initialized.")

        if self.prox_method in ("original", "oob"):
            Q = W.copy()
        else:
            Q = self._build_Q_matrix(leaves, is_training=True)

        if self.max_normalize and self.prox_method == "rfgap" and self.non_zero_diagonal:
            diagonal = Q.multiply(self.W_mat).sum(axis=1).A.ravel()
            diagonal[diagonal == 0] = 1.0
            self._csr_row_scale_inplace(Q, 1.0 / diagonal)

        if self.symm_mode is not None and self.prox_method == "rfgap":
            P = self._block_symmetrize(Q, W, mode=cast(Literal["arithmetic", "geometric"], self.symm_mode))
        else:
            P = cast(sparse.csr_matrix, Q.dot(W.T).tocsr())

        if self.prox_method == "oob":
            P.data = np.minimum(P.data, 1.0)

        return P.toarray() if self.matrix_type == "dense" else P

    def prox_extend(self, X_new: Any) -> MatrixLike:
        """
        Compute proximities between new (test) samples and training samples.

        Since test points were not used to build any tree, every test point is
        treated as out-of-bag for every tree.  For supervised forests with
        ``prox_method='rfgap'``, the returned matrix can be used as weights:
        ``P @ y_train`` reproduces the forest's test predictions.

        Parameters
        ----------
        X_new : array-like of shape (n_test, n_features)
            New observations.

        Returns
        -------
        scipy.sparse.csr_matrix or numpy.ndarray
            `(n_test, n_train)` proximity matrix.
        """
        self._check_proximity_ready()

        # RotationForest rfgap: use original-tree W (c_j=1) so that
        # P_ext @ y_train reproduces predict(X_test).
        orig_W = getattr(self, "_rotation_original_W_mat_", None)
        orig_enc = getattr(self, "_rotation_original_leaf_encoding_", None)
        if (
            _is_rotation_forest_instance(self)
            and self.prox_method == "rfgap"
            and orig_W is not None
            and orig_enc is not None
        ):
            apply_fn = getattr(self, "apply", None)
            if apply_fn is None:
                raise RuntimeError("Estimator must implement apply(X) to use proximity methods.")
            leaves_new = np.asarray(apply_fn(X_new))
            Q_new = self._build_Q_matrix_with_encoding(
                leaves_new, orig_enc, orig_W.shape[1],
            )
            P_new = cast(sparse.csr_matrix, Q_new.dot(orig_W.T).tocsr())
            return P_new.toarray() if self.matrix_type == "dense" else P_new

        surrogate_estimators = getattr(self, "_oob_surrogate_estimators_", None)
        if _is_rotation_forest_instance(self) and surrogate_estimators is not None:
            leaves_new = self._rotation_forest_apply_surrogates(X_new)
        else:
            apply_fn = getattr(self, "apply", None)
            if apply_fn is None:
                raise RuntimeError("Estimator must implement apply(X) to use proximity methods.")
            leaves_new = np.asarray(apply_fn(X_new))

        Q_new = self._build_Q_matrix(leaves_new, is_training=False)
        W = self.W_mat
        if W is None:
            raise RuntimeError("W matrix is not initialized.")
        P_new = cast(sparse.csr_matrix, Q_new.dot(W.T).tocsr())

        if self.prox_method == "oob":
            P_new.data = np.minimum(P_new.data, 1.0)

        return P_new.toarray() if self.matrix_type == "dense" else P_new

    def prox_predict(self, X: Any = None) -> np.ndarray:
        """
        Predict using rfgap proximity-weighted training labels.

        For classifiers, returns class probability estimates (matching
        ``predict_proba`` for test data or ``oob_decision_function_`` for
        training data).  For regressors, returns predicted values (matching
        ``predict`` for test data or ``oob_prediction_`` for training data).
        When ``X`` is ``None``, out-of-bag predictions for the training
        data are returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            Test data.  If ``None``, returns OOB predictions for training
            data.

        Returns
        -------
        predictions : np.ndarray
            Class probabilities of shape (n_samples, n_classes) for
            classifiers, or predicted values of shape (n_samples,) for
            regressors.
        """
        if self.prox_method != "rfgap":
            raise ValueError("prox_predict requires prox_method='rfgap'")
        if self._y_train is None:
            raise RuntimeError(
                "No training labels available. Refit with labels to use prox_predict."
            )

        P = self.get_proximities() if X is None else self.prox_extend(X)

        classes = getattr(self, "classes_", None)
        if classes is not None:
            classes = np.asarray(classes)
            y_idx = np.searchsorted(classes, self._y_train)
            y_hot = sparse.csc_matrix(
                (np.ones(len(y_idx), dtype=np.float64), (np.arange(len(y_idx)), y_idx)),
                shape=(len(y_idx), classes.shape[0]),
            )
            prox_prob = np.asarray((P @ y_hot).todense() if sparse.issparse(P) else P @ y_hot)
            if X is None:
                rs = prox_prob.sum(axis=1, keepdims=True)
                rs[rs == 0] = 1.0
                prox_prob /= rs
            return prox_prob

        y = self._y_train.astype(np.float64)
        result = P @ y
        return np.asarray(result.todense()).ravel() if sparse.issparse(result) else np.asarray(result).ravel()

    def _check_proximity_ready(self) -> None:
        check_is_fitted(cast(Any, self))
        if self.W_mat is None or self.leaf_matrix_ is None or self._leaf_encoding is None:
            raise RuntimeError("Proximity structures are not initialized. Call fit(X, y) first.")

    def _build_leaf_encoding(self, leaves: np.ndarray) -> _LeafEncoding:
        n_trees = leaves.shape[1]
        offsets = np.zeros(n_trees, dtype=np.int64)
        values_per_tree: List[np.ndarray] = []

        total = 0
        for t in range(n_trees):
            uniq = np.unique(leaves[:, t])
            values_per_tree.append(uniq)
            offsets[t] = total
            total += uniq.shape[0]

        return _LeafEncoding(offsets=offsets, values_per_tree=values_per_tree, total_features=total)

    def _map_leaves_to_global(self, leaves: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Map `(sample, tree, leaf_id)` entries to sparse COO triplets.

        Returns
        -------
        rows, cols, tree_ids : np.ndarray
            Flattened arrays for sparse matrix construction.
        """
        if self._leaf_encoding is None:
            raise RuntimeError("Leaf encoding is not initialized.")

        n_samples, n_trees = leaves.shape
        all_rows: List[np.ndarray] = []
        all_cols: List[np.ndarray] = []
        all_trees: List[np.ndarray] = []

        for t in range(n_trees):
            known = self._leaf_encoding.values_per_tree[t]
            test_vals = leaves[:, t]

            pos = np.searchsorted(known, test_vals)
            in_range = pos < known.shape[0]
            eq = np.zeros_like(in_range, dtype=bool)
            idx_in = np.nonzero(in_range)[0]
            if idx_in.size > 0:
                eq[idx_in] = known[pos[idx_in]] == test_vals[idx_in]
            valid = in_range & eq
            if not np.any(valid):
                continue

            rows_t = np.nonzero(valid)[0]
            cols_t = self._leaf_encoding.offsets[t] + pos[valid]
            trees_t = np.full(rows_t.shape[0], t, dtype=np.int32)

            all_rows.append(rows_t)
            all_cols.append(cols_t.astype(np.int64))
            all_trees.append(trees_t)

        if not all_rows:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int32),
            )

        return (
            np.concatenate(all_rows).astype(np.int64),
            np.concatenate(all_cols).astype(np.int64),
            np.concatenate(all_trees),
        )

    def _build_W_matrix(self) -> None:
        if self.leaf_matrix_ is None or self._leaf_encoding is None:
            raise RuntimeError("Cannot build W before leaf structures are initialized.")

        n_samples, n_trees = self.leaf_matrix_.shape
        rows, cols, tree_ids = self._map_leaves_to_global(self.leaf_matrix_)
        if rows.size == 0:
            raise RuntimeError("No valid leaf mappings found while building W matrix.")

        total_cols = int(self._leaf_encoding.total_features)

        if self.prox_method == "original":
            weights = np.full(rows.shape[0], 1.0 / np.sqrt(n_trees), dtype=np.float32)

        elif self.prox_method == "oob":
            if self.oob_mask_ is None:
                raise RuntimeError("oob_mask_ must be available for prox_method='oob'.")

            keep = self.oob_mask_[rows, tree_ids] == 1
            rows = rows[keep]
            cols = cols[keep]
            tree_ids = tree_ids[keep]

            M = self.oob_mask_.sum(axis=1).astype(np.float32)
            M[M == 0] = 1.0
            weights = (np.sqrt(n_trees) / M[rows]).astype(np.float32)

        elif self.prox_method == "rfgap":
            if self.in_bag_counts_ is None:
                raise RuntimeError("in_bag_counts_ must be available for prox_method='rfgap'.")

            c_j_t = self.in_bag_counts_[rows, tree_ids].astype(np.float32)

            leaf_mass = np.bincount(
                cols,
                weights=c_j_t,
                minlength=total_cols,
            ).astype(np.float32)

            with np.errstate(divide="ignore", invalid="ignore"):
                inv_leaf_mass = 1.0 / leaf_mass
            inv_leaf_mass[~np.isfinite(inv_leaf_mass)] = 0.0

            weights = c_j_t * inv_leaf_mass[cols]

        else:
            raise ValueError(f"Unknown prox_method={self.prox_method}")

        if self.non_zero_diagonal and self.prox_method == "rfgap":
            if self.in_bag_counts_ is None:
                raise RuntimeError("in_bag_counts_ must be available for non-zero diagonal RF-GAP.")
            row_sums = np.bincount(rows, weights=weights, minlength=n_samples).astype(np.float32)
            denom = (self.in_bag_counts_ > 0).sum(axis=1).astype(np.float32)
            denom[denom == 0] = 1.0
            diag_vals = row_sums / denom

            self._diag_offset = total_cols
            diag_rows = np.arange(n_samples, dtype=np.int64)
            diag_cols = self._diag_offset + diag_rows

            rows = np.concatenate([rows, diag_rows])
            cols = np.concatenate([cols, diag_cols])
            weights = np.concatenate([weights, diag_vals.astype(np.float32)])
            total_cols += n_samples
        else:
            self._diag_offset = None

        keep_nonzero = weights > 0
        self.W_mat = sparse.csr_matrix(
            (weights[keep_nonzero], (rows[keep_nonzero], cols[keep_nonzero])),
            shape=(n_samples, total_cols),
            dtype=np.float32,
        )
        self._n_total_features = total_cols

    def _build_W_uniform(
        self, leaves: np.ndarray, encoding: _LeafEncoding,
    ) -> sparse.csr_matrix:
        """Build an rfgap-style W matrix with uniform in-bag counts (c_j=1).

        Used for RotationForest where trees are fitted on all data without
        bootstrap.  W[j, leaf] = 1 / m_leaf where m_leaf is the number of
        training samples in that leaf.
        """
        n_samples, n_trees = leaves.shape
        rows, cols, _ = self._map_leaves_to_global_with_encoding(leaves, encoding)
        if rows.size == 0:
            return sparse.csr_matrix(
                (n_samples, encoding.total_features), dtype=np.float32,
            )

        total_cols = encoding.total_features

        leaf_mass = np.bincount(cols, minlength=total_cols).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_leaf_mass = 1.0 / leaf_mass
        inv_leaf_mass[~np.isfinite(inv_leaf_mass)] = 0.0

        weights = inv_leaf_mass[cols]

        keep_nonzero = weights > 0
        return sparse.csr_matrix(
            (weights[keep_nonzero], (rows[keep_nonzero], cols[keep_nonzero])),
            shape=(n_samples, total_cols),
            dtype=np.float32,
        )

    def _build_Q_matrix_with_encoding(
        self,
        leaves: np.ndarray,
        encoding: _LeafEncoding,
        total_features: int,
    ) -> sparse.csr_matrix:
        """Build a Q matrix for test data using a given leaf encoding.

        Each test point is treated as OOB for every tree, so Q values are 1/T.
        """
        n_samples, n_trees = leaves.shape
        rows, cols, _ = self._map_leaves_to_global_with_encoding(leaves, encoding)
        vals = np.full(rows.shape[0], 1.0 / n_trees, dtype=np.float32)

        return sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(n_samples, total_features),
            dtype=np.float32,
        )

    def _map_leaves_to_global_with_encoding(
        self,
        leaves: np.ndarray,
        encoding: _LeafEncoding,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map leaf indices to global column indices using a given encoding."""
        n_samples, n_trees = leaves.shape
        all_rows: List[np.ndarray] = []
        all_cols: List[np.ndarray] = []
        all_trees: List[np.ndarray] = []

        for t in range(n_trees):
            known = encoding.values_per_tree[t]
            test_vals = leaves[:, t]

            pos = np.searchsorted(known, test_vals)
            in_range = pos < known.shape[0]
            eq = np.zeros_like(in_range, dtype=bool)
            idx_in = np.nonzero(in_range)[0]
            if idx_in.size > 0:
                eq[idx_in] = known[pos[idx_in]] == test_vals[idx_in]
            valid = in_range & eq
            if not np.any(valid):
                continue

            rows_t = np.nonzero(valid)[0]
            cols_t = encoding.offsets[t] + pos[valid]
            trees_t = np.full(rows_t.shape[0], t, dtype=np.int32)

            all_rows.append(rows_t)
            all_cols.append(cols_t.astype(np.int64))
            all_trees.append(trees_t)

        if not all_rows:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int32),
            )

        return (
            np.concatenate(all_rows).astype(np.int64),
            np.concatenate(all_cols).astype(np.int64),
            np.concatenate(all_trees),
        )

    def _build_Q_matrix(self, leaves: np.ndarray, is_training: bool) -> sparse.csr_matrix:
        if self._leaf_encoding is None or self._n_total_features is None:
            raise RuntimeError("Cannot build Q before leaf encoding and W are initialized.")

        n_samples, n_trees = leaves.shape
        rows, cols, tree_ids = self._map_leaves_to_global(leaves)

        if self.prox_method == "original":
            vals = np.full(rows.shape[0], 1.0 / np.sqrt(n_trees), dtype=np.float32)

        elif self.prox_method == "oob":
            vals = np.full(rows.shape[0], np.sqrt(n_trees) / n_trees, dtype=np.float32)

        elif self.prox_method == "rfgap":
            if is_training:
                if self.oob_mask_ is None:
                    raise RuntimeError("oob_mask_ must be available for prox_method='rfgap'.")

                keep = self.oob_mask_[rows, tree_ids] == 1
                rows = rows[keep]
                cols = cols[keep]

                S_i = self.oob_mask_.sum(axis=1).astype(np.float32)
                S_i[S_i == 0] = 1.0
                vals = (1.0 / S_i[rows]).astype(np.float32)
            else:
                vals = np.full(rows.shape[0], 1.0 / n_trees, dtype=np.float32)

        else:
            raise ValueError(f"Unknown prox_method={self.prox_method}")

        if self.non_zero_diagonal and self.prox_method == "rfgap" and is_training:
            if self._diag_offset is None:
                raise RuntimeError("Diagonal offset is not initialized.")
            diag_rows = np.arange(n_samples, dtype=np.int64)
            diag_cols = self._diag_offset + diag_rows
            diag_vals = np.ones(n_samples, dtype=np.float32)
            rows = np.concatenate([rows, diag_rows])
            cols = np.concatenate([cols, diag_cols])
            vals = np.concatenate([vals, diag_vals])

        return sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(n_samples, self._n_total_features),
            dtype=np.float32,
        )

    def _rotation_forest_apply_surrogates(self, X: Any) -> np.ndarray:
        surrogate_estimators = getattr(self, "_oob_surrogate_estimators_", None)
        if surrogate_estimators is None:
            raise RuntimeError("RotationForest surrogate estimators are not initialized.")

        with _suppress_base_validate_data_futurewarning():
            X_proc = self._check_X(X)
            X_proc = validate_data(self, X=X_proc, reset=False, accept_sparse=False)
            X_proc = X_proc[:, self._useful_atts]
            X_proc = (X_proc - self._min) / self._ptp

            leaves: List[np.ndarray] = []
            for t, surrogate in enumerate(surrogate_estimators):
                pcas = self._pcas[t]
                groups = self._groups[t]

                X_t = np.concatenate(
                    [pcas[j].transform(X_proc[:, group]) for j, group in enumerate(groups)],
                    axis=1,
                )
                X_t = X_t.astype(np.float32)
                X_t = np.nan_to_num(
                    X_t,
                    copy=False,
                    nan=0.0,
                    posinf=float(np.finfo(np.float32).max),
                    neginf=float(np.finfo(np.float32).min),
                )

                leaves.append(surrogate.apply(X_t))

        return np.column_stack(leaves)

    def _compute_bootstrap_stats(
        self,
        n_samples: int,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute OOB mask and in-bag multiplicities for each sample/tree.

        Returns
        -------
        oob_mask : np.ndarray, shape (n_samples, n_estimators)
            1 when sample is OOB for a tree, 0 otherwise.
        in_bag_counts : np.ndarray, shape (n_samples, n_estimators)
            Number of bootstrap draws of sample for each tree.
        """
        estimators = getattr(self, "estimators_", None)
        if estimators is None:
            raise RuntimeError("Estimator must expose estimators_ after fit.")

        n_trees = len(estimators)
        oob_mask = np.zeros((n_samples, n_trees), dtype=np.int8)
        in_bag_counts = np.zeros((n_samples, n_trees), dtype=np.float32)

        if _is_rotation_forest_instance(self):
            if X is not None and y is not None:
                try:
                    train_estimate_fn = getattr(self, "_train_probas_for_estimator", None)
                    if train_estimate_fn is None:
                        train_estimate_fn = getattr(self, "_train_preds_for_estimator", None)

                    if callable(train_estimate_fn):
                        y_arr = np.asarray(y)

                        with _suppress_base_validate_data_futurewarning():
                            X_proc = self._check_X(X)
                            X_proc = validate_data(self, X=X_proc, reset=False, accept_sparse=False)
                            X_proc = X_proc[:, self._useful_atts]
                            X_proc = (X_proc - self._min) / self._ptp

                            X_t = [
                                np.nan_to_num(
                                    np.concatenate(
                                        [
                                            self._pcas[t][g_idx].transform(X_proc[:, group])
                                            for g_idx, group in enumerate(self._groups[t])
                                        ],
                                        axis=1,
                                    ).astype(np.float32),
                                    copy=False,
                                    nan=0.0,
                                    posinf=float(np.finfo(np.float32).max),
                                    neginf=float(np.finfo(np.float32).min),
                                )
                                for t in range(n_trees)
                            ]

                        n_cases = int(getattr(self, "n_cases_", n_samples))
                        if n_cases != n_samples:
                            raise ValueError("RotationForest n_cases_ does not match fit sample size.")

                        rng = check_random_state(getattr(self, "random_state", None))
                        max_int = np.iinfo(np.int32).max
                        surrogate_estimators: List[Any] = []
                        surrogate_leaves: List[np.ndarray] = []

                        base_estimator = getattr(self, "_base_estimator", None)

                        for t in range(n_trees):
                            est_seed = rng.randint(max_int)
                            est_rng = check_random_state(est_seed)

                            # Pull OOB indices from RotationForest's own
                            # train-estimate routine so masks match the
                            # original model definition exactly.
                            with _suppress_base_validate_data_futurewarning():
                                train_result = train_estimate_fn(
                                    X_t,
                                    y_arr,
                                    t,
                                    check_random_state(est_seed),
                                )
                            if not isinstance(train_result, (tuple, list)) or len(train_result) != 2:
                                raise RuntimeError("RotationForest train-estimate function returned an unexpected result.")
                            oob_idx_model = train_result[1]
                            oob_idx_model = np.asarray(oob_idx_model, dtype=np.int64)

                            sampled = est_rng.choice(n_cases, size=n_cases, replace=True)

                            if base_estimator is not None:
                                surrogate = clone(base_estimator)
                            else:
                                surrogate = clone(self.estimators_[t])
                            if hasattr(surrogate, "random_state"):
                                surrogate.random_state = est_rng
                            surrogate.fit(X_t[t][sampled], y_arr[sampled])
                            surrogate_estimators.append(surrogate)
                            surrogate_leaves.append(np.asarray(surrogate.apply(X_t[t])))

                            idx, cnt = np.unique(sampled, return_counts=True)
                            in_bag_counts[idx, t] = cnt.astype(np.float32)
                            oob_mask[:, t] = 0
                            oob_mask[oob_idx_model, t] = 1

                        self._oob_surrogate_estimators_ = surrogate_estimators
                        self._oob_surrogate_leaf_matrix_ = np.column_stack(surrogate_leaves)

                        return oob_mask, in_bag_counts
                except Exception:
                    raise

            rng = check_random_state(getattr(self, "random_state", None))
            max_int = np.iinfo(np.int32).max

            for t in range(n_trees):
                est_rng = check_random_state(rng.randint(max_int))
                sampled = est_rng.choice(n_samples, size=n_samples, replace=True)

                idx, cnt = np.unique(sampled, return_counts=True)
                in_bag_counts[idx, t] = cnt.astype(np.float32)
                oob_mask[:, t] = 1
                oob_mask[idx, t] = 0

            return oob_mask, in_bag_counts

        estimators_samples = getattr(self, "estimators_samples_", None)
        if estimators_samples is not None:
            for t, sample_indices in enumerate(estimators_samples):
                idx, cnt = np.unique(np.asarray(sample_indices), return_counts=True)
                in_bag_counts[idx, t] = cnt.astype(np.float32)
                oob_mask[:, t] = 1
                oob_mask[idx, t] = 0
            return oob_mask, in_bag_counts

        get_indices_fn = getattr(self, "_get_estimators_indices", None)
        if callable(get_indices_fn):
            try:
                for t, estimators_index in enumerate(cast(Any, get_indices_fn()) or []):
                    sample_indices = estimators_index
                    if isinstance(estimators_index, tuple) and len(estimators_index) == 2:
                        sample_indices = estimators_index[1]

                    idx, cnt = np.unique(np.asarray(sample_indices), return_counts=True)
                    in_bag_counts[idx, t] = cnt.astype(np.float32)
                    oob_mask[:, t] = 1
                    oob_mask[idx, t] = 0
                return oob_mask, in_bag_counts
            except AttributeError:
                # Some sktree forests (e.g. UnsupervisedRandomForest) expose
                # _get_estimators_indices but lack _n_samples; fall through to
                # the _generate_sample_indices fallback.
                oob_mask[:] = 0
                in_bag_counts[:] = 0

        if _generate_sample_indices is None or _generate_unsampled_indices is None:
            raise RuntimeError(
                "Bootstrap internals unavailable. For non-sklearn estimators, override "
                "_compute_bootstrap_stats to provide OOB and in-bag counts."
            )

        for t, est in enumerate(estimators):
            if not hasattr(est, "random_state"):
                raise RuntimeError(
                    "Each base estimator must expose random_state, or override "
                    "_compute_bootstrap_stats for your model class."
                )

            rs = est.random_state
            sampled = _generate_sample_indices(rs, n_samples, n_samples)
            unsampled = _generate_unsampled_indices(rs, n_samples, n_samples)

            idx, cnt = np.unique(sampled, return_counts=True)
            in_bag_counts[idx, t] = cnt.astype(np.float32)
            oob_mask[unsampled, t] = 1

        return oob_mask, in_bag_counts

    @staticmethod
    def _csr_row_scale_inplace(A: sparse.csr_matrix, scale: np.ndarray) -> None:
        if not sparse.isspmatrix_csr(A):
            raise ValueError("Matrix must be CSR for in-place row scaling.")
        scale = np.asarray(scale, dtype=A.data.dtype)
        nnz_per_row = np.diff(A.indptr)
        A.data *= np.repeat(scale, nnz_per_row)

    @staticmethod
    def _block_symmetrize(Q: sparse.csr_matrix, W: sparse.csr_matrix, mode: Literal["arithmetic", "geometric"]) -> sparse.csr_matrix:
        if mode == "arithmetic":
            left = sparse.hstack([Q, W], format="csr", dtype=np.float32)
            right_t = sparse.vstack([W.T, Q.T], format="csc", dtype=np.float32)
            prod = left.dot(right_t)
            P = sparse.csr_matrix(prod, dtype=np.float32)
            P.data *= 0.5
            return P

        if mode == "geometric":
            P_asym = Q.dot(W.T)
            P = P_asym.multiply(P_asym.T)
            np.sqrt(P.data, out=P.data)
            return cast(sparse.csr_matrix, P.tocsr())

        raise ValueError(f"Unknown symmetrization mode: {mode}")


def with_sparse_proximity(base_cls: Type[TForest]) -> Type[Any]:
    """
    Class decorator/factory to attach `SparseProximityMixin` to an estimator class.

    Can also be called with an **instance** for convenience; in that case a new
    ``ProximityEnabled`` class is built from the instance's class and a fresh
    instance is returned with the same parameters.

    Parameters
    ----------
    base_cls : type or estimator instance
        Forest-like estimator class (or instance) to extend.

    Returns
    -------
    type or estimator instance
        New class inheriting `(SparseProximityMixin, base_cls)`, or an instance
        of that class when an instance was passed.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from forestgap import with_sparse_proximity
    >>> ProxRF = with_sparse_proximity(RandomForestClassifier)
    >>> model = ProxRF(n_estimators=200, prox_method='rfgap', bootstrap=True)

    As a decorator:
    >>> @with_sparse_proximity
    ... class ProxExtraTrees(ExtraTreesClassifier):
    ...     pass

    With an already-constructed instance:
    >>> model = with_sparse_proximity(RandomForestClassifier(n_estimators=200))
    """

    # If an instance was passed, wrap its class and return a new instance
    # with the same parameters.
    if not isinstance(base_cls, type):
        instance = base_cls
        cls = with_sparse_proximity(type(instance))
        return cls(**instance.get_params())

    is_rotation_forest = _is_rotation_forest_class(base_cls)
    is_isolation_forest = _is_isolation_forest_class(base_cls)
    needs_rotation_apply_adapter = is_rotation_forest and not callable(getattr(base_cls, "apply", None))
    needs_isolation_apply_adapter = is_isolation_forest and not callable(getattr(base_cls, "apply", None))

    wrapped_base: Type[Any] = base_cls
    if is_rotation_forest:
        if needs_rotation_apply_adapter:
            class RotationForestWithAdapters(_RotationForestOOBAdapter, _RotationForestApplyAdapter, base_cls):
                pass
        else:
            class RotationForestWithAdapters(_RotationForestOOBAdapter, base_cls):
                pass

        RotationForestWithAdapters.__name__ = f"{base_cls.__name__}WithApplyAdapter"
        RotationForestWithAdapters.__qualname__ = RotationForestWithAdapters.__name__
        wrapped_base = RotationForestWithAdapters
    elif is_isolation_forest and needs_isolation_apply_adapter:
        class IsolationForestWithAdapters(_IsolationForestApplyAdapter, base_cls):
            pass

        IsolationForestWithAdapters.__name__ = f"{base_cls.__name__}WithApplyAdapter"
        IsolationForestWithAdapters.__qualname__ = IsolationForestWithAdapters.__name__
        wrapped_base = IsolationForestWithAdapters

    class ProximityEnabled(SparseProximityMixin, wrapped_base):
        def __init__(
            self,
            *args: Any,
            prox_method: ProxMethod = "rfgap",
            matrix_type: MatrixType = "sparse",
            non_zero_diagonal: bool = False,
            symm_mode: SymmMode = None,
            max_normalize: bool = False,
            **kwargs: Any,
        ) -> None:
            base_sig = inspect.signature(base_cls.__init__)
            has_oob_param = "oob_score" in base_sig.parameters

            # Determine wrapper-level oob_score request with sane defaults.
            if "oob_score" in kwargs:
                requested_oob_score = bool(kwargs["oob_score"])
            elif has_oob_param and base_sig.parameters["oob_score"].default is not inspect._empty:
                requested_oob_score = bool(base_sig.parameters["oob_score"].default)
            elif is_rotation_forest:
                requested_oob_score = True
            else:
                requested_oob_score = False

            # Support user-passed oob_score even for classes that do not natively
            # expose it (e.g., aeon RotationForest).
            if not has_oob_param and "oob_score" in kwargs:
                kwargs.pop("oob_score")

            # Quietly disable incompatible bootstrap/oob combinations instead of
            # propagating model-specific ValueError in adapter usage.
            effective_oob_score = requested_oob_score
            if has_oob_param:
                bootstrap_param = base_sig.parameters.get("bootstrap")
                bootstrap_val = kwargs.get("bootstrap", None)
                if bootstrap_val is None and bootstrap_param is not None and bootstrap_param.default is not inspect._empty:
                    bootstrap_val = bootstrap_param.default

                if requested_oob_score and bootstrap_val is False:
                    kwargs["oob_score"] = False
                    effective_oob_score = False
                elif "oob_score" in kwargs:
                    kwargs["oob_score"] = bool(kwargs["oob_score"])
                    effective_oob_score = bool(kwargs["oob_score"])

            # IsolationForest does not provide a meaningful supervised-style OOB
            # score path for this mixin; keep it as a quiet no-op flag.
            if is_isolation_forest and has_oob_param and "oob_score" not in kwargs:
                kwargs["oob_score"] = False
                effective_oob_score = False

            # sktree ForestCluster (UnsupervisedRandomForest) has a buggy
            # _set_oob_score_and_attributes; always disable the base model's
            # oob_score and let the mixin handle OOB computation itself.
            _is_forest_cluster = any(
                c.__name__ == "ForestCluster" for c in base_cls.mro()
            )
            if _is_forest_cluster and has_oob_param:
                kwargs["oob_score"] = False
                effective_oob_score = False

            bound = base_sig.bind_partial(self, *args, **kwargs)
            base_cls.__init__(*bound.args, **bound.kwargs)
            self._init_sparse_proximity(
                prox_method=prox_method,
                matrix_type=matrix_type,
                non_zero_diagonal=non_zero_diagonal,
                symm_mode=symm_mode,
                max_normalize=max_normalize,
            )
            if is_rotation_forest:
                self._rotation_oob_score = bool(effective_oob_score)
            if is_isolation_forest:
                self._isolation_oob_score = bool(requested_oob_score)
                effective_oob_score = False
            self.oob_score = bool(effective_oob_score)

    base_sig = inspect.signature(base_cls.__init__)
    base_params = [p for n, p in base_sig.parameters.items() if n != "self"]
    prox_params = [
        Parameter("prox_method", kind=Parameter.KEYWORD_ONLY, default="rfgap"),
        Parameter("matrix_type", kind=Parameter.KEYWORD_ONLY, default="sparse"),
        Parameter("non_zero_diagonal", kind=Parameter.KEYWORD_ONLY, default=False),
        Parameter("symm_mode", kind=Parameter.KEYWORD_ONLY, default=None),
        Parameter("max_normalize", kind=Parameter.KEYWORD_ONLY, default=False),
    ]
    if (is_rotation_forest or is_isolation_forest) and "oob_score" not in base_sig.parameters:
        prox_params.append(Parameter("oob_score", kind=Parameter.KEYWORD_ONLY, default=True))
    # Insert prox_params before any VAR_KEYWORD (**kwargs) parameter to maintain
    # valid Python parameter ordering.
    var_kw = [p for p in base_params if p.kind == Parameter.VAR_KEYWORD]
    non_var_kw = [p for p in base_params if p.kind != Parameter.VAR_KEYWORD]
    ordered_params = [Parameter("self", kind=Parameter.POSITIONAL_OR_KEYWORD)] + non_var_kw + prox_params + var_kw
    init_sig = Signature(parameters=ordered_params)
    ProximityEnabled.__init__.__signature__ = init_sig

    ProximityEnabled.__name__ = f"{base_cls.__name__}WithSparseProximity"
    ProximityEnabled.__qualname__ = ProximityEnabled.__name__
    return ProximityEnabled


class _RotationForestApplyAdapter:
    # def apply(self, X):
    #     X_proc = self._check_X(X)
    #     X_proc = validate_data(self, X=X_proc, reset=False, accept_sparse=False)
    #     X_proc = X_proc[:, self._useful_atts]
    #     X_proc = (X_proc - self._min) / self._ptp

    #     leaves = []
    #     for i in range(self._n_estimators):
    #         estimator = self.estimators_[i]
    #         pcas = self._pcas[i]
    #         groups = self._groups[i]

    #         X_t = np.concatenate(
    #             [pcas[j].transform(X_proc[:, group]) for j, group in enumerate(groups)],
    #             axis=1,
    #         )
    #         X_t = X_t.astype(np.float32)
    #         X_t = np.nan_to_num(
    #             X_t,
    #             copy=False,
    #             nan=0.0,
    #             posinf=float(np.finfo(np.float32).max),
    #             neginf=float(np.finfo(np.float32).min),
    #         )
    #         leaves.append(estimator.apply(X_t))

    #     return np.column_stack(leaves)

    # CHAT Version
    def apply(self, X):
        """
        Return terminal node indices for each sample and each estimator.

        Returns
        -------
        nodes : np.ndarray, shape=(n_samples, n_estimators)
            nodes[i, j] is the terminal node id for sample i in estimator j.
        """
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            from sklearn.exceptions import NotFittedError
            raise NotFittedError("This RotationForestClassifier instance is not fitted yet.")

        # same input checks / preprocessing used in predict_proba
        with _suppress_base_validate_data_futurewarning():
            X = self._check_X(X)
            X = validate_data(self, X=X, reset=False, accept_sparse=False)

        # remove attributes that were dropped at fit
        X = X[:, self._useful_atts]

        # normalize using stored min/ptp from training
        X = (X - self._min) / self._ptp

        # for each estimator, build its transformed feature matrix and call .apply()
        n_samples = X.shape[0]
        n_estimators = self._n_estimators
        nodes = np.empty((n_samples, n_estimators), dtype=np.int64)

        # optional: use joblib to parallelize if you want speed
        from joblib import Parallel, delayed

        def _apply_est(i):
            pcas = self._pcas[i]      # list of PCA objects used for estimator i
            groups = self._groups[i]  # list of attribute index arrays for estimator i
            # reconstruct transformed features for this estimator
            X_t = np.concatenate([pcas[j].transform(X[:, grp]) for j, grp in enumerate(groups)], axis=1)
            X_t = X_t.astype(np.float32)
            X_t = np.nan_to_num(X_t, False, 0, np.finfo(np.float32).max, np.finfo(np.float32).min)
            return self.estimators_[i].apply(X_t)

        results = Parallel(n_jobs=self._n_jobs, prefer="threads")(
            delayed(_apply_est)(i) for i in range(n_estimators)
        )

        # results is list of length n_estimators, each shape (n_samples,)
        # stack as (n_samples, n_estimators)
        nodes = np.column_stack(results)
        return nodes


class _IsolationForestApplyAdapter:
    def apply(self, X: Any) -> np.ndarray:
        X_checked = validate_data(self, X=X, reset=False, accept_sparse=True)

        estimators = getattr(self, "estimators_", None)
        if estimators is None:
            raise RuntimeError("IsolationForest estimators_ are not initialized.")

        features_per_estimator = getattr(self, "estimators_features_", None)
        if features_per_estimator is None:
            raise RuntimeError("IsolationForest estimators_features_ are not initialized.")

        leaves: List[np.ndarray] = []
        for est, features in zip(estimators, features_per_estimator):
            X_sub = X_checked[:, features]
            leaves.append(np.asarray(est.apply(X_sub)))

        return np.column_stack(leaves)

class _RotationForestOOBAdapter:
    def _set_rotation_forest_oob_attributes(self, X: Any, y: Optional[Any]) -> None:
        if not bool(getattr(self, "_rotation_oob_score", True)):
            return

        if y is None:
            return
        y_arr = np.asarray(y)

        try:
            train_proba_fn = getattr(self, "_train_probas_for_estimator", None)
            train_pred_fn = getattr(self, "_train_preds_for_estimator", None)
            n_trees = int(getattr(self, "_n_estimators", 0))

            if (
                hasattr(self, "estimators_")
                and n_trees > 0
                and hasattr(self, "_pcas")
                and hasattr(self, "_groups")
                and hasattr(self, "_check_X")
                and (callable(train_proba_fn) or callable(train_pred_fn))
            ):
                with _suppress_base_validate_data_futurewarning():
                    X_proc = self._check_X(X)
                    X_proc = validate_data(self, X=X_proc, reset=False, accept_sparse=False)
                    X_proc = X_proc[:, self._useful_atts]
                    X_proc = (X_proc - self._min) / self._ptp

                    X_t = [
                        np.nan_to_num(
                            np.concatenate(
                                [pcas[j].transform(X_proc[:, group]) for j, group in enumerate(groups)],
                                axis=1,
                            ).astype(np.float32),
                            copy=False,
                            nan=0.0,
                            posinf=float(np.finfo(np.float32).max),
                            neginf=float(np.finfo(np.float32).min),
                        )
                        for pcas, groups in zip(self._pcas, self._groups)
                    ]

                rng = check_random_state(getattr(self, "random_state", None))
                max_int = np.iinfo(np.int32).max
                n_samples = X_proc.shape[0]
                surrogate_estimators = getattr(self, "_oob_surrogate_estimators_", None)
                oob_mask = getattr(self, "oob_mask_", None)

                if surrogate_estimators is not None and oob_mask is not None:
                    divisors = oob_mask.sum(axis=1).astype(np.float64)

                    if hasattr(self, "classes_"):
                        classes = np.asarray(self.classes_)
                        class_to_col = {cls: i for i, cls in enumerate(classes)}
                        results = np.zeros((n_samples, classes.shape[0]), dtype=np.float64)

                        for t, surrogate in enumerate(surrogate_estimators):
                            oob_idx = np.flatnonzero(oob_mask[:, t] == 1)
                            if oob_idx.size == 0:
                                continue

                            probas = np.asarray(surrogate.predict_proba(X_t[t][oob_idx]), dtype=np.float64)
                            mapped = np.zeros((oob_idx.size, classes.shape[0]), dtype=np.float64)
                            surrogate_classes = np.asarray(getattr(surrogate, "classes_", classes))
                            for j, cls in enumerate(surrogate_classes):
                                mapped[:, class_to_col[cls]] = probas[:, j]

                            results[oob_idx] += mapped

                        if (divisors == 0).any():
                            warn(
                                (
                                    "Some inputs do not have OOB scores. This probably means "
                                    "too few trees were used to compute any reliable OOB "
                                    "estimates."
                                ),
                                UserWarning,
                            )
                            divisors[divisors == 0] = 1.0

                        self.oob_decision_function_ = results / divisors[:, None]
                        y_pred = classes[np.argmax(self.oob_decision_function_, axis=1)]
                        self.oob_score_ = float(accuracy_score(y_arr, y_pred))
                        return

                    results = np.zeros(n_samples, dtype=np.float64)
                    for t, surrogate in enumerate(surrogate_estimators):
                        oob_idx = np.flatnonzero(oob_mask[:, t] == 1)
                        if oob_idx.size == 0:
                            continue
                        results[oob_idx] += np.asarray(surrogate.predict(X_t[t][oob_idx]), dtype=np.float64)

                    if (divisors == 0).any():
                        warn(
                            (
                                "Some inputs do not have OOB scores. This probably means "
                                "too few trees were used to compute any reliable OOB "
                                "estimates."
                            ),
                            UserWarning,
                        )
                        divisors[divisors == 0] = 1.0

                    self.oob_prediction_ = results / divisors
                    self.oob_score_ = float(r2_score(y_arr, self.oob_prediction_))
                    return

                if hasattr(self, "classes_") and callable(train_proba_fn):
                    with _suppress_base_validate_data_futurewarning():
                        p = [
                            train_proba_fn(
                                X_t,
                                y_arr,
                                t,
                                check_random_state(rng.randint(max_int)),
                            )
                            for t in range(n_trees)
                        ]
                    y_probas, oobs = zip(*p)

                    results = np.sum(y_probas, axis=0)
                    divisors = np.zeros(n_samples, dtype=np.float64)
                    for oob in oobs:
                        for inst in oob:
                            divisors[inst] += 1.0

                    if (divisors == 0).any():
                        warn(
                            (
                                "Some inputs do not have OOB scores. This probably means "
                                "too few trees were used to compute any reliable OOB "
                                "estimates."
                            ),
                            UserWarning,
                        )
                        divisors[divisors == 0] = 1.0

                    results = results / divisors[:, None]

                    classes = np.asarray(self.classes_)
                    self.oob_decision_function_ = np.asarray(results, dtype=np.float64)
                    y_pred = classes[np.argmax(self.oob_decision_function_, axis=1)]
                    self.oob_score_ = float(accuracy_score(y_arr, y_pred))
                    return

                if callable(train_pred_fn):
                    with _suppress_base_validate_data_futurewarning():
                        p = [
                            train_pred_fn(
                                X_t,
                                y_arr,
                                t,
                                check_random_state(rng.randint(max_int)),
                            )
                            for t in range(n_trees)
                        ]
                    y_preds, oobs = zip(*p)

                    results = np.sum(y_preds, axis=0)
                    divisors = np.zeros(n_samples, dtype=np.float64)
                    for oob in oobs:
                        for inst in oob:
                            divisors[inst] += 1.0

                    if (divisors == 0).any():
                        warn(
                            (
                                "Some inputs do not have OOB scores. This probably means "
                                "too few trees were used to compute any reliable OOB "
                                "estimates."
                            ),
                            UserWarning,
                        )
                        divisors[divisors == 0] = 1.0

                    results = results / divisors

                    self.oob_prediction_ = np.asarray(results, dtype=np.float64)
                    self.oob_score_ = float(r2_score(y_arr, self.oob_prediction_))
                    return
        except Exception:
            raise

        try:
            oob_model = clone(self)

            if hasattr(self, "classes_") and hasattr(oob_model, "fit_predict_proba"):
                with _suppress_base_validate_data_futurewarning():
                    oob_proba = np.asarray(oob_model.fit_predict_proba(X, y_arr), dtype=np.float64)
                classes = np.asarray(getattr(oob_model, "classes_", self.classes_))

                self.oob_decision_function_ = oob_proba
                pred_idx = np.argmax(oob_proba, axis=1)
                y_pred = classes[pred_idx]
                self.oob_score_ = float(accuracy_score(y_arr, y_pred))
                return

            if hasattr(oob_model, "fit_predict"):
                with _suppress_base_validate_data_futurewarning():
                    oob_prediction = np.asarray(oob_model.fit_predict(X, y_arr), dtype=np.float64)
                self.oob_prediction_ = oob_prediction
                self.oob_score_ = float(r2_score(y_arr, oob_prediction))
                return
        except Exception:
            raise
