from __future__ import annotations

from dataclasses import dataclass
import inspect
from inspect import Parameter, Signature
from typing import Any, List, Literal, Optional, Protocol, Sequence, Tuple, Type, TypeVar, Union, cast, runtime_checkable

import numpy as np
from scipy import sparse
from sklearn.utils.validation import check_is_fitted, validate_data

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
        if y is None:
            base_fit(X, **fit_kwargs)
        else:
            base_fit(X, y, **fit_kwargs)

        self.prox_fit(X)
        return self

    def prox_fit(self, X: Any) -> None:
        """
        Build sparse factors required for proximity computations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data used during model fitting.
        """
        check_is_fitted(cast(Any, self))

        apply_fn = getattr(self, "apply", None)
        if apply_fn is None:
            raise RuntimeError("Estimator must implement apply(X) to use proximity methods.")
        leaves = np.asarray(apply_fn(X))
        if leaves.ndim != 2:
            raise ValueError("apply(X) must return a 2D array of shape (n_samples, n_estimators)")

        n_samples, n_trees = leaves.shape
        self._n_train_samples = n_samples
        self.leaf_matrix_ = leaves
        self._leaf_encoding = self._build_leaf_encoding(leaves)

        if self.prox_method in ("oob", "rfgap"):
            oob_mask, in_bag_counts = self._compute_bootstrap_stats(n_samples)
            self.oob_mask_ = oob_mask.astype(np.int8)
            self.in_bag_counts_ = in_bag_counts.astype(np.float32)

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
        Compute proximities from new samples to fitted/training samples.

        Parameters
        ----------
        X_new : array-like of shape (n_samples_new, n_features)
            New observations.

        Returns
        -------
        scipy.sparse.csr_matrix or numpy.ndarray
            `(n_samples_new, n_train)` proximity matrix.
        """
        self._check_proximity_ready()

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

    def _compute_bootstrap_stats(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
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

        estimators_samples = getattr(self, "estimators_samples_", None)
        if estimators_samples is not None:
            for t, sample_indices in enumerate(estimators_samples):
                idx, cnt = np.unique(np.asarray(sample_indices), return_counts=True)
                in_bag_counts[idx, t] = cnt.astype(np.float32)
                oob_mask[:, t] = 1
                oob_mask[idx, t] = 0
            return oob_mask, in_bag_counts

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

    Parameters
    ----------
    base_cls : type
        Forest-like estimator class to extend.

    Returns
    -------
    type
        New class inheriting `(SparseProximityMixin, base_cls)`.

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
    """

    needs_rotation_adapter = not callable(getattr(base_cls, "apply", None)) and any(
        "RotationForest" in cls.__name__ for cls in base_cls.mro()
    )

    wrapped_base: Type[Any] = base_cls
    if needs_rotation_adapter:
        class RotationForestWithApply(_RotationForestApplyAdapter, base_cls):
            pass

        RotationForestWithApply.__name__ = f"{base_cls.__name__}WithApplyAdapter"
        RotationForestWithApply.__qualname__ = RotationForestWithApply.__name__
        wrapped_base = RotationForestWithApply

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
            bound = base_sig.bind_partial(self, *args, **kwargs)
            base_cls.__init__(*bound.args, **bound.kwargs)
            self._init_sparse_proximity(
                prox_method=prox_method,
                matrix_type=matrix_type,
                non_zero_diagonal=non_zero_diagonal,
                symm_mode=symm_mode,
                max_normalize=max_normalize,
            )

    base_sig = inspect.signature(base_cls.__init__)
    base_params = [p for n, p in base_sig.parameters.items() if n != "self"]
    prox_params = [
        Parameter("prox_method", kind=Parameter.KEYWORD_ONLY, default="rfgap"),
        Parameter("matrix_type", kind=Parameter.KEYWORD_ONLY, default="sparse"),
        Parameter("non_zero_diagonal", kind=Parameter.KEYWORD_ONLY, default=False),
        Parameter("symm_mode", kind=Parameter.KEYWORD_ONLY, default=None),
        Parameter("max_normalize", kind=Parameter.KEYWORD_ONLY, default=False),
    ]
    init_sig = Signature(parameters=[Parameter("self", kind=Parameter.POSITIONAL_OR_KEYWORD)] + base_params + prox_params)
    ProximityEnabled.__init__.__signature__ = init_sig

    ProximityEnabled.__name__ = f"{base_cls.__name__}WithSparseProximity"
    ProximityEnabled.__qualname__ = ProximityEnabled.__name__
    return ProximityEnabled


class _RotationForestApplyAdapter:
    def apply(self, X):
        X_proc = self._check_X(X)
        X_proc = validate_data(self, X=X_proc, reset=False, accept_sparse=False)
        X_proc = X_proc[:, self._useful_atts]
        X_proc = (X_proc - self._min) / self._ptp

        leaves = []
        for i in range(self._n_estimators):
            estimator = self.estimators_[i]
            pcas = self._pcas[i]
            groups = self._groups[i]

            X_t = np.concatenate(
                [pcas[j].transform(X_proc[:, group]) for j, group in enumerate(groups)],
                axis=1,
            )
            X_t = X_t.astype(np.float32)
            X_t = np.nan_to_num(
                X_t,
                copy=False,
                nan=0.0,
                posinf=np.finfo(np.float32).max,
                neginf=np.finfo(np.float32).min,
            )
            leaves.append(estimator.apply(X_t))

        return np.column_stack(leaves)
