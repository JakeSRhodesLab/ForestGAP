"""Composition-based forest proximity construction and relation dispatch."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Protocol, Sequence, Tuple, Union, cast, runtime_checkable
import warnings
from warnings import warn

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

import pandas as pd

from forestgap.adapters import (
    BOOSTED_FAMILIES,
    BoostedTreeAdapter,
    EstimatorCapabilities,
    UnsupportedRelationError,
    capabilities_for,
    make_boosted_adapter,
    validate_relation,
)
from forestgap.adapters.aeon import (
    rotation_forest_decision_path,
    rotation_forest_leaf_ids,
    rotation_forest_node_depths,
    rotation_forest_transforms,
)
from forestgap.adapters.sklearn import isolation_leaf_ids
from forestgap.manifold.proximity import (
    is_symmetric,
    prepare_proximity_affinity,
    proximity_to_distance,
    symmetrize_affinity,
)

__all__ = [
    "Proximity",
    "UnsupportedRelationError",
    "is_symmetric",
    "prepare_proximity_affinity",
    "proximity_to_distance",
    "symmetrize_affinity",
]




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
ProxMethod = Literal[
    "original",
    "oob",
    "rfgap",
    "honest_gap",
    "kerf",
    "boosted",
    "path",
    "englund",
    "random_height",
    "collision",
    "conditional",
    "survival_leaf",
]
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


def _is_honest_forest_instance(obj: Any) -> bool:
    if any("HonestForest" in cls.__name__ for cls in type(obj).mro()):
        return True
    return (
        all(
            hasattr(obj, attr)
            for attr in ("honest_indices_", "structure_indices_", "estimators_", "classes_")
        )
        and callable(getattr(obj, "predict_proba", None))
    )


@runtime_checkable
class ForestLike(Protocol):
    estimators_: Sequence[Any]
    n_estimators: int

    def apply(self, X: Any) -> np.ndarray:
        ...


@dataclass
class _LeafEncoding:
    """Compact leaf-to-column encoding for sparse proximity construction."""

    offsets: np.ndarray
    values_per_tree: List[np.ndarray]
    total_features: int


class Proximity:
    """
    Composition-based sparse forest relation index.

    This mixin is designed for forest-like estimators that expose:
    - `fit(X, y, ...)`
    - `apply(X)` returning leaf indices of shape `(n_samples, n_estimators)`
    - `estimators_` and `n_estimators` after fitting

    Supported proximity methods:
    - `"original"`: classical RF proximity
    - `"oob"`: OOB-only approximation in separable sparse form
    - `"rfgap"`: RF-GAP asymmetric proximity
    - `"honest_gap"`: prediction-preserving HonestForest classifier proximity
    - `"englund"`: Englund–Verikas terminal-node branch-distance kernel

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

    def __init__(
        self,
        estimator: Any,
        relation: ProxMethod = "rfgap",
        *,
        matrix_type: MatrixType = "sparse",
        non_zero_diagonal: bool = False,
        symm_mode: SymmMode = None,
        max_normalize: bool = False,
        random_state: int | None = None,
        englund_weight: float = 1.0,
    ) -> None:
        """Create an unfitted relation index around an estimator.

        ``fit`` fits the supplied estimator before indexing it; ``attach``
        indexes an estimator that has already been fitted.  The estimator is
        never dynamically subclassed and remains available as ``estimator``.
        """

        if estimator is None:
            raise TypeError("estimator must be a forest-like estimator instance")
        if isinstance(estimator, type):
            raise TypeError("estimator must be an instance, not an estimator class")
        self.estimator = estimator
        self.relation = relation
        self.random_state = random_state
        if not np.isfinite(englund_weight) or englund_weight < 0:
            raise ValueError("englund_weight must be a finite non-negative number")
        self.englund_weight = float(englund_weight)
        self.capabilities_: EstimatorCapabilities | None = None
        self.relation_metadata_: dict[str, Any] = {}
        self._relation_feature_cache: dict[str, Any] = {}
        self._init_sparse_proximity(
            prox_method=relation,
            matrix_type=matrix_type,
            non_zero_diagonal=non_zero_diagonal,
            symm_mode=symm_mode,
            max_normalize=max_normalize,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate estimator state used by the sparse factor implementation."""

        estimator = self.__dict__.get("estimator")
        if estimator is None:
            raise AttributeError(name)
        try:
            return getattr(estimator, name)
        except AttributeError as exc:
            raise AttributeError(name) from exc

    def _init_sparse_proximity(
        self,
        prox_method: ProxMethod = "rfgap",
        matrix_type: MatrixType = "sparse",
        non_zero_diagonal: bool = False,
        symm_mode: SymmMode = None,
        max_normalize: bool = False,
    ) -> None:
        valid_relations = {
            "original", "oob", "rfgap", "honest_gap", "kerf", "path",
            "boosted", "englund", "random_height", "collision", "conditional",
            "survival_leaf",
        }
        if prox_method not in valid_relations:
            raise ValueError(
                f"relation must be one of {sorted(valid_relations)!r}"
            )
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
        self._X_train: Optional[np.ndarray] = None
        self._X_train_native: Any | None = None
        self._X_train_index_: Optional[pd.Index] = None
        self._X_train_columns_: Optional[pd.Index] = None
        self._X_train_was_dataframe_: bool = False
        self._boosted_adapter: BoostedTreeAdapter | None = None
        self.tree_weights_: np.ndarray | None = None
        self.tree_weighting: str | None = None

    def fit(self, X: Any, y: Optional[Any] = None, **fit_kwargs: Any) -> "Proximity":
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
        Proximity
            Fitted estimator with sparse proximity factors cached.
        """
        base_fit = getattr(self.estimator, "fit")
        if _is_rotation_forest_instance(self.estimator):
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

        return self.attach(X, y)

    def attach(self, X: Any, y: Optional[Any] = None) -> "Proximity":
        """Index ``X`` with an already-fitted estimator and build relation state."""

        check_is_fitted(self.estimator)
        self.capabilities_ = validate_relation(self.estimator, self.relation)
        self._boosted_adapter = (
            make_boosted_adapter(self.estimator)
            if self.capabilities_.family in BOOSTED_FAMILIES
            else None
        )
        self._index_training_data(X, y)
        self._set_rotation_forest_oob_attributes(X, y)
        self.relation_metadata_ = self._base_relation_metadata()
        return self

    def _set_rotation_forest_oob_attributes(self, X: Any, y: Optional[Any]) -> None:
        if _is_rotation_forest_instance(self.estimator) and self.relation in {"oob", "rfgap"}:
            _RotationForestOOBAdapter._set_rotation_forest_oob_attributes(self, X, y)

    def predict(self, X: Any) -> Any:
        base_predict = getattr(self.estimator, "predict")
        if _is_rotation_forest_instance(self.estimator):
            with _suppress_base_validate_data_futurewarning():
                return base_predict(X)
        return base_predict(X)

    def apply(self, X: Any) -> np.ndarray:
        """Return leaf IDs through the estimator-specific apply adapter."""

        if self._boosted_adapter is not None:
            return self._boosted_adapter.leaf_ids(X)
        apply_fn = getattr(self.estimator, "apply", None)
        if callable(apply_fn):
            return np.asarray(apply_fn(X))
        if _is_rotation_forest_instance(self.estimator):
            return rotation_forest_leaf_ids(self.estimator, X)
        if _is_isolation_forest_instance(self.estimator):
            return isolation_leaf_ids(self.estimator, X)
        raise UnsupportedRelationError(
            self.capabilities_.family if self.capabilities_ else "unknown",
            self.relation,
            "the estimator does not expose leaf assignments",
        )

    def _index_training_data(self, X: Any, y: Optional[Any] = None) -> None:
        """
        Build sparse factors required for proximity computations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data used during model fitting.
        """
        check_is_fitted(self.estimator)

        self._y_train = np.asarray(y) if y is not None else None
        self._store_training_features(X)
        self._oob_surrogate_estimators_ = None
        self._oob_surrogate_leaf_matrix_ = None
        self._rotation_original_leaf_encoding_: Optional[_LeafEncoding] = None
        self._rotation_original_W_mat_: Optional[sparse.csr_matrix] = None

        apply_fn = getattr(self, "apply", None)
        if apply_fn is None:
            raise RuntimeError("Estimator must implement apply(X) to use proximity methods.")

        if self.prox_method == "honest_gap":
            self._validate_honest_gap_inputs()

        leaves: Optional[np.ndarray] = None

        if self.prox_method in ("oob", "rfgap", "honest_gap"):
            n_samples = int(len(X))
            try:
                oob_mask, in_bag_counts = self._compute_bootstrap_stats(n_samples, X=X, y=y)
            except RuntimeError:
                if self.prox_method != "honest_gap":
                    raise
                estimators = getattr(self, "estimators_", None)
                if estimators is None:
                    raise
                oob_mask, in_bag_counts = self._honest_gap_oob_stats_from_indices(
                    n_samples,
                    len(estimators),
                )
            self.oob_mask_ = oob_mask.astype(np.int8)
            self.in_bag_counts_ = in_bag_counts.astype(np.float32)

            surrogate_leaves = getattr(self, "_oob_surrogate_leaf_matrix_", None)
            if (
                self.prox_method in ("oob", "rfgap")
                and surrogate_leaves is not None
                and surrogate_leaves.shape[0] == n_samples
            ):
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

        if self.prox_method == "boosted":
            if self._boosted_adapter is None:
                raise RuntimeError("Boosted relation requires a fitted-contribution adapter.")
            weights = np.asarray(
                self._boosted_adapter.contribution_weights(
                    self._relation_training_features(),
                ),
                dtype=np.float64,
            )
            if weights.shape != (n_trees,):
                raise RuntimeError("Boosted tree-weight inventory does not match leaf columns.")
            if np.any(~np.isfinite(weights)) or np.any(weights < 0):
                raise RuntimeError("Boosted tree weights must be finite and non-negative.")
            total_weight = float(weights.sum())
            if total_weight <= 0:
                raise RuntimeError("Boosted tree weights must have positive total mass.")
            self.tree_weights_ = weights / total_weight
            self.tree_weighting = "training_leaf_contribution_variance"

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

        if self.prox_method in {"kerf", "path", "englund", "random_height"}:
            result = self._kernel_relation(self.prox_method, X_new=None)
            self.relation_metadata_.update(self.diagnostics(result))
            return result

        W = self.W_mat
        if W is None:
            raise RuntimeError("W matrix is not initialized.")

        leaves = self.leaf_matrix_
        if leaves is None:
            raise RuntimeError("Leaf matrix is not initialized.")

        if self.prox_method in {
            "original", "oob", "boosted", "collision", "survival_leaf",
        }:
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

        result = P.toarray() if self.matrix_type == "dense" else P
        self.relation_metadata_.update(self.diagnostics(result))
        return result

    def get_relation(self, relation: str | None = None) -> MatrixLike:
        """Return the indexed training relation, optionally selecting another valid relation."""

        if relation is None or relation == self.relation:
            return self.get_proximities()
        return self._alternate_relation(relation).get_proximities()

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

        if self.prox_method in {"kerf", "path", "englund", "random_height"}:
            return self._kernel_relation(self.prox_method, X_new=X_new)

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
        elif self.prox_method == "honest_gap":
            self._csr_row_normalize_inplace(P_new)

        return P_new.toarray() if self.matrix_type == "dense" else P_new

    def extend_relation(self, X_new: Any, relation: str | None = None) -> MatrixLike:
        """Return a query-to-training relation for new observations."""

        if relation is None or relation == self.relation:
            return self.prox_extend(X_new)
        return self._alternate_relation(relation).prox_extend(X_new)

    def diagnostics(self, matrix: MatrixLike | None = None) -> dict[str, Any]:
        """Summarize directionality, mass, sparsity, symmetry, and PSD evidence."""

        if matrix is None:
            matrix = self.get_proximities()
        shape = tuple(int(value) for value in matrix.shape)
        if sparse.issparse(matrix):
            csr = cast(sparse.csr_matrix, matrix.tocsr())
            nnz = int(csr.nnz)
            values = csr.data
            row_mass = np.asarray(csr.sum(axis=1)).ravel()
            symmetric = shape[0] == shape[1] and (csr - csr.T).nnz == 0
            dense_for_psd = csr.toarray() if symmetric and shape[0] <= 512 else None
        else:
            dense = np.asarray(matrix, dtype=float)
            nnz = int(np.count_nonzero(dense))
            values = dense.ravel()
            row_mass = dense.sum(axis=1)
            symmetric = shape[0] == shape[1] and bool(np.allclose(dense, dense.T))
            dense_for_psd = dense if symmetric and shape[0] <= 512 else None
        min_eigenvalue: float | None = None
        psd: bool | None = None
        if dense_for_psd is not None:
            min_eigenvalue = float(np.linalg.eigvalsh((dense_for_psd + dense_for_psd.T) / 2).min())
            psd = bool(min_eigenvalue >= -1e-7)
        return {
            "relation": self.relation,
            "shape": shape,
            "directed": self.relation in {"rfgap", "honest_gap", "conditional"},
            "symmetric": symmetric,
            "psd": psd,
            "min_eigenvalue": min_eigenvalue,
            "nnz": nnz,
            "density": float(nnz / (shape[0] * shape[1])) if shape[0] and shape[1] else 0.0,
            "value_min": float(np.min(values)) if values.size else 0.0,
            "value_max": float(np.max(values)) if values.size else 0.0,
            "row_mass_min": float(np.min(row_mass)) if row_mass.size else 0.0,
            "row_mass_max": float(np.max(row_mass)) if row_mass.size else 0.0,
            **(
                {
                    "tree_weighting": self.tree_weighting,
                    "tree_weight_sum": float(np.sum(self.tree_weights_)),
                    "tree_weight_min": float(np.min(self.tree_weights_)),
                    "tree_weight_max": float(np.max(self.tree_weights_)),
                }
                if self.relation == "boosted" and self.tree_weights_ is not None
                else {}
            ),
        }

    def _base_relation_metadata(self) -> dict[str, Any]:
        capabilities = self.capabilities_ or capabilities_for(self.estimator)
        metadata = {
            "relation": self.relation,
            "estimator_family": capabilities.family,
            "task_type": capabilities.task_type,
            "direction": "query_to_training" if self.relation in {"rfgap", "honest_gap", "conditional"} else "symmetric",
            "symmetrization": self.symm_mode,
            "prediction_identity": self.relation in {"rfgap", "honest_gap"},
            "random_state": self.random_state,
        }
        if self.relation == "englund":
            metadata["englund_weight"] = self.englund_weight
        if self.relation == "boosted" and self.tree_weights_ is not None:
            metadata["tree_weighting"] = self.tree_weighting
            metadata["tree_weights"] = self.tree_weights_.tolist()
        oob_valid = getattr(self, "oob_valid_mask_", None)
        if self.relation in {"oob", "rfgap"} and oob_valid is not None:
            valid = np.asarray(oob_valid, dtype=bool)
            metadata["oob_valid_rows"] = int(np.count_nonzero(valid))
            metadata["oob_invalid_rows"] = int(valid.size - np.count_nonzero(valid))
        return metadata

    def _alternate_relation(self, relation: str) -> "Proximity":
        alternate = Proximity(
            self.estimator,
            relation=cast(ProxMethod, relation),
            matrix_type=self.matrix_type,
            non_zero_diagonal=self.non_zero_diagonal,
            symm_mode=self.symm_mode,
            max_normalize=self.max_normalize,
            random_state=self.random_state,
            englund_weight=self.englund_weight,
        )
        training_features = self._relation_training_features()
        if training_features is None:
            raise RuntimeError("Training features are unavailable for relation dispatch.")
        alternate.attach(training_features, self._y_train)
        return alternate

    def _kernel_relation(self, relation: str, X_new: Any | None) -> MatrixLike:
        if relation == "kerf":
            train_features = self._kerf_features(
                self._relation_training_features(), training=True,
            )
            query_features = train_features if X_new is None else self._kerf_features(X_new, training=False)
        elif relation == "path":
            train_features = self._path_features(
                self._relation_training_features(), training=True,
            )
            query_features = train_features if X_new is None else self._path_features(X_new, training=False)
        elif relation == "englund":
            return self._englund_relation(X_new)
        elif relation == "random_height":
            train_features = self._random_height_features(
                self._relation_training_features(), training=True,
            )
            query_features = train_features if X_new is None else self._random_height_features(X_new, training=False)
        else:  # pragma: no cover - internal dispatch guard
            raise ValueError(f"Unknown kernel relation: {relation}")
        result = cast(sparse.csr_matrix, query_features @ train_features.T).tocsr()
        result.eliminate_zeros()
        return result.toarray() if self.matrix_type == "dense" else result

    def _kerf_features(self, X: Any, *, training: bool) -> sparse.csr_matrix:
        if training and "kerf" in self._relation_feature_cache:
            return self._relation_feature_cache["kerf"]
        if self._leaf_encoding is None or self.leaf_matrix_ is None:
            raise RuntimeError("Leaf encoding is unavailable.")
        leaves = self.leaf_matrix_ if training else self.apply(X)
        rows, cols, _ = self._map_leaves_to_global(leaves)
        if training:
            masses = np.bincount(cols, minlength=self._leaf_encoding.total_features).astype(float)
            self._relation_feature_cache["kerf_masses"] = masses
        else:
            masses = np.asarray(self._relation_feature_cache.get("kerf_masses"))
        if masses.size == 0:
            raise RuntimeError("KeRF leaf masses are unavailable.")
        n_trees = leaves.shape[1]
        values = np.zeros(rows.shape[0], dtype=float)
        nonzero = masses[cols] > 0
        values[nonzero] = 1.0 / np.sqrt(n_trees * masses[cols[nonzero]])
        features = sparse.csr_matrix(
            (values[nonzero], (rows[nonzero], cols[nonzero])),
            shape=(leaves.shape[0], self._leaf_encoding.total_features),
        )
        if training:
            self._relation_feature_cache["kerf"] = features
        return features

    def _decision_path_indicator(self, X: Any) -> tuple[sparse.csr_matrix, np.ndarray]:
        if self._boosted_adapter is not None:
            result = self._boosted_adapter.decision_path(X)
        else:
            result = (
                rotation_forest_decision_path(self.estimator, X)
                if _is_rotation_forest_instance(self.estimator)
                else self.estimator.decision_path(X)
            )
        if not isinstance(result, tuple) or len(result) != 2:
            raise UnsupportedRelationError(
                self.capabilities_.family if self.capabilities_ else "unknown",
                self.relation,
                "decision_path must return (indicator, node_ptr)",
            )
        indicator, node_ptr = result
        indicator = sparse.csr_matrix(indicator)
        node_ptr = np.asarray(node_ptr, dtype=np.int64)
        n_trees = (
            self._boosted_adapter.tree_count
            if self._boosted_adapter is not None
            else len(getattr(self.estimator, "estimators_", ()))
        )
        if (
            node_ptr.ndim != 1
            or node_ptr.shape[0] != n_trees + 1
            or node_ptr[0] != 0
            or node_ptr[-1] != indicator.shape[1]
            or np.any(np.diff(node_ptr) < 0)
        ):
            raise UnsupportedRelationError(
                self.capabilities_.family if self.capabilities_ else "unknown",
                self.relation,
                "decision_path must return stable per-tree node offsets",
            )
        return indicator, node_ptr

    def _path_features(self, X: Any, *, training: bool) -> sparse.csr_matrix:
        if training and "path" in self._relation_feature_cache:
            return self._relation_feature_cache["path"]
        indicator, node_ptr = self._decision_path_indicator(X)
        if training:
            self._relation_feature_cache["path_ptr"] = node_ptr
        elif not np.array_equal(node_ptr, self._relation_feature_cache.get("path_ptr")):
            raise RuntimeError("Estimator path inventory changed after attachment.")
        row_norm = np.sqrt(np.asarray(indicator.multiply(indicator).sum(axis=1)).ravel())
        row_norm[row_norm == 0] = 1.0
        features = indicator.multiply((1.0 / row_norm)[:, None]).tocsr()
        if training:
            self._relation_feature_cache["path"] = features
        return features

    def _englund_paths(
        self, X: Any, *, training: bool,
    ) -> list[tuple[np.ndarray, dict[int, np.ndarray]]]:
        """Return terminal assignments and root-to-terminal paths per tree.

        Englund and Verikas define the per-tree similarity as
        ``exp(-w * g)``, where ``g`` is the number of tree branches on the
        path between the two occupied terminal nodes.  Keeping one path per
        distinct terminal node lets the pairwise kernel reuse the same tree
        geometry for all observations in that leaf.
        """

        cache_key = "englund_paths"
        if training and cache_key in self._relation_feature_cache:
            return self._relation_feature_cache[cache_key]

        indicator, node_ptr = self._decision_path_indicator(X)
        if training:
            self._relation_feature_cache["englund_ptr"] = node_ptr
        elif not np.array_equal(node_ptr, self._relation_feature_cache.get("englund_ptr")):
            raise RuntimeError("Estimator path inventory changed after attachment.")

        paths_by_tree: list[tuple[np.ndarray, dict[int, np.ndarray]]] = []
        for tree_idx in range(node_ptr.shape[0] - 1):
            start, stop = int(node_ptr[tree_idx]), int(node_ptr[tree_idx + 1])
            block = indicator[:, start:stop].tocsr()
            if block.shape[1] == 0 or np.any(np.diff(block.indptr) == 0):
                raise RuntimeError("Every observation must have a non-empty tree decision path.")

            row_ends = block.indptr[1:] - 1
            terminals = block.indices[row_ends].astype(np.int64, copy=False)
            _, first_rows = np.unique(terminals, return_index=True)
            paths: dict[int, np.ndarray] = {}
            for row in first_rows:
                row_start, row_stop = block.indptr[row], block.indptr[row + 1]
                terminal = int(terminals[row])
                paths[terminal] = block.indices[row_start:row_stop].astype(
                    np.int64, copy=True,
                )
            paths_by_tree.append((terminals, paths))

        if training:
            self._relation_feature_cache[cache_key] = paths_by_tree
        return paths_by_tree

    @staticmethod
    def _englund_tree_kernel(
        query_terminals: np.ndarray,
        query_paths: dict[int, np.ndarray],
        train_terminals: np.ndarray,
        train_paths: dict[int, np.ndarray],
        weight: float,
    ) -> np.ndarray:
        """Compute one tree's terminal-node branch-distance kernel."""

        query_unique, query_inverse = np.unique(query_terminals, return_inverse=True)
        train_unique, train_inverse = np.unique(train_terminals, return_inverse=True)
        leaf_kernel = np.empty(
            (query_unique.size, train_unique.size), dtype=np.float64,
        )

        for query_idx, query_terminal in enumerate(query_unique):
            query_path = query_paths[int(query_terminal)]
            for train_idx, train_terminal in enumerate(train_unique):
                train_path = train_paths[int(train_terminal)]
                common_length = min(query_path.size, train_path.size)
                mismatch = np.flatnonzero(
                    query_path[:common_length] != train_path[:common_length]
                )
                common_length = int(mismatch[0]) if mismatch.size else common_length
                branch_distance = query_path.size + train_path.size - 2 * common_length
                leaf_kernel[query_idx, train_idx] = np.exp(
                    -weight * float(branch_distance)
                )

        return leaf_kernel[query_inverse[:, None], train_inverse[None, :]]

    def _englund_relation(self, X_new: Any | None) -> MatrixLike:
        """Compute the Englund–Verikas PBK over all fitted forest trees."""

        training_features = self._relation_training_features()
        if training_features is None:
            raise RuntimeError("Training features are unavailable for relation dispatch.")
        train_paths = self._englund_paths(training_features, training=True)
        query_paths = train_paths if X_new is None else self._englund_paths(X_new, training=False)
        if len(query_paths) != len(train_paths):
            raise RuntimeError("Query and training forests have different tree counts.")

        n_query = query_paths[0][0].shape[0] if query_paths else 0
        n_train = train_paths[0][0].shape[0] if train_paths else 0
        result = np.zeros((n_query, n_train), dtype=np.float64)
        for (query_terminals, query_tree_paths), (train_terminals, train_tree_paths) in zip(
            query_paths, train_paths,
        ):
            result += self._englund_tree_kernel(
                query_terminals,
                query_tree_paths,
                train_terminals,
                train_tree_paths,
                self.englund_weight,
            )
        if train_paths:
            result /= float(len(train_paths))

        if self.matrix_type == "dense":
            return result
        return sparse.csr_matrix(result)

    @staticmethod
    def _node_depths(tree: Any) -> np.ndarray:
        children_left = np.asarray(tree.children_left)
        children_right = np.asarray(tree.children_right)
        depths = np.zeros(tree.node_count, dtype=np.int32)
        stack = [0]
        while stack:
            node = stack.pop()
            for child in (children_left[node], children_right[node]):
                if child >= 0:
                    depths[child] = depths[node] + 1
                    stack.append(int(child))
        return depths

    def _forest_node_depths(self) -> list[np.ndarray]:
        """Return node depths using the estimator's declared tree layout."""

        if self._boosted_adapter is not None:
            return self._boosted_adapter.node_depths()
        if _is_rotation_forest_instance(self.estimator):
            return rotation_forest_node_depths(self.estimator)

        depths: list[np.ndarray] = []
        for tree in self.estimators_:
            adapter_depths = getattr(tree, "node_depths_", None)
            if adapter_depths is not None:
                depths.append(np.asarray(adapter_depths, dtype=np.int32))
                continue
            tree_state = getattr(tree, "tree_", None)
            if tree_state is None:
                raise UnsupportedRelationError(
                    self.capabilities_.family if self.capabilities_ else "unknown",
                    self.relation,
                    "the estimator does not expose per-tree node depths",
                )
            depths.append(self._node_depths(tree_state))
        return depths

    def _random_height_features(self, X: Any, *, training: bool) -> sparse.csr_matrix:
        cache_key = "random_height"
        if training and cache_key in self._relation_feature_cache:
            return self._relation_feature_cache[cache_key]
        indicator, node_ptr = self._decision_path_indicator(X)
        n_trees = int(node_ptr.size - 1)
        if training:
            rng = np.random.default_rng(self.random_state)
            depths = self._forest_node_depths()
            if len(depths) != n_trees:
                raise RuntimeError("Node-depth inventory does not match fitted tree inventory.")
            for tree_idx, depth in enumerate(depths):
                expected = int(node_ptr[tree_idx + 1] - node_ptr[tree_idx])
                if depth.shape != (expected,):
                    raise RuntimeError("Node-depth inventory does not match decision-path offsets.")
            heights = np.asarray(
                [rng.integers(0, int(depth.max()) + 1) for depth in depths],
                dtype=np.int32,
            )
            self._relation_feature_cache["random_height_depths"] = depths
            self._relation_feature_cache["random_height_heights"] = heights
            self._relation_feature_cache["random_height_ptr"] = node_ptr
            self.relation_metadata_["tree_cut_heights"] = heights.tolist()
        else:
            depths = self._relation_feature_cache["random_height_depths"]
            heights = self._relation_feature_cache["random_height_heights"]
            if not np.array_equal(node_ptr, self._relation_feature_cache["random_height_ptr"]):
                raise RuntimeError("Estimator path inventory changed after attachment.")
        rows: list[int] = []
        cols: list[int] = []
        for tree_idx, (depth, height) in enumerate(zip(depths, heights)):
            start, stop = int(node_ptr[tree_idx]), int(node_ptr[tree_idx + 1])
            block = indicator[:, start:stop].tocsr()
            for row in range(block.shape[0]):
                nodes = block.indices[block.indptr[row]:block.indptr[row + 1]]
                eligible = nodes[depth[nodes] <= height]
                if eligible.size:
                    selected = int(eligible[np.argmax(depth[eligible])])
                    rows.append(row)
                    cols.append(start + selected)
        values = np.full(len(rows), 1.0 / np.sqrt(n_trees), dtype=float)
        features = sparse.csr_matrix(
            (values, (rows, cols)), shape=(indicator.shape[0], indicator.shape[1]),
        )
        if training:
            self._relation_feature_cache[cache_key] = features
        return features

    def prox_predict(self, X: Any = None) -> np.ndarray:
        """
        Predict using prediction-preserving proximity-weighted training labels.

        For classifiers, returns class estimates (matching ``predict_proba``
        for test data, and ``oob_decision_function_`` for OOB training data).
        Honest-GAP preserves HonestForest's native OOB scale; with
        ``kernel_method=True`` this can be raw/count-like rather than
        probability-normalized. For regressors, returns predicted values
        (matching ``predict`` for test data or ``oob_prediction_`` for
        training data). When ``X`` is ``None``, out-of-bag predictions for the
        training data are returned.

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
        if self.prox_method not in ("rfgap", "honest_gap"):
            raise ValueError("prox_predict requires prox_method='rfgap' or 'honest_gap'")
        if self._y_train is None:
            raise RuntimeError(
                "No training labels available. Refit with labels to use prox_predict."
            )

        P = self.get_proximities() if X is None else self.prox_extend(X)

        classes = getattr(self, "classes_", None)
        if classes is not None:
            classes = np.asarray(classes)
            y_idx = self._class_indices_for_values(np.asarray(self._y_train), classes)
            y_hot = sparse.csc_matrix(
                (np.ones(len(y_idx), dtype=np.float64), (np.arange(len(y_idx)), y_idx)),
                shape=(len(y_idx), classes.shape[0]),
            )
            if sparse.issparse(P):
                prox_prob = np.asarray((P @ y_hot).todense())
            else:
                prox_prob = np.asarray(P @ y_hot.toarray())
            if X is None and self.prox_method == "rfgap":
                rs = prox_prob.sum(axis=1, keepdims=True)
                rs[rs == 0] = 1.0
                prox_prob /= rs
            return prox_prob

        y = self._y_train.astype(np.float64)
        result = P @ y
        return np.asarray(result.todense()).ravel() if sparse.issparse(result) else np.asarray(result).ravel()

    def _impute(
        self,
        X: Any,
        categorical_features: Optional[Sequence[int]] = None,
        X_test: Optional[Any] = None,
        initialization: str = "median",
    ) -> Any:
        """
        Impute missing values using proximity-weighted replacement.

        For each missing continuous value ``X[n, m]``, the fill is a
        weighted average of non-missing values of feature *m*, where
        weights are the proximities between sample *n* and all samples
        with observed feature *m*.

        For each missing categorical value, the fill is the most frequent
        non-missing category weighted by proximity.

        Accepts numpy arrays or pandas DataFrames.  When a DataFrame is
        passed the result is returned as a DataFrame with the original
        index and columns preserved.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data with ``np.nan`` / ``NaN`` marking missing entries.
        categorical_features : sequence of int, optional
            Column indices of categorical features.  All other columns
            are treated as continuous.
        X_test : array-like or DataFrame of shape (n_test, n_features), optional
            Test data with ``np.nan`` / ``NaN`` marking missing entries.
            If provided, imputed using ``prox_extend`` proximities against
            training points.
        initialization : str, default ``'median'``
            Strategy used to rough-fill NaN values before computing test
            proximities (``prox_extend`` requires NaN-free input).
            Options: ``'mean'``, ``'median'``, ``'knn'``.

        Returns
        -------
        X_imputed : np.ndarray or DataFrame
            Copy of *X* with missing values filled.
        X_test_imputed : np.ndarray or DataFrame
            Returned only when *X_test* is provided.
        """
        if initialization not in ("mean", "median", "knn"):
            raise ValueError(
                f"initialization must be 'mean', 'median', or 'knn', got {initialization!r}"
            )
        self._check_proximity_ready()

        try:
            import pandas as pd
            _has_pandas = True
        except ImportError:
            _has_pandas = False

        X_is_df = _has_pandas and isinstance(X, pd.DataFrame)
        Xt_is_df = _has_pandas and X_test is not None and isinstance(X_test, pd.DataFrame)

        X_idx = X.index if X_is_df else None
        X_cols = X.columns if X_is_df else None
        Xt_idx = X_test.index if Xt_is_df else None
        Xt_cols = X_test.columns if Xt_is_df else None

        cat_set = set(categorical_features) if categorical_features is not None else set()

        X_vals = X.to_numpy(dtype=np.float64, copy=True, na_value=np.nan) if X_is_df else np.array(X, dtype=np.float64, copy=True)
        X_imp = X_vals
        missing_train = np.isnan(X_imp)

        P_train = self.get_proximities()
        if not sparse.issparse(P_train):
            P_train = sparse.csr_matrix(P_train)

        # Training imputation: query=train, reference=train
        self._prox_impute_inplace(
            X_query=X_imp, missing_query=missing_train,
            X_ref=X_imp, missing_ref=missing_train,
            P=P_train, cat_set=cat_set,
        )

        if X_test is not None:
            X_test_vals = X_test.to_numpy(dtype=np.float64, copy=True, na_value=np.nan) if Xt_is_df else np.array(X_test, dtype=np.float64, copy=True)
            X_test_imp = X_test_vals
            missing_test = np.isnan(X_test_imp)

            # prox_extend calls apply() which rejects NaN, so pass a
            # rough-filled copy for leaf assignment only.
            X_test_filled = X_test_imp.copy()
            if initialization == "knn":
                from sklearn.impute import KNNImputer
                imputer = KNNImputer()
                # Fit on already-imputed training data, transform test
                imputer.fit(X_imp)
                X_test_filled = imputer.transform(X_test_filled)
            else:
                if initialization == "mean":
                    col_stats = np.nanmean(X_imp, axis=0)
                else:  # median
                    col_stats = np.nanmedian(X_imp, axis=0)
                for col in range(X_test_filled.shape[1]):
                    nans = np.isnan(X_test_filled[:, col])
                    if nans.any():
                        X_test_filled[nans, col] = col_stats[col]

            P_test = self.prox_extend(X_test_filled)
            if not sparse.issparse(P_test):
                P_test = sparse.csr_matrix(P_test)

            # Test imputation: query=test, reference=train (already imputed)
            self._prox_impute_inplace(
                X_query=X_test_imp, missing_query=missing_test,
                X_ref=X_imp, missing_ref=missing_train,
                P=P_test, cat_set=cat_set,
            )
            out_tr = pd.DataFrame(X_imp, index=X_idx, columns=X_cols) if X_is_df else X_imp
            out_te = pd.DataFrame(X_test_imp, index=Xt_idx, columns=Xt_cols) if Xt_is_df else X_test_imp
            return out_tr, out_te

        return pd.DataFrame(X_imp, index=X_idx, columns=X_cols) if X_is_df else X_imp

    @staticmethod
    def _prox_impute_inplace(
        X_query: np.ndarray,
        missing_query: np.ndarray,
        X_ref: np.ndarray,
        missing_ref: np.ndarray,
        P: sparse.csr_matrix,
        cat_set: set,
    ) -> None:
        """Fill ``np.nan`` entries in *X_query* using proximity matrix *P*.

        *P* has shape ``(n_query, n_ref)``; reference values come from
        *X_ref*.  For each feature column:

        - Zero out proximity weights for reference rows where the feature
          is missing, then row-normalise.
        - Continuous: weighted average of non-missing reference values.
        - Categorical: weighted vote among non-missing reference categories.
        """
        n_features = X_query.shape[1]

        for m in range(n_features):
            col_missing_query = missing_query[:, m]
            if not col_missing_query.any():
                continue

            # Observed mask for the *reference* data
            observed_ref = (~missing_ref[:, m]).astype(np.float64)  # (n_ref,)

            # Scale proximity columns by the observed mask: sets weight
            # to 0 for reference samples that are missing feature m.
            P_obs = P.multiply(observed_ref[np.newaxis, :])
            P_obs = sparse.csr_matrix(P_obs)

            # Row-normalise so weights for each query row sum to 1
            row_sums = np.asarray(P_obs.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1.0
            diag_inv = sparse.diags(1.0 / row_sums, format="csr")
            P_norm = diag_inv @ P_obs

            query_rows = np.where(col_missing_query)[0]

            ref_vals = X_ref[:, m].copy()
            ref_vals[missing_ref[:, m]] = 0.0

            if m in cat_set:
                # Categorical: weighted vote
                categories = np.unique(X_ref[:, m][~missing_ref[:, m]])
                if categories.size == 0:
                    continue
                cat_idx = np.searchsorted(categories, ref_vals)
                cat_idx = np.clip(cat_idx, 0, categories.size - 1)
                one_hot = sparse.csc_matrix(
                    (observed_ref, (np.arange(len(ref_vals)), cat_idx)),
                    shape=(len(ref_vals), categories.size),
                )
                votes = P_norm[query_rows] @ one_hot
                votes_dense = np.asarray(votes.todense() if sparse.issparse(votes) else votes)
                best = np.argmax(votes_dense, axis=1)
                X_query[query_rows, m] = categories[best]
            else:
                # Continuous: weighted average
                result = P_norm[query_rows] @ ref_vals
                if sparse.issparse(result):
                    result = np.asarray(result.todense()).ravel()
                X_query[query_rows, m] = result

    def _outlier_scores(
        self,
        y: Any = None,
        scaling: Optional[Literal["normalize", "log"]] = None,
        method: Literal["breiman", "ene"] = "breiman",
    ) -> Any:
        """
        Compute outlier scores from the proximity matrix.

        When *y* is provided (classification), scores are computed
        class-relative following Breiman's definition:

        1. For each case *n* in class *j*, compute
           ``avg_prox(n) = sum_k prox(n, k)^2`` over same-class cases.
        2. Raw outlier measure: ``n_j / avg_prox(n)``.
        3. Within each class, standardise by subtracting the median and
           dividing by the median absolute deviation (MAD).

        When *y* is ``None`` (unsupervised / label-free), global
        proximities are used instead:

        1. ``avg_prox(n) = sum_k prox(n, k)^2`` over *all* samples.
        2. Raw outlier measure: ``n / avg_prox(n)``.
        3. MAD-standardise globally.

        Accepts numpy arrays or pandas Series / DataFrames for *y*.  When
        a pandas object with an index is passed, the result is returned as
        a ``pandas.Series`` with the same index.

        Parameters
        ----------
        y : array-like, Series, DataFrame, or None
            Class labels for training samples.  If ``None``, a global
            (label-free) outlier score is computed using all pairwise
            proximities.
        scaling : {None, 'normalize', 'log'}, optional
            Post-processing applied to the standardised scores.

            - ``None``: return the raw MAD-standardised scores.
            - ``'normalize'``: rescale to [0, 1].
            - ``'log'``: apply ``log(1 + score)``.
        method : {'breiman', 'ene'}, default 'breiman'
            Formula used to compute the raw outlier measure.

            - ``'breiman'``: Breiman's sum-of-squared-proximity approach.
              ``raw(n) = N / sum_k P(n,k)^2``.  Sensitive to the number
              of proximity neighbours, which can cause central inliers in
              IsolationForest to receive spuriously high scores.
            - ``'ene'``: Effective Number of Neighbours (inverse
              participation ratio).
              ``ENE(n) = (sum_k P(n,k))^2 / (sum_k P(n,k)^2)``
              and ``raw(n) = N / ENE(n)``.  This normalises out the
              confounding effect of neighbour count, making it
              well-suited for IsolationForest proximities where dense
              regions accumulate many weak connections.

        Returns
        -------
        scores : np.ndarray or pandas.Series of shape (n_samples,)
            Outlier scores (higher = more outlier-like).
        """
        self._check_proximity_ready()

        try:
            import pandas as pd
            _has_pandas = True
        except ImportError:
            _has_pandas = False

        y_is_pandas = _has_pandas and y is not None and isinstance(y, (pd.Series, pd.DataFrame))
        y_index = y.index if y_is_pandas else None

        # Get proximities with zero diagonal and no symmetrisation so the
        # raw self-proximity does not inflate within-class sums.
        orig_nzd = self.non_zero_diagonal
        orig_sym = self.symm_mode
        self.non_zero_diagonal = False
        self.symm_mode = None
        try:
            P = self.get_proximities()
        finally:
            self.non_zero_diagonal = orig_nzd
            self.symm_mode = orig_sym

        if not sparse.issparse(P):
            P = sparse.csr_matrix(P)

        n_samples = P.shape[0]

        if y is None:
            # --- Label-free (unsupervised) path ---
            P_sq = P.multiply(P)
            sum_P_sq = np.asarray(P_sq.sum(axis=1)).ravel()

            if method == "ene":
                sum_P = np.asarray(P.sum(axis=1)).ravel()
                sum_P_sq_safe = sum_P_sq.copy()
                sum_P_sq_safe[sum_P_sq_safe == 0] = 1e-10
                ene = sum_P ** 2 / sum_P_sq_safe
                ene[ene == 0] = 1e-10
                raw_scores = n_samples / ene
            else:
                sum_P_sq[sum_P_sq == 0] = 1e-10
                raw_scores = n_samples / sum_P_sq

            med = np.median(raw_scores)
            mad = np.median(np.abs(raw_scores - med))
            if mad == 0:
                scores = np.zeros(n_samples, dtype=np.float64)
            else:
                scores = np.abs(raw_scores - med) / mad
        else:
            # --- Class-relative (supervised) path ---
            y_arr = np.asarray(y)

            raw_scores = np.zeros(n_samples, dtype=np.float64)
            classes = np.unique(y_arr)

            if method == "ene":
                for cls in classes:
                    idx = np.where(y_arr == cls)[0]
                    P_sub = P[idx][:, idx]
                    sum_P_sub = np.asarray(P_sub.sum(axis=1)).ravel()
                    P_sub_sq = P_sub.multiply(P_sub)
                    sum_P_sub_sq = np.asarray(P_sub_sq.sum(axis=1)).ravel()
                    sum_P_sub_sq[sum_P_sub_sq == 0] = 1e-10
                    ene = sum_P_sub ** 2 / sum_P_sub_sq
                    ene[ene == 0] = 1e-10
                    raw_scores[idx] = len(idx) / ene
            else:
                avg_prox = np.zeros(n_samples, dtype=np.float64)
                for cls in classes:
                    idx = np.where(y_arr == cls)[0]
                    P_sub = P[idx][:, idx]
                    P_sub_sq = P_sub.multiply(P_sub)
                    avg_prox[idx] = np.asarray(P_sub_sq.sum(axis=1)).ravel()
                avg_prox[avg_prox == 0] = 1e-10
                for cls in classes:
                    idx = np.where(y_arr == cls)[0]
                    raw_scores[idx] = len(idx) / avg_prox[idx]

            scores = np.zeros(n_samples, dtype=np.float64)
            for cls in classes:
                idx = np.where(y_arr == cls)[0]
                class_raw = raw_scores[idx]
                med = np.median(class_raw)
                mad = np.median(np.abs(class_raw - med))
                if mad == 0:
                    scores[idx] = 0.0
                else:
                    scores[idx] = np.abs(class_raw - med) / mad

        if scaling == "log":
            scores = np.log1p(scores)
        elif scaling == "normalize":
            lo, hi = scores.min(), scores.max()
            if hi - lo > 0:
                scores = (scores - lo) / (hi - lo)
            else:
                scores = np.zeros_like(scores)

        if y_is_pandas:
            return pd.Series(scores, index=y_index, name="outlier_score")
        return scores

    def _get_prototypes(
        self,
        *,
        k: int = 10,
        n_prototypes: int = 1,
        categorical_features: Optional[Sequence[Any]] = None,
        standardize_continuous: bool = False,
    ) -> dict[Any, List[dict[str, Any]]]:
        """
        Compute classwise proximity prototypes following Breiman's description.

        For each class, the method searches over remaining cases of that class
        and selects the case whose ``k`` nearest available proximity neighbours
        contain the largest number of same-class points. The selected
        neighbourhood is then summarised feature-by-feature: continuous
        variables receive their 25th percentile, median, and 75th percentile;
        categorical variables receive their modal value plus the full category
        frequency table. Additional prototypes for the same class are obtained
        by removing the previously selected neighbourhood from consideration and
        repeating the search.

        Parameters
        ----------
        k : int, default 10
            Size of the proximity neighbourhood used to score candidate
            prototype centres and to summarise each selected prototype.
        n_prototypes : int, default 1
            Maximum number of prototypes to return per class.
        categorical_features : sequence of column names or positions, optional
            Features that should be treated as categorical even if they are
            numerically encoded. When omitted, non-numeric columns are inferred
            as categorical automatically.
        standardize_continuous : bool, default False
            When True, continuous prototype summaries are standardised by the
            fitted training column's 5th and 95th percentiles, matching
            Breiman's output convention.

        Returns
        -------
        prototypes : dict[class_label, list[dict[str, Any]]]
            Classwise prototype records. Each record contains:

            - ``prototype_number``: 1-based prototype index within the class.
            - ``center_position`` / ``center_index``: selected case.
            - ``neighbor_positions`` / ``neighbor_index``: selected
              neighbourhood.
            - ``same_class_neighbor_count`` and
              ``same_class_neighbor_fraction``.
            - ``prototype``: per-feature prototype values (median for
              continuous, mode for categorical).
            - ``continuous_summary``: DataFrame with ``q25``, ``median``,
              and ``q75`` rows for continuous variables.
            - ``categorical_summary``: DataFrame with modal category
              information.
            - ``categorical_frequencies``: DataFrame of all category counts
              and frequencies for categorical variables.
        """
        self._check_proximity_ready()

        if self._y_train is None or not hasattr(self, "classes_"):
            raise ValueError(
                "get_prototypes is only defined for classifiers with training "
                "class labels."
            )
        if self._X_train is None:
            raise RuntimeError(
                "Training features were not cached in a dense matrix, so "
                "prototypes are unavailable for this estimator fit."
            )
        if int(k) < 1:
            raise ValueError("k must be at least 1.")
        if int(n_prototypes) < 1:
            raise ValueError("n_prototypes must be at least 1.")

        X_train, _, X_index, X_columns = self._prepare_local_feature_matrix(None)
        train_labels = np.asarray(self._y_train)
        n_samples, n_features = X_train.shape
        feature_names = self._prototype_feature_names(n_features, X_columns)
        categorical_idx = self._resolve_prototype_categorical_indices(
            X_train, X_columns, categorical_features,
        )
        categorical_set = set(categorical_idx)
        continuous_idx = [
            feature_idx
            for feature_idx in range(n_features)
            if feature_idx not in categorical_set
        ]

        P = self._ensure_csr_matrix(self.get_proximities())
        results: dict[Any, List[dict[str, Any]]] = {}

        for cls in np.unique(train_labels):
            available_mask = np.ones(n_samples, dtype=bool)
            class_records: List[dict[str, Any]] = []

            for prototype_number in range(1, int(n_prototypes) + 1):
                class_candidates = np.flatnonzero((train_labels == cls) & available_mask)
                if class_candidates.size == 0:
                    break

                best_center: Optional[int] = None
                best_neighbors: Optional[np.ndarray] = None
                best_neighbor_values: Optional[np.ndarray] = None
                best_key: Optional[Tuple[int, float, float, int]] = None

                for center_position in class_candidates:
                    neighbor_positions, neighbor_values = self._prototype_neighbor_set(
                        P, int(center_position), available_mask, int(k),
                    )
                    if neighbor_positions.size == 0:
                        continue

                    same_class_mask = train_labels[neighbor_positions] == cls
                    same_class_count = int(same_class_mask.sum())
                    same_class_proximity = float(neighbor_values[same_class_mask].sum())
                    total_proximity = float(neighbor_values.sum())
                    key = (
                        same_class_count,
                        same_class_proximity,
                        total_proximity,
                        -int(center_position),
                    )
                    if best_key is None or key > best_key:
                        best_center = int(center_position)
                        best_neighbors = neighbor_positions
                        best_neighbor_values = neighbor_values
                        best_key = key

                if (
                    best_center is None
                    or best_neighbors is None
                    or best_neighbor_values is None
                ):
                    break

                continuous_summary, continuous_medians = self._summarize_prototype_continuous_features(
                    X_train,
                    best_neighbors,
                    continuous_idx,
                    feature_names,
                    standardize_continuous,
                )
                (
                    categorical_summary,
                    categorical_frequencies,
                    categorical_modes,
                ) = self._summarize_prototype_categorical_features(
                    X_train,
                    best_neighbors,
                    categorical_idx,
                    feature_names,
                )

                prototype_values: List[Any] = []
                for feature_idx, feature_name in enumerate(feature_names):
                    if feature_idx in categorical_set:
                        prototype_values.append(categorical_modes.get(feature_name, np.nan))
                    else:
                        prototype_values.append(continuous_medians.get(feature_name, np.nan))

                neighbor_index = (
                    np.array(best_neighbors, copy=True)
                    if X_index is None
                    else X_index.take(best_neighbors)
                )
                same_class_neighbor_count = int((train_labels[best_neighbors] == cls).sum())
                class_records.append(
                    {
                        "prototype_number": prototype_number,
                        "class_label": cls,
                        "center_position": best_center,
                        "center_index": (
                            int(best_center)
                            if X_index is None
                            else X_index[best_center]
                        ),
                        "neighbor_positions": np.array(best_neighbors, copy=True),
                        "neighbor_index": neighbor_index,
                        "neighbor_proximities": np.array(best_neighbor_values, copy=True),
                        "same_class_neighbor_count": same_class_neighbor_count,
                        "same_class_neighbor_fraction": (
                            same_class_neighbor_count / max(best_neighbors.size, 1)
                        ),
                        "prototype": pd.Series(
                            prototype_values,
                            index=pd.Index(feature_names, name="feature"),
                            name="prototype",
                        ),
                        "continuous_summary": continuous_summary,
                        "categorical_summary": categorical_summary,
                        "categorical_frequencies": categorical_frequencies,
                    }
                )

                available_mask[best_neighbors] = False

            results[cls] = class_records

        return results

    def _local_feature_importance(
        self,
        X: Optional[Any] = None,
        labels: Optional[Any] = None,
        *,
        perturbation: Literal["auto", "flip", "permute", "quantile"] = "auto",
        n_repeats: int = 30,
        random_state: Optional[Any] = None,
        return_closeness: bool = False,
        return_directionality: bool = False,
    ) -> Any:
        """
        Compute Whitmore et al. (2018) local feature contributions.

        The paper perturbs one feature at a time, recomputes the sample's
        proximity profile against the fixed training set, and compares the
        perturbed and unperturbed profiles with an in-group / out-group sign
        vector. Positive values mean the original feature value supported the
        focal class; negative values mean it pushed the sample away from that
        class.

        When ``return_directionality=True``, the same per-perturbation score is
        also split by whether the replacement value is numerically above or
        below the observed feature value. This tracks directional sensitivity
        in perturbation space for ordered numeric features without changing the
        default Whitmore et al. summary.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features), optional
            Query samples to explain. If omitted, the fitted training samples
            are explained.
        labels : array-like, Series, DataFrame, or None, optional
            Focal class for each query sample. When omitted, the estimator's
            predicted class is used, matching the paper's "decided class"
            interpretation. Passing the true label is useful for analyzing
            misclassifications.
        perturbation : {'auto', 'flip', 'permute', 'quantile'}, default 'auto'
            Feature perturbation strategy.

            - ``'flip'``: swap each binary value to the other observed level.
            - ``'permute'``: replace each feature value with a draw from the
              fitted training column.
            - ``'quantile'``: replace the feature with a succession of evenly
              spaced quantiles of the fitted training column.
            - ``'auto'``: use ``'flip'`` for binary columns, ``'quantile'``
              for non-binary numeric columns, and ``'permute'`` for
              non-numeric columns.
        n_repeats : int, default 30
            Number of perturbation draws per feature. Repeats are averaged.
            Ignored for deterministic binary flips. For
            ``perturbation='quantile'``, this is the number of evenly spaced
            quantiles used for perturbation.
        random_state : int, RandomState, or None, optional
            Controls reproducible feature permutations.
        return_closeness : bool, default False
            When True, also return the paper's "closeness" summaries as
            average squared perturbed proximities to the focal class and its
            complement.
        return_directionality : bool, default False
            When True, also return directional local summaries for numeric
            perturbations. Perturbations replacing a feature with a value above
            the observed value contribute to ``'upward'``; those below the
            observed value contribute to ``'downward'``. Non-numeric or
            unchanged perturbations are excluded from these directional
            summaries.

        Returns
        -------
        importance : np.ndarray or pandas.DataFrame of shape (n_samples, n_features)
            Local feature contributions for each sample and feature.
        closeness : dict[str, np.ndarray | pandas.DataFrame], optional
            Returned only when ``return_closeness=True`` with keys
            ``'in_group'`` and ``'out_group'``.
        directionality : dict[str, np.ndarray | pandas.DataFrame], optional
            Returned only when ``return_directionality=True`` with keys
            ``'upward'``, ``'downward'``, ``'upward_count'``, and
            ``'downward_count'``. The directional arrays are averages over
            perturbations in that direction for each sample-feature pair, and
            are ``NaN`` when no perturbations occurred in that direction.
        """
        self._check_proximity_ready()

        if self._y_train is None or not hasattr(self, "classes_"):
            raise ValueError(
                "local_feature_importance is only defined for classifiers with "
                "training class labels."
            )
        if perturbation not in ("auto", "flip", "permute", "quantile"):
            raise ValueError(
                "perturbation must be one of {'auto', 'flip', 'permute', 'quantile'}"
            )
        if int(n_repeats) < 1:
            raise ValueError("n_repeats must be at least 1")

        X_query, X_is_df, X_index, X_columns = self._prepare_local_feature_matrix(X)
        X_query_obj = self._materialize_local_feature_matrix(
            X_query, X_is_df, X_index, X_columns,
        )

        if self._X_train is None:
            raise RuntimeError(
                "Training features were not cached in a dense matrix, so local "
                "feature importance is unavailable for this estimator fit."
            )

        train_labels = np.asarray(self._y_train)
        query_labels = self._prepare_local_feature_labels(
            labels, X_query, X_query_obj,
        )

        groups: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = []
        for cls in np.unique(query_labels):
            row_idx = np.flatnonzero(query_labels == cls)
            in_mask = train_labels == cls
            if not np.any(in_mask):
                raise ValueError(
                    f"Focal class {cls!r} was not observed in the fitted training labels."
                )
            out_mask = ~in_mask
            groups.append(
                (
                    row_idx,
                    in_mask,
                    out_mask,
                    int(in_mask.sum()),
                    int(out_mask.sum()),
                )
            )

        P_base = self._ensure_csr_matrix(self.prox_extend(X_query_obj))

        n_samples, n_features = X_query.shape
        importance = np.zeros((n_samples, n_features), dtype=np.float64)

        closeness_in = np.zeros_like(importance) if return_closeness else None
        closeness_out = np.zeros_like(importance) if return_closeness else None
        directional_up_sum = np.zeros_like(importance) if return_directionality else None
        directional_down_sum = np.zeros_like(importance) if return_directionality else None
        directional_up_count = (
            np.zeros((n_samples, n_features), dtype=np.int64)
            if return_directionality
            else None
        )
        directional_down_count = (
            np.zeros((n_samples, n_features), dtype=np.int64)
            if return_directionality
            else None
        )

        rng = check_random_state(random_state)
        X_ref = self._X_train

        for feature_idx in range(n_features):
            query_col = X_query[:, feature_idx]
            ref_col = X_ref[:, feature_idx]
            mode = self._resolve_local_perturbation_mode(
                query_col, ref_col, perturbation,
            )
            if mode == "flip":
                perturbations = [self._flip_binary_feature(query_col, ref_col)]
            elif mode == "permute":
                perturbations = [
                    self._sample_permuted_feature(query_col, ref_col, rng)
                    for _ in range(int(n_repeats))
                ]
            else:
                perturbations = self._quantile_feature_perturbations(
                    query_col, ref_col, int(n_repeats),
                )
            repeats = len(perturbations)

            for perturbed_col in perturbations:
                if return_directionality:
                    upward_mask, downward_mask = self._directional_masks(
                        query_col, perturbed_col,
                    )
                else:
                    upward_mask = None
                    downward_mask = None

                X_perturbed = np.array(X_query, copy=True)
                X_perturbed[:, feature_idx] = perturbed_col
                X_perturbed_obj = self._materialize_local_feature_matrix(
                    X_perturbed, X_is_df, X_index, X_columns,
                )

                P_perturbed = self._ensure_csr_matrix(
                    self.prox_extend(X_perturbed_obj),
                )
                delta_sq = (P_perturbed - P_base).multiply(P_perturbed - P_base).tocsr()
                perturbed_sq = (
                    P_perturbed.multiply(P_perturbed).tocsr()
                    if return_closeness
                    else None
                )

                for row_idx, in_mask, out_mask, n_in, n_out in groups:
                    in_shift = np.asarray(
                        delta_sq[row_idx][:, in_mask].sum(axis=1),
                        dtype=np.float64,
                    ).ravel()
                    out_shift = np.asarray(
                        delta_sq[row_idx][:, out_mask].sum(axis=1),
                        dtype=np.float64,
                    ).ravel()
                    contribution = out_shift - in_shift

                    # Positive values indicate the original feature value
                    # supported the focal class more than its complement.
                    importance[row_idx, feature_idx] += contribution

                    if (
                        upward_mask is not None
                        and downward_mask is not None
                        and directional_up_sum is not None
                        and directional_down_sum is not None
                        and directional_up_count is not None
                        and directional_down_count is not None
                    ):
                        group_upward = upward_mask[row_idx]
                        if np.any(group_upward):
                            selected_rows = row_idx[group_upward]
                            directional_up_sum[selected_rows, feature_idx] += contribution[group_upward]
                            directional_up_count[selected_rows, feature_idx] += 1

                        group_downward = downward_mask[row_idx]
                        if np.any(group_downward):
                            selected_rows = row_idx[group_downward]
                            directional_down_sum[selected_rows, feature_idx] += contribution[group_downward]
                            directional_down_count[selected_rows, feature_idx] += 1

                    if perturbed_sq is not None and closeness_in is not None and closeness_out is not None:
                        closeness_in[row_idx, feature_idx] += (
                            np.asarray(
                                perturbed_sq[row_idx][:, in_mask].sum(axis=1),
                                dtype=np.float64,
                            ).ravel()
                            / max(n_in, 1)
                        )
                        if n_out > 0:
                            closeness_out[row_idx, feature_idx] += (
                                np.asarray(
                                    perturbed_sq[row_idx][:, out_mask].sum(axis=1),
                                    dtype=np.float64,
                                ).ravel()
                                / n_out
                            )

            importance[:, feature_idx] /= repeats
            if closeness_in is not None and closeness_out is not None:
                closeness_in[:, feature_idx] /= repeats
                closeness_out[:, feature_idx] /= repeats

        wrapped_importance = self._wrap_local_feature_output(
            importance, X_is_df, X_index, X_columns,
        )
        wrapped_directionality = None
        if (
            return_directionality
            and directional_up_sum is not None
            and directional_down_sum is not None
            and directional_up_count is not None
            and directional_down_count is not None
        ):
            upward = np.full_like(importance, np.nan)
            downward = np.full_like(importance, np.nan)

            valid_upward = directional_up_count > 0
            valid_downward = directional_down_count > 0
            upward[valid_upward] = (
                directional_up_sum[valid_upward] / directional_up_count[valid_upward]
            )
            downward[valid_downward] = (
                directional_down_sum[valid_downward] / directional_down_count[valid_downward]
            )

            wrapped_directionality = {
                "upward": self._wrap_local_feature_output(
                    upward, X_is_df, X_index, X_columns,
                ),
                "downward": self._wrap_local_feature_output(
                    downward, X_is_df, X_index, X_columns,
                ),
                "upward_count": self._wrap_local_feature_output(
                    directional_up_count, X_is_df, X_index, X_columns,
                ),
                "downward_count": self._wrap_local_feature_output(
                    directional_down_count, X_is_df, X_index, X_columns,
                ),
            }

        if not return_closeness or closeness_in is None or closeness_out is None:
            if wrapped_directionality is not None:
                return wrapped_importance, wrapped_directionality
            return wrapped_importance

        wrapped_closeness = {
            "in_group": self._wrap_local_feature_output(
                closeness_in, X_is_df, X_index, X_columns,
            ),
            "out_group": self._wrap_local_feature_output(
                closeness_out, X_is_df, X_index, X_columns,
            ),
        }
        if wrapped_directionality is not None:
            return wrapped_importance, wrapped_closeness, wrapped_directionality
        return wrapped_importance, wrapped_closeness

    def _local_variable_importance(self, *args: Any, **kwargs: Any) -> Any:
        """Alias for :meth:`local_feature_importance`."""
        return self._local_feature_importance(*args, **kwargs)

    def _check_proximity_ready(self) -> None:
        check_is_fitted(cast(Any, self))
        if self.W_mat is None or self.leaf_matrix_ is None or self._leaf_encoding is None:
            raise RuntimeError("Proximity structures are not initialized. Call fit(X, y) first.")

    def _store_training_features(self, X: Any) -> None:
        if sparse.issparse(X):
            self._X_train_native = X.copy()
            self._X_train = None
            self._X_train_index_ = None
            self._X_train_columns_ = None
            self._X_train_was_dataframe_ = False
            return

        if isinstance(X, pd.DataFrame):
            self._X_train_native = X.copy(deep=True)
            self._X_train = X.to_numpy(copy=True)
            self._X_train_index_ = X.index.copy()
            self._X_train_columns_ = X.columns.copy()
            self._X_train_was_dataframe_ = True
            return

        if (
            type(X).__module__.split(".", 1)[0] == "catboost"
            and type(X).__name__ == "Pool"
        ):
            # Pool carries categorical/text feature metadata required by native
            # CatBoost leaf inference.  It is immutable from this API's point
            # of view, so retain it rather than coercing it to an object array.
            self._X_train_native = X
            try:
                pool_features = np.asarray(X.get_features())
            except Exception:
                pool_features = None
            self._X_train = pool_features
            self._X_train_index_ = None
            self._X_train_columns_ = None
            self._X_train_was_dataframe_ = False
            return

        X_arr = np.array(X, copy=True)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        self._X_train = X_arr
        self._X_train_native = X_arr
        self._X_train_index_ = None
        self._X_train_columns_ = None
        self._X_train_was_dataframe_ = False

    def _relation_training_features(self) -> Any | None:
        """Return training features in the estimator's native input format."""

        if self._X_train_native is not None:
            return self._X_train_native
        return self._X_train

    def _prepare_local_feature_matrix(
        self,
        X: Optional[Any],
    ) -> Tuple[np.ndarray, bool, Optional[pd.Index], Optional[pd.Index]]:
        if X is None:
            if self._X_train is None:
                raise RuntimeError(
                    "local_feature_importance requires cached dense training features "
                    "when X is omitted."
                )
            return (
                np.array(self._X_train, copy=True),
                self._X_train_was_dataframe_,
                self._X_train_index_,
                self._X_train_columns_,
            )

        if sparse.issparse(X):
            raise TypeError(
                "local_feature_importance currently requires dense feature matrices."
            )

        if isinstance(X, pd.DataFrame):
            return X.to_numpy(copy=True), True, X.index.copy(), X.columns.copy()

        X_arr = np.array(X, copy=True)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return X_arr, False, None, None

    @staticmethod
    def _materialize_local_feature_matrix(
        X: np.ndarray,
        X_is_df: bool,
        X_index: Optional[pd.Index],
        X_columns: Optional[pd.Index],
    ) -> Any:
        if X_is_df:
            return pd.DataFrame(X, index=X_index, columns=X_columns)
        return X

    def _prepare_local_feature_labels(
        self,
        labels: Optional[Any],
        X_query: np.ndarray,
        X_query_obj: Any,
    ) -> np.ndarray:
        if labels is None:
            labels_arr = np.asarray(self.predict(X_query_obj))
        else:
            labels_arr = np.asarray(labels)

        if labels_arr.ndim == 2:
            if labels_arr.shape[1] != 1:
                raise ValueError("labels must be one-dimensional.")
            labels_arr = labels_arr.ravel()
        elif labels_arr.ndim != 1:
            raise ValueError("labels must be one-dimensional.")

        if labels_arr.shape[0] != X_query.shape[0]:
            raise ValueError(
                "labels must have one entry per query sample; got "
                f"{labels_arr.shape[0]} labels for {X_query.shape[0]} samples."
            )
        return labels_arr

    @staticmethod
    def _ensure_csr_matrix(P: MatrixLike) -> sparse.csr_matrix:
        if sparse.issparse(P):
            return P.tocsr()
        return sparse.csr_matrix(np.asarray(P))

    @staticmethod
    def _wrap_local_feature_output(
        values: np.ndarray,
        X_is_df: bool,
        X_index: Optional[pd.Index],
        X_columns: Optional[pd.Index],
    ) -> Any:
        if X_is_df:
            return pd.DataFrame(values, index=X_index, columns=X_columns)
        return values

    @staticmethod
    def _prototype_feature_names(
        n_features: int,
        X_columns: Optional[pd.Index],
    ) -> List[Any]:
        if X_columns is not None:
            return list(X_columns)
        return [f"feature_{idx}" for idx in range(n_features)]

    @staticmethod
    def _resolve_prototype_categorical_indices(
        X_train: np.ndarray,
        X_columns: Optional[pd.Index],
        categorical_features: Optional[Sequence[Any]],
    ) -> List[int]:
        inferred = {
            feature_idx
            for feature_idx in range(X_train.shape[1])
            if not Proximity._is_numeric_local_feature(X_train[:, feature_idx])
        }
        if categorical_features is None:
            return sorted(inferred)

        resolved = set(inferred)
        for feature in categorical_features:
            if isinstance(feature, (int, np.integer)):
                feature_idx = int(feature)
                if feature_idx < 0 or feature_idx >= X_train.shape[1]:
                    raise ValueError(
                        f"Categorical feature index {feature_idx} is out of bounds for "
                        f"{X_train.shape[1]} features."
                    )
                resolved.add(feature_idx)
                continue

            if X_columns is None:
                raise ValueError(
                    "String categorical feature names require fitting with a DataFrame "
                    "or passing integer column positions."
                )
            matches = np.flatnonzero(X_columns == feature)
            if matches.size == 0:
                raise ValueError(
                    f"Categorical feature {feature!r} was not found in the fitted columns."
                )
            resolved.add(int(matches[0]))

        return sorted(resolved)

    @staticmethod
    def _prototype_neighbor_set(
        P: sparse.csr_matrix,
        row_idx: int,
        available_mask: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if k < 1 or row_idx < 0 or row_idx >= P.shape[0] or not available_mask[row_idx]:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )

        row = P.getrow(row_idx)
        available_count = int(available_mask.sum())
        k_effective = min(int(k), available_count)

        selected_positions = [int(row_idx)]
        self_mask = row.indices == row_idx
        if np.any(self_mask):
            self_value = float(row.data[self_mask][0])
        else:
            row_max = float(row.data.max()) if row.nnz > 0 else 0.0
            self_value = float(np.nextafter(max(row_max, 0.0), np.inf))
        selected_values = [self_value]

        mask = available_mask[row.indices]
        candidate_positions = row.indices[mask]
        candidate_values = row.data[mask]

        nonself_mask = candidate_positions != row_idx
        candidate_positions = candidate_positions[nonself_mask]
        candidate_values = candidate_values[nonself_mask]

        if candidate_positions.size > 0:
            order = np.lexsort((candidate_positions, -candidate_values))
            candidate_positions = candidate_positions[order]
            candidate_values = candidate_values[order]

        remaining = k_effective - 1
        take = min(remaining, candidate_positions.size)
        if take > 0:
            selected_positions.extend(candidate_positions[:take].tolist())
            selected_values.extend(candidate_values[:take].astype(np.float64, copy=False).tolist())

        if len(selected_positions) < k_effective:
            used = np.zeros_like(available_mask, dtype=bool)
            used[np.asarray(selected_positions, dtype=np.int64)] = True
            zero_fill = np.flatnonzero(available_mask & ~used)
            needed = k_effective - len(selected_positions)
            if needed > 0 and zero_fill.size > 0:
                fill = zero_fill[:needed]
                selected_positions.extend(fill.tolist())
                selected_values.extend([0.0] * fill.size)

        return (
            np.asarray(selected_positions, dtype=np.int64),
            np.asarray(selected_values, dtype=np.float64),
        )

    @staticmethod
    def _summarize_prototype_continuous_features(
        X_train: np.ndarray,
        neighbor_positions: np.ndarray,
        continuous_idx: Sequence[int],
        feature_names: Sequence[Any],
        standardize_continuous: bool,
    ) -> Tuple[pd.DataFrame, dict[Any, float]]:
        rows: List[dict[str, Any]] = []
        medians: dict[Any, float] = {}

        for feature_idx in continuous_idx:
            feature_name = feature_names[feature_idx]
            feature_values = pd.to_numeric(
                pd.Series(X_train[neighbor_positions, feature_idx]),
                errors="coerce",
            ).to_numpy(dtype=np.float64, copy=False)
            feature_values = feature_values[~np.isnan(feature_values)]

            if feature_values.size == 0:
                q25 = median = q75 = np.nan
            else:
                q25, median, q75 = np.quantile(feature_values, [0.25, 0.5, 0.75])

            if standardize_continuous:
                ref_values = pd.to_numeric(
                    pd.Series(X_train[:, feature_idx]),
                    errors="coerce",
                ).to_numpy(dtype=np.float64, copy=False)
                ref_values = ref_values[~np.isnan(ref_values)]
                if ref_values.size > 0:
                    ref_q05, ref_q95 = np.quantile(ref_values, [0.05, 0.95])
                    scale = ref_q95 - ref_q05
                    q25 = q25 - ref_q05
                    median = median - ref_q05
                    q75 = q75 - ref_q05
                    if scale > 0:
                        q25 /= scale
                        median /= scale
                        q75 /= scale

            rows.append(
                {
                    "feature": feature_name,
                    "q25": float(q25) if np.isfinite(q25) else np.nan,
                    "median": float(median) if np.isfinite(median) else np.nan,
                    "q75": float(q75) if np.isfinite(q75) else np.nan,
                }
            )
            medians[feature_name] = float(median) if np.isfinite(median) else np.nan

        if rows:
            summary = pd.DataFrame(rows).set_index("feature")
        else:
            summary = pd.DataFrame(columns=["q25", "median", "q75"])
            summary.index = pd.Index([], name="feature")
        return summary, medians

    @staticmethod
    def _summarize_prototype_categorical_features(
        X_train: np.ndarray,
        neighbor_positions: np.ndarray,
        categorical_idx: Sequence[int],
        feature_names: Sequence[Any],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict[Any, Any]]:
        summary_rows: List[dict[str, Any]] = []
        frequency_rows: List[dict[str, Any]] = []
        modes: dict[Any, Any] = {}
        neighborhood_size = max(int(neighbor_positions.size), 1)

        for feature_idx in categorical_idx:
            feature_name = feature_names[feature_idx]
            counts = pd.Series(X_train[neighbor_positions, feature_idx]).value_counts(dropna=False)
            if counts.empty:
                mode_value = np.nan
                mode_count = 0
            else:
                mode_count = int(counts.max())
                mode_candidates = [
                    value for value, count in counts.items()
                    if int(count) == mode_count
                ]
                mode_value = min(
                    mode_candidates,
                    key=Proximity._prototype_category_sort_key,
                )

            modes[feature_name] = mode_value
            summary_rows.append(
                {
                    "feature": feature_name,
                    "mode": mode_value,
                    "mode_count": mode_count,
                    "mode_frequency": mode_count / neighborhood_size,
                }
            )
            for category, count in counts.items():
                frequency_rows.append(
                    {
                        "feature": feature_name,
                        "category": category,
                        "count": int(count),
                        "frequency": int(count) / neighborhood_size,
                    }
                )

        if summary_rows:
            summary = pd.DataFrame(summary_rows).set_index("feature")
        else:
            summary = pd.DataFrame(columns=["mode", "mode_count", "mode_frequency"])
            summary.index = pd.Index([], name="feature")

        if frequency_rows:
            frequencies = pd.DataFrame(frequency_rows).set_index(["feature", "category"])
        else:
            frequencies = pd.DataFrame(columns=["count", "frequency"])
            frequencies.index = pd.MultiIndex.from_arrays(
                [[], []], names=["feature", "category"],
            )

        return summary, frequencies, modes

    @staticmethod
    def _prototype_category_sort_key(value: Any) -> Tuple[str, str]:
        return (type(value).__name__, repr(value))

    @staticmethod
    def _resolve_local_perturbation_mode(
        query_col: np.ndarray,
        ref_col: np.ndarray,
        perturbation: Literal["auto", "flip", "permute", "quantile"],
    ) -> Literal["flip", "permute", "quantile"]:
        if perturbation != "auto":
            return perturbation

        combined = np.concatenate([np.asarray(ref_col).ravel(), np.asarray(query_col).ravel()])
        unique_nonmissing = pd.unique(combined[~pd.isna(combined)])
        if unique_nonmissing.size == 2:
            return "flip"
        return "quantile" if Proximity._is_numeric_local_feature(combined) else "permute"

    @staticmethod
    def _is_numeric_local_feature(values: np.ndarray) -> bool:
        try:
            numeric = pd.to_numeric(pd.Series(np.asarray(values).ravel()), errors="raise")
        except Exception:
            return False

        numeric_values = numeric.to_numpy(dtype=np.float64, copy=False)
        return np.any(~np.isnan(numeric_values))

    @staticmethod
    def _flip_binary_feature(query_col: np.ndarray, ref_col: np.ndarray) -> np.ndarray:
        combined = np.concatenate([np.asarray(ref_col).ravel(), np.asarray(query_col).ravel()])
        unique_nonmissing = pd.unique(combined[~pd.isna(combined)])
        if unique_nonmissing.size != 2:
            raise ValueError(
                "Binary flip perturbation requires exactly two observed feature values."
            )

        lo, hi = unique_nonmissing[0], unique_nonmissing[1]
        flipped = np.array(query_col, copy=True)
        for idx, value in enumerate(query_col):
            if pd.isna(value):
                flipped[idx] = lo
            elif Proximity._scalar_values_equal(value, lo):
                flipped[idx] = hi
            elif Proximity._scalar_values_equal(value, hi):
                flipped[idx] = lo
            else:
                flipped[idx] = lo
        return flipped

    @staticmethod
    def _sample_permuted_feature(
        query_col: np.ndarray,
        ref_col: np.ndarray,
        rng: Any,
    ) -> np.ndarray:
        ref_arr = np.asarray(ref_col).ravel()
        if ref_arr.size == 0:
            return np.array(query_col, copy=True)

        sampled = np.array(
            ref_arr[rng.randint(0, ref_arr.shape[0], size=query_col.shape[0])],
            copy=True,
        )
        unique_nonmissing = pd.unique(ref_arr[~pd.isna(ref_arr)])
        if unique_nonmissing.size <= 1:
            return sampled

        for idx, value in enumerate(query_col):
            if not Proximity._scalar_values_equal(sampled[idx], value):
                continue

            alternatives = [
                candidate
                for candidate in unique_nonmissing
                if not Proximity._scalar_values_equal(candidate, value)
            ]
            if alternatives:
                sampled[idx] = alternatives[rng.randint(0, len(alternatives))]
        return sampled

    @staticmethod
    def _quantile_feature_perturbations(
        query_col: np.ndarray,
        ref_col: np.ndarray,
        n_quantiles: int,
    ) -> List[np.ndarray]:
        if n_quantiles < 1:
            raise ValueError("n_repeats must be at least 1 for quantile perturbation.")

        try:
            ref_numeric = pd.to_numeric(pd.Series(np.asarray(ref_col).ravel()), errors="raise")
        except Exception as exc:
            raise TypeError(
                "Quantile perturbation requires numeric training feature columns."
            ) from exc

        ref_values = ref_numeric.to_numpy(dtype=np.float64, copy=False)
        ref_values = ref_values[~np.isnan(ref_values)]
        if ref_values.size == 0:
            return [np.array(query_col, copy=True) for _ in range(n_quantiles)]

        if n_quantiles == 1:
            quantile_levels = np.array([0.5], dtype=np.float64)
        else:
            quantile_levels = np.linspace(0.0, 1.0, n_quantiles, dtype=np.float64)

        perturbations: List[np.ndarray] = []
        for level in quantile_levels:
            quantile_value = float(np.quantile(ref_values, level))
            perturbations.append(
                np.full(query_col.shape[0], quantile_value, dtype=np.float64)
            )
        return perturbations

    @staticmethod
    def _directional_masks(
        query_col: np.ndarray,
        perturbed_col: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        query_numeric = pd.to_numeric(
            pd.Series(np.asarray(query_col).ravel()), errors="coerce",
        )
        perturbed_numeric = pd.to_numeric(
            pd.Series(np.asarray(perturbed_col).ravel()), errors="coerce",
        )

        query_values = query_numeric.to_numpy(dtype=np.float64, copy=False)
        perturbed_values = perturbed_numeric.to_numpy(dtype=np.float64, copy=False)
        valid = ~(np.isnan(query_values) | np.isnan(perturbed_values))

        upward = np.zeros(query_values.shape[0], dtype=bool)
        downward = np.zeros(query_values.shape[0], dtype=bool)
        upward[valid] = perturbed_values[valid] > query_values[valid]
        downward[valid] = perturbed_values[valid] < query_values[valid]
        return upward, downward

    @staticmethod
    def _scalar_values_equal(lhs: Any, rhs: Any) -> bool:
        lhs_missing = pd.isna(lhs)
        rhs_missing = pd.isna(rhs)
        if bool(lhs_missing) and bool(rhs_missing):
            return True
        if bool(lhs_missing) or bool(rhs_missing):
            return False
        try:
            return bool(lhs == rhs)
        except Exception:
            return False

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

    def _validate_honest_gap_inputs(self) -> None:
        if not _is_honest_forest_instance(self):
            raise ValueError(
                "prox_method='honest_gap' requires a fitted HonestForestClassifier-like "
                "estimator exposing honest_indices_, structure_indices_, estimators_, "
                "classes_, and predict_proba."
            )
        if self._y_train is None:
            raise ValueError("prox_method='honest_gap' requires training labels.")
        if np.asarray(self._y_train).ndim != 1:
            raise ValueError("prox_method='honest_gap' currently supports single-output classification only.")
        classes = getattr(self, "classes_", None)
        if classes is None or isinstance(classes, list) or np.asarray(classes).ndim != 1:
            raise ValueError("prox_method='honest_gap' currently supports single-output classification only.")
        if not callable(getattr(self, "predict_proba", None)):
            raise ValueError("prox_method='honest_gap' requires a classifier with predict_proba.")

        estimators = getattr(self, "estimators_", None)
        honest_indices = getattr(self, "honest_indices_", None)
        structure_indices = getattr(self, "structure_indices_", None)
        if estimators is None or honest_indices is None or structure_indices is None:
            raise ValueError(
                "prox_method='honest_gap' requires fitted honest_indices_, "
                "structure_indices_, and estimators_."
            )
        if len(honest_indices) != len(estimators) or len(structure_indices) != len(estimators):
            raise ValueError(
                "prox_method='honest_gap' requires one honest_indices_ and "
                "structure_indices_ entry per fitted estimator."
            )

    @staticmethod
    def _valid_index_array(indices: Any, n_samples: int) -> np.ndarray:
        if indices is None:
            return np.zeros(0, dtype=np.int64)
        arr = np.asarray(indices, dtype=np.int64).ravel()
        if arr.size == 0:
            return arr
        return arr[(arr >= 0) & (arr < n_samples)]

    @staticmethod
    def _class_indices_for_values(values: np.ndarray, classes: np.ndarray) -> np.ndarray:
        mapping = {cls: i for i, cls in enumerate(classes.tolist())}
        indices = np.empty(values.shape[0], dtype=np.int64)
        for i, value in enumerate(values):
            try:
                indices[i] = mapping[value]
            except KeyError:
                matches = np.flatnonzero(classes == value)
                if matches.size == 0:
                    raise ValueError("Training labels contain values not present in classes_.")
                indices[i] = int(matches[0])
        return indices

    def _honest_gap_oob_stats_from_indices(
        self,
        n_samples: int,
        n_trees: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        honest_indices = getattr(self, "honest_indices_", None)
        structure_indices = getattr(self, "structure_indices_", None)
        if honest_indices is None or structure_indices is None:
            raise ValueError("prox_method='honest_gap' requires honest_indices_ and structure_indices_.")

        oob_mask = np.ones((n_samples, n_trees), dtype=np.int8)
        in_bag_counts = np.zeros((n_samples, n_trees), dtype=np.float32)

        for t in range(n_trees):
            honest_idx = self._valid_index_array(honest_indices[t], n_samples)
            structure_idx = self._valid_index_array(structure_indices[t], n_samples)
            if honest_idx.size == 0 and structure_idx.size == 0:
                continue

            sampled_idx = np.concatenate([honest_idx, structure_idx])
            oob_mask[np.unique(sampled_idx), t] = 0
            idx, cnt = np.unique(sampled_idx, return_counts=True)
            in_bag_counts[idx, t] = cnt.astype(np.float32)

        return oob_mask, in_bag_counts

    def _tree_leaf_class_value_matrix(self, estimator: Any, forest_classes: np.ndarray) -> np.ndarray:
        tree = getattr(estimator, "tree_", None)
        values = getattr(tree, "value", None)
        if values is None:
            raise ValueError("prox_method='honest_gap' requires each estimator to expose tree_.value.")

        value_arr = np.asarray(values, dtype=np.float64)
        if value_arr.ndim == 3:
            if value_arr.shape[1] != 1:
                raise ValueError("prox_method='honest_gap' currently supports single-output classification only.")
            value_arr = value_arr[:, 0, :]
        elif value_arr.ndim != 2:
            raise ValueError("Unexpected tree_.value shape for prox_method='honest_gap'.")

        tree_classes_raw = getattr(estimator, "classes_", forest_classes)
        if isinstance(tree_classes_raw, list):
            if len(tree_classes_raw) != 1:
                raise ValueError("prox_method='honest_gap' currently supports single-output classification only.")
            tree_classes_raw = tree_classes_raw[0]
        tree_classes = np.asarray(tree_classes_raw).ravel()

        if (
            value_arr.shape[1] == forest_classes.shape[0]
            and tree_classes.shape[0] == forest_classes.shape[0]
            and np.array_equal(tree_classes, forest_classes)
        ):
            return value_arr

        aligned = np.zeros((value_arr.shape[0], forest_classes.shape[0]), dtype=np.float64)
        n_cols = min(value_arr.shape[1], tree_classes.shape[0])
        for src_col, cls_value in enumerate(tree_classes[:n_cols]):
            matches = np.flatnonzero(forest_classes == cls_value)
            if matches.size > 0:
                aligned[:, int(matches[0])] = value_arr[:, src_col]
        return aligned

    def _build_honest_gap_W_matrix(self) -> sparse.csr_matrix:
        if self.leaf_matrix_ is None or self._leaf_encoding is None:
            raise RuntimeError("Cannot build honest_gap W before leaf structures are initialized.")
        if self._y_train is None:
            raise ValueError("prox_method='honest_gap' requires training labels.")

        classes = np.asarray(getattr(self, "classes_", None))
        y_train = np.asarray(self._y_train)
        y_idx = self._class_indices_for_values(y_train, classes)

        estimators = getattr(self, "estimators_", None)
        honest_indices = getattr(self, "honest_indices_", None)
        structure_indices = getattr(self, "structure_indices_", None)
        if estimators is None or honest_indices is None or structure_indices is None:
            raise ValueError("prox_method='honest_gap' requires fitted honest forest indices.")

        n_samples, n_trees = self.leaf_matrix_.shape
        total_cols = int(self._leaf_encoding.total_features)
        kernel_method = bool(getattr(self, "kernel_method", True))

        all_rows: List[np.ndarray] = []
        all_cols: List[np.ndarray] = []
        all_weights: List[np.ndarray] = []

        for t in range(n_trees):
            honest_idx = self._valid_index_array(honest_indices[t], n_samples)
            structure_idx = self._valid_index_array(structure_indices[t], n_samples)
            if structure_idx.size > 0 and honest_idx.size > 0:
                honest_idx = np.setdiff1d(honest_idx, structure_idx, assume_unique=False)
            if honest_idx.size == 0:
                continue

            leaf_ids = self.leaf_matrix_[honest_idx, t].astype(np.int64, copy=False)
            known = self._leaf_encoding.values_per_tree[t]
            pos = np.searchsorted(known, leaf_ids)
            in_range = pos < known.shape[0]
            eq = np.zeros_like(in_range, dtype=bool)
            idx_in = np.nonzero(in_range)[0]
            if idx_in.size > 0:
                eq[idx_in] = known[pos[idx_in]] == leaf_ids[idx_in]
            valid = in_range & eq
            if not np.any(valid):
                continue

            honest_idx = honest_idx[valid]
            leaf_ids = leaf_ids[valid]
            cols = self._leaf_encoding.offsets[t] + pos[valid]
            leaf_values = self._tree_leaf_class_value_matrix(estimators[t], classes)
            node_valid = (leaf_ids >= 0) & (leaf_ids < leaf_values.shape[0])
            if not np.any(node_valid):
                continue

            honest_idx = honest_idx[node_valid]
            leaf_ids = leaf_ids[node_valid]
            cols = cols[node_valid].astype(np.int64, copy=False)
            y_idx_t = y_idx[honest_idx]

            if not kernel_method:
                leaf_totals = leaf_values.sum(axis=1, keepdims=True)
                leaf_values = np.divide(
                    leaf_values,
                    leaf_totals,
                    out=np.zeros_like(leaf_values),
                    where=leaf_totals != 0,
                )

            for col in np.unique(cols):
                in_leaf = cols == col
                if not np.any(in_leaf):
                    continue
                node_id = int(leaf_ids[np.nonzero(in_leaf)[0][0]])
                class_ids = np.unique(y_idx_t[in_leaf])
                for class_id in class_ids:
                    class_mask = in_leaf & (y_idx_t == class_id)
                    sample_rows = honest_idx[class_mask]
                    if sample_rows.size == 0:
                        continue
                    mass = float(leaf_values[node_id, class_id])
                    if not np.isfinite(mass) or mass <= 0.0:
                        continue
                    all_rows.append(sample_rows.astype(np.int64, copy=False))
                    all_cols.append(np.full(sample_rows.shape[0], int(col), dtype=np.int64))
                    all_weights.append(
                        np.full(sample_rows.shape[0], mass / sample_rows.shape[0], dtype=np.float64)
                    )

        if not all_rows:
            return sparse.csr_matrix((n_samples, total_cols), dtype=np.float64)

        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        weights = np.concatenate(all_weights)
        keep_nonzero = weights > 0
        return sparse.csr_matrix(
            (weights[keep_nonzero], (rows[keep_nonzero], cols[keep_nonzero])),
            shape=(n_samples, total_cols),
            dtype=np.float64,
        )

    def _build_W_matrix(self) -> None:
        if self.leaf_matrix_ is None or self._leaf_encoding is None:
            raise RuntimeError("Cannot build W before leaf structures are initialized.")

        n_samples, n_trees = self.leaf_matrix_.shape
        if self.prox_method == "honest_gap":
            self.W_mat = self._build_honest_gap_W_matrix()
            self._n_total_features = int(self._leaf_encoding.total_features)
            self._diag_offset = None
            return

        if self.prox_method == "conditional":
            # Meinshausen-style conditional-distribution weights: a query has
            # mass 1/T in each tree and training observations split that mass
            # uniformly within the query's terminal leaf.
            self.W_mat = self._build_W_uniform(
                self.leaf_matrix_, self._leaf_encoding,
            )
            self._n_total_features = int(self._leaf_encoding.total_features)
            self._diag_offset = None
            return

        rows, cols, tree_ids = self._map_leaves_to_global(self.leaf_matrix_)
        if rows.size == 0:
            raise RuntimeError("No valid leaf mappings found while building W matrix.")

        total_cols = int(self._leaf_encoding.total_features)

        if self.prox_method in {
            "original", "collision", "survival_leaf",
            "kerf", "path", "englund", "random_height",
        }:
            weights = np.full(rows.shape[0], 1.0 / np.sqrt(n_trees), dtype=np.float32)

        elif self.prox_method == "boosted":
            if self.tree_weights_ is None:
                raise RuntimeError("tree_weights_ must be available for relation='boosted'.")
            weights = np.sqrt(self.tree_weights_[tree_ids]).astype(np.float32)

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

        if self.prox_method == "conditional":
            vals = np.full(rows.shape[0], 1.0 / n_trees, dtype=np.float32)

        elif self.prox_method in {
            "original", "collision", "survival_leaf",
            "kerf", "path", "englund", "random_height",
        }:
            vals = np.full(rows.shape[0], 1.0 / np.sqrt(n_trees), dtype=np.float32)

        elif self.prox_method == "boosted":
            if self.tree_weights_ is None:
                raise RuntimeError("tree_weights_ must be available for relation='boosted'.")
            vals = np.sqrt(self.tree_weights_[tree_ids]).astype(np.float32)

        elif self.prox_method == "oob":
            vals = np.full(rows.shape[0], np.sqrt(n_trees) / n_trees, dtype=np.float32)

        elif self.prox_method in ("rfgap", "honest_gap"):
            if is_training:
                if self.oob_mask_ is None:
                    raise RuntimeError(f"oob_mask_ must be available for prox_method='{self.prox_method}'.")

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
            transformed = rotation_forest_transforms(self.estimator, X)
            leaves = [
                surrogate.apply(X_tree)
                for surrogate, X_tree in zip(surrogate_estimators, transformed)
            ]

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
            train_estimate_fn = getattr(self, "_train_probas_for_estimator", None)
            if not callable(train_estimate_fn):
                train_estimate_fn = getattr(self, "_train_preds_for_estimator", None)
            required = (
                X is not None,
                y is not None,
                callable(train_estimate_fn),
                hasattr(self, "_base_estimator"),
                hasattr(self, "n_cases_"),
            )
            if not all(required):
                raise RuntimeError(
                    "RotationForest OOB/RF-GAP requires aeon's private "
                    "training-estimate contract (_train_*_for_estimator), "
                    "fitted rotations, and the original training labels."
                )

            y_arr = np.asarray(y)
            n_cases = int(self.n_cases_)
            if n_cases != n_samples:
                raise RuntimeError("RotationForest n_cases_ does not match attached training rows.")

            with _suppress_base_validate_data_futurewarning():
                transformed = rotation_forest_transforms(self.estimator, X)
            if len(transformed) != n_trees:
                raise RuntimeError("RotationForest transformed-tree inventory changed after fitting.")

            # aeon's fit and train-estimate routines advance from the parent
            # estimator seed.  Proximity.random_state controls relation cuts,
            # not this reconstructed resampling protocol.
            rng = check_random_state(getattr(self.estimator, "random_state", None))
            max_int = np.iinfo(np.int32).max
            surrogate_estimators: List[Any] = []
            surrogate_leaves: List[np.ndarray] = []

            for t in range(n_trees):
                est_seed = int(rng.randint(max_int))
                est_rng = check_random_state(est_seed)

                # This call is the contract source of truth for aeon's OOB
                # role assignment.  Reconstruct the matching bootstrap sample
                # from the same seed, then refuse to continue if that private
                # contract no longer agrees with the reconstructed roles.
                with _suppress_base_validate_data_futurewarning():
                    train_result = train_estimate_fn(
                        transformed, y_arr, t, check_random_state(est_seed),
                    )
                if not isinstance(train_result, (tuple, list)) or len(train_result) != 2:
                    raise RuntimeError(
                        "RotationForest training-estimate function returned an unexpected result."
                    )
                oob_idx = np.asarray(train_result[1], dtype=np.int64)
                if np.any((oob_idx < 0) | (oob_idx >= n_cases)):
                    raise RuntimeError("RotationForest training-estimate returned invalid OOB indices.")

                sampled = est_rng.choice(n_cases, size=n_cases, replace=True)
                expected_oob = np.setdiff1d(np.arange(n_cases), np.unique(sampled))
                if not np.array_equal(np.sort(oob_idx), expected_oob):
                    raise RuntimeError(
                        "RotationForest training-estimate resampling no longer matches "
                        "the synchronized aeon seed contract."
                    )

                surrogate = clone(self._base_estimator)
                if hasattr(surrogate, "random_state"):
                    surrogate.random_state = est_rng
                surrogate.fit(transformed[t][sampled], y_arr[sampled])
                surrogate_estimators.append(surrogate)
                surrogate_leaves.append(np.asarray(surrogate.apply(transformed[t])))

                idx, cnt = np.unique(sampled, return_counts=True)
                in_bag_counts[idx, t] = cnt.astype(np.float32)
                oob_mask[oob_idx, t] = 1

            self._oob_surrogate_estimators_ = surrogate_estimators
            self._oob_surrogate_leaf_matrix_ = np.column_stack(surrogate_leaves)
            return oob_mask, in_bag_counts

        estimators_samples = getattr(self, "estimators_samples_", None)
        if estimators_samples is not None:
            for t, sample_indices in enumerate(estimators_samples):
                sample_indices = np.asarray(sample_indices, dtype=np.int64)
                # SyntheticNullForest trees are fitted on observed rows followed
                # by generated-null rows.  The indexed relation intentionally
                # contains observed rows only.
                sample_indices = sample_indices[
                    (sample_indices >= 0) & (sample_indices < n_samples)
                ]
                idx, cnt = np.unique(sample_indices, return_counts=True)
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
                # Some treeple forests (e.g. UnsupervisedRandomForest) expose
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

    @classmethod
    def _csr_row_normalize_inplace(cls, A: sparse.csr_matrix) -> None:
        if not sparse.isspmatrix_csr(A):
            raise ValueError("Matrix must be CSR for in-place row normalization.")
        row_sums = np.asarray(A.sum(axis=1)).ravel()
        scale = np.ones(row_sums.shape[0], dtype=np.float64)
        nonzero = row_sums != 0
        scale[nonzero] = 1.0 / row_sums[nonzero]
        cls._csr_row_scale_inplace(A, scale)

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


class _RotationForestOOBAdapter:
    def _set_rotation_forest_oob_attributes(self, X: Any, y: Optional[Any]) -> None:
        if y is None:
            return
        y_arr = np.asarray(y)

        surrogate_estimators = getattr(self, "_oob_surrogate_estimators_", None)
        oob_mask = getattr(self, "oob_mask_", None)
        if surrogate_estimators is None or oob_mask is None:
            raise RuntimeError(
                "RotationForest OOB attributes require the synchronized aeon "
                "training-estimate adapter to reconstruct surrogate trees."
            )
        if len(surrogate_estimators) != oob_mask.shape[1]:
            raise RuntimeError("RotationForest OOB surrogate inventory does not match its OOB mask.")

        with _suppress_base_validate_data_futurewarning():
            transformed = rotation_forest_transforms(self.estimator, X)
        if len(transformed) != len(surrogate_estimators):
            raise RuntimeError("RotationForest transformed-tree inventory changed after fitting.")

        counts = oob_mask.sum(axis=1).astype(np.int32)
        valid = counts > 0
        divisors = counts.astype(np.float64)
        if not np.all(valid):
            warn(
                "Some inputs do not have OOB scores because no refitted OOB tree "
                "excluded them; inspect oob_valid_mask_ before using OOB outputs.",
                UserWarning,
            )
            divisors[~valid] = 1.0

        def publish(name: str, value: Any) -> None:
            setattr(self, name, value)
            setattr(self.estimator, name, value)

        publish("oob_counts_", counts)
        publish("oob_valid_mask_", valid)

        if hasattr(self.estimator, "classes_"):
            classes = np.asarray(self.estimator.classes_)
            class_to_col = {cls: index for index, cls in enumerate(classes)}
            results = np.zeros((len(y_arr), len(classes)), dtype=np.float64)
            for tree_index, surrogate in enumerate(surrogate_estimators):
                oob_rows = np.flatnonzero(oob_mask[:, tree_index])
                if oob_rows.size == 0:
                    continue
                probabilities = np.asarray(
                    surrogate.predict_proba(transformed[tree_index][oob_rows]),
                    dtype=np.float64,
                )
                for column, label in enumerate(np.asarray(surrogate.classes_)):
                    results[oob_rows, class_to_col[label]] += probabilities[:, column]
            oob_decision = results / divisors[:, None]
            publish("oob_decision_function_", oob_decision)
            publish("oob_score_", float(accuracy_score(y_arr, classes[np.argmax(oob_decision, axis=1)])))
            return

        results = np.zeros(len(y_arr), dtype=np.float64)
        for tree_index, surrogate in enumerate(surrogate_estimators):
            oob_rows = np.flatnonzero(oob_mask[:, tree_index])
            if oob_rows.size:
                results[oob_rows] += np.asarray(
                    surrogate.predict(transformed[tree_index][oob_rows]), dtype=np.float64,
                )
        oob_prediction = results / divisors
        publish("oob_prediction_", oob_prediction)
        publish("oob_score_", float(r2_score(y_arr, oob_prediction)))
        return
