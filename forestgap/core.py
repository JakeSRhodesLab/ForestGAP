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

import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import pandas as pd


def patch_treeple_regressor_tags() -> bool:
    """Patch treeple regressor sklearn tags across sklearn versions.

    treeple 0.10.3 tries to set ``tags.regressor_tags.multi_label = True`` in
    its bundled sklearn forest regressor implementation. Some scikit-learn
    versions do not define ``multi_label`` on ``RegressorTags``, which raises
    ``AttributeError`` during fit.

    Returns
    -------
    bool
        True when the patch is installed (or already present), False when
        treeple is unavailable.
    """

    try:
        from treeple._lib.sklearn.ensemble._forest import ForestRegressor
    except Exception:
        return False

    if getattr(ForestRegressor, "_forestgap_tags_patch", False):
        return True

    def _patched_sklearn_tags(self):
        tags = super(ForestRegressor, self).__sklearn_tags__()
        regressor_tags = getattr(tags, "regressor_tags", None)
        if regressor_tags is not None:
            try:
                regressor_tags.multi_label = True
            except AttributeError:
                pass
        return tags

    ForestRegressor.__sklearn_tags__ = _patched_sklearn_tags
    ForestRegressor._forestgap_tags_patch = True
    return True


def patch_treeple_sample_weight_signature() -> bool:
    """Patch treeple's vendored sklearn tree helper for sklearn>=1.7.

    treeple 0.10.3 still calls ``_check_sample_weight(sample_weight, X, DOUBLE)``
    positionally, while sklearn 1.7 makes ``dtype`` keyword-only. Patch the
    treeple-local reference so existing call sites keep working.

    Returns
    -------
    bool
        True when the patch is installed (or already present), False when
        treeple is unavailable.
    """

    try:
        from sklearn.utils.validation import _check_sample_weight as sklearn_check_sample_weight
        import treeple._lib.sklearn.tree._classes as treeple_tree_classes
    except Exception:
        return False

    if getattr(treeple_tree_classes, "_forestgap_sample_weight_patch", False):
        return True

    def _patched_check_sample_weight(
        sample_weight: Any,
        X: Any,
        dtype: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if args:
            if dtype is None:
                dtype = args[0]
            else:
                raise TypeError("dtype provided both positionally and by keyword.")
            if len(args) > 1:
                raise TypeError(
                    "_check_sample_weight accepts at most one positional argument after X."
                )
        return sklearn_check_sample_weight(
            sample_weight,
            X,
            dtype=dtype,
            **kwargs,
        )

    treeple_tree_classes._check_sample_weight = _patched_check_sample_weight
    treeple_tree_classes._forestgap_sample_weight_patch = True
    return True


patch_treeple_regressor_tags()
patch_treeple_sample_weight_signature()


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
        self._X_train: Optional[np.ndarray] = None
        self._X_train_index_: Optional[pd.Index] = None
        self._X_train_columns_: Optional[pd.Index] = None
        self._X_train_was_dataframe_: bool = False

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
        self._store_training_features(X)
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

    def embed(self, method: str = "phate", **kwargs: Any) -> np.ndarray:
        """
        Create a manifold embedding from this fitted forest's proximities.

        Parameters
        ----------
        method : {'phate', 'tsne', 'isomap', 'umap'}
            Manifold method to fit from the training proximity matrix.
        **kwargs : dict
            Method-specific options forwarded to ``forestgap.manifold``.

        Returns
        -------
        embedding : ndarray of shape (n_train_samples, n_components)
            Low-dimensional coordinates for the training samples.
        """
        self._check_proximity_ready()
        from forestgap.manifold import embed_from_forest

        return embed_from_forest(self, method=method, **kwargs)

    def proximity_embedding(self, method: str = "phate", **kwargs: Any) -> np.ndarray:
        """Alias for :meth:`embed`."""
        return self.embed(method=method, **kwargs)

    def phate(self, **kwargs: Any) -> np.ndarray:
        """Fit a PHATE embedding from this fitted forest's proximities."""
        return self.embed(method="phate", **kwargs)

    def embed_transform(self, X_new: Any, method: str = "phate", **kwargs: Any) -> np.ndarray:
        """
        Project new observations through a stored manifold model when supported.

        ``method='phate'`` is supported after calling ``embed(method='phate')``.
        ``method='isomap'`` is supported when sklearn accepts precomputed
        new-to-training distances for the fitted Isomap model. t-SNE does not
        provide an out-of-sample transform.
        """
        self._check_proximity_ready()
        from forestgap.manifold import transform_from_forest

        return transform_from_forest(self, X_new, method=method, **kwargs)

    def phate_transform(self, X_new: Any, **kwargs: Any) -> np.ndarray:
        """Project new observations through a stored PHATE proximity embedding."""
        return self.embed_transform(X_new, method="phate", **kwargs)

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

    def impute(
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

    def outlier_scores(
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

    def get_prototypes(
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

    def local_feature_importance(
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

    def local_variable_importance(self, *args: Any, **kwargs: Any) -> Any:
        """Alias for :meth:`local_feature_importance`."""
        return self.local_feature_importance(*args, **kwargs)

    def plot_mds(
        self,
        X_test: Optional[Any] = None,
        y: Optional[Any] = None,
        y_test: Optional[Any] = None,
    ) -> Any:
        """
        Visualize proximity matrix using metric MDS and scatterplot.
        If X_test is provided, uses test proximities; else training.
        Colors points by y (categorical or continuous).
        Returns the matplotlib figure object.
        """

        # Obtain proximities and keep sparse when possible
        if X_test is not None:
            pr = self.prox_extend(X_test)
            labels = y_test
        else:
            pr = self.get_proximities()
            labels = y

        # Ensure sparse representation until we must densify for MDS
        if sparse.issparse(pr):
            pr_sp = pr.tocsr()
        else:
            pr_sp = sparse.csr_matrix(np.asarray(pr))

        # For rfgap, recompute proximities with non-zero diagonal & symm_mode
        if self.prox_method == "rfgap":
            orig_nzd = self.non_zero_diagonal
            orig_sym = self.symm_mode
            self.non_zero_diagonal = True
            self.symm_mode = self.symm_mode or "arithmetic"
            try:
                pr_tmp = self.get_proximities() if X_test is None else self.prox_extend(X_test)
                pr_sp = pr_tmp.tocsr() if sparse.issparse(pr_tmp) else sparse.csr_matrix(np.asarray(pr_tmp))
            finally:
                self.non_zero_diagonal = orig_nzd
                self.symm_mode = orig_sym

        # Compute sparse distance matrix: dist = sqrt(max(1 - prox, 0))
        dist_sp = pr_sp.copy().astype(np.float64)
        # data := 1 - data
        np.subtract(1.0, dist_sp.data, out=dist_sp.data)
        # clip negatives to zero then sqrt
        dist_sp.data[dist_sp.data < 0] = 0.0
        np.sqrt(dist_sp.data, out=dist_sp.data)

        # MDS requires dense dissimilarity matrix — convert only now
        dist = dist_sp.toarray()

        # Fit MDS
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        coords = mds.fit_transform(dist)

        # Infer categorical vs. continuous for coloring
        if labels is not None:
            if pd is not None and isinstance(labels, (pd.Series, pd.DataFrame)):
                arr = labels.values if hasattr(labels, "values") else np.asarray(labels)
            else:
                arr = np.asarray(labels)
            if np.issubdtype(arr.dtype, np.integer) or (np.unique(arr).size <= 10):
                unique = np.unique(arr)
                cmap = plt.get_cmap("tab10") if unique.size <= 10 else plt.get_cmap("tab20")
                colors = [cmap(i % cmap.N) for i in range(unique.size)]
                color_dict = {v: colors[i] for i, v in enumerate(unique)}
                color_vals = [color_dict[v] for v in arr]
                scatter_kwargs = {"c": color_vals, "label": None}
            else:
                scatter_kwargs = {"c": arr, "cmap": "viridis", "label": None}
        else:
            scatter_kwargs = {"c": "gray", "label": None}

        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(coords[:, 0], coords[:, 1], s=60, **scatter_kwargs)
        ax.set_xlabel("MDS 1")
        ax.set_ylabel("MDS 2")
        ax.set_title("MDS of Proximities")
        if labels is not None:
            if np.issubdtype(arr.dtype, np.integer) or (np.unique(arr).size <= 10):
                unique = np.unique(arr)
                from matplotlib.lines import Line2D

                handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[v], markersize=8, label=str(v)) for v in unique]
                ax.legend(handles=handles, title="Class")
            else:
                fig.colorbar(sc, ax=ax, label="y")
        plt.tight_layout()
        plt.show()
        return fig

        
    def _check_proximity_ready(self) -> None:
        check_is_fitted(cast(Any, self))
        if self.W_mat is None or self.leaf_matrix_ is None or self._leaf_encoding is None:
            raise RuntimeError("Proximity structures are not initialized. Call fit(X, y) first.")

    def _store_training_features(self, X: Any) -> None:
        if sparse.issparse(X):
            self._X_train = None
            self._X_train_index_ = None
            self._X_train_columns_ = None
            self._X_train_was_dataframe_ = False
            return

        if isinstance(X, pd.DataFrame):
            self._X_train = X.to_numpy(copy=True)
            self._X_train_index_ = X.index.copy()
            self._X_train_columns_ = X.columns.copy()
            self._X_train_was_dataframe_ = True
            return

        X_arr = np.array(X, copy=True)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        self._X_train = X_arr
        self._X_train_index_ = None
        self._X_train_columns_ = None
        self._X_train_was_dataframe_ = False

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
            if not SparseProximityMixin._is_numeric_local_feature(X_train[:, feature_idx])
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
                    key=SparseProximityMixin._prototype_category_sort_key,
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
        return "quantile" if SparseProximityMixin._is_numeric_local_feature(combined) else "permute"

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
            elif SparseProximityMixin._scalar_values_equal(value, lo):
                flipped[idx] = hi
            elif SparseProximityMixin._scalar_values_equal(value, hi):
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
            if not SparseProximityMixin._scalar_values_equal(sampled[idx], value):
                continue

            alternatives = [
                candidate
                for candidate in unique_nonmissing
                if not SparseProximityMixin._scalar_values_equal(candidate, value)
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

                max_samples_param = base_sig.parameters.get("max_samples")
                max_samples_val = kwargs.get("max_samples", None)
                if (
                    max_samples_val is None
                    and max_samples_param is not None
                    and max_samples_param.default is not inspect._empty
                ):
                    max_samples_val = max_samples_param.default

                if bootstrap_val is False and max_samples_val is not None:
                    kwargs["max_samples"] = None

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

            # treeple ForestCluster (UnsupervisedRandomForest) has a buggy
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
