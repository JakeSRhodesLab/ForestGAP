"""Shi--Horvath observed-versus-synthetic-null forest meta-estimator."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted


class SyntheticNullForest(ClassifierMixin, BaseEstimator):
    """Fit a classifier that separates observed rows from a synthetic null.

    ``generator='independent_marginals'`` independently permutes each feature,
    preserving univariate marginals while destroying multivariate dependence.
    The generated null, generator name, and resolved seed are persisted for
    provenance.  Proximities are indexed on observed rows only by
    :class:`forestgap.proximity.Proximity`.
    """

    _forestgap_synthetic_null = True

    def __init__(
        self,
        estimator: Any | None = None,
        *,
        generator: Literal["independent_marginals", "uniform"] = "independent_marginals",
        synthetic_ratio: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.generator = generator
        self.synthetic_ratio = synthetic_ratio
        self.random_state = random_state

    def _make_synthetic(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n_synth = max(1, int(round(X.shape[0] * self.synthetic_ratio)))
        if self.generator == "independent_marginals":
            indices = rng.integers(0, X.shape[0], size=(n_synth, X.shape[1]))
            return np.take_along_axis(X.T, indices.T, axis=1).T
        if self.generator == "uniform":
            lo = np.nanmin(X, axis=0)
            hi = np.nanmax(X, axis=0)
            return rng.uniform(lo, hi, size=(n_synth, X.shape[1]))
        raise ValueError("generator must be 'independent_marginals' or 'uniform'")

    def fit(self, X: Any, y: Any = None) -> "SyntheticNullForest":
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D feature matrix")
        base = self.estimator
        if base is None:
            base = RandomForestClassifier(
                n_estimators=200,
                bootstrap=True,
                oob_score=True,
                random_state=self.random_state,
            )
        self.estimator_ = clone(base)
        self.synthetic_X_ = self._make_synthetic(X_arr)
        self.observed_X_ = X_arr.copy()
        joined = np.vstack([X_arr, self.synthetic_X_])
        labels = np.r_[np.ones(X_arr.shape[0], dtype=int), np.zeros(self.synthetic_X_.shape[0], dtype=int)]
        self.estimator_.fit(joined, labels)
        self.generator_ = self.generator
        self.synthetic_seed_ = self.random_state
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = self.estimator_.n_features_in_
        self.estimators_ = self.estimator_.estimators_
        self.n_estimators = len(self.estimators_)
        self.bootstrap = bool(getattr(self.estimator_, "bootstrap", False))
        self.max_samples = getattr(self.estimator_, "max_samples", None)
        return self

    def apply(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.apply(X)

    def decision_path(self, X: Any) -> Any:
        check_is_fitted(self, "estimator_")
        return self.estimator_.decision_path(X)

    @property
    def estimators_samples_(self) -> Any:
        check_is_fitted(self, "estimator_")
        return self.estimator_.estimators_samples_

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    def score_samples(self, X: Any) -> np.ndarray:
        """Observed-class probability, where lower values are more null-like."""

        proba = self.predict_proba(X)
        observed_col = int(np.flatnonzero(self.classes_ == 1)[0])
        return proba[:, observed_col]
