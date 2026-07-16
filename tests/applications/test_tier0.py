import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from forestgap import (
    Proximity,
    get_prototypes,
    impute,
    local_feature_importance,
    native_anomaly_scores,
    outlier_scores,
)


def _proximity():
    X, y = make_classification(
        n_samples=70, n_features=6, n_informative=4, random_state=20,
    )
    proximity = Proximity(
        RandomForestClassifier(n_estimators=20, random_state=21), relation="original",
    ).fit(X, y)
    return X, y, proximity


def test_imputation_anomaly_and_prototypes_are_standalone():
    X, y, proximity = _proximity()
    missing = X.copy()
    missing[0, 0] = np.nan
    filled = impute(proximity, missing)
    assert np.isfinite(filled).all()
    scores = outlier_scores(proximity, y, method="ene")
    assert scores.shape == (len(X),)
    prototypes = get_prototypes(proximity, k=5, n_prototypes=1)
    assert set(prototypes) == set(np.unique(y))


def test_local_importance_has_query_feature_shape():
    X, _, proximity = _proximity()
    importance = local_feature_importance(
        proximity, X[:2], perturbation="permute", n_repeats=2, random_state=22,
    )
    assert importance.shape == (2, X.shape[1])


def test_native_isolation_score_does_not_require_a_relation():
    X, _, _ = _proximity()
    estimator = IsolationForest(n_estimators=10, random_state=23).fit(X)
    scores = native_anomaly_scores(estimator, X)
    np.testing.assert_allclose(scores, -estimator.score_samples(X))
