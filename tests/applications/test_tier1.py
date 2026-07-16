import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from forestgap import (
    Proximity,
    ProximityKernelPredictor,
    conditional_distribution_weights,
    conditional_quantiles,
    instance_attributions,
    knn_predict,
    neighborhood_feature_scores,
    proximity_clustering,
    retrieve_neighbors,
    retrieval_metrics,
    select_active_batch,
    select_features,
    split_conformal_classification,
    split_conformal_regression,
)


def test_prediction_clustering_and_feature_selection():
    X, y = make_classification(n_samples=70, n_features=6, n_informative=4, random_state=30)
    proximity = Proximity(
        RandomForestClassifier(n_estimators=30, random_state=31), relation="original",
    ).fit(X, y)
    assert ProximityKernelPredictor(proximity).fit(y).predict().shape == (len(X),)
    assert knn_predict(proximity, y, n_neighbors=3).shape == (len(X),)
    assert proximity_clustering(proximity, 2, random_state=32).shape == (len(X),)
    assert neighborhood_feature_scores(proximity, X).shape == (X.shape[1],)
    assert select_features(proximity, X, k=2).shape == (2,)


def test_quantiles_conformal_attribution_retrieval_and_active_learning():
    X, y = make_regression(n_samples=80, n_features=6, random_state=33)
    estimator = RandomForestRegressor(n_estimators=35, random_state=34).fit(X[:60], y[:60])
    conditional = Proximity(estimator, relation="original").attach(X[:60], y[:60])
    weights = conditional_distribution_weights(conditional, X[60:])
    np.testing.assert_allclose(np.asarray(weights.sum(axis=1)).ravel(), 1.0)
    assert conditional_quantiles(conditional, y[:60], [0.1, 0.5, 0.9], X=X[60:]).shape == (20, 3)
    intervals = split_conformal_regression(y[50:60], estimator.predict(X[50:60]), estimator.predict(X[60:]))
    assert intervals.shape == (20, 2)

    directed = Proximity(estimator, relation="rfgap").attach(X[:60], y[:60])
    attribution = instance_attributions(directed, X[60:62], target_values=y[:60], top_k=4)
    assert attribution["indices"].shape == (2, 4)
    indices, scores = retrieve_neighbors(directed, X[60:63], k=5)
    assert indices.shape == scores.shape == (3, 5)
    assert set(retrieval_metrics(indices, [{int(indices[0, 0])}, set(), set()])) == {"recall_at_k", "map_at_k", "ndcg_at_k"}

    affinity = conditional.get_proximities().toarray()
    chosen = select_active_batch(np.linspace(0, 1, len(affinity)), affinity, 5)
    assert len(np.unique(chosen)) == 5

    probabilities = np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    sets = split_conformal_classification([0, 1], probabilities[:2], probabilities[2:], classes=[0, 1])
    assert len(sets) == 2


def test_tier1_helpers_reject_misaligned_or_empty_inputs():
    with pytest.raises(ValueError, match="must align"):
        split_conformal_regression(np.ones((2, 1)), np.ones(3), [0.0])
    with pytest.raises(ValueError, match="at least one"):
        split_conformal_regression([], [], [0.0])
    with pytest.raises(ValueError, match="same class columns"):
        split_conformal_classification([0, 1], np.ones((2, 2)), np.ones((1, 3)), classes=[0, 1])
    with pytest.raises(ValueError, match="one label"):
        split_conformal_classification([0, 1], np.ones((2, 2)), np.ones((1, 2)), classes=[0])
    with pytest.raises(ValueError, match="one set"):
        retrieval_metrics([[0, 1], [1, 2]], [{0}])
    with pytest.raises(ValueError, match="at least one"):
        select_active_batch([], np.empty((0, 0)), 1)
