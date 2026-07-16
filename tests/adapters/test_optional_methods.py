import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from forestgap import Proximity, conditional_distribution_weights
from forestgap.adapters import (
    capabilities_for,
    censoring_safe_survival_metrics,
    native_quantiles,
    native_survival_outputs,
)
from forestgap.adapters.registry import method_capabilities


def test_optional_availability_is_reported_without_import_failure(monkeypatch):
    monkeypatch.setattr("forestgap.adapters.registry.find_spec", lambda name: None)
    assert not method_capabilities("quantile_forest").available
    assert not method_capabilities("random_survival_forest").available


def test_quantile_forest_conditional_weights_match_leaf_mass_definition():
    quantile_forest = pytest.importorskip("quantile_forest")
    rng = np.random.default_rng(11)
    X = rng.normal(size=(55, 4))
    y = X[:, 0] + rng.normal(scale=0.4, size=len(X))
    estimator = quantile_forest.RandomForestQuantileRegressor(
        n_estimators=9, bootstrap=False, random_state=12,
    ).fit(X, y)
    proximity = Proximity(estimator, relation="conditional").attach(X, y)
    query = X[:5] + 0.01
    actual = conditional_distribution_weights(proximity, query).toarray()
    train_leaves = estimator.apply(X)
    query_leaves = estimator.apply(query)
    expected = np.zeros_like(actual)
    for row in range(len(query)):
        for tree in range(estimator.n_estimators):
            members = train_leaves[:, tree] == query_leaves[row, tree]
            expected[row, members] += 1.0 / (estimator.n_estimators * members.sum())
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0)
    assert native_quantiles(estimator, query, [0.1, 0.5, 0.9]).shape == (5, 3)


def test_random_survival_forest_outputs_relation_and_ipcw_metrics():
    pytest.importorskip("sksurv")
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv

    rng = np.random.default_rng(13)
    X = rng.normal(size=(130, 5))
    event_time = rng.exponential(np.exp(-0.7 * X[:, 0]) * 8.0)
    censor_time = rng.exponential(11.0, size=len(X))
    event = event_time <= censor_time
    observed = np.minimum(event_time, censor_time) + 0.05
    y = Surv.from_arrays(event, observed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=14,
    )
    estimator = RandomSurvivalForest(
        n_estimators=20, min_samples_leaf=3, random_state=15, n_jobs=1,
    ).fit(X_train, y_train)
    proximity = Proximity(estimator, relation="survival_leaf").attach(X_train, y_train)
    assert proximity.prox_extend(X_test[:4]).shape == (4, len(X_train))
    outputs = native_survival_outputs(estimator, X_test[:4])
    assert {"risk", "survival", "cumulative_hazard"} <= set(outputs)
    metrics = censoring_safe_survival_metrics(estimator, X_test, y_train, y_test)
    assert 0.0 <= metrics["ipcw_concordance"] <= 1.0
    assert np.isfinite(metrics["integrated_brier_score"])


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_aeon_rotation_forest_adapter_preserves_oob_test_and_path_contracts(task):
    pytest.importorskip("aeon")

    rng = np.random.default_rng(16)
    X = rng.normal(size=(70, 5))
    if task == "classification":
        from aeon.classification.sklearn import RotationForestClassifier

        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        estimator = RotationForestClassifier(
            n_estimators=11, random_state=17, n_jobs=1,
        ).fit(X, y)
        expected_test = estimator.predict_proba(X[:6])
        oob_name = "oob_decision_function_"
    else:
        from aeon.regression.sklearn import RotationForestRegressor

        _, y = make_regression(n_samples=len(X), n_features=X.shape[1], random_state=18)
        estimator = RotationForestRegressor(
            n_estimators=11, random_state=17, n_jobs=1,
        ).fit(X, y)
        expected_test = estimator.predict(X[:6])
        oob_name = "oob_prediction_"

    proximity = Proximity(estimator, relation="rfgap").attach(X, y)
    assert proximity.capabilities_.family == "rotation_forest"
    assert {"original", "kerf", "oob", "rfgap", "path", "englund", "random_height"} <= set(
        proximity.capabilities_.valid_relations
    )
    assert proximity.get_proximities().shape == (len(X), len(X))
    np.testing.assert_allclose(
        proximity.prox_predict(X[:6]), expected_test, atol=3e-5,
    )
    valid = proximity.oob_valid_mask_
    assert valid.any()
    np.testing.assert_allclose(
        proximity.prox_predict()[valid], getattr(estimator, oob_name)[valid], atol=3e-5,
    )
    np.testing.assert_allclose(getattr(proximity, oob_name), getattr(estimator, oob_name))
    assert proximity.relation_metadata_["oob_valid_rows"] == int(valid.sum())

    repeated = Proximity(estimator, relation="rfgap").attach(X, y)
    np.testing.assert_array_equal(repeated.oob_mask_, proximity.oob_mask_)
    np.testing.assert_allclose(repeated.prox_predict(), proximity.prox_predict(), atol=3e-5)

    for relation in ("path", "englund", "random_height"):
        path_proximity = Proximity(
            estimator, relation=relation, random_state=19,
        ).attach(X, y)
        matrix = path_proximity.get_proximities().toarray()
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-7)
        assert np.linalg.eigvalsh(matrix).min() >= -1e-6
        assert path_proximity.prox_extend(X[:4]).shape == (4, len(X))


def test_aeon_rotation_forest_rejects_missing_training_estimate_contract(monkeypatch):
    pytest.importorskip("aeon")
    from aeon.classification.sklearn import RotationForestClassifier

    rng = np.random.default_rng(20)
    X = rng.normal(size=(45, 5))
    y = (X[:, 0] > 0).astype(int)
    estimator = RotationForestClassifier(
        n_estimators=5, random_state=21, n_jobs=1,
    ).fit(X, y)
    monkeypatch.setattr(type(estimator), "_train_probas_for_estimator", None)
    assert "rfgap" not in capabilities_for(estimator).valid_relations
    with pytest.raises(ValueError, match="synchronized training-estimate"):
        Proximity(estimator, relation="rfgap").attach(X, y)


def test_aeon_rotation_forest_marks_rows_without_an_oob_tree():
    pytest.importorskip("aeon")
    from aeon.classification.sklearn import RotationForestClassifier

    rng = np.random.default_rng(31)
    X = rng.normal(size=(45, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    estimator = RotationForestClassifier(
        n_estimators=1, random_state=32, n_jobs=1,
    ).fit(X, y)
    with pytest.warns(UserWarning, match="oob_valid_mask_"):
        proximity = Proximity(estimator, relation="rfgap").attach(X, y)

    invalid = ~proximity.oob_valid_mask_
    assert invalid.any()
    assert proximity.relation_metadata_["oob_invalid_rows"] == int(invalid.sum())
    np.testing.assert_allclose(estimator.oob_decision_function_[invalid], 0.0)


def test_treeple_oblique_honest_and_extended_isolation_contracts():
    treeple = pytest.importorskip("treeple")
    from forestgap.adapters import capabilities_for

    rng = np.random.default_rng(18)
    X = rng.normal(size=(50, 5))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)

    oblique = treeple.ObliqueRandomForestClassifier(
        n_estimators=5, random_state=19, n_jobs=1,
    ).fit(X, y)
    assert {"original", "oob", "rfgap", "path"} <= set(
        capabilities_for(oblique).valid_relations
    )
    oblique_proximity = Proximity(oblique, relation="rfgap").attach(X, y)
    assert oblique_proximity.get_proximities().shape == (50, 50)
    np.testing.assert_allclose(
        oblique_proximity.prox_predict(X[:6]), oblique.predict_proba(X[:6]), atol=1e-7,
    )

    honest = treeple.HonestForestClassifier(
        n_estimators=5, max_depth=4, random_state=20, n_jobs=1,
    ).fit(X, y)
    assert {"honest_gap", "kerf", "original"} <= set(capabilities_for(honest).valid_relations)
    honest_proximity = Proximity(honest, relation="honest_gap").attach(X, y)
    assert honest_proximity.get_proximities().shape == (50, 50)
    np.testing.assert_allclose(
        honest_proximity.prox_predict(X[:6]), honest.predict_proba(X[:6]), atol=1e-7,
    )

    extended = treeple.ExtendedIsolationForest(
        n_estimators=5, random_state=21, n_jobs=1,
    ).fit(X)
    assert capabilities_for(extended).valid_relations == ("collision",)
    assert Proximity(extended, relation="collision").attach(X).get_proximities().shape == (50, 50)
