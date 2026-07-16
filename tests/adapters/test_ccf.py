import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.utils.estimator_checks import check_estimator

from forestgap import (
    CanonicalCorrelationForestClassifier,
    CanonicalCorrelationForestRegressor,
    Proximity,
)
from forestgap.adapters import (
    UnsupportedRelationError,
    capabilities_for,
    load_ccf_oracle_fixture,
    validate_ccf_oracle,
)


def test_classifier_and_regressor_pass_packaged_matlab_oracle_fixture():
    fixture = load_ccf_oracle_fixture()
    X_train = np.asarray(fixture["X_train"])
    X_test = np.asarray(fixture["X_test"])
    parameters = fixture["estimator_parameters"]
    for estimator_class, task in (
        (CanonicalCorrelationForestClassifier, "classification"),
        (CanonicalCorrelationForestRegressor, "regression"),
    ):
        section = fixture[task]
        result = validate_ccf_oracle(
            estimator_class(**parameters),
            X_train,
            np.asarray(section["y_train"]),
            X_test,
            np.asarray(section["expected_predictions"]),
        )
        assert result.passed
        assert result.max_absolute_error == 0.0


def test_ccf_sklearn_lifecycle_apply_and_original_relation():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(90, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    prototype = CanonicalCorrelationForestClassifier(
        n_estimators=7, max_depth=7, random_state=7,
    )
    estimator = clone(prototype).fit(X, y)
    assert estimator.score(X, y) > 0.9
    assert estimator.apply(X).shape == (90, 7)
    assert estimator.predict_proba(X[:4]).shape == (4, 2)
    proximity = Proximity(estimator, relation="original").attach(X, y)
    assert proximity.get_proximities().shape == (90, 90)
    assert set(capabilities_for(estimator).valid_relations) == {
        "englund", "kerf", "original", "path", "random_height",
    }
    with pytest.raises(UnsupportedRelationError):
        Proximity(estimator, relation="rfgap").attach(X, y)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_ccf_bag_oob_and_rfgap_preserve_native_prediction_outputs(task):
    if task == "classification":
        X, y = make_classification(n_samples=100, n_features=6, random_state=21)
        estimator = CanonicalCorrelationForestClassifier(
            n_estimators=17, max_depth=7, bootstrap=True, oob_score=True,
            random_state=22,
        ).fit(X, y)
        expected_test = estimator.predict_proba(X[:9])
        expected_oob = estimator.oob_decision_function_
    else:
        X, y = make_regression(n_samples=100, n_features=6, random_state=23)
        estimator = CanonicalCorrelationForestRegressor(
            n_estimators=17, max_depth=7, bootstrap=True, oob_score=True,
            random_state=24,
        ).fit(X, y)
        expected_test = estimator.predict(X[:9])
        expected_oob = estimator.oob_prediction_

    capabilities = capabilities_for(estimator)
    assert {"oob", "rfgap"} <= set(capabilities.valid_relations)
    assert estimator.oob_valid_mask_.any()

    proximity = Proximity(estimator, relation="rfgap").attach(X, y)
    np.testing.assert_allclose(proximity.prox_predict(X[:9]), expected_test, atol=3e-5)
    np.testing.assert_allclose(
        proximity.prox_predict()[estimator.oob_valid_mask_],
        expected_oob[estimator.oob_valid_mask_],
        atol=3e-5,
    )


@pytest.mark.parametrize("relation", ["path", "englund", "random_height"])
def test_ccf_path_relations_are_sparse_dense_psd_and_extend(relation):
    X, y = make_classification(n_samples=75, n_features=5, random_state=25)
    estimator = CanonicalCorrelationForestClassifier(
        n_estimators=9, max_depth=6, random_state=26,
    ).fit(X, y)
    sparse_proximity = Proximity(
        estimator, relation=relation, random_state=27,
    ).attach(X, y)
    dense_proximity = Proximity(
        estimator, relation=relation, matrix_type="dense", random_state=27,
    ).attach(X, y)
    sparse_relation = sparse_proximity.get_proximities()
    assert sparse.isspmatrix_csr(sparse_relation)
    dense_relation = sparse_relation.toarray()
    np.testing.assert_allclose(dense_relation, dense_proximity.get_proximities())
    np.testing.assert_allclose(dense_relation, dense_relation.T, atol=1e-8)
    assert np.linalg.eigvalsh(dense_relation).min() >= -1e-6
    assert sparse_proximity.prox_extend(X[:6]).shape == (6, len(X))


def test_ccf_rejects_oob_score_without_ccf_bag():
    X, y = make_classification(n_samples=30, n_features=4, random_state=28)
    with pytest.raises(ValueError, match="bootstrap=True"):
        CanonicalCorrelationForestClassifier(oob_score=True).fit(X, y)


def test_ccf_regressor_supports_single_and_multi_output_predictions():
    X, y = make_regression(n_samples=70, n_features=5, noise=0.1, random_state=8)
    single = CanonicalCorrelationForestRegressor(
        n_estimators=5, max_depth=8, random_state=9,
    ).fit(X, y)
    assert single.predict(X[:3]).shape == (3,)
    multi = CanonicalCorrelationForestRegressor(
        n_estimators=3, max_depth=7, random_state=10,
    ).fit(X, np.column_stack([y, -y]))
    assert multi.predict(X[:3]).shape == (3, 2)


@pytest.mark.parametrize(
    "estimator",
    [
        CanonicalCorrelationForestClassifier(n_estimators=2, max_depth=3),
        CanonicalCorrelationForestRegressor(n_estimators=2, max_depth=3),
    ],
)
def test_ccf_estimators_satisfy_sklearn_estimator_contract(estimator):
    check_estimator(estimator)
