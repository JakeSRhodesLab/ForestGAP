import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_regression
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
    assert capabilities_for(estimator).valid_relations == ("original",)
    with pytest.raises(UnsupportedRelationError):
        Proximity(estimator, relation="rfgap").attach(X, y)


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
