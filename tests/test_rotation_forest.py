from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.estimator_checks import check_estimator

from forestgap import (
    Proximity,
    RotationForestClassifier,
    RotationForestRegressor,
)
from forestgap.adapters import capabilities_for


@pytest.mark.parametrize(
    "estimator",
    [
        RotationForestClassifier(
            n_estimators=2, min_group=1, max_group=2, random_state=1
        ),
        RotationForestRegressor(
            n_estimators=2, min_group=1, max_group=2, random_state=1
        ),
    ],
)
def test_rotation_forests_pass_sklearn_estimator_checks(estimator: object) -> None:
    check_estimator(estimator)


def test_rotation_classifier_has_sklearn_lifecycle_and_tree_interfaces() -> None:
    X, y = make_classification(
        n_samples=64,
        n_features=7,
        n_informative=5,
        random_state=3,
    )
    estimator = RotationForestClassifier(
        n_estimators=7,
        min_group=2,
        max_group=4,
        random_state=4,
        n_jobs=1,
    )
    copied = clone(estimator)
    assert copied.get_params() == estimator.get_params()
    assert estimator.fit(X, y) is estimator

    probabilities = estimator.predict_proba(X[:6])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert estimator.predict(X[:6]).shape == (6,)
    assert estimator.apply(X[:6]).shape == (6, estimator.n_estimators)

    indicator, offsets = estimator.decision_path(X[:6])
    assert sparse.isspmatrix_csr(indicator)
    assert offsets.shape == (estimator.n_estimators + 1,)
    assert offsets[-1] == indicator.shape[1]
    assert len(estimator.rotation_matrices_) == estimator.n_estimators
    assert estimator.feature_importances_.shape == (X.shape[1],)
    np.testing.assert_allclose(estimator.feature_importances_.sum(), 1.0)

    capabilities = capabilities_for(estimator)
    assert capabilities.family == "rotation_forest"
    assert {
        "original",
        "oob",
        "rfgap",
        "kerf",
        "path",
        "englund",
        "random_height",
    } <= set(capabilities.valid_relations)


def test_rotation_regressor_supports_multioutput_and_is_deterministic() -> None:
    X, first = make_regression(
        n_samples=70,
        n_features=8,
        n_informative=6,
        noise=0.1,
        random_state=5,
    )
    targets = np.column_stack((first, 0.5 * first + X[:, 0]))
    parameters = {
        "n_estimators": 9,
        "min_group": 2,
        "max_group": 4,
        "base_estimator": DecisionTreeRegressor(max_depth=5),
        "random_state": 6,
        "n_jobs": 1,
    }
    first_fit = RotationForestRegressor(**parameters).fit(X, targets)
    second_fit = RotationForestRegressor(**parameters).fit(X, targets)

    assert first_fit.n_outputs_ == 2
    assert first_fit.predict(X[:7]).shape == (7, 2)
    np.testing.assert_allclose(first_fit.predict(X[:7]), second_fit.predict(X[:7]))
    np.testing.assert_array_equal(first_fit.apply(X[:7]), second_fit.apply(X[:7]))


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_native_rotation_rfgap_preserves_external_predictions(task: str) -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(75, 6))
    query = rng.normal(size=(8, 6))
    if task == "classification":
        y = np.where(X[:, 0] + X[:, 1] > 0, "positive", "negative")
        estimator = RotationForestClassifier(
            n_estimators=11, random_state=8, n_jobs=1
        ).fit(X, y)
        expected = estimator.predict_proba(query)
        oob_name = "oob_decision_function_"
    else:
        y = np.column_stack((X[:, 0] - X[:, 1], X[:, 2] + 0.25 * X[:, 3]))
        estimator = RotationForestRegressor(
            n_estimators=11, random_state=8, n_jobs=1
        ).fit(X, y)
        expected = estimator.predict(query)
        oob_name = "oob_prediction_"

    proximity = Proximity(estimator, relation="rfgap").attach(X, y)
    np.testing.assert_allclose(proximity.prox_predict(query), expected, atol=2e-5)
    valid = proximity.oob_valid_mask_
    assert valid.any()
    np.testing.assert_allclose(
        proximity.prox_predict()[valid],
        getattr(estimator, oob_name)[valid],
        atol=2e-5,
    )


def test_rotation_fit_predict_oob_and_constant_column_handling() -> None:
    rng = np.random.default_rng(9)
    X = np.column_stack((rng.normal(size=(45, 4)), np.ones(45)))
    y = X[:, 0] - X[:, 1]
    estimator = RotationForestRegressor(n_estimators=9, random_state=10, n_jobs=1)
    predictions = estimator.fit_predict(X, y)

    assert predictions.shape == y.shape
    np.testing.assert_array_equal(predictions, estimator.oob_prediction_)
    assert estimator.oob_valid_mask_.any()
    assert estimator.feature_importances_[-1] == 0.0
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"n_estimators": 0}, "positive integer"),
        ({"min_group": 4, "max_group": 3}, "less than or equal"),
        ({"remove_proportion": 0}, "lie in"),
        ({"n_jobs": 0}, "may not be zero"),
    ],
)
def test_rotation_parameter_validation_is_explicit(
    parameters: dict[str, object], message: str
) -> None:
    X = np.arange(24, dtype=float).reshape(8, 3)
    y = np.arange(8, dtype=float)
    with pytest.raises((TypeError, ValueError), match=message):
        RotationForestRegressor(**parameters).fit(X, y)
