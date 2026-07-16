import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from forestgap import Proximity


@pytest.fixture
def data():
    return make_classification(
        n_samples=80, n_features=7, n_informative=5, random_state=4,
    )


def test_fit_composes_without_changing_estimator_type(data):
    X, y = data
    estimator = RandomForestClassifier(n_estimators=20, random_state=5)
    proximity = Proximity(estimator, relation="original").fit(X, y)
    assert proximity.estimator is estimator
    assert type(estimator) is RandomForestClassifier
    assert proximity.get_proximities().shape == (len(X), len(X))


def test_attach_indexes_an_already_fitted_estimator(data):
    X, y = data
    estimator = RandomForestClassifier(n_estimators=20, random_state=6).fit(X, y)
    proximity = Proximity(estimator, relation="original").attach(X, y)
    np.testing.assert_array_equal(proximity.predict(X), estimator.predict(X))


def test_estimator_class_is_rejected():
    with pytest.raises(TypeError, match="instance"):
        Proximity(RandomForestClassifier)
