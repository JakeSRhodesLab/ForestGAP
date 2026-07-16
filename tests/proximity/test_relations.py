import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier, RandomForestRegressor

from forestgap import Proximity, UnsupportedRelationError


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=100, n_features=8, n_informative=5, random_state=10,
    )
    return X[:80], y[:80], X[80:], y[80:]


@pytest.mark.parametrize("relation", ["original", "kerf", "path", "englund", "random_height"])
def test_symmetric_kernel_relations_are_psd_and_extend(classification_data, relation):
    X, y, X_test, _ = classification_data
    proximity = Proximity(
        RandomForestClassifier(n_estimators=25, random_state=11),
        relation=relation,
        random_state=12,
    ).fit(X, y)
    matrix = proximity.get_proximities()
    dense = matrix.toarray()
    np.testing.assert_allclose(dense, dense.T, atol=1e-7)
    assert np.linalg.eigvalsh(dense).min() >= -1e-6
    assert proximity.prox_extend(X_test).shape == (len(X_test), len(X))
    assert proximity.diagnostics(matrix)["psd"] is True


def test_englund_uses_terminal_node_branch_distance(classification_data):
    X, y, X_test, _ = classification_data
    estimator = RandomForestClassifier(
        n_estimators=1, bootstrap=False, max_depth=4, random_state=23,
    ).fit(X, y)
    weight = 0.7
    proximity = Proximity(
        estimator, relation="englund", matrix_type="dense", englund_weight=weight,
    ).attach(X, y)
    actual = proximity.get_proximities()

    indicator, node_ptr = estimator.decision_path(X)
    paths = []
    block = indicator[:, node_ptr[0]:node_ptr[1]].tocsr()
    for row in range(block.shape[0]):
        paths.append(block.indices[block.indptr[row]:block.indptr[row + 1]])

    expected = np.empty((len(X), len(X)), dtype=float)
    for i, left in enumerate(paths):
        for j, right in enumerate(paths):
            common = min(len(left), len(right))
            mismatch = np.flatnonzero(left[:common] != right[:common])
            common = int(mismatch[0]) if mismatch.size else common
            branches = len(left) + len(right) - 2 * common
            expected[i, j] = np.exp(-weight * branches)

    np.testing.assert_allclose(actual, expected)
    assert np.all(np.diag(actual) == 1.0)
    assert proximity.prox_extend(X_test).shape == (len(X_test), len(X))
    assert proximity.relation_metadata_["englund_weight"] == weight


def test_random_height_is_seeded_and_cached(classification_data):
    X, y, _, _ = classification_data
    estimator = RandomForestClassifier(n_estimators=20, random_state=13).fit(X, y)
    left = Proximity(estimator, relation="random_height", random_state=9).attach(X, y)
    right = Proximity(estimator, relation="random_height", random_state=9).attach(X, y)
    np.testing.assert_allclose(left.get_proximities().toarray(), right.get_proximities().toarray())
    assert left.relation_metadata_["tree_cut_heights"] == right.relation_metadata_["tree_cut_heights"]


def test_sparse_dense_parity(classification_data):
    X, y, _, _ = classification_data
    estimator = RandomForestClassifier(n_estimators=18, random_state=14).fit(X, y)
    sparse_relation = Proximity(estimator, relation="original").attach(X, y).get_proximities()
    dense_relation = Proximity(estimator, relation="original", matrix_type="dense").attach(X, y).get_proximities()
    assert sparse.isspmatrix_csr(sparse_relation)
    np.testing.assert_allclose(sparse_relation.toarray(), dense_relation)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_rfgap_reconstructs_test_predictions(task):
    if task == "classification":
        X, y = make_classification(n_samples=130, n_features=7, random_state=16)
        estimator = RandomForestClassifier(n_estimators=80, random_state=17)
        proximity = Proximity(estimator, relation="rfgap").fit(X[:100], y[:100])
        np.testing.assert_allclose(
            proximity.prox_predict(X[100:]), estimator.predict_proba(X[100:]), atol=2e-6,
        )
    else:
        X, y = make_regression(n_samples=130, n_features=7, random_state=18)
        estimator = RandomForestRegressor(n_estimators=80, random_state=19)
        proximity = Proximity(estimator, relation="rfgap").fit(X[:100], y[:100])
        np.testing.assert_allclose(
            proximity.prox_predict(X[100:]), estimator.predict(X[100:]), atol=2e-4,
        )


def test_invalid_bootstrap_and_isolation_requests_are_rejected(classification_data):
    X, y, _, _ = classification_data
    with pytest.raises(UnsupportedRelationError):
        Proximity(ExtraTreesClassifier(bootstrap=False), relation="rfgap").fit(X, y)
    with pytest.raises(UnsupportedRelationError):
        Proximity(IsolationForest(random_state=0), relation="rfgap").fit(X)
    collision = Proximity(IsolationForest(n_estimators=10, random_state=0), relation="collision").fit(X)
    assert collision.get_proximities().shape == (len(X), len(X))
