from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from forestgap import Proximity, UnsupportedRelationError
from forestgap.adapters import capabilities_for


def _data(task: str) -> tuple[np.ndarray, np.ndarray]:
    if task == "binary":
        return make_classification(
            n_samples=54, n_features=6, n_informative=4, random_state=101,
        )
    if task == "multiclass":
        return make_classification(
            n_samples=60, n_features=7, n_informative=5, n_redundant=0,
            n_classes=3, random_state=102,
        )
    return make_regression(n_samples=54, n_features=6, random_state=103)


def _estimator(backend: str, task: str):
    classifier = task != "regression"
    if backend == "gradient_boosting":
        cls = GradientBoostingClassifier if classifier else GradientBoostingRegressor
        return cls(n_estimators=5, max_depth=2, random_state=104)
    if backend == "xgboost":
        xgboost = pytest.importorskip("xgboost")
        cls = xgboost.XGBClassifier if classifier else xgboost.XGBRegressor
        return cls(n_estimators=5, max_depth=2, random_state=104, n_jobs=1)
    if backend == "lightgbm":
        lightgbm = pytest.importorskip("lightgbm")
        cls = lightgbm.LGBMClassifier if classifier else lightgbm.LGBMRegressor
        return cls(
            n_estimators=5, max_depth=2, num_leaves=4, random_state=104,
            n_jobs=1, verbosity=-1,
        )
    catboost = pytest.importorskip("catboost")
    cls = catboost.CatBoostClassifier if classifier else catboost.CatBoostRegressor
    return cls(
        iterations=5, depth=2, random_seed=104, verbose=False, thread_count=1,
        allow_writing_files=False,
    )


@pytest.mark.parametrize(
    "backend", ["gradient_boosting", "xgboost", "lightgbm", "catboost"],
)
@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_leaf_relations_match_direct_collisions_and_reject_forest_gap(backend, task):
    X, y = _data(task)
    estimator = _estimator(backend, task).fit(X, y)
    capabilities = capabilities_for(estimator)
    assert capabilities.family == backend
    assert {"original", "kerf", "boosted"} <= set(capabilities.valid_relations)
    assert capabilities.contribution_weights
    assert not capabilities.oob
    assert not capabilities.in_bag

    original = Proximity(estimator, relation="original").attach(X, y)
    leaves = original.apply(X)
    collisions = leaves[:, None, :] == leaves[None, :, :]
    np.testing.assert_allclose(
        original.get_proximities().toarray(), collisions.mean(axis=2), atol=2e-6,
    )

    kerf = Proximity(estimator, relation="kerf").attach(X, y)
    expected_kerf = np.zeros((len(X), len(X)), dtype=float)
    for tree in range(leaves.shape[1]):
        _, inverse, counts = np.unique(
            leaves[:, tree], return_inverse=True, return_counts=True,
        )
        expected_kerf += collisions[:, :, tree] / counts[inverse][:, None]
    expected_kerf /= leaves.shape[1]
    np.testing.assert_allclose(kerf.get_proximities().toarray(), expected_kerf, atol=2e-6)

    boosted = Proximity(estimator, relation="boosted").attach(X, y)
    expected_boosted = np.sum(
        collisions * boosted.tree_weights_[None, None, :], axis=2,
    )
    np.testing.assert_allclose(
        boosted.get_proximities().toarray(), expected_boosted, atol=2e-6,
    )
    assert np.all(np.isfinite(boosted.tree_weights_))
    assert np.all(boosted.tree_weights_ >= 0)
    assert boosted.tree_weights_.sum() == pytest.approx(1.0)
    assert boosted.tree_weighting == "training_leaf_contribution_variance"
    assert boosted.relation_metadata_["tree_weighting"] == boosted.tree_weighting
    assert boosted.relation_metadata_["prediction_identity"] is False

    query_leaves = boosted.apply(X[:4])
    expected_extend = np.sum(
        (query_leaves[:, None, :] == leaves[None, :, :])
        * boosted.tree_weights_[None, None, :],
        axis=2,
    )
    np.testing.assert_allclose(boosted.prox_extend(X[:4]).toarray(), expected_extend, atol=2e-6)

    for relation in ("oob", "rfgap"):
        with pytest.raises(UnsupportedRelationError, match="exchangeable bootstrap"):
            Proximity(estimator, relation=relation).attach(X, y)


@pytest.mark.parametrize(
    "backend", ["gradient_boosting", "xgboost", "lightgbm", "catboost"],
)
@pytest.mark.parametrize("relation", ["path", "englund", "random_height"])
def test_boosted_path_relations_are_symmetric_psd_seeded_and_extend(backend, relation):
    X, y = _data("binary")
    estimator = _estimator(backend, "binary").fit(X, y)
    sparse_index = Proximity(
        estimator, relation=relation, random_state=105,
    ).attach(X, y)
    dense_index = Proximity(
        estimator, relation=relation, matrix_type="dense", random_state=105,
    ).attach(X, y)
    matrix = sparse_index.get_proximities().toarray()
    np.testing.assert_allclose(matrix, dense_index.get_proximities(), atol=2e-7)
    np.testing.assert_allclose(matrix, matrix.T, atol=2e-7)
    assert np.linalg.eigvalsh(matrix).min() >= -2e-6
    assert sparse_index.prox_extend(X[:4]).shape == (4, len(X))
    if relation == "random_height":
        assert (
            sparse_index.relation_metadata_["tree_cut_heights"]
            == dense_index.relation_metadata_["tree_cut_heights"]
        )


def test_gradient_boosting_zero_contribution_variance_falls_back_to_uniform():
    rng = np.random.default_rng(106)
    X = rng.normal(size=(35, 4))
    y = np.ones(len(X))
    estimator = GradientBoostingRegressor(n_estimators=4, random_state=107).fit(X, y)
    proximity = Proximity(estimator, relation="boosted").attach(X, y)
    np.testing.assert_allclose(proximity.tree_weights_, np.full(4, 0.25))


def test_optional_boosters_are_not_imported_by_forestgap_import():
    code = (
        "import sys; import forestgap; "
        "assert not ({'xgboost', 'lightgbm', 'catboost'} & set(sys.modules))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_catboost_preserves_categorical_dataframe_and_pool_and_gates_growth_policy():
    catboost = pytest.importorskip("catboost")
    X = pd.DataFrame({
        "category": ["a", "b", "a", "c", "b", "c"] * 8,
        "value": np.tile(np.arange(8, dtype=float), 6),
    })
    y = ((X["category"] == "a") | (X["value"] > 4)).astype(int)
    estimator = catboost.CatBoostClassifier(
        iterations=5, depth=3, random_seed=108, verbose=False, thread_count=1,
        cat_features=["category"], allow_writing_files=False,
    ).fit(X, y)
    dataframe_index = Proximity(estimator, relation="boosted").attach(X, y)
    assert isinstance(dataframe_index._X_train_native, pd.DataFrame)
    assert dataframe_index.prox_extend(X.iloc[:3]).shape == (3, len(X))

    pool = catboost.Pool(X, y, cat_features=["category"])
    pool_index = Proximity(estimator, relation="boosted").attach(pool, y)
    assert pool_index._X_train_native is pool
    assert pool_index.get_proximities().shape == (len(X), len(X))

    nonsymmetric = catboost.CatBoostClassifier(
        iterations=4, depth=3, grow_policy="Depthwise", random_seed=109,
        verbose=False, thread_count=1, cat_features=["category"],
        allow_writing_files=False,
    ).fit(X, y)
    capabilities = capabilities_for(nonsymmetric)
    assert {"original", "kerf", "boosted"} <= set(capabilities.valid_relations)
    assert not ({"path", "englund", "random_height"} & set(capabilities.valid_relations))
    for relation in ("path", "englund", "random_height"):
        with pytest.raises(UnsupportedRelationError, match="SymmetricTree"):
            Proximity(nonsymmetric, relation=relation).attach(X, y)
