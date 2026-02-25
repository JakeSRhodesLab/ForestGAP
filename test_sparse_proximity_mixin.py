import unittest
import warnings

import numpy as np
from scipy import sparse
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from forestgap import with_sparse_proximity

# Suppress sklearn deprecation warnings from aeon's RotationForest internal usage
# (aeon uses deprecated _validate_data internally, which is not our code's responsibility)
warnings.filterwarnings(
    "ignore",
    message=".*BaseEstimator._validate_data.*",
    category=FutureWarning,
)

try:
    from aeon.classification.sklearn import RotationForestClassifier
    from aeon.regression.sklearn import RotationForestRegressor
    _HAS_AEON = True
except Exception:
    RotationForestClassifier = None
    RotationForestRegressor = None
    _HAS_AEON = False


class TestSparseProximityMixin(unittest.TestCase):
    @staticmethod
    def _check_common_outputs(model, X, expect_dense: bool = False):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*BaseEstimator._validate_data.*",
                category=FutureWarning,
            )
            model.fit(X["train"], X["target"])

        prox = model.get_proximities()
        prox_ext = model.prox_extend(X["test"])

        n_train = X["train"].shape[0]
        n_test = X["test"].shape[0]

        if expect_dense:
            assert isinstance(prox, np.ndarray)
            assert isinstance(prox_ext, np.ndarray)
        else:
            assert sparse.isspmatrix_csr(prox)
            assert sparse.isspmatrix_csr(prox_ext)

        assert prox.shape == (n_train, n_train)
        assert prox_ext.shape == (n_test, n_train)

    @staticmethod
    def _to_dense(mat):
        return mat.toarray() if sparse.isspmatrix(mat) else mat

    def _assert_regression_prediction_equivalence(self, model, X_train, y_train, X_test):
        model.fit(X_train, y_train)

        prox_test = self._to_dense(model.prox_extend(X_test))
        prox_preds = prox_test @ y_train
        model_preds = model.predict(X_test)

        np.testing.assert_allclose(prox_preds, model_preds, rtol=1e-5, atol=1e-6)

    def _assert_classification_prediction_equivalence(self, model, X_train, y_train, X_test):
        model.fit(X_train, y_train)

        prox_test = self._to_dense(model.prox_extend(X_test))
        classes = model.classes_

        y_train_idx = np.searchsorted(classes, y_train)
        y_one_hot = np.eye(classes.shape[0], dtype=np.float64)[y_train_idx]

        vote_scores = prox_test @ y_one_hot
        prox_pred_idx = np.argmax(vote_scores, axis=1)
        prox_preds = classes[prox_pred_idx]
        model_preds = model.predict(X_test)

        mismatches = np.where(prox_preds != model_preds)[0]
        if mismatches.size > 0:
            max_scores = vote_scores.max(axis=1, keepdims=True)
            tied_max = np.isclose(vote_scores, max_scores, rtol=1e-8, atol=1e-12)
            model_pred_idx = np.searchsorted(classes, model_preds)

            for i in mismatches:
                self.assertTrue(
                    tied_max[i, model_pred_idx[i]],
                    msg=(
                        "Classification mismatch is not explained by a top-vote tie "
                        f"at row {i}."
                    ),
                )

    def test_random_forest_classifier(self):
        X, y = make_classification(
            n_samples=180,
            n_features=12,
            n_informative=8,
            random_state=7,
        )
        data = {"train": X[:140], "target": y[:140], "test": X[140:]}

        ProxRFC = with_sparse_proximity(RandomForestClassifier)
        model = ProxRFC(
            n_estimators=30,
            random_state=1,
            prox_method="rfgap",
            matrix_type="sparse",
            bootstrap=True,
        )
        self._check_common_outputs(model, data)

    def test_random_forest_regressor(self):
        reg_data = make_regression(
            n_samples=180,
            n_features=10,
            n_informative=7,
            noise=0.2,
            random_state=11,
        )
        X, y = reg_data[0], reg_data[1]
        data = {"train": X[:140], "target": y[:140], "test": X[140:]}

        ProxRFR = with_sparse_proximity(RandomForestRegressor)
        model = ProxRFR(
            n_estimators=30,
            random_state=2,
            prox_method="oob",
            matrix_type="sparse",
            bootstrap=True,
        )
        self._check_common_outputs(model, data)

    def test_extra_trees_classifier(self):
        X, y = make_classification(
            n_samples=180,
            n_features=14,
            n_informative=9,
            random_state=19,
        )
        data = {"train": X[:140], "target": y[:140], "test": X[140:]}

        ProxETC = with_sparse_proximity(ExtraTreesClassifier)
        model = ProxETC(
            n_estimators=30,
            random_state=3,
            prox_method="rfgap",
            matrix_type="sparse",
            bootstrap=True,
        )
        self._check_common_outputs(model, data)

    def test_extra_trees_regressor_dense_output(self):
        reg_data = make_regression(
            n_samples=180,
            n_features=9,
            n_informative=6,
            noise=0.3,
            random_state=29,
        )
        X, y = reg_data[0], reg_data[1]
        data = {"train": X[:140], "target": y[:140], "test": X[140:]}

        ProxETR = with_sparse_proximity(ExtraTreesRegressor)
        model = ProxETR(
            n_estimators=30,
            random_state=4,
            prox_method="original",
            matrix_type="dense",
            bootstrap=True,
        )
        self._check_common_outputs(model, data, expect_dense=True)

    def test_rfgap_prediction_equivalence_random_forest_regressor(self):
        reg_data = make_regression(
            n_samples=220,
            n_features=12,
            n_informative=8,
            noise=0.25,
            random_state=101,
        )
        X, y = reg_data[0], reg_data[1]
        X_train, y_train, X_test = X[:170], y[:170], X[170:]

        ProxRFR = with_sparse_proximity(RandomForestRegressor)
        model = ProxRFR(
            n_estimators=80,
            random_state=10,
            prox_method="rfgap",
            matrix_type="sparse",
            bootstrap=True,
        )
        self._assert_regression_prediction_equivalence(model, X_train, y_train, X_test)

    def test_rfgap_prediction_equivalence_random_forest_classifier(self):
        X, y = make_classification(
            n_samples=220,
            n_features=12,
            n_informative=8,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=102,
        )
        X_train, y_train, X_test = X[:170], y[:170], X[170:]

        ProxRFC = with_sparse_proximity(RandomForestClassifier)
        model = ProxRFC(
            n_estimators=80,
            random_state=11,
            prox_method="rfgap",
            matrix_type="sparse",
            bootstrap=True,
        )
        self._assert_classification_prediction_equivalence(model, X_train, y_train, X_test)

    def test_rfgap_prediction_equivalence_extra_trees_regressor(self):
        reg_data = make_regression(
            n_samples=220,
            n_features=11,
            n_informative=7,
            noise=0.3,
            random_state=103,
        )
        X, y = reg_data[0], reg_data[1]
        X_train, y_train, X_test = X[:170], y[:170], X[170:]

        ProxETR = with_sparse_proximity(ExtraTreesRegressor)
        model = ProxETR(
            n_estimators=80,
            random_state=12,
            prox_method="rfgap",
            matrix_type="sparse",
            bootstrap=True,
        )
        self._assert_regression_prediction_equivalence(model, X_train, y_train, X_test)

    def test_rfgap_prediction_equivalence_extra_trees_classifier(self):
        X, y = make_classification(
            n_samples=220,
            n_features=13,
            n_informative=9,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=104,
        )
        X_train, y_train, X_test = X[:170], y[:170], X[170:]

        ProxETC = with_sparse_proximity(ExtraTreesClassifier)
        model = ProxETC(
            n_estimators=80,
            random_state=13,
            prox_method="rfgap",
            matrix_type="sparse",
            bootstrap=True,
        )
        self._assert_classification_prediction_equivalence(model, X_train, y_train, X_test)

    @unittest.skipUnless(_HAS_AEON, "aeon is required for RotationForest tests")
    def test_rotation_forest_classifier_original_proximity(self):
        """
        Test RotationForestClassifier with original proximity.
        
        Note: Prediction equivalence is NOT guaranteed for RotationForestClassifier because:
        - Proximity-based predictions use hard voting (argmax of proximity-weighted class counts)
        - RotationForest.predict() uses soft voting (argmax of averaged probabilities)
        
        These can differ when probability distributions are skewed differently than vote counts.
        """
        X, y = make_classification(
            n_samples=220,
            n_features=12,
            n_informative=8,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=201,
        )
        X_train, y_train, X_test = X[:170], y[:170], X[170:]

        ProxRotC = with_sparse_proximity(RotationForestClassifier)
        model = ProxRotC(
            n_estimators=40,
            random_state=21,
            prox_method="original",
            matrix_type="sparse",
        )
        self._check_common_outputs(model, {"train": X_train, "target": y_train, "test": X_test})
        
        # Test proximity computation works, but don't enforce prediction equivalence
        # due to hard vs soft voting difference

    @unittest.skipUnless(_HAS_AEON, "aeon is required for RotationForest tests")
    def test_rotation_forest_regressor_original_proximity(self):
        reg_data = make_regression(
            n_samples=220,
            n_features=10,
            n_informative=7,
            noise=0.2,
            random_state=202,
        )
        X, y = reg_data[0], reg_data[1]
        X_train, y_train, X_test = X[:170], y[:170], X[170:]

        ProxRotR = with_sparse_proximity(RotationForestRegressor)
        model = ProxRotR(
            n_estimators=40,
            random_state=22,
            prox_method="original",
            matrix_type="sparse",
        )
        self._check_common_outputs(model, {"train": X_train, "target": y_train, "test": X_test})
        self._assert_regression_prediction_equivalence(model, X_train, y_train, X_test)


if __name__ == "__main__":
    unittest.main()
