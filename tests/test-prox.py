import unittest
import warnings
import inspect

import numpy as np
from scipy import sparse
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    IsolationForest,
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
    PROX_METHODS = ("original", "oob", "rfgap")

    @staticmethod
    def _to_dense(mat):
        return mat.toarray() if sparse.isspmatrix(mat) else mat

    @staticmethod
    def _fit(model, X_train, y_train=None):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*BaseEstimator._validate_data.*",
                category=FutureWarning,
            )
            if y_train is None:
                model.fit(X_train)
            else:
                model.fit(X_train, y_train)

    @staticmethod
    def _build_model(base_cls, prox_method, n_estimators=20, random_state=0, **kwargs):
        Prox = with_sparse_proximity(base_cls)
        return Prox(
            n_estimators=n_estimators,
            random_state=random_state,
            prox_method=prox_method,
            matrix_type="sparse",
            **kwargs,
        )

    @staticmethod
    def _classification_data():
        X, y = make_classification(
            n_samples=120,
            n_features=10,
            n_informative=7,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=123,
        )
        return X[:90], y[:90], X[90:], y[90:]

    @staticmethod
    def _regression_data():
        X, y = make_regression(
            n_samples=120,
            n_features=10,
            n_informative=7,
            noise=0.2,
            random_state=124,
        )
        return X[:90], y[:90], X[90:], y[90:]

    @staticmethod
    def _unsupervised_data():
        X, _ = make_regression(
            n_samples=120,
            n_features=10,
            n_informative=7,
            noise=0.2,
            random_state=125,
        )
        return X[:90], X[90:]

    def _classification_models(self):
        models = [
            ("RandomForestClassifier", RandomForestClassifier),
            ("ExtraTreesClassifier", ExtraTreesClassifier),
        ]
        if _HAS_AEON:
            models.append(("RotationForestClassifier", RotationForestClassifier))
        return models

    def _regression_models(self):
        models = [
            ("RandomForestRegressor", RandomForestRegressor),
            ("ExtraTreesRegressor", ExtraTreesRegressor),
        ]
        if _HAS_AEON:
            models.append(("RotationForestRegressor", RotationForestRegressor))
        return models

    def test_fit_predict_get_proximities_all_models_methods(self):
        Xc_tr, yc_tr, Xc_te, _ = self._classification_data()
        Xr_tr, yr_tr, Xr_te, _ = self._regression_data()
        Xu_tr, Xu_te = self._unsupervised_data()

        for name, cls in self._classification_models():
            for method in self.PROX_METHODS:
                with self.subTest(model=name, prox_method=method):
                    model = self._build_model(cls, method, n_estimators=20, random_state=1)
                    self._fit(model, Xc_tr, yc_tr)
                    _ = model.predict(Xc_te)
                    prox = model.get_proximities()
                    prox_ext = model.prox_extend(Xc_te)
                    self.assertEqual(prox.shape, (Xc_tr.shape[0], Xc_tr.shape[0]))
                    self.assertEqual(prox_ext.shape, (Xc_te.shape[0], Xc_tr.shape[0]))

        for name, cls in self._regression_models():
            for method in self.PROX_METHODS:
                with self.subTest(model=name, prox_method=method):
                    model = self._build_model(cls, method, n_estimators=20, random_state=2)
                    self._fit(model, Xr_tr, yr_tr)
                    _ = model.predict(Xr_te)
                    prox = model.get_proximities()
                    prox_ext = model.prox_extend(Xr_te)
                    self.assertEqual(prox.shape, (Xr_tr.shape[0], Xr_tr.shape[0]))
                    self.assertEqual(prox_ext.shape, (Xr_te.shape[0], Xr_tr.shape[0]))

        for method in self.PROX_METHODS:
            with self.subTest(model="IsolationForest", prox_method=method):
                model = self._build_model(IsolationForest, method, n_estimators=20, random_state=3)
                self._fit(model, Xu_tr)
                _ = model.predict(Xu_te)
                prox = model.get_proximities()
                prox_ext = model.prox_extend(Xu_te)
                self.assertEqual(prox.shape, (Xu_tr.shape[0], Xu_tr.shape[0]))
                self.assertEqual(prox_ext.shape, (Xu_te.shape[0], Xu_tr.shape[0]))

    def test_supervised_rfgap_proximity_matches_oob_outputs(self):
        Xc_tr, yc_tr, _, _ = self._classification_data()
        Xr_tr, yr_tr, _, _ = self._regression_data()

        for name, cls in self._classification_models():
            with self.subTest(task="classification", model=name, prox_method="rfgap"):
                kwargs = {"oob_score": True}
                if "bootstrap" in inspect.signature(cls.__init__).parameters:
                    kwargs["bootstrap"] = True

                model = self._build_model(cls, "rfgap", n_estimators=20, random_state=4, **kwargs)
                self._fit(model, Xc_tr, yc_tr)

                classes = np.asarray(model.classes_)
                y_idx = np.searchsorted(classes, yc_tr)
                y_hot = np.eye(classes.shape[0], dtype=np.float64)[y_idx]

                prox = self._to_dense(model.get_proximities())
                prox_prob = prox @ y_hot
                rs = prox_prob.sum(axis=1, keepdims=True)
                rs[rs == 0] = 1.0
                prox_prob = prox_prob / rs

                oob_prob = np.asarray(model.oob_decision_function_, dtype=np.float64)
                np.testing.assert_allclose(prox_prob, oob_prob, rtol=1e-6, atol=1e-6)

        for name, cls in self._regression_models():
            with self.subTest(task="regression", model=name, prox_method="rfgap"):
                kwargs = {"oob_score": True}
                if "bootstrap" in inspect.signature(cls.__init__).parameters:
                    kwargs["bootstrap"] = True

                model = self._build_model(cls, "rfgap", n_estimators=20, random_state=5, **kwargs)
                self._fit(model, Xr_tr, yr_tr)

                prox = self._to_dense(model.get_proximities())
                prox_pred = prox @ yr_tr
                oob_pred = np.asarray(model.oob_prediction_, dtype=np.float64)
                np.testing.assert_allclose(prox_pred, oob_pred, rtol=1e-6, atol=1e-6)

    def test_fit_predict_for_oob_score_and_bootstrap_combinations(self):
        Xc_tr, yc_tr, Xc_te, _ = self._classification_data()
        Xr_tr, yr_tr, Xr_te, _ = self._regression_data()
        Xu_tr, Xu_te = self._unsupervised_data()

        families = []
        families.extend((name, cls, "classification") for name, cls in self._classification_models())
        families.extend((name, cls, "regression") for name, cls in self._regression_models())
        families.append(("IsolationForest", IsolationForest, "unsupervised"))

        for name, cls, task in families:
            base_sig = inspect.signature(cls.__init__)
            bootstrap_values = [None]
            if "bootstrap" in base_sig.parameters:
                bootstrap_values = [False, True]

            for oob_score in (False, True):
                for bootstrap in bootstrap_values:
                    with self.subTest(model=name, task=task, oob_score=oob_score, bootstrap=bootstrap):
                        kwargs = {"oob_score": oob_score}
                        if bootstrap is not None:
                            kwargs["bootstrap"] = bootstrap

                        model = self._build_model(cls, "rfgap", n_estimators=20, random_state=6, **kwargs)

                        if task == "classification":
                            self._fit(model, Xc_tr, yc_tr)
                            _ = model.predict(Xc_te)
                            prox = model.get_proximities()
                            self.assertEqual(prox.shape, (Xc_tr.shape[0], Xc_tr.shape[0]))
                        elif task == "regression":
                            self._fit(model, Xr_tr, yr_tr)
                            _ = model.predict(Xr_te)
                            prox = model.get_proximities()
                            self.assertEqual(prox.shape, (Xr_tr.shape[0], Xr_tr.shape[0]))
                        else:
                            self._fit(model, Xu_tr)
                            _ = model.predict(Xu_te)
                            prox = model.get_proximities()
                            self.assertEqual(prox.shape, (Xu_tr.shape[0], Xu_tr.shape[0]))

    def test_proximity_matrices_are_finite_and_nonnegative(self):
        Xc_tr, yc_tr, _, _ = self._classification_data()
        model = self._build_model(RandomForestClassifier, "rfgap", n_estimators=20, random_state=7, bootstrap=True)
        self._fit(model, Xc_tr, yc_tr)
        prox = self._to_dense(model.get_proximities())
        self.assertTrue(np.isfinite(prox).all())
        self.assertGreaterEqual(prox.min(), 0.0)


if __name__ == "__main__":
    unittest.main()
