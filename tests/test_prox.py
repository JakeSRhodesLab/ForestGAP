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

try:
    from treeple import (
        ObliqueRandomForestClassifier,
        ObliqueRandomForestRegressor,
        PatchObliqueRandomForestClassifier,
        PatchObliqueRandomForestRegressor,
        ExtraObliqueRandomForestClassifier,
        ExtraObliqueRandomForestRegressor,
        HonestForestClassifier,
        UnsupervisedRandomForest,
        ExtendedIsolationForest,
    )
    _HAS_TREEPLE = True
except Exception:
    _HAS_TREEPLE = False


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

    @staticmethod
    def _binary_signal_data():
        rng = np.random.RandomState(321)
        X = rng.randint(0, 2, size=(160, 3)).astype(float)
        y = X[:, 0].astype(int)
        return X[:120], y[:120], X[120:], y[120:]

    @staticmethod
    def _prototype_data():
        import pandas as pd

        X = pd.DataFrame(
            {
                "x1": [0] * 8 + [5] * 8,
                "x2": [0] * 4 + [4] * 4 + [0] * 4 + [4] * 4,
                "texture_code": [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4,
            },
            index=pd.Index([f"case_{idx}" for idx in range(16)], name="sample"),
        )
        y = np.array([0] * 8 + [1] * 8, dtype=int)
        return X, y

    def _classification_models(self):
        models = [
            ("RandomForestClassifier", RandomForestClassifier),
            ("ExtraTreesClassifier", ExtraTreesClassifier),
        ]
        if _HAS_AEON:
            models.append(("RotationForestClassifier", RotationForestClassifier))
        if _HAS_TREEPLE:
            models.extend([
                ("ObliqueRandomForestClassifier", ObliqueRandomForestClassifier),
                ("PatchObliqueRandomForestClassifier", PatchObliqueRandomForestClassifier),
                ("ExtraObliqueRandomForestClassifier", ExtraObliqueRandomForestClassifier),
                ("HonestForestClassifier", HonestForestClassifier),
            ])
        return models

    def _regression_models(self):
        models = [
            ("RandomForestRegressor", RandomForestRegressor),
            ("ExtraTreesRegressor", ExtraTreesRegressor),
        ]
        if _HAS_AEON:
            models.append(("RotationForestRegressor", RotationForestRegressor))
        if _HAS_TREEPLE:
            models.extend([
                ("ObliqueRandomForestRegressor", ObliqueRandomForestRegressor),
                ("PatchObliqueRandomForestRegressor", PatchObliqueRandomForestRegressor),
                ("ExtraObliqueRandomForestRegressor", ExtraObliqueRandomForestRegressor),
            ])
        return models

    def _unsupervised_models(self):
        models = [
            ("IsolationForest", IsolationForest),
        ]
        if _HAS_TREEPLE:
            models.extend([
                ("UnsupervisedRandomForest", UnsupervisedRandomForest),
                ("ExtendedIsolationForest", ExtendedIsolationForest),
            ])
        return models

    def _classification_models_oob_matchable(self):
        """Classification models whose OOB outputs match prox @ y exactly.

        HonestForestClassifier is excluded because honest splitting computes
        OOB predictions from a held-out honest subset, which structurally
        differs from the bootstrap-based proximity OOB.
        """
        return [(n, c) for n, c in self._classification_models()
                if "Honest" not in n]

    def _regression_models_oob_matchable(self):
        return self._regression_models()

    def _classification_models_extend_matchable(self):
        """Classification models whose extended rfgap proximities reproduce predict.

        HonestForestClassifier is excluded because honest splitting computes
        predictions from a held-out honest subset.
        """
        return [(n, c) for n, c in self._classification_models()
                if "Honest" not in n]

    def _regression_models_extend_matchable(self):
        """Regression models whose extended rfgap proximities reproduce predict."""
        return self._regression_models()

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

        for name, cls in self._unsupervised_models():
            for method in self.PROX_METHODS:
                with self.subTest(model=name, prox_method=method):
                    kwargs = {}
                    # UnsupervisedRandomForest defaults to bootstrap=False;
                    # enable it for oob/rfgap methods.
                    base_sig = inspect.signature(cls.__init__)
                    if "bootstrap" in base_sig.parameters:
                        kwargs["bootstrap"] = method in ("oob", "rfgap")
                    model = self._build_model(cls, method, n_estimators=20, random_state=3, **kwargs)
                    self._fit(model, Xu_tr)
                    _ = model.predict(Xu_te)
                    prox = model.get_proximities()
                    prox_ext = model.prox_extend(Xu_te)
                    self.assertEqual(prox.shape, (Xu_tr.shape[0], Xu_tr.shape[0]))
                    self.assertEqual(prox_ext.shape, (Xu_te.shape[0], Xu_tr.shape[0]))

    def test_supervised_rfgap_proximity_matches_oob_outputs(self):
        Xc_tr, yc_tr, _, _ = self._classification_data()
        Xr_tr, yr_tr, _, _ = self._regression_data()

        for name, cls in self._classification_models_oob_matchable():
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

        for name, cls in self._regression_models_oob_matchable():
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
        families.extend((name, cls, "unsupervised") for name, cls in self._unsupervised_models())

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

    def test_supervised_rfgap_extend_matches_predictions(self):
        """Extended rfgap proximities used as weights reproduce forest predictions."""
        Xc_tr, yc_tr, Xc_te, _ = self._classification_data()
        Xr_tr, yr_tr, Xr_te, _ = self._regression_data()

        for name, cls in self._classification_models_extend_matchable():
            with self.subTest(task="classification", model=name):
                kwargs = {"bootstrap": True}
                if "bootstrap" not in inspect.signature(cls.__init__).parameters:
                    kwargs = {}
                model = self._build_model(cls, "rfgap", n_estimators=20, random_state=8, **kwargs)
                self._fit(model, Xc_tr, yc_tr)

                prox_ext = self._to_dense(model.prox_extend(Xc_te))  # (n_test, n_train)

                classes = np.asarray(model.classes_)
                y_idx = np.searchsorted(classes, yc_tr)
                y_hot = np.eye(classes.shape[0], dtype=np.float64)[y_idx]

                # prox_ext @ y_one_hot should match predict_proba
                prox_prob = prox_ext @ y_hot

                forest_prob = np.asarray(model.predict_proba(Xc_te), dtype=np.float64)
                np.testing.assert_allclose(prox_prob, forest_prob, rtol=1e-4, atol=1e-4)

        for name, cls in self._regression_models_extend_matchable():
            with self.subTest(task="regression", model=name):
                kwargs = {"bootstrap": True}
                if "bootstrap" not in inspect.signature(cls.__init__).parameters:
                    kwargs = {}
                model = self._build_model(cls, "rfgap", n_estimators=20, random_state=9, **kwargs)
                self._fit(model, Xr_tr, yr_tr)

                prox_ext = self._to_dense(model.prox_extend(Xr_te))  # (n_test, n_train)

                # prox_ext @ y_train should match predict
                prox_pred = prox_ext @ yr_tr

                forest_pred = np.asarray(model.predict(Xr_te), dtype=np.float64)
                np.testing.assert_allclose(prox_pred, forest_pred, rtol=1e-4, atol=1e-4)

    def test_prox_predict_test_data_matches_forest_predict(self):
        """prox_predict(X_test) reproduces the forest's test predictions."""
        Xc_tr, yc_tr, Xc_te, _ = self._classification_data()
        Xr_tr, yr_tr, Xr_te, _ = self._regression_data()

        for name, cls in self._classification_models_extend_matchable():
            with self.subTest(task="classification", model=name):
                kwargs = {"bootstrap": True}
                if "bootstrap" not in inspect.signature(cls.__init__).parameters:
                    kwargs = {}
                model = self._build_model(cls, "rfgap", n_estimators=20, random_state=10, **kwargs)
                self._fit(model, Xc_tr, yc_tr)

                prox_prob = model.prox_predict(Xc_te)
                forest_prob = np.asarray(model.predict_proba(Xc_te), dtype=np.float64)
                np.testing.assert_allclose(prox_prob, forest_prob, rtol=1e-4, atol=1e-4)

        for name, cls in self._regression_models_extend_matchable():
            with self.subTest(task="regression", model=name):
                kwargs = {"bootstrap": True}
                if "bootstrap" not in inspect.signature(cls.__init__).parameters:
                    kwargs = {}
                model = self._build_model(cls, "rfgap", n_estimators=20, random_state=11, **kwargs)
                self._fit(model, Xr_tr, yr_tr)

                prox_pred = model.prox_predict(Xr_te)
                forest_pred = np.asarray(model.predict(Xr_te), dtype=np.float64)
                np.testing.assert_allclose(prox_pred, forest_pred, rtol=1e-4, atol=1e-4)

    def test_prox_predict_training_data_matches_oob(self):
        """prox_predict() with no args returns OOB predictions for training data."""
        Xc_tr, yc_tr, _, _ = self._classification_data()
        Xr_tr, yr_tr, _, _ = self._regression_data()

        for name, cls in self._classification_models_oob_matchable():
            with self.subTest(task="classification", model=name):
                kwargs = {"oob_score": True}
                if "bootstrap" in inspect.signature(cls.__init__).parameters:
                    kwargs["bootstrap"] = True
                model = self._build_model(cls, "rfgap", n_estimators=20, random_state=12, **kwargs)
                self._fit(model, Xc_tr, yc_tr)

                prox_prob = model.prox_predict()
                oob_prob = np.asarray(model.oob_decision_function_, dtype=np.float64)
                np.testing.assert_allclose(prox_prob, oob_prob, rtol=1e-6, atol=1e-6)

        for name, cls in self._regression_models_oob_matchable():
            with self.subTest(task="regression", model=name):
                kwargs = {"oob_score": True}
                if "bootstrap" in inspect.signature(cls.__init__).parameters:
                    kwargs["bootstrap"] = True
                model = self._build_model(cls, "rfgap", n_estimators=20, random_state=13, **kwargs)
                self._fit(model, Xr_tr, yr_tr)

                prox_pred = model.prox_predict()
                oob_pred = np.asarray(model.oob_prediction_, dtype=np.float64)
                np.testing.assert_allclose(prox_pred, oob_pred, rtol=1e-6, atol=1e-6)

    def test_proximity_matrices_are_finite_and_nonnegative(self):
        Xc_tr, yc_tr, _, _ = self._classification_data()
        model = self._build_model(RandomForestClassifier, "rfgap", n_estimators=20, random_state=7, bootstrap=True)
        self._fit(model, Xc_tr, yc_tr)
        prox = self._to_dense(model.get_proximities())
        self.assertTrue(np.isfinite(prox).all())
        self.assertGreaterEqual(prox.min(), 0.0)

    def test_impute_fills_missing_values(self):
        """impute fills NaN entries and leaves observed entries unchanged."""
        rng = np.random.RandomState(42)
        Xc_tr, yc_tr, Xc_te, _ = self._classification_data()

        # Inject ~10% missing values at random
        mask_tr = rng.rand(*Xc_tr.shape) < 0.10
        mask_te = rng.rand(*Xc_te.shape) < 0.10
        X_miss_tr = Xc_tr.copy()
        X_miss_te = Xc_te.copy()
        X_miss_tr[mask_tr] = np.nan
        X_miss_te[mask_te] = np.nan

        model = self._build_model(RandomForestClassifier, "rfgap", n_estimators=50, random_state=14, bootstrap=True)
        # Fit on the rough-filled data (median init)
        X_init = X_miss_tr.copy()
        col_medians = np.nanmedian(X_init, axis=0)
        for m in range(X_init.shape[1]):
            nans = np.isnan(X_init[:, m])
            X_init[nans, m] = col_medians[m]
        self._fit(model, X_init, yc_tr)

        # Impute with train + test
        X_imp_tr, X_imp_te = model.impute(X_miss_tr, X_test=X_miss_te)

        # 1. No NaN remaining
        self.assertFalse(np.isnan(X_imp_tr).any(), "Train imputation left NaN values")
        self.assertFalse(np.isnan(X_imp_te).any(), "Test imputation left NaN values")

        # 2. Observed entries are unchanged
        np.testing.assert_array_equal(X_imp_tr[~mask_tr], Xc_tr[~mask_tr])
        np.testing.assert_array_equal(X_imp_te[~mask_te], Xc_te[~mask_te])

        # 3. All values are finite
        self.assertTrue(np.isfinite(X_imp_tr).all())
        self.assertTrue(np.isfinite(X_imp_te).all())

    def test_impute_categorical(self):
        """impute correctly imputes categorical features via weighted vote."""
        rng = np.random.RandomState(99)
        Xc_tr, yc_tr, _, _ = self._classification_data()

        # Add a categorical column (integer-valued)
        cat_col = yc_tr.copy().astype(float)
        X_with_cat = np.column_stack([Xc_tr, cat_col])

        # Inject missing in the categorical column
        miss_idx = rng.choice(X_with_cat.shape[0], size=5, replace=False)
        X_miss = X_with_cat.copy()
        X_miss[miss_idx, -1] = np.nan

        model = self._build_model(RandomForestClassifier, "rfgap", n_estimators=50, random_state=15, bootstrap=True)
        X_init = X_miss.copy()
        col_medians = np.nanmedian(X_init, axis=0)
        for m in range(X_init.shape[1]):
            nans = np.isnan(X_init[:, m])
            X_init[nans, m] = col_medians[m]
        self._fit(model, X_init, yc_tr)

        X_imp = model.impute(X_miss, categorical_features=[X_with_cat.shape[1] - 1])

        # No NaN remaining
        self.assertFalse(np.isnan(X_imp).any())

        # Categorical column should contain only original category values
        orig_cats = set(np.unique(cat_col))
        imputed_cats = set(np.unique(X_imp[:, -1]))
        self.assertTrue(imputed_cats.issubset(orig_cats))

    def test_impute_initialization_options(self):
        """impute works with mean, median, and knn initialization for test data."""
        rng = np.random.RandomState(42)
        Xc_tr, yc_tr, Xc_te, _ = self._classification_data()

        mask_tr = rng.rand(*Xc_tr.shape) < 0.10
        mask_te = rng.rand(*Xc_te.shape) < 0.10
        X_miss_tr = Xc_tr.copy()
        X_miss_te = Xc_te.copy()
        X_miss_tr[mask_tr] = np.nan
        X_miss_te[mask_te] = np.nan

        # Fit on rough-filled data
        X_init = X_miss_tr.copy()
        col_medians = np.nanmedian(X_init, axis=0)
        for m in range(X_init.shape[1]):
            nans = np.isnan(X_init[:, m])
            X_init[nans, m] = col_medians[m]

        model = self._build_model(
            RandomForestClassifier, "rfgap", n_estimators=50,
            random_state=14, bootstrap=True,
        )
        self._fit(model, X_init, yc_tr)

        for init in ("mean", "median", "knn"):
            with self.subTest(initialization=init):
                X_imp_tr, X_imp_te = model.impute(
                    X_miss_tr, X_test=X_miss_te, initialization=init,
                )
                self.assertFalse(np.isnan(X_imp_tr).any())
                self.assertFalse(np.isnan(X_imp_te).any())
                self.assertTrue(np.isfinite(X_imp_tr).all())
                self.assertTrue(np.isfinite(X_imp_te).all())
                # Observed entries unchanged
                np.testing.assert_array_equal(X_imp_tr[~mask_tr], Xc_tr[~mask_tr])
                np.testing.assert_array_equal(X_imp_te[~mask_te], Xc_te[~mask_te])

    def test_outlier_scores_basic(self):
        """outlier_scores returns finite, non-negative scores for every sample."""
        Xc_tr, yc_tr, _, _ = self._classification_data()

        for method in self.PROX_METHODS:
            with self.subTest(prox_method=method):
                model = self._build_model(
                    RandomForestClassifier, method, n_estimators=50,
                    random_state=16, bootstrap=True,
                )
                self._fit(model, Xc_tr, yc_tr)

                scores = model.outlier_scores(yc_tr)
                self.assertEqual(scores.shape, (Xc_tr.shape[0],))
                self.assertTrue(np.isfinite(scores).all())
                self.assertGreaterEqual(scores.min(), 0.0)

    def test_outlier_scores_detects_corruption(self):
        """outlier_scores assigns higher scores to corrupted samples."""
        rng = np.random.RandomState(77)
        Xc_tr, yc_tr, _, _ = self._classification_data()

        # Corrupt 3 samples by adding very large noise
        corrupt_idx = rng.choice(Xc_tr.shape[0], size=3, replace=False)
        X_corrupt = Xc_tr.copy()
        X_corrupt[corrupt_idx] += rng.normal(0, 50, size=(3, Xc_tr.shape[1]))

        model = self._build_model(
            RandomForestClassifier, "rfgap", n_estimators=100,
            random_state=17, bootstrap=True,
        )
        self._fit(model, X_corrupt, yc_tr)

        scores = model.outlier_scores(yc_tr)

        # Corrupted samples should have above-median scores
        median_score = np.median(scores)
        self.assertTrue(
            (scores[corrupt_idx] >= median_score).all(),
            "Corrupted samples should have scores at or above the median",
        )

    def test_outlier_scores_scaling(self):
        """outlier_scores respects normalize and log scaling options."""
        Xc_tr, yc_tr, _, _ = self._classification_data()

        model = self._build_model(
            RandomForestClassifier, "rfgap", n_estimators=50,
            random_state=18, bootstrap=True,
        )
        self._fit(model, Xc_tr, yc_tr)

        raw = model.outlier_scores(yc_tr)
        normed = model.outlier_scores(yc_tr, scaling="normalize")
        logged = model.outlier_scores(yc_tr, scaling="log")

        # Normalised scores lie in [0, 1]
        self.assertGreaterEqual(normed.min(), 0.0)
        self.assertLessEqual(normed.max(), 1.0)

        # Log scores are non-negative and finite
        self.assertTrue(np.isfinite(logged).all())
        self.assertGreaterEqual(logged.min(), 0.0)

    # ---- DataFrame tests ------------------------------------------------

    def test_impute_dataframe(self):
        """impute accepts DataFrames and returns DataFrames with preserved index/columns."""
        import pandas as pd

        rng = np.random.RandomState(42)
        Xc_tr, yc_tr, Xc_te, _ = self._classification_data()

        cols = [f"feat_{i}" for i in range(Xc_tr.shape[1])]
        idx_tr = pd.RangeIndex(100, 100 + Xc_tr.shape[0])
        idx_te = pd.RangeIndex(200, 200 + Xc_te.shape[0])

        df_tr = pd.DataFrame(Xc_tr, columns=cols, index=idx_tr)
        df_te = pd.DataFrame(Xc_te, columns=cols, index=idx_te)

        # Inject ~10% missing values
        mask_tr = rng.rand(*Xc_tr.shape) < 0.10
        mask_te = rng.rand(*Xc_te.shape) < 0.10
        df_miss_tr = df_tr.copy()
        df_miss_te = df_te.copy()
        df_miss_tr[mask_tr] = np.nan
        df_miss_te[mask_te] = np.nan

        model = self._build_model(
            RandomForestClassifier, "rfgap", n_estimators=50,
            random_state=14, bootstrap=True,
        )
        X_init = df_miss_tr.fillna(df_miss_tr.median())
        self._fit(model, X_init.values, yc_tr)

        imp_tr, imp_te = model.impute(df_miss_tr, X_test=df_miss_te)

        # Returns DataFrames
        self.assertIsInstance(imp_tr, pd.DataFrame)
        self.assertIsInstance(imp_te, pd.DataFrame)

        # Index and columns preserved
        pd.testing.assert_index_equal(imp_tr.index, idx_tr)
        pd.testing.assert_index_equal(imp_te.index, idx_te)
        self.assertListEqual(list(imp_tr.columns), cols)
        self.assertListEqual(list(imp_te.columns), cols)

        # No NaN remaining
        self.assertFalse(imp_tr.isna().any().any())
        self.assertFalse(imp_te.isna().any().any())

        # Observed entries unchanged
        np.testing.assert_array_equal(
            imp_tr.values[~mask_tr], Xc_tr[~mask_tr],
        )

    def test_impute_dataframe_train_only(self):
        """impute returns a single DataFrame (not tuple) when X_test is omitted."""
        import pandas as pd

        rng = np.random.RandomState(55)
        Xc_tr, yc_tr, _, _ = self._classification_data()

        cols = [f"f{i}" for i in range(Xc_tr.shape[1])]
        df = pd.DataFrame(Xc_tr, columns=cols)
        mask = rng.rand(*Xc_tr.shape) < 0.10
        df_miss = df.copy()
        df_miss[mask] = np.nan

        model = self._build_model(
            RandomForestClassifier, "rfgap", n_estimators=50,
            random_state=14, bootstrap=True,
        )
        self._fit(model, df_miss.fillna(df_miss.median()).values, yc_tr)

        result = model.impute(df_miss)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(list(result.columns), cols)
        self.assertFalse(result.isna().any().any())

    def test_outlier_scores_dataframe(self):
        """outlier_scores accepts a pandas Series and returns a Series with preserved index."""
        import pandas as pd

        Xc_tr, yc_tr, _, _ = self._classification_data()
        idx = pd.RangeIndex(500, 500 + len(yc_tr))
        y_series = pd.Series(yc_tr, index=idx, name="label")

        model = self._build_model(
            RandomForestClassifier, "rfgap", n_estimators=50,
            random_state=16, bootstrap=True,
        )
        self._fit(model, Xc_tr, yc_tr)

        scores = model.outlier_scores(y_series)

        self.assertIsInstance(scores, pd.Series)
        pd.testing.assert_index_equal(scores.index, idx)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreaterEqual(scores.min(), 0.0)

    def test_fit_predict_dataframe(self):
        """fit / predict / get_proximities work when X and y are DataFrames / Series."""
        import pandas as pd

        Xc_tr, yc_tr, Xc_te, yc_te = self._classification_data()
        cols = [f"feat_{i}" for i in range(Xc_tr.shape[1])]

        df_tr = pd.DataFrame(Xc_tr, columns=cols)
        df_te = pd.DataFrame(Xc_te, columns=cols)
        s_tr = pd.Series(yc_tr, name="target")

        for method in self.PROX_METHODS:
            with self.subTest(prox_method=method):
                model = self._build_model(
                    RandomForestClassifier, method, n_estimators=20,
                    random_state=1, bootstrap=True,
                )
                self._fit(model, df_tr, s_tr)
                preds = model.predict(df_te)
                prox = model.get_proximities()
                prox_ext = model.prox_extend(df_te)

                self.assertEqual(preds.shape[0], Xc_te.shape[0])
                self.assertEqual(prox.shape, (Xc_tr.shape[0], Xc_tr.shape[0]))
                self.assertEqual(prox_ext.shape, (Xc_te.shape[0], Xc_tr.shape[0]))

    def test_prox_predict_dataframe(self):
        """prox_predict(DataFrame) reproduces forest predictions when fitted with DataFrames."""
        import pandas as pd

        Xc_tr, yc_tr, Xc_te, _ = self._classification_data()
        Xr_tr, yr_tr, Xr_te, _ = self._regression_data()

        cols_c = [f"f{i}" for i in range(Xc_tr.shape[1])]
        cols_r = [f"f{i}" for i in range(Xr_tr.shape[1])]

        # Classification
        df_tr = pd.DataFrame(Xc_tr, columns=cols_c)
        df_te = pd.DataFrame(Xc_te, columns=cols_c)
        model = self._build_model(
            RandomForestClassifier, "rfgap", n_estimators=20,
            random_state=10, bootstrap=True,
        )
        self._fit(model, df_tr, yc_tr)

        prox_prob = model.prox_predict(df_te)
        forest_prob = np.asarray(model.predict_proba(df_te), dtype=np.float64)
        np.testing.assert_allclose(prox_prob, forest_prob, rtol=1e-4, atol=1e-4)

        # Regression
        df_tr_r = pd.DataFrame(Xr_tr, columns=cols_r)
        df_te_r = pd.DataFrame(Xr_te, columns=cols_r)
        model_r = self._build_model(
            RandomForestRegressor, "rfgap", n_estimators=20,
            random_state=11, bootstrap=True,
        )
        self._fit(model_r, df_tr_r, yr_tr)

        prox_pred = model_r.prox_predict(df_te_r)
        forest_pred = np.asarray(model_r.predict(df_te_r), dtype=np.float64)
        np.testing.assert_allclose(prox_pred, forest_pred, rtol=1e-4, atol=1e-4)

    def test_prox_predict_oob_dataframe(self):
        """prox_predict() OOB matches oob_decision_function_ when fitted with DataFrames."""
        import pandas as pd

        Xc_tr, yc_tr, _, _ = self._classification_data()
        cols = [f"f{i}" for i in range(Xc_tr.shape[1])]
        df_tr = pd.DataFrame(Xc_tr, columns=cols)

        model = self._build_model(
            RandomForestClassifier, "rfgap", n_estimators=20,
            random_state=12, oob_score=True, bootstrap=True,
        )
        self._fit(model, df_tr, yc_tr)

        prox_prob = model.prox_predict()
        oob_prob = np.asarray(model.oob_decision_function_, dtype=np.float64)
        np.testing.assert_allclose(prox_prob, oob_prob, rtol=1e-6, atol=1e-6)

    def test_extend_matches_predictions_dataframe(self):
        """Extended rfgap proximities reproduce predictions when using DataFrames."""
        import pandas as pd

        Xc_tr, yc_tr, Xc_te, _ = self._classification_data()
        cols = [f"f{i}" for i in range(Xc_tr.shape[1])]
        df_tr = pd.DataFrame(Xc_tr, columns=cols)
        df_te = pd.DataFrame(Xc_te, columns=cols)

        model = self._build_model(
            RandomForestClassifier, "rfgap", n_estimators=20,
            random_state=8, bootstrap=True,
        )
        self._fit(model, df_tr, yc_tr)

        prox_ext = self._to_dense(model.prox_extend(df_te))
        classes = np.asarray(model.classes_)
        y_idx = np.searchsorted(classes, yc_tr)
        y_hot = np.eye(classes.shape[0], dtype=np.float64)[y_idx]

        prox_prob = prox_ext @ y_hot
        forest_prob = np.asarray(model.predict_proba(df_te), dtype=np.float64)
        np.testing.assert_allclose(prox_prob, forest_prob, rtol=1e-4, atol=1e-4)

    def test_oob_proximity_matches_dataframe(self):
        """OOB rfgap proximities match oob_decision_function_ when fitted with DataFrames."""
        import pandas as pd

        Xc_tr, yc_tr, _, _ = self._classification_data()
        cols = [f"f{i}" for i in range(Xc_tr.shape[1])]
        df_tr = pd.DataFrame(Xc_tr, columns=cols)

        model = self._build_model(
            RandomForestClassifier, "rfgap", n_estimators=20,
            random_state=4, oob_score=True, bootstrap=True,
        )
        self._fit(model, df_tr, yc_tr)

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

    def test_local_feature_importance_identifies_binary_signal(self):
        X_tr, y_tr, X_te, y_te = self._binary_signal_data()

        model = self._build_model(
            RandomForestClassifier,
            "original",
            n_estimators=80,
            random_state=19,
            bootstrap=False,
            max_features=None,
        )
        self._fit(model, X_tr, y_tr)

        importance = np.asarray(
            model.local_feature_importance(
                X_te[:12], labels=y_te[:12], perturbation="flip",
            ),
            dtype=np.float64,
        )

        self.assertEqual(importance.shape, (12, X_te.shape[1]))
        self.assertGreater(
            np.abs(importance[:, 0]).mean(),
            np.abs(importance[:, 1]).mean(),
        )
        self.assertGreater(
            np.abs(importance[:, 0]).mean(),
            np.abs(importance[:, 2]).mean(),
        )

    def test_local_feature_importance_respects_focal_class_choice(self):
        X_tr, y_tr, X_te, y_te = self._binary_signal_data()

        model = self._build_model(
            RandomForestClassifier,
            "original",
            n_estimators=80,
            random_state=20,
            bootstrap=False,
            max_features=None,
        )
        self._fit(model, X_tr, y_tr)

        correct = np.asarray(
            model.local_feature_importance(
                X_te[:12], labels=y_te[:12], perturbation="flip",
            ),
            dtype=np.float64,
        )
        alternate = np.asarray(
            model.local_variable_importance(
                X_te[:12], labels=1 - y_te[:12], perturbation="flip",
            ),
            dtype=np.float64,
        )

        np.testing.assert_allclose(alternate, -correct, atol=1e-12)

    def test_local_feature_importance_dataframe_and_closeness(self):
        """local_feature_importance preserves DataFrame structure and optional closeness output."""
        import pandas as pd

        X_tr, y_tr, X_te, y_te = self._binary_signal_data()
        columns = ["signal", "noise_1", "noise_2"]
        index = pd.RangeIndex(700, 712)

        df_tr = pd.DataFrame(X_tr, columns=columns)
        df_te = pd.DataFrame(X_te[:12], columns=columns, index=index)

        model = self._build_model(
            RandomForestClassifier,
            "original",
            n_estimators=80,
            random_state=21,
            bootstrap=False,
            max_features=None,
        )
        self._fit(model, df_tr, y_tr)

        importance, closeness = model.local_feature_importance(
            df_te,
            labels=y_te[:12],
            perturbation="flip",
            return_closeness=True,
        )

        self.assertIsInstance(importance, pd.DataFrame)
        pd.testing.assert_index_equal(importance.index, index)
        self.assertListEqual(list(importance.columns), columns)
        self.assertSetEqual(set(closeness.keys()), {"in_group", "out_group"})

        for value in closeness.values():
            self.assertIsInstance(value, pd.DataFrame)
            pd.testing.assert_index_equal(value.index, index)
            self.assertListEqual(list(value.columns), columns)
            self.assertTrue(np.isfinite(value.to_numpy()).all())

    def test_local_feature_importance_quantile_mode_is_deterministic(self):
        """quantile perturbation uses a deterministic quantile grid, not RNG draws."""
        import pandas as pd

        Xc_tr, yc_tr, Xc_te, _ = self._classification_data()
        columns = [f"feature_{i}" for i in range(Xc_tr.shape[1])]

        df_tr = pd.DataFrame(Xc_tr, columns=columns)
        df_te = pd.DataFrame(Xc_te[:8], columns=columns)

        model = self._build_model(
            RandomForestClassifier,
            "original",
            n_estimators=60,
            random_state=22,
            bootstrap=False,
            max_features=None,
        )
        self._fit(model, df_tr, yc_tr)

        quant_a = model.local_feature_importance(
            df_te,
            perturbation="quantile",
            n_repeats=5,
            random_state=0,
        )
        quant_b = model.local_feature_importance(
            df_te,
            perturbation="quantile",
            n_repeats=5,
            random_state=999,
        )
        quant_median = model.local_feature_importance(
            df_te,
            perturbation="quantile",
            n_repeats=1,
        )

        self.assertIsInstance(quant_a, pd.DataFrame)
        self.assertEqual(quant_a.shape, (8, Xc_tr.shape[1]))
        self.assertTrue(np.isfinite(quant_a.to_numpy()).all())
        self.assertTrue(np.isfinite(quant_median.to_numpy()).all())
        pd.testing.assert_frame_equal(quant_a, quant_b)

    def test_local_feature_importance_auto_uses_quantiles_for_numeric_columns(self):
        """auto perturbation should route non-binary numeric columns to quantile perturbations."""
        import pandas as pd

        Xc_tr, yc_tr, Xc_te, _ = self._classification_data()
        columns = [f"feature_{i}" for i in range(Xc_tr.shape[1])]

        df_tr = pd.DataFrame(Xc_tr, columns=columns)
        df_te = pd.DataFrame(Xc_te[:8], columns=columns)

        model = self._build_model(
            RandomForestClassifier,
            "original",
            n_estimators=60,
            random_state=24,
            bootstrap=False,
            max_features=None,
        )
        self._fit(model, df_tr, yc_tr)

        auto_importance = model.local_feature_importance(
            df_te,
            perturbation="auto",
            n_repeats=5,
            random_state=999,
        )
        quant_importance = model.local_feature_importance(
            df_te,
            perturbation="quantile",
            n_repeats=5,
            random_state=0,
        )

        self.assertIsInstance(auto_importance, pd.DataFrame)
        pd.testing.assert_frame_equal(auto_importance, quant_importance)

    def test_local_perturbation_mode_auto_uses_permute_for_non_numeric_columns(self):
        """auto perturbation should not send non-numeric categorical columns through quantiles."""
        query_col = np.array(["red", "blue", "green"], dtype=object)
        ref_col = np.array(["red", "green", "blue", "green"], dtype=object)

        mode = with_sparse_proximity(RandomForestClassifier)._resolve_local_perturbation_mode(
            query_col,
            ref_col,
            "auto",
        )
        self.assertEqual(mode, "permute")

    def test_local_feature_importance_directionality_tracks_quantile_direction(self):
        """Directional summaries split quantile perturbations into upward and downward moves."""
        import pandas as pd

        Xc_tr, yc_tr, Xc_te, _ = self._classification_data()
        columns = [f"feature_{i}" for i in range(Xc_tr.shape[1])]
        index = pd.RangeIndex(900, 908)

        df_tr = pd.DataFrame(Xc_tr, columns=columns)
        df_te = pd.DataFrame(Xc_te[:8], columns=columns, index=index)

        model = self._build_model(
            RandomForestClassifier,
            "original",
            n_estimators=60,
            random_state=23,
            bootstrap=False,
            max_features=None,
        )
        self._fit(model, df_tr, yc_tr)

        importance, closeness, directionality = model.local_feature_importance(
            df_te,
            perturbation="quantile",
            n_repeats=5,
            return_closeness=True,
            return_directionality=True,
        )

        self.assertIsInstance(importance, pd.DataFrame)
        self.assertSetEqual(set(closeness.keys()), {"in_group", "out_group"})
        self.assertSetEqual(
            set(directionality.keys()),
            {"upward", "downward", "upward_count", "downward_count"},
        )

        upward = directionality["upward"]
        downward = directionality["downward"]
        upward_count = directionality["upward_count"]
        downward_count = directionality["downward_count"]

        for value in (
            upward,
            downward,
            upward_count,
            downward_count,
            closeness["in_group"],
            closeness["out_group"],
        ):
            self.assertIsInstance(value, pd.DataFrame)
            pd.testing.assert_index_equal(value.index, index)
            self.assertListEqual(list(value.columns), columns)

        quantile_levels = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        expected_upward = np.zeros((df_te.shape[0], df_te.shape[1]), dtype=np.int64)
        expected_downward = np.zeros_like(expected_upward)

        for feature_idx, column in enumerate(columns):
            ref_values = df_tr[column].to_numpy(dtype=np.float64, copy=False)
            quantile_values = np.quantile(ref_values, quantile_levels)
            observed = df_te[column].to_numpy(dtype=np.float64, copy=False)
            expected_upward[:, feature_idx] = (
                quantile_values[np.newaxis, :] > observed[:, np.newaxis]
            ).sum(axis=1)
            expected_downward[:, feature_idx] = (
                quantile_values[np.newaxis, :] < observed[:, np.newaxis]
            ).sum(axis=1)

        expected_upward_df = pd.DataFrame(expected_upward, index=index, columns=columns)
        expected_downward_df = pd.DataFrame(expected_downward, index=index, columns=columns)

        pd.testing.assert_frame_equal(upward_count, expected_upward_df)
        pd.testing.assert_frame_equal(downward_count, expected_downward_df)

        upward_values = upward.to_numpy(dtype=np.float64)
        downward_values = downward.to_numpy(dtype=np.float64)
        self.assertTrue(np.isnan(upward_values[expected_upward == 0]).all())
        self.assertTrue(np.isnan(downward_values[expected_downward == 0]).all())
        self.assertTrue(np.isfinite(upward_values[expected_upward > 0]).all())
        self.assertTrue(np.isfinite(downward_values[expected_downward > 0]).all())

        reconstructed = (
            np.nan_to_num(upward_values, nan=0.0) * expected_upward
            + np.nan_to_num(downward_values, nan=0.0) * expected_downward
        ) / 5.0
        np.testing.assert_allclose(
            reconstructed,
            importance.to_numpy(dtype=np.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_get_prototypes_returns_disjoint_classwise_summaries(self):
        import pandas as pd

        X_tr, y_tr = self._prototype_data()

        model = self._build_model(
            RandomForestClassifier,
            "original",
            n_estimators=80,
            random_state=31,
            bootstrap=False,
            max_features=None,
        )
        self._fit(model, X_tr, y_tr)

        prototypes = model.get_prototypes(
            k=4,
            n_prototypes=2,
            categorical_features=["texture_code"],
        )

        self.assertSetEqual(set(prototypes.keys()), {0, 1})
        for cls, records in prototypes.items():
            self.assertEqual(len(records), 2)

            first_neighbors = set(records[0]["neighbor_positions"].tolist())
            second_neighbors = set(records[1]["neighbor_positions"].tolist())
            self.assertTrue(first_neighbors.isdisjoint(second_neighbors))

            x2_medians = sorted(
                float(record["continuous_summary"].loc["x2", "median"])
                for record in records
            )
            self.assertEqual(x2_medians, [0.0, 4.0])

            texture_modes = sorted(
                int(record["categorical_summary"].loc["texture_code", "mode"])
                for record in records
            )
            self.assertEqual(texture_modes, [0, 1])

            for record in records:
                self.assertIsInstance(record["prototype"], pd.Series)
                self.assertIsInstance(record["neighbor_index"], pd.Index)
                self.assertEqual(record["neighbor_index"].shape[0], 4)
                self.assertEqual(record["continuous_summary"].index.name, "feature")
                self.assertEqual(record["categorical_summary"].index.name, "feature")
                self.assertListEqual(
                    list(record["continuous_summary"].columns),
                    ["q25", "median", "q75"],
                )
                self.assertListEqual(
                    list(record["categorical_summary"].columns),
                    ["mode", "mode_count", "mode_frequency"],
                )
                self.assertListEqual(
                    list(record["categorical_frequencies"].columns),
                    ["count", "frequency"],
                )
                self.assertEqual(record["categorical_frequencies"].index.names, ["feature", "category"])
                self.assertGreater(record["same_class_neighbor_count"], 0)
                self.assertLessEqual(record["same_class_neighbor_count"], 4)
                self.assertTrue(np.isfinite(record["neighbor_proximities"]).all())
                self.assertEqual(record["prototype"].index.name, "feature")
                self.assertEqual(record["prototype"]["x1"], float(cls * 5))

    def test_get_prototypes_standardizes_continuous_summaries(self):
        X_tr, y_tr = self._prototype_data()

        model = self._build_model(
            RandomForestClassifier,
            "original",
            n_estimators=80,
            random_state=32,
            bootstrap=False,
            max_features=None,
        )
        self._fit(model, X_tr, y_tr)

        prototypes = model.get_prototypes(
            k=4,
            n_prototypes=2,
            categorical_features=["texture_code"],
            standardize_continuous=True,
        )

        x2_medians = sorted(
            float(record["continuous_summary"].loc["x2", "median"])
            for record in prototypes[0]
        )
        np.testing.assert_allclose(x2_medians, [0.0, 1.0], atol=1e-12)

        x1_medians = sorted(
            float(record["continuous_summary"].loc["x1", "median"])
            for record in prototypes[0] + prototypes[1]
        )
        np.testing.assert_allclose(x1_medians, [0.0, 0.0, 1.0, 1.0], atol=1e-12)

    def test_get_prototypes_rejects_regressors(self):
        Xr_tr, yr_tr, _, _ = self._regression_data()

        model = self._build_model(
            RandomForestRegressor,
            "original",
            n_estimators=40,
            random_state=33,
            bootstrap=False,
            max_features=None,
        )
        self._fit(model, Xr_tr, yr_tr)

        with self.assertRaisesRegex(ValueError, "only defined for classifiers"):
            model.get_prototypes()


if __name__ == "__main__":
    unittest.main()
