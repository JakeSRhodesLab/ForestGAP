import inspect
import unittest
import warnings

import numpy as np
from scipy import sparse
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from forestgap import with_sparse_proximity

try:
    from treeple import HonestForestClassifier

    _HAS_TREEPLE = True
except Exception:
    HonestForestClassifier = None
    _HAS_TREEPLE = False


def _to_dense(mat):
    return mat.toarray() if sparse.isspmatrix(mat) else np.asarray(mat)


def _one_hot(y, classes):
    mapping = {cls: i for i, cls in enumerate(np.asarray(classes).tolist())}
    y_idx = np.array([mapping[value] for value in y], dtype=np.int64)
    return sparse.csc_matrix(
        (np.ones(y_idx.shape[0], dtype=np.float64), (np.arange(y_idx.shape[0]), y_idx)),
        shape=(y_idx.shape[0], len(classes)),
    )


def _finite_rows(*arrays):
    mask = None
    for arr in arrays:
        arr = np.asarray(arr, dtype=np.float64)
        finite = np.isfinite(arr) if arr.ndim == 1 else np.all(np.isfinite(arr), axis=1)
        mask = finite if mask is None else (mask & finite)
    return np.asarray(mask, dtype=bool)


class TestHonestGapValidation(unittest.TestCase):
    def test_honest_gap_rejects_non_honest_forest(self):
        X, y = make_classification(
            n_samples=80,
            n_features=6,
            n_informative=4,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=101,
        )
        ProxRF = with_sparse_proximity(RandomForestClassifier)
        model = ProxRF(
            n_estimators=8,
            random_state=102,
            bootstrap=True,
            prox_method="honest_gap",
        )

        with self.assertRaisesRegex(ValueError, "honest_gap.*HonestForestClassifier"):
            model.fit(X, y)


@unittest.skipUnless(_HAS_TREEPLE, "treeple is required for HonestForestClassifier tests")
class TestHonestGapPredictionIdentity(unittest.TestCase):
    @staticmethod
    def _data():
        X, y = make_classification(
            n_samples=260,
            n_features=10,
            n_informative=7,
            n_redundant=1,
            n_classes=3,
            n_clusters_per_class=1,
            class_sep=1.4,
            random_state=201,
        )
        return train_test_split(X, y, test_size=0.25, stratify=y, random_state=202)

    def _model(self, prox_method="honest_gap", kernel_method=True, stratify=None, random_state=203):
        if HonestForestClassifier is None:
            self.skipTest("treeple is required for HonestForestClassifier tests")

        sig = inspect.signature(HonestForestClassifier.__init__)
        kwargs = {
            "n_estimators": 80,
            "random_state": random_state,
            "prox_method": prox_method,
            "matrix_type": "sparse",
        }
        for name, value in (
            ("bootstrap", True),
            ("oob_score", True),
            ("honest_fraction", 0.5),
            ("max_depth", 6),
            ("min_samples_leaf", 2),
        ):
            if name in sig.parameters:
                kwargs[name] = value
        if kernel_method is not None:
            if "kernel_method" not in sig.parameters:
                self.skipTest("Installed treeple lacks kernel_method.")
            kwargs["kernel_method"] = kernel_method
        if stratify is not None:
            if "stratify" not in sig.parameters:
                self.skipTest("Installed treeple lacks stratify.")
            kwargs["stratify"] = stratify

        ProxHonest = with_sparse_proximity(HonestForestClassifier)
        return ProxHonest(**kwargs)

    @staticmethod
    def _fit(model, X, y):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Some inputs do not have OOB scores.*")
            warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
            model.fit(X, y)

    def _assert_test_prediction_identity(self, *, kernel_method, stratify, random_state):
        X_train, X_test, y_train, _ = self._data()
        model = self._model(
            prox_method="honest_gap",
            kernel_method=kernel_method,
            stratify=stratify,
            random_state=random_state,
        )
        self._fit(model, X_train, y_train)

        prox_prob = np.asarray(model.prox_predict(X_test), dtype=np.float64)
        forest_prob = np.asarray(model.predict_proba(X_test), dtype=np.float64)
        finite = _finite_rows(prox_prob, forest_prob)
        self.assertTrue(np.any(finite))
        np.testing.assert_allclose(
            prox_prob[finite],
            forest_prob[finite],
            rtol=1e-6,
            atol=1e-6,
        )

        prox_ext = _to_dense(model.prox_extend(X_test))
        y_hot = _one_hot(y_train, model.classes_)
        reconstructed = np.asarray(prox_ext @ y_hot, dtype=np.float64)
        finite = _finite_rows(reconstructed, forest_prob)
        self.assertTrue(np.any(finite))
        np.testing.assert_allclose(
            reconstructed[finite],
            forest_prob[finite],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_rfgap_warns_for_honest_forest_and_still_fits(self):
        X_train, X_test, y_train, _ = self._data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = self._model(prox_method="rfgap", kernel_method=True, random_state=204)
            self._fit(model, X_train, y_train)

        warning_text = "\n".join(str(w.message) for w in caught)
        self.assertIn("honest_gap", warning_text)
        prox_prob = model.prox_predict(X_test)
        self.assertEqual(prox_prob.shape, model.predict_proba(X_test).shape)

    def test_honest_gap_test_predictions_match_default_kernel_method(self):
        self._assert_test_prediction_identity(
            kernel_method=True,
            stratify=None,
            random_state=205,
        )

    def test_honest_gap_test_predictions_match_kernel_method_false(self):
        self._assert_test_prediction_identity(
            kernel_method=False,
            stratify=None,
            random_state=206,
        )

    def test_honest_gap_test_predictions_match_stratified_honest_forest(self):
        self._assert_test_prediction_identity(
            kernel_method=True,
            stratify=True,
            random_state=207,
        )

    def test_honest_gap_oob_predictions_match_native_oob_output(self):
        X_train, _, y_train, _ = self._data()
        model = self._model(
            prox_method="honest_gap",
            kernel_method=True,
            stratify=True,
            random_state=208,
        )
        self._fit(model, X_train, y_train)

        prox_oob = np.asarray(model.prox_predict(), dtype=np.float64)
        native_oob = np.asarray(model.oob_decision_function_, dtype=np.float64)
        finite = _finite_rows(prox_oob, native_oob)
        self.assertTrue(np.any(finite))
        np.testing.assert_allclose(
            prox_oob[finite],
            native_oob[finite],
            rtol=1e-6,
            atol=1e-6,
        )

        row_sums = native_oob[finite].sum(axis=1)
        raw_like = np.nanmax(native_oob[finite]) > 1.0 or not np.allclose(row_sums, 1.0, atol=1e-6)
        if raw_like:
            self.assertFalse(np.allclose(prox_oob[finite].sum(axis=1), 1.0, atol=1e-6))

    def test_honest_gap_oob_predictions_match_kernel_method_false_probabilities(self):
        X_train, _, y_train, _ = self._data()
        model = self._model(
            prox_method="honest_gap",
            kernel_method=False,
            stratify=None,
            random_state=209,
        )
        self._fit(model, X_train, y_train)

        prox_oob = np.asarray(model.prox_predict(), dtype=np.float64)
        native_oob = np.asarray(model.oob_decision_function_, dtype=np.float64)
        finite = _finite_rows(prox_oob, native_oob)
        self.assertTrue(np.any(finite))
        np.testing.assert_allclose(
            prox_oob[finite],
            native_oob[finite],
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
