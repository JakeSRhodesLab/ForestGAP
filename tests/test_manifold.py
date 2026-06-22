import unittest

import numpy as np
from scipy import sparse
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from forestgap import with_sparse_proximity
from forestgap.manifold import (
    is_symmetric,
    prepare_proximity_affinity,
    proximity_to_distance,
)


class TestProximityManifoldUtilities(unittest.TestCase):
    def test_prepare_affinity_auto_symmetrizes_asymmetric_matrix(self):
        P = np.array(
            [
                [1.0, 0.2, 0.0],
                [0.8, 1.0, 0.4],
                [0.0, 0.1, 1.0],
            ]
        )

        A = prepare_proximity_affinity(P, symmetrize="auto", diagonal="one")

        self.assertTrue(is_symmetric(A))
        np.testing.assert_allclose(A[0, 1], 0.5)
        np.testing.assert_allclose(A[1, 2], 0.25)
        np.testing.assert_allclose(np.diag(A), np.ones(3))

    def test_sparse_zero_proximities_become_max_distance_when_dense(self):
        P = sparse.csr_matrix(
            np.array(
                [
                    [1.0, 0.25, 0.0],
                    [0.0, 1.0, 0.81],
                    [0.16, 0.0, 1.0],
                ]
            )
        )

        D = proximity_to_distance(P, dense=True)

        self.assertIsInstance(D, np.ndarray)
        np.testing.assert_allclose(np.diag(D), np.zeros(3))
        np.testing.assert_allclose(D[0, 1], np.sqrt(0.75))
        np.testing.assert_allclose(D[1, 2], np.sqrt(0.19))
        np.testing.assert_allclose(D[0, 2], 1.0)

    def test_sparse_distance_requires_dense_or_graph_mode(self):
        P = sparse.eye(4, format="csr")

        with self.assertRaisesRegex(ValueError, "implicit zero proximities"):
            proximity_to_distance(P, dense=False)

        D_graph = proximity_to_distance(P, dense=False, sparse_mode="graph")
        self.assertTrue(sparse.isspmatrix_csr(D_graph))
        self.assertEqual(D_graph.nnz, 0)


class TestForestManifoldEmbeddings(unittest.TestCase):
    @staticmethod
    def _data():
        X, y = make_classification(
            n_samples=54,
            n_features=8,
            n_informative=5,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=11,
        )
        return X[:42], y[:42], X[42:], y[42:]

    @staticmethod
    def _model(random_state=0):
        ProxRF = with_sparse_proximity(RandomForestClassifier)
        return ProxRF(
            n_estimators=25,
            random_state=random_state,
            prox_method="rfgap",
            matrix_type="sparse",
            bootstrap=True,
        )

    def test_embed_requires_fitted_forest(self):
        model = self._model()

        with self.assertRaisesRegex(RuntimeError, "Proximity structures"):
            model.embed(method="isomap")

    def test_isomap_embedding_from_fitted_forest(self):
        X_train, y_train, X_test, _ = self._data()
        model = self._model(random_state=1)
        model.fit(X_train, y_train)

        embedding = model.embed(
            method="isomap",
            n_components=2,
            n_neighbors=5,
            symm_mode="arithmetic",
        )

        self.assertEqual(embedding.shape, (X_train.shape[0], 2))
        self.assertIn("isomap", model.manifold_models_)
        self.assertIn("isomap", model.manifold_embeddings_)

        projected = model.embed_transform(X_test, method="isomap")
        self.assertEqual(projected.shape, (X_test.shape[0], 2))

    def test_tsne_embedding_from_fitted_forest(self):
        X_train, y_train, _, _ = self._data()
        model = self._model(random_state=2)
        model.fit(X_train, y_train)

        embedding = model.embed(
            method="tsne",
            n_components=2,
            perplexity=5,
            max_iter=250,
            learning_rate="auto",
            symm_mode="arithmetic",
        )

        self.assertEqual(embedding.shape, (X_train.shape[0], 2))
        self.assertIn("tsne", model.manifold_models_)

    def test_phate_embedding_from_fitted_forest_when_installed(self):
        try:
            import phate  # noqa: F401
            import graphtools  # noqa: F401
        except Exception:
            self.skipTest("PHATE optional dependencies are not installed.")

        X_train, y_train, X_test, _ = self._data()
        model = self._model(random_state=3)
        model.fit(X_train, y_train)

        embedding = model.embed(
            method="phate",
            n_components=2,
            knn=5,
            t=1,
            n_landmark=20,
            verbose=0,
        )

        self.assertEqual(embedding.shape, (X_train.shape[0], 2))
        projected = model.embed_transform(X_test, method="phate")
        self.assertEqual(projected.shape, (X_test.shape[0], 2))


if __name__ == "__main__":
    unittest.main()
