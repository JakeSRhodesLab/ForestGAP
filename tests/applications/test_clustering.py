import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score

from forestgap import Proximity, proximity_clustering


@pytest.fixture(scope="module")
def clustered_proximity():
    X, y = make_blobs(
        n_samples=90,
        centers=3,
        cluster_std=0.55,
        random_state=40,
    )
    proximity = Proximity(
        RandomForestClassifier(n_estimators=100, random_state=41), relation="original",
    ).fit(X, y)
    return X, y, proximity


@pytest.mark.parametrize("method", ["spectral", "kmeans"])
def test_affinity_clustering_recovers_separated_groups(clustered_proximity, method):
    _, y, proximity = clustered_proximity
    labels = proximity_clustering(proximity, n_clusters=3, method=method, random_state=42)
    assert labels.shape == y.shape
    assert adjusted_rand_score(y, labels) > 0.9


def test_hdbscan_clusters_from_proximity_distance(clustered_proximity):
    _, y, proximity = clustered_proximity
    labels = proximity_clustering(proximity, method="hdbscan", min_cluster_size=8)
    assert labels.shape == y.shape
    assert np.issubdtype(labels.dtype, np.integer)
    assert set(np.unique(labels)).issubset(set(range(len(y))) | {-1})


def test_directed_proximities_are_symmetrized_before_clustering(clustered_proximity):
    X, y, _ = clustered_proximity
    directed = Proximity(
        RandomForestClassifier(
            n_estimators=100,
            bootstrap=True,
            random_state=43,
        ),
        relation="rfgap",
    ).fit(X, y)
    labels = proximity_clustering(directed, n_clusters=3, method="spectral", random_state=44)
    assert labels.shape == y.shape


@pytest.mark.parametrize("method", ["spectral", "kmeans", "agglomerative"])
def test_cluster_count_is_required_when_the_estimator_needs_it(clustered_proximity, method):
    _, _, proximity = clustered_proximity
    with pytest.raises(ValueError, match="n_clusters is required"):
        proximity_clustering(proximity, method=method)


def test_unknown_clustering_method_is_rejected(clustered_proximity):
    _, _, proximity = clustered_proximity
    with pytest.raises(ValueError, match="method must be one of"):
        proximity_clustering(proximity, n_clusters=3, method="unknown")
