import numpy as np
import pytest
from scipy import sparse

from forestgap.manifold import is_symmetric, prepare_proximity_affinity, proximity_to_distance


def test_asymmetric_affinity_is_explicitly_symmetrized():
    matrix = np.array([[1.0, 0.2], [0.8, 1.0]])
    affinity = prepare_proximity_affinity(matrix, symmetrize="auto", diagonal="one")
    assert is_symmetric(affinity)
    assert affinity[0, 1] == pytest.approx(0.5)


def test_implicit_sparse_zero_affinity_is_maximal_dense_distance():
    affinity = sparse.csr_matrix([[1.0, 0.25, 0.0], [0.25, 1.0, 0.81], [0.0, 0.81, 1.0]])
    distance = proximity_to_distance(affinity, dense=True)
    assert distance[0, 2] == pytest.approx(1.0)
    assert distance[0, 1] == pytest.approx(np.sqrt(0.75))
    np.testing.assert_allclose(np.diag(distance), 0.0)
