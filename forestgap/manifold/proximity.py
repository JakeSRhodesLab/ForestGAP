from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
from scipy import sparse


MatrixLike = Union[np.ndarray, sparse.spmatrix]
SymmetrizeMode = Optional[
    Literal["auto", "arithmetic", "geometric", "max", "none"]
]
DiagonalMode = Optional[Literal["one", "zero", "preserve"]]
DistanceMethod = Literal["sqrt_one_minus", "one_minus"]


def _as_matrix(P: MatrixLike) -> MatrixLike:
    if sparse.issparse(P):
        mat = P.tocsr().astype(np.float64, copy=True)
    else:
        mat = np.asarray(P, dtype=np.float64)

    if mat.ndim != 2:
        raise ValueError("Expected a 2D proximity matrix.")
    return mat


def _is_square(P: MatrixLike) -> bool:
    return P.shape[0] == P.shape[1]


def is_symmetric(P: MatrixLike, *, rtol: float = 1e-7, atol: float = 1e-10) -> bool:
    """Return whether a square dense or sparse matrix is numerically symmetric."""

    mat = _as_matrix(P)
    if not _is_square(mat):
        return False

    if sparse.issparse(mat):
        diff = (mat - mat.T).tocoo()
        if diff.nnz == 0:
            return True
        scale = np.maximum(np.abs(mat.data).max(initial=0.0), 1.0)
        return bool(np.all(np.abs(diff.data) <= (atol + rtol * scale)))

    return bool(np.allclose(mat, mat.T, rtol=rtol, atol=atol))


def symmetrize_affinity(P: MatrixLike, mode: SymmetrizeMode = "arithmetic") -> MatrixLike:
    """Symmetrize a square proximity affinity matrix."""

    mat = _as_matrix(P)
    if mode in (None, "none"):
        return mat
    if mode == "auto":
        mode = "arithmetic" if not is_symmetric(mat) else "none"
    if mode in (None, "none"):
        return mat
    if not _is_square(mat):
        raise ValueError("Only square proximity matrices can be symmetrized.")

    if sparse.issparse(mat):
        if mode == "arithmetic":
            return ((mat + mat.T) * 0.5).tocsr()
        if mode == "max":
            return mat.maximum(mat.T).tocsr()
        if mode == "geometric":
            out = mat.multiply(mat.T).tocsr()
            np.sqrt(out.data, out=out.data)
            out.eliminate_zeros()
            return out
    else:
        if mode == "arithmetic":
            return (mat + mat.T) * 0.5
        if mode == "max":
            return np.maximum(mat, mat.T)
        if mode == "geometric":
            return np.sqrt(np.maximum(mat * mat.T, 0.0))

    raise ValueError(
        "symmetrize must be one of {None, 'none', 'auto', "
        "'arithmetic', 'geometric', 'max'}."
    )


def _clip_affinity(P: MatrixLike, *, lower: float = 0.0, upper: float = 1.0) -> MatrixLike:
    if sparse.issparse(P):
        out = P.tocsr(copy=True)
        np.clip(out.data, lower, upper, out=out.data)
        out.eliminate_zeros()
        return out
    return np.clip(np.asarray(P, dtype=np.float64), lower, upper)


def _set_square_diagonal(P: MatrixLike, value: float) -> MatrixLike:
    if not _is_square(P):
        return P
    if sparse.issparse(P):
        out = P.tocsr(copy=True)
        out.setdiag(value)
        out.eliminate_zeros()
        return out
    out = np.array(P, dtype=np.float64, copy=True)
    np.fill_diagonal(out, value)
    return out


def prepare_proximity_affinity(
    P: MatrixLike,
    *,
    symmetrize: SymmetrizeMode = "auto",
    diagonal: DiagonalMode = "one",
    clip: bool = True,
) -> MatrixLike:
    """
    Prepare forest proximities for affinity-based downstream methods.

    Parameters
    ----------
    P : array-like or sparse matrix
        Proximity matrix. Square matrices represent training affinities;
        rectangular matrices usually represent new-to-training affinities.
    symmetrize : {None, 'none', 'auto', 'arithmetic', 'geometric', 'max'}
        Symmetrization mode for square matrices. ``'auto'`` symmetrizes only
        when the matrix is detectably asymmetric.
    diagonal : {'one', 'zero', 'preserve', None}
        Diagonal handling for square matrices.
    clip : bool
        Whether to clip explicit proximity values into ``[0, 1]``.
    """

    mat = _as_matrix(P)
    if clip:
        mat = _clip_affinity(mat)

    mat = symmetrize_affinity(mat, symmetrize)

    if diagonal in (None, "preserve"):
        return mat
    if diagonal == "one":
        return _set_square_diagonal(mat, 1.0)
    if diagonal == "zero":
        return _set_square_diagonal(mat, 0.0)

    raise ValueError("diagonal must be one of {None, 'preserve', 'one', 'zero'}.")


def proximity_to_distance(
    P: MatrixLike,
    *,
    method: DistanceMethod = "sqrt_one_minus",
    dense: bool = True,
    sparse_mode: Literal["raise", "graph"] = "raise",
) -> MatrixLike:
    """
    Convert proximity affinities to precomputed distances.

    For dense output, implicit zero proximity becomes maximal distance. For
    sparse output, this is impossible to represent exactly because sparse
    matrices use implicit zeros; ``sparse_mode='graph'`` therefore converts
    only explicit affinity edges and treats missing entries as absent graph
    edges for estimators that support sparse precomputed graphs.
    """

    mat = _clip_affinity(_as_matrix(P))

    if sparse.issparse(mat):
        if dense:
            arr = mat.toarray()
            dist = _dense_proximity_to_distance(arr, method)
            if _is_square(dist):
                np.fill_diagonal(dist, 0.0)
            return dist
        if sparse_mode != "graph":
            raise ValueError(
                "Sparse proximity matrices cannot exactly represent dense "
                "precomputed distances because implicit zero proximities map "
                "to nonzero distances. Use dense=True or sparse_mode='graph'."
            )
        out = mat.tocsr(copy=True)
        out.data = _distance_values_from_proximity(out.data, method)
        if _is_square(out):
            out.setdiag(0.0)
        out.eliminate_zeros()
        return out

    dist = _dense_proximity_to_distance(np.asarray(mat, dtype=np.float64), method)
    if _is_square(dist):
        np.fill_diagonal(dist, 0.0)
    return dist


def _dense_proximity_to_distance(P: np.ndarray, method: DistanceMethod) -> np.ndarray:
    return _distance_values_from_proximity(P, method)


def _distance_values_from_proximity(values: np.ndarray, method: DistanceMethod) -> np.ndarray:
    if method == "one_minus":
        return np.maximum(1.0 - values, 0.0)
    if method == "sqrt_one_minus":
        return np.sqrt(np.maximum(1.0 - values, 0.0))
    raise ValueError("method must be one of {'sqrt_one_minus', 'one_minus'}.")
