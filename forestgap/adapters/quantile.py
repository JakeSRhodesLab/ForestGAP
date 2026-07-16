"""Optional quantile-forest native API adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse


def native_quantiles(estimator: Any, X: Any, quantiles: Any) -> np.ndarray:
    """Evaluate the dependency's native conditional quantiles."""

    return np.asarray(estimator.predict(X, quantiles=quantiles))


def proximity_counts(estimator: Any, X: Any, n_train: int) -> sparse.csr_matrix:
    """Convert quantile-forest ``proximity_counts`` output to CSR."""

    rows, cols, data = [], [], []
    counts = estimator.proximity_counts(X)
    for row, neighbors in enumerate(counts):
        for col, count in neighbors:
            rows.append(row)
            cols.append(int(col))
            data.append(float(count))
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(counts), n_train))
