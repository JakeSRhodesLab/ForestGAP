"""Optional scikit-survival native output adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.utils import _safe_indexing


def native_survival_outputs(estimator: Any, X: Any) -> dict[str, Any]:
    """Return risk plus available survival and cumulative-hazard functions."""

    output: dict[str, Any] = {"risk": np.asarray(estimator.predict(X))}
    survival = getattr(estimator, "predict_survival_function", None)
    hazard = getattr(estimator, "predict_cumulative_hazard_function", None)
    if callable(survival):
        output["survival"] = survival(X)
    if callable(hazard):
        output["cumulative_hazard"] = hazard(X)
    return output


def censoring_safe_survival_metrics(
    estimator: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    *,
    times: Any | None = None,
) -> dict[str, Any]:
    """Evaluate risk and survival curves with IPCW censoring correction.

    The censoring distribution is estimated from ``y_train`` only.  Returned
    metrics are Uno's IPCW concordance and integrated Brier score, avoiding
    the leakage and optimism caused by treating censored cases as events.
    """

    try:
        from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "censoring-safe evaluation requires the 'survival' extra"
        ) from exc

    train = np.asarray(y_train)
    test = np.asarray(y_test)
    if train.dtype.names is None or test.dtype.names is None:
        raise TypeError("y_train and y_test must be structured survival arrays")
    event_name, time_name = test.dtype.names[:2]
    train_time_name = train.dtype.names[1]
    train_times = np.asarray(train[train_time_name], dtype=float)
    test_times = np.asarray(test[time_name], dtype=float)
    # IPCW is identifiable only inside the training censoring support.  Apply
    # this predeclared support restriction instead of extrapolating censoring
    # weights for longer-followed test cases.
    supported = test_times < float(np.max(train_times))
    n_excluded = int(np.count_nonzero(~supported))
    if np.count_nonzero(supported) < 2:
        raise ValueError("fewer than two test cases lie inside training follow-up support")
    test = test[supported]
    test_times = test_times[supported]
    X_supported = _safe_indexing(X_test, supported)
    risk = np.asarray(estimator.predict(X_supported), dtype=float)

    upper = min(float(np.max(train_times)), float(np.max(test_times)))
    lower = max(float(np.min(train_times)), float(np.min(test_times)))
    if not lower < upper:
        raise ValueError("train and test follow-up ranges do not overlap")
    tau = float(np.nextafter(upper, lower))
    concordance = float(concordance_index_ipcw(train, test, risk, tau=tau)[0])

    if times is None:
        evaluation_times = np.linspace(lower, upper, 25, endpoint=False)[1:]
    else:
        evaluation_times = np.asarray(times, dtype=float)
    evaluation_times = evaluation_times[
        (evaluation_times > lower) & (evaluation_times < upper)
    ]
    if evaluation_times.size < 2:
        raise ValueError("at least two evaluation times must lie inside follow-up support")

    functions = estimator.predict_survival_function(X_supported)
    survival = np.asarray(
        [[float(function(time)) for time in evaluation_times] for function in functions]
    )
    brier = float(integrated_brier_score(train, test, survival, evaluation_times))
    return {
        "ipcw_concordance": concordance,
        "integrated_brier_score": brier,
        "evaluation_times": evaluation_times,
        "n_events": int(np.count_nonzero(test[event_name])),
        "n_censored": int(len(test) - np.count_nonzero(test[event_name])),
        "n_excluded_outside_train_support": n_excluded,
    }


__all__ = ["censoring_safe_survival_metrics", "native_survival_outputs"]
