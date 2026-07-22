from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from experiments.common import REGRESSION_SUITE_ID, prepare_dataset, suite_metadata
from experiments.analysis import (
    aggregate_folds_within_dataset,
    average_ranks,
    bootstrap_confidence_interval,
    holm_adjust,
    paired_effects,
)
from experiments.harness import run_cells
from experiments.program import (
    APPLICATION_RELATION_SCOPES,
    application_relations,
    calibration_split_indices,
    capability_aware_cells,
    default_trial_manifests,
    inner_fold_indices,
    relation_use_policy,
    training_only_anchors,
)
from forestgap.datasets import OpenMLDataset, OpenMLTask, OpenMLTaskFold


def _metrics(proximity, cell):
    return {"rows": proximity.get_proximities().shape[0]}


def test_study_455_metadata_and_official_fold_preparation_preserve_indices():
    assert REGRESSION_SUITE_ID == 455
    assert suite_metadata("regression")[0] == 455
    dataset = OpenMLDataset(
        name="study-455-toy",
        dataset_id=4550,
        version=1,
        target_name="target",
        X=pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "x2": [3.0, 2.0, 1.0, 0.0]}),
        y=pd.Series([0.0, 10.0, 20.0, 30.0]),
        feature_names=["x1", "x2"],
        categorical_indicator=[False, False],
    )
    task = OpenMLTask(
        task_id=361085,
        dataset=dataset,
        target_name="target",
        repeats=1,
        folds=2,
        samples=1,
    )
    official_fold = OpenMLTaskFold(
        task=task,
        repeat=0,
        fold=1,
        sample=0,
        train_indices=np.array([1, 3]),
        test_indices=np.array([0, 2]),
    )

    prepared = prepare_dataset(
        "regression",
        task_id=361085,
        task=SimpleNamespace(),
        random_state=99,
        max_rows=None,
        test_size=0.3,
        official_fold=official_fold,
    )

    assert prepared.suite_id == 455
    assert prepared.fold_id == "r0-f1-s0"
    assert prepared.repeat == 0 and prepared.fold == 1 and prepared.sample == 0
    np.testing.assert_array_equal(prepared.train_indices, [1, 3])
    np.testing.assert_array_equal(prepared.test_indices, [0, 2])
    np.testing.assert_allclose(prepared.X_train[:, 0], [1.0, 3.0])
    np.testing.assert_allclose(prepared.X_test[:, 0], [0.0, 2.0])
    np.testing.assert_allclose(prepared.y_train, [10.0, 30.0])
    np.testing.assert_allclose(prepared.y_test, [0.0, 20.0])


def test_capability_generation_preserves_invalid_request_and_skips_deferred_factory():
    X, y = make_classification(n_samples=60, n_features=6, random_state=101)
    factories = {"calls": 0}

    def non_bootstrap_forest():
        factories["calls"] += 1
        return RandomForestClassifier(n_estimators=7, bootstrap=False, random_state=102)

    cells = capability_aware_cells(
        family="random_forest",
        estimator="rf",
        estimator_factory=non_bootstrap_forest,
        application="prediction",
        application_fn=_metrics,
        X_train=X,
        y_train=y,
        data_regime="synthetic",
        requested_relations=["original", "rfgap"],
    )
    assert len(cells) == 2
    assert cells[0].declared_status is None
    assert cells[1].declared_status == "unsupported"
    rows = run_cells(cells)
    assert [row["status"] for row in rows] == ["ok", "unsupported"]
    assert rows[1]["capability_snapshot"]["bootstrap"] is False

    deferred_calls = {"count": 0}

    def must_not_run():
        deferred_calls["count"] += 1
        raise AssertionError("deferred family factory was invoked")

    deferred = capability_aware_cells(
        family="random_survival_forest",
        estimator="survival",
        estimator_factory=must_not_run,
        application="prediction",
        application_fn=_metrics,
        X_train=X,
        y_train=y,
        data_regime="synthetic",
    )
    assert deferred[0].declared_status == "excluded"
    assert run_cells(deferred)[0]["status"] == "excluded"
    assert deferred_calls["count"] == 0


def test_trial_and_partition_helpers_are_deterministic_and_outer_safe():
    parent_trials, relation_trials = default_trial_manifests(
        study="smoke", parent_family="forest", seed=103,
    )
    assert len(parent_trials) == 50
    assert len(relation_trials) == 25
    assert parent_trials[0].to_dict() == default_trial_manifests(
        study="smoke", parent_family="forest", seed=103,
    )[0][0].to_dict()

    outer = np.arange(50, 100)
    y = np.tile([0, 1], 25)
    inner = inner_fold_indices(outer, y, task="classification", seed=104)
    assert len(inner) == 5
    for train, validate in inner:
        assert set(train) <= set(outer)
        assert set(validate) <= set(outer)
        assert not set(train) & set(validate)
    fit, calibration = calibration_split_indices(
        outer, y, task="classification", seed=105,
    )
    anchors = training_only_anchors(outer, fraction=0.3, seed=106)
    assert set(fit) <= set(outer)
    assert set(calibration) <= set(outer)
    assert not set(fit) & set(calibration)
    assert set(anchors) <= set(outer)
    assert len(anchors) == 15


def test_application_scopes_match_the_declared_experiment_protocol():
    assert "quantiles" not in APPLICATION_RELATION_SCOPES
    assert "conformal" not in APPLICATION_RELATION_SCOPES
    assert {"rfgap", "honest_gap"} <= APPLICATION_RELATION_SCOPES["anomaly"]
    assert application_relations(
        "anomaly", ["original", "rfgap", "honest_gap", "conditional"]
    ) == ["original", "rfgap", "honest_gap"]
    assert relation_use_policy("honest_gap", "anomaly")["use"] == "direct_predictive_weights"


def test_fold_aware_analysis_aggregates_before_paired_comparison():
    rows = pd.DataFrame([
        {"status": "ok", "suite_id": 99, "task_id": 1, "dataset_id": 10, "method": "base", "score": 0.4},
        {"status": "ok", "suite_id": 99, "task_id": 1, "dataset_id": 10, "method": "base", "score": 0.6},
        {"status": "ok", "suite_id": 99, "task_id": 1, "dataset_id": 10, "method": "new", "score": 0.8},
        {"status": "ok", "suite_id": 99, "task_id": 1, "dataset_id": 10, "method": "new", "score": 1.0},
        {"status": "ok", "suite_id": 99, "task_id": 2, "dataset_id": 20, "method": "base", "score": 0.2},
        {"status": "ok", "suite_id": 99, "task_id": 2, "dataset_id": 20, "method": "new", "score": 0.5},
    ])
    aggregated = aggregate_folds_within_dataset(
        rows, metric="score", comparison_columns=("method",),
    )
    assert len(aggregated) == 4
    paired = paired_effects(
        aggregated, metric="score", baseline="base", comparison="new",
    )
    np.testing.assert_allclose(paired["effect"], [0.4, 0.3])
    mean, low, high = bootstrap_confidence_interval(paired["effect"], n_resamples=100, seed=107)
    assert low <= mean <= high
    ranks = average_ranks(aggregated, metric="score")
    assert ranks.iloc[0]["method"] == "new"
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
