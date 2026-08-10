from argparse import Namespace
import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from experiments.common import (
    available_model_names,
    build_outlier_model,
    build_proximity_model,
    configure_single_threaded_worker,
    extra_setup_selected,
    fit_outlier_model,
    proximity_methods_for_model,
    requested_model_names,
    requested_outlier_model_names,
    selected_model_names,
    selected_task_ids,
    setup_selected,
    stable_result_order,
)
from forestgap import capabilities_for


def test_parallel_selection_restricts_task_seed_model_and_extra_item():
    args = Namespace(
        _parallel_task_ids={"classification": [17], "regression": []},
        _parallel_setups=frozenset({("classification", 17, 3, "RandomForest")}),
        _parallel_models=frozenset({"RandomForest"}),
        _parallel_extra_items=frozenset({
            ("classification", 3, "synthetic_case", "RandomForest"),
        }),
    )

    assert selected_task_ids(args, "classification") == [17]
    assert selected_task_ids(args, "regression") == []
    assert setup_selected(args, "classification", 17, 3)
    assert not setup_selected(args, "classification", 17, 4)
    assert selected_model_names(args, ("RandomForest", "ExtraTrees")) == ("RandomForest",)
    assert extra_setup_selected(
        args,
        ("classification", 3, "synthetic_case", "RandomForest"),
    )


def test_worker_configuration_forces_single_threaded_ensemble(monkeypatch):
    monkeypatch.setenv("FORESTGAP_N_JOBS", "19")
    configure_single_threaded_worker()
    model = build_proximity_model(
        "classification",
        model_name="RandomForest",
        prox_method="original",
        random_state=11,
        n_estimators=3,
    )

    assert model.estimator.n_jobs == 1
    assert model.random_state == 11
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"


def test_common_registry_covers_supported_supervised_model_families():
    expected = {
        "CanonicalCorrelationForest",
        "CanonicalCorrelationForestBag",
        "GradientBoosting",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    }
    assert expected <= set(requested_model_names())
    assert {
        "RandomForest",
        "ExtraTrees",
        "CanonicalCorrelationForest",
        "CanonicalCorrelationForestBag",
        "GradientBoosting",
    } <= set(available_model_names("classification"))
    assert "SyntheticNullForest" in requested_outlier_model_names()


def test_common_registry_declares_family_specific_relation_sets():
    assert proximity_methods_for_model("CanonicalCorrelationForest") == (
        "original", "kerf", "path", "englund", "random_height",
    )
    assert proximity_methods_for_model("CanonicalCorrelationForestBag") == (
        "original", "oob", "rfgap", "kerf", "path", "englund", "random_height",
    )
    assert proximity_methods_for_model("GradientBoosting") == (
        "original", "kerf", "boosted", "path", "englund", "random_height",
    )
    assert proximity_methods_for_model("SyntheticNullForest") == (
        "original", "oob", "kerf", "path", "englund", "random_height",
    )


def test_common_builders_preserve_ccf_variants_and_support_new_builtin_models():
    X, y = make_classification(n_samples=42, n_features=6, random_state=12)
    ccf = build_proximity_model(
        "classification",
        model_name="CanonicalCorrelationForest",
        prox_method="original",
        random_state=13,
        n_estimators=3,
        bootstrap=True,
        oob_score=True,
    )
    assert ccf.estimator.bootstrap is False
    assert ccf.estimator.oob_score is False
    ccf.fit(X, y)
    assert "rfgap" not in capabilities_for(ccf.estimator).valid_relations

    ccf_bag = build_proximity_model(
        "classification",
        model_name="CanonicalCorrelationForestBag",
        prox_method="rfgap",
        random_state=14,
        n_estimators=3,
        bootstrap=False,
        oob_score=True,
    )
    assert ccf_bag.estimator.bootstrap is True
    ccf_bag.fit(X, y)
    assert "rfgap" in capabilities_for(ccf_bag.estimator).valid_relations

    boosted = build_proximity_model(
        "classification",
        model_name="GradientBoosting",
        prox_method="boosted",
        random_state=15,
        n_estimators=3,
    ).fit(X, y)
    assert boosted.get_proximities().shape == (len(X), len(X))

    synthetic, spec, metadata = build_outlier_model(
        "classification",
        model_name="SyntheticNullForest",
        prox_method="original",
        random_state=16,
        n_estimators=3,
        isolation_bootstrap=False,
    )
    fit_outlier_model(synthetic, spec, X, y)
    assert metadata["fit_bootstrap"] is True
    assert capabilities_for(synthetic.estimator).family == "synthetic_null_urf"
    assert np.asarray(synthetic.score_samples(X)).shape == (len(X),)


def test_parallel_result_order_is_deterministic():
    frame = pd.DataFrame([
        {
            "task_kind": "regression",
            "task_id": 2,
            "random_state": 1,
            "model_name": "RandomForest",
            "status": "ok",
            "value": 2,
        },
        {
            "task_kind": "classification",
            "task_id": 1,
            "random_state": 0,
            "model_name": "ExtraTrees",
            "status": "ok",
            "value": 1,
        },
    ])

    ordered = stable_result_order(frame)

    assert ordered["value"].tolist() == [1, 2]
