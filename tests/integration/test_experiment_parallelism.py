from argparse import Namespace
import os

import pandas as pd

from experiments.common import (
    build_proximity_model,
    configure_single_threaded_worker,
    extra_setup_selected,
    selected_model_names,
    selected_task_ids,
    setup_selected,
    stable_result_order,
)


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
