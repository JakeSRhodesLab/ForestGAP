from __future__ import annotations

from argparse import Namespace
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from experiments.common import run_module_parallel
from experiments.harness import run_cells
import experiments.run_protocol_matrix as protocol_matrix
from experiments.run_imputation import compile_outputs as compile_imputation_outputs
from experiments.run_local_feature_importance import (
    compile_outputs as compile_local_importance_outputs,
)
from experiments.run_mds_visualization import compile_outputs as compile_mds_outputs
from experiments.run_outlier_detection import compile_outputs as compile_outlier_outputs
from experiments.run_prediction_identity import compile_outputs as compile_identity_outputs
from experiments.run_prototypes import compile_outputs as compile_prototype_outputs


def _runner_that_crashes_once(args: Namespace) -> pd.DataFrame:
    """Terminate one child process once so scheduler recovery is observable."""

    item = next(iter(args._parallel_extra_items))
    item_name = str(item[0])
    marker = Path(args._crash_once_marker)
    if item_name == "crash":
        try:
            descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
            os._exit(23)
    return pd.DataFrame([{"status": "ok", "item": item_name}])


def test_parallel_scheduler_retries_unfinished_groups_after_worker_exit(tmp_path):
    marker = tmp_path / "worker-exited"
    args = Namespace(
        task="classification",
        random_states="0",
        workers=2,
        _crash_once_marker=str(marker),
    )

    frame = run_module_parallel(
        __name__,
        args,
        runner_name="_runner_that_crashes_once",
        model_names=(),
        include_openml=False,
        extra_items=(("crash", "model_a"), ("complete", "model_b")),
    )

    assert marker.exists()
    assert set(frame["item"]) == {"crash", "complete"}


def test_application_output_compilers_accept_an_empty_result_set():
    empty = pd.DataFrame()

    compile_imputation_outputs(empty)
    compile_local_importance_outputs(empty)
    compile_mds_outputs(empty)
    compile_outlier_outputs(empty)
    compile_identity_outputs(empty)
    compile_prototype_outputs(empty, empty)


def test_protocol_matrix_records_dataset_preparation_errors(monkeypatch):
    official_fold = SimpleNamespace(
        task_id=17,
        dataset_id=42,
        fold_id="r0-f0-s0",
        repeat=0,
        fold=0,
        sample=0,
    )
    monkeypatch.setattr(
        protocol_matrix,
        "iter_openml_suite_folds",
        lambda _task_kind, _limit: iter((official_fold,)),
    )

    def fail_preparation(*_args, **_kwargs):
        raise ValueError("fixture dataset cannot be prepared")

    monkeypatch.setattr(protocol_matrix, "prepare_dataset", fail_preparation)

    cells = protocol_matrix.build_cells(
        task_kinds=("classification",),
        models=("random_forest", "extra_trees"),
        relations=("original",),
        seeds=(0, 1),
        dataset_limit=1,
        fold_limit=1,
        n_estimators=8,
    )
    rows = run_cells(cells)

    assert len(rows) == 4
    assert {row["status"] for row in rows} == {"error"}
    assert {row["error_type"] for row in rows} == {"ValueError"}
    assert all("fixture dataset cannot be prepared" in row["reason"] for row in rows)
