import json
import subprocess
import sys


def test_method_matrix_cli_smoke_writes_all_requested_cells(tmp_path):
    output = tmp_path / "cells.jsonl"
    completed = subprocess.run(
        [
            sys.executable, "-m", "experiments.run_method_matrix", "--smoke",
            "--estimators", "random_forest,ccf",
            "--relations", "original,rfgap",
            "--applications", "prediction",
            "--data-regimes", "axis_aligned",
            "--seeds", "0",
            "--output", str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout.strip().splitlines()[-1])
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["cells"] == 4 == len(rows)
    assert summary["status"] == {"error": 0, "ok": 3, "unsupported": 1}


def test_every_maintained_runner_exposes_a_cli_help_surface():
    modules = [
        "experiments.run_method_matrix",
        "experiments.run_imputation",
        "experiments.run_local_feature_importance",
        "experiments.run_mds_visualization",
        "experiments.run_outlier_detection",
        "experiments.run_prediction_identity",
        "experiments.run_prototypes",
    ]
    for module in modules:
        completed = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "usage:" in completed.stdout.lower()
