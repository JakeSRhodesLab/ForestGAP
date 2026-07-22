from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys
import time


SCRIPTS = [
    "run_prediction_identity.py",
    "run_imputation.py",
    "run_outlier_detection.py",
    "run_prototypes.py",
    "run_local_feature_importance.py",
    "run_mds_visualization.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("classification", "regression", "both"), default="both")
    parser.add_argument("--random-states", default="0,1,2")
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.3)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    results_dir = root.parent / "results"
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    process_rows: list[dict[str, str]] = []
    for script in SCRIPTS:
        cmd = [
            sys.executable,
            str(root / script),
            "--task",
            args.task,
            "--random-states",
            args.random_states,
            "--test-size",
            str(args.test_size),
        ]
        if args.dataset_limit is not None:
            cmd.extend(["--dataset-limit", str(args.dataset_limit)])
        if args.max_rows is not None:
            cmd.extend(["--max-rows", str(args.max_rows)])
        start = time.perf_counter()
        completed = subprocess.run(cmd, check=False)
        runtime_seconds = time.perf_counter() - start
        process_rows.append(
            {
                "script_name": script,
                "task": args.task,
                "random_states": args.random_states,
                "dataset_limit": "" if args.dataset_limit is None else str(args.dataset_limit),
                "max_rows": "" if args.max_rows is None else str(args.max_rows),
                "test_size": str(args.test_size),
                "return_code": str(completed.returncode),
                "status": "ok" if completed.returncode == 0 else "error",
                "runtime_seconds": f"{runtime_seconds:.6f}",
            }
        )

    raw_path = raw_dir / "run_all_process_runtimes.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(process_rows[0].keys()))
        writer.writeheader()
        writer.writerows(process_rows)


if __name__ == "__main__":
    main()
