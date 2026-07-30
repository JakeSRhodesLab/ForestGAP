from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, TextIO


SCRIPTS = [
    "run_prediction_identity.py",
    "run_imputation.py",
    "run_outlier_detection.py",
    "run_prototypes.py",
    "run_local_feature_importance.py",
    "run_mds_visualization.py",
]
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _available_threads() -> int:
    return max(1, os.cpu_count() or 1)


def _child_env(thread_count: int) -> dict[str, str]:
    env = os.environ.copy()
    thread_count = str(max(1, thread_count))
    for env_name in THREAD_ENV_VARS:
        env[env_name] = thread_count
    env["FORESTGAP_N_JOBS"] = thread_count
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _python_command() -> list[str]:
    probe_env = os.environ.copy()
    probe_env["PYTHONNOUSERSITE"] = "1"
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sklearn.utils.validation import validate_data",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=probe_env,
    )
    if probe.returncode == 0:
        return [sys.executable, "-u"]

    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError(
            "Current Python environment is missing the sklearn.validate_data API "
            "required by the experiment runners, and conda is unavailable to "
            "launch the generalforest environment."
        )

    env_name = os.environ.get("GENERALFOREST_ENV", "generalforest")
    return [conda, "run", "--no-capture-output", "-n", env_name, "python", "-u"]


def _emit_terminal(lock: Any, message: str) -> None:
    with lock:
        sys.stdout.write(message)
        if not message.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()


def _tee_process_output(
    proc: subprocess.Popen[str],
    log_handle: TextIO,
    *,
    prefix: str,
    terminal_lock: Any,
) -> None:
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            log_handle.write(line)
            log_handle.flush()
            cleaned_line = line.rstrip("\r\n")
            _emit_terminal(terminal_lock, f"{prefix} {cleaned_line}")
    finally:
        proc.stdout.close()


def _write_process_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    log_dir = raw_dir / "run_all_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    process_rows: list[dict[str, str]] = []
    had_failure = False
    total_threads = _available_threads()
    child_thread_count = max(1, total_threads // len(SCRIPTS))
    python_command = _python_command()
    raw_path = raw_dir / "run_all_process_runtimes.csv"
    running: dict[str, dict[str, object]] = {}
    terminal_lock = threading.Lock()
    common_args = [
        "--task",
        args.task,
        "--random-states",
        args.random_states,
        "--test-size",
        str(args.test_size),
    ]
    if args.dataset_limit is not None:
        common_args.extend(["--dataset-limit", str(args.dataset_limit)])
    if args.max_rows is not None:
        common_args.extend(["--max-rows", str(args.max_rows)])

    for index, script in enumerate(SCRIPTS, start=1):
        log_path = log_dir / f"{index:02d}_{Path(script).stem}.log"
        cmd = [*python_command, str(root / script), *common_args]
        start = time.perf_counter()
        log_handle = log_path.open("w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd,
                env=_child_env(child_thread_count),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except Exception:
            log_handle.close()
            raise
        reader_thread = threading.Thread(
            target=_tee_process_output,
            args=(proc, log_handle),
            kwargs={
                "prefix": f"[{index}/{len(SCRIPTS)}:{Path(script).stem}]",
                "terminal_lock": terminal_lock,
            },
            daemon=True,
        )
        reader_thread.start()
        running[script] = {
            "index": index,
            "proc": proc,
            "start": start,
            "log_handle": log_handle,
            "log_path": log_path,
            "reader_thread": reader_thread,
        }
        _emit_terminal(
            terminal_lock,
            f"[{index}/{len(SCRIPTS)}] started {script} with {child_thread_count} threads "
            f"log={log_path}",
        )

    last_status_report = time.perf_counter()
    while running:
        completed_scripts: list[str] = []
        for script, info in list(running.items()):
            proc = info["proc"]
            assert isinstance(proc, subprocess.Popen)
            return_code = proc.poll()
            if return_code is None:
                continue
            completed_scripts.append(script)
            runtime_seconds = time.perf_counter() - float(info["start"])
            reader_thread = info["reader_thread"]
            assert isinstance(reader_thread, threading.Thread)
            reader_thread.join()
            log_handle = info["log_handle"]
            assert hasattr(log_handle, "close")
            log_handle.close()
            row = {
                "script_name": script,
                "task": args.task,
                "random_states": args.random_states,
                "dataset_limit": "" if args.dataset_limit is None else str(args.dataset_limit),
                "max_rows": "" if args.max_rows is None else str(args.max_rows),
                "test_size": str(args.test_size),
                "return_code": str(return_code),
                "status": "ok" if return_code == 0 else "error",
                "runtime_seconds": f"{runtime_seconds:.6f}",
            }
            process_rows.append(row)
            process_rows.sort(key=lambda item: SCRIPTS.index(item["script_name"]))
            _write_process_rows(raw_path, process_rows)
            _emit_terminal(
                terminal_lock,
                f"[{info['index']}/{len(SCRIPTS)}] finished {script} "
                f"rc={return_code} runtime={runtime_seconds:.2f}s log={info['log_path']}",
            )
            had_failure = had_failure or return_code != 0
        for script in completed_scripts:
            running.pop(script, None)
        if running and time.perf_counter() - last_status_report >= 30.0:
            active_scripts = ", ".join(
                f"{info['index']}:{script}" for script, info in running.items()
            )
            _emit_terminal(
                terminal_lock,
                f"{len(running)} scripts still running with {child_thread_count} threads each: "
                f"{active_scripts}",
            )
            last_status_report = time.perf_counter()
        if running:
            time.sleep(1.0)

    if had_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
