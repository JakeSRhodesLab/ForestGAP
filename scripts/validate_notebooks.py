"""Validate tutorial freshness and optionally execute every notebook in memory."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "honest_forest_outlier_demo_labeled.ipynb",
    "honest_gap_prediction_identity.ipynb",
    "imputation_and_outlier_detection.ipynb",
    "isolation_forest_demo.ipynb",
    "isolation_forest_outlier_comparison.ipynb",
    "local_feature_importance_demo.ipynb",
    "manifold.ipynb",
    "manifold_mwe.ipynb",
    "oblique_random_forest.ipynb",
    "oblique_rf_simulation_comparison.ipynb",
    "prototypes_demo.ipynb",
    "rotation_forest_proximities.ipynb",
    "spectral_cluster.ipynb",
    "canonical_correlation_forest.ipynb",
    "quantile_forest_proximities.ipynb",
    "random_survival_forest_proximities.ipynb",
    "synthetic_null_urf.ipynb",
}
STALE = ("with_sparse_proximity", "forestgap.core", "SparseProximityMixin")


def validate(path: Path, *, execute: bool) -> list[str]:
    notebook = nbformat.read(path, as_version=4)
    errors: list[str] = []
    sources = "\n".join(str(cell.source) for cell in notebook.cells)
    for token in STALE:
        if token in sources:
            errors.append(f"contains removed API token {token!r}")
    parameter_cells = [
        cell for cell in notebook.cells
        if "parameters" in cell.get("metadata", {}).get("tags", [])
    ]
    if len(parameter_cells) != 1:
        errors.append("must contain exactly one tagged parameters cell")
    metadata = notebook.metadata.get("forestgap", {})
    if not metadata.get("smoke") or metadata.get("network") is not False:
        errors.append("must declare deterministic no-network smoke metadata")
    if not metadata.get("executed_smoke"):
        errors.append("committed smoke output is stale or missing")
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if any(cell.execution_count is None for cell in code_cells):
        errors.append("one or more code cells lack committed execution counts")
    if any(output.get("output_type") == "error" for cell in code_cells for output in cell.get("outputs", [])):
        errors.append("contains a committed execution error")
    if path.name.startswith("isolation_forest"):
        invalid = ('relation="rfgap"', "relation='rfgap'", 'relation="oob"', "relation='oob'")
        if any(token in sources for token in invalid):
            errors.append("isolation tutorial requests an invalid RF-GAP/OOB relation")
    if path.name == "honest_gap_prediction_identity.ipynb" and "relation=\"honest_gap\"" not in sources:
        errors.append("HonestForest prediction tutorial must use honest_gap")
    required_relations = {
        "canonical_correlation_forest.ipynb": ('relation="original"',),
        "quantile_forest_proximities.ipynb": ('relation="conditional"',),
        "random_survival_forest_proximities.ipynb": ('relation="survival_leaf"',),
        "synthetic_null_urf.ipynb": ('relation="original"', 'relation="oob"'),
        "rotation_forest_proximities.ipynb": (
            'relation="rfgap"', '"path"', '"random_height"',
            "prox_predict", "oob_decision_function_", "rotated_xor",
            "UnsupervisedRandomForest", "ExtendedIsolationForest",
        ),
    }
    for token in required_relations.get(path.name, ()):
        if token not in sources:
            errors.append(f"must demonstrate {token}")
    if execute and not errors:
        NotebookClient(
            notebook, timeout=180, kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute(cwd=str(ROOT))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--only", metavar="NOTEBOOK", help="Validate one notebook by filename.")
    args = parser.parse_args()
    if args.only:
        path = ROOT / "notebooks" / args.only
        if not path.is_file():
            parser.error(f"notebook not found: {args.only}")
        paths = [path]
        failures: list[str] = []
    else:
        paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
        found = {path.name for path in paths}
        failures = [f"missing notebook {name}" for name in sorted(EXPECTED - found)]
        failures += [f"unexpected notebook {name}" for name in sorted(found - EXPECTED)]
    for path in paths:
        failures.extend(f"{path.name}: {error}" for error in validate(path, execute=args.execute))
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"validated {len(paths)} notebooks" + (" with execution" if args.execute else ""))


if __name__ == "__main__":
    main()
