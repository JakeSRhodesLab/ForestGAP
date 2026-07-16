"""Fail CI when executable sources import the removed pre-1.0 wrapper API."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
REMOVED = {"with_sparse_proximity", "SparseProximityMixin"}


def _python_errors(source: str, label: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"{label}: cannot parse code while checking removed API: {exc}"]
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "forestgap.core":
                errors.append(f"{label}:{node.lineno}: imports removed forestgap.core")
            imported = {alias.name for alias in node.names}
            stale = sorted(imported & REMOVED)
            if stale:
                errors.append(f"{label}:{node.lineno}: imports removed API {stale}")
        elif isinstance(node, ast.Import):
            if any(alias.name == "forestgap.core" for alias in node.names):
                errors.append(f"{label}:{node.lineno}: imports removed forestgap.core")
    return errors


def main() -> None:
    failures: list[str] = []
    roots = [ROOT / "forestgap", ROOT / "experiments", ROOT / "scripts", ROOT / "tests"]
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            failures.extend(_python_errors(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT))))
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                failures.extend(_python_errors(cell.source, f"{path.relative_to(ROOT)} cell {index}"))
    if failures:
        raise SystemExit("\n".join(failures))
    print("no executable source imports the removed wrapper API")


if __name__ == "__main__":
    main()
