"""Execute every maintained tutorial in a clean kernel and commit its output."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", metavar="NOTEBOOK", help="Execute one notebook by filename.")
    args = parser.parse_args()
    if args.only:
        path = ROOT / "notebooks" / args.only
        if not path.is_file():
            parser.error(f"notebook not found: {args.only}")
        paths = [path]
    else:
        paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute(cwd=str(ROOT))
        notebook.metadata.setdefault("forestgap", {})["executed_smoke"] = True
        nbformat.write(notebook, path)
        print(f"executed {path.name}")


if __name__ == "__main__":
    main()
