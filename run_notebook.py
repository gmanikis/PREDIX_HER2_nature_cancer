#!/usr/bin/env python
"""Re-execute PREDIX_HER2_reproducibility.ipynb headless and save it in place.

Usage:
    python run_notebook.py

Requires nbformat, nbclient and ipykernel in addition to
environment/requirements.txt:

    pip install nbformat nbclient ipykernel

The notebook is executed with the package root (this file's directory) as the
working directory, kernel "python3", and a 3600-second per-cell timeout. The
executed notebook, outputs embedded, replaces the file on disk. Exit code 0
means every cell ran without error.
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
NB = HERE / "PREDIX_HER2_reproducibility.ipynb"


def main():
    nb = nbformat.read(NB, as_version=4)
    client = NotebookClient(
        nb,
        timeout=3600,
        kernel_name="python3",
        resources={"metadata": {"path": str(HERE)}},
    )
    print(f"Executing {NB.name} (cwd = {HERE}) ...")
    try:
        client.execute()
    finally:
        # Save whatever executed, so a failing cell's traceback is inspectable.
        nbformat.write(nb, NB)
    n_err = sum(
        1
        for c in nb.cells
        if c.cell_type == "code"
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    )
    print(f"Done. Cells with error output: {n_err}")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
