"""Test example notebooks in docs/source/notebooks.

Tests if notebooks run without errors using papermill. Generally these notebooks
are a bit heavy, so we mark the tests with the `notebook` marker, and also run
only on self-hosted runners. Run with:

.. code-block:: bash
    pytest -s -m 'notebook'

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
from pathlib import Path

import papermill as pm
import pytest

# Automatically discover notebooks
NOTEBOOKS_DIR = Path("docs/source/notebooks")
NOTEBOOKS = list(NOTEBOOKS_DIR.rglob("*.ipynb"))


def pytest_sessionstart(session):
    print(f"📚 Preparing to test {len(NOTEBOOKS)} notebooks from {NOTEBOOKS_DIR}")


@pytest.mark.notebook
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda x: x.name)
def test_notebook_runs(notebook, tmp_path):
    print(f"\n📘 Starting notebook: {notebook.name}")

    output_path = tmp_path / notebook.name
    start = time.time()

    pm.execute_notebook(
        input_path=str(notebook),
        output_path=str(output_path),
        kernel_name="python3",
        parameters={},
    )

    duration = time.time() - start
    print(f"✅ Finished {notebook.name} in {duration:.1f}s")
