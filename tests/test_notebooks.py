"""Test example notebooks in docs/source/notebooks.

Tests if notebooks run without errors using papermill. Generally these notebooks
are a bit heavy, so we mark the tests with the `notebook` marker, and also run
only on self-hosted runners. Run with:

.. code-block:: bash
    pytest -m 'notebook'

"""

from pathlib import Path

import papermill as pm
import pytest

# Automatically discover notebooks
NOTEBOOKS_DIR = Path("docs/source/notebooks")
NOTEBOOKS = list(NOTEBOOKS_DIR.rglob("*.ipynb"))


@pytest.mark.notebook
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda x: x.name)
def test_notebook_runs(notebook, tmp_path):
    output_path = tmp_path / notebook.name
    pm.execute_notebook(
        input_path=str(notebook),
        output_path=str(output_path),
        kernel_name="python3",
        parameters={},  # Optional: set notebook parameters
    )
