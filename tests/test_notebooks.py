"""Test example notebooks in docs/source/notebooks.

Tests if notebooks run without errors using papermill. Generally these notebooks
are a bit heavy, so we mark the tests with the `notebook` marker, and also run
only on self-hosted runners. Run with:

.. code-block:: bash

    pytest -s -m 'notebook'

Or to run a specific notebook:

.. code-block:: bash

    pytest -s -m 'notebook' --notebook dbua_example.ipynb

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
from pathlib import Path

import papermill as pm
import pytest

CONFIG_DIR = Path("configs")

# Automatically discover notebooks
NOTEBOOKS_DIR = Path("docs/source/notebooks")
NOTEBOOKS = list(NOTEBOOKS_DIR.rglob("*.ipynb"))

# Per-notebook parameters for CI testing (faster execution)
# these overwrite the default parameters in the notebooks
NOTEBOOK_PARAMETERS = {
    "diffusion_model_example.ipynb": {
        "n_unconditional_samples": 2,
        "n_unconditional_steps": 2,
        "n_conditional_samples": 2,
        "n_conditional_steps": 2,
    },
    "custom_models_example.ipynb": {
        "grid_size_x": 10,
        "grid_size_z": 10,
    },
    "agent_example.ipynb": {
        "n_prior_samples": 2,
        "n_unconditional_steps": 2,
        "n_initial_conditonal_steps": 1,
        "n_conditional_steps": 2,
        "n_conditional_samples": 2,
    },
    "task_based_perception_action_loop.ipynb": {
        "n_prior_steps": 2,
        "n_posterior_steps": 2,
        "n_particles": 2,
    },
    "3d_beamforming_example.ipynb": {
        "downscale_rate": 8,
    },
    "zea_sequence_example.ipynb": {
        "n_frames": 15,
        "n_tx": 1,
        "n_tx_total": 3,
    },
    "zea_data_example.ipynb": {
        "config_picmus_iq": f"{CONFIG_DIR}/config_picmus_iq.yaml",
    },
    "zea_local_data.ipynb": {
        "config_picmus_rf": f"{CONFIG_DIR}/config_picmus_rf.yaml",
    },
    "doppler_example.ipynb": {
        "n_frames": 3,
        "n_transmits": 2,
    },
    "speckle_tracking_example.ipynb": {
        "num_frames": 5,
        "num_points": 10,
        "max_iterations": 2,
    },
    "hvae_model_example.ipynb": {
        "inference_fractions": [0.03],
        "n_samples": 2,
        "batch_size": 1,
    },
    "dbua_example.ipynb": {
        "num_iterations": 2,
        "step_size": 1,
    },
    # Add more notebooks and their parameters here as needed
    # "other_notebook.ipynb": {
    #     "param1": value1,
    #     "param2": value2,
    # },
}

_notebook_names = [nb.name for nb in NOTEBOOKS]
for nbp_name in NOTEBOOK_PARAMETERS.keys():
    assert nbp_name in _notebook_names, (
        f"Notebook {nbp_name} not found in {NOTEBOOKS_DIR}. "
        "Wrong definition in NOTEBOOK_PARAMETERS?"
    )


def pytest_sessionstart(session):
    print(f"📚 Preparing to test {len(NOTEBOOKS)} notebooks from {NOTEBOOKS_DIR}")
    print(f"📝 Using custom parameters for {len(NOTEBOOK_PARAMETERS)} notebooks")


@pytest.mark.notebook
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda x: x.name)
def test_notebook_runs(notebook, tmp_path, request):
    # Filter by --notebook CLI option if provided
    notebook_filter = request.config.getoption("--notebook")
    if notebook_filter and notebook_filter not in notebook.name:
        pytest.skip(f"Skipped (--notebook={notebook_filter})")

    print(f"\n📘 Starting notebook: {notebook.name}")

    output_path = tmp_path / notebook.name
    start = time.time()

    # Get custom parameters for this notebook if they exist
    notebook_params = NOTEBOOK_PARAMETERS.get(notebook.name, {})
    if notebook_params:
        print(f"🔧 Using custom parameters: {notebook_params}")

    pm.execute_notebook(
        input_path=str(notebook),
        output_path=str(output_path),
        kernel_name="python3",
        parameters=notebook_params,
    )

    duration = time.time() - start
    print(f"✅ Finished {notebook.name} in {duration:.1f}s")
