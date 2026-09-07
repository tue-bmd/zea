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
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
from pathlib import Path

import nbformat
import papermill as pm
import pytest

from . import _notebook_timings

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
        "n_frames": 5,
        "n_points": 10,
        "max_iterations": 2,
    },
    "hvae_model_example.ipynb": {
        "inference_fractions": [0.03],
        "n_samples": 2,
        "batch_size": 2,
        "load_weights": False,
    },
    "dbua_example.ipynb": {
        "num_iterations": 2,
        "step_size": 1,
    },
    "nuclear_dehazing_example.ipynb": {
        "n_unconditional_samples": 1,
        "n_unconditional_steps": 2,
        "n_conditional_samples": 1,
        "n_conditional_steps": 2,
        "diffusion_steps": 2,
        "window_size": 2,
        "hard_project": True,
        "omega": 1.0,
        "gamma": 1.0,
        "haze_level": 0.5,
        "rank_weight_factor": 20,
        "initial_step": 0,
    },
    "refocus_pipeline_example.ipynb": {
        "num_transmits": 2,
        "grid_size_x": 16,
        "grid_size_z": 16,
    },
    "zea_simulation_example.ipynb": {
        "n_repeats": 2,
    },
    # Add more notebooks and their parameters here as needed
    # "other_notebook.ipynb": {
    #     "param1": value1,
    #     "param2": value2,
    # },
}

TENSORFLOW_NOTEBOOKS = {
    "doppler_example.ipynb",
    "taesd_autoencoder_example.ipynb",
    "carotid_segmentation_example.ipynb",
    "left_ventricle_segmentation_example.ipynb",
    "myocardial_quality_example.ipynb",
}

_notebook_names = [nb.name for nb in NOTEBOOKS]
for notebook_name in NOTEBOOK_PARAMETERS:
    assert notebook_name in _notebook_names, (
        f"Notebook {notebook_name} not found in {NOTEBOOKS_DIR}. "
        "Wrong definition in NOTEBOOK_PARAMETERS?"
    )
for notebook_name in TENSORFLOW_NOTEBOOKS:
    assert notebook_name in _notebook_names, (
        f"Notebook {notebook_name} not found in {NOTEBOOKS_DIR}. "
        "Wrong definition in TENSORFLOW_NOTEBOOKS?"
    )


# Upper bound on a single cell's run time. This is deliberately generous: the slowest
# notebook in CI (hvae_model_example) runs end-to-end in about four minutes, so no
# individual cell comes close. The bound is not a performance budget -- it exists so a
# cell that blocks forever on a stalled network call fails in minutes, naming the cell,
# instead of silently consuming the job's full timeout.
DEFAULT_CELL_TIMEOUT = 600


def _execute_notebook(notebook, output_path, parameters=None, cell_timeout=DEFAULT_CELL_TIMEOUT):
    """Run ``notebook`` through papermill with a per-cell time limit.

    ``execution_timeout`` bounds each cell separately rather than the notebook as a
    whole, so a notebook made of many slow cells is unaffected by the limit. When a cell
    exceeds it papermill raises ``CellTimeoutError`` (a builtin ``TimeoutError``) quoting
    the source of the cell that hung, which is what makes a stall diagnosable from a CI
    log alone.
    """
    return pm.execute_notebook(
        input_path=str(notebook),
        output_path=str(output_path),
        kernel_name="python3",
        parameters=parameters or {},
        execution_timeout=cell_timeout,
    )


def _notebook_case(notebook):
    marks = (pytest.mark.tensorflow,) if notebook.name in TENSORFLOW_NOTEBOOKS else ()
    return pytest.param(notebook, id=notebook.name, marks=marks)


NOTEBOOK_CASES = [_notebook_case(notebook) for notebook in NOTEBOOKS]


# Files the notebooks write into the working directory to be self-contained. They are
# part of what the notebook demonstrates, so they belong in the notebook rather than in
# a fixture -- but a test run should not leave them behind in a checkout.
NOTEBOOK_ARTIFACTS = (Path("users.yaml"), Path("zea-data"))


@pytest.fixture(autouse=True)
def clean_notebook_artifacts():
    """Remove working-directory files a notebook created, keeping anything pre-existing.

    ``zea_local_data.ipynb`` writes a ``users.yaml`` and a data folder so that it runs
    warning-free on a machine that has never seen zea. Anything already present belongs
    to the developer running the tests and is left untouched.
    """
    pre_existing = {path for path in NOTEBOOK_ARTIFACTS if path.exists()}
    yield
    for path in NOTEBOOK_ARTIFACTS:
        if path in pre_existing or not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


@pytest.mark.notebook
@pytest.mark.parametrize("notebook", NOTEBOOK_CASES)
def test_notebook_runs(notebook, tmp_path, request):
    # Filter by --notebook CLI option if provided
    notebook_filter = request.config.getoption("--notebook")
    if notebook_filter and notebook_filter not in notebook.name:
        pytest.skip(f"Skipped (--notebook={notebook_filter})")

    # Filter by --notebook-dir CLI option if provided
    notebook_dir_filter = request.config.getoption("--notebook-dir")
    if notebook_dir_filter and notebook.parent.name not in notebook_dir_filter:
        pytest.skip(f"Skipped (--notebook-dir={notebook_dir_filter})")

    print(f"\n📘 Starting notebook: {notebook.name}")

    output_path = tmp_path / notebook.name
    start = time.time()

    # Get custom parameters for this notebook if they exist
    notebook_params = NOTEBOOK_PARAMETERS.get(notebook.name, {})
    if notebook_params:
        print(f"🔧 Using custom parameters: {notebook_params}")

    _execute_notebook(
        notebook,
        output_path,
        parameters=notebook_params,
        cell_timeout=request.config.getoption("--notebook-cell-timeout") or DEFAULT_CELL_TIMEOUT,
    )

    duration = time.time() - start
    _notebook_timings[notebook.name] = (notebook.parent.name, duration)
    print(f"✅ Finished {notebook.name} in {duration:.1f}s")


def _write_notebook(path, sources):
    """Write a minimal notebook at ``path`` whose code cells hold ``sources``."""
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell(source) for source in sources]
    nbformat.write(notebook, str(path))
    return path


def test_cell_timeout_names_the_hanging_cell(tmp_path):
    """A cell that blocks past the limit fails, and the error quotes that cell.

    This is the whole point of bounding execution: a stalled cell has to be
    identifiable from the CI log, since ``%%capture`` and papermill's progress bar
    otherwise reduce a hang to a cell count.
    """
    marker = "sentinel_cell_that_hangs"
    notebook = _write_notebook(
        tmp_path / "hangs.ipynb",
        [f"import time  # {marker}\ntime.sleep(300)"],
    )

    start = time.time()
    with pytest.raises(TimeoutError) as excinfo:
        _execute_notebook(notebook, tmp_path / "out.ipynb", cell_timeout=5)
    elapsed = time.time() - start

    # Fails on the limit, not on the cell's own 300s sleep.
    assert elapsed < 120, f"timeout did not interrupt the cell (took {elapsed:.1f}s)"
    assert marker in str(excinfo.value), (
        "timeout error should quote the offending cell so it can be identified from "
        f"a log alone, got: {excinfo.value}"
    )


def test_cell_timeout_does_not_disturb_normal_execution(tmp_path):
    """Setting the limit must not change how a notebook that finishes in time runs."""
    notebook = _write_notebook(
        tmp_path / "quick.ipynb",
        ["value = 6 * 7", "assert value == 42"],
    )
    output_path = tmp_path / "out.ipynb"

    _execute_notebook(notebook, output_path, cell_timeout=DEFAULT_CELL_TIMEOUT)

    assert output_path.exists()
