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

# Per-notebook parameters for CI testing (faster execution). Parameters override the default
# notebook parameters. Required backends are mapped to pytest markers on the parametrized test case.
NOTEBOOK_OVERRIDES = {
    "diffusion_model_example.ipynb": {
        "parameters": {
            "n_unconditional_samples": 2,
            "n_unconditional_steps": 2,
            "n_conditional_samples": 2,
            "n_conditional_steps": 2,
        }
    },
    "custom_models_example.ipynb": {
        "parameters": {
            "grid_size_x": 10,
            "grid_size_z": 10,
        }
    },
    "agent_example.ipynb": {
        "parameters": {
            "n_prior_samples": 2,
            "n_unconditional_steps": 2,
            "n_initial_conditonal_steps": 1,
            "n_conditional_steps": 2,
            "n_conditional_samples": 2,
        }
    },
    "task_based_perception_action_loop.ipynb": {
        "parameters": {
            "n_prior_steps": 2,
            "n_posterior_steps": 2,
            "n_particles": 2,
        }
    },
    "3d_beamforming_example.ipynb": {
        "parameters": {
            "downscale_rate": 8,
        }
    },
    "zea_sequence_example.ipynb": {
        "parameters": {
            "n_frames": 15,
            "n_tx": 1,
            "n_tx_total": 3,
        }
    },
    "zea_data_example.ipynb": {
        "parameters": {
            "config_picmus_iq": f"{CONFIG_DIR}/config_picmus_iq.yaml",
        }
    },
    "zea_local_data.ipynb": {
        "parameters": {
            "config_picmus_rf": f"{CONFIG_DIR}/config_picmus_rf.yaml",
        }
    },
    "doppler_example.ipynb": {
        "parameters": {
            "n_frames": 3,
            "n_transmits": 2,
        },
        "required_backends": ("tensorflow",),
    },
    "speckle_tracking_example.ipynb": {
        "parameters": {
            "num_frames": 5,
            "num_points": 10,
            "max_iterations": 2,
        }
    },
    "hvae_model_example.ipynb": {
        "parameters": {
            "inference_fractions": [0.03],
            "n_samples": 2,
            "batch_size": 2,
            "load_weights": False,
        }
    },
    "dbua_example.ipynb": {
        "parameters": {
            "num_iterations": 2,
            "step_size": 1,
        }
    },
    "nuclear_dehazing_example.ipynb": {
        "parameters": {
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
        }
    },
    "refocus_pipeline_example.ipynb": {
        "parameters": {
            "num_transmits": 2,
            "grid_size_x": 16,
            "grid_size_z": 16,
        }
    },
    "taesd_autoencoder_example.ipynb": {
        "required_backends": ("tensorflow",),
    },
    "carotid_segmentation_example.ipynb": {
        "required_backends": ("tensorflow",),
    },
    "left_ventricle_segmentation_example.ipynb": {
        "required_backends": ("tensorflow",),
    },
    "myocardial_quality_example.ipynb": {
        "required_backends": ("tensorflow",),
    },
}

_notebook_names = [nb.name for nb in NOTEBOOKS]
for notebook_name in NOTEBOOK_OVERRIDES:
    assert notebook_name in _notebook_names, (
        f"Notebook {notebook_name} not found in {NOTEBOOKS_DIR}. "
        "Wrong definition in NOTEBOOK_OVERRIDES?"
    )


def _notebook_parameters(notebook_name):
    return NOTEBOOK_OVERRIDES.get(notebook_name, {}).get("parameters", {})


def _notebook_marks(notebook_name):
    required_backends = NOTEBOOK_OVERRIDES.get(notebook_name, {}).get("required_backends", ())
    return tuple(getattr(pytest.mark, backend) for backend in required_backends)


NOTEBOOK_CASES = [
    pytest.param(notebook, id=notebook.name, marks=_notebook_marks(notebook.name))
    for notebook in NOTEBOOKS
]


def pytest_sessionstart(session):
    print(f"📚 Preparing to test {len(NOTEBOOKS)} notebooks from {NOTEBOOKS_DIR}")
    notebooks_with_parameters = sum(
        bool(override.get("parameters")) for override in NOTEBOOK_OVERRIDES.values()
    )
    print(f"📝 Using custom parameters for {notebooks_with_parameters} notebooks")


@pytest.mark.notebook
@pytest.mark.parametrize("notebook", NOTEBOOK_CASES)
def test_notebook_runs(notebook, tmp_path, request):
    # Filter by --notebook CLI option if provided
    notebook_filter = request.config.getoption("--notebook")
    if notebook_filter and notebook_filter not in notebook.name:
        pytest.skip(f"Skipped (--notebook={notebook_filter})")

    print(f"\n📘 Starting notebook: {notebook.name}")

    output_path = tmp_path / notebook.name
    start = time.time()

    # Get custom parameters for this notebook if they exist
    notebook_params = _notebook_parameters(notebook.name)
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
