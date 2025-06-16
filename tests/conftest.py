"""This file contains fixtures that are used by all tests in the tests directory."""

import os
import tempfile

import matplotlib.pyplot as plt
import pytest

_tmp_cache_dir = tempfile.TemporaryDirectory(prefix="zea_test_cache_")

os.environ["ZEA_CACHE_DIR"] = _tmp_cache_dir.name  # set before importing zea

from zea.data.data_format import generate_example_dataset  # noqa: E402

from . import (  # noqa: E402
    DUMMY_DATASET_N_FRAMES,
    DUMMY_DATASET_N_X,
    DUMMY_DATASET_N_Z,
    backend_workers,
)

plt.rcParams["backend"] = "agg"


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run tests with GPU support if CUDA is available",
    )


@pytest.fixture(scope="session", autouse=True)
def run_once_after_all_tests():
    """Fixture to stop workers after all tests have run."""
    yield
    print("Stopping workers")
    backend_workers.stop_workers()


@pytest.fixture(scope="session", autouse=True)
def clean_cache_dir():
    """Fixture to clean the cache directory after all tests."""
    yield
    print("Cleaning cache directory")
    _tmp_cache_dir.cleanup()


@pytest.fixture
def dummy_dataset_path(tmp_path):
    """Fixture to create a temporary dataset"""
    for i in range(2):
        temp_file = tmp_path / f"test{i}.hdf5"
        generate_example_dataset(
            temp_file,
            add_optional_dtypes=True,
            n_frames=DUMMY_DATASET_N_FRAMES,
            n_z=DUMMY_DATASET_N_Z,
            n_x=DUMMY_DATASET_N_X,
        )

    yield str(tmp_path)
