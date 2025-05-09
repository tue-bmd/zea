"""This file contains fixtures that are used by all tests in the tests directory."""

import matplotlib.pyplot as plt
import pytest

from usbmd.data.data_format import generate_example_dataset

from . import backend_workers

plt.rcParams["backend"] = "agg"


@pytest.fixture(scope="session", autouse=True)
def run_once_after_all_tests():
    """Fixture to stop workers after all tests have run."""
    yield
    print("Stopping workers")
    backend_workers.stop_workers()


@pytest.fixture
def dummy_dataset_path(tmp_path):
    """Fixture to create a temporary dataset"""
    for i in range(2):
        temp_file = tmp_path / f"test{i}.hdf5"
        generate_example_dataset(temp_file, add_optional_dtypes=True, n_frames=4)

    yield str(tmp_path)
