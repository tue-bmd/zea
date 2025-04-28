"""This file contains fixtures that are used by all tests in the tests directory."""

import matplotlib.pyplot as plt
import pytest

from . import backend_workers

plt.rcParams["backend"] = "agg"


@pytest.fixture(scope="session", autouse=True)
def run_once_after_all_tests():
    """Fixture to stop workers after all tests have run."""
    yield
    print("Stopping workers")
    backend_workers.stop_workers()
