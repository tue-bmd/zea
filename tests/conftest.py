""" This file contains fixtures that are used by all tests in the tests directory. """

import pytest

from . import elp


@pytest.fixture(scope="session", autouse=True)
def run_once_after_all_tests():
    """Fixture to stop workers after all tests have run."""
    yield
    print("Stopping workers")
    elp.stop_workers()
