import pytest

from . import elp


@pytest.fixture(scope="session", autouse=True)
def run_once_after_all_tests():
    yield
    print("Stopping workers")
    elp.stop_workers()
