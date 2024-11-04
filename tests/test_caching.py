"""Tests for the caching utility."""

import time

import pytest

from usbmd.utils.cache import CACHE_DIR, cache_output

# Global variable for the expected duration of the expensive operation
EXPECTED_DURATION = 0.2


@cache_output("x")
def _expensive_operation_x(x, y):  # pylint: disable=unused-argument
    # Simulate an expensive operation
    result = x
    time.sleep(EXPECTED_DURATION)
    return result


@cache_output("y")
def _expensive_operation_y(x, y):  # pylint: disable=unused-argument
    # Simulate an expensive operation
    result = y
    time.sleep(EXPECTED_DURATION)
    return result


@cache_output()
def _expensive_operation(x, y):
    # Simulate an expensive operation
    result = x + y
    time.sleep(EXPECTED_DURATION)
    return result


@pytest.fixture(scope="module", autouse=True)
def clean_cache():
    """Fixture to clean up the cache directory before and after tests."""
    if CACHE_DIR.exists():
        for file in CACHE_DIR.glob("*.pkl"):
            file.unlink()
    yield
    if CACHE_DIR.exists():
        for file in CACHE_DIR.glob("*.pkl"):
            file.unlink()


def test_caching_x():
    """Test caching for expensive_operation_x."""
    start_time = time.time()
    result = _expensive_operation_x(2, 10)
    duration = time.time() - start_time
    assert (
        duration >= EXPECTED_DURATION
    ), f"Expected duration >= {EXPECTED_DURATION}, got {duration}"
    assert result == 2, f"Expected 2, got {result}"

    start_time = time.time()
    result = _expensive_operation_x(2, 20)
    duration = time.time() - start_time
    assert (
        duration < EXPECTED_DURATION
    ), f"Expected duration < {EXPECTED_DURATION}, got {duration}"
    assert result == 2, f"Expected 2, got {result}"

    start_time = time.time()
    result = _expensive_operation_x(3, 10)
    duration = time.time() - start_time
    assert (
        duration >= EXPECTED_DURATION
    ), f"Expected duration >= {EXPECTED_DURATION}, got {duration}"
    assert result == 3, f"Expected 3, got {result}"


def test_caching_y():
    """Test caching for expensive_operation_y."""

    start_time = time.time()
    result = _expensive_operation_y(2, 10)
    duration = time.time() - start_time
    assert (
        duration >= EXPECTED_DURATION
    ), f"Expected duration >= {EXPECTED_DURATION}, got {duration}"
    assert result == 10, f"Expected 10, got {result}"

    start_time = time.time()
    result = _expensive_operation_y(3, 10)
    duration = time.time() - start_time
    assert (
        duration < EXPECTED_DURATION
    ), f"Expected duration < {EXPECTED_DURATION}, got {duration}"
    assert result == 10, f"Expected 10, got {result}"

    start_time = time.time()
    result = _expensive_operation_y(2, 20)
    duration = time.time() - start_time
    assert (
        duration >= EXPECTED_DURATION
    ), f"Expected duration >= {EXPECTED_DURATION}, got {duration}"
    assert result == 20, f"Expected 20, got {result}"


def test_caching():
    """Test caching for expensive_operation."""
    start_time = time.time()
    result = _expensive_operation(2, 10)
    duration = time.time() - start_time
    assert (
        duration >= EXPECTED_DURATION
    ), f"Expected duration >= {EXPECTED_DURATION}, got {duration}"
    assert result == 2 + 10, f"Expected 2 + 10, got {result}"

    start_time = time.time()
    result = _expensive_operation(2, 10)
    duration = time.time() - start_time
    assert (
        duration < EXPECTED_DURATION
    ), f"Expected duration < {EXPECTED_DURATION}, got {duration}"
    assert result == 2 + 10, f"Expected 2 + 10, got {result}"

    start_time = time.time()
    result = _expensive_operation(3, 10)
    duration = time.time() - start_time
    assert (
        duration >= EXPECTED_DURATION
    ), f"Expected duration >= {EXPECTED_DURATION}, got {duration}"
    assert result == 3 + 10, f"Expected 3 + 10, got {result}"

    start_time = time.time()
    result = _expensive_operation(2, 20)
    duration = time.time() - start_time
    assert (
        duration >= EXPECTED_DURATION
    ), f"Expected duration >= {EXPECTED_DURATION}, got {duration}"
    assert result == 2 + 20, f"Expected 2 + 20, got {result}"
