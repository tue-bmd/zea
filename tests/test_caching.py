"""Tests for the caching utility."""

import os
import time

import keras
import numpy as np
import pytest

os.environ["USBMD_CACHE_DIR"] = "/tmp/test_cache"

from usbmd.internal.cache import (
    cache_output,
    cache_summary,
    clear_cache,
    get_function_source,
)
from usbmd.internal.core import Object

# Global variable for the expected duration of the expensive operation
EXPECTED_DURATION = 0.05


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


@cache_output()
def _expensive_operation_obj(obj):
    # Simulate an expensive operation
    result = obj.x + obj.y
    time.sleep(EXPECTED_DURATION)
    return result


@cache_output("seed_gen")
def _expensive_operation_seed(seed_gen):
    # Simulate an expensive operation
    result = keras.random.uniform([1], seed=seed_gen)
    time.sleep(EXPECTED_DURATION)
    return np.squeeze(keras.ops.convert_to_numpy(result))


def _some_random_func():
    # This comment is required for some tests!
    return 1


@cache_output("x")
def _expensive_nested_operation(x, y):  # pylint: disable=unused-argument
    result = x + _some_random_func()
    time.sleep(EXPECTED_DURATION)
    return result


class CustomObject(Object):
    """Custom core object for testing caching"""

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y


@pytest.fixture(scope="module", autouse=True)
def clean_cache():
    """Fixture to clean up the cache directory before and after tests."""
    clear_cache()
    yield
    clear_cache()


def test_get_function_source():
    """Test getting the source code of a function."""

    def some_nested_func():
        # This comment is also required for some tests!
        _some_random_func()

    src = get_function_source(some_nested_func)
    assert (
        "# This comment is also required for some tests!" in src
    ), "Did not get source code"
    assert (
        "# This comment is required for some tests!" in src
    ), "Did not get nested source code"


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


def test_caching_custom_object():
    """Test caching for expensive_operation with CustomObject."""

    # First time should not be cached
    obj1 = CustomObject(2, 10)
    start_time = time.time()
    result = _expensive_operation_obj(obj1)
    duration = time.time() - start_time
    assert (
        duration >= EXPECTED_DURATION
    ), f"Expected duration >= {EXPECTED_DURATION}, got {duration}"
    assert result == 2 + 10, f"Expected 2 + 10, got {result}"

    # Second time should be cached
    start_time = time.time()
    result = _expensive_operation_obj(obj1)
    duration = time.time() - start_time
    assert (
        duration < EXPECTED_DURATION
    ), f"Expected duration < {EXPECTED_DURATION}, got {duration}"
    assert result == 2 + 10, f"Expected 2 + 10, got {result}"

    # If we use kwarg instead of arg should still be cached, see #561
    start_time = time.time()
    result = _expensive_operation_obj(obj=obj1)
    duration = time.time() - start_time
    assert (
        duration < EXPECTED_DURATION
    ), f"Expected duration < {EXPECTED_DURATION}, got {duration}"
    assert result == 2 + 10, f"Expected 2 + 10, got {result}"

    # Another instance with the same values should also be cached
    obj1_identical = CustomObject(2, 10)
    start_time = time.time()
    result = _expensive_operation_obj(obj1_identical)
    duration = time.time() - start_time
    assert (
        duration < EXPECTED_DURATION
    ), f"Expected duration < {EXPECTED_DURATION}, got {duration}"
    assert result == 2 + 10, f"Expected 2 + 10, got {result}"

    # Another object with different values should not be cached
    obj2 = CustomObject(3, 10)
    start_time = time.time()
    result = _expensive_operation_obj(obj2)
    duration = time.time() - start_time
    assert (
        duration >= EXPECTED_DURATION
    ), f"Expected duration >= {EXPECTED_DURATION}, got {duration}"
    assert result == 3 + 10, f"Expected 3 + 10, got {result}"

    # Another object with different values should not be cached
    obj3 = CustomObject(2, 20)
    start_time = time.time()
    result = _expensive_operation_obj(obj3)
    duration = time.time() - start_time
    assert (
        duration >= EXPECTED_DURATION
    ), f"Expected duration >= {EXPECTED_DURATION}, got {duration}"
    assert result == 2 + 20, f"Expected 2 + 20, got {result}"


def test_cache_summary():
    """Test cache summary."""
    cache_summary()
    assert True


def test_clear_cache():
    """Test clear cache."""
    clear_cache()
    assert True


def test_nested_cache():
    """Test nested caching."""
    start_time = time.time()
    result1 = _expensive_nested_operation(2, 10)
    duration1 = time.time() - start_time
    assert (
        duration1 >= EXPECTED_DURATION
    ), f"Expected duration >= {EXPECTED_DURATION}, got {duration1}"

    start_time = time.time()
    result2 = _expensive_nested_operation(2, 10)
    duration2 = time.time() - start_time
    assert (
        duration2 < EXPECTED_DURATION
    ), f"Expected duration < {EXPECTED_DURATION}, got {duration2}"
    assert result1 == result2, "Results should be equal"


def test_caching_seed_generator():
    """Test caching for expensive_operation with keras.seed.SeedGenerator."""
    seed_gen = keras.random.SeedGenerator(42)

    # First time should not be cached
    start_time = time.time()
    result1 = _expensive_operation_seed(seed_gen)
    duration = time.time() - start_time
    assert (
        duration >= EXPECTED_DURATION
    ), f"Expected duration >= {EXPECTED_DURATION}, got {duration}"

    # Second time should not be cached unless we reset seed_gen

    start_time = time.time()
    result2 = _expensive_operation_seed(seed_gen)
    duration = time.time() - start_time
    assert (
        duration >= EXPECTED_DURATION
    ), f"Expected duration >= {EXPECTED_DURATION}, got {duration}"
    assert result1 != result2, "Results should not be equal"

    # Reset seed_gen
    seed_gen = keras.random.SeedGenerator(42)
    start_time = time.time()
    result3 = _expensive_operation_seed(seed_gen)
    duration = time.time() - start_time
    assert (
        duration < EXPECTED_DURATION
    ), f"Expected duration < {EXPECTED_DURATION}, got {duration}"
    assert result1 == result3, "Results should be equal"

    # Different seed_gen should not be cached
    seed_gen = keras.random.SeedGenerator(43)
    start_time = time.time()
    _expensive_operation_seed(seed_gen)
    duration = time.time() - start_time
    assert (
        duration >= EXPECTED_DURATION
    ), f"Expected duration >= {EXPECTED_DURATION}, got {duration}"
