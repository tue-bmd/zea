"""Tests for the zea.utils module.

Contains both tests for zea.utils and zea.internal.utils.
"""

import re

import numpy as np
import pytest
from keras import ops

from zea.backend import jit
from zea.internal.utils import (
    calculate_file_hash,
    find_first_nonzero_index,
    find_key,
    first_not_none_item,
)
from zea.utils import (
    block_until_ready,
    get_date_string,
    strtobool,
    update_dictionary,
)


def test_calculate_file_hash_omit_line(tmp_path):
    """Test that calculate_file_hash correctly omits lines containing a string."""

    # Create a temporary file
    file_content = [
        "Dataset: test_folder\n",
        "Validated on: 2025_10_14_120000\n",
        "hash: should_be_ignored\n",
    ]
    file_path = tmp_path / "validation_file.txt"
    file_path.write_text("".join(file_content), encoding="utf-8")

    # Calculate hash ignoring the 'hash' line
    hash_without_hash_line = calculate_file_hash(file_path, omit_line_str="hash")

    expected_hash = "02d7d3d3f7731f715cc3c886752196c67893267b12a880455f0aeca0ad4d7da9"

    assert hash_without_hash_line == expected_hash

    hash_with_hash_line = calculate_file_hash(file_path, omit_line_str=None)

    assert hash_with_hash_line != expected_hash


@pytest.mark.parametrize(
    "dict1, dict2, keep_none, expected_result",
    [
        (
            {1: "one", 2: "two"},
            {2: "new_two", 3: "three"},
            False,
            {1: "one", 2: "new_two", 3: "three"},
        ),
        (
            {1: "one", 2: "two"},
            {2: None, 3: "three"},
            False,
            {1: "one", 2: "two", 3: "three"},
        ),
        ({}, {1: "one"}, False, {1: "one"}),
        ({1: "one"}, {}, False, {1: "one"}),
        (
            {1: "one", 2: "two"},
            {2: None, 3: "three"},
            True,
            {1: "one", 2: None, 3: "three"},
        ),
        ({}, {}, False, {}),
    ],
)
def test_update_dictionary(dict1, dict2, keep_none, expected_result):
    """Tests the update_dictionary function using simple equality check."""
    result = update_dictionary(dict1, dict2, keep_none)
    assert result == expected_result


@pytest.mark.parametrize(
    "contains, case_sensitive",
    [["apple", False], ["apple", True], ["pie", True]],
)
def test_find_key(contains, case_sensitive):
    """Tests the find_key function by providing a test dictionary and checking the
    number of keys found."""
    dictionary = {
        "APPLES": 1,
        "apple pie": 2,
        "cherry pie": 3,
        "what apple": 4,
        "rainbow": 5,
    }

    result = find_key(dictionary, contains, case_sensitive)

    # Check that the result is a string
    assert isinstance(result, str), "Result is not a list"
    # Check that the result is actually in the dictionary
    assert result in dictionary.keys(), "Key not found in dictionary"

    # Check that the result contains the search string
    if not case_sensitive:
        result = result.lower()
        contains = contains.lower()

    assert contains in result, "Key does not contain the search string"


def test_nonexistent_key_raises_keyerror():
    """Tests that a KeyError is raised if the key is not found."""
    dictionary = {"APPLES": 1, "apple pie": 2, "cherry pie": 3, "rainbow": 5}

    with pytest.raises(KeyError):
        find_key(dictionary, "banana", case_sensitive=True)


def test_strtobool():
    """ "Test strtobool function with multiple user inputs."""
    # 1. Non string input raises assertion error
    with pytest.raises(AssertionError, match="Input value must be a string"):
        strtobool(1)

    # 2. strtobool is case insensitive
    assert strtobool("TRUE") is True
    assert strtobool("TruE") is True
    assert strtobool("true") is True

    # 3. valid 'true' values get mapped to True
    valid_true_values = ["y", "yes", "t", "true", "on", "1"]
    assert np.all([strtobool(v) for v in valid_true_values])

    # 4. valid 'false' values get mapped to False
    valid_false_values = ["n", "no", "f", "false", "off", "0"]
    assert not np.any([strtobool(v) for v in valid_false_values])

    # 5. any other value raises a ValueError
    sample_invalid_values = ["🤔", "invalid_value", "hello!"]
    for invalid_value in sample_invalid_values:
        with pytest.raises(ValueError, match=f"invalid truth value {invalid_value}"):
            strtobool(invalid_value)


def test_get_date_string():
    """Tests the get_date_string function."""

    # Test default date format
    date_string = get_date_string()
    assert isinstance(date_string, str), "Result is not a string"
    date_string = get_date_string()

    # Check if date string matches pattern YYYY_MM_DD_HHMMSS
    regex_pattern = r"^\d{4}_\d{2}_\d{2}_\d{6}$"
    assert re.match(regex_pattern, date_string), "Date string does not match pattern"

    # Test alternative date format
    date_string = get_date_string(string="%d-%m-%Y")
    assert isinstance(date_string, str), "Result is not a string"
    regex_pattern = r"^\d{2}-\d{2}-\d{4}$"
    assert re.match(regex_pattern, date_string), "Date string does not match pattern"

    # Test if the function raises an error at invalid input
    with pytest.raises(TypeError):
        get_date_string(string=1)

    with pytest.raises(TypeError):
        get_date_string(string=lambda x: x)


@pytest.mark.parametrize(
    "arr, axis, invalid_val, expected",
    [
        ((0, 0, 0, 5, 0, 3, 0), 0, -1, 3),
        ([[0, 0, 0], [4, 0, 0], [0, 0, 7]], 1, None, [None, 0, 2]),
    ],
)
def test_find_first_nonzero_index(arr, axis, invalid_val, expected):
    """Tests the find_first_nonzero_index function."""
    arr = np.array(arr)
    result = find_first_nonzero_index(arr, axis, invalid_val=invalid_val)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "arr, expected",
    [
        ([None, None], None),
        ([None, False, 0, 1, 2.0], False),
        ([2.0, None], 2.0),
    ],
)
def test_first_not_none_item(arr, expected):
    """Tests the find_first_nonzero_index function."""
    result = first_not_none_item(arr)
    np.testing.assert_equal(result, expected)


def test_block_until_ready_timing():
    """Tests that block_until_ready calls the correct backend-specific function."""
    from unittest.mock import patch

    import keras

    @jit
    def slow_computation(x):
        # Vectorized heavy computation (no Python for-loop):
        # - Build an outer-product matrix (n x n)
        # - Apply elementwise trig operations
        # - Reduce to a scalar
        y = ops.matmul(x[:, None], x[None, :])  # shape (n, n)
        z = ops.sin(y) + ops.cos(y)
        return ops.sum(z, axis=1)

    x = ops.ones(1000)

    # Compile first
    _ = slow_computation(x)

    backend_name = keras.backend.backend()

    if backend_name == "jax":
        # Test that jax.block_until_ready is called for JAX backend
        with patch("jax.block_until_ready") as mock_jax_block:
            # Make mock return the input unchanged
            mock_jax_block.side_effect = lambda x: x

            result = block_until_ready(slow_computation)(x)

            # Verify jax.block_until_ready was called
            mock_jax_block.assert_called_once()
            assert result is not None
    else:
        # Test that keras.ops.convert_to_numpy is called for other backends
        with patch("keras.ops.convert_to_numpy") as mock_convert:
            # Make mock return the input unchanged
            mock_convert.side_effect = lambda x: x

            result = block_until_ready(slow_computation)(x)

            # Verify keras.ops.convert_to_numpy was called
            mock_convert.assert_called_once()
            assert result is not None

    print(f"Backend: {backend_name}")
    print("block_until_ready backend-specific function test completed!")
