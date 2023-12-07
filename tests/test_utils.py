"""Tests for the usbmd.utils.utils module."""
import re

import numpy as np
import pytest

from usbmd.utils.utils import (
    find_first_nonzero_index,
    find_key,
    first_not_none_item,
    get_date_string,
    strtobool,
    translate,
    update_dictionary,
)


@pytest.mark.parametrize(
    "range_from, range_to",
    [((0, 100), (2, 5)), ((-60, 0), (0, 255))],
)
def test_translate(range_from, range_to):
    """Tests the translate function by providing a test array with its range_from and
    a range to."""
    arr = np.random.randint(low=range_from[0] + 1, high=range_from[1] - 2, size=10)
    right_min, right_max = range_to
    result = translate(arr, range_from, range_to)
    assert right_min <= np.min(result), "Minimum value is too small"
    assert np.max(result) <= right_max, "Maximum value is too large"


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
    """"Test strtobool function with multiple user inputs."""
    # 1. Non string input raises assertion error
    with pytest.raises(AssertionError, match="Input value must be a string"):
        strtobool(1)

    # 2. strtobool is case insensitive
    assert strtobool("TRUE") == True
    assert strtobool("TruE") == True
    assert strtobool("true") == True

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
