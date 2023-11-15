"""Tests for the usbmd.utils.utils module."""
import numpy as np
import pytest

from usbmd.utils.utils import find_first_nonzero_index, first_not_none_item, strtobool


def test_strtobool():
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
