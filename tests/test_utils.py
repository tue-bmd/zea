"""Tests for the usbmd.utils.utils module."""
import numpy as np
import pytest

from usbmd.utils.utils import find_first_nonzero_index, first_not_none_item


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
