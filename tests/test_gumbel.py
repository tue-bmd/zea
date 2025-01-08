""" Tests for usbmd.agent.gumbel """

import numpy as np
import pytest
from keras import ops

from usbmd.agent.gumbel import hard_straight_through


@pytest.mark.parametrize(
    "khot_orig, k, n_value_dims, expected_output",
    [
        # Test case 1: Simple case
        (ops.array([[0.1, 0.9, 0.2, 0.8]]), 2, 1, ops.array([[0, 1, 0, 1]])),
        # Test case 2: Larger tensor with n_value_dims=1
        (
            ops.array([[0.1, 0.9, 0.2, 0.8], [0.5, 0.4, 0.6, 0.7]]),
            2,
            1,
            ops.array([[0, 1, 0, 1], [0, 0, 1, 1]]),
        ),
        # Test case 3: 3D tensor with n_value_dims=2
        (
            ops.array([[[0.1, 0.9], [0.2, 0.8]], [[0.5, 0.4], [0.6, 0.7]]]),
            2,
            2,
            ops.array([[[0, 1], [0, 1]], [[0, 0], [1, 1]]]),
        ),
        # Test case 4: Edge case with k=0
        (ops.array([[0.1, 0.9, 0.2, 0.8]]), 0, 1, ops.array([[0, 0, 0, 0]])),
    ],
)
def test_hard_straight_through(khot_orig, k, n_value_dims, expected_output):
    """Test the hard_straight_through function"""
    output = hard_straight_through(khot_orig, k, n_value_dims)
    assert np.allclose(
        output, expected_output
    ), f"Expected {expected_output}, but got {output}"
