"""Test agent functions."""

import pytest
from keras import ops

from usbmd.agent.masks import equispaced_lines


def test_equispaced_lines():
    """Test equispaced_lines."""
    expected_lines = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    lines = equispaced_lines(n_actions=5, n_possible_actions=10)
    assert ops.all(lines == expected_lines)

    expected_lines = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    lines = equispaced_lines(n_actions=5, n_possible_actions=10, previous_mask=lines)
    assert ops.all(lines == expected_lines)

    expected_lines = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    lines = equispaced_lines(n_actions=5, n_possible_actions=10, previous_mask=lines)
    assert ops.all(lines == expected_lines)


def test_equispaced_lines_assertion():
    """Test equispaced_lines assertion."""
    # Should raise AssertionError when n_possible_actions is not divisible by n_actions
    with pytest.raises(AssertionError):
        equispaced_lines(n_actions=3, n_possible_actions=10)

    # Should not raise error when n_possible_actions is divisible by n_actions
    equispaced_lines(n_actions=2, n_possible_actions=10)
    equispaced_lines(n_actions=5, n_possible_actions=10)


if __name__ == "__main__":
    test_equispaced_lines()
    test_equispaced_lines_assertion()
