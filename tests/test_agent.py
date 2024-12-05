"""Test agent functions."""

from keras import ops
from usbmd.agent.masks import generate_equispaced_lines


def test_generate_equispaced_lines():
    """Test generate_equispaced_lines."""
    expected_lines = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    lines = generate_equispaced_lines(n_actions=5, n_possible_actions=10)
    assert ops.all(lines == expected_lines)

    expected_lines = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    lines = generate_equispaced_lines(
        n_actions=5, n_possible_actions=10, previous_mask=lines
    )
    assert ops.all(lines == expected_lines)

    expected_lines = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    lines = generate_equispaced_lines(
        n_actions=5, n_possible_actions=10, previous_mask=lines
    )
    assert ops.all(lines == expected_lines)


def test_generate_equispaced_lines_assertion():
    """Test generate_equispaced_lines assertion."""
    # Should raise AssertionError when n_possible_actions is not divisible by n_actions
    try:
        generate_equispaced_lines(n_actions=3, n_possible_actions=10)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Should not raise error when n_possible_actions is divisible by n_actions
    try:
        generate_equispaced_lines(n_actions=2, n_possible_actions=10)
        generate_equispaced_lines(n_actions=5, n_possible_actions=10)
    except AssertionError:
        assert False, "Should not have raised AssertionError"


if __name__ == "__main__":
    test_generate_equispaced_lines()
    test_generate_equispaced_lines_assertion()
