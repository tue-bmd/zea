"""Test agent functions."""

import numpy as np
import pytest
from keras import ops

from usbmd.agent import selection
from usbmd.agent.masks import equispaced_lines


def test_equispaced_lines():
    """Test equispaced_lines."""
    expected_lines = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    lines = equispaced_lines(n_actions=5, n_possible_actions=10)
    assert ops.all(lines == expected_lines)

    expected_lines = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    lines = equispaced_lines(n_actions=5, n_possible_actions=10, previous_mask=lines)
    assert ops.all(lines == expected_lines)

    expected_lines = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
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


def test_mask_action_model():
    """Test MaskActionModel."""
    model = selection.MaskActionModel()
    observation = ops.ones((2, 2))
    action = ops.eye(2)
    masked = model.apply(action, observation)
    expected_masked = ops.eye(2)
    assert ops.all(masked == expected_masked)


def test_lines_action_model():
    """Test LinesActionModel."""
    model = selection.LinesActionModel(
        n_actions=2, n_possible_actions=4, img_width=8, img_height=8
    )
    assert model.stack_n_cols == 2

    with pytest.raises(AssertionError):
        selection.LinesActionModel(
            n_actions=2, n_possible_actions=3, img_width=8, img_height=8
        )


def test_greedy_entropy():
    """Test GreedyEntropy action selection."""
    np.random.seed(2)
    h, w = 8, 8
    rand_img_1 = np.random.rand(h, w, 1).astype(np.float32)
    rand_img_2 = np.random.rand(h, w, 1).astype(np.float32)

    # manually make lines 2 and 3 very correlated
    rand_img_1[:, 2] = rand_img_1[:, 3]
    rand_img_2[:, 2] = rand_img_2[:, 3]

    particles = np.stack([rand_img_1, rand_img_2], axis=0)[:, None]
    particles = np.squeeze(particles, axis=-1)  # shape (n_particles, 1, h, w)

    n_actions = 1
    agent = selection.GreedyEntropy(n_actions, w, h, w)
    selected_lines, mask = agent.sample(particles)
    assert mask.shape == (1, h, w)
    assert selected_lines.shape == (1, w)
    first_row = mask[0, 0]
    assert np.count_nonzero(first_row) == n_actions
    assert np.count_nonzero(selected_lines[0]) == n_actions

    n_actions = 2
    agent = selection.GreedyEntropy(n_actions, w, h, w)
    selected_lines, mask = agent.sample(particles)
    assert mask.shape == (1, h, w)
    assert selected_lines.shape == (1, w)
    first_row = mask[0, 0]
    assert np.count_nonzero(first_row) == n_actions
    assert np.count_nonzero(selected_lines[0]) == n_actions


def test_covariance_sampling_lines():
    """Test CovarianceSamplingLines action selection."""
    np.random.seed(2)
    h, w = 8, 8
    rand_img_1 = np.random.rand(h, w, 1).astype(np.float32)
    rand_img_2 = np.random.rand(h, w, 1).astype(np.float32)

    # manually make lines 2 and 3 very correlated
    rand_img_1[:, 2] = rand_img_1[:, 3]
    rand_img_2[:, 2] = rand_img_2[:, 3]

    particles = np.stack([rand_img_1, rand_img_2], axis=0)[:, None]
    particles = np.squeeze(particles, axis=-1)  # shape (n_particles, 1, h, w)

    n_actions = 1
    agent = selection.CovarianceSamplingLines(n_actions, w, h, w, n_masks=200)
    mask = agent.sample(particles)
    assert mask.shape == (1, h, w)
    first_row = mask[0, 0]
    assert np.count_nonzero(first_row) == n_actions

    n_actions = 2
    agent = selection.CovarianceSamplingLines(n_actions, w, h, w, n_masks=200)
    mask = agent.sample(particles)
    assert mask.shape == (1, h, w)
    first_row = mask[0, 0]
    first_row = mask[0, 0]
    assert np.count_nonzero(first_row) == n_actions


def test_single_action():
    """Test single action."""
    np.random.seed(2)
    h, w = 8, 8
    particles = np.random.rand(2, 1, h, w).astype(np.float32)

    agent = selection.GreedyEntropy(1, w, h, w)
    selected_lines, mask = agent.sample(particles)
    assert mask.shape == (1, h, w)
    assert selected_lines.shape == (1, w)
    first_row = mask[0, 0]
    assert np.count_nonzero(first_row) == 1
    assert np.count_nonzero(selected_lines[0]) == 1

    agent = selection.CovarianceSamplingLines(1, w, h, w, n_masks=200)
    mask = agent.sample(particles)
    assert mask.shape == (1, h, w)
    first_row = mask[0, 0]
    assert np.count_nonzero(first_row) == 1


def test_maximum_actions():
    """Test maximum actions."""
    np.random.seed(2)
    h, w = 8, 8
    particles = np.random.rand(2, 1, h, w).astype(np.float32)

    agent = selection.GreedyEntropy(w, w, h, w)
    selected_lines, mask = agent.sample(particles)
    assert mask.shape == (1, h, w)
    assert selected_lines.shape == (1, w)
    first_row = mask[0, 0]
    assert np.count_nonzero(first_row) == w
    assert np.count_nonzero(selected_lines[0]) == w

    agent = selection.CovarianceSamplingLines(w, w, h, w, n_masks=200)
    mask = agent.sample(particles)
    assert mask.shape == (1, h, w)
    first_row = mask[0, 0]
    assert np.count_nonzero(first_row) == w


def test_non_divisible_actions():
    """Test non-divisible actions."""
    with pytest.raises(AssertionError):
        selection.GreedyEntropy(3, 10, 8, 8)
        selection.CovarianceSamplingLines(3, 10, 8, 8, n_masks=200)


def test_equispaced_lines_class():
    """Test EquispacedLines class."""
    b, h, w = 3, 8, 8  # batch_size=3

    # Test with 2 actions
    n_actions = 2
    agent = selection.EquispacedLines(n_actions, w, w, h, batch_size=b)
    mask = agent.sample()

    # Check mask shape (should include batch dimension)
    assert mask.shape == (b, h, w)

    # Check first row has correct number of ones for each batch
    for batch_idx in range(b):
        first_row = mask[batch_idx, 0]
        assert np.count_nonzero(first_row) == n_actions

    # Test successive calls return different but valid patterns
    mask1 = agent.sample()
    mask2 = agent.sample()

    # Masks should be different (alternating pattern) for each batch
    assert not np.array_equal(mask1, mask2)

    # Both should have correct number of actions for each batch
    for batch_idx in range(b):
        assert np.count_nonzero(mask1[batch_idx, 0]) == n_actions
        assert np.count_nonzero(mask2[batch_idx, 0]) == n_actions

        # Check that batch elements have the same pattern within a single call
        assert np.array_equal(mask1[0], mask1[batch_idx])
        assert np.array_equal(mask2[0], mask2[batch_idx])

    # Test with maximum number of actions
    agent = selection.EquispacedLines(w, w, h, w, batch_size=b)
    mask = agent.sample()
    for batch_idx in range(b):
        assert np.count_nonzero(mask[batch_idx, 0]) == w

    # Test with non-divisible actions (should raise AssertionError)
    with pytest.raises(AssertionError):
        selection.EquispacedLines(3, 10, h, w, batch_size=b)


def test_uniform_random_lines():
    """Test UniformRandomLines action selection."""
    np.random.seed(2)
    h, w = 8, 8
    batch_size = 3

    # Test with single action
    n_actions = 1
    agent = selection.UniformRandomLines(n_actions, w, h, w, batch_size=batch_size)
    selected_lines, mask = agent.sample()
    assert mask.shape == (batch_size, h, w)
    assert selected_lines.shape == (batch_size, w)

    # Check each batch has correct number of actions
    for b in range(batch_size):
        first_row = mask[b, 0]
        assert np.count_nonzero(first_row) == n_actions
        assert np.count_nonzero(selected_lines[b]) == n_actions

    # Test with multiple actions
    n_actions = 2
    agent = selection.UniformRandomLines(n_actions, w, h, w, batch_size=batch_size)
    selected_lines, mask = agent.sample()
    assert mask.shape == (batch_size, h, w)
    assert selected_lines.shape == (batch_size, w)

    # Check each batch has correct number of actions
    for b in range(batch_size):
        first_row = mask[b, 0]
        assert np.count_nonzero(first_row) == n_actions
        assert np.count_nonzero(selected_lines[b]) == n_actions

    # Test with maximum actions
    agent = selection.UniformRandomLines(w, w, h, w, batch_size=batch_size)
    selected_lines, mask = agent.sample()
    assert mask.shape == (batch_size, h, w)
    assert selected_lines.shape == (batch_size, w)

    # Check each batch has correct number of actions
    for b in range(batch_size):
        first_row = mask[b, 0]
        assert np.count_nonzero(first_row) == w
        assert np.count_nonzero(selected_lines[b]) == w

    # Test with non-divisible actions (should raise AssertionError)
    with pytest.raises(AssertionError):
        selection.UniformRandomLines(3, 10, h, w, batch_size=batch_size)


if __name__ == "__main__":
    test_equispaced_lines_class()
    pytest.main()
