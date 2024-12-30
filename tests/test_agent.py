"""Test agent functions."""

import pytest
import numpy as np
from keras import ops
from usbmd.agent.masks import equispaced_lines
import usbmd.agent.selection as selection


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
    mask = agent.sample(particles)
    assert mask.shape == (1, h, w)
    first_row = mask[0, 0]
    assert np.count_nonzero(first_row) == n_actions

    n_actions = 2
    agent = selection.GreedyEntropy(n_actions, w, h, w)
    mask = agent.sample(particles)
    assert mask.shape == (1, h, w)
    first_row = mask[0, 0]
    assert np.count_nonzero(first_row) == n_actions


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
    mask = agent.sample(particles)
    assert mask.shape == (1, h, w)
    first_row = mask[0, 0]
    assert np.count_nonzero(first_row) == 1

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
    mask = agent.sample(particles)
    assert mask.shape == (1, h, w)
    first_row = mask[0, 0]
    assert np.count_nonzero(first_row) == w

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


if __name__ == "__main__":
    pytest.main()
