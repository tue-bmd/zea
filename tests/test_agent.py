"""Test agent functions."""

import numpy as np
import pytest
from keras import ops

from zea.agent import masks, selection

from . import DEFAULT_TEST_SEED


def test_equispaced_lines():
    """Test equispaced_lines."""
    expected_lines = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    lines = masks.initial_equispaced_lines(n_actions=5, n_possible_actions=10)
    assert ops.all(lines == expected_lines)

    expected_lines = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    lines = masks.next_equispaced_lines(lines)
    assert ops.all(lines == expected_lines)

    expected_lines = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    lines = masks.next_equispaced_lines(lines)
    assert ops.all(lines == expected_lines)


def test_equispaced_lines_assertion():
    """Test equispaced_lines assertion."""
    # Should raise AssertionError when n_possible_actions is not divisible by n_actions
    with pytest.raises(AssertionError):
        masks.initial_equispaced_lines(n_actions=3, n_possible_actions=10)

    # Should not raise error when n_possible_actions is divisible by n_actions
    masks.initial_equispaced_lines(n_actions=2, n_possible_actions=10)
    masks.initial_equispaced_lines(n_actions=5, n_possible_actions=10)


def test_unequal_spacing():
    """Test equispaced_lines with unequal spacing."""
    # Should not raise error when n_possible_actions is not divisible by n_actions
    lines = masks.initial_equispaced_lines(
        n_actions=3, n_possible_actions=10, assert_equal_spacing=False
    )
    assert ops.shape(lines) == (10,)
    assert ops.sum(lines) == 3

    # Should not raise error when n_possible_actions is divisible by n_actions
    lines = masks.initial_equispaced_lines(
        n_actions=2, n_possible_actions=10, assert_equal_spacing=False
    )
    assert ops.shape(lines) == (10,)
    assert ops.sum(lines) == 2


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
    model = selection.LinesActionModel(n_actions=2, n_possible_actions=4, img_width=8, img_height=8)
    assert model.stack_n_cols == 2

    with pytest.raises(AssertionError):
        selection.LinesActionModel(n_actions=2, n_possible_actions=3, img_width=8, img_height=8)


def test_greedy_entropy():
    """Test GreedyEntropy action selection."""
    # Note: this test is hard-coded to work with rng seed 2, seed should not be a variable.
    np.random.seed(2)
    h, w = 8, 8
    rand_img_1 = np.random.rand(h, w, 1).astype(np.float32)
    rand_img_2 = np.random.rand(h, w, 1).astype(np.float32)

    # manually make lines 2 and 3 very correlated
    rand_img_1[:, 2] = rand_img_1[:, 3]
    rand_img_2[:, 2] = rand_img_2[:, 3]

    particles = np.stack([rand_img_1, rand_img_2], axis=0)
    particles = np.expand_dims(particles, axis=0)  # add batch dim
    particles = np.squeeze(particles, axis=-1)  # remove channel dim --> (batch, n_particles, h, w)

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

    # Test that the algorithm hasn't changed by comparing to a correct hard-coded value
    h, w = 64, 64
    rand_img_1 = np.random.rand(h, w, 1).astype(np.float32)
    rand_img_2 = np.random.rand(h, w, 1).astype(np.float32)

    # manually make lines 2 and 3 very correlated
    rand_img_1[:, 2] = rand_img_1[:, 3]
    rand_img_2[:, 2] = rand_img_2[:, 3]

    particles = np.stack([rand_img_1, rand_img_2], axis=0)
    particles = np.expand_dims(particles, axis=0)
    particles = np.squeeze(particles, axis=-1)

    n_actions = 1
    agent = selection.GreedyEntropy(n_actions, w, h, w)
    selected_lines, mask = agent.sample(particles)

    correct_line_index = 17
    correct_selected_lines = [False] * 64
    correct_selected_lines[correct_line_index] = True
    correct_selected_lines = [correct_selected_lines]
    assert np.all(selected_lines == correct_selected_lines)

    # Test compute_pairwise_pixel_gaussian_error with n_possible_actions = None
    _ = agent.compute_pairwise_pixel_gaussian_error(particles, n_possible_actions=None)


def test_greedy_entropy_average_across_batch():
    """Test GreedyEntropy with average_entropy_across_batch=True for 3D plane selection."""
    np.random.seed(42)
    h, w = 8, 8
    batch_size = 3
    n_particles = 2
    n_actions = 1

    particles = np.random.rand(batch_size, n_particles, h, w).astype(np.float32)

    # Verify it runs without error when averaging across batch
    agent = selection.GreedyEntropy(n_actions, w, h, w, average_entropy_across_batch=True)
    selected_lines, mask = agent.sample(particles)

    assert mask.shape == (1, h, w)
    assert selected_lines.shape == (1, w)


def test_covariance_sampling_lines():
    """Test CovarianceSamplingLines action selection."""
    # Note: this test is hard-coded to work with rng seed 2, seed should not be a variable.
    rng = np.random.default_rng(2)
    h, w = 16, 16
    rand_img_1 = rng.uniform(0, 1, (h, w)).astype(np.float32)
    rand_img_2 = rng.uniform(0, 1, (h, w)).astype(np.float32)

    # manually make lines 2 and 3 very correlated
    rand_img_1[:, 2] = rand_img_1[:, 3]
    rand_img_2[:, 2] = rand_img_2[:, 3]

    # manually add a line on the right edge (i.e. high variance with eachother)
    rand_img_1[:, -1] = 1.0
    rand_img_2[:, -1] = 0.0

    particles = np.stack([rand_img_1, rand_img_2], axis=0)[None]  # (batch, n_particles, h, w)

    # CASE 1: Single action
    n_actions = 1
    agent = selection.CovarianceSamplingLines(n_actions, w, h, w, n_masks=200)
    selected_lines, mask = agent.sample(particles)
    assert selected_lines.ndim == 2
    selected_lines = ops.squeeze(selected_lines, axis=0)  # remove batch dim
    assert mask.shape == (1, h, w)
    assert np.count_nonzero(selected_lines) == n_actions

    # regression
    assert 15 in np.flatnonzero(selected_lines)

    # CASE 2: Two actions
    n_actions = 2
    agent = selection.CovarianceSamplingLines(n_actions, w, h, w, n_masks=200)
    selected_lines, mask = agent.sample(particles)
    assert selected_lines.ndim == 2
    selected_lines = ops.squeeze(selected_lines, axis=0)  # remove batch dim
    assert mask.shape == (1, h, w)
    assert np.count_nonzero(selected_lines) == n_actions

    # regression
    assert 15 in np.flatnonzero(selected_lines) and 0 in np.flatnonzero(selected_lines)


def test_single_action():
    """Test single action."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    h, w = 8, 8
    particles = rng.standard_normal((1, 2, h, w)).astype(np.float32)

    agent = selection.GreedyEntropy(1, w, h, w)
    selected_lines, mask = agent.sample(particles)
    assert mask.shape == (1, h, w)
    assert selected_lines.shape == (1, w)
    first_row = mask[0, 0]
    assert np.count_nonzero(first_row) == 1
    assert np.count_nonzero(selected_lines[0]) == 1

    agent = selection.CovarianceSamplingLines(1, w, h, w, n_masks=200)
    _, mask = agent.sample(particles)
    assert mask.shape == (1, h, w)
    first_row = mask[0, 0]
    assert np.count_nonzero(first_row) == 1


def test_maximum_actions():
    """Test maximum actions."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    h, w = 8, 8
    particles = rng.random((1, 2, h, w)).astype(np.float32)

    agent = selection.GreedyEntropy(w, w, h, w)
    selected_lines, mask = agent.sample(particles)
    assert mask.shape == (1, h, w)
    assert selected_lines.shape == (1, w)
    first_row = mask[0, 0]
    assert np.count_nonzero(first_row) == w
    assert np.count_nonzero(selected_lines[0]) == w

    agent = selection.CovarianceSamplingLines(w, w, h, w, n_masks=200)
    _, mask = agent.sample(particles)
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
    agent = selection.EquispacedLines(n_actions, w, w, h)
    _, mask = agent.sample(batch_size=b)

    # Check mask shape (should include batch dimension)
    assert mask.shape == (b, h, w)

    # Check first row has correct number of ones for each batch
    for batch_idx in range(b):
        first_row = mask[batch_idx, 0]
        assert np.count_nonzero(first_row) == n_actions

    # Test successive calls return different but valid patterns
    lines1, mask1 = agent.sample(batch_size=b)
    _, mask2 = agent.sample(lines1)

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
    agent = selection.EquispacedLines(w, w, h, w)
    _, mask = agent.sample(batch_size=b)
    for batch_idx in range(b):
        assert np.count_nonzero(mask[batch_idx, 0]) == w

    # Test with non-divisible actions (should raise AssertionError)
    with pytest.raises(AssertionError):
        selection.EquispacedLines(3, 10, h, w)


def test_uniform_random_lines():
    """Test UniformRandomLines action selection."""
    h, w = 8, 8
    batch_size = 3

    # Test with single action
    n_actions = 1
    agent = selection.UniformRandomLines(n_actions, w, h, w)
    selected_lines, mask = agent.sample(batch_size=batch_size)
    assert mask.shape == (batch_size, h, w)
    assert selected_lines.shape == (batch_size, w)

    # Check each batch has correct number of actions
    for b in range(batch_size):
        first_row = mask[b, 0]
        assert np.count_nonzero(first_row) == n_actions
        assert np.count_nonzero(selected_lines[b]) == n_actions

    # Test with multiple actions
    n_actions = 2
    agent = selection.UniformRandomLines(n_actions, w, h, w)
    selected_lines, mask = agent.sample(batch_size=batch_size)
    assert mask.shape == (batch_size, h, w)
    assert selected_lines.shape == (batch_size, w)

    # Check each batch has correct number of actions
    for b in range(batch_size):
        first_row = mask[b, 0]
        assert np.count_nonzero(first_row) == n_actions
        assert np.count_nonzero(selected_lines[b]) == n_actions

    # Test with maximum actions
    agent = selection.UniformRandomLines(w, w, h, w)
    selected_lines, mask = agent.sample(batch_size=batch_size)
    assert mask.shape == (batch_size, h, w)
    assert selected_lines.shape == (batch_size, w)

    # Check each batch has correct number of actions
    for b in range(batch_size):
        first_row = mask[b, 0]
        assert np.count_nonzero(first_row) == w
        assert np.count_nonzero(selected_lines[b]) == w

    # Test with non-divisible actions (should raise AssertionError)
    with pytest.raises(AssertionError):
        selection.UniformRandomLines(3, 10, h, w)


def test_task_based_lines():
    """Test TaskBasedLines action selection."""
    # Note: this test is hard-coded to work with rng seed 2, seed should not be a variable.
    np.random.seed(2)
    h, w = 8, 8
    rand_img_1 = np.random.rand(h, w, 1).astype(np.float32)
    rand_img_2 = np.random.rand(h, w, 1).astype(np.float32)

    # manually make lines 2 and 3 very correlated
    rand_img_1[:, 2] = rand_img_1[:, 3]
    rand_img_2[:, 2] = rand_img_2[:, 3]

    particles = np.stack([rand_img_1, rand_img_2], axis=0)
    particles = np.expand_dims(particles, axis=0)  # add batch dim
    particles = np.squeeze(particles, axis=-1)  # remove channel dim --> (batch, n_particles, h, w)

    # Define a simple downstream task function: sum of squared pixel values
    def downstream_task_fn(x):
        return ops.sum(x**2)

    n_actions = 1
    agent = selection.TaskBasedLines(n_actions, w, h, w, downstream_task_fn)
    selected_lines, mask, pixelwise_contribution = agent.sample(particles)

    # Test output shapes
    assert mask.shape == (1, h, w)
    assert selected_lines.shape == (1, w)
    assert pixelwise_contribution.shape == (1, h, w)

    # Test that correct number of lines are selected
    first_row = mask[0, 0]
    assert np.count_nonzero(first_row) == n_actions
    assert np.count_nonzero(selected_lines[0]) == n_actions

    # Test multiple actions
    n_actions = 2
    agent = selection.TaskBasedLines(n_actions, w, h, w, downstream_task_fn)
    selected_lines, mask, pixelwise_contribution = agent.sample(particles)

    assert mask.shape == (1, h, w)
    assert selected_lines.shape == (1, w)
    assert pixelwise_contribution.shape == (1, h, w)

    first_row = mask[0, 0]
    assert np.count_nonzero(first_row) == n_actions
    assert np.count_nonzero(selected_lines[0]) == n_actions

    # Test that pixelwise contribution values are non-negative (variance * squared gradient)
    assert np.all(pixelwise_contribution >= 0)

    # Test with batch size > 1
    batch_size = 3
    # Create particles for multiple batches
    particles_batch = np.tile(particles, (batch_size, 1, 1, 1))  # (batch_size, n_particles, h, w)

    n_actions = 1
    agent = selection.TaskBasedLines(n_actions, w, h, w, downstream_task_fn)
    selected_lines_batch, mask_batch, pixelwise_contribution_batch = agent.sample(particles_batch)

    # Test batch output shapes
    assert mask_batch.shape == (batch_size, h, w)
    assert selected_lines_batch.shape == (batch_size, w)
    assert pixelwise_contribution_batch.shape == (batch_size, h, w)

    # Test that correct number of lines are selected for each batch
    for b in range(batch_size):
        first_row = mask_batch[b, 0]
        assert np.count_nonzero(first_row) == n_actions
        assert np.count_nonzero(selected_lines_batch[b]) == n_actions
        # All pixelwise contributions should be non-negative
        assert np.all(pixelwise_contribution_batch[b] >= 0)

    # Test with a different downstream task function: mean pixel value
    def mean_task_fn(x):
        return ops.mean(x)

    agent_mean = selection.TaskBasedLines(n_actions, w, h, w, mean_task_fn)
    selected_lines_mean, mask_mean, pixelwise_contribution_mean = agent_mean.sample(particles)

    # Should have same shapes
    assert mask_mean.shape == (1, h, w)
    assert selected_lines_mean.shape == (1, w)
    assert pixelwise_contribution_mean.shape == (1, h, w)
    assert np.count_nonzero(selected_lines_mean[0]) == n_actions

    # regression tests
    assert 2 in np.flatnonzero(selected_lines)
    assert 6 in np.flatnonzero(selected_lines)
