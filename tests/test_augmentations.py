"""Tests for RandomCircleInclusion augmentation."""

import numpy as np
from keras import ops
from keras import random as keras_random

from zea.data.augmentations import RandomCircleInclusion

from . import DEFAULT_TEST_SEED


def assert_circle_pixels(image, center, radius, fill_value, tol=1e-5, min_fraction=0.9):
    """Check that pixels inside the circle are set to fill_value."""
    h, w = image.shape[-2:]
    cx, cy = int(round(center[0])), int(round(center[1]))
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius**2

    inside = image[mask]
    correct = np.isclose(inside, fill_value, atol=tol)
    fraction = np.sum(correct) / correct.size if correct.size > 0 else 1.0

    assert fraction >= min_fraction, (
        f"Only {fraction:.2%} of pixels inside circle match fill_value."
    )


def test_random_circle_inclusion_2d_with_batch():
    """Test 2D batch augmentation."""
    images = np.zeros((4, 28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(radius=5, fill_value=1.0, circle_axes=(1, 2), with_batch_dim=True)
    seed = keras_random.SeedGenerator(DEFAULT_TEST_SEED)
    out = layer(ops.convert_to_tensor(images), seed=seed)
    out_np = ops.convert_to_numpy(out)
    assert out_np.shape == images.shape
    assert np.all([np.any(np.isclose(im, 1.0)) for im in out_np])


def test_random_circle_inclusion_2d_no_batch():
    """Test 2D single image augmentation."""
    image = np.zeros((28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5, fill_value=1.0, circle_axes=(0, 1), with_batch_dim=False
    )
    seed = keras_random.SeedGenerator(DEFAULT_TEST_SEED)
    out = layer(ops.convert_to_tensor(image), seed=seed)
    out_np = ops.convert_to_numpy(out)
    assert out_np.shape == image.shape
    assert np.any(np.isclose(out_np, 1.0))


def test_random_circle_inclusion_3d_with_batch():
    """Test 3D batch augmentation."""
    images = np.zeros((2, 8, 28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5,
        fill_value=1.0,
        circle_axes=(2, 3),
        with_batch_dim=True,
    )
    seed = keras_random.SeedGenerator(DEFAULT_TEST_SEED)
    out = layer(ops.convert_to_tensor(images), seed=seed)
    out_np = ops.convert_to_numpy(out)
    assert out_np.shape == images.shape
    assert np.all([np.any(np.isclose(im, 1.0)) for im in out_np.reshape(-1, 28, 28)])


def test_random_circle_inclusion_3d_no_batch():
    """Test 3D single image augmentation."""
    image = np.zeros((8, 28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5,
        fill_value=1.0,
        circle_axes=(1, 2),
        with_batch_dim=False,
    )
    seed = keras_random.SeedGenerator(DEFAULT_TEST_SEED)
    out = layer(ops.convert_to_tensor(image), seed=seed)
    out_np = ops.convert_to_numpy(out)
    assert out_np.shape == image.shape
    assert np.all([np.any(np.isclose(im, 1.0)) for im in out_np])


def test_random_circle_inclusion_2d_with_batch_centers():
    """Test 2D batch augmentation with returned centers."""
    images = np.zeros((4, 28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5,
        fill_value=1.0,
        circle_axes=(1, 2),
        with_batch_dim=True,
        return_centers=True,
    )
    seed = keras_random.SeedGenerator(DEFAULT_TEST_SEED)
    out, centers = layer(ops.convert_to_tensor(images), seed=seed)
    out_np = ops.convert_to_numpy(out)
    centers_np = ops.convert_to_numpy(centers)
    assert out_np.shape == images.shape
    assert centers_np.shape == (images.shape[0], 2)
    for img, (cx, cy) in zip(out_np, centers_np):
        assert_circle_pixels(img, (cx, cy), 5, 1.0)


def test_random_circle_inclusion_2d_no_batch_centers():
    """Test 2D single image augmentation with returned center."""
    image = np.zeros((28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5,
        fill_value=1.0,
        circle_axes=(0, 1),
        with_batch_dim=False,
        return_centers=True,
    )
    seed = keras_random.SeedGenerator(DEFAULT_TEST_SEED)
    out, center = layer(ops.convert_to_tensor(image), seed=seed)
    out_np = ops.convert_to_numpy(out)
    center_np = ops.convert_to_numpy(center)
    assert out_np.shape == image.shape
    assert center_np.shape == (2,)
    assert_circle_pixels(out_np, center_np, 5, 1.0)


def test_evaluate_recovered_circle_accuracy_2d_with_batch_centers():
    """Test recovery accuracy for 2D batch with centers."""
    images = np.zeros((4, 28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5,
        fill_value=1.0,
        circle_axes=(1, 2),
        with_batch_dim=True,
        return_centers=True,
    )
    seed = keras_random.SeedGenerator(DEFAULT_TEST_SEED)
    out, centers = layer(ops.convert_to_tensor(images), seed=seed)
    acc, _ = layer.evaluate_recovered_circle_accuracy(out, centers, recovery_threshold=1e-5)
    assert np.all(np.isclose(acc, 1.0)), f"Expected 1.0, got {acc}"


def test_evaluate_recovered_circle_accuracy_3d_with_batch_centers():
    """Test recovery accuracy for 3D batch with centers."""
    images = np.zeros((2, 8, 28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5,
        fill_value=1.0,
        circle_axes=(2, 3),
        with_batch_dim=True,
        return_centers=True,
    )
    seed = keras_random.SeedGenerator(DEFAULT_TEST_SEED)
    out, centers = layer(ops.convert_to_tensor(images), seed=seed)
    acc, _ = layer.evaluate_recovered_circle_accuracy(out, centers, recovery_threshold=1e-5)
    assert np.all(np.isclose(acc, 1.0)), f"Expected 1.0, got {acc}"


def test_evaluate_recovered_circle_accuracy_3d_no_batch_centers():
    """Test recovery accuracy for 3D single image with centers."""
    image = np.zeros((8, 28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5,
        fill_value=1.0,
        circle_axes=(1, 2),
        with_batch_dim=False,
        return_centers=True,
    )
    seed = keras_random.SeedGenerator(DEFAULT_TEST_SEED)
    out, centers = layer(ops.convert_to_tensor(image), seed=seed)
    acc, _ = layer.evaluate_recovered_circle_accuracy(out, centers, recovery_threshold=1e-5)
    assert np.all(acc > 0.90), f"Expected circle recovery accuracy of  >0.90, got {acc}"


def test_random_circle_inclusion_batch_vs_no_batch_randomization():
    """Test that randomize_location_across_batch controls per-batch randomization."""
    images = np.zeros((3, 16, 16), dtype=np.float32)
    # With randomize_location_across_batch=True, each image gets a different center
    layer_random = RandomCircleInclusion(
        radius=3,
        fill_value=1.0,
        circle_axes=(1, 2),
        with_batch_dim=True,
        return_centers=True,
        randomize_location_across_batch=True,
    )
    seed = keras_random.SeedGenerator(33)
    out_rand, centers_rand = layer_random(ops.convert_to_tensor(images), seed=seed)
    centers_rand_np = ops.convert_to_numpy(centers_rand)
    # All centers should not be the same
    assert len({tuple(c) for c in centers_rand_np}) > 1, "Centers should differ across batch"

    # With randomize_location_across_batch=False, all images get the same center
    layer_fixed = RandomCircleInclusion(
        radius=3,
        fill_value=1.0,
        circle_axes=(1, 2),
        with_batch_dim=True,
        return_centers=True,
        randomize_location_across_batch=False,
    )
    out_fixed, centers_fixed = layer_fixed(ops.convert_to_tensor(images), seed=seed)
    centers_fixed_np = ops.convert_to_numpy(centers_fixed)
    # All centers should be the same
    assert np.all(centers_fixed_np == centers_fixed_np[0]), (
        "Centers should be identical across batch"
    )

    # With with_batch_dim=False, the input is treated as a single volume,
    # and returns a single center (2,) not (3, 2)
    layer_no_batch = RandomCircleInclusion(
        radius=3,
        fill_value=1.0,
        circle_axes=(1, 2),
        with_batch_dim=False,
        return_centers=True,
    )
    out_no_batch, centers_no_batch = layer_no_batch(ops.convert_to_tensor(images), seed=seed)
    centers_no_batch_np = ops.convert_to_numpy(centers_no_batch)
    # Should return centers with shape (3, 2) - one center per "slice" along axis 0
    # All slices share the same circle location
    assert centers_no_batch_np.shape == (3, 2), (
        f"Expected shape (3, 2), got {centers_no_batch_np.shape}"
    )
    assert np.all(centers_no_batch_np == centers_no_batch_np[0]), (
        "All slices should have identical centers when with_batch_dim=False"
    )


def test_evaluate_recovered_circle_accuracy_partial_recovery():
    """Test partial recovery accuracy."""
    image = np.zeros((28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5,
        fill_value=1.0,
        circle_axes=(0, 1),
        with_batch_dim=False,
    )
    center = (14, 14)
    Y, X = np.ogrid[:28, :28]
    mask = (X - center[0]) ** 2 + (Y - center[1]) ** 2 <= 5**2
    mask_indices = np.argwhere(mask)
    half = len(mask_indices) // 2
    for idx in mask_indices[:half]:
        image[tuple(idx)] = 1.0
    acc, _ = layer.evaluate_recovered_circle_accuracy(image, center, recovery_threshold=1e-5)
    assert 0.4 < acc < 0.6, f"Expected ~0.5, got {acc}"


def test_random_circle_inclusion_with_height_width_ranges():
    """Test 2D augmentation with width_range and height_range parameters."""
    image = np.zeros((28, 28), dtype=np.float32)
    # Restrict center to a small region
    width_range = (10, 12)
    height_range = (15, 17)
    layer = RandomCircleInclusion(
        radius=5,
        fill_value=1.0,
        circle_axes=(0, 1),
        with_batch_dim=False,
        return_centers=True,
        width_range=width_range,
        height_range=height_range,
    )
    seed = keras_random.SeedGenerator(123)
    out, center = layer(ops.convert_to_tensor(image), seed=seed)
    out_np = ops.convert_to_numpy(out)
    center_np = ops.convert_to_numpy(center)
    assert out_np.shape == image.shape
    assert center_np.shape == (2,)
    # Check that the center is within the specified ranges
    assert width_range[0] <= center_np[0] < width_range[1]
    assert height_range[0] <= center_np[1] < height_range[1]
    assert_circle_pixels(out_np, center_np, 5, 1.0)
