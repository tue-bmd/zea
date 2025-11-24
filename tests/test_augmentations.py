"""Tests for RandomCircleInclusion augmentation."""

import numpy as np
from keras import ops
from keras import random as keras_random

from zea.data.augmentations import RandomCircleInclusion

from . import DEFAULT_TEST_SEED


def assert_circle_pixels(image, center, radius, fill_value, tol=1e-5):
    """Check that pixels inside the circle are set to fill_value."""
    h, w = image.shape[-2:]
    cx, cy = center
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius**2
    assert np.allclose(image[mask], fill_value, atol=tol)
    if np.any(~mask):
        assert not np.allclose(image[~mask], fill_value, atol=tol)


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
    acc = layer.evaluate_recovered_circle_accuracy(out, centers, recovery_threshold=1e-5)
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
    acc = layer.evaluate_recovered_circle_accuracy(out, centers, recovery_threshold=1e-5)
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
    acc = layer.evaluate_recovered_circle_accuracy(out, centers, recovery_threshold=1e-5)
    assert np.all(np.isclose(acc, 1.0)), f"Expected 1.0, got {acc}"


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
    acc = layer.evaluate_recovered_circle_accuracy(image, center, recovery_threshold=1e-5)
    assert 0.4 < acc < 0.6, f"Expected ~0.5, got {acc}"
