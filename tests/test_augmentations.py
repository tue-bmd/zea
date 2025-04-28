import numpy as np
from keras import ops
from usbmd.data.augmentations import RandomCircleInclusion


def assert_circle_pixels(image, center, radius, fill_value, tol=1e-5):
    """Assert that pixels within the circle are set to fill_value."""
    h, w = image.shape[-2:]
    cx, cy = center
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius**2
    # All pixels inside the circle should be close to fill_value
    assert np.allclose(image[mask], fill_value, atol=tol)
    # At least some pixels outside the circle should not be fill_value
    if np.any(~mask):
        assert not np.allclose(image[~mask], fill_value, atol=tol)


def test_random_circle_inclusion_2d_with_batch():
    # Create a dummy 2D image batch (batch, height, width)
    images = np.zeros((4, 28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5, fill_value=1.0, circle_axes=(1, 2), with_batch_dim=True
    )
    out = layer(ops.convert_to_tensor(images))
    out_np = ops.convert_to_numpy(out)
    # Output shape should match input
    assert out_np.shape == images.shape
    # Should have some pixels set to fill_value in each image
    assert np.all([np.any(np.isclose(im, 1.0)) for im in out_np])


def test_random_circle_inclusion_2d_no_batch():
    # Create a dummy 2D image (height, width)
    image = np.zeros((28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5, fill_value=1.0, circle_axes=(0, 1), with_batch_dim=False
    )
    out = layer(ops.convert_to_tensor(image))
    out_np = ops.convert_to_numpy(out)
    # Output shape should match input
    assert out_np.shape == image.shape
    # Should have some pixels set to fill_value
    assert np.any(np.isclose(out_np, 1.0))


def test_random_circle_inclusion_3d_with_batch():
    # Create a dummy 3D image batch (batch, depth, height, width)
    images = np.zeros((2, 8, 28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5,
        fill_value=1.0,
        circle_axes=(2, 3),
        with_batch_dim=True,
    )
    out = layer(ops.convert_to_tensor(images))
    out_np = ops.convert_to_numpy(out)
    # Output shape should match input
    assert out_np.shape == images.shape
    # Should have some pixels set to fill_value in each image
    assert np.all([np.any(np.isclose(im, 1.0)) for im in out_np.reshape(-1, 28, 28)])


def test_random_circle_inclusion_3d_no_batch():
    # Create a dummy 3D image (depth, height, width)
    image = np.zeros((8, 28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5,
        fill_value=1.0,
        circle_axes=(1, 2),
        with_batch_dim=False,
    )
    out = layer(ops.convert_to_tensor(image))
    out_np = ops.convert_to_numpy(out)
    # Output shape should match input
    assert out_np.shape == image.shape
    # Should have some pixels set to fill_value in each slice
    assert np.all([np.any(np.isclose(im, 1.0)) for im in out_np])


def test_random_circle_inclusion_2d_with_batch_centers():
    images = np.zeros((4, 28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5,
        fill_value=1.0,
        circle_axes=(1, 2),
        with_batch_dim=True,
        return_centers=True,
    )
    out, centers = layer(ops.convert_to_tensor(images))
    out_np = ops.convert_to_numpy(out)
    centers_np = ops.convert_to_numpy(centers)
    assert out_np.shape == images.shape
    assert centers_np.shape == (images.shape[0], 2)
    for img, (cx, cy) in zip(out_np, centers_np):
        assert_circle_pixels(img, (cx, cy), 5, 1.0)


def test_random_circle_inclusion_2d_no_batch_centers():
    image = np.zeros((28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5,
        fill_value=1.0,
        circle_axes=(0, 1),
        with_batch_dim=False,
        return_centers=True,
    )
    out, center = layer(ops.convert_to_tensor(image))
    out_np = ops.convert_to_numpy(out)
    center_np = ops.convert_to_numpy(center)
    assert out_np.shape == image.shape
    assert center_np.shape == (2,)
    assert_circle_pixels(out_np, center_np, 5, 1.0)


def test_evaluate_recovered_circle_accuracy_2d_with_batch_centers():
    # Batched 2D images, with known centers
    images = np.zeros((4, 28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5,
        fill_value=1.0,
        circle_axes=(1, 2),
        with_batch_dim=True,
        return_centers=True,
    )
    out, centers = layer(ops.convert_to_tensor(images))
    out_np = ops.convert_to_numpy(out)
    centers_np = ops.convert_to_numpy(centers)
    for img, center in zip(out_np, centers_np):
        acc = layer.evaluate_recovered_circle_accuracy(img, center, threshold=1e-5)
        assert np.isclose(acc, 1.0), f"Expected 1.0, got {acc}"


def test_evaluate_recovered_circle_accuracy_3d_with_batch_centers():
    # Batched 3D images (batch, depth, height, width)
    images = np.zeros((2, 8, 28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5,
        fill_value=1.0,
        circle_axes=(2, 3),
        with_batch_dim=True,
        return_centers=True,
    )
    out, centers = layer(ops.convert_to_tensor(images))
    out_np = ops.convert_to_numpy(out)
    centers_np = ops.convert_to_numpy(centers)
    # centers_np shape: (batch*depth, 2) or (batch, depth, 2) depending on your implementation
    # out_np shape: (batch, depth, height, width)
    # We'll flatten batch and depth for simplicity
    flat_imgs = out_np.reshape(-1, 28, 28)
    flat_centers = centers_np.reshape(-1, 2)
    for img, center in zip(flat_imgs, flat_centers):
        acc = layer.evaluate_recovered_circle_accuracy(img, center, threshold=1e-5)
        assert np.isclose(acc, 1.0), f"Expected 1.0, got {acc}"


def test_evaluate_recovered_circle_accuracy_3d_no_batch_centers():
    # 3D image (depth, height, width)
    image = np.zeros((8, 28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5,
        fill_value=1.0,
        circle_axes=(1, 2),
        with_batch_dim=False,
        return_centers=True,
    )
    out, centers = layer(ops.convert_to_tensor(image))
    out_np = ops.convert_to_numpy(out)
    centers_np = ops.convert_to_numpy(centers)
    # out_np shape: (8, 28, 28), centers_np shape: (8, 2)
    for img, center in zip(out_np, centers_np):
        acc = layer.evaluate_recovered_circle_accuracy(img, center, threshold=1e-5)
        assert np.isclose(acc, 1.0), f"Expected 1.0, got {acc}"


def test_evaluate_recovered_circle_accuracy_partial_recovery():
    # Single 2D image, partial recovery
    image = np.zeros((28, 28), dtype=np.float32)
    layer = RandomCircleInclusion(
        radius=5,
        fill_value=1.0,
        circle_axes=(0, 1),
        with_batch_dim=False,
    )
    center = (14, 14)
    # Draw a partial circle at the center
    Y, X = np.ogrid[:28, :28]
    mask = (X - center[0]) ** 2 + (Y - center[1]) ** 2 <= 5**2
    mask_indices = np.argwhere(mask)
    half = len(mask_indices) // 2
    for idx in mask_indices[:half]:
        image[tuple(idx)] = 1.0
    acc = layer.evaluate_recovered_circle_accuracy(image, center, threshold=1e-5)
    assert 0.4 < acc < 0.6, f"Expected ~0.5, got {acc}"
