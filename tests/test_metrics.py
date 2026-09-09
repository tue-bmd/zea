"""Module for testing loss functions"""

import inspect

import numpy as np
import pytest
from keras import ops

from zea import metrics
from zea.internal.registry import metrics_registry

from . import DEFAULT_TEST_SEED, backend_equality_check
from .backend_utils import runs_on


@backend_equality_check(decimal=3)
def test_smsle():
    """Test SMSLE loss function"""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    y_true = rng.standard_normal((2, 32, 32, 1)).astype(np.float32)
    y_pred = rng.standard_normal((2, 32, 32, 1)).astype(np.float32)

    loss = metrics.smsle(y_true, y_pred)

    # Loss reduces to a scalar, is non-negative, and is zero for identical inputs
    assert loss.shape == ()
    assert float(loss) > 0
    assert float(metrics.smsle(y_true, y_true)) == pytest.approx(0.0, abs=1e-5)

    # The metric is scale invariant: both inputs are normalized by their own maximum
    np.testing.assert_allclose(
        ops.convert_to_numpy(metrics.smsle(y_true * 10, y_pred)),
        ops.convert_to_numpy(loss),
        rtol=1e-5,
        atol=1e-5,
    )

    # Sign matters: flipping the sign of the prediction changes the loss
    assert float(metrics.smsle(y_true, -y_true)) > 0

    # A smaller dynamic range compresses the errors, so the loss must not grow
    assert float(metrics.smsle(y_true, y_pred, dynamic_range=20)) < float(loss)

    return loss


@pytest.mark.parametrize("metric_name", metrics_registry.registered_names())
@backend_equality_check(decimal=3)
def test_metrics(metric_name):
    """Test all losses and metrics.
    Most metrics do not have a batch axis, so we test with single images."""
    if metric_name == "lpips":
        metric = metrics.get_metric(metric_name, image_range=[0, 255])
    else:
        metric = metrics.get_metric(metric_name)
    paired = metrics_registry.get_parameter(metric_name, "paired")

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    y_true = rng.uniform(0, 255, (16, 16, 3)).astype(np.float32)
    y_pred = rng.uniform(0, 255, (16, 16, 3)).astype(np.float32)
    y_true = ops.convert_to_tensor(y_true)
    y_pred = ops.convert_to_tensor(y_pred)

    if paired:
        metric_value = metric(y_true, y_pred)
    else:
        metric_value = metric(y_pred)

    assert metric_value.shape == (), f"Metric {metric_name} did not return a scalar value"

    # Regression test against the TensorFlow implementation of PSNR
    if metric_name == "psnr" and runs_on("tensorflow"):
        import tensorflow as tf

        expected_value = tf.image.psnr(
            ops.convert_to_numpy(y_true),
            ops.convert_to_numpy(y_pred),
            max_val=255.0,
        )
        np.testing.assert_allclose(metric_value, expected_value, rtol=1e-5, atol=1e-5)

    return metric_value


@backend_equality_check(decimal=2)
def test_metrics_class():
    """Test Metrics class, which computes multiple metrics at once on batched data."""
    batch_size = 2
    img_size = (16, 16, 3)

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    y_true = rng.uniform(0, 255, (batch_size, *img_size)).astype(np.float32)
    y_pred = rng.uniform(0, 255, (batch_size, *img_size)).astype(np.float32)
    y_true = ops.convert_to_tensor(y_true)
    y_pred = ops.convert_to_tensor(y_pred)

    METRIC_NAMES = ["mse", "psnr", "lpips"]  # ssim does not work with torch.vmap
    metrics_instance = metrics.Metrics(METRIC_NAMES, [0, 255])

    results = metrics_instance(y_true, y_pred, average_batches=True)
    assert all(name in results for name in METRIC_NAMES)
    assert all(np.isscalar(value.item()) for value in results.values())

    results_no_avg = metrics_instance(y_true, y_pred, average_batches=False)
    assert all(name in results_no_avg for name in METRIC_NAMES)
    assert all(value.shape == (batch_size,) for value in results_no_avg.values())

    # Compare backends for a single metric
    return results_no_avg["mse"]


@backend_equality_check(decimal=2)
def test_metrics_class_batch_size():
    """Test Metrics class with batch_size parameter"""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    y_true = rng.random((4, 16, 16, 3)).astype(np.float32) * 255.0
    y_pred = rng.random((4, 16, 16, 3)).astype(np.float32) * 255.0
    y_true = ops.convert_to_tensor(y_true)
    y_pred = ops.convert_to_tensor(y_pred)

    METRIC_NAMES = ["mse", "psnr", "lpips"]
    metrics_instance = metrics.Metrics(METRIC_NAMES, [0, 255])

    # Compute without batch_size (baseline)
    results_no_batch_size = metrics_instance(y_true, y_pred, average_batches=False)

    # Compute with batch_size=2 (should process in chunks)
    results_with_batch_size = metrics_instance(
        y_true, y_pred, average_batches=False, mapped_batch_size=2
    )

    # Results should be the same regardless of batch_size
    for name in METRIC_NAMES:
        np.testing.assert_allclose(
            results_no_batch_size[name],
            results_with_batch_size[name],
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Metric {name} differs with batch_size parameter",
        )

    # Verify shapes are correct
    assert all(value.shape[0] == 4 for value in results_with_batch_size.values())

    # Compare backends for a single metric
    return results_with_batch_size["mse"]


def test_metrics_registry():
    """Test if all metrics are in the registry"""

    metrics_funcs = inspect.getmembers(metrics, inspect.isfunction)
    for _, _func in metrics_funcs:
        if _func.__module__.startswith("zea.metrics."):
            metrics_registry.get_name(_func)  # this raises an error if the class is not registered


def test_sector_reweight_image():
    """Test sector reweight util function"""
    # TODO: redo this test to not reimplement the function
    # arrange
    cube_of_ones = np.ones((3, 3, 3)).astype(np.float32)
    cube_of_ones = ops.convert_to_tensor(cube_of_ones)

    # act
    reweighted_cube = metrics._sector_reweight_image(cube_of_ones, 180, axis=1)

    # assert
    # depths are set at the 'center' of each pixel index
    expected_depths = np.array([0.5, 1.5, 2.5])
    expected_reweighting_per_depth = np.pi  # (180 / 360) * 2 * pi = pi
    expected_result = cube_of_ones * expected_depths[:, None] * expected_reweighting_per_depth
    assert np.all(expected_result == reweighted_cube)


@backend_equality_check(decimal=4)
def test_ssim():
    """Test the properties SSIM should satisfy, on single images and on batches."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    y_true = rng.uniform(0, 1, (3, 32, 32, 1)).astype(np.float32)
    y_pred = rng.uniform(0, 1, (3, 32, 32, 1)).astype(np.float32)

    def ssim(a, b, max_val=1.0):
        return ops.convert_to_numpy(metrics.ssim(a, b, max_val=max_val))

    # One value per image for a batch, a scalar for a single image
    value = ssim(y_true, y_pred)
    assert value.shape == (3,)
    assert ssim(y_true[0], y_pred[0]).shape == ()

    # Bounded by 1, and exactly 1 for identical images
    assert np.all(value <= 1.0)
    np.testing.assert_allclose(ssim(y_true, y_true), np.ones(3), rtol=1e-5, atol=1e-5)

    # Symmetric in its arguments
    np.testing.assert_allclose(ssim(y_pred, y_true), value, rtol=1e-5, atol=1e-5)

    # Invariant to the scale of the inputs, as long as max_val scales along
    np.testing.assert_allclose(ssim(y_true * 255, y_pred * 255, 255.0), value, rtol=1e-4, atol=1e-4)

    # Decreases as more noise is added
    mild = ssim(y_true, y_true + rng.normal(0, 0.05, y_true.shape).astype(np.float32))
    severe = ssim(y_true, y_true + rng.normal(0, 0.3, y_true.shape).astype(np.float32))
    assert np.all(mild > severe)
    assert np.all(severe > value)  # noise still beats an unrelated image

    return value
