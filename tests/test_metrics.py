"""Module for testing loss functions"""

import inspect

import numpy as np
import pytest
from keras import ops

from zea import metrics
from zea.backend.tensorflow.losses import SMSLE
from zea.internal.registry import metrics_registry

from . import DEFAULT_TEST_SEED, backend_equality_check


def test_smsle():
    """Test SMSLE loss function"""
    # Create random y_true and y_pred data
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    y_true = rng.standard_normal((1, 11, 128, 512, 2)).astype(np.float32)
    y_pred = rng.standard_normal((1, 11, 128, 512, 2)).astype(np.float32)

    # Calculate SMSLE loss
    smsle = SMSLE()
    loss = smsle(y_true, y_pred)

    # Check if loss is a scalar
    assert loss.shape == ()


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

    # Regression test against TensorFlow implementations for SSIM and PSNR
    if metric_name == "ssim":
        import tensorflow as tf

        expected_value = tf.image.ssim(
            ops.convert_to_numpy(y_true),
            ops.convert_to_numpy(y_pred),
            max_val=255.0,
        )
        np.testing.assert_allclose(metric_value, expected_value, rtol=1e-5, atol=1e-5)
    elif metric_name == "psnr":
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
