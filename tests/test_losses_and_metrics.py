""" Module for testing loss functions """

import inspect

import numpy as np
import pytest

import usbmd
from usbmd.backend.tensorflow.losses import SMSLE
from usbmd.registry import metrics_registry
from usbmd.utils.metrics import get_metric, _sector_reweight_image


def test_smsle():
    """Test SMSLE loss function"""
    # Create random y_true and y_pred data
    y_true = np.random.rand(1, 11, 128, 512, 2).astype(np.float32)
    y_pred = np.random.rand(1, 11, 128, 512, 2).astype(np.float32)

    # Calculate SMSLE loss
    smsle = SMSLE()
    loss = smsle(y_true, y_pred)

    # Check if loss is a scalar
    assert loss.shape == ()


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        (
            np.random.rand(11, 128, 512, 2).astype(np.float32),
            np.random.rand(11, 128, 512, 2).astype(np.float32),
        ),
        (
            np.random.rand(200).astype(np.float32),
            np.random.rand(200).astype(np.float32),
        ),
        (
            np.random.rand(300, 20).astype(np.float32),
            np.random.rand(300, 20).astype(np.float32),
        ),
    ],
)
def test_metrics(y_true, y_pred):
    """Test all losses and metrics"""
    supervised_metrics = metrics_registry.filter_by_argument("supervised", True)
    unsupervised_metrics = metrics_registry.filter_by_argument("supervised", False)

    for metric_name in metrics_registry:
        metric = get_metric(metric_name)
        try:
            if metric_name in supervised_metrics:
                metric_value = metric(y_true, y_pred)
            elif metric_name in unsupervised_metrics:
                metric_value = metric(y_pred)
            else:
                raise ValueError("Metric is not supervised or unsupervised")
        except NotImplementedError:
            continue

        assert (
            metric_value.shape == ()
        ), f"{metric_name} function does not return a scalar"


def test_metrics_registry():
    """Test if all metrics are in the registry"""
    metrics_module = inspect.getmodule(usbmd.utils.metrics)
    metrics_funcs = inspect.getmembers(metrics_module, inspect.isfunction)
    metrics_func_names = [func[0] for func in metrics_funcs]

    for metric in metrics_func_names:
        if metric == "get_metric" or metric.startswith("_"):
            continue
        assert metric in metrics_registry, f"{metric} is not in the metrics registry"


def test_sector_reweight_image():
    """Test sector reweight util function"""
    # arrange
    cube_of_ones = np.ones((3, 3, 3))

    # act
    reweighted_cube = _sector_reweight_image(cube_of_ones, 180)

    # assert
    # depths are set at the 'center' of each pixel index
    expected_depths = np.array([0.5, 1.5, 2.5])
    expected_reweighting_per_depth = np.pi  # (180 / 360) * 2 * pi = pi
    expected_result = (
        cube_of_ones * expected_depths[:, None] * expected_reweighting_per_depth
    )
    assert np.all(expected_result == reweighted_cube)
