"""This tests the zea.keras module.

As we cannot test all functions, we will only test a few
of them to ensure that the wrapping works correctly."""

import inspect

import numpy as np
import pytest

from zea.internal.registry import ops_registry
from zea.ops import keras_ops


def test_swapaxes():
    """Test the Swapaxes operation."""
    with pytest.raises(ValueError):
        keras_ops.Swapaxes(axis2=1)

    output = keras_ops.Swapaxes(
        axis1=0,
        axis2=1,
        with_batch_dim=False,
    )(data=np.ones((10, 20)))["data"]
    assert output.shape == (20, 10)

    output = keras_ops.Swapaxes(
        axis1=0,
        axis2=2,
        with_batch_dim=True,
    )(data=np.ones((5, 10, 15, 3)))["data"]
    assert output.shape == (5, 3, 15, 10)


def test_registry():
    """Test that all keras.ops functions are registered in ops_registry."""

    classes = inspect.getmembers(keras_ops, inspect.isclass)
    for _, _class in classes:
        if _class.__module__.startswith("zea.ops.keras_ops."):
            ops_registry.get_name(_class)  # this raises an error if the class is not registered
