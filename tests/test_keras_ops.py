"""This tests the zea.keras module.

As we cannot test all functions, we will only test a few
of them to ensure that the wrapping works correctly."""

import inspect

import numpy as np
import pytest

import zea.keras_ops
from zea.ops import ops_registry


def test_swapaxes():
    """Test the Swapaxes operation."""
    with pytest.raises(ValueError):
        zea.keras_ops.Swapaxes(axis2=1)

    output = zea.keras_ops.Swapaxes(axis1=0, axis2=1)(data=np.ones((10, 20)))["data"]
    assert output.shape == (20, 10)


def test_registry():
    """Test that all keras.ops functions are registered in ops_registry."""

    classes = inspect.getmembers(zea.keras_ops, inspect.isclass)
    for _, _class in classes:
        if _class.__module__.startswith("zea.keras_ops."):
            ops_registry.get_name(_class)  # this raises an error if the class is not registered
