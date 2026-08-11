"""Tests for zea.internal.core."""

import numpy as np
import pytest

from zea.internal.core import dict_to_tensor


def test_dict_to_tensor_ignores_framework_to_tensor_methods():
    """Objects with a `to_tensor` method whose signature doesn't accept `keep_as_is`
    (e.g. `tf.RaggedTensor`) must not be dispatched through zea's conversion protocol.
    """
    tf = pytest.importorskip("tensorflow")

    ragged = tf.ragged.constant([[1, 2], [3]])
    result = dict_to_tensor({"ragged": ragged, "value": 1.0})

    assert result["ragged"] is ragged
    assert np.asarray(result["value"]) == 1.0
