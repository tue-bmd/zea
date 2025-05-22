"""Tests for the Parameters base class.

This test suite verifies the following features of the Parameters system:

- Type validation and error handling for parameter assignment
- Dependency tracking and lazy computation of properties
- Cache invalidation and recomputation on parameter change
- Restriction to setting only leaf parameters (not computed properties)
- Informative error messages for missing dependencies
- Tensor conversion of parameters and computed properties

"""

import time

import keras
import numpy as np
import pytest

from usbmd.internal.parameters import Parameters, cache_with_dependencies


class DummyParameters(Parameters):
    """A simple test class with parameters and computed properties.

    This class is used for testing the Parameter framework with simple
    dependencies between properties.

    Args:
        param1: First parameter (equivalent to Nx in the original)
        param2: Second parameter (equivalent to Nz in the original)
        param3: Third parameter with default value (like sound_speed)
        param4: Fourth parameter (like sampling_frequency)
        param5: Optional fifth parameter
        param6: Optional sixth parameter

    Attributes:
        computed1: A computed property depending on param1 and param2
        computed2: A computed property depending on computed1
        computed3: A computed property depending on param3 and param4
    """

    VALID_PARAMS = {
        "param1": {"type": int, "default": None},
        "param2": {"type": int, "default": None},
        "param3": {"type": float, "default": 1540.0},
        "param4": {"type": float, "default": None},
        "param5": {"type": float, "default": None},
        "param6": {"type": float, "default": None},
    }

    def _timestamp(self):
        return time.time()

    @cache_with_dependencies("param1", "param2")
    def computed1(self):
        # Use a call counter for robust cache testing
        if not hasattr(self, "_computed1_count"):
            self._computed1_count = 0
        self._computed1_count += 1
        p1, p2 = self.param1, self.param2
        return np.meshgrid(np.arange(p1), np.arange(p2), indexing="ij")

    @cache_with_dependencies("computed1")
    def computed2(self):
        if not hasattr(self, "_computed2_count"):
            self._computed2_count = 0
        self._computed2_count += 1
        x, z = self.computed1
        dx = np.mean(np.diff(x[:, 0]))
        dz = np.mean(np.diff(z[0, :]))
        return dx, dz

    @cache_with_dependencies("param3", "param4")
    def computed3(self):
        if not hasattr(self, "_computed3_count"):
            self._computed3_count = 0
        self._computed3_count += 1
        return self.param3 / self.param4


@pytest.fixture
def dummy_params():
    """Fixture for a fresh DummyParameters instance with required params."""
    return DummyParameters(param1=5, param2=10, param3=1500.0, param4=5e6)


def test_type_validation_on_init():
    """Test that invalid parameter names and types raise errors on init."""
    with pytest.raises(ValueError, match="Invalid parameter: invalid_param"):
        DummyParameters(param1=1, param2=2, invalid_param=3)
    with pytest.raises(TypeError, match="Parameter 'param4' expected type float"):
        DummyParameters(param1=1, param2=2, param3=1500.0, param4="not_a_float")


def test_type_validation_on_set(dummy_params):
    """Test that setting invalid parameter names/types after init raises errors."""
    with pytest.raises(ValueError, match="Invalid parameter: invalid_param"):
        dummy_params.invalid_param = 42
    with pytest.raises(TypeError, match="Parameter 'param3' expected type float"):
        dummy_params.param3 = "not_a_float"


def test_dependency_tracking_and_lazy_computation(dummy_params):
    """Test that computed properties are lazily evaluated and cached."""
    # Access computed1 and check it is computed
    _ = dummy_params.computed1
    assert dummy_params._computed1_count == 1
    # Access again, should not recompute
    _ = dummy_params.computed1
    assert dummy_params._computed1_count == 1
    # Changing a dependency invalidates cache
    dummy_params.param1 = 6
    _ = dummy_params.computed1
    assert dummy_params._computed1_count == 2


def test_chain_dependency_and_cache(dummy_params):
    """Test that chained dependencies are resolved and cached correctly."""
    _ = dummy_params.computed2
    assert dummy_params._computed1_count == 1
    assert dummy_params._computed2_count == 1
    # Changing a leaf param invalidates all dependents
    dummy_params.param2 = 11
    _ = dummy_params.computed1
    _ = dummy_params.computed2
    assert dummy_params._computed1_count == 2
    assert dummy_params._computed2_count == 2  # fails here
    _ = dummy_params.computed1
    _ = dummy_params.computed2
    # However, when accessing computed1 and 2 again, without
    # changing any leaf params, they should not recompute
    assert dummy_params._computed1_count == 2
    assert dummy_params._computed2_count == 2


def test_setting_computed_property_raises(dummy_params):
    """Test that setting a computed property raises informative AttributeError."""
    with pytest.raises(AttributeError) as excinfo:
        dummy_params.computed1 = 123
    msg = str(excinfo.value)
    assert "Cannot set computed property 'computed1'" in msg
    assert "param1" in msg and "param2" in msg


def test_missing_dependency_error_message():
    """Test that missing dependencies raise informative errors."""
    s = DummyParameters()
    with pytest.raises(AttributeError) as excinfo:
        _ = s.computed2
    msg = str(excinfo.value)
    assert "param1" in msg and "param2" in msg


def test_to_tensor_includes_all(dummy_params):
    """Test that to_tensor includes all parameters and computed properties."""
    tensors = dummy_params.to_tensor(compute_missing=True)
    # Should include all direct params and computed1, computed2, computed3
    for key in [
        "param1",
        "param2",
        "param3",
        "param4",
        "computed1",
        "computed2",
        "computed3",
    ]:
        assert key in tensors
    # Check tensor value for computed3
    assert np.isclose(
        float(keras.ops.convert_to_numpy(tensors["computed3"])),
        dummy_params.param3 / dummy_params.param4,
    )


def test_to_tensor_only_computed(dummy_params):
    """Test that to_tensor(compute_missing=False) only includes already computed properties."""
    # Before accessing any computed property
    tensors = dummy_params.to_tensor(compute_missing=False)
    assert set(tensors.keys()) == {"param1", "param2", "param3", "param4"}
    # After accessing computed1
    _ = dummy_params.computed1
    tensors2 = dummy_params.to_tensor(compute_missing=False)
    assert "computed1" in tensors2


def test_repr_and_str(dummy_params):
    """Test __repr__ and __str__ output for Parameters."""
    r = repr(dummy_params)
    s = str(dummy_params)
    assert "DummyParameters" in r
    assert "param1=" in r
    assert "DummyParameters" in s
    assert "param1=" in s
