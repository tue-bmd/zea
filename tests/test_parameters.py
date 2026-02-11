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

from zea.internal.parameters import MissingDependencyError, Parameters, cache_with_dependencies


class DummyCircularParameters(Parameters):
    """A simple test class with a circular dependency."""

    VALID_PARAMS = {
        "param1": {"type": int},
    }

    @cache_with_dependencies("param1", "computed2")
    def computed1(self):
        return self.computed2 + self.param1

    @cache_with_dependencies("computed1")
    def computed2(self):
        return self.computed1


class DummyInvalidParameters(Parameters):
    """A simple test class with an invalid parameter type."""

    VALID_PARAMS = {
        "param1": {"type": int},
    }

    @cache_with_dependencies("param1")
    def computed1(self):
        return self.param1 * 2

    @cache_with_dependencies("non_existing_dependency")  # Invalid dependency
    def computed2(self):
        return self.computed1


class DummyParameters(Parameters):
    """A simple test class with parameters and computed properties.

    This class is used for testing the Parameter framework with simple
    dependencies between properties.

    Args:
        param1: First parameter (equivalent to grid_size_x in the original)
        param2: Second parameter (equivalent to grid_size_z in the original)
        param3: Third parameter with default value (like sound_speed)
        param4: Fourth parameter (like sampling_frequency)
        param5: Optional fifth parameter
        param6: Optional sixth parameter
        optional_param: Optional parameter that can be set directly or computed from dependencies

    Attributes:
        computed1: A computed property depending on param1 and param2
        computed2: A computed property depending on computed1
        computed3: A computed property depending on param3 and param4
        optional_param: A property that is either set directly or computed from dependencies
    """

    VALID_PARAMS = {
        "param1": {"type": int},
        "param2": {"type": int},
        "param3": {"type": float, "default": 1540.0},
        "param4": {"type": float},
        "param5": {"type": float},
        "param6": {"type": float},
        "optional_param": {"type": (list, type(None))},
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

    @cache_with_dependencies("param3", "param1", "param4")
    def optional_param(self):
        # Use the underlying param if set
        if self._params.get("optional_param", None) is not None:
            return self._params["optional_param"]
        # Otherwise, compute from dependencies
        if None in (self.param3, self.param1, self.param4):
            raise MissingDependencyError("Missing dependencies for optional_param")
        return [0, self.param3 * self.param1 / self.param4 / 2]

    @cache_with_dependencies("optional_param", "param6")
    def dependent_on_optional(self):
        # If optional_param is set, use it, else use computed value
        base = self.optional_param
        if self.param6 is None:
            raise MissingDependencyError("Missing dependency: param6")
        # Just for test: sum the second value of optional_param and param6
        return base[1] + self.param6


@pytest.fixture
def dummy_params():
    """Fixture for a fresh DummyParameters instance with required params."""
    return DummyParameters(param1=5, param2=10, param3=1500.0, param4=5e6)


def test_catch_circular_dependency():
    """Test that circular dependencies raise an error."""
    with pytest.raises(RuntimeError, match="Circular dependency detected"):
        DummyCircularParameters(param1=5)


def test_catch_invalid_dependency():
    """Test that invalid dependencies raise an error."""
    with pytest.raises(ValueError):
        DummyInvalidParameters(param1=5)


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
    """Test that setting a computed property raises informative MissingDependencyError."""
    with pytest.raises(ValueError) as excinfo:
        dummy_params.computed1 = 123
    msg = str(excinfo.value)
    assert "Cannot set computed property 'computed1'" in msg
    assert "param1" in msg and "param2" in msg


def test_missing_dependency_error_message():
    """Test that missing dependencies raise informative errors."""
    s = DummyParameters()
    with pytest.raises(MissingDependencyError) as excinfo:
        _ = s.computed2
    msg = str(excinfo.value)
    assert "param1" in msg and "param2" in msg


def test_to_tensor_includes_all(dummy_params: DummyParameters):
    """Test that to_tensor includes all parameters and computed properties."""
    tensors = dummy_params.to_tensor()
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


def test_to_tensor_excludes(dummy_params: Parameters):
    """Test that to_tensor excludes specified keys."""
    # Exclude computed1 and param2
    tensors = dummy_params.to_tensor(exclude=["computed1", "param2"])
    assert "computed1" not in tensors
    assert "param2" not in tensors
    # Should still include other params and computed properties
    for key in ["param1", "param3", "param4", "computed2", "computed3"]:
        assert key in tensors

    # Exclude a non-existent key, should not raise error
    dummy_params.to_tensor(exclude=["non_existent"])


def test_to_tensor_partial_computed_subset(dummy_params):
    """Test that to_tensor only computes the requested subset."""
    # Access no computed properties yet
    tensors = dummy_params.to_tensor(include=["computed1"])
    # Only computed1 should be present (besides direct params)
    assert "computed1" in tensors
    assert "computed2" not in tensors
    assert "computed3" not in tensors
    # Now try with multiple keys
    tensors2 = dummy_params.to_tensor(include=["computed1", "computed3"])
    assert "computed1" in tensors2
    assert "computed3" in tensors2
    assert "computed2" not in tensors2
    # If a key is not a computed property, it should be ignored (no error)
    tensors3 = dummy_params.to_tensor(include=["computed1", "param1"])
    assert "computed1" in tensors3
    assert "param1" in tensors3

    # An empty include list should not return an empty dict, since no keys are specified.
    tensors4 = dummy_params.to_tensor(include=[])
    assert set(tensors4.keys()) != {
        "param1",
        "param2",
        "param3",
        "param4",
        "computed1",
        "computed3",
    }

    # Access computed2 manually
    _ = dummy_params.computed2
    # Now call to_tensor with include requesting computed2
    tensors5 = dummy_params.to_tensor(include=["computed2"])
    # It should be present, and should not be recomputed (counter stays the same)
    assert "computed2" in tensors5
    count = dummy_params._computed2_count
    _ = dummy_params.to_tensor(include=["computed2"])
    assert dummy_params._computed2_count == count  # No recompute


def test_repr_and_str(dummy_params):
    """Test __repr__ and __str__ output for Parameters."""
    r = repr(dummy_params)
    s = str(dummy_params)
    assert "DummyParameters" in r
    assert "param1=" in r
    assert "DummyParameters" in s
    assert "param1=" in s


def test_optional_param_leaf_or_dependency_behavior():
    """Test that optional_param can be set as a leaf or computed as a dependency."""
    # Case 1: optional_param provided, uses it directly
    p = DummyParameters(param1=10, param2=5, param3=1500.0, param4=5e6, optional_param=[1, 2])
    assert p.optional_param == [1, 2]

    # Case 2: optional_param not provided, computed from dependencies
    p2 = DummyParameters(param1=10, param2=5, param3=1500.0, param4=5e6)
    expected = [0, 1500.0 * 10 / 5e6 / 2]
    assert np.allclose(p2.optional_param, expected)

    # Case 3: optional_param set after init, uses new value
    p2.optional_param = [3, 4]
    assert p2.optional_param == [3, 4]

    # Case 4: optional_param set to None, falls back to dependency
    p2.optional_param = None
    assert np.allclose(p2.optional_param, expected)


def test_optional_parm_with_dependent_behavior():
    """Test that dependent_on_optional behaves correctly with optional_param."""
    # Case 1: optional_param provided, dependent uses it
    p = DummyParameters(
        param1=10,
        param2=5,
        param3=1500.0,
        param4=5e6,
        optional_param=[1, 2],
        param6=7.0,
    )
    assert p.optional_param == [1, 2]
    assert p.dependent_on_optional == 2 + 7.0

    # Case 2: optional_param not provided, dependent uses computed value
    p2 = DummyParameters(param1=10, param2=5, param3=1500.0, param4=5e6, param6=8.0)
    expected = [0, 1500.0 * 10 / 5e6 / 2]
    assert np.allclose(p2.optional_param, expected)
    assert np.isclose(p2.dependent_on_optional, expected[1] + 8)

    # Case 3: optional_param set after init, dependent uses new value
    p2.optional_param = [3, 4]
    assert p2.optional_param == [3, 4]
    assert p2.dependent_on_optional == 4 + 8.0

    # Case 4: optional_param set to None, dependent falls back to computed
    p2.optional_param = None
    assert np.allclose(p2.optional_param, expected)
    assert np.isclose(p2.dependent_on_optional, expected[1] + 8)
