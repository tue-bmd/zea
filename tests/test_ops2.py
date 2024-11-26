"""Tests for the Operation and Pipeline classes in ops_v2.py."""

# pylint: disable=arguments-differ

import keras
import pytest

from usbmd.ops_v2 import AddOperation, MultiplyOperation, Operation, Pipeline

# TODO: Run tests for all backends


class TestOperation(Operation):
    """Simple addition operation for testing."""

    def call(self, x, y=0):
        """Adds `x` and `y`."""
        return {"result": x + y}


@pytest.fixture
def test_operation():
    """Returns a TestOperation instance."""
    return TestOperation(cache_inputs=True, cache_outputs=True, jit_compile=False)


@pytest.fixture
def test_pipeline():
    """Returns a Pipeline with basic operations."""
    pipeline = Pipeline()
    pipeline.add_operation(MultiplyOperation(cache_outputs=True))
    pipeline.add_operation(AddOperation(cache_outputs=True))
    return pipeline


# 1. Operation Class Tests


def test_operation_initialization(test_operation):
    """Tests initialization of an Operation."""
    assert test_operation.cache_inputs is True
    assert test_operation.cache_outputs is True
    assert test_operation.jit_compile is False
    assert test_operation._input_cache == {}
    assert test_operation._output_cache == {}


def test_operation_abstract_call():
    """Ensures Operation cannot be instantiated directly."""
    class IncompleteOperation(Operation):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        _ = IncompleteOperation()


@pytest.mark.parametrize("jit_compile", [True, False])
def test_operation_input_validation(test_operation, jit_compile):
    """Tests input validation and handling of unexpected keys."""
    test_operation.jit_compile = jit_compile
    outputs = test_operation(x=5, y=3, z=10)
    assert outputs["z"] == 10
    assert outputs["result"] == 8


@pytest.mark.parametrize("jit_compile", [True, False])
def test_operation_output_caching(test_operation, jit_compile):
    """Tests output caching behavior."""
    test_operation.jit_compile = jit_compile
    output1 = test_operation(x=5, y=3)
    output2 = test_operation(x=5, y=3)
    assert output1 == output2
    output3 = test_operation(x=5, y=4)
    assert output1 != output3


@pytest.mark.parametrize("jit_compile", [True, False])
def test_operation_input_caching(test_operation, jit_compile):
    """Tests input caching behavior."""
    test_operation.jit_compile = jit_compile
    test_operation.set_input_cache(input_cache={"x": 10})
    result = test_operation(y=5)
    assert result["result"] == 15


def test_operation_jit_compilation():
    """Ensures JIT compilation works."""
    op = TestOperation(jit_compile=True)
    assert callable(op.call)


def test_operation_cache_persistence():
    """Tests persistence of output cache."""
    op = TestOperation(cache_outputs=True)
    result1 = op(x=5, y=3)
    assert result1["result"] == 8
    assert len(op._output_cache) == 1
    result2 = op(x=5, y=3)
    assert result2 == result1
    assert len(op._output_cache) == 1


# 2. Pipeline Class Tests TODO


# 3. Edge Case Tests


def test_operation_empty_input(test_operation):
    """Ensures error is raised for missing required inputs."""
    with pytest.raises(TypeError):
        test_operation()


def test_operation_large_data_inputs():
    """Tests operation with large data inputs."""
    class MatrixMultiplyOperation(Operation):
        def call(self, matrix_a, matrix_b):
            return {"result": keras.ops.dot(matrix_a, matrix_b)}

    matrix_a = keras.random.normal(shape=(512, 512))
    matrix_b = keras.random.normal(shape=(512, 512))

    op = MatrixMultiplyOperation()
    result = op(matrix_a=matrix_a, matrix_b=matrix_b)
    assert result["result"].shape == (512, 512)


if __name__ == "__main__":
    pytest.main()
