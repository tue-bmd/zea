"""Tests for the Operation and Pipeline classes in ops_v2.py."""

# pylint: disable=arguments-differ, abstract-class-instantiated

import keras
import pytest

from usbmd.ops_v2 import DataTypes, Operation, Pipeline
from usbmd.probes import Dummy
from usbmd.scan import Scan

# TODO: Run tests for all backends


class MultiplyOperation(Operation):
    """Multiply Operation for testing purposes."""

    def call(self, x, y):
        """
        Multiplies the input x by the specified factor.
        """
        return {"x": keras.ops.multiply(x, y)}


class AddOperation(Operation):
    """Add Operation for testing purposes."""

    def call(self, x, y):
        """
        Adds the result from MultiplyOperation with y.
        """
        # print(f"Processing AddOperation: result={result}, y={y}")
        return {"z": keras.ops.add(x, y)}


class LargeMatrixMultiplicationOperation(Operation):
    """Large Matrix Multiplication Operation for testing purposes."""

    def call(self, matrix_a, matrix_b):
        """
        Performs large matrix multiplication using Keras ops.
        """
        # print("Processing LargeMatrixMultiplicationOperation...")
        # Perform matrix multiplication
        result = keras.ops.matmul(matrix_a, matrix_b)
        result2 = keras.ops.matmul(result, matrix_a)
        result3 = keras.ops.matmul(result2, matrix_b)
        return {"matrix_result": result3}


class ElementwiseMatrixOperation(Operation):
    """Elementwise Matrix Operation for testing purposes."""

    def call(self, matrix, scalar):
        """
        Performs elementwise operations on a matrix (adds and multiplies by scalar).
        """
        # print("Processing ElementwiseMatrixOperation...")
        # Perform elementwise addition and multiplication
        result = keras.ops.add(matrix, scalar)
        result = keras.ops.multiply(result, scalar)
        return {"elementwise_result": result}


@pytest.fixture
def test_operation():
    """Returns a MultiplyOperation instance."""
    return AddOperation(cache_inputs=True, cache_outputs=True, jit_compile=False)


# 1. Operation Class Tests


def test_operation_initialization(test_operation):
    """Tests initialization of an Operation."""
    assert test_operation.cache_inputs is True
    assert test_operation.cache_outputs is True
    assert test_operation._jit_compile is False
    assert test_operation._input_cache == {}
    assert test_operation._output_cache == {}


@pytest.mark.parametrize("jit_compile", [True, False])
def test_operation_input_validation(test_operation, jit_compile):
    """Tests input validation and handling of unexpected keys."""
    test_operation.set_jit(jit_compile)
    outputs = test_operation(x=5, y=3, other=10)
    assert outputs["other"] == 10
    assert outputs["z"] == 8


@pytest.mark.parametrize("jit_compile", [True, False])
def test_operation_output_caching(test_operation, jit_compile):
    """Tests output caching behavior."""
    test_operation.set_jit(jit_compile)
    output1 = test_operation(x=5, y=3)
    output2 = test_operation(x=5, y=3)
    assert output1 == output2
    output3 = test_operation(x=5, y=4)
    assert output1 != output3


@pytest.mark.parametrize("jit_compile", [True, False])
def test_operation_input_caching(test_operation, jit_compile):
    """Tests input caching behavior."""
    test_operation.set_jit(jit_compile)
    test_operation.set_input_cache(input_cache={"x": 10})
    result = test_operation(y=5)
    assert result["z"] == 15


def test_operation_jit_compilation():
    """Ensures JIT compilation works."""
    op = AddOperation(jit_compile=True)
    assert callable(op.call)


def test_operation_cache_persistence():
    """Tests persistence of output cache."""
    op = AddOperation(cache_outputs=True)
    result1 = op(x=5, y=3)
    assert result1["z"] == 8
    assert len(op._output_cache) == 1
    result2 = op(x=5, y=3)
    assert result2 == result1
    assert len(op._output_cache) == 1


def test_string_representation(verbose=False):
    """Print the string representation of the Pipeline"""
    operations = [MultiplyOperation(), AddOperation()]
    pipeline = Pipeline(operations=operations)
    if verbose:
        print(str(pipeline))
    assert str(pipeline) == "MultiplyOperation -> AddOperation"


# 2. Pipeline Class Tests


def test_pipeline_initialization():
    """Tests initialization of a Pipeline."""
    operations = [MultiplyOperation(), AddOperation()]
    pipeline = Pipeline(operations=operations)
    assert len(pipeline.operations) == 2
    assert isinstance(pipeline.operations[0], MultiplyOperation)
    assert isinstance(pipeline.operations[1], AddOperation)


def test_pipeline_call():
    """Tests the call method of the Pipeline."""
    operations = [MultiplyOperation(), AddOperation()]
    pipeline = Pipeline(operations=operations)
    result = pipeline(x=2, y=3)
    assert result["z"] == 9  # (2 * 3) + 3


def test_pipeline_with_large_matrix_multiplication():
    """Tests the Pipeline with a large matrix multiplication operation."""
    operations = [LargeMatrixMultiplicationOperation()]
    pipeline = Pipeline(operations=operations)
    matrix_a = keras.random.normal(shape=(512, 512))
    matrix_b = keras.random.normal(shape=(512, 512))
    result = pipeline(matrix_a=matrix_a, matrix_b=matrix_b)
    assert result["matrix_result"].shape == (512, 512)


def test_pipeline_with_elementwise_operation():
    """Tests the Pipeline with an elementwise matrix operation."""
    operations = [ElementwiseMatrixOperation()]
    pipeline = Pipeline(operations=operations)
    matrix = keras.random.normal(shape=(512, 512))
    scalar = 2
    result = pipeline(matrix=matrix, scalar=scalar)
    assert result["elementwise_result"].shape == (512, 512)


def test_pipeline_jit_options():
    """Tests the JIT options for the Pipeline."""
    operations = [MultiplyOperation(), AddOperation()]
    pipeline = Pipeline(operations=operations, jit_options="pipeline")
    assert callable(pipeline.call)

    pipeline = Pipeline(operations=operations, jit_options="ops")
    for operation in pipeline.operations:
        assert operation._jit_compile is True

    pipeline = Pipeline(operations=operations, jit_options=None)
    for operation in pipeline.operations:
        assert operation._jit_compile is False


def test_pipeline_set_params():
    """Tests setting parameters for the Pipeline."""
    operations = [MultiplyOperation(), AddOperation()]
    pipeline = Pipeline(operations=operations)
    pipeline.set_params(x=5, y=3)
    params = pipeline.get_params()
    assert params["x"] == 5
    assert params["y"] == 3


def test_pipeline_get_params_per_operation():
    """Tests getting parameters per operation in the Pipeline."""
    operations = [MultiplyOperation(), AddOperation()]
    pipeline = Pipeline(operations=operations)
    pipeline.set_params(x=5, y=3)
    params = pipeline.get_params(per_operation=True)
    assert params[0]["x"] == 5
    assert params[1]["y"] == 3


def test_pipeline_validation():
    """Tests the validation of the Pipeline."""
    operations = [
        MultiplyOperation(output_data_type=DataTypes.RAW_DATA),
        AddOperation(input_data_type=DataTypes.RAW_DATA),
    ]
    _ = Pipeline(operations=operations)

    operations = [
        MultiplyOperation(output_data_type=DataTypes.RAW_DATA),
        AddOperation(input_data_type=DataTypes.IMAGE),
    ]
    with pytest.raises(ValueError):
        _ = Pipeline(operations=operations)


def test_pipeline_with_scan_probe_config():
    """Tests the Pipeline with Scan, Probe, and Config objects as inputs."""

    probe = Dummy()
    scan = Scan(
        n_tx=128,
        n_ax=256,
        n_el=128,
        n_ch=2,
        center_frequency=5.0,
        sampling_frequency=5.0,
        xlims=(-2e-3, 2e-3),
    )

    # TODO: Add Config object as input to the Pipeline, currently config is not an Object

    operations = [MultiplyOperation(), AddOperation()]
    pipeline = Pipeline(operations=operations)

    result = pipeline(scan, probe, x=2, y=3)
    assert "z" in result
    assert "n_tx" in result  # Check if we parsed the scan object correctly
    assert "probe_geometry" in result  # Check if we parsed the probe object correctly


# 3. Edge Case Tests


def test_operation_empty_input(test_operation):
    """Ensures error is raised for missing required inputs."""
    with pytest.raises(TypeError):
        test_operation()


def test_operation_large_data_inputs():
    """Tests operation with large data inputs."""

    class MatrixMultiplyOperation(Operation):
        """Matrix multiplication operation for testing."""

        def call(self, matrix_a, matrix_b):
            return {"result": keras.ops.dot(matrix_a, matrix_b)}

    matrix_a = keras.random.normal(shape=(512, 512))
    matrix_b = keras.random.normal(shape=(512, 512))

    op = MatrixMultiplyOperation()
    result = op(matrix_a=matrix_a, matrix_b=matrix_b)
    assert result["result"].shape == (512, 512)


if __name__ == "__main__":
    pytest.main()
