"""Tests for the Operation and Pipeline classes in ops_v2.py.

# TODO: Run tests for all backends
# TODO: merge with original ops
"""

# pylint: disable=arguments-differ, abstract-class-instantiated, pointless-string-statement

import json

import keras
import pytest

from usbmd.config.config import Config
from usbmd.ops_v2 import Operation, Pipeline, pipeline_from_config, pipeline_from_json
from usbmd.probes import Dummy
from usbmd.registry import ops_registry
from usbmd.scan import Scan

"""Some operations for testing"""


@ops_registry("testmultiply")
class TestMultiply(Operation):
    """TestMultiply Operation for testing purposes."""

    def __init__(self, useless_parameter: int = None, **kwargs):
        super().__init__(**kwargs)
        self.useless_parameter = useless_parameter

    def call(self, x, y):
        """
        Multiplies x and y.
        """
        return {"x": keras.ops.multiply(x, y)}


@ops_registry("testadd")
class TestAdd(Operation):
    """Add Operation for testing purposes."""

    def call(self, x, y=None, **kwargs):
        if y is None:
            y = kwargs.get("input_y")
        return {"z": keras.ops.add(x, y)}


@ops_registry("large_matrix_multiplication")
class LargeMatrixMultiplicationOperation(Operation):
    """Large Matrix Multiplication Operation for testing purposes."""

    def call(self, matrix_a, matrix_b):
        """
        Performs large matrix multiplication using Keras ops.
        """
        result = keras.ops.matmul(matrix_a, matrix_b)
        result2 = keras.ops.matmul(result, matrix_a)
        result3 = keras.ops.matmul(result2, matrix_b)
        return {"matrix_result": result3}


@ops_registry("elementwise_matrix_operation")
class ElementwiseMatrixOperation(Operation):
    """Elementwise Matrix Operation for testing purposes."""

    def call(self, matrix, scalar):
        """
        Performs elementwise operations on a matrix.
        """
        result = keras.ops.add(matrix, scalar)
        result = keras.ops.multiply(result, scalar)
        return {"elementwise_result": result}


"""Fixtures"""


@pytest.fixture
def test_operation():
    """Returns an TestAdd instance with caching."""
    return TestAdd(cache_inputs=True, cache_outputs=True, jit_compile=False)


@pytest.fixture
def pipeline_config():
    """Returns a test sequential pipeline configuration."""
    return {
        "operations": [
            {"op": "testmultiply", "params": {}},
            {"op": "testadd", "params": {}},
        ]
    }


@pytest.fixture
def pipeline_config_with_params():
    """Returns a test sequential pipeline configuration with parameters."""
    return {
        "operations": [
            {"op": "testmultiply", "params": {"useless_parameter": 10}},
            {"op": "testadd"},
        ]
    }


@pytest.fixture
def pipeline_config_with_branch():
    """
    Returns a test branched pipeline configuration.

    In this configuration:
      - 'prod' (a multiply op) computes {"x": x*y}
      - 'branch1' (an add op) takes the full output from 'prod'
        (i.e. it will receive all key–value pairs from prod) and global input 'input_y'
        to compute {"z": add(prod.x, input_y)}
      - 'branch2' (an add op) does the same as branch1.
      - 'rename1' (a rename op) renames the output 'z' from 'branch1' to 'z1'.
      - 'rename2' (a rename op) renames the output 'z' from 'branch2' to 'z2'.
      - 'merge' (a merge op) merges the outputs from 'rename1' and 'rename2'.

    With inputs x=2, y=3, input_y=10:
      - prod: 2*3 = 6, so its output is {"x": 6}
      - branch1 and branch2 each receive the full output from 'prod' (i.e. {"x": 6})
        and also (from the global input) input_y=10. Thus, they call add(x=6, input_y=10)
        (the add op will use input_y as y), yielding {"z": 16}.
      - rename1 renames 'z' from branch1 to 'z1', so its output is {"z1": 16}
      - rename2 renames 'z' from branch2 to 'z2', so its output is {"z2": 16}
      - merge receives the outputs from rename1 and rename2 and merges them,
        so the final output contains {"z1": 16, "z2": 16}.
    """
    return {
        "operations": [
            {"id": "prod", "op": "testmultiply", "params": {}},
            {"id": "branch1", "op": "testadd", "params": {}, "inputs": ["prod"]},
            {"id": "branch2", "op": "testadd", "params": {}, "inputs": ["prod"]},
            {
                "id": "rename1",
                "op": "rename_v2",
                "params": {"mapping": {"z": "z1"}},
                "inputs": ["branch1"],
            },
            {
                "id": "rename2",
                "op": "rename_v2",
                "params": {"mapping": {"z": "z2"}},
                "inputs": ["branch2"],
            },
            {
                "id": "merge",
                "op": "merge_v2",
                "params": {},
                "inputs": ["rename1", "rename2"],
            },
        ]
    }


"""Operation Class Tests"""


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
    # 'other' should be passed through unchanged.
    assert outputs["other"] == 10
    assert outputs["z"] == 8  # 5 + 3


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
    assert result["z"] == 15  # 10 + 5


def test_operation_jit_compilation():
    """Ensures JIT compilation works."""
    op = TestAdd(jit_compile=True)
    assert callable(op.call)


def test_operation_cache_persistence():
    """Tests persistence of output cache."""
    op = TestAdd(cache_outputs=True)
    result1 = op(x=5, y=3)
    assert result1["z"] == 8
    assert len(op._output_cache) == 1
    result2 = op(x=5, y=3)
    assert result2 == result1
    assert len(op._output_cache) == 1


def test_string_representation(verbose=False):
    """Print the string representation of the Pipeline"""
    operations = [TestMultiply(), TestAdd()]
    pipeline = Pipeline(operations=operations)
    if verbose:
        print(str(pipeline))
    assert str(pipeline) == "op_0:TestMultiply -> op_1:TestAdd"


"""Pipeline Class Tests"""


def test_pipeline_initialization():
    """Tests initialization of a Pipeline."""
    operations = [TestMultiply(), TestAdd()]
    pipeline = Pipeline(operations=operations)
    # For sequential pipelines, operations are stored in _ops_list.
    assert len(pipeline._ops_list) == 2
    assert isinstance(pipeline._ops_list[0], TestMultiply)
    assert isinstance(pipeline._ops_list[1], TestAdd)


def test_pipeline_call():
    """Tests the call method of the Pipeline."""
    operations = [TestMultiply(), TestAdd()]
    pipeline = Pipeline(operations=operations)
    result = pipeline(x=2, y=3)
    # multiply: 2*3 = 6; add: 6+3 = 9.
    assert result["z"] == 9


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
    operations = [TestMultiply(), TestAdd()]
    pipeline = Pipeline(operations=operations, jit_options="pipeline")
    assert callable(pipeline.call)

    pipeline = Pipeline(operations=operations, jit_options="ops")
    for op in pipeline._ops_list:
        assert op._jit_compile is True

    pipeline = Pipeline(operations=operations, jit_options=None)
    for op in pipeline._ops_list:
        assert op._jit_compile is False


def test_pipeline_cycle_detection():
    """Test that a circular dependency in the pipeline raises a ValueError."""
    # Define two dummy operations with cyclic dependencies.
    op_a = TestMultiply(uid="op_a")
    op_a.inputs = ["op_b"]  # op_a depends on op_b

    op_b = TestAdd(uid="op_b")
    op_b.inputs = ["op_a"]  # op_b depends on op_a

    with pytest.raises(ValueError, match="Cycle detected"):
        Pipeline(operations=[op_a, op_b])


def test_operation_invalid_output():
    """Test that an operation returning a non-dict raises a TypeError."""

    class BadOperation(Operation):
        """Operation that returns a list instead of a dict."""

        def call(self, **kwargs):
            return [1, 2, 3]  # Not a dict!

    op = BadOperation()
    with pytest.raises(TypeError, match="must return a dictionary"):
        op(x=1)


def test_operation_cache_clearing():
    """Test that clearing an operation's cache forces re-computation."""
    op = TestAdd(cache_outputs=True, jit_compile=False)
    result1 = op(x=1, y=2)
    # Cache should now have an entry.
    assert op._output_cache
    op.clear_cache()
    assert not op._output_cache
    result2 = op(x=1, y=2)
    # Ensure the recomputed result is equal to the original result.
    assert result1 == result2


def test_pipeline_missing_dependency():
    """Test that a pipeline referencing a non-existent op ID raises a ValueError."""
    op = TestMultiply(uid="op1")
    # This add op references a non-existent op "missing_op"
    add_op = TestAdd(uid="op2")
    add_op.inputs = ["missing_op"]

    with pytest.raises(ValueError, match="missing from the Operation chain"):
        Pipeline(operations=[op, add_op])


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
        probe_geometry=probe.probe_geometry,
    )

    # TODO: Add Config object as input to the Pipeline, currently config is not an Object
    operations = [TestMultiply(), TestAdd()]
    pipeline = Pipeline(operations=operations)
    result = pipeline(scan, probe, x=2, y=3)
    assert "z" in result
    assert "n_tx" in result  # Check if we parsed the scan object correctly
    assert "probe_geometry" in result  # Check if we parsed the probe object correctly


"""Pipeline build from config / json tests"""


@pytest.mark.parametrize(
    "config_fixture", ["pipeline_config", "pipeline_config_with_params"]
)
def test_pipeline_from_json(config_fixture, request):
    """Tests creating a pipeline from a JSON string."""
    config = request.getfixturevalue(config_fixture)
    json_string = json.dumps(config)
    pipeline = pipeline_from_json(json_string, jit_options=None)

    assert len(pipeline._ops_list) == 2
    assert isinstance(pipeline._ops_list[0], TestMultiply)
    assert isinstance(pipeline._ops_list[1], TestAdd)

    result = pipeline(x=2, y=3)
    if config_fixture == "pipeline_config_with_params":
        assert pipeline._ops_list[0].useless_parameter == 10

    assert result["z"] == 9  # (2 * 3) + 3


@pytest.mark.parametrize(
    "config_fixture",
    ["pipeline_config", "pipeline_config_with_params", "pipeline_config_with_branch"],
)
def test_pipeline_from_config(config_fixture, request):
    """Tests creating a pipeline from a Config object."""
    config_dict = request.getfixturevalue(config_fixture)
    config = Config(**config_dict)
    pipeline = pipeline_from_config(config, jit_options=None)

    # For branched pipelines, _ops_list contains all ops.
    if config_fixture == "pipeline_config_with_branch":
        # Expect 6 operations in branched pipeline.
        assert len(pipeline._ops_list) == 6
    else:
        assert len(pipeline._ops_list) == 2

    # For the branched pipeline, we pass global inputs: x=2, y=3, input_y=10.
    result = pipeline(x=2, y=3)
    if config_fixture != "pipeline_config_with_branch":
        assert result["z"] == 9  # (2 * 3) + 3
    else:
        assert "z1" in result and "z2" in result
        assert result["z1"] == 9
        assert result["z2"] == 9


"""Pipeline Save/Load Tests"""


def test_pipeline_save_and_load(tmp_path):
    """Tests saving and loading a sequential pipeline in JSON format."""
    operations = [TestMultiply(), TestAdd()]
    pipeline = Pipeline(operations=operations, jit_options=None)
    file_path = tmp_path / "pipeline.json"
    pipeline.save(str(file_path), file_format="json")
    loaded_pipeline = Pipeline.load(str(file_path))
    result_original = pipeline(x=2, y=3)
    result_loaded = loaded_pipeline(x=2, y=3)
    assert result_original == result_loaded


def test_pipeline_save_and_load_yaml(tmp_path):
    """Tests saving and loading a sequential pipeline in YAML format."""
    operations = [TestMultiply(), TestAdd()]
    pipeline = Pipeline(operations=operations, jit_options=None)
    file_path = tmp_path / "pipeline.yaml"
    pipeline.save(str(file_path), file_format="yaml")
    loaded_pipeline = Pipeline.load(str(file_path))
    result_original = pipeline(x=2, y=3)
    result_loaded = loaded_pipeline(x=2, y=3)
    assert result_original == result_loaded


"""Edge Case Tests"""


def test_operation_empty_input(test_operation):
    """Ensures error is raised for missing required inputs."""
    with pytest.raises(TypeError):
        test_operation()


def test_operation_large_data_inputs():
    """Tests operation with large data inputs."""

    class MatrixTestMultiplyOperation(Operation):
        """Matrix multiplication operation for testing."""

        def call(self, matrix_a, matrix_b):
            return {"result": keras.ops.dot(matrix_a, matrix_b)}

    matrix_a = keras.random.normal(shape=(512, 512))
    matrix_b = keras.random.normal(shape=(512, 512))
    op = MatrixTestMultiplyOperation()
    result = op(matrix_a=matrix_a, matrix_b=matrix_b)
    assert result["result"].shape == (512, 512)


if __name__ == "__main__":
    pytest.main()
