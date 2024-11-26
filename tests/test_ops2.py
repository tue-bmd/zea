import keras
import pytest

from usbmd.ops_v2 import AddOperation, MultiplyOperation, Operation, Pipeline

# TODO: We should run these tests for all backends


# Custom test subclass of Operation
class TestOperation(Operation):
    def call(self, x, y=0):
        return {"result": x + y}


@pytest.fixture
def test_operation():
    return TestOperation(cache_inputs=True, cache_outputs=True, jit_compile=False)


@pytest.fixture
def test_pipeline():
    pipeline = Pipeline()
    pipeline.add_operation(MultiplyOperation(cache_outputs=True))
    pipeline.add_operation(AddOperation(cache_outputs=True))
    return pipeline


# 1. Operation Class Tests


def test_operation_initialization(test_operation):
    assert test_operation.cache_inputs is True
    assert test_operation.cache_outputs is True
    assert test_operation.jit_compile is False
    assert test_operation._input_cache == {}
    assert test_operation._output_cache == {}


def test_operation_abstract_call():
    class IncompleteOperation(Operation):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        _ = IncompleteOperation()


@pytest.mark.parametrize("jit_compile", [True, False])
def test_operation_input_validation(test_operation, jit_compile):
    test_operation.jit_compile = jit_compile

    outputs = test_operation(x=5, y=3, z=10)  # 'z' is an unexpected key
    assert outputs["z"] == 10  # Unexpected key should be passed through
    assert outputs["result"] == 8  # Valid keys processed correctly


@pytest.mark.parametrize("jit_compile", [True, False])
def test_operation_output_caching(test_operation, jit_compile):
    test_operation.jit_compile = jit_compile

    output1 = test_operation(x=5, y=3)
    output2 = test_operation(x=5, y=3)  # Same inputs
    assert output1 == output2  # Cached output reused

    output3 = test_operation(x=5, y=4)  # Different inputs
    assert (
        output1 != output3
    )  # New computation, no cache hit should return different output


@pytest.mark.parametrize("jit_compile", [True, False])
def test_operation_input_caching(test_operation, jit_compile):
    test_operation.jit_compile = jit_compile
    test_operation.set_input_cache(input_cache={"x": 10})
    result = test_operation(y=5)  # Should still merge cached input
    assert result["result"] == 15


def test_operation_jit_compilation():
    op = TestOperation(jit_compile=True)
    assert callable(op.call)  # Ensure JIT compilation doesn't break functionality


def test_operation_cache_persistence():
    op = TestOperation(cache_outputs=True)

    # First run to populate the cache
    result1 = op(x=5, y=3)
    assert result1["result"] == 8  # Correct computation
    assert len(op._output_cache) == 1  # Cache should now be populated

    # Second run with the same inputs
    result2 = op(x=5, y=3)
    assert result2 == result1  # Result should match the first run
    assert len(op._output_cache) == 1  # Cache size should remain unchanged


# 2. Pipeline Class Tests TODO


# 3. Edge Case Tests


def test_operation_empty_input(test_operation):
    with pytest.raises(TypeError):  # 'x' is required in TestOperation
        test_operation()


def test_operation_large_data_inputs():
    class MatrixMultiplyOperation(Operation):
        def call(self, matrix_a, matrix_b):
            return {"result": keras.ops.dot(matrix_a, matrix_b)}

    matrix_a = keras.random.normal(shape=(512, 512))
    matrix_b = keras.random.normal(shape=(512, 512))

    op = MatrixMultiplyOperation()
    result = op(matrix_a=matrix_a, matrix_b=matrix_b)
    assert result["result"].shape == (512, 512)  # Matrix multiplication output

if __name__ == "__main__":
    pytest.main()
