""" Experimental version of the USBMD ops module"""

import os
import timeit
import hashlib
import inspect
from abc import ABC, abstractmethod
from typing import Callable, Any, Tuple, Dict, List, Union
import numpy as np
import json

# Set the Keras backend
# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["KERAS_BACKEND"] = "torch"
# os.environ["KERAS_BACKEND"] = "numpy"
import keras

from ops import Operation, Pipeline


print("WARNING: This module is work in progress and may not work as expected!")


"""
TODO: jit_compile should allow for 2 different modes: 
    1. Operation-based: each operation is compiled separately by setting Operation(jit_compile=True).
    This means the __call__ method is not compiled and most of the usbmd logic can be executed on the fly,
    preserving the caching functionality. (DONE)
    2. Pipeline-based: the entire pipeline is compiled by setting Pipeline(jit_compile=True). 
    This means the entire pipeline is compiled and executed as a single function, which may be faster but
    may not preserve the caching functionality. 
"""


class Operation(ABC):
    """
    A base abstract class for operations in the pipeline with caching functionality.
    """

    def __init__(
        self,
        cache_inputs: Union[bool, List[str]] = False,
        cache_outputs: bool = False,
        jit_compile=True,
    ):
        """
        args:
            cache_inputs: A list of input keys to cache or True to cache all inputs
            cache_outputs: A list of output keys to cache or True to cache all outputs
            jit_compile: Whether to JIT compile the 'call' method for faster execution
        """
        self.cache_inputs = cache_inputs
        self.cache_outputs = cache_outputs
        self.jit_compile = jit_compile

        # Initialize input and output caches
        self._input_cache = {}
        self._output_cache = {}

        # Obtain the input signature of the `call` method
        self._input_signature = None
        self._valid_keys = None  # Keys valid for the `call` method
        self._trace_signatures()

        # Compile the `call` method if necessary
        self.call = jit_compile(self.call) if self.jit_compile else self.call

    def _trace_signatures(self):
        """
        Analyze and store the input/output signatures of the `call` method.
        """
        self.input_signature = inspect.signature(self.call)
        self.valid_keys = set(self.input_signature.parameters.keys())

    @abstractmethod
    def call(self, **kwargs):
        """
        Abstract method that defines the processing logic for the operation.
        Subclasses must implement this method.
        """
        pass

    def set_cache(self, input_cache: Dict[str, Any], output_cache: Dict[str, Any]):
        """
        Set a cache for inputs or outputs, then retrace the function if necessary.

        args:
            input_cache: A dictionary containing cached inputs.
            output_cache: A dictionary containing cached outputs.
        """
        self.input_cache.update(input_cache)
        self.output_cache.update(output_cache)
        self._trace_signatures()  # Retrace after updating cache to ensure correctness.

    def clear_cache(self):
        """
        Clear the input and output caches.
        """
        self.input_cache.clear()
        self.output_cache.clear()

    def _hash_inputs(self, kwargs: Dict) -> str:
        """
        Generate a hash for the given inputs to use as a cache key.

        :param kwargs: Keyword arguments.
        :return: A unique hash representing the inputs.
        """
        input_json = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.md5(input_json.encode()).hexdigest()

    def __call__(self, **kwargs) -> Dict:
        """
        Process the input keyword arguments and return the processed results.

        :param kwargs: Keyword arguments to be processed.
        :return: Combined input and output as kwargs.
        """
        # Merge cached inputs with provided ones
        merged_kwargs = {**self._input_cache, **kwargs}

        # Return cached output if available
        if self.cache_outputs:
            cache_key = self._hash_inputs(merged_kwargs)
            if cache_key in self._output_cache:
                return {**merged_kwargs, **self._output_cache[cache_key]}

        # Filter kwargs to match the valid keys of the `call` method
        filtered_kwargs = {
            k: v for k, v in merged_kwargs.items() if k in self._valid_keys
        }

        # Call the processing function
        processed_output = self.call(**filtered_kwargs)

        # Ensure the output is always a dictionary
        if not isinstance(processed_output, dict):
            raise TypeError(
                f"The `call` method must return a dictionary. Got {type(processed_output)}."
            )

        # Merge outputs with inputs
        combined_kwargs = {**merged_kwargs, **processed_output}

        # Cache the result if caching is enabled
        if self.cache_outputs:
            if isinstance(self.cache_outputs, list):
                cached_output = {
                    k: v for k, v in processed_output.items() if k in self.cache_outputs
                }
            else:
                cached_output = processed_output
            self._output_cache[cache_key] = cached_output

        return combined_kwargs


class Operation_keras(keras.Operation):
    def __init__(
        self, cache_outputs: bool = False, dtype: Any = None, name: str = None
    ):
        super().__init__(dtype=dtype, name=name)

        self.cache_outputs = cache_outputs
        self.input_cache = {}
        self.output_cache = {}
        self.valid_keys = None
        self._trace_signatures()

    def _trace_signatures(self):
        """
        Analyze and store the input/output signatures of the `process` method.
        """
        self.input_signature = inspect.signature(self.call)
        # Extract valid keys from the signature for filtering
        self.valid_keys = set(self.input_signature.parameters.keys())

    def set_cache(self, cache: Dict[str, Any]):
        """
        Set a cache for inputs or outputs, then retrace the function if necessary.

        :param cache: A dictionary containing cached inputs and/or outputs.
        """
        self.input_cache.update(cache.get("inputs", {}))
        self.output_cache.update(cache.get("outputs", {}))
        self._trace_signatures()  # Retrace after updating cache to ensure correctness.

    @abstractmethod
    def call(self, **kwargs):
        """
        Abstract method that defines the processing logic for the operation.
        Subclasses must implement this method.
        """
        pass

    def __call__(self, **kwargs) -> Dict:
        """
        Process the input keyword arguments and return the processed results.

        :param kwargs: Keyword arguments to be processed.
        :return: Combined input and output as kwargs.
        """
        # Merge cached inputs with provided ones
        merged_kwargs = {**self.input_cache, **kwargs}

        # Return cached output if available
        if self.cache_outputs:

            cache_key = self._hash_inputs(merged_kwargs)

            if cache_key in self.output_cache:
                return {**merged_kwargs, **self.output_cache[cache_key]}

        # Filter kwargs to match the valid keys of the `process` method
        filtered_kwargs = {
            k: v for k, v in merged_kwargs.items() if k in self.valid_keys
        }

        # Call the processing function
        processed_output = self.call(**filtered_kwargs)

        # Ensure the output is always a dictionary
        # if not isinstance(processed_output, dict):
        #     raise TypeError(f"The `process` method must return a dictionary. Got {type(processed_output)}.")

        # Merge outputs with inputs
        combined_kwargs = {**merged_kwargs, **processed_output}

        # Cache the result if caching is enabled
        if self.cache_outputs:
            self.output_cache[cache_key] = processed_output

        return combined_kwargs

    def _hash_inputs(self, kwargs: Dict) -> str:
        """
        Generate a hash for the given inputs to use as a cache key.

        :param kwargs: Keyword arguments.
        :return: A unique hash representing the inputs.
        """
        input_json = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.md5(input_json.encode()).hexdigest()


class Pipeline:
    """
    A modular and flexible data pipeline class.
    """

    def __init__(self):
        """
        Initialize an empty pipeline.
        """
        self.operations: List[Operation] = []

    def add_operation(self, operation: Operation):
        """
        Add an operation to the pipeline.

        :param operation: An instance of the Operation class.
        """
        self.operations.append(operation)

    def run(self, **kwargs) -> Dict:
        """
        Execute all operations in the pipeline sequentially.

        :param kwargs: Initial keyword arguments.
        :return: Final processed keyword arguments.
        """
        for operation in self.operations:
            kwargs = operation(**kwargs)  # Only kwargs are passed and returned
        return kwargs


class PipelineModel(keras.models.Model):
    def __init__(self, pipeline: Pipeline, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline

    def call(self, **inputs) -> Dict:
        # Assuming inputs is a dictionary of inputs required by the pipeline
        outputs = self.pipeline.run(**inputs)
        return outputs


### TESTS ###


def jit_compile(func):
    """
    Applies JIT compilation to the given function based on the current Keras backend.

    Args:
        func (callable): The function to be JIT compiled.

    Returns:
        callable: The JIT-compiled function.

    Raises:
        ValueError: If the backend is unsupported or if the necessary libraries are not installed.
    """
    backend = os.environ.get("KERAS_BACKEND", "tensorflow")

    if backend == "tensorflow":
        try:
            import tensorflow as tf

            return tf.function(func, jit_compile=True)
        except ImportError:
            raise ImportError(
                "TensorFlow is not installed. Please install it to use this backend."
            )
    elif backend == "jax":
        try:
            import jax

            return jax.jit(func)
        except ImportError:
            raise ImportError(
                "JAX is not installed. Please install it to use this backend."
            )
    else:
        print(
            f"Unsupported backend: {backend}. Supported backends are 'tensorflow' and 'jax'."
        )
        print("Falling back to non-compiled mode.")
        return func


class MultiplyOperation(Operation):
    def call(self, x, factor=1):
        """
        Multiplies the input x by the specified factor.
        """
        # print(f"Processing MultiplyOperation: x={x}, factor={factor}")
        return {"result": keras.ops.multiply(x, factor)}


class AddOperation(Operation):
    def call(self, result, y):
        """
        Adds the result from MultiplyOperation with y.
        """
        # print(f"Processing AddOperation: result={result}, y={y}")
        return {"final_result": keras.ops.add(result, y)}


class LargeMatrixMultiplicationOperation(Operation):
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
    def call(self, matrix, scalar):
        """
        Performs elementwise operations on a matrix (adds and multiplies by scalar).
        """
        # print("Processing ElementwiseMatrixOperation...")
        # Perform elementwise addition and multiplication
        result = keras.ops.add(matrix, scalar)
        result = keras.ops.multiply(result, scalar)
        return {"elementwise_result": result}


def test_pipeline_with_gpu_operations():
    """
    A test function to validate the pipeline with GPU-heavy operations and measure execution times.
    """
    # Initialize matrices
    matrix_size = 2048
    matrix_a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    matrix_b = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    scalar = 2.5

    matrix_a = keras.ops.convert_to_tensor(matrix_a)
    matrix_b = keras.ops.convert_to_tensor(matrix_b)

    # framework warm-up
    _ = keras.ops.matmul(matrix_a, matrix_b)
    _ = jit_compile(keras.ops.matmul)(matrix_a, matrix_b)

    # Create operations
    multiply_op = MultiplyOperation(cache_outputs=False)
    add_op = AddOperation(cache_outputs=False)
    large_matmul_op = LargeMatrixMultiplicationOperation(cache_outputs=False)
    elementwise_op = ElementwiseMatrixOperation(cache_outputs=False)

    # Create a pipeline and add operations
    pipeline = Pipeline()
    pipeline.add_operation(multiply_op)
    pipeline.add_operation(add_op)
    pipeline.add_operation(large_matmul_op)
    pipeline.add_operation(elementwise_op)

    # Define the run function
    def run_pipeline():
        pipeline.run(
            x=2,
            factor=3,
            y=5,
            matrix_a=matrix_a,
            matrix_b=matrix_b,
            matrix=matrix_a,
            scalar=scalar,
        )

    run_pipeline = jit_compile(run_pipeline)

    # Timing the pipeline
    print("\nTiming the pipeline:")

    N = 20  # Number of iterations for timing

    # No cache
    print("\nNo cache:")
    time = timeit.timeit(run_pipeline, number=N)
    print(f"Time per run: {time/N:.4f} seconds")

    # With cache
    multiply_op.cache_outputs = True
    add_op.cache_outputs = True
    large_matmul_op.cache_outputs = True
    elementwise_op.cache_outputs = False

    print("\nWith cache:")
    run_pipeline()  # Warm-up run
    time = timeit.timeit(run_pipeline, number=N)
    print(f"Time per run: {time/N:.4f} seconds")

    # With cache and different inputs
    def run_pipeline_different_inputs():
        pipeline.run(
            x=2,
            factor=4,
            y=5,
            matrix_a=matrix_a,
            matrix_b=matrix_b,
            matrix=matrix_a,
            scalar=scalar,
        )

    run_pipeline_different_inputs = jit_compile(run_pipeline_different_inputs)

    print("\nWith cache (different inputs):")
    run_pipeline_different_inputs()  # Warm-up run
    time = timeit.timeit(run_pipeline_different_inputs, number=N)
    print(f"Time per run: {time/N:.4f} seconds")

    # test model
    from ops import PipelineModel

    print("\n Without cache, keras model:")
    multiply_op = MultiplyOperation(cache_outputs=False)
    add_op = AddOperation(cache_outputs=False)
    large_matmul_op = LargeMatrixMultiplicationOperation(cache_outputs=False)
    elementwise_op = ElementwiseMatrixOperation(cache_outputs=False)

    pipeline = Pipeline()
    pipeline.add_operation(multiply_op)
    pipeline.add_operation(add_op)
    pipeline.add_operation(large_matmul_op)
    pipeline.add_operation(elementwise_op)
    model = PipelineModel(pipeline)

    inputs = {
        "x": 2,
        "factor": 3,
        "y": 5,
        "matrix_a": matrix_a,
        "matrix_b": matrix_b,
        "matrix": matrix_a,
        "scalar": scalar,
    }

    def convert_dict_to_tensor(inputs):
        return {k: keras.ops.convert_to_tensor(v) for k, v in inputs.items()}

    inputs = convert_dict_to_tensor(inputs)

    from time import perf_counter, sleep

    _ = model(**inputs)  # Warm-up run
    start = perf_counter()
    for _ in range(20):
        outputs = model(**inputs)
    end = perf_counter()
    print(f"Time per run: {(end - start) / 100:.4f} seconds")

    import tensorflow as tf

    # run in async scope
    model = jit_compile(model)
    _ = model(**inputs)  # Warm-up run
    start = perf_counter()
    for _ in range(20):
        outputs = model(**inputs)
    end = perf_counter()
    print(f"Time per run compiled: {(end - start) / 100:.4f} seconds")


if __name__ == "__main__":
    test_pipeline_with_gpu_operations()
