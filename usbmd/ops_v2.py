""" Experimental version of the USBMD ops module"""

# pylint: disable=arguments-differ

import hashlib
import inspect
import json
import os
import timeit
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Dict, List, Union

import numpy as np

# Set the Keras backend
# os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["KERAS_BACKEND"] = "torch"
# os.environ["KERAS_BACKEND"] = "numpy"
import keras

print("WARNING: This module is work in progress and may not work as expected!")


# TODO: jit_compile should allow for 2 different modes:
# 1. Operation-based: each operation is compiled separately by setting Operation(jit_compile=True).
# This means the __call__ method is not compiled and most of the usbmd logic can be executed on the
# fly, preserving the caching functionality. (DONE)
# 2. Pipeline-based: the entire pipeline is compiled by setting Pipeline(jit_compile=True).
# This means the entire pipeline is compiled and executed as a single function, which may be faster
# but may not preserve the caching functionality (need to check).


# TODO: check if inheriting from keras.Operation is better than using the ABC class.
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
        self._call = jit(self.call) if self.jit_compile else self.call

    def _trace_signatures(self):
        """
        Analyze and store the input/output signatures of the `call` method.
        """
        self._input_signature = inspect.signature(self.call)
        self._valid_keys = set(self._input_signature.parameters.keys())

    @abstractmethod
    def call(self, *args, **kwargs):
        """
        Abstract method that defines the processing logic for the operation.
        Subclasses must implement this method.
        """

    def set_input_cache(self, input_cache: Dict[str, Any]):
        """
        Set a cache for inputs, then retrace the function if necessary.

        args:
            input_cache: A dictionary containing cached inputs.
        """
        self._input_cache.update(input_cache)
        self._trace_signatures()  # Retrace after updating cache to ensure correctness.

    def set_output_cache(self, output_cache: Dict[str, Any]):
        """
        Set a cache for outputs, then retrace the function if necessary.

        args:
            output_cache: A dictionary containing cached outputs.
        """
        self._output_cache.update(output_cache)
        self._trace_signatures()  # Retrace after updating cache to ensure correctness.

    def clear_cache(self):
        """
        Clear the input and output caches.
        """
        self._input_cache.clear()
        self._output_cache.clear()

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
        processed_output = self._call(**filtered_kwargs)

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
    """Test pipeline that inherits from Keras Model."""

    def __init__(self, pipeline: Pipeline, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline

    def call(self, **inputs) -> Dict:
        # Assuming inputs is a dictionary of inputs required by the pipeline
        outputs = self.pipeline.run(**inputs)
        return outputs


### TESTS ###


def jit(func):
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
            import tensorflow as tf  # pylint: disable=import-outside-toplevel

            return tf.function(func, jit_compile=True)
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is not installed. Please install it to use this backend."
            ) from exc
    elif backend == "jax":
        try:
            import jax  # pylint: disable=import-outside-toplevel

            return jax.jit(func)
        except ImportError as exc:
            raise ImportError(
                "JAX is not installed. Please install it to use this backend."
            ) from exc
    else:
        print(
            f"Unsupported backend: {backend}. Supported backends are 'tensorflow' and 'jax'."
        )
        print("Falling back to non-compiled mode.")
        return func


class Merge(Operation):
    """Operation that merges sets of input dictionaries."""

    def call(self, *args) -> Dict:
        """
        Merges the input dictionaries. Priority is given to the last input.
        """
        merged = {}
        for arg in args:
            if not isinstance(arg, dict):
                raise TypeError("All inputs must be dictionaries.")
            merged.update(arg)
        return merged


# def test_pipeline_with_gpu_operations():
#     """
#     A test function to validate the pipeline with GPU-heavy operations and measure execution times.
#     """
#     # Initialize matrices
#     matrix_size = 2048
#     matrix_a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
#     matrix_b = np.random.rand(matrix_size, matrix_size).astype(np.float32)
#     scalar = 2.5

#     matrix_a = keras.ops.convert_to_tensor(matrix_a)
#     matrix_b = keras.ops.convert_to_tensor(matrix_b)

#     # framework warm-up
#     _ = keras.ops.matmul(matrix_a, matrix_b)
#     _ = jit(keras.ops.matmul)(matrix_a, matrix_b)

#     # Create operations
#     multiply_op = MultiplyOperation(cache_outputs=False)
#     add_op = AddOperation(cache_outputs=False)
#     large_matmul_op = LargeMatrixMultiplicationOperation(cache_outputs=False)
#     elementwise_op = ElementwiseMatrixOperation(cache_outputs=False)

#     # Create a pipeline and add operations
#     pipeline = Pipeline()
#     pipeline.add_operation(multiply_op)
#     pipeline.add_operation(add_op)
#     pipeline.add_operation(large_matmul_op)
#     pipeline.add_operation(elementwise_op)

#     # Define the run function
#     def run_pipeline():
#         pipeline.run(
#             x=2,
#             factor=3,
#             y=5,
#             matrix_a=matrix_a,
#             matrix_b=matrix_b,
#             matrix=matrix_a,
#             scalar=scalar,
#         )

#     run_pipeline = jit(run_pipeline)

#     # Timing the pipeline
#     print("\nTiming the pipeline:")

#     N = 20  # Number of iterations for timing

#     # No cache
#     print("\nNo cache:")
#     time = timeit.timeit(run_pipeline, number=N)
#     print(f"Time per run: {time/N:.4f} seconds")

#     # With cache
#     multiply_op.cache_outputs = True
#     add_op.cache_outputs = True
#     large_matmul_op.cache_outputs = True
#     elementwise_op.cache_outputs = False

#     print("\nWith cache:")
#     run_pipeline()  # Warm-up run
#     time = timeit.timeit(run_pipeline, number=N)
#     print(f"Time per run: {time/N:.4f} seconds")

#     # With cache and different inputs
#     def run_pipeline_different_inputs():
#         pipeline.run(
#             x=2,
#             factor=4,
#             y=5,
#             matrix_a=matrix_a,
#             matrix_b=matrix_b,
#             matrix=matrix_a,
#             scalar=scalar,
#         )

#     run_pipeline_different_inputs = jit(run_pipeline_different_inputs)

#     print("\nWith cache (different inputs):")
#     run_pipeline_different_inputs()  # Warm-up run
#     time = timeit.timeit(run_pipeline_different_inputs, number=N)
#     print(f"Time per run: {time/N:.4f} seconds")

#     print("\n Without cache, keras model:")
#     multiply_op = MultiplyOperation(cache_outputs=False)
#     add_op = AddOperation(cache_outputs=False)
#     large_matmul_op = LargeMatrixMultiplicationOperation(cache_outputs=False)
#     elementwise_op = ElementwiseMatrixOperation(cache_outputs=False)

#     pipeline = Pipeline()
#     pipeline.add_operation(multiply_op)
#     pipeline.add_operation(add_op)
#     pipeline.add_operation(large_matmul_op)
#     pipeline.add_operation(elementwise_op)
#     model = PipelineModel(pipeline)

#     inputs = {
#         "x": 2,
#         "factor": 3,
#         "y": 5,
#         "matrix_a": matrix_a,
#         "matrix_b": matrix_b,
#         "matrix": matrix_a,
#         "scalar": scalar,
#     }

#     def convert_dict_to_tensor(inputs):
#         return {k: keras.ops.convert_to_tensor(v) for k, v in inputs.items()}

#     inputs = convert_dict_to_tensor(inputs)

#     _ = model(**inputs)  # Warm-up run
#     start = perf_counter()
#     for _ in range(20):
#         _ = model(**inputs)
#     end = perf_counter()
#     print(f"Time per run: {(end - start) / 100:.4f} seconds")

#     # run in async scope
#     model = jit(model)
#     _ = model(**inputs)  # Warm-up run
#     start = perf_counter()
#     for _ in range(20):
#         _ = model(**inputs)
#     end = perf_counter()
#     print(f"Time per run compiled: {(end - start) / 100:.4f} seconds")


# if __name__ == "__main__":
#     test_pipeline_with_gpu_operations()
