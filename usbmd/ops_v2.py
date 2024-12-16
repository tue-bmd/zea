""" Experimental version of the USBMD ops module"""

# pylint: disable=arguments-differ

import enum
import hashlib
import importlib
import inspect
import json
import os
import timeit
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Dict, List, Union

import numpy as np

from usbmd.utils import log
from usbmd.backend import jit

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


# _DATA_TYPES = [
#     "raw_data",
#     "aligned_data",
#     "beamformed_data",
#     "envelope_data",
#     "image",
#     "image_sc",
# ]


class DataTypes(enum.Enum):
    """Enum class for USBMD data types."""

    RAW_DATA = "raw_data"
    ALIGNED_DATA = "aligned_data"
    BEAMFORMED_DATA = "beamformed_data"
    ENVELOPE_DATA = "envelope_data"
    IMAGE = "image"
    IMAGE_SC = "image_sc"


# TODO: check if inheriting from keras.Operation is better than using the ABC class.
class Operation(keras.Operation):
    """
    A base abstract class for operations in the pipeline with caching functionality.
    """

    def __init__(
        self,
        input_data_type: Union[DataTypes, None] = None,
        output_data_type: Union[DataTypes, None] = None,
        cache_inputs: Union[bool, List[str]] = False,
        cache_outputs: bool = False,
        jit_compile: bool = True,
    ):
        """
        args:
            cache_inputs: A list of input keys to cache or True to cache all inputs
            cache_outputs: A list of output keys to cache or True to cache all outputs
            jit_compile: Whether to JIT compile the 'call' method for faster execution
        """
        super().__init__()

        self.input_data_type = input_data_type
        self.output_data_type = output_data_type
        self.cache_inputs = cache_inputs
        self.cache_outputs = cache_outputs

        self._jit_compile = jit_compile

        # Initialize input and output caches
        self._input_cache = {}
        self._output_cache = {}

        # Obtain the input signature of the `call` method
        self._input_signature = None
        self._valid_keys = None  # Keys valid for the `call` method
        self._trace_signatures()

        # Compile the `call` method if necessary
        self._call = jit(self.call) if self.jit_compile else self.call

    def set_jit(self, jit_compile: bool):
        """Set the JIT compilation flag and set the `_call` method accordingly."""
        self._jit_compile = jit_compile
        self._call = jit(self.call) if self._jit_compile else self.call

    def _trace_signatures(self):
        """
        Analyze and store the input/output signatures of the `call` method.
        """
        self._input_signature = inspect.signature(self.call)
        self._valid_keys = set(self._input_signature.parameters.keys())

    def call(self, *args, **kwargs):
        """
        Abstract method that defines the processing logic for the operation.
        Subclasses must implement this method.
        """
        raise NotImplementedError

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


class Pipeline(keras.Pipeline):
    """Pipeline class for processing ultrasound data through a series of operations."""

    def __init__(
        self,
        operations: List[Operation],
        with_batch_dim: bool = True,
        device: Union[str, None] = None,
        jit_options: Union[str, None] = "ops",
    ):
        """Initialize a pipeline

        Args:
            operations (list): A list of Operation instances representing the operations
                to be performed.
            with_batch_dim (bool, optional): Whether to include batch dimension in the operations.
                Defaults to True.
            device (str, optional): The device to use for the operations. Defaults to None.
                Can be `cpu` or `cuda`, `cuda:0`, etc.
            jit_options (str, optional): The JIT options to use. Must be "pipeline", "ops", or None.
            - "pipeline" compiles the entire pipeline as a single function. This may be faster but,
            does not preserve python control flow, such as caching.
            - "ops" compiles each operation separately. This preserves python control flow and
            caching functionality, but speeds up the operations.
            - None disables JIT compilation.
            Defaults to "ops".
        """
        super().__init__()

        if jit_options not in ["pipeline", "ops", None]:
            raise ValueError("jit_options must be 'pipeline', 'ops', or None")

        self.operations = operations
        self.device = self._check_device(device)

        for operation in self.operations:
            operation.with_batch_dim = with_batch_dim
            operation.set_jit(True if jit_options == "ops" else False)

        self.validate()

        self._call = jit(self.call) if jit_options == "pipeline" else self.call

    def call(self, *args, return_numpy=False, **kwargs):
        """Process input data through the pipeline."""

        # TODO: convert args and kwargs to tensors

        if self._jitted_process is None:
            processing_func = self._process
        else:
            processing_func = self._jitted_process

        kwargs = self.

        if return_numpy:
            return keras.ops.convert_to_numpy(data_out)
        return data_out

    def prepare_input(self, *args)
        """ Convert input data and parameters to dictionary of tensors following the CCC"""


        return kwargs

    def prepare_output(self, kwargs):
        """ Convert output data to dictionary of tensors following the CCC"""

        return data_out

    def run(self, *args, **kwargs):
        """Execute all operations in the pipeline"""

        # TODO: compatiblity with Stack operation
        for operation in self.operations:
            kwargs = operation(*args, **kwargs)  # TODO: check if args are needed
        return kwargs

    @property
    def with_batch_dim(self):
        """Get the with_batch_dim property of the pipeline."""
        return self.operations[0].with_batch_dim

    def validate(self):
        """Validate the pipeline by checking the compatibility of the operations."""
        for i in range(len(self.operations) - 1):
            if self.operations[i].output_data_type is None:
                continue
            if self.operations[i + 1].input_data_type is None:
                continue
            if (
                self.operations[i].output_data_type
                != self.operations[i + 1].input_data_type
            ):
                raise ValueError(
                    f"Operation {self.operations[i].name} output data type is not compatible "
                    f"with the input data type of operation {self.operations[i + 1].name}"
                )

    # TODO: Ben: is it actually possible to change backend at runtime?
    # Should this be handled by the pipeline or at a higher level?
    def on_device(self, func, data, device=None, return_numpy=False):
        """On device function for running pipeline on specific device."""
        backend = keras.backend.backend()
        if backend == "numpy":
            return func(data)
        elif backend == "tensorflow":
            on_device_tf = importlib.import_module(
                "usbmd.backend.tensorflow"
            ).on_device_tf
            return on_device_tf(func, data, device=device, return_numpy=return_numpy)
        elif backend == "torch":
            on_device_torch = importlib.import_module(
                "usbmd.backend.torch"
            ).on_device_torch
            return on_device_torch(func, data, device=device, return_numpy=return_numpy)
        elif backend == "jax":
            on_device_jax = importlib.import_module("usbmd.backend.jax").on_device_jax
            return on_device_jax(func, data, device=device, return_numpy=return_numpy)
        else:
            raise ValueError(f"Unsupported operations package {backend}.")

    def set_params(self, **params):
        """Set parameters for the operations in the pipeline by adding them to the cache."""
        raise NotImplementedError

    def _process(self, data):
        for operation in self.operations:
            if isinstance(data, list) and operation.__class__.__name__ != "Stack":
                data = [operation(_data) for _data in data]
            else:
                data = operation(data)
        return data

    def prepare_tensor(self, x, dtype=None, device=None):
        """Convert input array to appropriate tensor type for the operations package."""
        if len(self.operations) == 0:
            return x
        return self.operations[0].prepare_tensor(x, dtype=dtype, device=device)

    def __str__(self):
        """String representation of the pipeline.

        Will print on two parallel pipeline lines if it detects a splitting operations
        (such as multi_bandpass_filter)
        Will merge the pipeline lines if it detects a stacking operation (such as stack)
        """
        split_operations = ["MultiBandPassFilter"]
        merge_operations = ["Stack"]

        operations = [operation.__class__.__name__ for operation in self.operations]
        string = " -> ".join(operations)

        if any(operation in split_operations for operation in operations):
            # a second line is needed with same length as the first line
            split_line = " " * len(string)
            # find the splitting operation and index and print \-> instead of -> after
            split_detected = False
            merge_detected = False
            split_operation = None
            for operation in operations:
                if operation in split_operations:
                    index = string.index(operation)
                    index = index + len(operation)
                    split_line = (
                        split_line[:index] + "\\->" + split_line[index + len("\\->") :]
                    )
                    split_detected = True
                    merge_detected = False
                    split_operation = operation
                    continue

                if operation in merge_operations:
                    index = string.index(operation)
                    index = index - 4
                    split_line = split_line[:index] + "/" + split_line[index + 1 :]
                    split_detected = False
                    merge_detected = True
                    continue

                if split_detected:
                    # print all operations in the second line
                    index = string.index(operation)
                    split_line = (
                        split_line[:index]
                        + operation
                        + " -> "
                        + split_line[index + len(operation) + len(" -> ") :]
                    )
            assert merge_detected is True, log.error(
                "Pipeline was never merged back together (with Stack operation), even "
                f"though it was split with {split_operation}. "
                "Please properly define your operation chain."
            )
            return f"\n{string}\n{split_line}\n"

        return string

    def __repr__(self):
        """String representation of the pipeline."""
        operations = [operation.__class__.__name__ for operation in self.operations]
        return ",".join(operations)

    def _check_device(self, device):
        if device is None:
            return None

        if device == "cpu":
            return "cpu"

        backend = keras.backend.backend()

        if backend == "numpy":
            if device not in [None, "cpu"]:
                log.warning(
                    f"Device {device} is not supported for numpy operations, using cpu."
                )
            return "cpu"

        else:
            # assert device to be cpu, cuda, cuda:{int} or int or None
            assert isinstance(
                device, (str, int)
            ), f"device should be a string or int, got {device}"
            if isinstance(device, str):
                if backend == "tensorflow":
                    assert device.startswith(
                        "gpu"
                    ), f"device should be 'cpu' or 'gpu:*', got {device}"
                elif backend == "torch":
                    assert device.startswith(
                        "cuda"
                    ), f"device should be 'cpu' or 'cuda:*', got {device}"
                elif backend == "jax":
                    assert device.startswith(
                        ("gpu", "cuda")
                    ), f"device should be 'cpu', 'gpu:*', or 'cuda:*', got {device}"
                else:
                    raise ValueError(f"Unsupported backend {backend}.")
            return device


# class Pipeline:
#     """
#     A modular and flexible data pipeline class.
#     """

#     def __init__(self):
#         """
#         Initialize an empty pipeline.
#         """
#         self.operations: List[Operation] = []

#     def add_operation(self, operation: Operation):
#         """
#         Add an operation to the pipeline.

#         :param operation: An instance of the Operation class.
#         """
#         self.operations.append(operation)

#     def run(self, **kwargs) -> Dict:
#         """
#         Execute all operations in the pipeline sequentially.

#         :param kwargs: Initial keyword arguments.
#         :return: Final processed keyword arguments.
#         """
#         for operation in self.operations:
#             kwargs = operation(**kwargs)  # Only kwargs are passed and returned
#         return kwargs


## Helper functions




## Operations


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


class Stack(Operation):
    """Stack multiple data arrays along a new axis.
    Useful to merge data from parallel pipelines.
    """

    def __init__(
        self,
        keys: Union[str, List[str], None] = None,
        axis: Union[int, List[int], None] = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.keys = keys
        self.axis = axis

    def call(self, **kwargs) -> Dict:
        """
        Stacks the inputs corresponding to the specified keys along the specified axis.
        If a list of axes is provided, the length must match the number of keys.
        If an integer axis is provided, all inputs are stacked along the same axis.
        """

        raise NotImplementedError


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
