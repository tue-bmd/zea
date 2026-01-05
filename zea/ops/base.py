import hashlib
import inspect
import json
from functools import partial
from typing import Any, Dict, List, Union

import keras
from keras import ops

from zea import log
from zea.backend import jit
from zea.internal.checks import _assert_keys_and_axes
from zea.internal.core import (
    DataTypes,
)
from zea.internal.registry import ops_registry
from zea.utils import (
    deep_compare,
)


def get_ops(ops_name):
    """Get the operation from the registry."""
    return ops_registry[ops_name]


class Operation(keras.Operation):
    """
    A base abstract class for operations in the pipeline with caching functionality.
    """

    def __init__(
        self,
        input_data_type: Union[DataTypes, None] = None,
        output_data_type: Union[DataTypes, None] = None,
        key: Union[str, None] = "data",
        output_key: Union[str, None] = None,
        cache_inputs: Union[bool, List[str]] = False,
        cache_outputs: bool = False,
        jit_compile: bool = True,
        with_batch_dim: bool = True,
        jit_kwargs: dict | None = None,
        jittable: bool = True,
        additional_output_keys: List[str] = None,
        **kwargs,
    ):
        """
        Args:
            input_data_type (DataTypes): The data type of the input data
            output_data_type (DataTypes): The data type of the output data
            key: The key for the input data (operation will operate on this key)
                Defaults to "data".
            output_key: The key for the output data (operation will output to this key)
                Defaults to the same as the input key. If you want to store intermediate
                results, you can set this to a different key. But make sure to update the
                input key of the next operation to match the output key of this operation.
            cache_inputs: A list of input keys to cache or True to cache all inputs
            cache_outputs: A list of output keys to cache or True to cache all outputs
            jit_compile: Whether to JIT compile the 'call' method for faster execution
            with_batch_dim: Whether operations should expect a batch dimension in the input
            jit_kwargs: Additional keyword arguments for the JIT compiler
            jittable: Whether the operation can be JIT compiled
            additional_output_keys: A list of additional output keys produced by the operation.
                These are used to track if all keys are available for downstream operations.
                If the operation has a conditional output, it is best to add all possible
                output keys here.
        """
        super().__init__(**kwargs)

        self.input_data_type = input_data_type
        self.output_data_type = output_data_type

        self.key = key  # Key for input data
        self.output_key = output_key  # Key for output data
        if self.output_key is None:
            self.output_key = self.key
        self.additional_output_keys = additional_output_keys or []

        self.inputs = []  # Source(s) of input data (name of a previous operation)
        self.allow_multiple_inputs = False  # Only single input allowed by default

        self.cache_inputs = cache_inputs
        self.cache_outputs = cache_outputs

        # Initialize input and output caches
        self._input_cache = {}
        self._output_cache = {}

        # Obtain the input signature of the `call` method
        self._trace_signatures()

        if jit_kwargs is None:
            jit_kwargs = {}

        if keras.backend.backend() == "jax" and self.static_params:
            jit_kwargs |= {"static_argnames": self.static_params}

        self.jit_kwargs = jit_kwargs

        self.with_batch_dim = with_batch_dim
        self._jittable = jittable

        # Set the jit compilation flag and compile the `call` method
        # Set zea logger level to suppress warnings regarding
        # torch not being able to compile the function
        with log.set_level("ERROR"):
            self.set_jit(jit_compile)

    @property
    def output_keys(self) -> List[str]:
        """Get the output keys of the operation."""
        return [self.output_key] + self.additional_output_keys

    @property
    def static_params(self):
        """Get the static parameters of the operation."""
        return getattr(self.__class__, "STATIC_PARAMS", [])

    def set_jit(self, jit_compile: bool):
        """Set the JIT compilation flag and set the `_call` method accordingly."""
        self._jit_compile = jit_compile
        if self._jit_compile and self.jittable:
            self._call = jit(self.call, **self.jit_kwargs)
        else:
            self._call = self.call

    def _trace_signatures(self):
        """
        Analyze and store the input/output signatures of the `call` method.
        """
        self._input_signature = inspect.signature(self.call)
        self._valid_keys = set(self._input_signature.parameters.keys()) | {self.key}

    @property
    def valid_keys(self) -> set:
        """Get the valid keys for the `call` method."""
        return self._valid_keys

    @property
    def needs_keys(self) -> set:
        """Get a set of all input keys needed by the operation."""
        return self.valid_keys

    @property
    def jittable(self):
        """Check if the operation can be JIT compiled."""
        return self._jittable

    def call(self, **kwargs):
        """
        Abstract method that defines the processing logic for the operation.
        Subclasses must implement this method.
        """
        raise NotImplementedError

    def set_input_cache(self, input_cache: Dict[str, Any]):
        """
        Set a cache for inputs, then retrace the function if necessary.

        Args:
            input_cache: A dictionary containing cached inputs.
        """
        self._input_cache.update(input_cache)
        self._trace_signatures()  # Retrace after updating cache to ensure correctness.

    def set_output_cache(self, output_cache: Dict[str, Any]):
        """
        Set a cache for outputs, then retrace the function if necessary.

        Args:
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

        Args:
            kwargs: Keyword arguments.

        Returns:
            A unique hash representing the inputs.
        """
        input_json = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.md5(input_json.encode()).hexdigest()

    def __call__(self, *args, **kwargs) -> Dict:
        """
        Process the input keyword arguments and return the processed results.

        Args:
            kwargs: Keyword arguments to be processed.

        Returns:
            Combined input and output as kwargs.
        """
        if args:
            example_usage = f"    result = {ops_registry.get_name(self)}({self.key}=my_data"
            valid_keys_no_kwargs = self.valid_keys - {"kwargs"}
            if valid_keys_no_kwargs:
                example_usage += f", {list(valid_keys_no_kwargs)[0]}=param1, ..., **kwargs)"
            else:
                example_usage += ", **kwargs)"
            raise TypeError(
                f"{self.__class__.__name__}.__call__() only accepts keyword arguments. "
                "Positional arguments are not allowed.\n"
                f"Received positional arguments: {args}\n"
                "Example usage:\n"
                f"{example_usage}"
            )

        # Merge cached inputs with provided ones
        merged_kwargs = {**self._input_cache, **kwargs}

        # Return cached output if available
        if self.cache_outputs:
            cache_key = self._hash_inputs(merged_kwargs)
            if cache_key in self._output_cache:
                return {**merged_kwargs, **self._output_cache[cache_key]}

        # Filter kwargs to match the valid keys of the `call` method
        if "kwargs" not in self.valid_keys:
            filtered_kwargs = {k: v for k, v in merged_kwargs.items() if k in self.valid_keys}
        else:
            filtered_kwargs = merged_kwargs

        # Call the processing function
        # If you want to jump in with debugger please set `jit_compile=False`
        # when initializing the pipeline.
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

    def get_dict(self):
        """Get the configuration of the operation. Inherit from keras.Operation."""
        config = {}
        config.update({"name": ops_registry.get_name(self)})
        config["params"] = {
            "key": self.key,
            "output_key": self.output_key,
            "cache_inputs": self.cache_inputs,
            "cache_outputs": self.cache_outputs,
            "jit_compile": self._jit_compile,
            "with_batch_dim": self.with_batch_dim,
            "jit_kwargs": self.jit_kwargs,
        }
        return config

    def __eq__(self, other):
        """Check equality of two operations based on type and configuration."""
        if not isinstance(other, Operation):
            return False

        # Compare the class name and parameters
        if self.__class__.__name__ != other.__class__.__name__:
            return False

        # Compare the name assigned to the operation
        name = ops_registry.get_name(self)
        other_name = ops_registry.get_name(other)
        if name != other_name:
            return False

        # Compare the parameters of the operations
        if not deep_compare(self.get_dict(), other.get_dict()):
            return False

        return True


@ops_registry("identity")
class Identity(Operation):
    """Identity operation."""

    def call(self, **kwargs) -> Dict:
        """Returns the input as is."""
        return {}


class ImageOperation(Operation):
    """
    Base class for image processing operations.

    This class extends the Operation class to provide a common interface
    for operations that process image data, with shape (batch, height, width, channels)
    or (height, width, channels) if batch dimension is not present.

    Subclasses should implement the `call` method to define the image processing logic, and call
    ``super().call(**kwargs)`` to validate the input data shape.
    """

    def call(self, **kwargs):
        """
        Validate input data shape for image operations.

        Args:
            **kwargs: Keyword arguments containing input data.

        Raises:
            AssertionError: If input data does not have the expected number of dimensions.
        """
        data = kwargs[self.key]

        if self.with_batch_dim:
            assert ops.ndim(data) == 4, "Input data must have 4 dimensions (b, h, w, c)."
        else:
            assert ops.ndim(data) == 3, "Input data must have 3 dimensions (h, w, c)."


@ops_registry("lambda")
class Lambda(Operation):
    """Use any function as an operation."""

    def __init__(self, func, **kwargs):
        # Split kwargs into kwargs for partial and __init__
        sig = inspect.signature(func)
        func_params = set(sig.parameters.keys())

        func_kwargs = {k: v for k, v in kwargs.items() if k in func_params}
        op_kwargs = {k: v for k, v in kwargs.items() if k not in func_params}

        Lambda._check_if_unary(func, **func_kwargs)

        super().__init__(**op_kwargs)
        self.func = partial(func, **func_kwargs)

    @staticmethod
    def _check_if_unary(func, **kwargs):
        """Checks if the kwargs are sufficient to call the function as a unary operation."""
        sig = inspect.signature(func)
        # Remove arguments that are already provided in func_kwargs
        params = list(sig.parameters.values())
        remaining = [p for p in params if p.name not in kwargs]
        # Count required positional arguments (excluding self/cls)
        required_positional = [
            p
            for p in remaining
            if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if len(required_positional) != 1:
            raise ValueError(
                f"Partial of {func.__name__} must be callable with exactly one required "
                f"positional argument, we still need: {required_positional}."
            )

    def call(self, **kwargs):
        data = kwargs[self.key]
        if self.with_batch_dim:
            data = ops.map(self.func, data)
        else:
            data = self.func(data)
        return {self.output_key: data}


@ops_registry("mean")
class Mean(Operation):
    """Take the mean of the input data along a specific axis."""

    def __init__(self, keys, axes, **kwargs):
        super().__init__(**kwargs)

        self.keys, self.axes = _assert_keys_and_axes(keys, axes)

    def call(self, **kwargs):
        for key, axis in zip(self.keys, self.axes):
            kwargs[key] = ops.mean(kwargs[key], axis=axis)

        return kwargs


@ops_registry("merge")
class Merge(Operation):
    """Operation that merges sets of input dictionaries."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.allow_multiple_inputs = True

    def call(self, *args, **kwargs) -> Dict:
        """
        Merges the input dictionaries. Priority is given to the last input.
        """
        merged = {}
        for arg in args:
            if not isinstance(arg, dict):
                raise TypeError("All inputs must be dictionaries.")
            merged.update(arg)
        return merged


@ops_registry("stack")
class Stack(Operation):
    """Stack multiple data arrays along a new axis.
    Useful to merge data from parallel pipelines.
    """

    def __init__(
        self,
        keys: Union[str, List[str], None],
        axes: Union[int, List[int], None],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.keys, self.axes = _assert_keys_and_axes(keys, axes)

    def call(self, **kwargs) -> Dict:
        """
        Stacks the inputs corresponding to the specified keys along the specified axis.
        If a list of axes is provided, the length must match the number of keys.
        """
        for key, axis in zip(self.keys, self.axes):
            kwargs[key] = keras.ops.stack([kwargs[key] for key in self.keys], axis=axis)
        return kwargs
