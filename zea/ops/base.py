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
    map_negative_indices,
)


def get_ops(ops_name):
    """Get the operation from the registry."""
    return ops_registry[ops_name]


def _to_native(value):
    """Convert non-serializable types (e.g. numpy) to native Python equivalents."""
    if hasattr(value, "ndim") and callable(getattr(value, "tolist", None)):
        return value.tolist()
    if isinstance(value, tuple):
        return tuple(_to_native(v) for v in value)
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    return value


class Operation(keras.Operation):
    """
    A base abstract class for operations in the pipeline with caching functionality.
    """

    ADD_OUTPUT_KEYS: List[str] = []

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
        if additional_output_keys is None:
            additional_output_keys = getattr(self.__class__, "ADD_OUTPUT_KEYS", [])
        self.additional_output_keys = (
            list(additional_output_keys) if additional_output_keys is not None else []
        )

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

        self._user_jit_kwargs = jit_kwargs.copy()

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

    def get_dict(self, compact=True):
        """Get the configuration of the operation.

        Args:
            compact (bool): If True (default), only include
                parameters that differ from their defaults.
                If False, include all parameters for full reproducibility.
        """
        config = {"name": ops_registry.get_name(self)}
        params = {}

        # Collect subclass-specific params from the MRO (excluding Operation base)
        base_param_names = set(inspect.signature(Operation.__init__).parameters.keys())
        seen = set()

        for cls in type(self).__mro__:
            if not issubclass(cls, Operation) or cls is Operation:
                continue
            init_fn = cls.__dict__.get("__init__")
            if init_fn is None:
                continue
            for name, param in inspect.signature(init_fn).parameters.items():
                if name == "self" or name in base_param_names or name in seen:
                    continue
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                seen.add(name)

                value = _to_native(getattr(self, name, None))
                if callable(value):
                    if name == "func":
                        op_name = ops_registry.get_name(self)
                        if op_name == "lambda":
                            raise TypeError(
                                "Cannot serialize generic 'lambda' operation with an arbitrary "
                                "callable. Use a registered operation class instead (e.g. "
                                "zea.ops.keras_ops wrappers) or create a custom Operation "
                                "subclass."
                            )
                        continue
                    raise TypeError(
                        f"Parameter '{name}' of '{type(self).__name__}' is callable and cannot "
                        "be serialized to config. Override get_dict() to skip it."
                    )
                if compact:
                    if param.default is inspect.Parameter.empty or value != param.default:
                        params[name] = value
                else:
                    params[name] = value

        # Base Operation parameters
        if compact:
            if self.key != "data":
                params["key"] = self.key
            if self.output_key != self.key:
                params["output_key"] = self.output_key
            if self.cache_inputs:
                params["cache_inputs"] = self.cache_inputs
            if self.cache_outputs:
                params["cache_outputs"] = self.cache_outputs
            if not self._jit_compile:
                params["jit_compile"] = self._jit_compile
            if not self.with_batch_dim:
                params["with_batch_dim"] = self.with_batch_dim
            if self._user_jit_kwargs:
                params["jit_kwargs"] = self._user_jit_kwargs
        else:
            params["key"] = self.key
            params["output_key"] = self.output_key
            params["cache_inputs"] = self.cache_inputs
            params["cache_outputs"] = self.cache_outputs
            params["jit_compile"] = self._jit_compile
            params["with_batch_dim"] = self.with_batch_dim
            params["jit_kwargs"] = self._user_jit_kwargs

        if params:
            config["params"] = params

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


class Filter(Operation):
    def _resolve_filter_axes(self, data, axes=None):
        """
        Resolve the axes to filter over based on the axes parameter and with_batch_dim flag.

        Args:
            data: Input tensor
            axes: Tuple of axes to filter over, or None to filter all (non-batch) axes

        Returns:
            Tuple of resolved axes indices

        Raises:
            ValueError: If batch dimension is included in axes when with_batch_dim is True
        """

        if axes is None:
            if self.with_batch_dim:
                return tuple(range(1, data.ndim))
            else:
                return tuple(range(data.ndim))
        else:
            axes = map_negative_indices(axes, data.ndim)
            if self.with_batch_dim and 0 in axes:
                raise ValueError("Batch dimension cannot be one of the axes to filter over.")
            return axes


@ops_registry("identity")
class Identity(Operation):
    """Identity operation."""

    def call(self, **kwargs) -> Dict:
        """Returns the input as is."""
        return {}


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

    def get_dict(self, compact=True):
        """Serialize lambda-based operations.

        Generic ``zea.ops.Lambda`` instances are intentionally rejected because
        arbitrary callables cannot be reliably serialized. Registered subclasses
        (e.g. ``zea.ops.keras_ops`` wrappers) are serialized by operation name and
        the callable keyword arguments.
        """
        config = super().get_dict(compact=compact)

        func = self.func.func if isinstance(self.func, partial) else self.func
        func_sig = inspect.signature(func)
        func_kwargs = self.func.keywords or {}

        serialized_func_params = {}
        for name, param in func_sig.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if name in func_kwargs:
                serialized_func_params[name] = _to_native(func_kwargs[name])
            elif not compact and param.default is not inspect.Parameter.empty:
                serialized_func_params[name] = _to_native(param.default)

        if serialized_func_params:
            existing_params = config.get("params", {})
            existing_params.update(serialized_func_params)
            config["params"] = existing_params

        return config


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
