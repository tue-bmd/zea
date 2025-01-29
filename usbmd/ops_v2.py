""" Experimental version of the USBMD ops module"""

import hashlib
import importlib
import inspect
import json
from typing import Any, Dict, List, Union

import keras
from keras import ops

from usbmd.backend import jit
from usbmd.config.config import Config
from usbmd.core import DataTypes, Object
from usbmd.probes import Probe
from usbmd.registry import ops_registry
from usbmd.scan import Scan
from usbmd.utils import log

log.warning("WARNING: This module is work in progress and may not work as expected!")

# make sure to reload all modules that import keras
# to be able to set backend properly
# importlib.reload(bmf)
# importlib.reload(pfield)
# importlib.reload(lens_correction)
# importlib.reload(display)

# clear registry upon import
# ops_registry.clear()


def get_ops(ops_name):
    """Get the operation from the registry."""
    return ops_registry[ops_name]


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
        Args:
            cache_inputs: A list of input keys to cache or True to cache all inputs
            cache_outputs: A list of output keys to cache or True to cache all outputs
            jit_compile: Whether to JIT compile the 'call' method for faster execution
        """
        super().__init__()

        self.input_data_type = input_data_type
        self.output_data_type = output_data_type
        self.cache_inputs = cache_inputs
        self.cache_outputs = cache_outputs

        # Initialize input and output caches
        self._input_cache = {}
        self._output_cache = {}

        # Obtain the input signature of the `call` method
        self._input_signature = None
        self._valid_keys = None  # Keys valid for the `call` method
        self._trace_signatures()

        # Set the jit compilation flag and compile the `call` method
        self.set_jit(jit_compile)

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

    def __call__(self, **kwargs) -> Dict:
        """
        Process the input keyword arguments and return the processed results.

        Args:
            kwargs: Keyword arguments to be processed.

        Returns:
            Combined input and output as kwargs.
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
    """Pipeline class for processing ultrasound data through a series of operations."""

    def __init__(
        self,
        operations: List[Operation],
        with_batch_dim: bool = True,
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

        # add functionality here
        # operations = ...

        self._pipeline_layers = operations

        if jit_options not in ["pipeline", "ops", None]:
            raise ValueError("jit_options must be 'pipeline', 'ops', or None")

        for operation in self.operations:  # We use self.layers from keras.Pipeline here
            operation.with_batch_dim = with_batch_dim
            operation.set_jit(jit_options == "ops")

        self.validate()

        # pylint: disable=method-hidden
        self._call_pipeline = jit(self.call) if jit_options == "pipeline" else self.call

    @property
    def operations(self):
        """Alias for self.layers to match the USBMD naming convention"""
        return self._pipeline_layers

    def call(self, inputs):
        """Process input data through the pipeline."""
        for operation in self._pipeline_layers:
            outputs = operation(**inputs)
            inputs = outputs
        return outputs

    def __call__(self, *args, return_numpy=False, **kwargs):
        """Process input data through the pipeline."""

        if "probe" or "scan" or "config" in kwargs:
            raise ValueError(
                "Probe, Scan and Config objects should be passed as positional arguments. "
                "e.g. pipeline(probe, scan, config, **kwargs)"
            )

        # Extract from args Probe, Scan and Config objects
        probe, scan, config = {}, {}, {}
        for arg in args:
            if isinstance(arg, Probe):
                probe = arg.to_tensor()
            elif isinstance(arg, Scan):
                scan = arg.to_tensor()
            elif isinstance(arg, Config):
                config = arg.to_tensor()  # TODO
            else:
                raise ValueError(
                    f"Unsupported input type for pipeline *args: {type(arg)}. "
                    "Pipeline call expects a `usbmd.core.Object` (Probe, Scan, Config) as args. "
                    "Alternatively, pass the input as keyword argument (kwargs)."
                )

        # combine probe, scan, config and kwargs
        # explicitly so we know which keys overwrite which
        # kwargs > config > scan > probe
        inputs = {**probe, **scan, **config, **kwargs}

        ## PROCESSING
        outputs = self._call_pipeline(inputs)

        if return_numpy:
            outputs = {k: v.numpy() for k, v in outputs.items()}

        ## PREPARE OUTPUT

        # TODO: if we can in-place update the Scan, Probe and Config objects, we can output those.

        # update probe, scan, config with outputs
        # for arg in args:
        #     if isinstance(arg, Probe):
        #         arg.update(outputs)
        #     elif isinstance(arg, Scan):
        #         arg.update(outputs)
        #     elif isinstance(arg, Config):
        #         arg.update(outputs)

        return outputs

    def prepare_input(self, *args):
        """Convert input data and parameters to dictionary of tensors following the CCC"""
        raise NotImplementedError

    def prepare_output(self, kwargs):
        """Convert output data to dictionary of tensors following the CCC"""
        raise NotImplementedError

    @property
    def with_batch_dim(self):
        """Get the with_batch_dim property of the pipeline."""
        return self.operations[0].with_batch_dim

    def validate(self):
        """Validate the pipeline by checking the compatibility of the operations."""
        operations = self.operations
        for i in range(len(operations) - 1):
            if operations[i].output_data_type is None:
                continue
            if operations[i + 1].input_data_type is None:
                continue
            if operations[i].output_data_type != operations[i + 1].input_data_type:
                raise ValueError(
                    f"Operation {operations[i].name} output data type is not compatible "
                    f"with the input data type of operation {operations[i + 1].name}"
                )

    def set_params(self, **params):
        """Set parameters for the operations in the pipeline by adding them to the cache."""
        for operation in self.operations:
            operation_params = {
                key: value
                for key, value in params.items()
                if key in operation._valid_keys
            }
            if operation_params:
                operation.set_input_cache(operation_params)

    def get_params(self, per_operation: bool = False):
        """Get a snapshot of the current parameters of the operations in the pipeline.

        Args:
            per_operation (bool): If True, return a list of dictionaries for each operation.
                                  If False, return a single dictionary with all parameters combined.
        """
        if per_operation:
            return [operation._input_cache.copy() for operation in self.operations]
        else:
            params = {}
            for operation in self.operations:
                params.update(operation._input_cache)
            return params

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


def make_operation_chain(operation_chain: List[Union[str, Dict]]) -> List[Operation]:
    """Make an operation chain from a custom list of operations.

    Args:
        operation_chain (list): List of operations to be performed.
            Each operation can be a string or a dictionary.
            if a string, the operation is initialized with default parameters.
            if a dictionary, the operation is initialized with the parameters
            provided in the dictionary, which should have the keys 'name' and 'params'.

    Returns:
        list: List of operations to be performed.

    """
    chain = []
    for operation in operation_chain:
        assert isinstance(
            operation, (str, dict, Config)
        ), f"Operation {operation} should be a string, dictionary or Config object"
        if isinstance(operation, str):
            operation = get_ops(operation)()
        else:
            if isinstance(operation, Config):
                operation = operation.serialize()
            # should have either name or name and params keys
            assert set(operation.keys()).issubset({"name", "params"}), (
                f"Operation {operation} should have keys 'name' and 'params'"
                f"or only 'name' got {operation.keys()}"
            )
            if operation.get("params") is None:
                operation["params"] = {}
            operation = get_ops(operation["name"])(**operation["params"])
        chain.append(operation)

    return chain


def pipeline_from_json(json_string: str, **kwargs) -> Pipeline:
    """Create a pipeline from a json string."""
    pipeline_config = json.loads(json_string)
    operations = make_operation_chain(pipeline_config["operations"])
    return Pipeline(operations=operations, **kwargs)


def pipeline_from_yaml(yaml_path: str, **kwargs) -> Pipeline:
    """Create a pipeline from a yaml file."""
    config = Config.load_from_yaml(yaml_path)
    operations = make_operation_chain(config.operations)
    return Pipeline(operations=operations, **kwargs)


def pipeline_from_config(config: Config, **kwargs) -> Pipeline:
    """Create a pipeline from a Config / dict kobject."""
    operations = make_operation_chain(config.operations)
    return Pipeline(operations=operations, **kwargs)


## Base Operations


@ops_registry("identity_v2")
class Identity(Operation):
    """Identity operation."""

    def call(self, *args, **kwargs) -> Dict:
        """Returns the input as is."""
        return kwargs


@ops_registry("merge_v2")
class Merge(Operation):
    """Operation that merges sets of input dictionaries."""

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


@ops_registry("split_v2")
class Split(Operation):
    """Operation that splits an input dictionary  n copies."""

    def __init__(self, n: int, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def call(self, *args, **kwargs) -> List[Dict]:
        """
        Splits the input dictionary into n copies.
        """
        return [kwargs.copy() for _ in range(self.n)]


@ops_registry("stack_v2")
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

    def call(self, *args, **kwargs) -> Dict:
        """
        Stacks the inputs corresponding to the specified keys along the specified axis.
        If a list of axes is provided, the length must match the number of keys.
        If an integer axis is provided, all inputs are stacked along the same axis.
        """

        raise NotImplementedError


@ops_registry("mean_v2")
class Mean(Operation):
    """Take the mean of the input data along a specific axis."""

    def __init__(self, axis, keys, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.keys = keys

    def call(self, **kwargs):
        for key in self.keys:
            kwargs[key] = ops.mean(kwargs[key], axis=self.axis)

        return kwargs
