""" Experimental version of the USBMD ops module"""

import hashlib
import inspect
import json
from typing import Any, Dict, List, Set, Union

import keras
from keras import ops

from usbmd.backend import jit
from usbmd.config.config import Config
from usbmd.core import DataTypes
from usbmd.ops import channels_to_complex, upmix
from usbmd.probes import Probe
from usbmd.registry import ops_registry
from usbmd.scan import Scan
from usbmd.utils import log
from usbmd.utils.checks import _assert_keys_and_axes

log.warning("WARNING: This module is work in progress and may not work as expected!")

# pylint: disable=arguments-differ

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

        self.inputs = None  # List of Operation outputs that are input to this Operation

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

    @property
    def connected(self):
        """Check if the operation is connected to another operation."""
        return self.inputs is not None

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
    """
    A Pipeline that supports both sequential (simple) and branched pipelines.

    In the simple (sequential) case, you may provide a list of Operation instances
    without needing to specify IDs or input keys. In that case each operation receives
    the entire global input dictionary (or the output of the previous op).

    In the branched case, at least one operation must have an explicit 'id' or define an
    'input_keys' attribute. In this mode every operation is assigned (or must provide) a
    unique ID. Operations can then specify in their "input_keys" a list of keys of the form
    "other_op_id.suffix" so that their inputs come from the output of a particular op.
    """

    def __init__(
        self,
        operations: List[Any],  # A list of Operation instances.
        with_batch_dim: bool = True,
        jit_options: Union[str, None] = "ops",
    ):
        """
        Args:
            operations: A list of Operation instances.
                • For a sequential pipeline, no extra info is needed.
                • For a branched pipeline, each op must have a unique 'id' (or one is assigned)
                  and can optionally define 'input_keys' (a list of strings) to select inputs.
            with_batch_dim: Whether to include a batch dimension.
            jit_options: "pipeline", "ops", or None.
                - "pipeline": jit-compiles the entire pipeline.
                - "ops": jit-compiles each op separately.
                - None: no JIT.
        """
        if jit_options not in ["pipeline", "ops", None]:
            raise ValueError("jit_options must be 'pipeline', 'ops', or None")

        # Decide the mode:
        # If any op has an "id" attribute or an "input_keys" attribute, we assume branched.
        self._branched = any(
            hasattr(op, "id") or hasattr(op, "input_keys") for op in operations
        )

        if not self._branched:
            # SIMPLE / SEQUENTIAL MODE:
            self._pipeline_layers = operations
            for op in self._pipeline_layers:
                op.with_batch_dim = with_batch_dim
                op.set_jit(jit_options == "ops")
        else:
            # BRANCHED MODE:
            # Ensure every op has a unique id. (If missing, assign one automatically.)
            for idx, op in enumerate(operations):
                if not hasattr(op, "id"):
                    op.id = f"op_{idx}"
            # It’s okay if some ops do not define 'input_keys': those will simply receive the full data-store.
            self._op_dict: Dict[str, Any] = {op.id: op for op in operations}
            for op in operations:
                op.with_batch_dim = with_batch_dim
                op.set_jit(jit_options == "ops")
            self._top_order = self._topological_sort(operations)
            self.validate()  # Validate branched dependencies

        # Wrap the pipeline call in jit if requested
        self._call_pipeline = jit(self.call) if jit_options == "pipeline" else self.call

    def _build_dependency_graph(self, operations: List[Any]) -> Dict[str, Set[str]]:
        """
        Build a dependency graph where each key is an op id and each value is the set of op ids
        that must be executed before it (because its input_keys reference those ops).
        """
        dep_graph: Dict[str, Set[str]] = {op.id: set() for op in operations}
        for op in operations:
            if hasattr(op, "input_keys") and op.input_keys:
                for key in op.input_keys:
                    if "." in key:
                        source_op_id = key.split(".")[0]
                        if source_op_id in dep_graph:
                            dep_graph[op.id].add(source_op_id)
        return dep_graph

    def _topological_sort(self, operations: List[Any]) -> List[str]:
        """
        Return a list of op ids sorted in topological order (dependencies come first).
        Raises an error if a cycle is detected.
        """
        dep_graph = self._build_dependency_graph(operations)
        in_degree = {op_id: len(deps) for op_id, deps in dep_graph.items()}
        zero_in_degree = [op_id for op_id, deg in in_degree.items() if deg == 0]
        sorted_order = []
        while zero_in_degree:
            current = zero_in_degree.pop(0)
            sorted_order.append(current)
            for op_id in dep_graph:
                if current in dep_graph[op_id]:
                    dep_graph[op_id].remove(current)
                    in_degree[op_id] -= 1
                    if in_degree[op_id] == 0:
                        zero_in_degree.append(op_id)
        if len(sorted_order) != len(operations):
            raise ValueError("Cycle detected in operation dependencies.")
        return sorted_order

    def call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline.

        In sequential mode, each op receives the full data dictionary (or the output of the previous op).
        In branched mode, operations are executed in topological order using their input_keys to select data
        from a shared data_store.
        """
        if not self._branched:
            # SIMPLE / SEQUENTIAL MODE:
            data = inputs.copy()
            for op in self._pipeline_layers:
                data = op(**data)
            return data
        else:
            # BRANCHED MODE:
            data_store = inputs.copy()
            for op_id in self._top_order:
                op = self._op_dict[op_id]
                # If the op defines input_keys, pick those from the data_store; else pass full data_store.
                if hasattr(op, "input_keys") and op.input_keys:
                    op_inputs = {}
                    for key in op.input_keys:
                        if key in data_store:
                            op_inputs[key] = data_store[key]
                        else:
                            raise ValueError(
                                f"Input key '{key}' for operation '{op_id}' not found in data_store."
                            )
                else:
                    op_inputs = data_store
                outputs = op(**op_inputs)
                data_store.update(outputs)
            return outputs

    def __call__(self, *args, return_numpy=False, **kwargs) -> Dict[str, Any]:
        """
        Prepare input data and execute the pipeline.

        (This version assumes that Probe, Scan, Config objects (if any) are passed as positional
         arguments and have a .to_tensor() method, similar to your original design.)
        """
        if any(key in kwargs for key in ["probe", "scan", "config"]):
            raise ValueError(
                "Probe, Scan and Config objects should be passed as positional arguments. "
                "e.g. pipeline(probe, scan, config, **kwargs)"
            )
        probe, scan, config = {}, {}, {}
        for arg in args:
            if hasattr(arg, "to_tensor"):
                tensorized = arg.to_tensor()
                type_name = type(arg).__name__.lower()
                if type_name in ["probe", "scan", "config"]:
                    if type_name == "probe":
                        probe = tensorized
                    elif type_name == "scan":
                        scan = tensorized
                    elif type_name == "config":
                        config = tensorized
            else:
                raise ValueError(f"Unsupported input type: {type(arg)}")
        inputs = {**probe, **scan, **config, **kwargs}
        outputs = self._call_pipeline(inputs)
        if return_numpy:
            outputs = {k: v.numpy() for k, v in outputs.items()}
        return outputs

    def validate(self):
        """
        In branched mode, ensure that all input_keys referencing a previous op actually refer to a valid op.
        (For sequential pipelines, no extra validation is needed.)
        """
        if self._branched:
            for op in self._op_dict.values():
                if hasattr(op, "input_keys") and op.input_keys:
                    for key in op.input_keys:
                        if "." in key:
                            source_op_id = key.split(".")[0]
                            if source_op_id not in self._op_dict:
                                raise ValueError(
                                    f"Operation '{op.id}' expects input from '{source_op_id}', which is not defined."
                                )

    def __str__(self):
        """
        Provide a string representation of the pipeline.
        In sequential mode, this prints a simple arrow chain.
        In branched mode, you might extend this to visualize the DAG.
        """
        if not self._branched:
            ops_str = " -> ".join(
                [op.__class__.__name__ for op in self._pipeline_layers]
            )
            return ops_str
        else:
            ops_str = " -> ".join(
                [
                    f"{op_id}:{self._op_dict[op_id].__class__.__name__}"
                    for op_id in self._top_order
                ]
            )
            return f"Branched Pipeline: {ops_str}"

    def __repr__(self):
        return self.__str__()


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

    TODO: add support for nested operations such that parallel pipelines can be defined.

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
            assert set(operation.keys()).issubset({"op", "params", "inputs"}), (
                f"Operation {operation} should have keys 'op', 'params' and 'inputs'"
                f"or only 'op' got {operation.keys()}"
            )
            if operation.get("params") is None:
                operation["params"] = {}
            operation = get_ops(operation["op"])(**operation["params"])
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
        keys: Union[str, List[str], None],
        axes: Union[int, List[int], None],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.keys, self.axes = _assert_keys_and_axes(keys, axes)

    def call(self, *args, **kwargs) -> Dict:
        """
        Stacks the inputs corresponding to the specified keys along the specified axis.
        If a list of axes is provided, the length must match the number of keys.
        If an integer axis is provided, all inputs are stacked along the same axis.
        """

        raise NotImplementedError


@ops_registry("concatenate_v2")
class Concatenate(Operation):
    """Concatenate multiple data arrays along an existing axis."""

    def __init__(self, keys: List[str], axis: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.keys = keys
        self.axis = axis

    def call(self, **kwargs) -> Dict:
        """
        Concatenates the inputs corresponding to the specified keys along the specified axis.
        """
        for key, axis in zip(self.keys, self.axis):
            kwargs[key] = keras.ops.concat(
                [kwargs[key] for key in self.keys], axis=axis
            )
        return kwargs


@ops_registry("rename_v2")
class Rename(Operation):
    """Rename keys in the input dictionary."""

    def __init__(self, mapping: Dict[str, str], **kwargs):
        super().__init__(**kwargs)
        self.mapping = mapping

    def call(self, **kwargs) -> Dict:
        """
        Renames the keys in the input dictionary according to the mapping.
        """
        renamed = {self.mapping.get(k, k): v for k, v in kwargs.items()}
        return renamed


@ops_registry("filter_v2")
class Filter(Operation):
    """Filter keys in the input dictionary."""

    def __init__(self, keys: List[str], **kwargs):
        super().__init__(**kwargs)
        self.keys = keys

    def call(self, *args, **kwargs) -> Dict:
        """
        Filters the input dictionary to include only the specified keys.
        """
        filtered = {k: v for k, v in kwargs.items() if k in self.keys}
        return filtered


@ops_registry("output_v2")
class Output(Operation):
    """Output operation. This operation is used to mark outputs in the pipeline.
    Optionally, keys can be specified to only output a subset of the input dictionary. Otherwise
    the entire input dictionary is returned.
    """

    def __init__(self, keys: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.keys = keys

    def call(self, *args, **kwargs) -> Dict:
        """Returns the input dictionary."""
        if self.keys:
            return {k: v for k, v in kwargs.items() if k in self.keys}
        return kwargs


@ops_registry("mean_v2")
class Mean(Operation):
    """Take the mean of the input data along a specific axis."""

    def __init__(self, keys, axes, **kwargs):
        super().__init__(**kwargs)

        self.keys, self.axes = _assert_keys_and_axes(keys, axes)

    def call(self, **kwargs):
        for key, axis in zip(self.keys, self.axes):
            kwargs[key] = ops.mean(kwargs[key], axis=axis)

        return kwargs


@ops_registry("upmix_v2")
class UpMix(Operation):
    """Upmix IQ data to RF data."""

    def __init__(self, key: str, **kwargs):
        super().__init__(**kwargs)
        self.key = key

    def call(self, fs=None, fc=None, upsampling_rate=6, **kwargs):

        data = kwargs[self.key]

        if data.shape[-1] == 1:
            log.warning("Upmixing is not applicable to RF data.")
            return data
        elif data.shape[-1] == 2:
            data = channels_to_complex(data)

        data = upmix(data, fs, fc, upsampling_rate)
        data = ops.expand_dims(data, axis=-1)
        return data
