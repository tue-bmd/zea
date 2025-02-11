""" Experimental version of the USBMD ops module"""

import hashlib
import inspect
import json
from typing import Any, Dict, List, Set, Union

import keras
import yaml
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
from usbmd.simulator import simulate_rf

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


def get_op(op_name):
    """Get the operation from the registry."""
    return ops_registry[op_name]


# TODO: check if inheriting from keras.Operation is better than using the ABC class.
class Operation(keras.Operation):
    """
    A base abstract class for operations in the pipeline with caching functionality.
    """

    def __init__(
        self,
        uid: str = None,
        input_data_type: Union[DataTypes, None] = None,
        output_data_type: Union[DataTypes, None] = None,
        cache_inputs: Union[bool, List[str]] = False,
        cache_outputs: bool = False,
        jit_compile: bool = True,
    ):
        """
        Args:
            input_data_type: The expected data type of the input data
            output_data_type: The expected data type of the output
            cache_inputs: A list of input keys to cache or True to cache all inputs
            cache_outputs: A list of output keys to cache or True to cache all outputs
            jit_compile: Whether to JIT compile the 'call' method for faster execution
        """
        super().__init__()

        self.uid = uid
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
        if not "kwargs" in self._valid_keys:
            filtered_kwargs = {
                k: v for k, v in merged_kwargs.items() if k in self._valid_keys
            }
        else:
            filtered_kwargs = merged_kwargs

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
    A Pipeline that supports branching. Each operation is assigned a unique ID.

    The configuration for an operation may include an "inputs" attribute—a list of op IDs
    from which it should receive its input dictionary. Typically, an op receives one input,
    but for operations like merge the list may contain more than one op ID.

    When executing the pipeline (TODO)
      ...

    A sequential pipeline is simply one in which no op defines "inputs" (or the ops are provided
    in a list without "inputs"), and the operations are executed in the order provided.
    """

    def __init__(
        self,
        operations: List[Any],
        with_batch_dim: bool = True,
        jit_options: Union[str, None] = "ops",
    ):
        if jit_options not in ["pipeline", "ops", None]:
            raise ValueError("jit_options must be 'pipeline', 'ops', or None")

        # Ensure that each element in the operations list is an Operation instance.
        # If an element is a string or dict, create the Operation instance.

        # TODO: this has shared code with make_operation_chain, refactor to avoid duplication

        for idx, op in enumerate(operations):
            if not isinstance(op, Operation):
                # If op is a string, then use it as the op name with default parameters.
                if isinstance(op, str):
                    new_op = get_op(op)()
                elif isinstance(op, dict):
                    # Expect dict to contain at least "op" and optionally "params", "id", "inputs"
                    params = op.get("params", {})
                    new_op = get_op(op["op"])(**params)
                    if "id" in op:
                        new_op.uid = op["id"]
                    if "inputs" in op:
                        new_op.inputs = op["inputs"]
                else:
                    raise ValueError(f"Unsupported operation type: {type(op)}")
                operations[idx] = new_op  # Replace with the newly created op.

        self._ops_list = operations
        for idx, op in enumerate(self._ops_list):
            if not hasattr(op, "id"):
                op.uid = f"op_{idx}"
            if not hasattr(op, "inputs"):
                op.inputs = None
            op.with_batch_dim = with_batch_dim
            op.set_jit(jit_options == "ops")

        # Build mapping from op ID to op instance.
        self._op_dict: Dict[str, Any] = {op.uid: op for op in self._ops_list}
        # If no op specifies "inputs", assume sequential order.
        if not any(op.inputs for op in self._ops_list):
            self._top_order = [op.uid for op in self._ops_list]
        else:
            self._top_order = self._topological_sort(self._ops_list)
            self.validate()

        self._call_pipeline = jit(self.call) if jit_options == "pipeline" else self.call

    def _build_dependency_graph(self, operations: List[Any]) -> Dict[str, Set[str]]:
        """
        Build a dependency graph mapping each op ID to the set of op IDs that
        must run before it. For an op that defines an "inputs" list, each element
        of that list is assumed to be an op ID.
        """
        dep_graph: Dict[str, Set[str]] = {op.uid: set() for op in operations}
        for op in operations:
            if op.inputs:
                for source_uid in op.inputs:
                    dep_graph[op.uid].add(source_uid)
        return dep_graph

    def _topological_sort(self, operations: List[Any]) -> List[str]:
        dep_graph = self._build_dependency_graph(operations)
        in_degree = {uid: len(deps) for uid, deps in dep_graph.items()}
        zero_in_degree = [uid for uid, deg in in_degree.items() if deg == 0]
        sorted_order = []

        while zero_in_degree:
            current = zero_in_degree.pop(0)
            sorted_order.append(current)
            for uid in dep_graph:
                if current in dep_graph[uid]:
                    dep_graph[uid].remove(current)
                    in_degree[uid] -= 1
                    if in_degree[uid] == 0:
                        zero_in_degree.append(uid)

        # Find dependencies that are referenced but not present in the dep_graph keys.
        missing_dependencies = {
            dep for deps in dep_graph.values() for dep in deps if dep not in dep_graph
        }

        if missing_dependencies:
            raise ValueError(
                f"Following ops are provided as input, but are missing from the Operation chain: "
                f"{missing_dependencies}"
            )

        if len(sorted_order) != len(operations):
            # If there is a cycle in the graph, the sorted order will be incomplete because
            # some ops cannot be added to the sorted order due to their dependencies.
            raise ValueError("Cycle detected in operation dependencies.")

        return sorted_order

    def call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline by iterating over operations in topological order.
        For an op that defines "inputs", merge the outputs from each specified upstream op
        (by their IDs) into a single dictionary and pass that as kwargs.
        If an op does not define "inputs", pass the entire global data dictionary.
        After execution, store the op's output under its own ID and merge it into the
        global data dictionary.
        """
        data_store = inputs.copy()
        for uid in self._top_order:
            op = self._op_dict[uid]
            if op.inputs:
                op_inputs = {}
                for source_id in op.inputs:
                    if source_id in data_store:
                        upstream_output = data_store[source_id]
                        if not isinstance(upstream_output, dict):
                            raise TypeError(
                                f"Expected dict for op '{source_id}', got {type(upstream_output)}"
                            )
                        op_inputs.update(upstream_output)
                    else:
                        raise ValueError(
                            f"Input op '{source_id}' for operation '{op.uid}' not found."
                        )
            else:
                op_inputs = data_store
            outputs = op(**op_inputs)
            data_store[op.uid] = outputs
            data_store.update(outputs)
        return outputs

    def __call__(self, *args, return_numpy=False, **kwargs) -> Dict[str, Any]:
        """
        Prepares inputs (merging Probe, Scan, and Config if provided as positional args)
        and then executes the pipeline.
        """
        if any(key in kwargs for key in ["probe", "scan", "config"]):
            raise ValueError(
                "Probe, Scan and Config objects should be passed as positional arguments. "
                "e.g. pipeline(probe, scan, config, **kwargs)"
            )
        dicts = {
            "probe": {},
            "scan": {},
            "config": {},
        }
        for arg in args:
            if not isinstance(arg, (Probe, Scan, Config)):
                raise ValueError(
                    f"Expected Probe, Scan, or Config object, got {type(arg).__name__}"
                )
            tensorized = arg.to_tensor()

            type_name = type(arg).__name__.lower()
            if type_name in ("probe", "scan", "config"):
                dicts[type_name] = tensorized

        inputs = {**dicts["probe"], **dicts["scan"], **dicts["config"], **kwargs}
        outputs = self._call_pipeline(inputs)
        if return_numpy:
            outputs = {k: v.numpy() for k, v in outputs.items()}
        return outputs

    def validate(self):
        """
        For each op that defines an "inputs" list, ensure that every referenced op ID exists.
        """
        for op in self._ops_list:
            if not op.inputs:
                continue

            for source_id in op.inputs:
                if source_id not in self._op_dict:
                    raise ValueError(
                        f"Operation '{op.uid}' expects input from '{source_id}', which is missing."
                    )

    def __str__(self):  # Ensure branches are displayed correctly
        ops_str = []
        for uid in self._top_order:
            op = self._op_dict[uid]
            if op.inputs:
                inputs_str = ", ".join(op.inputs)
                ops_str.append(f"{inputs_str} -> {uid}:{op.__class__.__name__}")
            else:
                ops_str.append(f"{uid}:{op.__class__.__name__}")
        return " -> ".join(ops_str)

    def __repr__(self):
        return self.__str__()

    @classmethod
    def load(cls, file_path: str, **kwargs) -> "Pipeline":
        """Load a pipeline from a JSON or YAML file."""
        if file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                json_str = f.read()
            return pipeline_from_json(json_str, **kwargs)
        elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
            return pipeline_from_yaml(file_path, **kwargs)
        else:
            raise ValueError("File must have extension .json, .yaml, or .yml")

    def save(self, file_path: str, file_format: str = "json") -> None:
        """Save the pipeline to a JSON or YAML file."""
        if file_format.lower() == "json":
            config_str = pipeline_to_json(self)
        elif file_format.lower() == "yaml":
            config_str = pipeline_to_yaml(self)
        else:
            raise ValueError("file_format must be either 'json' or 'yaml'.")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(config_str)


########################################################################
# Pipeline Construction and Serialization Functions
########################################################################


def make_operation_chain(
    operation_chain: List[Union[str, Dict, Config]]
) -> List[Operation]:
    """
    Build an operation chain from a list of operations.

    For a simple (sequential) pipeline, you can specify each op as a string
    (which initializes the op with default parameters) or as a dict with keys
    "op" and "params". For branched pipelines, you may also specify extra keys:
      • "id": a unique identifier for the op (optional in simple pipelines)
      • "inputs": a list of strings (e.g. "op1.out1") that indicate which outputs
                  from previous ops should be passed as inputs.

    Returns:
        A list of Operation instances.
    """
    chain = []
    for op_def in operation_chain:
        assert isinstance(
            op_def, (str, dict, Config)
        ), f"Operation {op_def} must be a string, dict, or Config object"
        if isinstance(op_def, str):
            op_instance = get_op(op_def)()
        else:
            if isinstance(op_def, Config):
                op_def = op_def.serialize()
            allowed = {"op", "params", "inputs", "id"}
            assert set(op_def.keys()).issubset(
                allowed
            ), f"Operation {op_def} has invalid keys. Allowed keys: {allowed}"
            if op_def.get("params") is None:
                op_def["params"] = {}
            op_instance = get_op(op_def["op"])(**op_def["params"])
            if "id" in op_def:
                op_instance.uid = op_def["id"]
            if "inputs" in op_def:
                op_instance.input_keys = op_def["inputs"]
        chain.append(op_instance)
    return chain


def pipeline_from_json(json_string: str, **kwargs) -> Pipeline:
    """
    Create a Pipeline instance from a JSON string.
    """
    pipeline_config = json.loads(json_string)["operations"]
    return Pipeline(operations=pipeline_config, **kwargs)


def pipeline_from_yaml(yaml_path: str, **kwargs) -> Pipeline:
    """
    Create a Pipeline instance from a YAML file.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        pipeline_config = yaml.safe_load(f)
    pipeline_config = pipeline_config["operations"]
    return Pipeline(operations=pipeline_config, **kwargs)


def pipeline_from_config(config: Config, **kwargs) -> Pipeline:
    """
    Create a Pipeline instance from a Config object.
    """
    operations = make_operation_chain(config.operations)
    return Pipeline(operations=operations, **kwargs)


########################################################################
# Pipeline Saving / Serialization Functions (External)
########################################################################


def pipeline_to_dict(pipeline: Pipeline) -> Dict:
    """
    Serialize the pipeline configuration to a dictionary.

    For a simple (sequential) pipeline, each operation is represented with just its
    op name and parameters. For a branched pipeline (i.e. if any op defined input_keys),
    the extra keys ("id" and "inputs") are included.

    Note: This function assumes that each operation supports a serialize() method.
    """
    op_list = []
    # Determine if any op has explicit input_keys.
    is_sequential = all(
        getattr(op, "input_keys", None) is None for op in pipeline._ops_list
    )
    if is_sequential:
        for op in pipeline._ops_list:
            op_conf = {  # get op name from ops_registry
                "op": op.__class__.__name__.lower(),
                "params": op.serialize() if hasattr(op, "serialize") else {},
            }
            op_list.append(op_conf)
    else:
        # In branched mode include the extra keys.
        for uid in pipeline._top_order:
            op = pipeline._op_dict[uid]
            op_conf = {
                "id": op.uid,
                "op": op.__class__.__name__,
                "params": op.serialize() if hasattr(op, "serialize") else {},
            }
            if getattr(op, "input_keys", None) is not None:
                op_conf["inputs"] = op.input_keys
            op_list.append(op_conf)
    return {"operations": op_list}


def pipeline_to_json(pipeline: Pipeline, indent: int = 4) -> str:
    """
    Serialize the pipeline to a JSON string.
    """
    return json.dumps(pipeline_to_dict(pipeline), indent=indent)


def pipeline_to_yaml(pipeline: Pipeline) -> str:
    """
    Serialize the pipeline to a YAML string.
    """
    return yaml.dump(pipeline_to_dict(pipeline))


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
            kwargs[key] = keras.ops.concat(  # pylint: disable=no-member
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


@ops_registry("simulate_rf_v2")
class Simulate(Operation):
    """Simulate RF data."""

    def __init__(self, n_ax, apply_lens_correction=True, **kwargs):
        super().__init__(
            output_data_type=DataTypes.RAW_DATA,
            jit_compile=False,
            **kwargs,
        )
        self.apply_lens_correction = apply_lens_correction
        self.n_ax = n_ax

    def call(
        self,
        scatterer_positions,
        scatterer_magnitudes,
        probe_geometry,
        lens_thickness,
        lens_sound_speed,
        sound_speed,
        center_frequency,
        sampling_frequency,
        t0_delays,
        initial_times,
        element_width,
        attenuation_coef,
        tx_apodizations,
    ):
        return {
            "raw_data": simulate_rf(
                scatterer_positions,
                scatterer_magnitudes,
                probe_geometry=probe_geometry,
                apply_lens_correction=self.apply_lens_correction,
                lens_thickness=lens_thickness,
                lens_sound_speed=lens_sound_speed,
                sound_speed=sound_speed,
                n_ax=self.n_ax,
                center_frequency=center_frequency,
                sampling_frequency=sampling_frequency,
                t0_delays=t0_delays,
                initial_times=initial_times,
                element_width=element_width,
                attenuation_coef=attenuation_coef,
                tx_apodizations=tx_apodizations,
            )
        }
