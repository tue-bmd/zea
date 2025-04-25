"""Experimental version of the USBMD ops module"""

import copy
import hashlib
import inspect
import json
from typing import Any, Dict, List, Union

import keras
import numpy as np
import yaml
from keras import ops

from usbmd.backend import jit
from usbmd.beamformer import tof_correction_flatgrid
from usbmd.config.config import Config
from usbmd.core import STATIC, DataTypes
from usbmd.core import Object as USBMDObject
from usbmd.core import USBMDDecoderJSON, USBMDEncoderJSON
from usbmd.display import scan_convert
from usbmd.ops import channels_to_complex, demodulate, hilbert, upmix
from usbmd.probes import Probe
from usbmd.registry import ops_v2_registry as ops_registry
from usbmd.scan import Scan
from usbmd.simulator import simulate_rf
from usbmd.tensor_ops import patched_map, reshape_axis
from usbmd.utils import deep_compare, log, translate
from usbmd.utils.checks import _assert_keys_and_axes

log.warning("WARNING: This module is work in progress and may not work as expected!")

DEFAULT_DYNAMIC_RANGE = (-60, 0)

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
        key: Union[str, None] = "data",
        output_key: Union[str, None] = None,
        cache_inputs: Union[bool, List[str]] = False,
        cache_outputs: bool = False,
        jit_compile: bool = True,
        with_batch_dim: bool = True,
        jit_kwargs: dict | None = None,
        jittable: bool = True,
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
        """
        super().__init__()

        self.input_data_type = input_data_type
        self.output_data_type = output_data_type

        self.key = key  # Key for input data
        self.output_key = output_key  # Key for output data
        if self.output_key is None:
            self.output_key = self.key

        self.inputs = []  # Source(s) of input data (name of a previous operation)
        self.allow_multiple_inputs = False  # Only single input allowed by default

        self.cache_inputs = cache_inputs
        self.cache_outputs = cache_outputs

        # Initialize input and output caches
        self._input_cache = {}
        self._output_cache = {}

        # Obtain the input signature of the `call` method
        self._input_signature = None
        self._valid_keys = None  # Keys valid for the `call` method
        self._trace_signatures()

        if jit_kwargs is None:
            # TODO: set static_argnames only for operations that require it
            if keras.backend.backend() == "jax":
                jit_kwargs = {"static_argnames": STATIC}
            else:
                jit_kwargs = {}
        self.jit_kwargs = jit_kwargs

        self.with_batch_dim = with_batch_dim
        self._jittable = jittable

        # Set the jit compilation flag and compile the `call` method
        self.set_jit(jit_compile)

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
        self._valid_keys = set(self._input_signature.parameters.keys())

    @property
    def jittable(self):
        """Check if the operation can be JIT compiled."""
        return self._jittable

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


@ops_registry("pipeline")
class Pipeline:
    """Pipeline class for processing ultrasound data through a series of operations."""

    def __init__(
        self,
        operations: List[Operation],
        with_batch_dim: bool = True,
        jit_options: Union[str, None] = "ops",
        jit_kwargs: dict | None = None,
        name="pipeline",
        validate=True,
    ):
        """Initialize a pipeline

        Args:
            operations (list): A list of Operation instances representing the operations
                to be performed.
            with_batch_dim (bool, optional): Whether operations should expect a batch dimension.
                Defaults to True.
            jit_options (str, optional): The JIT options to use. Must be "pipeline", "ops", or None.
                - "pipeline" compiles the entire pipeline as a single function.
                    This may be faster but, does not preserve python control flow, such as caching.
                - "ops" compiles each operation separately. This preserves python control flow and
                    caching functionality, but speeds up the operations.
                - None disables JIT compilation.
                Defaults to "ops".
            jit_kwargs (dict, optional): Additional keyword arguments for the JIT compiler.
            name (str, optional): The name of the pipeline. Defaults to "pipeline".
            validate (bool, optional): Whether to validate the pipeline. Defaults to True.
        """
        self._call_pipeline = self.call
        self.name = name

        # add functionality here
        # operations = ...

        self._pipeline_layers = operations

        if jit_options not in ["pipeline", "ops", None]:
            raise ValueError("jit_options must be 'pipeline', 'ops', or None")

        self.with_batch_dim = with_batch_dim

        if validate:
            self.validate()
        else:
            log.warning(
                "Pipeline validation is disabled, make sure to validate manually."
            )

        # pylint: disable=method-hidden
        if jit_kwargs is None:
            if keras.backend.backend() == "jax":
                jit_kwargs = {"static_argnames": STATIC}
            else:
                jit_kwargs = {}
        self.jit_kwargs = jit_kwargs
        self.jit_options = jit_options  # will handle the jit compilation

    def needs(self, key):
        """Check if the pipeline needs a specific key."""
        for operation in self.operations:
            if isinstance(operation, Pipeline):
                return operation.needs(key)
            if key in operation._valid_keys:
                return True

    @classmethod
    def from_default(cls, num_patches=20, **kwargs) -> "Pipeline":
        """Create a default pipeline."""
        operations = []

        # Add the demodulate operation
        operations.append(Demodulate())

        # Get beamforming ops
        beamforming = [
            TOFCorrection(apply_phase_rotation=True),
            PfieldWeighting(),
            DelayAndSum(),
        ]

        # Optionally add patching
        if num_patches > 1:
            beamforming = [PatchedGrid(operations=beamforming, num_patches=num_patches)]

        # Add beamforming ops
        operations += beamforming

        # Add display ops
        operations += [
            EnvelopeDetect(),
            Normalize(),
            LogCompress(),
        ]
        return cls(operations, **kwargs)

    def prepend(self, operation: Operation):
        """Prepend an operation to the pipeline."""
        self._pipeline_layers.insert(0, operation)
        self.reset_jit()

    def append(self, operation: Operation):
        """Append an operation to the pipeline."""
        self._pipeline_layers.append(operation)
        self.reset_jit()

    @property
    def operations(self):
        """Alias for self.layers to match the USBMD naming convention"""
        return self._pipeline_layers

    def call(self, **inputs):
        """Process input data through the pipeline."""
        for operation in self._pipeline_layers:
            outputs = operation(**inputs)
            inputs = outputs
        return outputs

    def __call__(self, return_numpy=False, **inputs):
        """Process input data through the pipeline."""

        if any(key in inputs for key in ["probe", "scan", "config"]):
            raise ValueError(
                "Probe, Scan and Config objects should be first processed with "
                "`Pipeline.prepare_parameters` before calling the pipeline. "
                "e.g. inputs = Pipeline.prepare_parameters(probe, scan, config)"
            )

        if any(isinstance(arg, USBMDObject) for arg in inputs.values()):
            raise ValueError(
                "Probe, Scan and Config objects should be first processed with "
                "`Pipeline.prepare_parameters` before calling the pipeline. "
                "e.g. inputs = Pipeline.prepare_parameters(probe, scan, config)"
            )

        if any(isinstance(arg, str) for arg in inputs.values()):
            raise ValueError(
                "Pipeline does not support string inputs. "
                "Please ensure all inputs are convertible to tensors."
            )

        ## PROCESSING
        outputs = self._call_pipeline(**inputs)

        ## PREPARE OUTPUT
        if return_numpy:
            # Convert tensors to numpy arrays but preserve None values
            outputs = {
                k: ops.convert_to_numpy(v) if v is ops.is_tensor(v) else v
                for k, v in outputs.items()
            }

        return outputs

    def reset_jit(self):
        """Reset the JIT compilation of the pipeline."""
        # TODO: kind of hacky...
        self.jit_options = self._jit_options

    @property
    def jit_options(self):
        """Get the jit_options property of the pipeline."""
        return self._jit_options

    @jit_options.setter
    def jit_options(self, value: Union[str, None]):
        """Set the jit_options property of the pipeline."""
        self._jit_options = value
        if value == "pipeline":
            assert self.jittable, log.error(
                "jit_options 'pipeline' cannot be used as the entire pipeline is not jittable. "
                "The following operations are not jittable: "
                f"{self.unjitable_ops}. "
                "Try setting jit_options to 'ops' or None."
            )
            self.jit()
            return
        else:
            self.unjit()

        for operation in self.operations:
            if isinstance(operation, Pipeline):
                operation.jit_options = value
            else:
                if operation.jittable:
                    operation.set_jit(value == "ops")

    def jit(self):
        """JIT compile the pipeline."""
        self._call_pipeline = jit(self.call, **self.jit_kwargs)

    def unjit(self):
        """Un-JIT compile the pipeline."""
        self._call_pipeline = self.call

    @property
    def jittable(self):
        """Check if all operations in the pipeline are jittable."""
        return all(operation.jittable for operation in self.operations)

    @property
    def unjitable_ops(self):
        """Get a list of operations that are not jittable."""
        return [operation for operation in self.operations if not operation.jittable]

    @property
    def with_batch_dim(self):
        """Get the with_batch_dim property of the pipeline."""
        return self.operations[0].with_batch_dim

    @with_batch_dim.setter
    def with_batch_dim(self, value):
        """Set the with_batch_dim property of the pipeline."""
        for operation in self.operations:
            operation.with_batch_dim = value

    @property
    def input_data_type(self):
        """Get the input_data_type property of the pipeline."""
        return self.operations[0].input_data_type

    @property
    def output_data_type(self):
        """Get the output_data_type property of the pipeline."""
        return self.operations[-1].output_data_type

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
                    f"Operation {operations[i].__class__.__name__} output data type "
                    f"({operations[i].output_data_type}) is not compatible "
                    f"with the input data type ({operations[i + 1].input_data_type}) "
                    f"of operation {operations[i + 1].__class__.__name__}"
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
        operations = []
        for operation in self.operations:
            if isinstance(operation, Pipeline):
                operations.append(repr(operation))
            else:
                operations.append(operation.__class__.__name__)
        return f"<Pipeline {self.name}=({', '.join(operations)})>"

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

    def get_dict(self) -> dict:
        """Convert the pipeline to a dictionary."""
        config = {}
        config["name"] = ops_registry.get_name(self)
        config["operations"] = self._pipeline_to_list(self)
        config["params"] = {
            "with_batch_dim": self.with_batch_dim,
            "jit_options": self.jit_options,
            "jit_kwargs": self.jit_kwargs,
        }
        return config

    @staticmethod
    def _pipeline_to_list(pipeline):
        """Convert the pipeline to a list of operations."""
        ops_list = []
        for op in pipeline.operations:
            ops_list.append(op.get_dict())
        return ops_list

    @classmethod
    def from_config(cls, config: Dict, **kwargs) -> "Pipeline":
        """Create a pipeline from a dictionary or `usbmd.Config` object.

        Args:
            config (dict or Config): Configuration dictionary or `usbmd.Config` object.
            **kwargs: Additional keyword arguments to be passed to the pipeline.

        Note:
            Must have the a `pipeline` key with a subkey `operations`.

        Example:
        ```python
        config = Config({
            "operations": [
                "identity",
            ],
        })
        pipeline = Pipeline.from_config(config)
        """
        return pipeline_from_config(Config(config), **kwargs)

    @classmethod
    def from_yaml(cls, file_path: str, **kwargs) -> "Pipeline":
        """Create a pipeline from a YAML file.

        Args:
            file_path (str): Path to the YAML file.
            **kwargs: Additional keyword arguments to be passed to the pipeline.

        Note:
            Must have the a `pipeline` key with a subkey `operations`.

        Example:
        ```python
        pipeline = Pipeline.from_yaml("pipeline.yaml")
        ```
        """
        return pipeline_from_yaml(file_path, **kwargs)

    @classmethod
    def from_json(cls, json_string: str, **kwargs) -> "Pipeline":
        """Create a pipeline from a JSON string.

        Args:
            json_string (str): JSON string representing the pipeline.
            **kwargs: Additional keyword arguments to be passed to the pipeline.

        Note:
            Must have the `operations` key.

        Example:
        ```python
        json_string = '{"operations": ["identity"]}'
        pipeline = Pipeline.from_json(json_string)
        ```
        """
        return pipeline_from_json(json_string, **kwargs)

    def to_config(self) -> Config:
        """Convert the pipeline to a `usbmd.Config` object."""
        return pipeline_to_config(self)

    def to_json(self) -> str:
        """Convert the pipeline to a JSON string."""
        return pipeline_to_json(self)

    def to_yaml(self, file_path: str) -> None:
        """Convert the pipeline to a YAML file."""
        pipeline_to_yaml(self, file_path)

    @property
    def key(self) -> str:
        """Input key of the pipeline."""
        return self.operations[0].key

    @property
    def output_key(self) -> str:
        """Output key of the pipeline."""
        return self.operations[-1].output_key

    def __eq__(self, other):
        """Check if two pipelines are equal."""
        if not isinstance(other, Pipeline):
            return False

        # Compare the operations in both pipelines
        if len(self.operations) != len(other.operations):
            return False

        for op1, op2 in zip(self.operations, other.operations):
            if not op1 == op2:
                return False

        return True

    def prepare_parameters(
        self,
        probe: Probe = None,
        scan: Scan = None,
        config: Config = None,
        **kwargs,
    ):
        """Prepare Probe, Scan and Config objects for the pipeline.

        Serializes `usbmd.core.Object` instances and converts them to
        dictionary of tensors.

        Args:
            probe: Probe object.
            scan: Scan object.
            config: Config object.
            **kwargs: Additional keyword arguments to be included in the inputs.

        Returns:
            dict: Dictionary of inputs with all values as tensors.
        """
        # Initialize dictionaries for probe, scan, and config
        probe_dict, scan_dict, config_dict = {}, {}, {}
        other_dicts = {}

        # Process args to extract Probe, Scan, and Config objects
        if probe is not None:
            assert isinstance(
                probe, Probe
            ), f"Expected an instance of `usbmd.probes.Probe`, got {type(probe)}"
            probe_dict = probe.to_tensor()

        if scan is not None:
            assert isinstance(
                scan, Scan
            ), f"Expected an instance of `usbmd.scan.Scan`, got {type(scan)}"
            except_tensors = []
            for key in scan._on_request:
                if not self.needs(key):
                    except_tensors.append(key)
            scan_dict = scan.to_tensor(except_tensors)

        if config is not None:
            # TODO: currently nothing...
            assert isinstance(
                config, Config
            ), f"Expected an instance of `usbmd.config.Config`, got {type(config)}"
            config_dict.update(config.to_tensor())

        # Convert all kwargs to tensors
        tensor_kwargs = {}
        for key, value in kwargs.items():
            try:
                if isinstance(value, USBMDObject):
                    tensor_kwargs[key] = value.to_tensor()
                else:
                    tensor_kwargs[key] = ops.convert_to_tensor(value)
            except Exception as e:
                raise ValueError(
                    f"Error converting key '{key}' to tensor: {e}. "
                    f"Please ensure all inputs are convertible to tensors."
                ) from e

        # combine probe, scan, config and kwargs
        # explicitly so we know which keys overwrite which
        # kwargs > config > scan > probe
        inputs = {
            **probe_dict,
            **scan_dict,
            **config_dict,
            **other_dicts,
            **tensor_kwargs,
        }

        # Dropping str inputs as they are not supported in jax.jit
        # TODO: will this break any operations?
        inputs.pop("probe_type", None)

        return inputs


def make_operation_chain(
    operation_chain: List[Union[str, Dict, Config, Operation, Pipeline]],
) -> List[Operation]:
    """Make an operation chain from a custom list of operations.
    Args:
        operation_chain (list): List of operations to be performed.
            Each operation can be:
            - A string: operation initialized with default parameters
            - A dictionary: operation initialized with parameters in the dictionary
            - A Config object: converted to a dictionary and initialized
            - An Operation/Pipeline instance: used as-is
    Returns:
        list: List of operations to be performed.
    """
    chain = []
    for operation in operation_chain:
        # Handle already instantiated Operation or Pipeline objects
        if isinstance(operation, (Operation, Pipeline)):
            chain.append(operation)
            continue

        assert isinstance(
            operation, (str, dict, Config)
        ), f"Operation {operation} should be a string, dict, Config object, Operation, or Pipeline"

        if isinstance(operation, str):
            operation_instance = get_ops(operation)()

        else:
            if isinstance(operation, Config):
                operation = operation.serialize()

            params = operation.get("params", {})
            op_name = operation.get("name")
            operation_cls = get_ops(op_name)

            # Handle branches for branched pipeline
            if op_name == "branched_pipeline" and "branches" in operation:
                branch_configs = operation.get("branches", {})
                branches = []

                # Convert each branch configuration to an operation chain
                for _, branch_config in branch_configs.items():
                    if isinstance(branch_config, (list, np.ndarray)):
                        # This is a list of operations
                        branch = make_operation_chain(branch_config)
                    elif "operations" in branch_config:
                        # This is a pipeline-like branch
                        branch = make_operation_chain(branch_config["operations"])
                    else:
                        # This is a single operation branch
                        branch_op_cls = get_ops(branch_config["name"])
                        branch_params = branch_config.get("params", {})
                        branch = branch_op_cls(**branch_params)

                    branches.append(branch)

                # Create the branched pipeline instance
                operation_instance = operation_cls(branches=branches, **params)
            # Check for nested operations at the same level as params
            elif "operations" in operation:
                nested_operations = make_operation_chain(operation["operations"])

                # Instantiate pipeline-type operations with nested operations
                if issubclass(operation_cls, Pipeline):
                    operation_instance = operation_cls(
                        operations=nested_operations, **params
                    )
                else:
                    operation_instance = operation_cls(
                        operations=nested_operations, **params
                    )
            else:
                operation_instance = operation_cls(**params)

        chain.append(operation_instance)

    return chain


def pipeline_from_config(config: Config, **kwargs) -> Pipeline:
    """
    Create a Pipeline instance from a Config object.
    """
    assert (
        "operations" in config
    ), "Config object must have an 'operations' key for pipeline creation."
    assert isinstance(
        config.operations, (list, np.ndarray)
    ), "Config object must have a list or numpy array of operations for pipeline creation."

    operations = make_operation_chain(config.operations)

    # merge pipeline config without operations with kwargs
    pipeline_config = copy.deepcopy(config)
    pipeline_config.pop("operations")

    kwargs = {**pipeline_config, **kwargs}
    return Pipeline(operations=operations, **kwargs)


def pipeline_from_json(json_string: str, **kwargs) -> Pipeline:
    """
    Create a Pipeline instance from a JSON string.
    """
    pipeline_config = Config(json.loads(json_string, cls=USBMDDecoderJSON))
    return pipeline_from_config(pipeline_config, **kwargs)


def pipeline_from_yaml(yaml_path: str, **kwargs) -> Pipeline:
    """
    Create a Pipeline instance from a YAML file.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        pipeline_config = yaml.safe_load(f)
    operations = pipeline_config["operations"]
    return pipeline_from_config(Config({"operations": operations}), **kwargs)


def pipeline_to_config(pipeline: Pipeline) -> Config:
    """
    Convert a Pipeline instance into a Config object.
    """
    # TODO: we currently add the full pipeline as 1 operation to the config.
    # In another PR we should add a "pipeline" entry to the config instead of the "operations"
    # entry. This allows us to also have non-default pipeline classes as top level op.
    pipeline_dict = {"operations": [pipeline.get_dict()]}

    # HACK: If the top level operation is a single pipeline, collapse it into the operations list.
    ops = pipeline_dict["operations"]
    if ops[0]["name"] == "pipeline" and len(ops) == 1:
        pipeline_dict = {"operations": ops[0]["operations"]}

    return Config(pipeline_dict)


def pipeline_to_json(pipeline: Pipeline) -> str:
    """
    Convert a Pipeline instance into a JSON string.
    """
    pipeline_dict = {"operations": [pipeline.get_dict()]}

    # HACK: If the top level operation is a single pipeline, collapse it into the operations list.
    ops = pipeline_dict["operations"]
    if ops[0]["name"] == "pipeline" and len(ops) == 1:
        pipeline_dict = {"operations": ops[0]["operations"]}

    return json.dumps(pipeline_dict, cls=USBMDEncoderJSON, indent=4)


def pipeline_to_yaml(pipeline: Pipeline, file_path: str) -> None:
    """
    Convert a Pipeline instance into a YAML file.
    """
    pipeline_dict = pipeline.get_dict()

    # HACK: If the top level operation is a single pipeline, collapse it into the operations list.
    ops = pipeline_dict["operations"]
    if ops[0]["name"] == "pipeline" and len(ops) == 1:
        pipeline_dict = {"operations": ops[0]["operations"]}

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(pipeline_dict, f, Dumper=yaml.Dumper, indent=4)


@ops_registry("patched_grid")
class PatchedGrid(Pipeline):
    """
    With this class you can form a pipeline that will be applied to patches of the grid.
    This is useful to avoid OOM errors when processing large grids.

    Somethings to NOTE about this class:
        - The ops have to use flatgrid and flat_pfield as inputs, these will be patched.
        - Changing anything other than `self.output_data_type` in the dict will not be propagated!
        - Will be jitted as a single operation, not the individual operations.
        - This class handles the batching.
    """

    def __init__(self, *args, num_patches=10, **kwargs):
        super().__init__(*args, name="patched_grid", **kwargs)
        self.num_patches = num_patches

        for operation in self.operations:
            if isinstance(operation, DelayAndSum):
                operation.reshape_grid = False

        self._jittable_call = self.jittable_call

    @property
    def jit_options(self):
        """Get the jit_options property of the pipeline."""
        return self._jit_options

    @jit_options.setter
    def jit_options(self, value):
        """Set the jit_options property of the pipeline."""
        self._jit_options = value
        if value in ["pipeline", "ops"]:
            self.jit()
        else:
            self.unjit()

    def jit(self):
        """JIT compile the pipeline."""
        self._jittable_call = jit(self.jittable_call, **self.jit_kwargs)

    def unjit(self):
        """Un-JIT compile the pipeline."""
        self._jittable_call = self.jittable_call
        self._call_pipeline = self.call

    @property
    def with_batch_dim(self):
        """Get the with_batch_dim property of the pipeline."""
        return self.pipeline_batched

    @with_batch_dim.setter
    def with_batch_dim(self, value):
        """Set the with_batch_dim property of the pipeline.
        The class handles the batching so the operations have to be set to False."""
        self.pipeline_batched = value
        for operation in self.operations:
            operation.with_batch_dim = False

    def call_item(self, inputs):
        """Process data in patches."""
        Nx = inputs["Nx"]
        Nz = inputs["Nz"]
        flatgrid = inputs.pop("flatgrid")
        flat_pfield = inputs.pop("flat_pfield")

        def patched_call(flatgrid, flat_pfield):
            out = super(PatchedGrid, self).call(  # pylint: disable=super-with-arguments
                flatgrid=flatgrid, flat_pfield=flat_pfield, **inputs
            )
            return out[self.output_key]

        out = patched_map(
            patched_call,
            flatgrid,
            self.num_patches,
            flat_pfield=flat_pfield,
            jit=bool(self.jit_options),
        )
        return ops.reshape(out, (Nz, Nx, *ops.shape(out)[1:]))

    def jittable_call(self, **inputs):
        """Process input data through the pipeline."""
        if self.pipeline_batched:
            input_data = inputs.pop(self.key)
            output = ops.map(
                lambda x: self.call_item({self.key: x, **inputs}),
                input_data,
            )
        else:
            output = self.call_item(inputs)

        return {self.output_key: output}

    def call(self, **inputs):
        """Process input data through the pipeline."""
        output = self._jittable_call(**inputs)
        inputs.update(output)
        return inputs

    def get_dict(self):
        """Get the configuration of the pipeline."""
        config = super().get_dict()
        config.update({"name": "patched_grid"})
        config["params"].update({"num_patches": self.num_patches})
        return config


## Base Operations


@ops_registry("identity")
class Identity(Operation):
    """Identity operation."""

    def call(self, *args, **kwargs) -> Dict:
        """Returns the input as is."""
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


@ops_registry("split")
class Split(Operation):
    """Operation that splits an input dictionary  n copies."""

    def __init__(self, n: int, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def call(self, **kwargs) -> List[Dict]:
        """
        Splits the input dictionary into n copies.
        """
        return [kwargs.copy() for _ in range(self.n)]


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


@ops_registry("simulate_rf")
class Simulate(Operation):
    """Simulate RF data."""

    def __init__(self, **kwargs):
        super().__init__(
            output_data_type=DataTypes.RAW_DATA,
            **kwargs,
        )

    def call(
        self,
        scatterer_positions,
        scatterer_magnitudes,
        probe_geometry,
        apply_lens_correction,
        lens_thickness,
        lens_sound_speed,
        sound_speed,
        n_ax,
        center_frequency,
        sampling_frequency,
        t0_delays,
        initial_times,
        element_width,
        attenuation_coef,
        tx_apodizations,
        **kwargs,
    ):
        return {
            self.output_key: simulate_rf(
                ops.convert_to_tensor(scatterer_positions),
                ops.convert_to_tensor(scatterer_magnitudes),
                probe_geometry=probe_geometry,
                apply_lens_correction=apply_lens_correction,
                lens_thickness=lens_thickness,
                lens_sound_speed=lens_sound_speed,
                sound_speed=sound_speed,
                n_ax=n_ax,
                center_frequency=center_frequency,
                sampling_frequency=sampling_frequency,
                t0_delays=t0_delays,
                initial_times=initial_times,
                element_width=element_width,
                attenuation_coef=attenuation_coef,
                tx_apodizations=tx_apodizations,
            ),
        }


@ops_registry("tof_correction")
class TOFCorrection(Operation):
    """Time-of-flight correction operation for ultrasound data."""

    def __init__(self, apply_phase_rotation=True, **kwargs):
        super().__init__(
            input_data_type=DataTypes.RAW_DATA,
            output_data_type=DataTypes.ALIGNED_DATA,
            **kwargs,
        )
        self.apply_phase_rotation = apply_phase_rotation

    def call(
        self,
        flatgrid=None,
        sound_speed=None,
        polar_angles=None,
        focus_distances=None,
        sampling_frequency=None,
        f_number=None,
        demodulation_frequency=None,
        t0_delays=None,
        tx_apodizations=None,
        initial_times=None,
        probe_geometry=None,
        apply_lens_correction=None,
        lens_thickness=None,
        lens_sound_speed=None,
        **kwargs,
    ):
        """Perform time-of-flight correction on raw RF data.

        Args:
            raw_data (ops.Tensor): Raw RF data to correct
            flatgrid (ops.Tensor): Grid points at which to evaluate the time-of-flight
            sound_speed (float): Sound speed in the medium
            polar_angles (ops.Tensor): Polar angles for scan lines
            focus_distances (ops.Tensor): Focus distances for scan lines
            sampling_frequency (float): Sampling frequency
            f_number (float): F-number for apodization
            demodulation_frequency (float): Demodulation frequency
            t0_delays (ops.Tensor): T0 delays
            tx_apodizations (ops.Tensor): Transmit apodizations
            initial_times (ops.Tensor): Initial times
            probe_geometry (ops.Tensor): Probe element positions
            apply_lens_correction (bool): Whether to apply lens correction
            lens_thickness (float): Lens thickness
            lens_sound_speed (float): Sound speed in the lens

        Returns:
            dict: Dictionary containing tof_corrected_data
        """

        raw_data = kwargs[self.key]

        kwargs = {
            "flatgrid": flatgrid,
            "sound_speed": sound_speed,
            "angles": polar_angles,
            "vfocus": focus_distances,
            "sampling_frequency": sampling_frequency,
            "fnum": f_number,
            "apply_phase_rotation": self.apply_phase_rotation,
            "demodulation_frequency": demodulation_frequency,
            "t0_delays": t0_delays,
            "tx_apodizations": tx_apodizations,
            "initial_times": initial_times,
            "probe_geometry": probe_geometry,
            "apply_lens_correction": apply_lens_correction,
            "lens_thickness": lens_thickness,
            "lens_sound_speed": lens_sound_speed,
        }

        if not self.with_batch_dim:
            tof_corrected = tof_correction_flatgrid(raw_data, **kwargs)
        else:
            tof_corrected = ops.map(
                lambda data: tof_correction_flatgrid(data, **kwargs),
                raw_data,
            )

        return {self.output_key: tof_corrected}


@ops_registry("pfield_weighting")
class PfieldWeighting(Operation):
    """Weighting aligned data with the pressure field."""

    def __init__(self, **kwargs):
        super().__init__(
            input_data_type=DataTypes.ALIGNED_DATA,
            output_data_type=DataTypes.ALIGNED_DATA,
            **kwargs,
        )

    def call(self, flat_pfield=None, **kwargs):
        """Weight data with pressure field.

        Args:
            flat_pfield (ops.Tensor): Pressure field weight mask of shape (n_pix, n_tx)

        Returns:
            dict: Dictionary containing weighted data
        """
        data = kwargs[self.key]

        if flat_pfield is None:
            return {self.output_key: data}

        # Swap (n_pix, n_tx) to (n_tx, n_pix)
        flat_pfield = ops.swapaxes(flat_pfield, 0, 1)

        # Perform element-wise multiplication with the pressure weight mask
        # Also add the required dimensions for broadcasting
        if self.with_batch_dim:
            pfield_expanded = ops.expand_dims(flat_pfield, axis=0)
        else:
            pfield_expanded = flat_pfield

        pfield_expanded = pfield_expanded[..., None, None]
        weighted_data = data * pfield_expanded

        return {self.output_key: weighted_data}


@ops_registry("sum")
class Sum(Operation):
    """Sum data along a specific axis."""

    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, **kwargs):
        data = kwargs[self.key]
        return {self.output_key: ops.sum(data, axis=self.axis)}


@ops_registry("delay_and_sum")
class DelayAndSum(Operation):
    """Sums time-delayed signals along channels and transmits."""

    def __init__(
        self,
        reshape_grid=True,
        **kwargs,
    ):
        super().__init__(
            input_data_type=None,
            output_data_type=DataTypes.BEAMFORMED_DATA,
            **kwargs,
        )
        self.reshape_grid = reshape_grid

    def process_image(self, data, rx_apo, tx_apo):
        """Performs DAS beamforming on tof-corrected input.

        Args:
            data (ops.Tensor): The TOF corrected input of shape `(n_tx, n_pix, n_el, n_ch)`

        Returns:
            ops.Tensor: The beamformed data of shape `(n_pix, n_ch)`
        """
        # Apply tx_apo
        data = tx_apo * data

        # Sum over the channels, i.e. DAS
        data = ops.sum(rx_apo * data, -2)

        # Sum over transmits, i.e. Compounding
        data = ops.sum(data, 0)

        return data

    def call(
        self,
        rx_apo=None,
        tx_apo=None,
        Nz=None,
        Nx=None,
        **kwargs,
    ):
        """Performs DAS beamforming on tof-corrected input.

        Args:
            tof_corrected_data (ops.Tensor): The TOF corrected input of shape
                `(n_tx, n_z*n_x, n_el, n_ch)` with optional batch dimension.
            rx_apo (ops.Tensor, optional): Receive apodization window. Defaults to 1.0.
            tx_apo (ops.Tensor, optional): Transmit apodization window. Defaults to 1.0.

        Returns:
            dict: Dictionary containing beamformed_data of shape `(n_z*n_x, n_ch)`
                when reshape_grid is False or `(n_z, n_x, n_ch)` when reshape_grid is True,
                with optional batch dimension.
        """
        if rx_apo is None:
            rx_apo = 1.0

        if tx_apo is None:
            tx_apo = 1.0

        data = kwargs[self.key]

        if not self.with_batch_dim:
            beamformed_data = self.process_image(data, rx_apo, tx_apo)
        else:
            # Apply process_image to each item in the batch
            beamformed_data = ops.map(
                lambda data: self.process_image(data, rx_apo, tx_apo), data
            )

        if self.reshape_grid:
            beamformed_data = reshape_axis(
                beamformed_data, (Nz, Nx), axis=int(self.with_batch_dim)
            )

        return {self.output_key: beamformed_data}


@ops_registry("envelope_detect")
class EnvelopeDetect(Operation):
    """Envelope detection of RF signals."""

    def __init__(
        self,
        axis=-3,
        **kwargs,
    ):
        super().__init__(
            input_data_type=DataTypes.BEAMFORMED_DATA,
            output_data_type=DataTypes.ENVELOPE_DATA,
            **kwargs,
        )
        self.axis = axis

    def call(self, **kwargs):
        """
        Args:
            - data (Tensor): The beamformed data of shape (..., n_z, n_x, n_ch).
        Returns:
            - envelope_data (Tensor): The envelope detected data of shape (..., n_z, n_x).
        """
        data = kwargs[self.key]

        if data.shape[-1] == 2:
            data = channels_to_complex(data)
        else:
            n_ax = data.shape[self.axis]
            M = 2 ** int(np.ceil(np.log2(n_ax)))
            # data = scipy.signal.hilbert(data, N=M, axis=self.axis)
            data = hilbert(data, N=M, axis=self.axis)
            indices = ops.arange(n_ax)

            data = ops.take(data, indices, axis=self.axis)
            data = ops.squeeze(data, axis=-1)

        # data = ops.abs(data)
        real = ops.real(data)
        imag = ops.imag(data)
        data = ops.sqrt(real**2 + imag**2)
        data = ops.cast(data, "float32")

        return {self.output_key: data}


@ops_registry("upmix")
class UpMix(Operation):
    """Upmix IQ data to RF data."""

    def call(
        self,
        sampling_frequency=None,
        center_frequency=None,
        upsampling_rate=6,
        **kwargs,
    ):
        data = kwargs[self.key]

        if data.shape[-1] == 1:
            log.warning("Upmixing is not applicable to RF data.")
            return data
        elif data.shape[-1] == 2:
            data = channels_to_complex(data)

        data = upmix(data, sampling_frequency, center_frequency, upsampling_rate)
        data = ops.expand_dims(data, axis=-1)
        return {self.output_key: data}


@ops_registry("log_compress")
class LogCompress(Operation):
    """Logarithmic compression of data."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            input_data_type=DataTypes.ENVELOPE_DATA,
            output_data_type=DataTypes.IMAGE,
            **kwargs,
        )

    def call(self, dynamic_range=None, **kwargs):
        """Apply logarithmic compression to data.

        Args:
            dynamic_range (tuple, optional): Dynamic range in dB. Defaults to (-60, 0).

        Returns:
            dict: Dictionary containing log-compressed data
        """
        data = kwargs[self.key]

        if dynamic_range is None:
            dynamic_range = DEFAULT_DYNAMIC_RANGE

        small_number = ops.convert_to_tensor(1e-16, dtype=data.dtype)
        data = ops.where(data == 0, small_number, data)
        compressed_data = 20 * ops.log10(data)
        compressed_data = ops.clip(compressed_data, *dynamic_range)

        return {self.output_key: compressed_data}


@ops_registry("normalize")
class Normalize(Operation):
    """Normalize data to a given range."""

    def __init__(self, output_range=None, input_range=None, **kwargs):
        super().__init__(**kwargs)
        self.output_range = self.to_float32(output_range)
        self.input_range = self.to_float32(input_range)
        assert output_range is None or len(output_range) == 2
        assert input_range is None or len(input_range) == 2

    @staticmethod
    def to_float32(data):
        """Converts an iterable to float32 and leaves None values as is."""
        return (
            [np.float32(x) if x is not None else None for x in data]
            if data is not None
            else None
        )

    def call(self, **kwargs):
        """Normalize data to a given range.

        Args:
            output_range (tuple, optional): Range to which data should be mapped.
                Defaults to (0, 1).
            input_range (tuple, optional): Range of input data. If None, the range
                of the input data will be computed. Defaults to None.

        Returns:
            dict: Dictionary containing normalized data
        """
        data = kwargs[self.key]

        output_range = _set_if_none(self.output_range, default=(0, 1))
        input_range = _set_if_none(self.input_range, default=(None, None))

        a_min, a_max = input_range
        if a_min is None:
            a_min = ops.min(data)
        if a_max is None:
            a_max = ops.max(data)
        data = ops.clip(data, a_min, a_max)
        input_range = (a_min, a_max)

        # Map the data to the output range
        normalized_data = translate(data, input_range, output_range)

        return {self.output_key: normalized_data}


def _set_if_none(variable, default):
    if variable is not None:
        return variable
    return default


@ops_registry("scan_convert")
class ScanConvert(Operation):
    """Scan convert images to cartesian coordinates."""

    def __init__(
        self,
        order=1,
        **kwargs,
    ):
        """Initialize the ScanConvert operation.

        Args:
            order (int, optional): Interpolation order. Defaults to 1. Currently only
                GPU support for order=1.
        """
        jittable = kwargs.pop("jittable", False)
        super().__init__(
            input_data_type=DataTypes.IMAGE,
            output_data_type=DataTypes.IMAGE_SC,
            jittable=jittable,  # if you provide coordinates, this operation can be jitted!
            **kwargs,
        )
        self.order = order

    def call(
        self,
        rho_range=None,
        theta_range=None,
        phi_range=None,
        resolution=None,
        coordinates=None,
        fill_value=None,
        **kwargs,
    ):
        """Scan convert images to cartesian coordinates.

        Args:
            rho_range (Tuple): Range of the rho axis in the polar coordinate system.
                Defined in meters.
            theta_range (Tuple): Range of the theta axis in the polar coordinate system.
                Defined in radians.
            phi_range (Tuple): Range of the phi axis in the polar coordinate system.
                Defined in radians.
            resolution (float): Resolution of the output image in meters per pixel.
                if None, the resolution is computed based on the input data.
            coordinates (Tensor): Coordinates for scan convertion. If None, will be computed
                based on rho_range, theta_range, phi_range and resolution. If provided, this
                operation can be jitted.
            fill_value (float): Value to fill the image with outside the defined region.

        """

        data = kwargs[self.key]

        data_out = scan_convert(
            data,
            rho_range,
            theta_range,
            phi_range,
            resolution,
            coordinates,
            fill_value,
            self.order,
        )

        return {self.output_key: data_out}


@ops_registry("demodulate")
class Demodulate(Operation):
    """Demodulates the input data to baseband."""

    def __init__(self, axis=-3, **kwargs):
        super().__init__(
            input_data_type=DataTypes.RAW_DATA,
            output_data_type=DataTypes.RAW_DATA,
            jittable=True,
            **kwargs,
        )
        self.axis = axis

    def call(self, center_frequency=None, sampling_frequency=None, **kwargs):
        data = kwargs[self.key]

        demodulation_frequency = center_frequency

        # Split the complex signal into two channels
        iq_data_two_channel = demodulate(
            data=data,
            center_frequency=center_frequency,
            sampling_frequency=sampling_frequency,
            axis=self.axis,
        )

        return {
            self.output_key: iq_data_two_channel,
            "demodulation_frequency": demodulation_frequency,
            "n_ch": 2,
        }


@ops_registry("clip")
class Clip(Operation):
    """Clip the input data to a given range."""

    def __init__(self, min_value=None, max_value=None, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, **kwargs):
        data = kwargs[self.key]
        data = ops.clip(data, self.min_value, self.max_value)
        return {self.output_key: data}


@ops_registry("branched_pipeline")
class BranchedPipeline(Operation):
    """Operation that processes data through multiple branches.

    This operation takes input data, processes it through multiple parallel branches,
    and then merges the results from those branches using the specified merge strategy.
    """

    def __init__(self, branches=None, merge_strategy="nested", **kwargs):
        """Initialize a branched pipeline.

        Args:
            branches (List[Union[List, Pipeline, Operation]]): List of branch operations
            merge_strategy (str or callable): How to merge the outputs from branches:
                - "nested" (default): Return outputs as a dictionary keyed by branch name
                - "flatten": Flatten outputs by prefixing keys with the branch name
                - "suffix": Flatten outputs by suffixing keys with the branch name
                - callable: A custom merge function that accepts the branch outputs dict
            **kwargs: Additional arguments for the Operation base class
        """
        super().__init__(**kwargs)

        # Convert branch specifications to operation chains
        if branches is None:
            branches = []

        self.branches = {}
        for i, branch in enumerate(branches, start=1):
            branch_name = f"branch_{i}"
            # Convert different branch specification types
            if isinstance(branch, list):
                # Convert list to operation chain
                self.branches[branch_name] = make_operation_chain(branch)
            elif isinstance(branch, (Pipeline, Operation)):
                # Already a pipeline or operation
                self.branches[branch_name] = branch
            else:
                raise ValueError(
                    f"Branch must be a list, Pipeline, or Operation, got {type(branch)}"
                )

        # Set merge strategy
        self.merge_strategy = merge_strategy
        if isinstance(merge_strategy, str):
            if merge_strategy == "nested":
                self._merge_function = lambda outputs: outputs
            elif merge_strategy == "flatten":
                self._merge_function = self.flatten_outputs
            elif merge_strategy == "suffix":
                self._merge_function = self.suffix_merge_outputs
            else:
                raise ValueError(f"Unknown merge_strategy: {merge_strategy}")
        elif callable(merge_strategy):
            self._merge_function = merge_strategy
        else:
            raise ValueError("Invalid merge_strategy type provided.")

    def call(self, **kwargs):
        """Process input through branches and merge results.

        Args:
            **kwargs: Input keyword arguments

        Returns:
            dict: Merged outputs from all branches according to merge strategy
        """
        branch_outputs = {}
        for branch_name, branch in self.branches.items():
            # Each branch gets a fresh copy of kwargs to avoid interference
            branch_kwargs = kwargs.copy()

            # Process through the branch
            branch_result = branch(**branch_kwargs)

            # Store branch outputs
            branch_outputs[branch_name] = branch_result

        # Apply merge strategy to combine outputs
        merged_outputs = self._merge_function(branch_outputs)

        return merged_outputs

    def flatten_outputs(self, outputs: dict) -> dict:
        """
        Flatten a nested dictionary by prefixing keys with the branch name.
        For each branch, the resulting key is "{branch_name}_{original_key}".
        """
        flat = {}
        for branch_name, branch_dict in outputs.items():
            for key, value in branch_dict.items():
                new_key = f"{branch_name}_{key}"
                if new_key in flat:
                    raise ValueError(f"Key collision detected for {new_key}")
                flat[new_key] = value
        return flat

    def suffix_merge_outputs(self, outputs: dict) -> dict:
        """
        Flatten a nested dictionary by suffixing keys with the branch name.
        For each branch, the resulting key is "{original_key}_{branch_name}".
        """
        flat = {}
        for branch_name, branch_dict in outputs.items():
            for key, value in branch_dict.items():
                new_key = f"{key}_{branch_name}"
                if new_key in flat:
                    raise ValueError(f"Key collision detected for {new_key}")
                flat[new_key] = value
        return flat

    def get_config(self):
        """Return the config dictionary for serialization."""
        config = super().get_config()

        # Add branch configurations
        branch_configs = {}
        for branch_name, branch in self.branches.items():
            if isinstance(branch, Pipeline):
                # Get the operations list from the Pipeline
                branch_configs[branch_name] = branch.get_config()
            elif isinstance(branch, list):
                # Convert list of operations to list of operation configs
                branch_op_configs = []
                for op in branch:
                    branch_op_configs.append(op.get_config())
                branch_configs[branch_name] = {"operations": branch_op_configs}
            else:
                # Single operation
                branch_configs[branch_name] = branch.get_config()

        # Add merge strategy
        if isinstance(self.merge_strategy, str):
            merge_strategy_config = self.merge_strategy
        else:
            # For custom functions, use the name if available
            merge_strategy_config = getattr(self.merge_strategy, "__name__", "custom")

        config.update(
            {
                "branches": branch_configs,
                "merge_strategy": merge_strategy_config,
            }
        )

        return config

    def get_dict(self):
        """Get the configuration of the operation."""
        config = super().get_dict()
        config.update({"name": "branched_pipeline"})

        # Add branches (recursively) to the config
        branches = {}
        for branch_name, branch in self.branches.items():
            if isinstance(branch, Pipeline):
                branches[branch_name] = branch.get_dict()
            elif isinstance(branch, list):
                branches[branch_name] = [op.get_dict() for op in branch]
            else:
                branches[branch_name] = branch.get_dict()
        config["branches"] = branches
        config["merge_strategy"] = self.merge_strategy
        return config
