"""Ops module for processing ultrasound data.

This module contains two important classes, Operation and Pipeline, which are used to
process ultrasound data. A pipeline is a sequence of operations that are applied to the data
in a specific order.

## Stand-alone manual usage
Operations can be run on their own:

Examples:
```python
data = np.random.randn(2000, 128, 1)
# static arguments are passed in the constructor
envelope_detect = EnvelopeDetect(axis=-1)
# other parameters can be passed here along with the data
envelope_data = envelope_detect(data=data)
```

## Using a pipeline
You can initialize with a default pipeline or create your own custom pipeline.
```python
pipeline = Pipeline.from_default()

operations = [
    EnvelopeDetect(),
    Normalize(),
    LogCompress(),
]
pipeline_custom = Pipeline(operations)
```

One can also load a pipeline from a config or yaml/json file:

```python
json_string = '{"operations": ["identity"]}'
pipeline = Pipeline.from_json(json_string)

yaml_file = "pipeline.yaml"
pipeline = Pipeline.from_yaml(yaml_file)
```

Example of a yaml file:
```yaml
pipeline:
  operations:
    - name: demodulate
    - name: "patched_grid"
      params:
        operations:
          - name: tof_correction
            params:
              apply_phase_rotation: true
          - name: pfield_weighting
          - name: delay_and_sum
        num_patches: 100
    - name: envelope_detect
    - name: normalize
    - name: log_compress

```
"""

import copy
import hashlib
import inspect
import json
from typing import Any, Dict, List, Union

import keras
import numpy as np
import scipy
import yaml
from keras import ops

from usbmd.backend import jit
from usbmd.beamform.beamformer import tof_correction
from usbmd.config.config import Config
from usbmd.core import STATIC, DataTypes
from usbmd.core import Object as USBMDObject
from usbmd.core import USBMDDecoderJSON, USBMDEncoderJSON
from usbmd.display import scan_convert
from usbmd import log
from usbmd.probes import Probe
from usbmd.internal.registry import ops_registry
from usbmd.scan import Scan
from usbmd.simulator import simulate_rf
from usbmd.tensor_ops import patched_map, resample, reshape_axis
from usbmd.utils import check_architecture, deep_compare, translate
from usbmd.internal.checks import _assert_keys_and_axes

DEFAULT_DYNAMIC_RANGE = (-60, 0)


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

            # Get global static parameters
            static_params = list(STATIC)

            # Add operation-specific static parameters
            op_static = list(getattr(self.__class__, "STATIC_PARAMS", []))
            if op_static:
                static_params = list(set(static_params + op_static))

            if keras.backend.backend() == "jax":
                jit_kwargs = {"static_argnames": static_params}
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

    # pylint: disable=arguments-differ
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
    def from_default(cls, num_patches=20, pfield=True, **kwargs) -> "Pipeline":
        """Create a default pipeline."""
        operations = []

        # Add the demodulate operation
        operations.append(Demodulate())

        # Get beamforming ops
        beamforming = [
            TOFCorrection(apply_phase_rotation=True),
            DelayAndSum(),
        ]
        if pfield:
            beamforming.insert(1, PfieldWeighting())

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
                if operation.jittable and operation._jit_compile:
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
                # TODO: maybe some logic of convert_to_tensor is needed
                if isinstance(value, USBMDObject):
                    tensor_kwargs[key] = value.to_tensor()
                elif value is None:
                    tensor_kwargs[key] = None
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
            elif operation["name"] in ["patched_grid"]:
                nested_operations = make_operation_chain(
                    operation["params"].pop("operations")
                )
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

        # Define a list of keys to look up for patching
        patch_keys = ["flat_pfield"]

        patch_arrays = {}
        for key in patch_keys:
            if key in inputs:
                patch_arrays[key] = inputs.pop(key)

        def patched_call(flatgrid, **patch_kwargs):
            patch_args = {k: v for k, v in patch_kwargs.items() if v is not None}
            out = super(PatchedGrid, self).call(  # pylint: disable=super-with-arguments
                flatgrid=flatgrid, **patch_args, **inputs
            )
            return out[self.output_key]

        out = patched_map(
            patched_call,
            flatgrid,
            self.num_patches,
            **patch_arrays,
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

    def call(self, **kwargs) -> Dict:
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


@ops_registry("transpose")
class Transpose(Operation):
    """Transpose the input data along the specified axes."""

    def __init__(self, axes, **kwargs):
        super().__init__(**kwargs)
        self.axes = axes

    def call(self, **kwargs):
        data = kwargs[self.key]
        transposed_data = ops.transpose(data, axes=self.axes)
        return {self.output_key: transposed_data}


@ops_registry("simulate_rf")
class Simulate(Operation):
    """Simulate RF data."""

    # Define operation-specific static parameters
    STATIC_PARAMS = ["n_ax"]

    def __init__(self, **kwargs):
        super().__init__(
            output_data_type=DataTypes.RAW_DATA,
            **kwargs,
        )

    # pylint: disable=arguments-differ
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

    # Define operation-specific static parameters
    STATIC_PARAMS = [
        "f_number",
        "apply_lens_correction",
        "apply_phase_rotation",
        "Nx",
        "Nz",
    ]

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
            tof_corrected = tof_correction(raw_data, **kwargs)
        else:
            tof_corrected = ops.map(
                lambda data: tof_correction(data, **kwargs),
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

    def __init__(
        self,
        upsampling_rate=1,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.upsampling_rate = upsampling_rate

    def call(
        self,
        sampling_frequency=None,
        center_frequency=None,
        **kwargs,
    ):
        data = kwargs[self.key]

        if data.shape[-1] == 1:
            log.warning("Upmixing is not applicable to RF data.")
            return data
        elif data.shape[-1] == 2:
            data = channels_to_complex(data)

        data = upmix(data, sampling_frequency, center_frequency, self.upsampling_rate)
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
        if output_range is None:
            output_range = (0, 1)
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

        # If input_range is not provided, try to get it from kwargs
        # This allows you to normalize based on the first frame in a sequence:
        # Example: https://github.com/tue-bmd/ultrasound-toolbox/pull/662
        if self.input_range is None:
            maxval = kwargs.get("maxval", None)
            minval = kwargs.get("minval", None)
        # If input_range is provided, use it
        else:
            minval, maxval = self.input_range

        # If input_range is still not provided, compute it from the data
        if minval is None:
            minval = ops.min(data)
        if maxval is None:
            maxval = ops.max(data)

        # Clip the data to the input range
        data = ops.clip(data, minval, maxval)

        # Map the data to the output range
        normalized_data = translate(data, (minval, maxval), self.output_range)

        return {self.output_key: normalized_data, "minval": minval, "maxval": maxval}


@ops_registry("scan_convert")
class ScanConvert(Operation):
    """Scan convert images to cartesian coordinates."""

    STATIC_PARAMS = ["fill_value"]

    def __init__(self, order=1, **kwargs):
        """Initialize the ScanConvert operation.

        Args:
            order (int, optional): Interpolation order. Defaults to 1. Currently only
                GPU support for order=1.
        """
        if order > 1:
            jittable = False
            log.warning(
                "GPU support for order > 1 is not available. "
                + "Disabling jit for ScanConvert."
            )
        else:
            jittable = True

        super().__init__(
            input_data_type=DataTypes.IMAGE,
            output_data_type=DataTypes.IMAGE_SC,
            jittable=jittable,
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
        if fill_value is None:
            fill_value = np.nan

        data = kwargs[self.key]

        if self._jit_compile and self.jittable:
            assert coordinates is not None, (
                "coordinates must be provided to jit scan conversion."
                "You can set ScanConvert(jit_compile=False) to disable jitting."
            )

        data_out, parameters = scan_convert(
            data,
            rho_range,
            theta_range,
            phi_range,
            resolution,
            coordinates,
            fill_value,
            self.order,
            with_batch_dim=self.with_batch_dim,
        )

        return {self.output_key: data_out, **parameters}


@ops_registry("gaussian_blur")
class GaussianBlur(Operation):
    """
    GaussianBlur is an operation that applies a Gaussian blur to an input image.
    Uses scipy.ndimage.gaussian_filter to create a kernel.
    """

    def __init__(
        self,
        sigma: float,
        kernel_size: int | None = None,
        pad_mode="symmetric",
        truncate=4.0,
        **kwargs,
    ):
        """
        Args:
            sigma (float): Standard deviation for Gaussian kernel.
            kernel_size (int, optional): The size of the kernel. If None, the kernel
                size is calculated based on the sigma and truncate. Default is None.
            pad_mode (str): Padding mode for the input image. Default is 'symmetric'.
            truncate (float): Truncate the filter at this many standard deviations.
        """
        super().__init__(**kwargs)
        if kernel_size is None:
            radius = round(truncate * sigma)
            self.kernel_size = 2 * radius + 1
        else:
            self.kernel_size = kernel_size
        self.sigma = sigma
        self.pad_mode = pad_mode
        self.radius = self.kernel_size // 2
        self.kernel = self.get_kernel()

    def get_kernel(self):
        """
        Create a gaussian kernel for blurring.

        Returns:
            kernel (Tensor): A gaussian kernel for blurring.
                Shape is (kernel_size, kernel_size, 1, 1).
        """
        n = np.zeros((self.kernel_size, self.kernel_size))
        n[self.radius, self.radius] = 1
        kernel = scipy.ndimage.gaussian_filter(
            n, sigma=self.sigma, mode="constant"
        ).astype(np.float32)
        kernel = kernel[:, :, None, None]
        return ops.convert_to_tensor(kernel)

    def call(self, **kwargs):

        data = kwargs[self.key]

        # Add batch dimension if not present
        if not self.with_batch_dim:
            data = data[None]

        # Add channel dimension to kernel
        kernel = ops.tile(self.kernel, (1, 1, data.shape[-1], data.shape[-1]))

        # Pad the input image according to the padding mode
        padded = ops.pad(
            data,
            [[0, 0], [self.radius, self.radius], [self.radius, self.radius], [0, 0]],
            mode=self.pad_mode,
        )

        # Apply the gaussian kernel to the padded image
        out = ops.conv(padded, kernel, padding="valid", data_format="channels_last")

        # Remove padding
        out = ops.slice(
            out,
            [0, 0, 0, 0],
            [out.shape[0], data.shape[1], data.shape[2], data.shape[3]],
        )

        # Remove batch dimension if it was not present before
        if not self.with_batch_dim:
            out = ops.squeeze(out, axis=0)

        return {self.output_key: out}


@ops_registry("lee_filter")
class LeeFilter(Operation):
    """
    The Lee filter is a speckle reduction filter commonly used in synthetic aperture radar (SAR)
    and ultrasound image processing. It smooths the image while preserving edges and details.
    This implementation uses Gaussian filter for local statistics and treats channels independently.

    Lee, J.S. (1980). Digital image enhancement and noise filtering by use of local statistics.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, (2), 165-168.
    """

    def __init__(self, sigma=3, kernel_size=None, pad_mode="symmetric", **kwargs):
        """
        Args:
            sigma (float): Standard deviation for Gaussian kernel. Default is 3.
            kernel_size (int, optional): Size of the Gaussian kernel. If None,
                it will be calculated based on sigma.
            pad_mode (str): Padding mode to be used for Gaussian blur. Default is "symmetric".
        """
        super().__init__(**kwargs)
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.pad_mode = pad_mode

        # Create a GaussianBlur instance for computing local statistics
        self.gaussian_blur = GaussianBlur(
            sigma=self.sigma,
            kernel_size=self.kernel_size,
            pad_mode=self.pad_mode,
            with_batch_dim=self.with_batch_dim,
            jittable=self._jittable,
            key=self.key,
        )

    def call(self, **kwargs):
        data = kwargs[self.key]

        # Apply Gaussian blur to get local mean
        img_mean = self.gaussian_blur.call(**kwargs)[self.gaussian_blur.output_key]

        # Apply Gaussian blur to squared data to get local squared mean
        data_squared = data**2
        kwargs[self.gaussian_blur.key] = data_squared
        img_sqr_mean = self.gaussian_blur.call(**kwargs)[self.gaussian_blur.output_key]

        # Calculate local variance
        img_variance = img_sqr_mean - img_mean**2

        # Calculate global variance (per channel)
        if self.with_batch_dim:
            overall_variance = ops.var(data, axis=(-3, -2), keepdims=True)
        else:
            overall_variance = ops.var(data, axis=(-2, -1), keepdims=True)

        # Calculate adaptive weights
        img_weights = img_variance / (img_variance + overall_variance)

        # Apply Lee filter formula
        img_output = img_mean + img_weights * (data - img_mean)

        return {self.output_key: img_output}


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


@ops_registry("companding")
class Companding(Operation):
    """Companding according to the A- or μ-law algorithm.

    Invertible compressing operation. Used to compress
    dynamic range of input data (and subsequently expand).

    μ-law companding:
    https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
    A-law companding:
    https://en.wikipedia.org/wiki/A-law_algorithm

    Args:
        expand (bool, optional): If set to False (default),
            data is compressed, else expanded.
        comp_type (str): either `a` or `mu`.
        mu (float, optional): compression parameter. Defaults to 255.
        A (float, optional): compression parameter. Defaults to 87.6.
    """

    def __init__(self, expand=False, comp_type="mu", **kwargs):
        super().__init__(**kwargs)
        self.expand = expand
        self.comp_type = comp_type.lower()
        if self.comp_type not in ["mu", "a"]:
            raise ValueError("comp_type must be 'mu' or 'a'.")

        if self.comp_type == "mu":
            self._compand_func = (
                self._mu_law_expand if self.expand else self._mu_law_compress
            )
        else:
            self._compand_func = (
                self._a_law_expand if self.expand else self._a_law_compress
            )

    # pylint: disable=unused-argument
    @staticmethod
    def _mu_law_compress(x, mu=255, **kwargs):
        x = ops.clip(x, -1, 1)
        return ops.sign(x) * ops.log(1.0 + mu * ops.abs(x)) / ops.log(1.0 + mu)

    # pylint: disable=unused-argument
    @staticmethod
    def _mu_law_expand(y, mu=255, **kwargs):
        y = ops.clip(y, -1, 1)
        return ops.sign(y) * ((1.0 + mu) ** ops.abs(y) - 1.0) / mu

    # pylint: disable=unused-argument
    @staticmethod
    def _a_law_compress(x, A=87.6, **kwargs):
        x = ops.clip(x, -1, 1)
        x_sign = ops.sign(x)
        x_abs = ops.abs(x)
        A_log = ops.log(A)
        val1 = x_sign * A * x_abs / (1.0 + A_log)
        val2 = x_sign * (1.0 + ops.log(A * x_abs)) / (1.0 + A_log)
        y = ops.where((x_abs >= 0) & (x_abs < (1.0 / A)), val1, val2)
        return y

    # pylint: disable=unused-argument
    @staticmethod
    def _a_law_expand(y, A=87.6, **kwargs):
        y = ops.clip(y, -1, 1)
        y_sign = ops.sign(y)
        y_abs = ops.abs(y)
        A_log = ops.log(A)
        val1 = y_sign * y_abs * (1.0 + A_log) / A
        val2 = y_sign * ops.exp(y_abs * (1.0 + A_log) - 1.0) / A
        x = ops.where((y_abs >= 0) & (y_abs < (1.0 / (1.0 + A_log))), val1, val2)
        return x

    def call(self, mu=255, A=87.6, **kwargs):
        data = kwargs[self.key]

        mu = ops.cast(mu, data.dtype)
        A = ops.cast(A, data.dtype)

        data_out = self._compand_func(data, mu=mu, A=A)
        return {self.output_key: data_out}


@ops_registry("downsample")
class Downsample(Operation):
    """Downsample data along a specific axis."""

    def __init__(self, factor: int = 1, phase: int = 0, axis: int = -3, **kwargs):
        super().__init__(
            **kwargs,
        )
        self.factor = factor
        self.phase = phase
        self.axis = axis

    def call(self, **kwargs):
        data = kwargs[self.key]
        length = ops.shape(data)[self.axis]
        sample_idx = ops.arange(self.phase, length, self.factor)
        data_downsampled = ops.take(data, sample_idx, axis=self.axis)

        # downsampling also affects the sampling frequency
        if "sampling_frequency" in kwargs:
            kwargs["sampling_frequency"] = kwargs["sampling_frequency"] / self.factor
            kwargs["n_ax"] = kwargs["n_ax"] // self.factor
        return {
            self.output_key: data_downsampled,
            "sampling_frequency": kwargs["sampling_frequency"],
            "n_ax": kwargs["n_ax"],
        }


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


@ops_registry("threshold")
class Threshold(Operation):
    """Threshold an array, setting values below/above a threshold to a fill value."""

    def __init__(
        self,
        threshold_type="hard",
        below_threshold=True,
        fill_value="min",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if threshold_type not in ("hard", "soft"):
            raise ValueError("threshold_type must be 'hard' or 'soft'")
        self.threshold_type = threshold_type
        self.below_threshold = below_threshold
        self._fill_value_type = fill_value

        # Define threshold function at init
        if threshold_type == "hard":
            if below_threshold:
                self._threshold_func = lambda data, threshold, fill: ops.where(
                    data < threshold, fill, data
                )
            else:
                self._threshold_func = lambda data, threshold, fill: ops.where(
                    data > threshold, fill, data
                )
        else:  # soft
            if below_threshold:
                self._threshold_func = (
                    lambda data, threshold, fill: ops.maximum(data - threshold, 0)
                    + fill
                )
            else:
                self._threshold_func = (
                    lambda data, threshold, fill: ops.minimum(data - threshold, 0)
                    + fill
                )

    def _resolve_fill_value(self, data, threshold):
        """Get the fill value based on the fill_value_type."""
        fv = self._fill_value_type
        if isinstance(fv, (int, float)):
            return ops.convert_to_tensor(fv, dtype=data.dtype)
        elif fv == "min":
            return ops.min(data)
        elif fv == "max":
            return ops.max(data)
        elif fv == "threshold":
            return threshold
        else:
            raise ValueError("Unknown fill_value")

    def call(
        self,
        threshold=None,
        percentile=None,
        **kwargs,
    ):
        """Threshold the input data.

        Args:
            threshold: Numeric threshold.
            percentile: Percentile to derive threshold from.
        Returns:
            Tensor with thresholding applied.
        """
        data = kwargs[self.key]
        if (threshold is None) == (percentile is None):
            raise ValueError(
                "Pass either threshold or percentile, not both or neither."
            )

        if percentile is not None:
            # Convert percentile to quantile value (0-1 range)
            threshold = ops.quantile(data, percentile / 100.0)

        fill_value = self._resolve_fill_value(data, threshold)
        result = self._threshold_func(data, threshold, fill_value)
        return {self.output_key: result}


@ops_registry("bm3d")
class BM3DDenoise(Operation):
    """Block Matching 3D (BM3D) denoising operation."""

    def __init__(self, stage="all_stages", **kwargs):
        super().__init__(
            jittable=False,
            **kwargs,
        )

        if "arm" in check_architecture():
            raise ValueError(
                log.error("BM3D denoiser is not supported on ARM architecture.")
            )

        # pylint: disable=import-outside-toplevel
        import bm3d as _bm3d_module

        self._bm3d_module = _bm3d_module

        str_to_stage = {
            "hard_thresholding": self._bm3d_module.BM3DStages.HARD_THRESHOLDING,
            "all_stages": self._bm3d_module.BM3DStages.ALL_STAGES,
        }
        self.stage = str_to_stage[stage]

    def call(self, sigma=0.1, **kwargs):
        """BM3D denoising operation.

        Args:
            sigma: Noise standard deviation.

        Returns:
            Denoised image(s), same shape as input.
        """
        data = kwargs[self.key]

        # Convert to numpy for bm3d (cpu)
        data_np = ops.convert_to_numpy(data)

        if not self.with_batch_dim:
            data_np = np.expand_dims(data_np, axis=0)

        denoised = [
            self._bm3d_module.bm3d(img, sigma, stage_arg=self.stage) for img in data_np
        ]
        denoised = np.stack(denoised)

        if not self.with_batch_dim:
            denoised = np.squeeze(denoised, axis=0)

        result = ops.convert_to_tensor(denoised, dtype=data.dtype)
        return {self.output_key: result}


@ops_registry("anisotropic_diffusion")
class AnisotropicDiffusion(Operation):
    """Speckle Reducing Anisotropic Diffusion (SRAD) filter.

    Reference:
    - https://www.researchgate.net/publication/5602035_Speckle_reducing_anisotropic_diffusion
    - https://nl.mathworks.com/matlabcentral/fileexchange/54044-image-despeckle-filtering-toolbox
    """

    def call(self, niter=100, lmbda=0.1, rect=None, eps=1e-6, **kwargs):
        """Anisotropic diffusion filter.

        Assumes input data is non-negative.

        Args:
            niter: Number of iterations.
            lmbda: Lambda parameter.
            rect: Rectangle [x1, y1, x2, y2] for homogeneous noise (optional).
            eps: Small epsilon for stability.
        Returns:
            Filtered image (2D tensor or batch of images).
        """
        data = kwargs[self.key]

        if not self.with_batch_dim:
            data = ops.expand_dims(data, axis=0)

        batch_size = ops.shape(data)[0]

        results = []
        for i in range(batch_size):
            image = data[i]
            image_out = self._anisotropic_diffusion_single(
                image, niter, lmbda, rect, eps
            )
            results.append(image_out)

        result = ops.stack(results, axis=0)

        if not self.with_batch_dim:
            result = ops.squeeze(result, axis=0)

        return {self.output_key: result}

    def _anisotropic_diffusion_single(self, image, niter, lmbda, rect, eps):
        """Apply anisotropic diffusion to a single image (2D)."""
        image = ops.exp(image)
        M, N = image.shape

        for _ in range(niter):
            iN = ops.concatenate(
                [image[1:], ops.zeros((1, N), dtype=image.dtype)], axis=0
            )
            iS = ops.concatenate(
                [ops.zeros((1, N), dtype=image.dtype), image[:-1]], axis=0
            )
            jW = ops.concatenate(
                [image[:, 1:], ops.zeros((M, 1), dtype=image.dtype)], axis=1
            )
            jE = ops.concatenate(
                [ops.zeros((M, 1), dtype=image.dtype), image[:, :-1]], axis=1
            )

            if rect is not None:
                x1, y1, x2, y2 = rect
                imageuniform = image[x1:x2, y1:y2]
                q0_squared = (
                    ops.std(imageuniform) / (ops.mean(imageuniform) + eps)
                ) ** 2

            dN = iN - image
            dS = iS - image
            dW = jW - image
            dE = jE - image

            G2 = (dN**2 + dS**2 + dW**2 + dE**2) / (image**2 + eps)
            L = (dN + dS + dW + dE) / (image + eps)
            num = (0.5 * G2) - ((1 / 16) * (L**2))
            den = (1 + ((1 / 4) * L)) ** 2
            q_squared = num / (den + eps)

            if rect is not None:
                den = (q_squared - q0_squared) / (q0_squared * (1 + q0_squared) + eps)
            c = 1.0 / (1 + den)
            cS = ops.concatenate([ops.zeros((1, N), dtype=image.dtype), c[:-1]], axis=0)
            cE = ops.concatenate(
                [ops.zeros((M, 1), dtype=image.dtype), c[:, :-1]], axis=1
            )

            D = (cS * dS) + (c * dN) + (cE * dE) + (c * dW)
            image = image + (lmbda / 4) * D

        result = ops.log(image)
        return result


def demodulate_not_jitable(
    rf_data,
    sampling_frequency=None,
    center_frequency=None,
    bandwidth=None,
    filter_coeff=None,
):
    """Demodulates an RF signal to complex base-band (IQ).

    Demodulates the radiofrequency (RF) bandpass signals and returns the
    Inphase/Quadrature (I/Q) components. IQ is a complex whose real (imaginary)
    part contains the in-phase (quadrature) component.

    This function operates (i.e. demodulates) on the RF signal over the
    (fast-) time axis which is assumed to be the last axis.

    Args:
        rf_data (ndarray): real valued input array of size [..., n_ax, n_el].
            second to last axis is fast-time axis.
        sampling_frequency (float): the sampling frequency of the RF signals (in Hz).
            Only not necessary when filter_coeff is provided.
        center_frequency (float, optional): represents the center frequency (in Hz).
            Defaults to None.
        bandwidth (float, optional): Bandwidth of RF signal in % of center
            frequency. Defaults to None.
            The bandwidth in % is defined by:
            B = Bandwidth_in_% = Bandwidth_in_Hz*(100/center_frequency).
            The cutoff frequency:
            Wn = Bandwidth_in_Hz/sampling_frequency, i.e:
            Wn = B*(center_frequency/100)/sampling_frequency.
        filter_coeff (list, optional): (b, a), numerator and denominator coefficients
            of FIR filter for quadratic band pass filter. All other parameters are ignored
            if filter_coeff are provided. Instead the given filter_coeff is directly used.
            If not provided, a filter is derived from the other params (sampling_frequency,
            center_frequency, bandwidth).
            see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html

    Returns:
        iq_data (ndarray): complex valued base-band signal.

    """
    rf_data = ops.convert_to_numpy(rf_data)
    assert np.isreal(
        rf_data
    ).all(), f"RF must contain real RF signals, got {rf_data.dtype}"

    input_shape = rf_data.shape
    n_dim = len(input_shape)
    if n_dim > 2:
        *_, n_ax, n_el = input_shape
    else:
        n_ax, n_el = input_shape

    if filter_coeff is None:
        assert (
            sampling_frequency is not None
        ), "provide sampling_frequency when no filter is given."
        # Time vector
        t = np.arange(n_ax) / sampling_frequency
        t0 = 0
        t = t + t0

        # Estimate center frequency
        if center_frequency is None:
            # Keep a maximum of 100 randomly selected scanlines
            idx = np.arange(n_el)
            if n_el > 100:
                idx = np.random.permutation(idx)[:100]
            # Power Spectrum
            P = np.sum(
                np.abs(np.fft.fft(np.take(rf_data, idx, axis=-1), axis=-2)) ** 2,
                axis=-1,
            )
            P = P[: n_ax // 2]
            # Carrier frequency
            idx = np.sum(np.arange(n_ax // 2) * P) / np.sum(P)
            center_frequency = idx * sampling_frequency / n_ax

        # Normalized cut-off frequency
        if bandwidth is None:
            Wn = min(2 * center_frequency / sampling_frequency, 0.5)
            bandwidth = center_frequency * Wn
        else:
            assert np.isscalar(
                bandwidth
            ), "The signal bandwidth (in %) must be a scalar."
            assert (bandwidth > 0) & (
                bandwidth <= 200
            ), "The signal bandwidth (in %) must be within the interval of ]0,200]."
            # bandwidth in Hz
            bandwidth = center_frequency * bandwidth / 100
            Wn = bandwidth / sampling_frequency
        assert (Wn > 0) & (Wn <= 1), (
            "The normalized cutoff frequency is not within the interval of (0,1). "
            "Check the input parameters!"
        )

        # Down-mixing of the RF signals
        carrier = np.exp(-1j * 2 * np.pi * center_frequency * t)
        # add the singleton dimensions
        carrier = np.reshape(carrier, (*[1] * (n_dim - 2), n_ax, 1))
        iq_data = rf_data * carrier

        # Low-pass filter
        N = 5
        b, a = scipy.signal.butter(N, Wn, "low")

        # factor 2: to preserve the envelope amplitude
        iq_data = scipy.signal.filtfilt(b, a, iq_data, axis=-2) * 2

        # Display a warning message if harmful aliasing is suspected
        # the RF signal is undersampled
        if sampling_frequency < (2 * center_frequency + bandwidth):
            # lower and higher frequencies of the bandpass signal
            fL = center_frequency - bandwidth / 2
            fH = center_frequency + bandwidth / 2
            n = fH // (fH - fL)
            harmless_aliasing = any(
                (2 * fH / np.arange(1, n) <= sampling_frequency)
                & (sampling_frequency <= 2 * fL / np.arange(1, n))
            )
            if not harmless_aliasing:
                log.warning(
                    "rf2iq:harmful_aliasing Harmful aliasing is present: the aliases"
                    " are not mutually exclusive!"
                )
    else:
        b, a = filter_coeff
        iq_data = scipy.signal.lfilter(b, a, rf_data, axis=-2) * 2

    return iq_data


def upmix(iq_data, sampling_frequency, center_frequency, upsampling_rate=6):
    """Upsamples and upmixes complex base-band signals (IQ) to RF.

    Args:
        iq_data (ndarray): complex valued input array of size [..., n_ax, n_el]. second
            to last axis is fast-time axis.
        sampling_frequency (float): the sampling frequency of the input IQ signal (in Hz).
            resulting sampling_frequency of RF data is upsampling_rate times higher.
        center_frequency (float, optional): represents the center frequency (in Hz).

    Returns:
        rf_data (ndarray): output real valued rf data.
    """
    assert iq_data.dtype in [
        "complex64",
        "complex128",
    ], "IQ must contain all complex signals."

    input_shape = iq_data.shape
    n_dim = len(input_shape)
    if n_dim > 2:
        *_, n_ax, _ = input_shape
    else:
        n_ax, _ = input_shape

    # Time vector
    n_ax_up = n_ax * upsampling_rate
    sampling_frequency_up = sampling_frequency * upsampling_rate

    t = ops.arange(n_ax_up, dtype="float32") / sampling_frequency_up
    t0 = 0
    t = t + t0

    iq_data_upsampled = resample(
        iq_data,
        n_samples=n_ax_up,
        axis=-2,
        order=1,
    )

    # Up-mixing of the IQ signals
    t = ops.cast(t, dtype="complex64")
    center_frequency = ops.cast(center_frequency, dtype="complex64")
    carrier = ops.exp(1j * 2 * np.pi * center_frequency * t)
    carrier = ops.reshape(carrier, (*[1] * (n_dim - 2), n_ax_up, 1))

    rf_data = iq_data_upsampled * carrier
    rf_data = ops.real(rf_data) * ops.sqrt(2)

    return ops.cast(rf_data, "float32")


def get_band_pass_filter(num_taps, sampling_frequency, f1, f2):
    """Band pass filter

    Args:
        num_taps (int): number of taps in filter.
        sampling_frequency (float): sample frequency in Hz.
        f1 (float): cutoff frequency in Hz of left band edge.
        f2 (float): cutoff frequency in Hz of right band edge.

    Returns:
        ndarray: band pass filter
    """
    bpf = scipy.signal.firwin(
        num_taps, [f1, f2], pass_zero=False, fs=sampling_frequency
    )
    return bpf


def get_low_pass_iq_filter(num_taps, sampling_frequency, f, bw):
    """Design low pass filter.

    LPF with num_taps points and cutoff at bw / 2

    Args:
        num_taps (int): number of taps in filter.
        sampling_frequency (float): sample frequency.
        f (float): center frequency.
        bw (float): bandwidth in Hz.
    Raises:
        AssertionError: if cutoff frequency (bw / 2) is not within (0, sampling_frequency / 2)

    Returns:
        ndarray: fx LP filter
    """
    assert (bw / 2 > 0) & (bw / 2 < sampling_frequency / 2), log.error(
        "Cutoff frequency must be within (0, sampling_frequency / 2), "
        f"got {bw / 2} Hz, must be within (0, {sampling_frequency / 2}) Hz"
    )
    t_qbp = np.arange(num_taps) / sampling_frequency
    lpf = scipy.signal.firwin(
        num_taps, bw / 2, pass_zero=True, fs=sampling_frequency
    ) * np.exp(1j * 2 * np.pi * f * t_qbp)
    return lpf


def complex_to_channels(complex_data, axis=-1):
    """Unroll complex data to separate channels.

    Args:
        complex_data (complex ndarray): complex input data.
        axis (int, optional): on which axis to extend. Defaults to -1.

    Returns:
        ndarray: real array with real and imaginary components
            unrolled over two channels at axis.
    """
    # assert ops.iscomplex(complex_data).any()
    q_data = ops.imag(complex_data)
    i_data = ops.real(complex_data)

    i_data = ops.expand_dims(i_data, axis=axis)
    q_data = ops.expand_dims(q_data, axis=axis)

    iq_data = ops.concatenate((i_data, q_data), axis=axis)
    return iq_data


def channels_to_complex(data):
    """Convert array with real and imaginary components at
    different channels to complex data array.

    Args:
        data (ndarray): input data, with at 0 index of axis
            real component and 1 index of axis the imaginary.

    Returns:
        ndarray: complex array with real and imaginary components.
    """
    assert data.shape[-1] == 2, "Data must have two channels."
    data = ops.cast(data, "complex64")
    return data[..., 0] + 1j * data[..., 1]


def hilbert(x, N: int = None, axis=-1):
    """Manual implementation of the Hilbert transform function. Tje function
    returns the analytical signal.

    Operated in the Fourier domain.

    Note:
        THIS IS NOT THE MATHEMATICAL THE HILBERT TRANSFORM as you will find it on
        wikipedia, but computes the analytical signal. The implementation reproduces
        the behavior of the `scipy.signal.hilbert` function.

    Args:
        x (ndarray): input data of any shape.
        N (int, optional): number of points in the FFT. Defaults to None.
        axis (int, optional): axis to operate on. Defaults to -1.
    Returns:
        x (ndarray): complex iq data of any shape.k

    """
    input_shape = x.shape
    n_dim = len(input_shape)

    n_ax = input_shape[axis]

    if axis < 0:
        axis = n_dim + axis

    if N is not None:
        if N < n_ax:
            raise ValueError("N must be greater or equal to n_ax.")
        # only pad along the axis, use manual padding
        pad = N - n_ax
        zeros = ops.zeros(
            input_shape[:axis] + (pad,) + input_shape[axis + 1 :],
        )

        x = ops.concatenate((x, zeros), axis=axis)
    else:
        N = n_ax

    # Create filter to zero out negative frequencies
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1 : N // 2] = 2
    else:
        h[0] = 1
        h[1 : (N + 1) // 2] = 2

    idx = list(range(n_dim))
    # make sure axis gets to the end for fft (operates on last axis)
    idx.remove(axis)
    idx.append(axis)
    x = ops.transpose(x, idx)

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[-1] = slice(None)
        h = h[tuple(ind)]

    h = ops.convert_to_tensor(h)
    h = ops.cast(h, "complex64")
    h = h + 1j * ops.zeros_like(h)

    Xf_r, Xf_i = ops.fft((x, ops.zeros_like(x)))

    Xf_r = ops.cast(Xf_r, "complex64")
    Xf_i = ops.cast(Xf_i, "complex64")

    Xf = Xf_r + 1j * Xf_i
    Xf = Xf * h

    # x = np.fft.ifft(Xf)
    # do manual ifft using fft
    Xf_r = ops.real(Xf)
    Xf_i = ops.imag(Xf)
    Xf_r_inv, Xf_i_inv = ops.fft((Xf_r, -Xf_i))

    Xf_i_inv = ops.cast(Xf_i_inv, "complex64")
    Xf_r_inv = ops.cast(Xf_r_inv, "complex64")

    x = Xf_r_inv / N
    x = x + 1j * (-Xf_i_inv / N)

    # switch back to original shape
    idx = list(range(n_dim))
    idx.insert(axis, idx.pop(-1))
    x = ops.transpose(x, idx)
    return x


def demodulate(data, center_frequency, sampling_frequency, axis=-3):
    """Demodulates the input data to baseband. The function computes the analytical
    signal (the signal with negative frequencies removed) and then shifts the spectrum
    of the signal to baseband by multiplying with a complex exponential. Where the
    spectrum was centered around `center_frequency` before, it is now centered around
    0 Hz. The baseband IQ data are complex-valued. The real and imaginary parts
    are stored in two real-valued channels.

    Args:
        data (ops.Tensor): The input data to demodulate of shape `(..., axis, ..., 1)`.
        center_frequency (float): The center frequency of the signal.
        sampling_frequency (float): The sampling frequency of the signal.
        axis (int, optional): The axis along which to demodulate. Defaults to -3.

    Returns:
        ops.Tensor: The demodulated IQ data of shape `(..., axis, ..., 2)`.
    """
    # Compute the analytical signal
    analytical_signal = hilbert(data, axis=axis)

    # Define frequency indices
    frequency_indices = ops.arange(analytical_signal.shape[axis])

    # Expand the frequency indices to match the shape of the RF data
    indexing = [None] * data.ndim
    indexing[axis] = slice(None)
    indexing = tuple(indexing)
    frequency_indices_shaped_like_rf = frequency_indices[indexing]

    # Cast to complex64
    center_frequency = ops.cast(center_frequency, dtype="complex64")
    sampling_frequency = ops.cast(sampling_frequency, dtype="complex64")
    frequency_indices_shaped_like_rf = ops.cast(
        frequency_indices_shaped_like_rf, dtype="complex64"
    )

    # Shift to baseband
    phasor_exponent = (
        -1j
        * 2
        * np.pi
        * center_frequency
        * frequency_indices_shaped_like_rf
        / sampling_frequency
    )
    iq_data_signal_complex = analytical_signal * ops.exp(phasor_exponent)

    # Split the complex signal into two channels
    iq_data_two_channel = complex_to_channels(iq_data_signal_complex[..., 0])

    return iq_data_two_channel
