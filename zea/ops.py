"""Operations and Pipelines for ultrasound data processing.

This module contains two important classes, :class:`Operation` and :class:`Pipeline`,
which are used to process ultrasound data. A pipeline is a sequence of operations
that are applied to the data in a specific order.

Stand-alone manual usage
------------------------

Operations can be run on their own:

Examples
^^^^^^^^
.. doctest::

    >>> import numpy as np
    >>> from zea.ops import EnvelopeDetect
    >>> data = np.random.randn(2000, 128, 1)
    >>> # static arguments are passed in the constructor
    >>> envelope_detect = EnvelopeDetect(axis=-1)
    >>> # other parameters can be passed here along with the data
    >>> envelope_data = envelope_detect(data=data)

Using a pipeline
----------------

You can initialize with a default pipeline or create your own custom pipeline.

.. doctest::

    >>> from zea.ops import Pipeline, EnvelopeDetect, Normalize, LogCompress
    >>> pipeline = Pipeline.from_default()

    >>> operations = [
    ...     EnvelopeDetect(),
    ...     Normalize(),
    ...     LogCompress(),
    ... ]
    >>> pipeline_custom = Pipeline(operations)

One can also load a pipeline from a config or yaml/json file:

.. doctest::

    >>> from zea import Pipeline

    >>> # From JSON string
    >>> json_string = '{"operations": ["identity"]}'
    >>> pipeline = Pipeline.from_json(json_string)

    >>> # from yaml file
    >>> import yaml
    >>> from zea import Config
    >>> # Create a sample pipeline YAML file
    >>> pipeline_dict = {
    ...     "operations": [
    ...         {"name": "identity"},
    ...     ]
    ... }
    >>> with open("pipeline.yaml", "w") as f:
    ...     yaml.dump(pipeline_dict, f)
    >>> yaml_file = "pipeline.yaml"
    >>> pipeline = Pipeline.from_yaml(yaml_file)

.. testcleanup::

    import os

    os.remove("pipeline.yaml")

Example of a yaml file:

.. code-block:: yaml

    pipeline:
      operations:
        - name: demodulate
        - name: "patched_grid"
          params:
            operations:
              - name: tof_correction
              - name: pfield_weighting
              - name: delay_and_sum
            num_patches: 100
        - name: envelope_detect
        - name: normalize
        - name: log_compress

"""

import hashlib
import inspect
import json
import uuid
from functools import partial
from typing import Any, Dict, List, Union

import keras
import numpy as np
import scipy
import yaml
from keras import ops
from keras.src.layers.preprocessing.data_layer import DataLayer

from zea import log
from zea.backend import jit
from zea.beamform.beamformer import tof_correction
from zea.config import Config
from zea.display import scan_convert
from zea.internal.checks import _assert_keys_and_axes
from zea.internal.core import (
    DEFAULT_DYNAMIC_RANGE,
    DataTypes,
    ZEADecoderJSON,
    ZEAEncoderJSON,
    dict_to_tensor,
)
from zea.internal.core import Object as ZEAObject
from zea.internal.registry import ops_registry
from zea.probes import Probe
from zea.scan import Scan
from zea.simulator import simulate_rf
from zea.tensor_ops import (
    apply_along_axis,
    correlate,
    extend_n_dims,
    resample,
    reshape_axis,
    translate,
    vmap,
)
from zea.utils import (
    FunctionTimer,
    deep_compare,
    map_negative_indices,
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
        timed: bool = False,
    ):
        """
        Initialize a pipeline.

        Args:
            operations (list): A list of Operation instances representing the operations
                to be performed.
            with_batch_dim (bool, optional): Whether operations should expect a batch dimension.
                Defaults to True.
            jit_options (str, optional): The JIT options to use. Must be "pipeline", "ops", or None.

                - "pipeline": compiles the entire pipeline as a single function.
                  This may be faster but does not preserve python control flow, such as caching.

                - "ops": compiles each operation separately. This preserves python control flow and
                  caching functionality, but speeds up the operations.

                - None: disables JIT compilation.

                Defaults to "ops".

            jit_kwargs (dict, optional): Additional keyword arguments for the JIT compiler.
            name (str, optional): The name of the pipeline. Defaults to "pipeline".
            validate (bool, optional): Whether to validate the pipeline. Defaults to True.
            timed (bool, optional): Whether to time each operation. Defaults to False.

        """
        self._call_pipeline = self.call
        self.name = name

        self._pipeline_layers = operations

        if jit_options not in ["pipeline", "ops", None]:
            raise ValueError("jit_options must be 'pipeline', 'ops', or None")

        self.with_batch_dim = with_batch_dim
        self._validate_flag = validate

        # Setup timer
        if jit_options == "pipeline" and timed:
            raise ValueError(
                "timed=True cannot be used with jit_options='pipeline' as the entire "
                "pipeline is compiled into a single function. Try setting jit_options to "
                "'ops' or None."
            )
        if timed:
            log.warning(
                "Timer has been initialized for the pipeline. To get an accurate timing estimate, "
                "the `block_until_ready()` is used, which will slow down the execution, so "
                "do not use for regular processing!"
            )
            self._callable_layers = self._get_timed_operations()
        else:
            self._callable_layers = self._pipeline_layers
        self._timed = timed

        if validate:
            self.validate()
        else:
            log.warning("Pipeline validation is disabled, make sure to validate manually.")

        if jit_kwargs is None:
            jit_kwargs = {}

        if keras.backend.backend() == "jax" and self.static_params != []:
            jit_kwargs = {"static_argnames": self.static_params}

        self.jit_kwargs = jit_kwargs
        self.jit_options = jit_options  # will handle the jit compilation

        self._logged_difference_keys = False

        # Do not log again for nested pipelines
        for nested_pipeline in self._nested_pipelines:
            nested_pipeline._logged_difference_keys = True

    def needs(self, key) -> bool:
        """Check if the pipeline needs a specific key at the input."""
        return key in self.needs_keys

    @property
    def _nested_pipelines(self):
        return [operation for operation in self.operations if isinstance(operation, Pipeline)]

    @property
    def output_keys(self) -> set:
        """All output keys the pipeline guarantees to produce."""
        output_keys = set()
        for operation in self.operations:
            output_keys.update(operation.output_keys)
        return output_keys

    @property
    def valid_keys(self) -> set:
        """Get a set of valid keys for the pipeline.

        This is all keys that can be passed to the pipeline as input.
        """
        valid_keys = set()
        for operation in self.operations:
            valid_keys.update(operation.valid_keys)
        return valid_keys

    @property
    def static_params(self) -> List[str]:
        """Get a list of static parameters for the pipeline."""
        static_params = []
        for operation in self.operations:
            static_params.extend(operation.static_params)
        return list(set(static_params))

    @property
    def needs_keys(self) -> set:
        """Get a set of all input keys needed by the pipeline.

        Will keep track of keys that are already provided by previous operations.
        """
        needs = set()
        has_so_far = set()
        previous_operation = None
        for operation in self.operations:
            if previous_operation is not None:
                has_so_far.update(previous_operation.output_keys)
            needs.update(operation.needs_keys - has_so_far)
            previous_operation = operation
        return needs

    @classmethod
    def from_default(
        cls, num_patches=100, baseband=False, pfield=False, timed=False, **kwargs
    ) -> "Pipeline":
        """Create a default pipeline.

        Args:
            num_patches (int): Number of patches for the PatchedGrid operation.
                Defaults to 100. If you get an out of memory error, try to increase this number.
            baseband (bool): If True, assume the input data is baseband (I/Q) data,
                which has 2 channels (last dim). Defaults to False, which assumes RF data,
                so input signal has a single channel dim and is still on carrier frequency.
            pfield (bool): If True, apply Pfield weighting. Defaults to False.
                This will calculate pressure field and only beamform the data to those locations.
            timed (bool, optional): Whether to time each operation. Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the Pipeline constructor.

        """
        operations = []

        # Add the demodulate operation
        if not baseband:
            operations.append(Demodulate())

        # Get beamforming ops
        beamforming = [
            TOFCorrection(),
            DelayAndSum(),
        ]
        if pfield:
            beamforming.insert(1, PfieldWeighting())

        # Optionally add patching
        if num_patches > 1:
            beamforming = [PatchedGrid(operations=beamforming, num_patches=num_patches, **kwargs)]

        # Add beamforming ops
        operations += beamforming

        # Add display ops
        operations += [
            ReshapeGrid(),
            EnvelopeDetect(),
            Normalize(),
            LogCompress(),
        ]
        return cls(operations, timed=timed, **kwargs)

    def copy(self) -> "Pipeline":
        """Create a copy of the pipeline."""
        return Pipeline(
            self._pipeline_layers.copy(),
            with_batch_dim=self.with_batch_dim,
            jit_options=self.jit_options,
            jit_kwargs=self.jit_kwargs,
            name=self.name,
            validate=self._validate_flag,
            timed=self._timed,
        )

    def reinitialize(self):
        """Reinitialize the pipeline in place."""
        self.__init__(
            self._pipeline_layers,
            with_batch_dim=self.with_batch_dim,
            jit_options=self.jit_options,
            jit_kwargs=self.jit_kwargs,
            name=self.name,
            validate=self._validate_flag,
            timed=self._timed,
        )

    def prepend(self, operation: Operation):
        """Prepend an operation to the pipeline."""
        self._pipeline_layers.insert(0, operation)
        self.reinitialize()

    def append(self, operation: Operation):
        """Append an operation to the pipeline."""
        self._pipeline_layers.append(operation)
        self.reinitialize()

    def insert(self, index: int, operation: Operation):
        """Insert an operation at a specific index in the pipeline."""
        if index < 0 or index > len(self._pipeline_layers):
            raise IndexError("Index out of bounds for inserting operation.")
        self._pipeline_layers.insert(index, operation)
        self.reinitialize()

    @property
    def operations(self):
        """Alias for self.layers to match the zea naming convention"""
        return self._pipeline_layers

    def reset_timer(self):
        """Reset the timer for timed operations."""
        if self._timed:
            self._callable_layers = self._get_timed_operations()
        else:
            log.warning(
                "Timer has not been initialized. Set timed=True when initializing the pipeline."
            )

    def _get_timed_operations(self):
        """Get a list of timed operations."""
        self.timer = FunctionTimer()
        return [self.timer(op, name=op.__class__.__name__) for op in self._pipeline_layers]

    def call(self, **inputs):
        """Process input data through the pipeline."""
        for operation in self._callable_layers:
            try:
                outputs = operation(**inputs)
            except KeyError as exc:
                raise KeyError(
                    f"[zea.Pipeline] Operation '{operation.__class__.__name__}' "
                    f"requires input key '{exc.args[0]}', "
                    "but it was not provided in the inputs.\n"
                    "Check whether the objects (such as `zea.Scan`) passed to "
                    "`pipeline.prepare_parameters()` contain all required keys.\n"
                    f"Current list of all passed keys: {list(inputs.keys())}\n"
                    f"Valid keys for this pipeline: {self.valid_keys}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"[zea.Pipeline] Error in operation '{operation.__class__.__name__}': {exc}"
                )
            inputs = outputs
        return outputs

    def __call__(self, return_numpy=False, **inputs):
        """Process input data through the pipeline."""

        if any(key in inputs for key in ["probe", "scan", "config"]) or any(
            isinstance(arg, ZEAObject) for arg in inputs.values()
        ):
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

        if not self._logged_difference_keys:
            difference_keys = set(inputs.keys()) - self.valid_keys
            if difference_keys:
                log.debug(
                    f"[zea.Pipeline] The following input keys are not used by the pipeline: "
                    f"{difference_keys}. Make sure this is intended. "
                    "This warning will only be shown once."
                )
                self._logged_difference_keys = True

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
        return self._with_batch_dim

    @with_batch_dim.setter
    def with_batch_dim(self, value):
        """Set the with_batch_dim property of the pipeline."""
        self._with_batch_dim = value
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
                key: value for key, value in params.items() if key in operation.valid_keys
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
        """String representation of the pipeline."""
        operations = [operation.__class__.__name__ for operation in self.operations]
        string = " -> ".join(operations)
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
        """Create a pipeline from a dictionary or ``zea.Config`` object.

        Args:
            config (dict or Config): Configuration dictionary or ``zea.Config`` object.
            **kwargs: Additional keyword arguments to be passed to the pipeline.

        Note:
            Must have a ``pipeline`` key with a subkey ``operations``.

        Example:
            .. doctest::

                >>> from zea import Config, Pipeline
                >>> config = Config(
                ...     {
                ...         "operations": [
                ...             "identity",
                ...         ],
                ...     }
                ... )
                >>> pipeline = Pipeline.from_config(config)
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
            .. doctest::

                >>> import yaml
                >>> from zea import Config
                >>> # Create a sample pipeline YAML file
                >>> pipeline_dict = {
                ...     "operations": [
                ...         "identity",
                ...     ],
                ... }
                >>> with open("pipeline.yaml", "w") as f:
                ...     yaml.dump(pipeline_dict, f)
                >>> from zea.ops import Pipeline
                >>> pipeline = Pipeline.from_yaml("pipeline.yaml", jit_options=None)
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
        """Convert the pipeline to a `zea.Config` object."""
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

        Serializes `zea.core.Object` instances and converts them to
        dictionary of tensors.

        Args:
            probe: Probe object.
            scan: Scan object.
            config: Config object.
            include (None, "all", or list): Only include these parameter/computed property names.
                If None or "all", include all.
            exclude (None or list): Exclude these parameter/computed property names.
                If provided, these keys will be excluded from the output.
                Only one of include or exclude can be set.

            **kwargs: Additional keyword arguments to be included in the inputs.

        Returns:
            dict: Dictionary of inputs with all values as tensors.
        """
        # Initialize dictionaries for probe, scan, and config
        probe_dict, scan_dict, config_dict = {}, {}, {}

        # Process args to extract Probe, Scan, and Config objects
        if probe is not None:
            assert isinstance(probe, Probe), (
                f"Expected an instance of `zea.probes.Probe`, got {type(probe)}"
            )
            probe_dict = probe.to_tensor(keep_as_is=self.static_params)

        if scan is not None:
            assert isinstance(scan, Scan), (
                f"Expected an instance of `zea.scan.Scan`, got {type(scan)}"
            )
            scan_dict = scan.to_tensor(include=self.needs_keys, keep_as_is=self.static_params)

        if config is not None:
            assert isinstance(config, Config), (
                f"Expected an instance of `zea.config.Config`, got {type(config)}"
            )
            config_dict.update(config.to_tensor(keep_as_is=self.static_params))

        # Convert all kwargs to tensors
        tensor_kwargs = dict_to_tensor(kwargs, keep_as_is=self.static_params)

        # combine probe, scan, config and kwargs
        # explicitly so we know which keys overwrite which
        # kwargs > config > scan > probe
        inputs = {
            **probe_dict,
            **scan_dict,
            **config_dict,
            **tensor_kwargs,
        }

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

    Example:
        .. doctest::

            >>> from zea.ops import make_operation_chain, LogCompress
            >>> SomeCustomOperation = LogCompress  # just for demonstration
            >>> chain = make_operation_chain(
            ...     [
            ...         "envelope_detect",
            ...         {"name": "normalize", "params": {"output_range": (0, 1)}},
            ...         SomeCustomOperation(),
            ...     ]
            ... )
    """
    chain = []
    for operation in operation_chain:
        # Handle already instantiated Operation or Pipeline objects
        if isinstance(operation, (Operation, Pipeline)):
            chain.append(operation)
            continue

        assert isinstance(operation, (str, dict, Config)), (
            f"Operation {operation} should be a string, dict, Config object, Operation, or Pipeline"
        )

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
                    operation_instance = operation_cls(operations=nested_operations, **params)
                else:
                    operation_instance = operation_cls(operations=nested_operations, **params)
            elif operation["name"] in ["patched_grid"]:
                nested_operations = make_operation_chain(operation["params"].pop("operations"))
                operation_instance = operation_cls(operations=nested_operations, **params)
            else:
                operation_instance = operation_cls(**params)

        chain.append(operation_instance)

    return chain


def pipeline_from_config(config: Config, **kwargs) -> Pipeline:
    """
    Create a Pipeline instance from a Config object.
    """
    assert "operations" in config, (
        "Config object must have an 'operations' key for pipeline creation."
    )
    assert isinstance(config.operations, (list, np.ndarray)), (
        "Config object must have a list or numpy array of operations for pipeline creation."
    )

    operations = make_operation_chain(config.operations)

    # merge pipeline config without operations with kwargs
    pipeline_config = config.copy()
    pipeline_config.pop("operations")

    kwargs = {**pipeline_config, **kwargs}
    return Pipeline(operations=operations, **kwargs)


def pipeline_from_json(json_string: str, **kwargs) -> Pipeline:
    """
    Create a Pipeline instance from a JSON string.
    """
    pipeline_config = Config(json.loads(json_string, cls=ZEADecoderJSON))
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

    return json.dumps(pipeline_dict, cls=ZEAEncoderJSON, indent=4)


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


@ops_registry("map")
class Map(Pipeline):
    """
    A pipeline that maps its operations over specified input arguments.

    This can be used to reduce memory usage by processing data in chunks.

    Notes
    -----
    - When `chunks` and `batch_size` are both None (default), this behaves like a normal Pipeline.
    - Changing anything other than ``self.output_key`` in the dict will not be propagated.
    - Will be jitted as a single operation, not the individual operations.
    - This class handles the batching.

    For more information on how to use ``in_axes``, ``out_axes``, `see the documentation for
    jax.vmap <https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html>`_.

    Example
    -------
        .. doctest::

            >>> from zea.ops import Map, Pipeline, Demodulate, TOFCorrection

            >>> # apply operations in batches of 8
            >>> # in this case, over the first axis of "data"
            >>> # or more specifically, process 8 transmits at a time

            >>> pipeline_mapped = Map(
            ...     [
            ...         Demodulate(),
            ...         TOFCorrection(),
            ...     ],
            ...     argnames="data",
            ...     batch_size=8,
            ... )

            >>> # you can also map a subset of the operations
            >>> # for example, demodulate in 4 chunks
            >>> # or more specifically, split the transmit axis into 4 parts

            >>> pipeline_mapped = Pipeline(
            ...     [
            ...         Map([Demodulate()], argnames="data", chunks=4),
            ...         TOFCorrection(),
            ...     ],
            ... )
    """

    def __init__(
        self,
        operations: List[Operation],
        argnames: List[str] | str,
        in_axes: List[Union[int, None]] | int = 0,
        out_axes: List[Union[int, None]] | int = 0,
        chunks: int | None = None,
        batch_size: int | None = None,
        **kwargs,
    ):
        """
        Args:
            operations (list): List of operations to be performed.
            argnames (str or list): List of argument names (or keys) to map over.
                Can also be a single string if only one argument is mapped over.
            in_axes (int or list): Axes to map over for each argument.
                If a single int is provided, it is used for all arguments.
            out_axes (int or list): Axes to map over for each output.
                If a single int is provided, it is used for all outputs.
            chunks (int, optional): Number of chunks to split the input data into.
                If None, no chunking is performed. Mutually exclusive with ``batch_size``.
            batch_size (int, optional): Size of batches to process at once.
                If None, no batching is performed. Mutually exclusive with ``chunks``.
        """
        super().__init__(operations, **kwargs)

        if batch_size is not None and chunks is not None:
            raise ValueError(
                "batch_size and chunks are mutually exclusive. Please specify only one."
            )

        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        if chunks is not None and chunks <= 0:
            raise ValueError("chunks must be a positive integer.")

        if isinstance(argnames, str):
            argnames = [argnames]

        self.argnames = argnames
        self.in_axes = in_axes
        self.out_axes = out_axes
        self.chunks = chunks
        self.batch_size = batch_size

        if chunks is None and batch_size is None:
            log.warning(
                "[zea.ops.Map] Both `chunks` and `batch_size` are None. "
                "This will behave like a normal Pipeline. "
                "Consider setting one of them to process data in chunks or batches."
            )

        def call_item(**inputs):
            """Process data in patches."""
            mapped_args = []
            for argname in argnames:
                mapped_args.append(inputs.pop(argname, None))

            def patched_call(*args):
                mapped_kwargs = [(k, v) for k, v in zip(argnames, args)]
                out = super(Map, self).call(**dict(mapped_kwargs), **inputs)

                # TODO: maybe it is possible to output everything?
                # e.g. prepend a empty dimension to all inputs and just map over everything?
                return out[self.output_key]

            out = vmap(
                patched_call,
                in_axes=in_axes,
                out_axes=out_axes,
                chunks=chunks,
                batch_size=batch_size,
                fn_supports_batch=True,
                disable_jit=not bool(self.jit_options),
            )(*mapped_args)

            return out

        self.call_item = call_item

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
        return self._with_batch_dim

    @with_batch_dim.setter
    def with_batch_dim(self, value):
        """Set the with_batch_dim property of the pipeline.
        The class handles the batching so the operations have to be set to False."""
        self._with_batch_dim = value
        for operation in self.operations:
            operation.with_batch_dim = False

    @property
    def valid_keys(self) -> set:
        """Get a set of valid keys for the pipeline.
        Adds the parameters that PatchedGrid itself operates on (even if not used by operations
        inside it)."""
        return super().valid_keys.union(self.argnames)

    @property
    def needs_keys(self) -> set:
        """Get a set of all input keys needed by the pipeline.
        Adds the parameters that PatchedGrid itself operates on (even if not used by operations
        inside it)."""
        return super().needs_keys.union(self.argnames)

    def jittable_call(self, **inputs):
        """Process input data through the pipeline."""
        if self._with_batch_dim:
            input_data = inputs.pop(self.key)
            output = ops.map(
                lambda x: self.call_item(**{self.key: x, **inputs}),
                input_data,
            )
        else:
            output = self.call_item(**inputs)

        return {self.output_key: output}

    def call(self, **inputs):
        """Process input data through the pipeline."""
        output = self._jittable_call(**inputs)
        inputs.update(output)
        return inputs

    def get_dict(self):
        """Get the configuration of the pipeline."""
        config = super().get_dict()
        config["params"].update(
            {
                "argnames": self.argnames,
                "in_axes": self.in_axes,
                "out_axes": self.out_axes,
                "chunks": self.chunks,
                "batch_size": self.batch_size,
            }
        )
        return config


@ops_registry("patched_grid")
class PatchedGrid(Map):
    """
    A pipeline that maps its operations over `flatgrid` and `flat_pfield` keys.

    This can be used to reduce memory usage by processing data in chunks.

    For more information and flexibility, see :class:`zea.ops.Map`.
    """

    def __init__(self, *args, num_patches=10, **kwargs):
        super().__init__(*args, argnames=["flatgrid", "flat_pfield"], chunks=num_patches, **kwargs)
        self.num_patches = num_patches

    def get_dict(self):
        """Get the configuration of the pipeline."""
        config = super().get_dict()
        config["params"].pop("argnames")
        config["params"].pop("chunks")
        config["params"].update({"num_patches": self.num_patches})
        return config


@ops_registry("reshape_grid")
class ReshapeGrid(Operation):
    """Reshape flat grid data to grid shape."""

    def __init__(self, axis=0, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, grid, **kwargs):
        """
        Args:
            - data (Tensor): The flat grid data of shape (..., n_pix, ...).
        Returns:
            - reshaped_data (Tensor): The reshaped data of shape (..., grid.shape, ...).
        """
        data = kwargs[self.key]
        reshaped_data = reshape_axis(data, grid.shape[:-1], self.axis + int(self.with_batch_dim))
        return {self.output_key: reshaped_data}


## Base Operations


@ops_registry("identity")
class Identity(Operation):
    """Identity operation."""

    def call(self, **kwargs) -> Dict:
        """Returns the input as is."""
        return {}


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

    # Define operation-specific static parameters
    STATIC_PARAMS = ["n_ax", "apply_lens_correction"]

    def __init__(self, **kwargs):
        super().__init__(
            output_data_type=DataTypes.RAW_DATA,
            additional_output_keys=["n_ch"],
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
            "n_ch": 1,  # Simulate always returns RF data (so single channel)
        }


@ops_registry("tof_correction")
class TOFCorrection(Operation):
    """Time-of-flight correction operation for ultrasound data."""

    # Define operation-specific static parameters
    STATIC_PARAMS = [
        "f_number",
        "apply_lens_correction",
        "grid_size_x",
        "grid_size_z",
    ]

    def __init__(self, **kwargs):
        super().__init__(
            input_data_type=DataTypes.RAW_DATA,
            output_data_type=DataTypes.ALIGNED_DATA,
            **kwargs,
        )

    def call(
        self,
        flatgrid,
        sound_speed,
        polar_angles,
        focus_distances,
        sampling_frequency,
        f_number,
        demodulation_frequency,
        t0_delays,
        tx_apodizations,
        initial_times,
        probe_geometry,
        t_peak,
        tx_waveform_indices,
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
            t_peak (float): Time to peak of the transmit pulse
            tx_waveform_indices (ops.Tensor): Index of the transmit waveform for each
                transmit. (All zero if there is only one waveform)
            apply_lens_correction (bool): Whether to apply lens correction
            lens_thickness (float): Lens thickness
            lens_sound_speed (float): Sound speed in the lens

        Returns:
            dict: Dictionary containing tof_corrected_data
        """

        raw_data = kwargs[self.key]

        tof_kwargs = {
            "flatgrid": flatgrid,
            "t0_delays": t0_delays,
            "tx_apodizations": tx_apodizations,
            "sound_speed": sound_speed,
            "probe_geometry": probe_geometry,
            "initial_times": initial_times,
            "sampling_frequency": sampling_frequency,
            "demodulation_frequency": demodulation_frequency,
            "f_number": f_number,
            "polar_angles": polar_angles,
            "focus_distances": focus_distances,
            "t_peak": t_peak,
            "tx_waveform_indices": tx_waveform_indices,
            "apply_lens_correction": apply_lens_correction,
            "lens_thickness": lens_thickness,
            "lens_sound_speed": lens_sound_speed,
        }

        if not self.with_batch_dim:
            tof_corrected = tof_correction(raw_data, **tof_kwargs)
        else:
            tof_corrected = ops.map(
                lambda data: tof_correction(data, **tof_kwargs),
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
        data = kwargs[self.key]  # must start with ((batch_size,) n_tx, n_pix, ...)

        if flat_pfield is None:
            return {self.output_key: data}

        # Swap (n_pix, n_tx) to (n_tx, n_pix)
        flat_pfield = ops.swapaxes(flat_pfield, 0, 1)

        # Add batch dimension if needed
        if self.with_batch_dim:
            pfield_expanded = ops.expand_dims(flat_pfield, axis=0)
        else:
            pfield_expanded = flat_pfield

        append_n_dims = ops.ndim(data) - ops.ndim(pfield_expanded)
        pfield_expanded = extend_n_dims(pfield_expanded, axis=-1, n_dims=append_n_dims)

        # Perform element-wise multiplication with the pressure weight mask
        weighted_data = data * pfield_expanded

        return {self.output_key: weighted_data}


@ops_registry("delay_and_sum")
class DelayAndSum(Operation):
    """Sums time-delayed signals along channels and transmits."""

    def __init__(self, **kwargs):
        super().__init__(
            input_data_type=DataTypes.ALIGNED_DATA,
            output_data_type=DataTypes.BEAMFORMED_DATA,
            **kwargs,
        )

    def process_image(self, data):
        """Performs DAS beamforming on tof-corrected input.

        Args:
            data (ops.Tensor): The TOF corrected input of shape `(n_tx, n_pix, n_el, n_ch)`

        Returns:
            ops.Tensor: The beamformed data of shape `(n_pix, n_ch)`
        """
        # Sum over the channels, i.e. DAS
        data = ops.sum(data, -2)

        # Sum over transmits, i.e. Compounding
        data = ops.sum(data, 0)

        return data

    def call(self, grid=None, **kwargs):
        """Performs DAS beamforming on tof-corrected input.

        Args:
            tof_corrected_data (ops.Tensor): The TOF corrected input of shape
                `(n_tx, grid_size_z*grid_size_x, n_el, n_ch)` with optional batch dimension.

        Returns:
            dict: Dictionary containing beamformed_data
                of shape `(grid_size_z*grid_size_x, n_ch)`
                with optional batch dimension.
        """
        data = kwargs[self.key]

        if not self.with_batch_dim:
            beamformed_data = self.process_image(data)
        else:
            # Apply process_image to each item in the batch
            beamformed_data = ops.map(self.process_image, data)

        return {self.output_key: beamformed_data}


def envelope_detect(data, axis=-3):
    """Envelope detection of RF signals.

    If the input data is real, it first applies the Hilbert transform along the specified axis
    and then computes the magnitude of the resulting complex signal.
    If the input data is complex, it computes the magnitude directly.

    Args:
        - data (Tensor): The beamformed data of shape (..., grid_size_z, grid_size_x, n_ch).
        - axis (int): Axis along which to apply the Hilbert transform. Defaults to -3.

    Returns:
        - envelope_data (Tensor): The envelope detected data
            of shape (..., grid_size_z, grid_size_x).
    """
    if data.shape[-1] == 2:
        data = channels_to_complex(data)
    else:
        n_ax = ops.shape(data)[axis]
        n_ax_float = ops.cast(n_ax, "float32")

        # Calculate next power of 2: M = 2^ceil(log2(n_ax))
        # see https://github.com/tue-bmd/zea/discussions/147
        log2_n_ax = ops.log2(n_ax_float)
        M = ops.cast(2 ** ops.ceil(log2_n_ax), "int32")

        data = hilbert(data, N=M, axis=axis)
        indices = ops.arange(n_ax)

        data = ops.take(data, indices, axis=axis)
        data = ops.squeeze(data, axis=-1)

    # data = ops.abs(data)
    real = ops.real(data)
    imag = ops.imag(data)
    data = ops.sqrt(real**2 + imag**2)
    data = ops.cast(data, "float32")
    return data


@ops_registry("multiply_and_sum")
class MultiplyAndSum(Operation):
    """Performs the operations for the Delay-Multiply-and-Sum beamformer except the delay.
    The delay should be performed by the TOF correction operation.
    """

    def __init__(self, **kwargs):
        super().__init__(
            input_data_type=DataTypes.ALIGNED_DATA,
            output_data_type=DataTypes.BEAMFORMED_DATA,
            **kwargs,
        )

    def process_image(self, data):
        """Performs DMAS beamforming on tof-corrected input.

        Args:
            data (ops.Tensor): The TOF corrected input of shape `(n_tx, n_pix, n_el, n_ch)`

        Returns:
            ops.Tensor: The beamformed data of shape `(n_pix, n_ch)`
        """

        if not data.shape[-1] == 2:
            raise ValueError(
                "MultiplyAndSum operation requires IQ data with 2 channels. "
                f"Got data with shape {data.shape}."
            )

        # Compute the correlation matrix
        data = channels_to_complex(data)

        data = self._multiply(data)
        data = self._select_lower_triangle(data)
        data = ops.sum(data, axis=(0, 2, 3))

        data = complex_to_channels(data)

        return data

    def _select_lower_triangle(self, data):
        """Select only the lower triangle of the correlation matrix."""
        n_el = data.shape[3]
        mask = ops.ones((n_el, n_el), dtype=data.dtype) - ops.eye(n_el, dtype=data.dtype)
        data = data * mask[None, None, :, :] / 2
        return data

    def _multiply(self, data):
        """Apply the DMAS multiplication step."""
        channel_products = data[:, :, :, None] * data[:, :, None, :]

        data = ops.sign(channel_products) * ops.cast(
            ops.sqrt(ops.abs(channel_products)), data.dtype
        )
        return data

    def call(self, grid=None, **kwargs):
        """Performs DMAS beamforming on tof-corrected input.

        Args:
            tof_corrected_data (ops.Tensor): The TOF corrected input of shape
                `(n_tx, grid_size_z*grid_size_x, n_el, n_ch)` with optional batch dimension.

        Returns:
            dict: Dictionary containing beamformed_data
                of shape `(grid_size_z*grid_size_x, n_ch)`
                with optional batch dimension.
        """
        data = kwargs[self.key]

        if not self.with_batch_dim:
            beamformed_data = self.process_image(data)
        else:
            # Apply process_image to each item in the batch
            beamformed_data = ops.map(self.process_image, data)

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
            - data (Tensor): The beamformed data of shape (..., grid_size_z, grid_size_x, n_ch).
        Returns:
            - envelope_data (Tensor): The envelope detected data
                of shape (..., grid_size_z, grid_size_x).
        """
        data = kwargs[self.key]

        data = envelope_detect(data, axis=self.axis)

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


def log_compress(data, eps=1e-16):
    """Apply logarithmic compression to data."""
    eps = ops.convert_to_tensor(eps, dtype=data.dtype)
    data = ops.where(data == 0, eps, data)  # Avoid log(0)
    return 20 * keras.ops.log10(data)


@ops_registry("log_compress")
class LogCompress(Operation):
    """Logarithmic compression of data."""

    def __init__(self, clip: bool = True, **kwargs):
        """Initialize the LogCompress operation.

        Args:
            clip (bool): Whether to clip the output to a dynamic range. Defaults to True.
        """
        super().__init__(
            input_data_type=DataTypes.ENVELOPE_DATA,
            output_data_type=DataTypes.IMAGE,
            **kwargs,
        )
        self.clip = clip

    def call(self, dynamic_range=None, **kwargs):
        """Apply logarithmic compression to data.

        Args:
            dynamic_range (tuple, optional): Dynamic range in dB. Defaults to (-60, 0).

        Returns:
            dict: Dictionary containing log-compressed data
        """
        data = kwargs[self.key]

        if dynamic_range is None:
            dynamic_range = ops.array(DEFAULT_DYNAMIC_RANGE)
        dynamic_range = ops.cast(dynamic_range, data.dtype)

        compressed_data = log_compress(data)
        if self.clip:
            compressed_data = ops.clip(compressed_data, dynamic_range[0], dynamic_range[1])

        return {self.output_key: compressed_data}


def normalize(data, output_range, input_range=None):
    """Normalize data to a given range.

    Equivalent to `translate` with clipping.

    Args:
        data (ops.Tensor): Input data to normalize.
        output_range (tuple): Range to which data should be mapped, e.g., (0, 1).
        input_range (tuple, optional): Range of input data.
            If None, the range will be computed from the data.
            Defaults to None.
    """
    if input_range is None:
        input_range = (None, None)
    minval, maxval = input_range
    if minval is None:
        minval = ops.min(data)
    if maxval is None:
        maxval = ops.max(data)
    data = ops.clip(data, minval, maxval)
    normalized_data = translate(data, (minval, maxval), output_range)
    return normalized_data


@ops_registry("normalize")
class Normalize(Operation):
    """Normalize data to a given range."""

    def __init__(self, output_range=None, input_range=None, **kwargs):
        super().__init__(additional_output_keys=["minval", "maxval"], **kwargs)
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
            [np.float32(x) if x is not None else None for x in data] if data is not None else None
        )

    def call(self, **kwargs):
        """Normalize data to a given range.

        Args:
            output_range (tuple, optional): Range to which data should be mapped.
                Defaults to (0, 1).
            input_range (tuple, optional): Range of input data. If None, the range
                of the input data will be computed. Defaults to None.

        Returns:
            dict: Dictionary containing normalized data, along with the computed
                  or provided input range (minval and maxval).
        """
        data = kwargs[self.key]

        # If input_range is not provided, try to get it from kwargs
        # This allows you to normalize based on the first frame in a sequence and avoid flicker
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

        normalized_data = normalize(
            data, output_range=self.output_range, input_range=(minval, maxval)
        )

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
                "GPU support for order > 1 is not available. " + "Disabling jit for ScanConvert."
            )
        else:
            jittable = True

        super().__init__(
            input_data_type=DataTypes.IMAGE,
            output_data_type=DataTypes.IMAGE_SC,
            jittable=jittable,
            additional_output_keys=[
                "resolution",
                "x_lim",
                "y_lim",
                "z_lim",
                "rho_range",
                "theta_range",
                "phi_range",
                "d_rho",
                "d_theta",
                "d_phi",
            ],
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
class GaussianBlur(ImageOperation):
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
        kernel = scipy.ndimage.gaussian_filter(n, sigma=self.sigma, mode="constant").astype(
            np.float32
        )
        kernel = kernel[:, :, None, None]
        return ops.convert_to_tensor(kernel)

    def call(self, **kwargs):
        """Apply a Gaussian filter to the input data.

        Args:
            data (ops.Tensor): Input image data of shape (height, width, channels) with
                optional batch dimension if ``self.with_batch_dim``.
        """
        super().call(**kwargs)
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
class LeeFilter(ImageOperation):
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
            key="data",
        )

    @property
    def with_batch_dim(self):
        """Get the with_batch_dim property of the LeeFilter operation."""
        return self._with_batch_dim

    @with_batch_dim.setter
    def with_batch_dim(self, value):
        """Set the with_batch_dim property of the LeeFilter operation."""
        self._with_batch_dim = value
        if hasattr(self, "gaussian_blur"):
            self.gaussian_blur.with_batch_dim = value

    def call(self, **kwargs):
        """Apply the Lee filter to the input data.

        Args:
            data (ops.Tensor): Input image data of shape (height, width, channels) with
                optional batch dimension if ``self.with_batch_dim``.
        """
        super().call(**kwargs)
        data = kwargs.pop(self.key)

        # Apply Gaussian blur to get local mean
        img_mean = self.gaussian_blur.call(data=data, **kwargs)[self.gaussian_blur.output_key]

        # Apply Gaussian blur to squared data to get local squared mean
        img_sqr_mean = self.gaussian_blur.call(
            data=data**2,
            **kwargs,
        )[self.gaussian_blur.output_key]

        # Calculate local variance
        img_variance = img_sqr_mean - img_mean**2

        # Calculate global variance (per channel)
        overall_variance = ops.var(data, axis=(-3, -2), keepdims=True)

        # Calculate adaptive weights
        img_weights = img_variance / (img_variance + overall_variance)

        # Apply Lee filter formula
        img_output = img_mean + img_weights * (data - img_mean)

        return {self.output_key: img_output}


@ops_registry("demodulate")
class Demodulate(Operation):
    """Demodulates the input data to baseband. After this operation, the carrier frequency
    is removed (0 Hz) and the data is in IQ format stored in two real valued channels."""

    def __init__(self, axis=-3, **kwargs):
        super().__init__(
            input_data_type=DataTypes.RAW_DATA,
            output_data_type=DataTypes.RAW_DATA,
            jittable=True,
            additional_output_keys=[
                "demodulation_frequency",
                "center_frequency",
                "n_ch",
            ],
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
            "center_frequency": 0.0,
            "n_ch": 2,
        }


@ops_registry("fir_filter")
class FirFilter(Operation):
    """Apply a FIR filter to the input signal using convolution.

    Looks for the filter taps in the input dictionary using the specified ``filter_key``.
    """

    def __init__(
        self,
        axis: int,
        complex_channels: bool = False,
        filter_key: str = "fir_filter_taps",
        **kwargs,
    ):
        """
        Args:
            axis (int): Axis along which to apply the filter. Cannot be the batch dimension.
                When using ``complex_channels=True``, the complex channels are removed to convert
                to complex numbers before filtering, so adjust the ``axis`` accordingly!
            complex_channels (bool): Whether the last dimension of the input signal represents
                complex channels (real and imaginary parts). When True, it will convert the signal
                to ``complex`` dtype before filtering and convert it back to two channels
                after filtering.
            filter_key (str): Key in the input dictionary where the FIR filter taps are stored.
                Default is "fir_filter_taps".
        """
        super().__init__(**kwargs)
        self._check_axis(axis)

        self.axis = axis
        self.complex_channels = complex_channels
        self.filter_key = filter_key

    def _check_axis(self, axis, ndim=None):
        """Check if the axis is valid."""
        if ndim is not None:
            if axis < -ndim or axis >= ndim:
                raise ValueError(f"Axis {axis} is out of bounds for array of dimension {ndim}.")

        if self.with_batch_dim and (axis == 0 or (ndim is not None and axis == -ndim)):
            raise ValueError("Cannot apply FIR filter along batch dimension.")

    @property
    def valid_keys(self):
        """Get the valid keys for the `call` method."""
        return self._valid_keys.union({self.filter_key})

    def call(self, **kwargs):
        signal = kwargs[self.key]
        fir_filter_taps = kwargs[self.filter_key]

        if self.complex_channels:
            signal = channels_to_complex(signal)

        self._check_axis(self.axis, ndim=ops.ndim(signal))

        def _convolve(signal):
            """Apply the filter to the signal using correlation."""
            return correlate(signal, fir_filter_taps[::-1], mode="same")

        filtered_signal = apply_along_axis(_convolve, self.axis, signal)

        if self.complex_channels:
            filtered_signal = complex_to_channels(filtered_signal)

        return {self.output_key: filtered_signal}


@ops_registry("low_pass_filter")
class LowPassFilter(FirFilter):
    """Apply a low-pass FIR filter to the input signal using convolution.

    It is recommended to use :class:`FirFilter` with pre-computed filter taps for jittable
    operations. The :class:`LowPassFilter` operation itself is not jittable and is provided
    for convenience only.

    Uses :func:`get_low_pass_iq_filter` to compute the filter taps.
    """

    def __init__(self, axis: int, complex_channels: bool = False, num_taps: int = 128, **kwargs):
        """Initialize the LowPassFilter operation.

        Args:
            axis (int): Axis along which to apply the filter. Cannot be the batch dimension.
                When using ``complex_channels=True``, the complex channels are removed to convert
                to complex numbers before filtering, so adjust the ``axis`` accordingly.
            complex_channels (bool): Whether the last dimension of the input signal represents
                complex channels (real and imaginary parts). When True, it will convert the signal
                to ``complex`` dtype before filtering and convert it back to two channels
                after filtering.
            num_taps (int): Number of taps in the FIR filter. Default is 128.
        """
        self._random_suffix = str(uuid.uuid4())
        kwargs.pop("filter_key", None)
        kwargs.pop("jittable", None)
        super().__init__(
            axis=axis,
            complex_channels=complex_channels,
            filter_key=f"low_pass_{self._random_suffix}",
            jittable=False,
            **kwargs,
        )
        self.num_taps = num_taps

    def call(self, bandwidth, sampling_frequency, center_frequency, **kwargs):
        lpf = get_low_pass_iq_filter(
            self.num_taps,
            ops.convert_to_numpy(sampling_frequency).item(),
            ops.convert_to_numpy(center_frequency).item(),
            ops.convert_to_numpy(bandwidth).item(),
        )
        kwargs[self.filter_key] = lpf
        return super().call(**kwargs)


@ops_registry("lambda")
class Lambda(Operation):
    """Use any function as an operation."""

    def __init__(self, func, **kwargs):
        # Split kwargs into kwargs for partial and __init__
        op_kwargs = {k: v for k, v in kwargs.items() if k not in func.__code__.co_varnames}
        func_kwargs = {k: v for k, v in kwargs.items() if k in func.__code__.co_varnames}
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


@ops_registry("pad")
class Pad(Operation, DataLayer):
    """Pad layer for padding tensors to a specified shape."""

    def __init__(
        self,
        target_shape: list | tuple,
        uniform: bool = True,
        axis: Union[int, List[int]] = None,
        fail_on_bigger_shape: bool = True,
        pad_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_shape = target_shape
        self.uniform = uniform
        self.axis = axis
        self.pad_kwargs = pad_kwargs or {}
        self.fail_on_bigger_shape = fail_on_bigger_shape

    @staticmethod
    def _format_target_shape(shape_array, target_shape, axis):
        if isinstance(axis, int):
            axis = [axis]
        assert len(axis) == len(target_shape), (
            "The length of axis must be equal to the length of target_shape."
        )
        axis = map_negative_indices(axis, len(shape_array))

        target_shape = [
            target_shape[axis.index(i)] if i in axis else shape_array[i]
            for i in range(len(shape_array))
        ]
        return target_shape

    def pad(
        self,
        z,
        target_shape: list | tuple,
        uniform: bool = True,
        axis: Union[int, List[int]] = None,
        fail_on_bigger_shape: bool = True,
        **kwargs,
    ):
        """
        Pads the input tensor `z` to the specified shape.

        Parameters:
            z (tensor): The input tensor to be padded.
            target_shape (list or tuple): The target shape to pad the tensor to.
            uniform (bool, optional): If True, ensures that padding is uniform (even on both sides).
                Default is False.
            axis (int or list of int, optional): The axis or axes along which `target_shape` was
                specified. If None, `len(target_shape) == `len(ops.shape(z))` must hold.
                Default is None.
            fail_on_bigger_shape (bool, optional): If True (default), raises an error if any target
                dimension is smaller than the input shape; if False, pads only where the
                target shape exceeds the input shape and leaves other dimensions unchanged.
            kwargs: Additional keyword arguments to pass to the padding function.

        Returns:
            tensor: The padded tensor with the specified shape.
        """
        shape_array = self.backend.shape(z)

        # When axis is provided, convert target_shape
        if axis is not None:
            target_shape = self._format_target_shape(shape_array, target_shape, axis)

        if not fail_on_bigger_shape:
            target_shape = [max(target_shape[i], shape_array[i]) for i in range(len(shape_array))]

        # Compute the padding required for each dimension
        pad_shape = np.array(target_shape) - shape_array

        # Create the paddings array
        if uniform:
            # if odd, pad more on the left, same as:
            # https://keras.io/api/layers/preprocessing_layers/image_preprocessing/center_crop/
            right_pad = pad_shape // 2
            left_pad = pad_shape - right_pad
            paddings = np.stack([right_pad, left_pad], axis=1)
        else:
            paddings = np.stack([np.zeros_like(pad_shape), pad_shape], axis=1)

        if np.any(paddings < 0):
            raise ValueError(
                f"Target shape {target_shape} must be greater than or equal "
                f"to the input shape {shape_array}."
            )

        return self.backend.numpy.pad(z, paddings, **kwargs)

    def call(self, **kwargs):
        data = kwargs[self.key]
        padded_data = self.pad(
            data,
            self.target_shape,
            self.uniform,
            self.axis,
            self.fail_on_bigger_shape,
            **self.pad_kwargs,
        )
        return {self.output_key: padded_data}


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
            self._compand_func = self._mu_law_expand if self.expand else self._mu_law_compress
        else:
            self._compand_func = self._a_law_expand if self.expand else self._a_law_compress

    @staticmethod
    def _mu_law_compress(x, mu=255, **kwargs):
        x = ops.clip(x, -1, 1)
        return ops.sign(x) * ops.log(1.0 + mu * ops.abs(x)) / ops.log(1.0 + mu)

    @staticmethod
    def _mu_law_expand(y, mu=255, **kwargs):
        y = ops.clip(y, -1, 1)
        return ops.sign(y) * ((1.0 + mu) ** ops.abs(y) - 1.0) / mu

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
            additional_output_keys=["sampling_frequency", "n_ax"],
            **kwargs,
        )
        self.factor = factor
        self.phase = phase
        self.axis = axis

    def call(self, sampling_frequency=None, n_ax=None, **kwargs):
        data = kwargs[self.key]
        length = ops.shape(data)[self.axis]
        sample_idx = ops.arange(self.phase, length, self.factor)
        data_downsampled = ops.take(data, sample_idx, axis=self.axis)

        output = {self.output_key: data_downsampled}
        # downsampling also affects the sampling frequency
        if sampling_frequency is not None:
            sampling_frequency = sampling_frequency / self.factor
            output["sampling_frequency"] = sampling_frequency
        if n_ax is not None:
            n_ax = n_ax // self.factor
            output["n_ax"] = n_ax
        return output


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
                    lambda data, threshold, fill: ops.maximum(data - threshold, 0) + fill
                )
            else:
                self._threshold_func = (
                    lambda data, threshold, fill: ops.minimum(data - threshold, 0) + fill
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
            raise ValueError("Pass either threshold or percentile, not both or neither.")

        if percentile is not None:
            # Convert percentile to quantile value (0-1 range)
            threshold = ops.quantile(data, percentile / 100.0)

        fill_value = self._resolve_fill_value(data, threshold)
        result = self._threshold_func(data, threshold, fill_value)
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
            image_out = self._anisotropic_diffusion_single(image, niter, lmbda, rect, eps)
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
            iN = ops.concatenate([image[1:], ops.zeros((1, N), dtype=image.dtype)], axis=0)
            iS = ops.concatenate([ops.zeros((1, N), dtype=image.dtype), image[:-1]], axis=0)
            jW = ops.concatenate([image[:, 1:], ops.zeros((M, 1), dtype=image.dtype)], axis=1)
            jE = ops.concatenate([ops.zeros((M, 1), dtype=image.dtype), image[:, :-1]], axis=1)

            if rect is not None:
                x1, y1, x2, y2 = rect
                imageuniform = image[x1:x2, y1:y2]
                q0_squared = (ops.std(imageuniform) / (ops.mean(imageuniform) + eps)) ** 2

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
            cE = ops.concatenate([ops.zeros((M, 1), dtype=image.dtype), c[:, :-1]], axis=1)

            D = (cS * dS) + (c * dN) + (cE * dE) + (c * dW)
            image = image + (lmbda / 4) * D

        result = ops.log(image)
        return result


@ops_registry("channels_to_complex")
class ChannelsToComplex(Operation):
    def call(self, **kwargs):
        data = kwargs[self.key]
        output = channels_to_complex(data)
        return {self.output_key: output}


@ops_registry("complex_to_channels")
class ComplexToChannels(Operation):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, **kwargs):
        data = kwargs[self.key]
        output = complex_to_channels(data, axis=self.axis)
        return {self.output_key: output}


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
    assert np.isreal(rf_data).all(), f"RF must contain real RF signals, got {rf_data.dtype}"

    input_shape = rf_data.shape
    n_dim = len(input_shape)
    if n_dim > 2:
        *_, n_ax, n_el = input_shape
    else:
        n_ax, n_el = input_shape

    if filter_coeff is None:
        assert sampling_frequency is not None, "provide sampling_frequency when no filter is given."
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
            assert np.isscalar(bandwidth), "The signal bandwidth (in %) must be a scalar."
            assert (bandwidth > 0) & (bandwidth <= 200), (
                "The signal bandwidth (in %) must be within the interval of ]0,200]."
            )
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
    bpf = scipy.signal.firwin(num_taps, [f1, f2], pass_zero=False, fs=sampling_frequency)
    return bpf


def get_low_pass_iq_filter(num_taps, sampling_frequency, center_frequency, bandwidth):
    """Design complex low-pass filter.

    The filter is a low-pass FIR filter modulated to the center frequency.

    Args:
        num_taps (int): number of taps in filter.
        sampling_frequency (float): sample frequency.
        center_frequency (float): center frequency.
        bandwidth (float): bandwidth in Hz.

    Raises:
        ValueError: if cutoff frequency (bandwidth / 2) is not within (0, sampling_frequency / 2)

    Returns:
        ndarray: Complex-valued low-pass filter
    """
    cutoff = bandwidth / 2
    if not (0 < cutoff < sampling_frequency / 2):
        raise ValueError(
            f"Cutoff frequency must be within (0, sampling_frequency / 2), "
            f"got {cutoff} Hz, must be within (0, {sampling_frequency / 2}) Hz"
        )
    # Design real-valued low-pass filter
    lpf = scipy.signal.firwin(num_taps, cutoff, pass_zero=True, fs=sampling_frequency)
    # Modulate to center frequency to make it complex
    time_points = np.arange(num_taps) / sampling_frequency
    lpf_complex = lpf * np.exp(1j * 2 * np.pi * center_frequency * time_points)
    return lpf_complex


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
    """Manual implementation of the Hilbert transform function. The function
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
    frequency_indices_shaped_like_rf = ops.cast(frequency_indices_shaped_like_rf, dtype="complex64")

    # Shift to baseband
    phasor_exponent = (
        -1j * 2 * np.pi * center_frequency * frequency_indices_shaped_like_rf / sampling_frequency
    )
    iq_data_signal_complex = analytical_signal * ops.exp(phasor_exponent)

    # Split the complex signal into two channels
    iq_data_two_channel = complex_to_channels(ops.squeeze(iq_data_signal_complex, axis=-1))

    return iq_data_two_channel


def compute_time_to_peak_stack(waveforms, center_frequencies, waveform_sampling_frequency=250e6):
    """Compute the time of the peak of each waveform in a stack of waveforms.

    Args:
        waveforms (ndarray): The waveforms of shape (n_waveforms, n_samples).
        center_frequencies (ndarray): The center frequencies of the waveforms in Hz of shape
            (n_waveforms,) or a scalar if all waveforms have the same center frequency.
        waveform_sampling_frequency (float): The sampling frequency of the waveforms in Hz.

    Returns:
        ndarray: The time to peak for each waveform in seconds.
    """
    t_peak = []
    center_frequencies = center_frequencies * ops.ones((waveforms.shape[0],))
    for waveform, center_frequency in zip(waveforms, center_frequencies):
        t_peak.append(compute_time_to_peak(waveform, center_frequency, waveform_sampling_frequency))
    return ops.stack(t_peak)


def compute_time_to_peak(waveform, center_frequency, waveform_sampling_frequency=250e6):
    """Compute the time of the peak of the waveform.

    Args:
        waveform (ndarray): The waveform of shape (n_samples).
        center_frequency (float): The center frequency of the waveform in Hz.
        waveform_sampling_frequency (float): The sampling frequency of the waveform in Hz.

    Returns:
        float: The time to peak for the waveform in seconds.
    """
    n_samples = waveform.shape[0]
    if n_samples == 0:
        raise ValueError("Waveform has zero samples.")

    waveforms_iq_complex_channels = demodulate(
        waveform[..., None], center_frequency, waveform_sampling_frequency, axis=-1
    )
    waveforms_iq_complex = channels_to_complex(waveforms_iq_complex_channels)
    envelope = ops.abs(waveforms_iq_complex)
    peak_idx = ops.argmax(envelope, axis=-1)
    t_peak = ops.cast(peak_idx, dtype="float32") / waveform_sampling_frequency
    return t_peak
