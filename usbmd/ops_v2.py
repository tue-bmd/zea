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
from usbmd.display import scan_convert_2d, scan_convert_3d
from usbmd.ops import channels_to_complex, hilbert, upmix
from usbmd.probes import Probe
from usbmd.registry import ops_v2_registry as ops_registry
from usbmd.scan import Scan
from usbmd.simulator import simulate_rf
from usbmd.tensor_ops import patched_map, reshape_axis, take
from usbmd.utils import log, translate
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
        key: Union[str, None] = None,
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


class Pipeline:
    """Pipeline class for processing ultrasound data through a series of operations."""

    def __init__(
        self,
        operations: List[Operation],
        with_batch_dim: bool = True,
        jit_options: Union[str, None] = "ops",
        jit_kwargs: dict | None = None,
        name="pipeline",
    ):
        """Initialize a pipeline

        Args:
            operations (list): A list of Operation instances representing the operations
                to be performed.
            with_batch_dim (bool, optional): Whether operations should expect a batch dimension.
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
        self._call_pipeline = self.call
        self.name = name

        # add functionality here
        # operations = ...

        self._pipeline_layers = operations

        if jit_options not in ["pipeline", "ops", None]:
            raise ValueError("jit_options must be 'pipeline', 'ops', or None")

        self.with_batch_dim = with_batch_dim

        self.validate()

        # pylint: disable=method-hidden
        if jit_kwargs is None:
            if keras.backend.backend() == "jax":
                jit_kwargs = {"static_argnames": STATIC}
            else:
                jit_kwargs = {}
        self.jit_kwargs = jit_kwargs
        self.jit_options = jit_options  # will handle the jit compilation

    @classmethod
    def default(cls, num_patches=20, **kwargs) -> "Pipeline":
        """Create a default pipeline."""
        # Get beamforming ops
        beamforming = [
            TOFCorrection(),
            PfieldWeighting(),
            DelayAndSum(),
        ]

        # Optionally add patching
        if num_patches > 1:
            beamforming = PatchedGrid(operations=beamforming, num_patches=num_patches)
            operations = [beamforming]
        else:
            operations = beamforming

        # Add display ops
        operations += [
            EnvelopeDetect(),
            LogCompress(output_key="image"),
            Normalize(key="image", output_key="image"),
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

    # TODO: a lot of time is spent on converting scan etc... to_tensor every call
    # If you want to call pipeline on every frame in a for loop (e.g. real time applications where
    # data is coming in on the fly), this will make it twice as slow
    def __call__(self, *args, return_numpy=False, **kwargs):
        """Process input data through the pipeline."""

        if any(key in kwargs for key in ["probe", "scan", "config"]):
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

        # Dropping str inputs as they are not supported in jax.jit
        # TODO: will this break any operations?
        inputs.pop("probe_type", None)

        ## PROCESSING
        outputs = self._call_pipeline(**inputs)

        ## PREPARE OUTPUT
        if return_numpy:
            # Convert tensors to numpy arrays but preserve None values
            outputs = {
                k: ops.convert_to_numpy(v) if v is ops.is_tensor(v) else v
                for k, v in outputs.items()
            }

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

    @classmethod
    def from_config(cls, config: Dict, **kwargs) -> "Pipeline":
        """Create a pipeline from a dictionary or `usbmd.Config` object.

        Must have an 'operations' key with a list of operations.

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

    @property
    def key(self) -> str:
        """Input key of the pipeline."""
        return self.operations[0].key

    @property
    def output_key(self) -> str:
        """Output key of the pipeline."""
        return self.operations[-1].output_key


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

            # this allows for nested operations
            if operation["params"].get("operations") is not None:
                operation["params"]["operations"] = make_operation_chain(
                    operation["params"]["operations"]
                )

            operation = get_ops(operation["name"])(**operation["params"])
        chain.append(operation)

    return chain


def pipeline_from_json(json_string: str, **kwargs) -> Pipeline:
    """
    Create a Pipeline instance from a JSON string.
    """
    pipeline_config = Config(json.loads(json_string))
    return pipeline_from_config(pipeline_config, **kwargs)


def pipeline_from_yaml(yaml_path: str, **kwargs) -> Pipeline:
    """
    Create a Pipeline instance from a YAML file.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        pipeline_config = yaml.safe_load(f)
    operations = pipeline_config["operations"]
    return pipeline_from_config(Config({"operations": operations}), **kwargs)


def pipeline_from_config(config: Config, **kwargs) -> Pipeline:
    """
    Create a Pipeline instance from a Config object.
    """
    assert (
        "operations" in config
    ), "Config object must have an 'operations' key for pipeline creation."
    assert isinstance(
        config.operations, list
    ), "Config object must have a list of operations for pipeline creation."

    operations = make_operation_chain(config.operations)

    # merge pipeline config without operations with kwargs
    pipeline_config = copy.deepcopy(config)
    pipeline_config.pop("operations")

    kwargs = {**pipeline_config, **kwargs}
    return Pipeline(operations=operations, **kwargs)


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


@ops_registry("upmix")
class UpMix(Operation):
    """Upmix IQ data to RF data."""

    def __init__(self, key: str, **kwargs):
        super().__init__(key=key, **kwargs)

    def call(self, fs=None, fc=None, upsampling_rate=6, **kwargs):

        data = kwargs[self.key]

        if data.shape[-1] == 1:
            log.warning("Upmixing is not applicable to RF data.")
            return data
        elif data.shape[-1] == 2:
            data = channels_to_complex(data)

        data = upmix(data, fs, fc, upsampling_rate)
        data = ops.expand_dims(data, axis=-1)
        return {self.output_key: data}


@ops_registry("simulate_rf")
class Simulate(Operation):
    """Simulate RF data."""

    def __init__(self, **kwargs):
        super().__init__(
            output_data_type=DataTypes.RAW_DATA,
            jit_compile=False,
            **kwargs,
        )

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
        apply_lens_correction,
        n_ax,
    ):
        return {
            "raw_data": simulate_rf(
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

    def __init__(self, key="raw_data", output_key="aligned_data", **kwargs):
        super().__init__(
            key=key,
            output_key=output_key,
            input_data_type=DataTypes.RAW_DATA,
            output_data_type=DataTypes.ALIGNED_DATA,
            **kwargs,
        )

    def call(
        self,
        flatgrid=None,
        sound_speed=None,
        polar_angles=None,
        focus_distances=None,
        sampling_frequency=None,
        f_number=None,
        fdemod=None,
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
            fdemod (float): Demodulation frequency
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
            "fdemod": fdemod,
            "apply_phase_rotation": fdemod,
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

    def __init__(self, key="aligned_data", output_key="aligned_data", **kwargs):
        """Initialize the PfieldWeighting operation.

        Args:
            key (str, optional): Key for input data. Defaults to "aligned_data".
            output_key (str, optional): Key for output data. Defaults to "aligned_data".
        """
        super().__init__(key=key, output_key=output_key, **kwargs)

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
            return out[self.output_data_type.value]

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
            input_data = inputs.pop(self.input_data_type.value)
            output = ops.map(
                lambda x: self.call_item({self.input_data_type.value: x, **inputs}),
                input_data,
            )
        else:
            output = self.call_item(inputs)

        return {self.output_data_type.value: output}

    def call(self, **inputs):
        """Process input data through the pipeline."""
        output = self._jittable_call(**inputs)
        inputs.update(output)
        return inputs


@ops_registry("delay_and_sum")
class DelayAndSum(Operation):
    """Sums time-delayed signals along channels and transmits."""

    def __init__(
        self,
        key="aligned_data",
        output_key="beamformed_data",
        reshape_grid=True,
        **kwargs,
    ):
        super().__init__(
            key=key,
            output_key=output_key,
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
        key: str = "beamformed_data",
        output_key: str = "envelope_data",
        axis=-2,
        **kwargs,
    ):
        super().__init__(
            key=key,
            output_key=output_key,
            input_data_type=DataTypes.BEAMFORMED_DATA,
            output_data_type=DataTypes.ENVELOPE_DATA,
            **kwargs,
        )
        self.axis = axis

    def call(self, **kwargs):
        data = kwargs[self.key]

        if data.shape[-1] == 2:
            data = channels_to_complex(data)
        else:
            n_ax = data.shape[self.axis]
            M = 2 ** int(np.ceil(np.log2(n_ax)))
            # data = scipy.signal.hilbert(data, N=M, axis=self.axis)
            data = hilbert(data, N=M, axis=self.axis)
            indices = ops.arange(n_ax)

            data = take(data, indices, axis=self.axis)
            data = ops.squeeze(data, axis=-1)

        # data = ops.abs(data)
        real = ops.real(data)
        imag = ops.imag(data)
        data = ops.sqrt(real**2 + imag**2)
        data = ops.cast(data, "float32")

        return {self.output_key: data}


@ops_registry("log_compress")
class LogCompress(Operation):
    """Logarithmic compression of data."""

    def __init__(
        self,
        key: str = "envelope_data",
        output_key: str = "image",
        **kwargs,
    ):
        """Initialize the LogCompress operation.

        Args:
            key (str, optional): Key for input data. Defaults to "envelope_data".
            output_key (str, optional): Key for output data. Defaults to "image".
        """
        super().__init__(
            input_data_type=DataTypes.ENVELOPE_DATA,
            output_data_type=DataTypes.IMAGE,
            key=key,
            output_key=output_key,
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

    def __init__(
        self,
        key: str = "envelope_data",
        output_key: str = "envelope_data",
        **kwargs,
    ):
        """Initialize the Normalize operation.

        Args:
            key (str, optional): Key for input data. Defaults to "envelope_data".
            output_key (str, optional): Key for output data. Defaults to "envelope_data".
        """
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            key=key,
            output_key=output_key,
            **kwargs,
        )

    def call(self, output_range=None, input_range=None, **kwargs):
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

        if output_range is None:
            output_range = (0, 1)

        if input_range is None:
            minimum = ops.min(data)
            maximum = ops.max(data)
            input_range = (minimum, maximum)
        else:
            a_min, a_max = input_range
            data = ops.clip(data, a_min, a_max)

        normalized_data = translate(data, input_range, output_range)

        return {self.output_key: normalized_data}


@ops_registry("scan_convert")
class ScanConvert(Operation):
    """Scan convert images to cartesian coordinates."""

    def __init__(
        self,
        key: str = "image",
        output_key: str = "image_sc",
        order=1,
        **kwargs,
    ):
        """Initialize the ScanConvert operation.

        Args:
            key (str, optional): Key for input data. Defaults to "image".
            output_key (str, optional): Key for output data. Defaults to "image_sc".
            order (int, optional): Interpolation order. Defaults to 1. Currently only
                GPU support for order=1.
        """
        super().__init__(
            input_data_type=DataTypes.IMAGE,
            output_data_type=DataTypes.IMAGE_SC,
            key=key,
            output_key=output_key,
            jittable=False,  # currently not jittable due to variable size
            **kwargs,
        )
        self.order = order

    def call(
        self,
        rho_range=None,
        theta_range=None,
        phi_range=None,
        resolution=None,
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
            fill_value (float): Value to fill the image with outside the defined region.

        """

        data = kwargs[self.key]

        if phi_range is not None:
            data_out = scan_convert_3d(
                data,
                rho_range,
                theta_range,
                phi_range,
                resolution,
                fill_value,
                order=self.order,
            )
        else:
            data_out = scan_convert_2d(
                data,
                rho_range,
                theta_range,
                resolution,
                fill_value,
                order=self.order,
            )

        return {self.output_key: data_out}
