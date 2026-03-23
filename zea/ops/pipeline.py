import json
from typing import Dict, List, Union

import keras
import numpy as np
import yaml
from keras import ops

from zea import log
from zea.backend import jit
from zea.config import Config
from zea.func.tensor import (
    vmap,
)
from zea.func.ultrasound import channels_to_complex, complex_to_channels
from zea.internal.core import (
    DataTypes,
    ZEADecoderJSON,
    ZEAEncoderJSON,
    dict_to_tensor,
)
from zea.internal.core import Object as ZEAObject
from zea.internal.registry import ops_registry
from zea.internal.utils import deprecated
from zea.ops.base import (
    Operation,
    get_ops,
)
from zea.ops.tensor import Normalize
from zea.ops.ultrasound import (
    ApplyWindow,
    Demodulate,
    EnvelopeDetect,
    LogCompress,
    PfieldWeighting,
    ReshapeGrid,
    TOFCorrection,
)
from zea.probes import Probe
from zea.scan import Scan
from zea.utils import (
    FunctionTimer,
)


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

        self._user_jit_kwargs = jit_kwargs.copy()

        if keras.backend.backend() == "jax" and self.static_params != []:
            existing = jit_kwargs.get("static_argnames", [])
            if isinstance(existing, str):
                existing = [existing]
            jit_kwargs = {
                **jit_kwargs,
                "static_argnames": list(set(existing) | set(self.static_params)),
            }

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
        cls,
        beamformer="delay_and_sum",
        num_patches=100,
        baseband=False,
        enable_pfield=False,
        timed=False,
        **kwargs,
    ) -> "Pipeline":
        """Create a default pipeline.

        Args:
            beamformer (str): Type of beamformer to use. Currently supporting,
                "delay_and_sum" and "delay_multiply_and_sum". Defaults to "delay_and_sum".
            num_patches (int): Number of patches for the PatchedGrid operation.
                Defaults to 100. If you get an out of memory error, try to increase this number.
            baseband (bool): If True, assume the input data is baseband (I/Q) data,
                which has 2 channels (last dim). Defaults to False, which assumes RF data,
                so input signal has a single channel dim and is still on carrier frequency.
            enable_pfield (bool): If True, apply PfieldWeighting. Defaults to False.
                This will calculate pressure field and only beamform the data to those locations.
            timed (bool, optional): Whether to time each operation. Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the Pipeline constructor.

        """
        operations = []

        # Add the demodulate operation
        if not baseband:
            operations += [
                ApplyWindow(),
                Demodulate(),
            ]

        # Add beamforming ops
        operations.append(
            Beamform(
                beamformer=beamformer,
                num_patches=num_patches,
                enable_pfield=enable_pfield,
            ),
        )

        # Add display ops
        operations += [
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
    def operations(self) -> List[Union[Operation, "Pipeline"]]:
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
                k: ops.convert_to_numpy(v) if ops.is_tensor(v) else v for k, v in outputs.items()
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
        operations = []
        for operation in self.operations:
            if isinstance(operation, Pipeline):
                operations.append(f"{operation.__class__.__name__}({str(operation)})")
            else:
                operations.append(operation.__class__.__name__)
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
            return cls.from_path(file_path, **kwargs)
        else:
            raise ValueError("File must have extension .json, .yaml, or .yml")

    def get_dict(self, compact=True) -> dict:
        """Convert the pipeline to a dictionary.

        Args:
            compact (bool): If True (default), only include
                parameters that differ from their defaults.
                If False, include all parameters for full reproducibility.
        """
        config = {"name": ops_registry.get_name(self)}
        config["operations"] = self._pipeline_to_list(self, compact=compact)

        if compact:
            params = {}
            if not self.with_batch_dim:
                params["with_batch_dim"] = self.with_batch_dim
            if self.jit_options != "ops":
                params["jit_options"] = self.jit_options
            if self._user_jit_kwargs:
                params["jit_kwargs"] = self._user_jit_kwargs
            if params:
                config["params"] = params
        else:
            config["params"] = {
                "with_batch_dim": self.with_batch_dim,
                "jit_options": self.jit_options,
                "jit_kwargs": self._user_jit_kwargs,
            }

        return config

    @staticmethod
    def _pipeline_to_list(pipeline: "Pipeline", compact=True):
        """Convert the pipeline to a list of operations."""
        ops_list = []
        for op in pipeline.operations:
            if isinstance(op, Pipeline):
                ops_list.append(op.get_dict(compact=compact))
            else:
                d = op.get_dict(compact=compact)
                if compact:
                    params = d.get("params", {})
                    # Strip with_batch_dim when it is merely inherited from the pipeline
                    if params.get("with_batch_dim") == pipeline.with_batch_dim:
                        params.pop("with_batch_dim", None)
                    # Strip jit_compile=False when it is implied by pipeline-level JIT
                    if not params.get("jit_compile", True) and pipeline.jit_options in (
                        None,
                        "pipeline",
                    ):
                        params.pop("jit_compile", None)
                    if not params:
                        d.pop("params", None)
                    # Name-only dict → bare string shorthand
                    if list(d.keys()) == ["name"]:
                        d = d["name"]
                ops_list.append(d)
        return ops_list

    @classmethod
    def from_config(cls, config: Dict, **kwargs) -> "Pipeline":
        """Create a pipeline from a dictionary or ``zea.Config`` object.

        Args:
            config (dict or Config): Configuration dictionary or ``zea.Config`` object.
                Must have a ``pipeline`` key with a subkey ``operations``.
            **kwargs: Additional keyword arguments to be passed to the pipeline.

        Example:
            .. doctest::

                >>> from zea import Config, Pipeline
                >>> config = Config(
                ...     {
                ...         "pipeline": {
                ...             "operations": [
                ...                 "identity",
                ...             ],
                ...         }
                ...     }
                ... )
                >>> pipeline = Pipeline.from_config(config)
        """
        return pipeline_from_config(Config(config), **kwargs)

    @classmethod
    def from_path(cls, file_path: str, **kwargs) -> "Pipeline":
        """Create a pipeline from a YAML/config file path.

        Args:
            file_path (str): Path to the config file (local or ``hf://`` URI).
                Must have a ``pipeline`` key with a subkey ``operations``.
            **kwargs: Additional keyword arguments to be passed to the pipeline.

        Example:
            .. doctest::

                >>> from zea import Config, Pipeline
                >>> config = Config(
                ...     {
                ...         "pipeline": {
                ...             "operations": [
                ...                 "identity",
                ...             ],
                ...         }
                ...     }
                ... )
                >>> config.to_yaml("pipeline.yaml")
                >>> pipeline = Pipeline.from_path("pipeline.yaml")
        """
        config = Config.from_path(file_path)
        return pipeline_from_config(config, **kwargs)

    @classmethod
    @deprecated(replacement="Pipeline.from_path")
    def from_yaml(cls, file_path: str, **kwargs) -> "Pipeline":
        """Deprecated. Use :meth:`from_path` instead."""
        return pipeline_from_yaml(file_path, **kwargs)

    @classmethod
    def from_json(cls, json_string: str, **kwargs) -> "Pipeline":
        """Create a pipeline from a JSON string.

        Args:
            json_string (str): JSON string representing the pipeline.
                Must have a ``pipeline`` key with a subkey ``operations``.
            **kwargs: Additional keyword arguments to be passed to the pipeline.

        Example:
        ```python
        json_string = '{"pipeline": {"operations": ["identity"]}}'
        pipeline = Pipeline.from_json(json_string)
        ```
        """
        return pipeline_from_json(json_string, **kwargs)

    def to_config(self, compact=True) -> Config:
        """Convert the pipeline to a `zea.Config` object."""
        return pipeline_to_config(self, compact=compact)

    def to_json(self, compact=True) -> str:
        """Convert the pipeline to a JSON string."""
        return pipeline_to_json(self, compact=compact)

    def to_yaml(self, file_path: str, compact=True) -> None:
        """Convert the pipeline to a YAML file."""
        pipeline_to_yaml(self, file_path, compact=compact)

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
            **kwargs: Additional keyword arguments to be included in the inputs.

        Returns:
            dict: Dictionary of inputs with all values as tensors.
        """
        # Initialize dictionaries for probe, scan, and config
        probe_dict, scan_dict, config_dict = {}, {}, {}

        config_keys, kwargs_keys = set(), set()
        if config is not None:
            config_keys = set(config.keys())
        kwargs_keys = set(kwargs.keys())

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
            needs_keys = self.needs_keys - config_keys - kwargs_keys
            scan_dict = scan.to_tensor(include=needs_keys, keep_as_is=self.static_params)

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

    def get_dict(self, compact=True):
        """Get the configuration of the pipeline."""
        config = super().get_dict(compact=compact)
        config["name"] = "map"

        params = config.get("params", {})
        params["argnames"] = self.argnames
        if not compact or self.in_axes != 0:
            params["in_axes"] = self.in_axes
        if not compact or self.out_axes != 0:
            params["out_axes"] = self.out_axes
        if not compact or self.chunks is not None:
            params["chunks"] = self.chunks
        if not compact or self.batch_size is not None:
            params["batch_size"] = self.batch_size
        config["params"] = params
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

    def get_dict(self, compact=True):
        """Get the configuration of the pipeline."""
        config = super().get_dict(compact=compact)
        config["name"] = "patched_grid"

        params = config.get("params", {})
        params.pop("argnames", None)
        params.pop("chunks", None)
        params["num_patches"] = self.num_patches
        config["params"] = params
        return config


@ops_registry("beamform")
class Beamform(Pipeline):
    """Classical beamforming pipeline for ultrasound image formation.

    Expected input data type is `DataTypes.RF_DATA` which has shape `(n_tx, n_ax, n_el, n_ch)`.

    Will run the following operations in sequence:
    - TOFCorrection (output type `DataTypes.ALIGNED_DATA`: `(n_tx, n_ax, n_el, n_ch)`)
    - PfieldWeighting (optional, output type `DataTypes.ALIGNED_DATA`: `(n_tx, n_ax, n_el, n_ch)`)
    - Sum over channels (DAS)
    - Sum over transmits (Compounding) (output type `DataTypes.BEAMFORMED_DATA`: `(grid_size_z, grid_size_x, n_ch)`)
    - ReshapeGrid (flattened grid is also reshaped to `(grid_size_z, grid_size_x)`)
    """  # noqa: E501

    def __init__(self, beamformer="delay_and_sum", num_patches=100, enable_pfield=False, **kwargs):
        """Initialize a Delay-and-Sum beamforming `zea.Pipeline`.

        Args:
            beamformer (str): Type of beamformer to use. Currently supporting,
                "delay_and_sum" and "delay_multiply_and_sum".
            num_patches (int): Number of patches to split the grid into for patch-wise
                beamforming. If 1, no patching is performed.
            enable_pfield (bool): Whether to include pressure field weighting in the beamforming.

        """

        self.beamformer_type = beamformer
        self.num_patches = num_patches
        self.enable_pfield = enable_pfield

        # for backwards compatibility
        name_mapping = {
            "das": "delay_and_sum",
            "dmas": "delay_multiply_and_sum",
        }
        if beamformer in name_mapping:
            log.deprecated(
                f"Beamformer name '{beamformer}' is deprecated. "
                f"Please use '{name_mapping[beamformer]}' instead."
            )
            self.beamformer_type = name_mapping[beamformer]

        if self.beamformer_type not in ["delay_and_sum", "delay_multiply_and_sum"]:
            raise ValueError(
                f"Unsupported beamformer type: {self.beamformer_type}. "
                "Supported types are 'delay_and_sum' and 'delay_multiply_and_sum'."
            )

        # Get beamforming ops
        beamforming = [
            TOFCorrection(),
            # PfieldWeighting(),  # Inserted conditionally
            get_ops(self.beamformer_type)(),
        ]

        if self.enable_pfield:
            beamforming.insert(1, PfieldWeighting())

        # Optionally add patching
        if self.num_patches > 1:
            beamforming = [
                PatchedGrid(
                    operations=beamforming,
                    num_patches=self.num_patches,
                    **kwargs,
                )
            ]

        # Reshape the grid to image shape
        beamforming.append(ReshapeGrid())

        # Set the output data type of the last operation
        # which also defines the pipeline output type
        beamforming[-1].output_data_type = DataTypes.BEAMFORMED_DATA

        super().__init__(operations=beamforming, **kwargs)

    def __repr__(self):
        """String representation of the pipeline."""
        operations = []
        for operation in self.operations:
            if isinstance(operation, Pipeline):
                operations.append(repr(operation))
            else:
                operations.append(operation.__class__.__name__)
        return f"<Beamform {self.name}=({', '.join(operations)})>"

    def get_dict(self, compact=True) -> dict:
        """Convert the pipeline to a dictionary.

        Unlike Pipeline.get_dict(), this does NOT include the internal
        operations list, since Beamform auto-generates its operations
        from ``beamformer``, ``num_patches``, and ``enable_pfield``.
        """
        config = {"name": "beamform"}

        params = {}
        if not compact or self.beamformer_type != "delay_and_sum":
            params["beamformer"] = self.beamformer_type
        if not compact or self.num_patches != 100:
            params["num_patches"] = self.num_patches
        if not compact or self.enable_pfield:
            params["enable_pfield"] = self.enable_pfield

        # Pipeline-level params
        if compact:
            if not self.with_batch_dim:
                params["with_batch_dim"] = self.with_batch_dim
            if self.jit_options != "ops":
                params["jit_options"] = self.jit_options
            if self._user_jit_kwargs:
                params["jit_kwargs"] = self._user_jit_kwargs
        else:
            params["with_batch_dim"] = self.with_batch_dim
            params["jit_options"] = self.jit_options
            params["jit_kwargs"] = self._user_jit_kwargs

        if params:
            config["params"] = params

        return config


@ops_registry("delay_and_sum")
class DelayAndSum(Operation):
    """Sums time-delayed signals along channels and transmits."""

    def __init__(self, **kwargs):
        super().__init__(
            input_data_type=DataTypes.ALIGNED_DATA,
            output_data_type=DataTypes.BEAMFORMED_DATA,
            **kwargs,
        )

    def call(self, **kwargs):
        """Performs DAS beamforming on tof-corrected input.

        Args:
            tof_corrected_data (ops.Tensor): The TOF corrected input of shape
                `(n_tx, prod(grid.shape), n_el, n_ch)` with optional batch dimension.

        Returns:
            dict: Dictionary containing beamformed_data
                of shape `(prod(grid.shape), n_ch)`
                with optional batch dimension.
        """
        data = kwargs[self.key]

        # Sum over the channels (n_el), i.e. DAS
        beamformed_data = ops.sum(data, -2)
        # Sum over transmits (n_tx), i.e. Compounding
        beamformed_data = ops.sum(beamformed_data, -3)

        return {self.output_key: beamformed_data}


@ops_registry("delay_multiply_and_sum")
class DelayMultiplyAndSum(Operation):
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

    def call(self, **kwargs):
        """Performs DMAS beamforming on tof-corrected input.

        Args:
            tof_corrected_data (ops.Tensor): The TOF corrected input of shape
                `(n_tx, prod(grid.shape), n_el, n_ch)` with optional batch dimension.

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

            # Check for nested operations at the same level as params
            if "operations" in operation:
                nested_operations = make_operation_chain(operation["operations"])
                # Instantiate pipeline-type operations with nested operations
                if issubclass(operation_cls, Beamform):
                    # some pipelines, such as `zea.ops.Beamformer`, are initialized
                    # not with a list of operations but with other parameters that then
                    # internally create a list of operations
                    operation_instance = operation_cls(**params)
                elif issubclass(operation_cls, Pipeline):
                    # in most cases we want to pass an operations list to
                    # initialize a pipeline
                    operation_instance = operation_cls(operations=nested_operations, **params)
                else:
                    operation_instance = operation_cls(operations=nested_operations, **params)
            else:
                operation_instance = operation_cls(**params)

        chain.append(operation_instance)

    return chain


def pipeline_from_config(config: Config, **kwargs) -> Pipeline:
    """
    Create a Pipeline instance from a Config object.

    The config must have a top-level ``pipeline`` key containing an ``operations`` list.
    """
    if "pipeline" not in config:
        top_keys = list(config.keys()) if hasattr(config, "keys") else []
        raise ValueError(
            f"Cannot build Pipeline: missing top-level 'pipeline' key.\n"
            f"Expected a config with the format:\n"
            f"  pipeline:\n"
            f"    operations:\n"
            f"      - <operation_name>\n"
            f"      - ...\n"
            f"Found top-level keys: {top_keys}"
        )

    # Unwrap the pipeline subsection from a full config
    config = Config(config["pipeline"])

    if "operations" not in config:
        top_keys = list(config.keys()) if hasattr(config, "keys") else []
        raise ValueError(
            f"Cannot build Pipeline: missing 'operations' key.\n"
            f"Expected a config with the format:\n"
            f"  pipeline:\n"
            f"    operations:\n"
            f"      - <operation_name>\n"
            f"      - ...\n"
            f"Found top-level keys: {top_keys}"
        )

    if not isinstance(config.operations, (list, np.ndarray)):
        raise ValueError(
            f"Cannot build Pipeline: 'operations' must be a list, "
            f"got {type(config.operations).__name__}."
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


@deprecated(replacement="Pipeline.from_path")
def pipeline_from_yaml(yaml_path: str, **kwargs) -> Pipeline:  # pragma: no cover
    """
    Create a Pipeline instance from a YAML file.

    .. deprecated::
        Use :meth:`Pipeline.from_path` instead.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        pipeline_config = yaml.safe_load(f)
    return pipeline_from_config(Config(pipeline_config), **kwargs)


def _pipeline_to_serializable_dict(pipeline: Pipeline, compact=True) -> dict:
    """Convert a Pipeline to a dict suitable for serialization.

    The output format is ``{"pipeline": {"operations": [...], ...pipeline_kwargs}}``
    which can be loaded back via ``pipeline_from_config``.
    """
    pipeline_dict = {
        "operations": Pipeline._pipeline_to_list(pipeline, compact=compact),
    }

    if compact:
        if not pipeline.with_batch_dim:
            pipeline_dict["with_batch_dim"] = pipeline.with_batch_dim
        if pipeline.jit_options != "ops":
            pipeline_dict["jit_options"] = pipeline.jit_options
        if pipeline._user_jit_kwargs:
            pipeline_dict["jit_kwargs"] = pipeline._user_jit_kwargs
        if pipeline.name != "pipeline":
            pipeline_dict["name"] = pipeline.name
    else:
        pipeline_dict.update(
            {
                "with_batch_dim": pipeline.with_batch_dim,
                "jit_options": pipeline.jit_options,
                "jit_kwargs": pipeline._user_jit_kwargs,
                "name": pipeline.name,
            }
        )

    return {"pipeline": pipeline_dict}


def pipeline_to_config(pipeline: Pipeline, compact=True) -> Config:
    """
    Convert a Pipeline instance into a Config object.
    """
    return Config(_pipeline_to_serializable_dict(pipeline, compact=compact))


def pipeline_to_json(pipeline: Pipeline, compact=True) -> str:
    """
    Convert a Pipeline instance into a JSON string.
    """
    return json.dumps(
        _pipeline_to_serializable_dict(pipeline, compact=compact),
        cls=ZEAEncoderJSON,
        indent=4,
    )


def pipeline_to_yaml(pipeline: Pipeline, file_path: str, compact=True) -> None:
    """
    Convert a Pipeline instance into a YAML file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            _pipeline_to_serializable_dict(pipeline, compact=compact),
            f,
            indent=4,
        )
