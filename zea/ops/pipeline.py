import difflib
import inspect
import json
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Union, cast

import keras
import numpy as np
import yaml
from keras import ops

from zea import backend, log
from zea.backend import func_on_device, jit
from zea.config import Config
from zea.func.tensor import vmap
from zea.internal.core import DataTypes, ZEADecoderJSON, ZEAEncoderJSON, dict_to_tensor
from zea.internal.ops_list import OperationList
from zea.internal.precision import LOW_PRECISION_DTYPES
from zea.internal.registry import beamformer_registry, ops_registry
from zea.internal.utils import deprecated
from zea.ops.base import Operation, get_ops
from zea.ops.tensor import Normalize
from zea.ops.ultrasound import (
    AlignedApodization,
    ApplyWindow,
    Demodulate,
    EnvelopeDetect,
    LogCompress,
    PfieldWeighting,
    ReceiveApodization,
    ReshapeGrid,
    TOFCorrection,
)
from zea.utils import FunctionTimer

if TYPE_CHECKING:
    # Imported lazily at runtime (inside prepare_parameters) to avoid a circular
    # import: zea.parameters imports the data specs, which can pull in this module.
    from zea.parameters import Parameters


class PipelineError(RuntimeError):
    """Error raised when an operation inside a :class:`Pipeline` fails.

    Subclass of :class:`RuntimeError` so existing ``except RuntimeError`` handlers
    keep working. Instances carry a ``_zea_annotated`` marker so that enclosing
    pipelines do not re-wrap (and thus double-annotate) an error that an inner
    pipeline already reported.
    """

    _zea_annotated = True


class PipelineKeyError(KeyError):
    """Raised when a :class:`Pipeline` operation is missing a required input key.

    Subclass of :class:`KeyError` so existing ``except KeyError`` handlers keep
    working. ``__str__`` is overridden so the (multi-line) message renders as-is
    instead of the ``repr``-quoted single line that :class:`KeyError` produces.
    Carries the ``_zea_annotated`` marker to prevent double-wrapping.
    """

    _zea_annotated = True

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


def _summarize_inputs(inputs: Dict[str, Any]) -> str:
    """Compact ``key: shape dtype`` summary of pipeline inputs for error messages.

    Keeps large arrays out of tracebacks while still surfacing the shape/dtype
    information needed to diagnose most pipeline failures.
    """
    parts = []
    for key, value in inputs.items():
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None and dtype is not None:
            parts.append(f"{key}: {tuple(shape)} {dtype}")
        else:
            parts.append(f"{key}: {type(value).__name__}")
    return "{" + ", ".join(parts) + "}"


@ops_registry("pipeline")
class Pipeline:
    """Pipeline class for processing ultrasound data through a series of
    :class:`~zea.ops.base.Operation` objects.
    """

    def __init__(
        self,
        operations: Sequence[Union[Operation, "Pipeline"]],
        with_batch_dim: bool = True,
        jit_options: Union[str, None] = "ops",
        jit_kwargs: dict | None = None,
        name="pipeline",
        validate=True,
        timed: bool = False,
        device: Union[str, None] = None,
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
            device (str, optional): Default device for all pipeline calls, e.g.
                ``'cpu'``, ``'gpu:0'``, ``'cuda:1'``.  Can be overridden per-call
                by passing ``device=`` to ``__call__``.  Uses
                :func:`zea.backend.func_on_device` under the hood, which moves
                input tensors to the device for the ``torch`` backend and wraps
                the call in a device context for JAX / TensorFlow.  Defaults to
                ``None`` (no device placement).

        """
        self._call_pipeline = self.call
        self.name = name

        self._pipeline_layers: List[Union[Operation, "Pipeline"]] = list(operations)

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
        # True when an enclosing pipeline is JIT-compiled as a whole, so this
        # pipeline already runs inside that trace. Updated by the parent via
        # _configure_jit; defaults to False for a standalone/root pipeline.
        self._inside_outer_jit = False
        self.jit_options = jit_options  # will handle the jit compilation
        self.device = device

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
        enable_aligned_apodization=False,
        enable_receive_apodization=False,
        timed=False,
        **kwargs,
    ) -> "Pipeline":
        """Create a default pipeline.

        Args:
            beamformer (str): Type of beamformer to use.
                Currently supporting:
                - "delay_and_sum"
                - "delay_multiply_and_sum"
                - "coherence_factor"
                - "generalized_coherence_factor"
                - "minimum_variance"
                Defaults to "delay_and_sum".
            num_patches (int): Number of patches for the PatchedGrid operation.
                Defaults to 100. If you get an out of memory error, try to increase this number.
            baseband (bool): If True, assume the input data is baseband (I/Q) data,
                which has 2 channels (last dim). Defaults to False, which assumes RF data,
                so input signal has a single channel dim and is still on carrier frequency.
            enable_pfield (bool): If True, apply PfieldWeighting. Defaults to False.
                This will calculate pressure field and only beamform the data to those locations.
            enable_aligned_apodization (bool): If True, apply AlignedApodization (a per-pixel,
                per-transmit compounding weight) using ``parameters.flat_aligned_apodization``.
                Defaults to False. Used e.g. for scanline (line-by-line) imaging with
                ``parameters.enable_scanline = True``.
            enable_receive_apodization (bool): If True, apply ReceiveApodization (a custom
                per-pixel, per-element receive-aperture weight) using
                ``parameters.flat_receive_apodization``. Defaults to False.
            timed (bool, optional): Whether to time each operation. Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the Pipeline constructor.

        """
        operations: List[Union[Operation, "Pipeline"]] = []

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
                enable_aligned_apodization=enable_aligned_apodization,
                enable_receive_apodization=enable_receive_apodization,
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
            device=self.device,
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
            device=self.device,
        )

    @staticmethod
    def _check_op_is_instance(operation):
        """Raise a clear TypeError when a class is passed instead of an instance."""
        if isinstance(operation, type):
            raise TypeError(
                f"Expected an Operation instance, got class {operation.__name__!r}. "
                f"Did you forget the parentheses? "
                f"Use {operation.__name__}() instead of {operation.__name__}."
            )

    def prepend(self, operation: Operation):
        """Prepend an operation to the pipeline."""
        self._check_op_is_instance(operation)
        self._pipeline_layers.insert(0, operation)
        self.reinitialize()

    def append(self, operation: Operation):
        """Append an operation to the pipeline."""
        self._check_op_is_instance(operation)
        self._pipeline_layers.append(operation)
        self.reinitialize()

    def insert(self, index: int, operation: Operation):
        """Insert an operation at a specific index in the pipeline."""
        self._check_op_is_instance(operation)
        if index < 0 or index > len(self._pipeline_layers):
            raise IndexError("Index out of bounds for inserting operation.")
        self._pipeline_layers.insert(index, operation)
        self.reinitialize()

    @property
    def operations(self) -> List[Union[Operation, "Pipeline"]]:
        """Alias for self.layers to match the zea naming convention"""
        return self._pipeline_layers

    def __getitem__(self, key: str):
        """Look up an operation by name.

        Allows chaining directly on the pipeline object::

            pipeline["beamform"]["tof_correction"]

        Use :meth:`keys` to see available names.
        Duplicate operation names are disambiguated with a ``_N`` suffix,
        e.g. ``pipeline["normalize_0"]``.
        """
        return OperationList(self._pipeline_layers)[key]

    def keys(self):
        """Return the string keys that can be used with ``pipeline[key]``.

        Example::

            pipeline.keys()
            # ['apply_window', 'demodulate', 'beamform', ...]
        """
        return OperationList(self._pipeline_layers).keys()

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

    def call(self, **inputs) -> Dict[str, Any]:
        """Process input data through the pipeline."""

        for operation in self._callable_layers:
            try:
                outputs = operation(**inputs)
            except Exception as exc:
                # Already annotated by this or an inner pipeline: re-raise as-is so
                # we do not wrap the message (and traceback) a second time.
                if getattr(exc, "_zea_annotated", False):
                    raise
                if isinstance(exc, KeyError):
                    self._raise_missing_key(operation, exc, inputs)
                else:
                    self._raise_operation_error(operation, exc, inputs)
            inputs = outputs
        return outputs

    def _raise_missing_key(self, operation, exc: KeyError, inputs: Dict[str, Any]):
        """Re-raise a bare ``KeyError`` from an operation with actionable context."""
        missing = exc.args[0] if exc.args else "?"
        unused = [k for k in (set(inputs.keys()) - self.valid_keys) if k != "kwargs"]
        # If the caller passed something close to the missing key, it is likely a typo.
        typo = difflib.get_close_matches(str(missing), unused, n=1, cutoff=0.6)
        hint = (
            f" You provided '{typo[0]}', which is unused — did you mean '{missing}'?"
            if typo
            else ""
        )
        raise PipelineKeyError(
            f"[zea.Pipeline] Operation '{operation.__class__.__name__}' "
            f"requires input key '{missing}', but it was not provided.{hint}\n"
            "Check whether the objects (such as `zea.Parameters`) passed to "
            "`pipeline.prepare_parameters()` contain all required keys.\n"
            f"Provided keys: {sorted(inputs.keys())}\n"
            f"Valid keys for this pipeline: {sorted(self.valid_keys - {'kwargs'})}"
        ) from exc

    def _raise_operation_error(self, operation, exc: Exception, inputs: Dict[str, Any]):
        """Re-raise a generic operation failure as a :class:`PipelineError`.

        Adds the failing operation name and a compact shape/dtype summary of its
        inputs, and truncates the underlying message so a concrete (non-jit) array
        is never dumped in full into the traceback.
        """
        original = str(exc)
        if len(original) > 500:
            original = original[:500] + "… (truncated)"
        raise PipelineError(
            f"[zea.Pipeline] Operation '{operation.__class__.__name__}' failed with "
            f"{type(exc).__name__}: {original}\n"
            f"Inputs: {_summarize_inputs(inputs)}"
        ) from exc

    def __call__(
        self, return_numpy=False, device: Union[str, None] = None, **inputs
    ) -> Dict[str, Any]:
        """Process input data through the pipeline.

        Args:
            return_numpy (bool): If ``True``, convert output tensors to NumPy
                arrays before returning.
            device (str, optional): Device to run this call on, e.g.
                ``'cpu'``, ``'gpu:0'``, or ``'cuda:1'``.  Overrides the
                pipeline-level ``device`` set at construction time for this
                single invocation.  When ``None`` (default), the pipeline-level
                ``device`` attribute is used (which is also ``None`` by
                default, meaning no explicit device placement).
            **inputs: Tensor inputs forwarded to the operations.
        """
        from zea.internal.parameters import BaseParameters

        if any(key in inputs for key in ["probe", "scan", "config", "parameters"]) or any(
            isinstance(arg, BaseParameters) for arg in inputs.values()
        ):
            raise ValueError(
                "Parameters (and Probe/Config) objects should be first processed with "
                "`Pipeline.prepare_parameters` before calling the pipeline. "
                "e.g. `inputs = pipeline.prepare_parameters(parameters, **overrides)`"
            )

        if any(isinstance(arg, str) for arg in inputs.values()):
            raise ValueError(
                "Pipeline does not support string inputs. "
                "Please ensure all inputs are convertible to tensors, or use "
                "`inputs = Pipeline.prepare_parameters(parameters)` to convert "
                "all your parameters for you."
            )

        if not self._logged_difference_keys:
            difference_keys = set(inputs.keys()) - self.valid_keys
            if difference_keys:
                # Separate likely typos (close to a key the pipeline actually uses)
                # from benign pass-through keys (e.g. extra `zea.Parameters` fields).
                candidates = self.valid_keys - {"kwargs"}
                matches = {
                    key: difflib.get_close_matches(key, candidates, n=1, cutoff=0.6)
                    for key in difference_keys
                }
                typos = {key: match[0] for key, match in matches.items() if match}
                benign = difference_keys - set(typos)
                if typos:
                    hints = ", ".join(f"'{k}' -> did you mean '{v}'?" for k, v in typos.items())
                    log.warning(
                        f"[zea.Pipeline] Some input keys look like typos and are ignored: {hints}"
                    )
                if benign:
                    log.debug(
                        f"[zea.Pipeline] Ignoring input keys not used by the pipeline: "
                        f"{sorted(benign)}."
                    )
                self._logged_difference_keys = True

        ## PROCESSING
        _device = device if device is not None else self.device
        if _device is not None:
            outputs = func_on_device(self._call_pipeline, _device, **inputs)
        else:
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

    def set_jit(self, value: bool):
        """Set the JIT compilation for the pipeline."""
        if value:
            self._jit()
        else:
            self._unjit()

    @jit_options.setter
    def jit_options(self, value: Union[str, None]):
        """Set the jit_options property of the pipeline."""
        self._configure_jit(value, inside_outer_jit=self._inside_outer_jit)

    def _configure_jit(self, value: Union[str, None], inside_outer_jit: bool):
        """Recursively configure JIT for this pipeline and all descendants.

        Args:
            value: jit_options for this pipeline ("pipeline", "ops", or None).
            inside_outer_jit: True if an enclosing pipeline is JIT-compiled as a
                whole, so this pipeline already runs inside a trace.
        """
        if value not in ("pipeline", "ops", None):
            raise ValueError(f"jit_options must be 'pipeline', 'ops', or None, got {value!r}")

        self._jit_options = value
        self._inside_outer_jit = inside_outer_jit
        self.set_jit(value == "pipeline")

        # Children run inside a trace if we compile ourselves as a whole, or we
        # already do. When that happens they must not add their own JIT, so their
        # jit_options is forced to None.
        child_inside_outer_jit = inside_outer_jit or value == "pipeline"
        child_value = None if child_inside_outer_jit else value
        for operation in self.operations:
            if isinstance(operation, Pipeline):
                operation._configure_jit(child_value, inside_outer_jit=child_inside_outer_jit)
            else:
                operation.set_jit(child_value == "ops")
                operation._inside_outer_jit = child_inside_outer_jit

    def _jit(self):
        """JIT compile the pipeline."""
        if not self.jittable:
            raise ValueError(
                "Cannot JIT compile the pipeline because not all operations are jittable. "
                f"The following operations are not jittable: {self.unjitable_ops}"
                "Try setting jit_options to 'ops' or None."
            )
        self._call_pipeline = jit(self.call, **self.jit_kwargs)

    def _unjit(self):
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
        for i, op in enumerate(operations):
            if isinstance(op, type):
                raise TypeError(
                    f"Pipeline operation at index {i} is a class ({op.__name__!r}), "
                    "not an instance. "
                    f"Did you forget the parentheses? "
                    f"Use {op.__name__}() instead of {op.__name__}."
                )
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
            if isinstance(operation, Pipeline):
                operation.set_params(**params)
            elif isinstance(operation, Operation):
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
            result = []
            for operation in self.operations:
                if isinstance(operation, Pipeline):
                    result.extend(operation.get_params(per_operation=True))
                elif isinstance(operation, Operation):
                    result.append(operation._input_cache.copy())
            return result
        else:
            params = {}
            for operation in self.operations:
                if isinstance(operation, Pipeline):
                    params.update(operation.get_params(per_operation=False))
                elif isinstance(operation, Operation):
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
        return f"Pipeline(name={self.name!r}, operations=[{', '.join(operations)}])"

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
            if self.device is not None:
                params["device"] = self.device
            if params:
                config["params"] = params
        else:
            config["params"] = {
                "with_batch_dim": self.with_batch_dim,
                "jit_options": self.jit_options,
                "jit_kwargs": self._user_jit_kwargs,
                "device": self.device,
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
    def from_path(cls, file_path: str, revision: str | None = None, **kwargs) -> "Pipeline":
        """Create a pipeline from a YAML/config file path.

        Args:
            file_path (str): Path to the config file (local or ``hf://`` URI).
                Must have a ``pipeline`` key with a subkey ``operations``.
            revision (str, optional): Revision of the config file (for Hugging Face ``hf://`` URIs).
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

            .. testcleanup::

                import os
                os.remove("pipeline.yaml")

        """
        config = Config.from_path(file_path, revision=revision)
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
        parameters: Union["Parameters", None] = None,
        device: Union[str, None] = None,
        **overrides,
    ) -> Dict[str, Any]:
        """Prepare a :class:`~zea.Parameters` object for the pipeline.

        Converts the (validated and derived) parameters needed by this
        pipeline's operations into a dictionary of tensors, then overlays any
        manually supplied overrides (e.g. ``config.parameters`` or ad-hoc
        keyword arguments). Overrides take priority over the values in
        ``parameters``.

        Args:
            parameters: :class:`~zea.Parameters` object. Only the keys
                this pipeline ``needs`` (and that are not overridden) are
                converted, so derivation is lazy and minimal.
            device: Device to place the tensors on. Defaults to the pipeline
                device.
            **overrides: Additional parameters to include in the inputs
                (converted to tensors). These overwrite values taken from
                ``parameters``.

        Returns:
            dict: Dictionary of inputs with all values as tensors.

        Example:
            .. code-block:: python

                inputs = pipeline.prepare_parameters(parameters, **config.parameters)
                outputs = pipeline(data=raw_data, **inputs)
        """
        from zea.parameters import Parameters

        _device = device if device is not None else self.device

        params_dict = {}
        override_keys = set(overrides.keys())

        if parameters is not None:
            if not isinstance(parameters, Parameters):
                raise TypeError(f"Expected an instance of `zea.Parameters`, got {type(parameters)}")
            # Only convert keys the pipeline needs and that are not overridden,
            # so we avoid deriving unnecessary parameters.
            needs_keys = self.needs_keys - override_keys
            with backend.device(_device):
                params_dict = parameters.to_tensor(
                    include=list(needs_keys), keep_as_is=self.static_params
                )

        # Convert all overrides to tensors
        with backend.device(_device):
            tensor_overrides = dict_to_tensor(overrides, keep_as_is=self.static_params)

        # Overrides overwrite values taken from the parameters object.
        prepared = {**params_dict, **tensor_overrides}

        return prepared


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

    def call_item(self, **inputs):
        """Process data in patches."""
        mapped_args = []
        for argname in self.argnames:
            mapped_args.append(inputs.pop(argname, None))

        def patched_call(*args):
            mapped_kwargs = [(k, v) for k, v in zip(self.argnames, args)]
            out = super(Map, self).call(**dict(mapped_kwargs), **inputs)

            # TODO: maybe it is possible to output everything?
            # e.g. prepend a empty dimension to all inputs and just map over everything?
            return out[self.output_key]

        out = vmap(
            patched_call,
            in_axes=self.in_axes,
            out_axes=self.out_axes,
            chunks=self.chunks,
            batch_size=self.batch_size,
            fn_supports_batch=True,
            disable_jit=not bool(self.jit_options) and not self._inside_outer_jit,
        )(*mapped_args)

        return out

    def _configure_jit(self, value: Union[str, None], inside_outer_jit: bool):
        """Configure JIT for this Map and its inner operations.

        Map compiles its entire mapped call as a single unit whenever it self-jits
        (any non-None ``jit_options``). Its inner operations never JIT themselves;
        Map owns that scope. See :meth:`Pipeline._configure_jit`.
        """
        if value not in ("pipeline", "ops", None):
            raise ValueError(f"jit_options must be 'pipeline', 'ops', or None, got {value!r}")

        self._jit_options = value
        self._inside_outer_jit = inside_outer_jit
        self.set_jit(value is not None)

        # Inner ops run inside a trace if Map self-jits or an outer pipeline does.
        child_inside_outer_jit = value is not None or inside_outer_jit
        for operation in self.operations:
            if isinstance(operation, Pipeline):
                operation._configure_jit(None, inside_outer_jit=child_inside_outer_jit)
            else:
                operation.set_jit(False)
                operation._inside_outer_jit = child_inside_outer_jit

    def _jit(self):
        """JIT compile the pipeline."""
        self._jittable_call = jit(self.jittable_call, **self.jit_kwargs)

    def _unjit(self):
        """Un-JIT compile the pipeline."""
        self._jittable_call = self.jittable_call

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
    A pipeline that maps its operations over `flatgrid`, `flat_pfield`,
    `flat_aligned_apodization`, and `flat_receive_apodization` keys.

    This can be used to reduce memory usage by processing data in chunks.

    For more information and flexibility, see :class:`zea.ops.Map`.
    """

    def __init__(self, *args, num_patches=10, **kwargs):
        super().__init__(
            *args,
            argnames=[
                "flatgrid",
                "flat_pfield",
                "flat_aligned_apodization",
                "flat_receive_apodization",
            ],
            chunks=num_patches,
            **kwargs,
        )
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
    - ReceiveApodization (optional, output type `DataTypes.ALIGNED_DATA`: `(n_tx, n_ax, n_el, n_ch)`)
    - AlignedApodization (optional, output type `DataTypes.ALIGNED_DATA`: `(n_tx, n_ax, n_el, n_ch)`)
    - Sum over channels (DAS)
    - Sum over transmits (Compounding) (output type `DataTypes.BEAMFORMED_DATA`: `(grid_size_z, grid_size_x, n_ch)`)
    - ReshapeGrid (flattened grid is also reshaped to `(grid_size_z, grid_size_x)`)

    There are two distinct apodization stages, plus the built-in f-number mask
    (fused inside ``TOFCorrection``):

    - ``AlignedApodization`` weights the **transmit** axis (compounding) with a
      per-pixel, per-transmit mask (``parameters.flat_aligned_apodization``).
    - ``ReceiveApodization`` weights the **receive-element** axis with a custom
      per-pixel, per-element mask (``parameters.flat_receive_apodization``), on
      top of the built-in f-number receive-aperture mask.

    Scanline (line-by-line) imaging is a special case of this pipeline: set
    ``parameters.enable_scanline = True`` and ``enable_aligned_apodization=True``
    so that each pixel's grid column is beamformed from its own owning transmit
    only, with every other transmit masked to zero by
    :class:`~zea.ops.AlignedApodization` (fed
    :func:`zea.beamform.pixelgrid.scanline_aligned_apodization`) instead of
    compounding all transmits onto a shared grid.
    """  # noqa: E501

    def __init__(
        self,
        beamformer="delay_and_sum",
        num_patches=100,
        enable_pfield=False,
        enable_aligned_apodization=False,
        enable_receive_apodization=False,
        **kwargs,
    ):
        """Initialize a Delay-and-Sum beamforming `zea.Pipeline`.

        Args:
            beamformer (str): Type of beamformer to use.
                Currently supporting:
                - "delay_and_sum"
                - "delay_multiply_and_sum"
                - "coherence_factor"
                - "generalized_coherence_factor"
                - "minimum_variance"
                Defaults to "delay_and_sum".
            num_patches (int): Number of patches to split the grid into for patch-wise
                beamforming. If 1, no patching is performed.
            enable_pfield (bool): Whether to include pressure field weighting in the beamforming.
                Mutually exclusive with ``enable_aligned_apodization``.
            enable_aligned_apodization (bool): Whether to include an explicit per-pixel,
                per-transmit compounding apodization mask
                (``parameters.flat_aligned_apodization``) in the beamforming, e.g. to
                reconstruct scanline imaging (see class docstring). Mutually exclusive with
                ``enable_pfield`` (both weight the transmit axis).
            enable_receive_apodization (bool): Whether to include a custom per-pixel,
                per-element receive-aperture apodization mask
                (``parameters.flat_receive_apodization``) in the beamforming. Applied in
                addition to the built-in f-number mask; independent of and combinable with
                ``enable_pfield`` / ``enable_aligned_apodization``.
            **kwargs: Any keyword accepted by the chosen ``beamformer``'s own constructor
                (e.g. ``subarray_size`` / ``diagonal_loading`` for ``"minimum_variance"``)
                is forwarded to it. Remaining keywords are forwarded to the underlying
                ``Pipeline`` / ``PatchedGrid``.
        """
        if enable_pfield and enable_aligned_apodization:
            raise ValueError(
                "enable_pfield and enable_aligned_apodization are mutually exclusive. "
                "Please specify only one."
            )

        self.beamformer_type = beamformer
        self.num_patches = num_patches
        self.enable_pfield = enable_pfield
        self.enable_aligned_apodization = enable_aligned_apodization
        self.enable_receive_apodization = enable_receive_apodization

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

        if self.beamformer_type not in beamformer_registry:
            raise ValueError(
                f"Unsupported beamformer type: '{self.beamformer_type}'. "
                f"Supported types are: {beamformer_registry.registered_names()}."
            )

        # Pull out any kwargs meant for the beamformer op itself (e.g. `subarray_size` /
        # `diagonal_loading` for "minimum_variance"), so they don't leak into the
        # Pipeline / PatchedGrid kwargs below.
        beamformer_cls = beamformer_registry[self.beamformer_type]
        beamformer_params = {
            name
            for name, param in inspect.signature(beamformer_cls.__init__).parameters.items()
            if name != "self" and param.kind != inspect.Parameter.VAR_KEYWORD
        }
        # Never steal a keyword that the pipeline itself understands (e.g. `name`,
        # `with_batch_dim`): those must keep reaching Pipeline / PatchedGrid.
        pipeline_params = set(inspect.signature(Pipeline.__init__).parameters)
        beamformer_params -= pipeline_params
        self.beamformer_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in beamformer_params}

        # Get beamforming ops
        beamforming: List[Operation] = [
            TOFCorrection(),
            # ReceiveApodization() / AlignedApodization() / PfieldWeighting(),
            # inserted conditionally
            beamformer_cls(**self.beamformer_kwargs),
        ]

        if self.enable_receive_apodization:
            beamforming.insert(1, ReceiveApodization())

        if self.enable_aligned_apodization:
            beamforming.insert(1, AlignedApodization())

        if self.enable_pfield:
            beamforming.insert(1, PfieldWeighting())

        # Optionally add patching
        if self.num_patches > 1:
            beamforming = cast(  # type: ignore[assignment]
                List[Operation],
                [PatchedGrid(operations=beamforming, num_patches=self.num_patches, **kwargs)],
            )

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
        return f"Beamform(name={self.name!r}, operations=[{', '.join(operations)}])"

    def get_dict(self, compact=True) -> dict:
        """Convert the pipeline to a dictionary.

        Unlike Pipeline.get_dict(), this does NOT include the internal
        operations list, since Beamform auto-generates its operations
        from ``beamformer``, ``num_patches``, ``enable_pfield``,
        ``enable_aligned_apodization``, and ``enable_receive_apodization``.
        """
        config = super().get_dict(compact=compact)
        config.pop("operations", None)

        params = {}
        if not compact or self.beamformer_type != "delay_and_sum":
            params["beamformer"] = self.beamformer_type
        if not compact or self.num_patches != 100:
            params["num_patches"] = self.num_patches
        if not compact or self.enable_pfield:
            params["enable_pfield"] = self.enable_pfield
        if not compact or self.enable_aligned_apodization:
            params["enable_aligned_apodization"] = self.enable_aligned_apodization
        if not compact or self.enable_receive_apodization:
            params["enable_receive_apodization"] = self.enable_receive_apodization
        params.update(self.beamformer_kwargs)

        # Merge in the pipeline-level params from super().
        params.update(config.get("params", {}))

        if params:
            config["params"] = params
        else:
            config.pop("params", None)

        return config


@beamformer_registry("delay_and_sum")
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
        # Accumulate the delay-and-sum in float32. Under a mixed-precision policy
        # the aligned data is bfloat16 (produced cheaply by the TOF gather); summing
        # ~n_el * n_tx terms in bfloat16 would accumulate significant error, so the
        # reduction is up-cast. XLA fuses the cast into the reduction, so the input
        # is still read as bfloat16 (half the memory bandwidth) but accumulated in
        # float32. Under the default float32 policy this cast is a no-op.
        data = ops.cast(kwargs[self.key], "float32")

        # Sum over the channels (n_el), i.e. DAS
        beamformed_data = ops.sum(data, -2)
        # Sum over transmits (n_tx), i.e. Compounding
        beamformed_data = ops.sum(beamformed_data, -3)

        return {self.output_key: beamformed_data}


@beamformer_registry("delay_multiply_and_sum")
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

        # Avoid building the full (n_el, n_el) pairwise product matrix: rewrite
        # sum_{i<j} y_i y_j as 1/2 [(sum y_i)^2 - sum y_i^2]; O(n_el) instead of O(n_el^2).
        # There is no complex bfloat16, and under a mixed-precision policy the aligned
        # data arrives as bfloat16, so up-cast before the complex view. Only touch
        # low-precision input: a custom pipeline may already hand this op float32/float64.
        if keras.backend.standardize_dtype(data.dtype) in LOW_PRECISION_DTYPES:
            data = ops.cast(data, "float32")
        data = ops.view_as_complex(data)  # (n_tx, n_pix, n_el)

        # y_i = x_i / sqrt(|x_i|); eps guards |x_i| == 0 (then y_i -> 0).
        eps = keras.backend.epsilon()
        y = data / ops.cast(ops.sqrt(ops.abs(data)) + eps, data.dtype)

        sum_y = ops.sum(y, axis=-1)  # sum_i y_i        -> (n_tx, n_pix)
        sum_y2 = ops.sum(y * y, axis=-1)  # sum_i y_i^2 -> (n_tx, n_pix)
        per_tx = 0.5 * (sum_y * sum_y - sum_y2)  # sum_{i<j} y_i y_j

        # Compound over transmits.
        data = ops.sum(per_tx, axis=0)  # (n_pix,)

        return ops.view_as_real(data)

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


@beamformer_registry("coherence_factor")
@ops_registry("coherence_factor")
class CoherenceFactor(Operation):
    r"""Coherence Factor (CF) Beamformer.

    The Coherence Factor is a pixel-dependent weight used to quantify the focus
    quality of the beamformed signal. It is the ratio of the coherent power to
    the incoherent power of the signals received across the transducer aperture.

    For a set of delayed signals :math:`x_i` across :math:`N` elements:

    .. math::

        \mathrm{CF} = \frac{\left|\sum_{i=1}^{N} x_i\right|^2}
        {N \sum_{i=1}^{N} \left|x_i\right|^2}

    The CF ranges from 0 (completely incoherent) to 1 (perfectly coherent).
    The beamformed output is the standard DAS sum weighted by CF per transmit,
    then compounded across transmits.

    .. admonition:: Reference

        Hollman, K. W., Rigby, K. W., & O'Donnell, M. (1999).
        Coherence factor of speckle from a multi-row probe. IEEE Ultrasonics Symposium.

    Args:
        **kwargs: Additional arguments passed to the Operation base class.
    """

    def __init__(self, **kwargs):
        super().__init__(
            input_data_type=DataTypes.ALIGNED_DATA,
            output_data_type=DataTypes.BEAMFORMED_DATA,
            **kwargs,
        )

    def process_image(self, data):
        """Applies CF weighting and compounding on tof-corrected input.

        Args:
            data (ops.Tensor): TOF-corrected input of shape
                ``(n_tx, n_pix, n_el, n_ch)``, with optional batch dimension.

        Returns:
            ops.Tensor: Beamformed image of shape ``(n_pix, n_ch)``,
                with optional batch dimension.
        """
        # Coherence-factor power ratios are accumulated in float32 for stability;
        # under a mixed-precision policy the aligned data arrives as bfloat16. Only
        # touch low-precision input: a custom pipeline may already hand this op
        # float32/float64 (or complex) data.
        if keras.backend.standardize_dtype(data.dtype) in LOW_PRECISION_DTYPES:
            data = ops.cast(data, "float32")

        n_el = ops.shape(data)[-2]

        # DAS per transmit: sum over elements
        das_per_tx = ops.sum(data, axis=-2)

        # Coherent power: |sum_i(x_i)|^2, works for both RF (n_ch=1) and IQ (n_ch=2)
        coherent_power = ops.sum(ops.square(das_per_tx), axis=-1, keepdims=True)

        # Incoherent power: N * sum_i(|x_i|^2)
        incoherent_power = n_el * ops.sum(
            ops.sum(ops.square(data), axis=-1), axis=-1, keepdims=True
        )

        # CF weight, clipped to [0, 1] by construction when incoherent_power > 0
        cf_weight = coherent_power / (incoherent_power + keras.backend.epsilon())

        # Apply weight per transmit, then compound
        return ops.sum(das_per_tx * cf_weight, axis=-3)

    def call(self, **kwargs):
        """Performs CF beamforming on tof-corrected input.

        Args:
            tof_corrected_data (ops.Tensor): TOF-corrected input of shape
                ``(n_tx, n_pix, n_el, n_ch)``, with optional batch dimension.

        Returns:
            dict: Dictionary containing beamformed data of shape
                ``(n_pix, n_ch)``, with optional batch dimension.
        """
        data = kwargs[self.key]
        return {self.output_key: self.process_image(data)}


@beamformer_registry("generalized_coherence_factor")
@ops_registry("generalized_coherence_factor")
class GeneralizedCoherenceFactor(Operation):
    r"""Generalized Coherence Factor (GCF) Beamformer.

    The GCF is a coherence-based adaptive weighting technique used to improve
    the quality of ultrasound images by suppressing sidelobes and clutter.
    It is defined as the ratio of the power within a low-frequency region of the
    spatial spectrum to the total power across the aperture.

    For a given pixel, let :math:`A(k)` be the spatial Fourier transform of the
    delayed channel data across :math:`N` elements. The GCF is:

    .. math::

        \mathrm{GCF} = \frac{\sum_{k \in \mathcal{M}_0} \left|A(k)\right|^2}
        {\sum_{k=0}^{N-1} \left|A(k)\right|^2}

    where :math:`\mathcal{M}_0 = \{k : k \leq m_0\} \cup \{k : k \geq N - m_0\}`
    is the low spatial-frequency region controlled by :math:`m_0`.

    .. admonition:: Reference

        Li, P. C., & Li, M. L. (2003).
        "Adaptive imaging using the generalized coherence factor."
        IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control,
        50(2), 128-141.

    Args:
        m_zero (int): Cutoff frequency index for the low-frequency spatial region.
            Higher values increase tolerance to phase aberrations. Defaults to ``4``.
        **kwargs: Additional arguments passed to the Operation base class.
    """

    def __init__(self, m_zero=4, **kwargs):
        if not isinstance(m_zero, int) or m_zero < 0:
            raise ValueError(f"m_zero must be a non-negative integer, got {m_zero!r}.")
        super().__init__(
            input_data_type=DataTypes.ALIGNED_DATA,
            output_data_type=DataTypes.BEAMFORMED_DATA,
            **kwargs,
        )
        self.m_zero = m_zero

    def process_image(self, data, m_zero=None):
        """Applies GCF weighting and compounding on tof-corrected input.

        Args:
            data (ops.Tensor): TOF-corrected input of shape
                ``(n_tx, n_pix, n_el, n_ch)``, with optional batch dimension.
            m_zero (int, optional): Overrides the instance ``m_zero`` for this call.

        Returns:
            ops.Tensor: Beamformed image of shape ``(n_pix, n_ch)``,
                with optional batch dimension.
        """
        if m_zero is None:
            m_zero = self.m_zero

        # GCF uses a spatial FFT and power ratios; run it in float32 (bfloat16 FFT
        # is unsupported / inaccurate). Under a mixed-precision policy the aligned
        # data arrives as bfloat16, so up-cast at entry. Only touch low-precision
        # input: a custom pipeline may already hand this op float32/float64 data.
        if keras.backend.standardize_dtype(data.dtype) in LOW_PRECISION_DTYPES:
            data = ops.cast(data, "float32")

        n_el = ops.shape(data)[-2]
        n_ch = data.shape[-1]  # static Python int — safe for branching

        # Move n_el to last axis for spatial FFT: (..., n_tx, n_pix, n_ch, n_el)
        spatial_data = ops.moveaxis(data, -2, -1)

        real_in = spatial_data[..., 0, :]
        imag_in = spatial_data[..., 1, :] if n_ch == 2 else ops.zeros_like(real_in)

        # Spatial FFT power spectrum across elements
        real_fft, imag_fft = ops.fft((real_in, imag_in))
        power_spectrum = ops.square(real_fft) + ops.square(imag_fft)

        # Total energy and low-frequency energy
        total_energy = ops.sum(power_spectrum, axis=-1, keepdims=True)
        indices = ops.arange(n_el)
        low_freq_mask = ops.logical_or(
            ops.less_equal(indices, m_zero),
            ops.greater_equal(indices, n_el - m_zero),
        )
        low_freq_energy = ops.sum(
            ops.where(low_freq_mask, power_spectrum, 0.0),
            axis=-1,
            keepdims=True,
        )

        # GCF weight
        gcf_weight = low_freq_energy / (total_energy + keras.backend.epsilon())

        # DAS per transmit, apply weight, then compound
        das_per_tx = ops.sum(data, axis=-2)
        return ops.sum(das_per_tx * gcf_weight, axis=-3)

    def call(self, m_zero=None, **kwargs):
        """Performs GCF beamforming on tof-corrected input.

        Args:
            m_zero (int, optional): Cutoff frequency index, overrides the
                instance default when provided via pipeline parameters.
            tof_corrected_data (ops.Tensor): TOF-corrected input of shape
                ``(n_tx, n_pix, n_el, n_ch)``, with optional batch dimension.

        Returns:
            dict: Dictionary containing beamformed data of shape
                ``(n_pix, n_ch)``, with optional batch dimension.
        """
        data = kwargs[self.key]
        return {self.output_key: self.process_image(data, m_zero=m_zero)}


@beamformer_registry("minimum_variance")
@ops_registry("minimum_variance")
class MinimumVariance(Operation):
    r"""Minimum Variance (Capon/MVDR) beamformer.

    Instead of summing the delayed channels with unit weights (delay-and-sum), the
    weights :math:`\mathbf{w}` are chosen per pixel to minimise the output power
    :math:`\mathbf{w}^H \hat{\mathbf{R}} \mathbf{w}` while passing the look direction
    undistorted (:math:`\mathbf{w}^H \mathbf{e} = 1`, with :math:`\mathbf{e} =
    \mathbf{1}`). This adapts the receive aperture to suppress off-axis energy,
    improving lateral resolution and clutter rejection over delay-and-sum.

    The covariance :math:`\hat{\mathbf{R}}` is estimated with spatial smoothing:
    :math:`L = N_{el} - M + 1` overlapping sub-apertures of length :math:`M` are
    averaged, optionally together with :math:`2K + 1` axially adjacent pixels,

    .. math::

        \hat{\mathbf{R}}(p) = \frac{1}{L (2K+1)}
        \sum_{l,k} \mathbf{x}_{l}[p_k]\,\mathbf{x}_{l}^H[p_k],

    diagonally loaded by :math:`\delta\,\mathrm{tr}(\hat{\mathbf{R}})/M` to keep the
    inverse well conditioned. The weights follow in closed form,

    .. math::

        \mathbf{w}(p) = \frac{\hat{\mathbf{R}}_\delta^{-1}\,\mathbf{e}}
        {\mathbf{e}^H\,\hat{\mathbf{R}}_\delta^{-1}\,\mathbf{e}}.

    Each transmit is beamformed with its own weights and the results are summed
    (compounded), as in :class:`DelayAndSum`.

    .. warning::

        Spatial smoothing assumes every sub-aperture sees the full array. Zeroing
        receive channels breaks that: elements masked out by a nonzero ``f_number``
        span a null space of :math:`\hat{\mathbf{R}}` that the loaded inverse fills
        with weight, starving the live channels and darkening the lateral near field
        into a triangular artefact. Set ``parameters.f_number = 0`` and let MV adapt
        the aperture itself.

    .. admonition:: References

        Synnevåg, J.-F., Austeng, A. and Holm, S., "Adaptive beamforming applied to
        medical ultrasound imaging," *IEEE Trans. Ultrason. Ferroelectr. Freq.
        Control* **54** (8), 2007. https://doi.org/10.1109/TUFFC.2007.431

        Vignon, F. and Burcher, M. R., "Capon beamforming in medical ultrasound
        imaging with focused beams," *IEEE Trans. Ultrason. Ferroelectr. Freq.
        Control* **55** (3), 2008. https://doi.org/10.1109/TUFFC.2008.686

    Args:
        subarray_size (int or None): Sub-aperture length :math:`M`. Smaller values are
            more robust (shallow targets need this); ``n_el // 2`` is the maximum
            before the covariance turns singular. Defaults to ``n_el // 2``.
        diagonal_loading (float): Loading :math:`\delta` as a fraction of the mean
            eigenvalue :math:`\mathrm{tr}(\hat{\mathbf{R}})/M`. Larger values are more
            robust and tend toward delay-and-sum. Defaults to ``1e-2``.
        axial_averaging (int): Half-width :math:`K` of the axial covariance averaging
            window, in pixels. Requires a 2D ``grid`` pipeline parameter and enough
            axial pixels per patch. ``0`` disables it. Defaults to ``2``.
        **kwargs: Forwarded to :class:`~zea.ops.base.Operation`.
    """

    def __init__(self, subarray_size=None, diagonal_loading=1e-2, axial_averaging=2, **kwargs):
        if subarray_size is not None and (not isinstance(subarray_size, int) or subarray_size < 1):
            raise ValueError(
                f"subarray_size must be a positive integer or None, got {subarray_size!r}."
            )
        if diagonal_loading < 0:
            raise ValueError(f"diagonal_loading must be non-negative, got {diagonal_loading!r}.")
        if not isinstance(axial_averaging, int) or axial_averaging < 0:
            raise ValueError(
                f"axial_averaging must be a non-negative integer, got {axial_averaging!r}."
            )
        super().__init__(
            input_data_type=DataTypes.ALIGNED_DATA,
            output_data_type=DataTypes.BEAMFORMED_DATA,
            **kwargs,
        )
        self.subarray_size = subarray_size
        self.diagonal_loading = diagonal_loading
        self.axial_averaging = axial_averaging
        self._warned = False

    def _axial_average(self, matrices, stride):
        """Average each pixel's matrix with its ``2K`` axial neighbours.

        Axial neighbours sit ``stride`` apart in the flattened ``(n_z, n_x)`` grid;
        pixels near a patch edge average over fewer of them.
        """
        n_pix = matrices.shape[0]
        width = self.axial_averaging * stride
        padding = [[width, width]] + [[0, 0]] * (len(matrices.shape) - 1)
        padded = ops.pad(matrices, padding)
        weights = ops.pad(ops.ones_like(matrices[:, :1, :1]), padding)

        total = padded[:n_pix]
        count = weights[:n_pix]
        for offset in range(stride, 2 * width + 1, stride):
            total = total + padded[offset : offset + n_pix]
            count = count + weights[offset : offset + n_pix]
        return total / count

    def process_image(self, data, axial_stride=None):
        """Apply MV beamforming to one image.

        Args:
            data (ops.Tensor): TOF-corrected data of shape ``(n_tx, n_pix, n_el, 2)``.
            axial_stride (int, optional): Flat-index distance between axially adjacent
                pixels. Axial averaging is skipped when this is ``None``.

        Returns:
            ops.Tensor: Beamformed image of shape ``(n_pix, 2)``.
        """
        if not data.shape[-1] == 2:
            raise ValueError(
                "MinimumVariance operation requires IQ data with 2 channels. "
                f"Got data with shape {data.shape}."
            )

        n_el = data.shape[-2]
        if self.subarray_size is not None and self.subarray_size > n_el:
            raise ValueError(f"subarray_size ({self.subarray_size}) must not exceed n_el ({n_el}).")

        subarray_length = (
            self.subarray_size if self.subarray_size is not None else max(1, n_el // 2)
        )

        # One transmit at a time keeps only a single (n_pix, M, M) covariance in memory.
        per_transmit = ops.map(
            lambda transmit: self._beamform_transmit(transmit, subarray_length, axial_stride),
            data,
        )
        return ops.sum(per_transmit, axis=0)

    def _beamform_transmit(self, data, subarray_length, axial_stride):
        """Capon-beamform a single transmit of shape ``(n_pix, n_el, 2)``."""
        n_el = data.shape[-2]
        num_subarrays = n_el - subarray_length + 1

        x_c = ops.view_as_complex(data)
        sub_ap = ops.stack(
            [x_c[:, start : start + subarray_length] for start in range(num_subarrays)],
            axis=1,
        )

        # Hermitian sample covariance (n_pix, M, M). The 1/L scaling cancels in the
        # weights; it only keeps magnitudes in float32 range.
        covariance = ops.einsum("pli,plj->pij", sub_ap, ops.conj(sub_ap)) / ops.cast(
            ops.cast(num_subarrays, "float32"), "complex64"
        )
        R_re = ops.real(covariance)
        R_im = ops.imag(covariance)

        if self.axial_averaging > 0 and axial_stride:
            R_re = self._axial_average(R_re, axial_stride)
            R_im = self._axial_average(R_im, axial_stride)

        # Loading relative to the mean eigenvalue tr(R)/M, so it is independent of M.
        # The trace is clamped so all-zero pixels outside the image stay invertible.
        trace = ops.maximum(ops.einsum("...ii->...", R_re), keras.backend.epsilon())[
            ..., None, None
        ]
        R_re = R_re + self.diagonal_loading * trace / subarray_length * ops.eye(subarray_length)

        # linalg.solve has no complex kernel under TF/XLA, so the Hermitian system
        # (R_re + i R_im)(u + i v) = e is recast as the real 2M x 2M system
        # [[R_re, -R_im], [R_im, R_re]] [u; v] = [e; 0].
        block = ops.concatenate(
            [
                ops.concatenate([R_re, -R_im], axis=-1),
                ops.concatenate([R_im, R_re], axis=-1),
            ],
            axis=-2,
        )
        rhs = ops.concatenate(
            [ops.ones_like(R_re[..., :1]), ops.zeros_like(R_re[..., :1])], axis=-2
        )
        solution = ops.linalg.solve(block, rhs)[..., 0]
        R_inv_e = ops.view_as_complex(
            ops.stack([solution[..., :subarray_length], solution[..., subarray_length:]], axis=-1)
        )

        # w = R^-1 e / (e^H R^-1 e). The denominator is real and positive (R is loaded
        # to positive definite); no epsilon guard, which would break scale invariance.
        capon_weights = R_inv_e / ops.sum(R_inv_e, axis=-1, keepdims=True)

        y = ops.einsum("pm,plm->p", ops.conj(capon_weights), sub_ap) / ops.cast(
            num_subarrays, "complex64"
        )
        return ops.view_as_real(y)

    def _resolve_axial_stride(self, grid, n_pix, f_number):
        """Distance between axially adjacent pixels in the flattened grid."""
        if f_number and not self._warned:
            log.warning(
                "MinimumVariance is used with f_number="
                f"{f_number}. The f-number mask zeroes receive channels, which the "
                "Capon inverse fills with weight, suppressing the lateral near field. "
                "Set parameters.f_number = 0 to let MV adapt the aperture itself."
            )
        if self.axial_averaging == 0:
            return None

        stride = None
        if grid is not None and len(grid.shape) == 3:
            stride = grid.shape[1]
        if not self._warned:
            if stride is None:
                log.warning(
                    "MinimumVariance axial_averaging needs a 2D `grid` parameter to "
                    "locate axially adjacent pixels; averaging is disabled."
                )
            elif n_pix < (2 * self.axial_averaging + 1) * stride:
                log.warning(
                    f"MinimumVariance axial_averaging={self.axial_averaging} needs "
                    f"{2 * self.axial_averaging + 1} axial pixels per patch, but a patch "
                    f"holds only {n_pix // stride}. Lower `num_patches` to use the full "
                    "averaging window."
                )
        self._warned = True
        return stride

    def call(self, grid=None, f_number=None, **kwargs):
        """Apply MV beamforming to TOF-corrected data."""
        data = kwargs[self.key]
        n_pix = data.shape[-3]
        stride = self._resolve_axial_stride(grid, n_pix, f_number)

        if not self.with_batch_dim:
            beamformed_data = self.process_image(data, stride)
        else:
            beamformed_data = ops.map(lambda image: self.process_image(image, stride), data)
        return {self.output_key: beamformed_data}


@ops_registry("refocus")
class Refocus(Operation):
    r"""REFoCUS (Retrospective Encoding For Conventional Ultrasound Sequences).

    Decodes any transmit data into synthetic aperture
    (multistatic / full-matrix capture) data by inverting the transmit
    encoding model in the frequency domain.

    The transmit encoding is modelled as a matrix :math:`H` whose entry
    :math:`H_{t,e}` describes the complex phase shift applied to element
    :math:`e` during transmit event :math:`t`:

    .. math::

        H_{t,e}(f) = a_{t,e} \exp(-j 2\pi f \tau_{t,e})

    where :math:`\tau_{t,e}` is the transmit delay in samples and
    :math:`a_{t,e}` is the apodization.

    At each temporal frequency the received RF spectrum is decoded by
    multiplying with the pseudo-inverse :math:`H^{-1}`:

    .. math::

        \hat{U}(f) = H^{-1}(f) \, S(f)

    producing a synthetic aperture dataset where each decoded channel
    corresponds to a virtual single-element transmission.

    The **input** data has shape ``(n_tx, n_ax, n_el, n_ch)`` and the
    **output** has shape ``(n_el, n_ax, n_el, n_ch)``, where the new first
    axis indexes the decoded virtual transmit elements.

    .. admonition:: References

        Bottenus, N. (2018).
        "Recovery of the complete data set from focused transmit beams."
        *IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency
        Control*, 65(1), 30–38.

        Ali, R., Dahl, J., & Bottenus, N. (2019).
        "Extending Retrospective Encoding for Robust Recovery of the Multistatic Dataset."
        *IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency
        Control*, 67(5), 943–956.

        https://github.com/nbottenus/REFoCUS

    Args:
        method (str): Inversion method. One of:

            - ``'adjoint'``: Adjoint (matched-filter) pseudo-inverse with
              an optional ramp filter in frequency. Default.
            - ``'tikhonov'``: Tikhonov-regularized inverse.
            - ``'rsvd'``: Regularized SVD-based inverse.
            - ``'tsvd'``: Truncated SVD-based inverse.

        param (float or None): Regularization / filter parameter.

            - ``'adjoint'``: ``None`` applies a ramp filter (multiply by
              :math:`f`). Set to ``0`` to disable the ramp filter. Defaults to ``None``.
            - ``'tikhonov'``, ``'rsvd'``, ``'tsvd'``: Relative regularization
              strength. Defaults to ``1e-2`` when ``None``.

        **kwargs: Additional arguments forwarded to
            :class:`~zea.ops.Operation`.
    """

    _VALID_METHODS = ("adjoint", "tikhonov", "rsvd", "tsvd")

    def __init__(self, method="adjoint", param=None, **kwargs):
        if method not in self._VALID_METHODS:
            raise ValueError(f"method must be one of {self._VALID_METHODS}, got '{method}'")
        # SVD is not supported by TF XLA, so SVD-based methods cannot be JIT-compiled
        if method != "adjoint":
            if kwargs.get("jittable", True):
                log.warning(
                    f"Refocus method='{method}' uses SVD, which is not supported by the XLA "
                    "JIT compiler. Marking this operation as non-jittable."
                )
            kwargs["jittable"] = False
        super().__init__(
            input_data_type=DataTypes.RAW_DATA,
            output_data_type=DataTypes.RAW_DATA,
            **kwargs,
        )
        self.method = method
        self.param = param

    def _get_hinv(self, delays, f_vec, apod):
        """Compute batched Hinv for all normalised frequencies at once.

        Args:
            delays: ``(n_tx, n_el)`` delays in samples.
            f_vec: ``(n_freq,)`` normalised frequencies (cycles/sample).
            apod: ``(n_tx, n_el)`` apodization.

        Returns:
            Hinv: ``(n_freq, n_el, n_tx)`` complex64 tensor.
        """
        # H: (n_freq, n_tx, n_el)
        f_c = ops.cast(f_vec[:, None, None], "complex64")
        d_c = ops.cast(delays[None], "complex64")
        a_c = ops.cast(apod[None], "complex64")
        H = a_c * ops.exp(ops.cast(-1j * 2 * np.pi, "complex64") * f_c * d_c)

        if self.method == "adjoint":
            # param=None  → ramp filter (multiply by f)
            # param=0     → no ramp (multiply by 1, plain adjoint)
            Hinv = ops.conj(ops.transpose(H, (0, 2, 1)))
            ramp_vals = f_vec if self.param is None else ops.ones_like(f_vec)
            ramp = ops.cast(ramp_vals, "complex64")[:, None, None]
            return ramp * Hinv

        # SVD-based methods
        U, s, VH = ops.svd(H, full_matrices=False)
        lam = self.param if self.param is not None else 1e-2

        if self.method in ("tikhonov", "rsvd"):
            sinv = s / (s**2 + (lam * s[:, 0:1]) ** 2)
        else:  # tsvd
            threshold = lam * s[:, 0:1]
            safe_s = ops.where(s >= threshold, s, ops.ones_like(s))
            sinv = ops.where(s >= threshold, 1.0 / safe_s, ops.zeros_like(s))

        VHT = ops.conj(ops.transpose(VH, (0, 2, 1)))  # (n_freq, n_el, k)
        UT = ops.conj(ops.transpose(U, (0, 2, 1)))  # (n_freq, k, n_tx)
        sinv_c = ops.cast(sinv, "complex64")
        return ops.matmul(VHT * sinv_c[:, None, :], UT)

    def _decode(self, data, delays_samples, apod, demodulation_frequency, sampling_frequency):
        """REFoCUS decoding for a single (unbatched) volume.

        All channels and all frequency bins are processed in parallel via
        batched tensor operations.

        Args:
            data: ``(n_tx, n_ax, n_el, n_ch)`` float32 array.
            delays_samples: ``(n_tx, n_el)`` transmit delays in samples.
            apod: ``(n_tx, n_el)`` transmit apodization.

        Returns:
            decoded: ``(n_el, n_ax, n_el, n_ch)`` float32 array.
        """
        n_tx, n_ax, n_el, n_ch = data.shape
        n_elements = delays_samples.shape[1]
        #Refocus for RF
        if n_ch ==1:
            # --- FFT over all channels at once ---
            # data: (n_tx, n_ax, n_el, n_ch) -> (n_ch, n_el, n_tx, n_ax)
            rf = ops.cast(ops.transpose(data, (3, 2, 0, 1)), "float32")
            # (n_ch, n_el_recv, n_tx, n_freq)
            RF_enc_r, RF_enc_i = ops.rfft(rf)
            RF_enc = ops.cast(RF_enc_r, "complex64") + 1j * ops.cast(RF_enc_i, "complex64")
            n_freq = RF_enc.shape[-1]

            # Rearrange to (n_freq, n_tx, n_el_recv * n_ch) for batched matmul.
            # (n_ch, n_el_recv, n_tx, n_freq) -> (n_freq, n_tx, n_el_recv, n_ch)
            RF_enc = ops.transpose(RF_enc, (3, 2, 1, 0))
            # -> (n_freq, n_tx, n_el_recv * n_ch)
            RF_enc = ops.reshape(RF_enc, (n_freq, n_tx, n_el * n_ch))

            # --- Batched inverse encoding matrices (skip DC at index 0) ---
            frequency = ops.cast(ops.arange(n_freq), "float32") / n_ax
            freq_noDC = frequency[1:]  # (n_freq - 1,)
            # Hinv: (n_freq - 1, n_elements, n_tx)
            Hinv = self._get_hinv(delays_samples, freq_noDC, apod)

            # --- Single batched matmul over all frequencies and channels ---
            # (n_freq-1, n_elements, n_tx) @ (n_freq-1, n_tx, n_el_recv * n_ch)
            # -> (n_freq-1, n_elements, n_el_recv * n_ch)
            RF_dec = ops.matmul(Hinv, RF_enc[1:])

            # Prepend zeros for the DC bin: (n_freq, n_elements, n_el_recv * n_ch)
            dc = ops.zeros((1, n_elements, n_el * n_ch), dtype="complex64")
            RF_decoded = ops.concatenate([dc, RF_dec], axis=0)

            # --- IFFT back to time domain ---
            # Reshape to (n_freq, n_elements, n_el_recv, n_ch)
            RF_decoded = ops.reshape(RF_decoded, (n_freq, n_elements, n_el, n_ch))
            # irfft acts on the last axis: move n_freq last
            # -> (n_elements, n_el_recv, n_ch, n_freq)
            RF_decoded = ops.transpose(RF_decoded, (1, 2, 3, 0))
            # -> (n_elements, n_el_recv, n_ch, n_ax)
            rf_decoded = ops.irfft((ops.real(RF_decoded), ops.imag(RF_decoded)), fft_length=n_ax)
            # -> (n_elements, n_ax, n_el_recv, n_ch)
            rf_decoded = ops.transpose(rf_decoded, (0, 3, 1, 2))

            return ops.cast(rf_decoded, "float32")
        #Refocus for IQ
        elif n_ch == 2:
            # --- FFT over all channels at once ---
            # data: (n_tx, n_ax, n_el, n_ch) -> (n_ch, n_el, n_tx, n_ax)
            iq = ops.cast(ops.transpose(data, (3, 2, 0, 1)), "float32")
            # (n_ch, n_el_recv, n_tx, n_freq)
            IQ_enc_r, IQ_enc_i = ops.fft((iq[0,:,:,:], iq[1,:,:,:]))
            IQ_enc = ops.cast(IQ_enc_r, "complex64") + 1j * ops.cast(IQ_enc_i, "complex64")
            n_freq = IQ_enc.shape[-1]

            # Rearrange to (n_freq, n_tx, n_el_recv) for batched matmul.
            # (n_el_recv, n_tx, n_freq) -> (n_freq, n_tx, n_el_recv)
            IQ_enc = ops.transpose(IQ_enc, (2, 1, 0))
             # --- Batched inverse encoding matrices ---
            # FFT frequencies contain positive and negative frequencies for IQ
            k = ops.arange(n_ax)
            frequency = ops.where(
                k <= n_ax // 2,
                ops.cast(k, "float32") / n_ax,
                ops.cast(k - n_ax, "float32") / n_ax,
            )
            frequency = frequency + demodulation_frequency / sampling_frequency # relative to baseband

            # Hinv: (n_freq, n_elements, n_tx)
            Hinv = self._get_hinv(delays_samples, frequency, apod)

            # --- Single batched matmul over all frequencies and channels ---
            # (n_freq, n_elements, n_tx) @ (n_freq, n_tx, n_el_recv * n_ch)
            # -> (n_freq, n_elements, n_el_recv * n_ch)
            IQ_decoded = ops.matmul(Hinv, IQ_enc)
            # --- IFFT back to time domain ---
            # (n_freq, n_elements, n_el_recv)
    
            # -> (n_elements, n_el_recv, n_freq)
            IQ_decoded = ops.transpose(IQ_decoded, (1, 2, 0))
            # Use `ifft2` with a dummy axis so the inverse transform still
            # applies along the frequency axis while preserving the 1D layout.
            # We do this because keras does not supoort ifft 
            # -> (n_elements, n_el_recv, 1, n_freq)
            iq_decoded_real = ops.expand_dims(ops.real(IQ_decoded), axis=-2)
            iq_decoded_imag = ops.expand_dims(ops.imag(IQ_decoded), axis=-2)
            # -> (n_elements, n_el_recv, 1, n_ax)
            iq_decoded_r, iq_decoded_i = ops.ifft2((iq_decoded_real, iq_decoded_imag))
            # -> (n_elements, n_el_recv, n_ax)
            iq_decoded_r = ops.squeeze(iq_decoded_r, axis=-2)
            iq_decoded_i = ops.squeeze(iq_decoded_i, axis=-2)

            # Recreate the channel dimension.
            # -> (n_elements, n_el_recv, n_ax, n_ch)
            iq_decoded = ops.stack((iq_decoded_r, iq_decoded_i), axis=-1)

            # -> (n_elements, n_ax, n_el_recv, n_ch)
            iq_decoded = ops.transpose(iq_decoded, (0, 2, 1, 3))
            return ops.cast(iq_decoded, "float32")
    # ------------------------------------------------------------------
    # Operation interface
    # ------------------------------------------------------------------

    def call(
        self,
        t0_delays,
        sampling_frequency,
        demodulation_frequency,
        probe_geometry,
        initial_times,
        tx_apodizations=None,
        **kwargs,
    ):
        """Decode plane-wave / focused transmit data into multistatic data.

        After decoding the output is a synthetic-aperture (SA) dataset where
        each virtual transmit corresponds to a single element firing.  The
        pipeline parameters that describe the transmit sequence are updated
        accordingly so that downstream operations (TOF correction, pfield
        weighting, etc.) remain consistent with the new data shape.

        Args:
            t0_delays: ``(n_tx, n_el)`` transmit delays in **seconds**.
            sampling_frequency: Sampling frequency in Hz.
            probe_geometry: ``(n_el, 3)`` element positions in metres.
            tx_apodizations: ``(n_tx, n_el)`` transmit apodization weights.
                Defaults to all-ones (uniform apodization).
            **kwargs: Must contain the input data tensor under ``self.key``.

        Returns:
            dict with keys:

            * ``self.output_key`` — decoded data ``(n_el, n_ax, n_el, n_ch)``
              (or batched variant).
            * ``"t0_delays"`` — zeros ``(n_el, n_el)`` (SA: no extra delay).
            * ``"tx_apodizations"`` — identity ``(n_el, n_el)`` (one element
              per virtual transmit).
            * ``"polar_angles"`` — zeros ``(n_el,)`` (no steering).
            * ``"focus_distances"`` — zeros ``(n_el,)`` (no focus).
            * ``"transmit_origins"`` — element positions ``(n_el, 3)``.
            * ``"initial_times"`` — zeros ``(n_el,)``.
            * ``"t_peak"`` — shared transmit-waveform peak time ``(n_el,)``.
            * ``"flat_pfield"`` — ``None`` (resets pfield so downstream
              :class:`PfieldWeighting` becomes a no-op).
            * ``"flat_aligned_apodization"`` — ``None`` (resets the compounding
              apodization mask, which was sized for the old transmit count, so
              downstream :class:`AlignedApodization` becomes a no-op).
            * ``"flat_receive_apodization"`` — ``None`` (resets any custom
              receive-aperture apodization, which was sized for the old grid, so
              downstream :class:`ReceiveApodization` becomes a no-op).
        """
        data = kwargs[self.key]

        delays_samples = (t0_delays - initial_times[..., None]) * ops.cast(
            sampling_frequency, t0_delays.dtype
        )

        if tx_apodizations is None:
            apod = ops.ones_like(delays_samples)
        else:
            apod = tx_apodizations

        if self.with_batch_dim:
            decoded = vmap(self._decode, in_axes=[0, None, None,None,None])(data, delays_samples, apod, demodulation_frequency, sampling_frequency)
        else:
            decoded = self._decode(data, delays_samples, apod, demodulation_frequency, sampling_frequency)

        # Number of virtual SA transmits = number of elements
        n_el = ops.shape(probe_geometry)[0]
        dtype = t0_delays.dtype

        sa_t0_delays = ops.zeros((n_el, n_el), dtype=dtype)
        sa_tx_apodizations = ops.eye(n_el, dtype=dtype)
        sa_polar_angles = ops.zeros((n_el,), dtype=dtype)
        sa_focus_distances = ops.zeros((n_el,), dtype=dtype)
        sa_initial_times = ops.zeros((n_el,), dtype=dtype)

        t_peak = kwargs.get("t_peak")
        if t_peak is not None:
            t_peak_flat = ops.reshape(ops.cast(t_peak, dtype), (-1,))
            sa_t_peak = ops.broadcast_to(t_peak_flat[:1], (n_el,))
        else:
            sa_t_peak = ops.zeros((n_el,), dtype=dtype)

        return {
            self.output_key: decoded,
            "t0_delays": sa_t0_delays,
            "tx_apodizations": sa_tx_apodizations,
            "polar_angles": sa_polar_angles,
            "focus_distances": sa_focus_distances,
            "transmit_origins": probe_geometry,
            "initial_times": sa_initial_times,
            "t_peak": sa_t_peak,
            "flat_pfield": None,
            "flat_aligned_apodization": None,
            "flat_receive_apodization": None,
        }


def make_operation_chain(
    operation_chain: List[Union[str, Dict, Config, Operation, "Pipeline"]],
) -> List[Union[Operation, "Pipeline"]]:
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

        if not isinstance(operation, (str, dict, Config)):
            raise TypeError(
                f"Operation {operation} should be a string, dict, Config object, Operation, "
                "or Pipeline"
            )

        if isinstance(operation, str):
            operation_instance = get_ops(operation)()

        else:
            if isinstance(operation, Config):
                operation = operation.serialize()

            params = operation.get("params", {})
            op_name = operation.get("name")
            if op_name is None:
                raise ValueError(f"Operation dict is missing a 'name' key: {operation}")
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
