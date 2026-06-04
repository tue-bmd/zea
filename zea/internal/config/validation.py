"""Validate configuration dictionaries.

Config validation follows the same dataclass-Spec pattern used elsewhere in zea
for :class:`~zea.data.spec.ProbeSpec` / :class:`~zea.data.spec.ScanSpec`: each
config section is a :func:`dataclasses.dataclass` with typed fields, default
values, and validation in ``__post_init__``.

Unlike the array Specs in :mod:`zea.data.spec` (which validate numpy
``dtype``/``shape`` and named-dimension consistency), config values are plain
Python scalars / lists / dicts, so validation here uses small validator
callables (enums, numeric ranges, regexes).

The ``parameters`` section and the top-level config are *open*: they accept
arbitrary extra keys, which are stored and re-emitted unchanged.  This mirrors
:class:`zea.Parameters`, which keeps unknown keys as pass-through
``_custom_params``.  When adding a documented parameter, add a field (with a
default if optional) to the relevant ``*Config`` dataclass below.
"""

import re
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional, Type

from zea.internal.checks import _DATA_TYPES
from zea.metrics import metrics_registry

_ALLOWED_PLOT_LIBS = ("opencv", "matplotlib")


# ---------------------------------------------------------------------------
# Validator helpers
#
# Each validator is a ``Callable[[Any], Any]`` that returns the (possibly
# coerced) value or raises ``ValueError`` with a human-readable message.  These
# replace the ``schema`` library primitives (``And`` / ``Or`` / ``Regex`` /
# lambdas) that were previously used.
# ---------------------------------------------------------------------------


def boolean(value: Any) -> bool:
    """Validate a boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"must be a boolean, got {type(value).__name__}")
    return value


def string(value: Any) -> str:
    """Validate a string."""
    if not isinstance(value, str):
        raise ValueError(f"must be a string, got {type(value).__name__}")
    return value


def integer(value: Any) -> int:
    """Validate an integer (``bool`` is rejected)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"must be an integer, got {type(value).__name__}")
    return value


def any_number(value: Any) -> Any:
    """Validate a number (``int`` or ``float``, ``bool`` is rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "must be a number, scientific notation should be of form x.xe+xx, "
            "otherwise interpreted as string"
        )
    return value


def positive_integer(value: Any) -> int:
    """Validate a strictly positive integer."""
    integer(value)
    if value <= 0:
        raise ValueError(f"must be a positive integer, got {value}")
    return value


def positive_integer_and_zero(value: Any) -> int:
    """Validate a non-negative integer."""
    integer(value)
    if value < 0:
        raise ValueError(f"must be a non-negative integer, got {value}")
    return value


def positive_float(value: Any) -> float:
    """Validate a strictly positive float."""
    if isinstance(value, bool) or not isinstance(value, float):
        raise ValueError(f"must be a float, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"must be a positive float, got {value}")
    return value


def mapping(value: Any) -> dict:
    """Validate a dict (mapping)."""
    if not isinstance(value, dict):
        raise ValueError(f"must be a dict, got {type(value).__name__}")
    return value


def list_of_size_two(value: Any) -> list:
    """Validate a list of exactly two elements."""
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"must be a list of length two, got {value!r}")
    return value


def list_of_positive_integers(value: Any) -> list:
    """Validate a list of non-negative integers."""
    if not isinstance(value, list) or not all(
        isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in value
    ):
        raise ValueError(f"must be a list of non-negative integers, got {value!r}")
    return value


def string_or_path(value: Any) -> Any:
    """Validate a string or :class:`pathlib.Path`."""
    if not isinstance(value, (str, Path)):
        raise ValueError(f"must be a string or path, got {type(value).__name__}")
    return value


def enum(*allowed: Any) -> Callable[[Any], Any]:
    """Build a validator that accepts only one of ``allowed`` values."""

    def validate(value: Any) -> Any:
        if value not in allowed:
            raise ValueError(f"must be one of {list(allowed)}, got {value!r}")
        return value

    return validate


def regex(pattern: str) -> Callable[[Any], str]:
    """Build a validator that fully matches ``pattern``."""
    compiled = re.compile(pattern)

    def validate(value: Any) -> str:
        if not isinstance(value, str) or compiled.fullmatch(value) is None:
            raise ValueError(f"must match pattern {pattern!r}, got {value!r}")
        return value

    return validate


def any_of(*validators: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Build a validator that passes if any of ``validators`` passes."""

    def validate(value: Any) -> Any:
        errors = []
        for validator in validators:
            try:
                return validator(value)
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError(" or ".join(errors))

    return validate


def optional(validator: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Build a validator that also accepts ``None``."""

    def validate(value: Any) -> Any:
        if value is None:
            return None
        return validator(value)

    return validate


def operations_list(value: Any) -> list:
    """Validate the pipeline ``operations`` list.

    Each element is either an operation name (str) or a mapping with a ``name``
    (str) and optional ``params`` (dict).
    """
    if not isinstance(value, list):
        raise ValueError(f"must be a list of operations, got {type(value).__name__}")
    for op in value:
        if isinstance(op, str):
            continue
        if isinstance(op, dict):
            if not isinstance(op.get("name"), str):
                raise ValueError(f"operation {op!r} must have a string 'name'")
            unexpected = set(op) - {"name", "params"}
            if unexpected:
                raise ValueError(f"operation {op!r} has unexpected keys {sorted(unexpected)}")
            if "params" in op and not isinstance(op["params"], dict):
                raise ValueError(f"operation {op!r} 'params' must be a dict")
            continue
        raise ValueError(f"invalid operation {op!r}")
    return value


# ---------------------------------------------------------------------------
# Config Spec base class
# ---------------------------------------------------------------------------


@dataclass
class ConfigSpec:
    """Base class for config sections.

    Subclasses are dataclasses that declare their fields (with defaults for
    optional fields) and the following class variables:

    - ``VALIDATORS``: maps a field name to a validator callable.
    - ``NESTED``: maps a field name to a nested :class:`ConfigSpec` subclass.
    - ``ALLOW_EXTRA``: when ``True`` arbitrary extra keys are accepted and
      passed through unchanged (used for the open ``parameters`` section and the
      top-level config).
    """

    VALIDATORS: ClassVar[dict[str, Callable[[Any], Any]]] = {}
    NESTED: ClassVar[dict[str, Type["ConfigSpec"]]] = {}
    ALLOW_EXTRA: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if not hasattr(self, "_extra"):
            self._extra: dict[str, Any] = {}
        for name in self.field_names():
            value = getattr(self, name)
            nested = self.NESTED.get(name)
            if nested is not None:
                setattr(self, name, self._coerce_nested(name, nested, value))
                continue
            validator = self.VALIDATORS.get(name)
            if validator is not None:
                try:
                    value = validator(value)
                except ValueError as exc:
                    raise ValueError(f"{type(self).__name__}.{name}: {exc}") from exc
                setattr(self, name, value)

    def _coerce_nested(self, name: str, nested: Type["ConfigSpec"], value: Any) -> "ConfigSpec":
        if value is None:
            # Optional nested section: fall back to its defaults.
            return nested.from_dict({})
        if isinstance(value, nested):
            return value
        if isinstance(value, dict):
            try:
                return nested.from_dict(value)
            except ValueError as exc:
                raise ValueError(f"{type(self).__name__}.{name}: {exc}") from exc
        raise ValueError(
            f"{type(self).__name__}.{name}: expected a mapping for "
            f"{nested.__name__}, got {type(value).__name__}"
        )

    # -- construction / serialization --------------------------------------

    @classmethod
    def from_dict(cls, dictionary: Optional[dict]) -> "ConfigSpec":
        """Validate ``dictionary`` and return a populated spec instance."""
        if dictionary is None:
            dictionary = {}
        if not isinstance(dictionary, dict):
            raise ValueError(f"{cls.__name__}: expected a mapping, got {type(dictionary).__name__}")

        field_names = set(cls.field_names())
        known = {k: v for k, v in dictionary.items() if k in field_names}
        extra = {k: v for k, v in dictionary.items() if k not in field_names}

        if extra and not cls.ALLOW_EXTRA:
            raise ValueError(f"{cls.__name__}: unexpected keys {sorted(extra)}")

        missing = [name for name in cls.required_fields() if name not in known]
        if missing:
            raise ValueError(f"{cls.__name__}: missing required keys {missing}")

        obj = cls(**known)
        if extra:
            obj._extra.update(extra)
        return obj

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict with defaults filled and nested specs expanded."""
        result: dict[str, Any] = {}
        for name in self.field_names():
            value = getattr(self, name)
            if isinstance(value, ConfigSpec):
                value = value.to_dict()
            result[name] = value
        result.update(self._extra)
        return result

    # -- introspection (used by tooling / docs) ----------------------------

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Return the names of all declared fields."""
        return tuple(f.name for f in fields(cls))

    @classmethod
    def required_fields(cls) -> tuple[str, ...]:
        """Return the names of fields without a default value."""
        return tuple(
            f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING
        )

    @classmethod
    def optional_fields(cls) -> tuple[str, ...]:
        """Return the names of fields with a default value."""
        required = set(cls.required_fields())
        return tuple(name for name in cls.field_names() if name not in required)

    @classmethod
    def all_field_paths(cls, prefix: str = "") -> set[str]:
        """Return all documented field paths in dot notation, recursing nested specs."""
        paths: set[str] = set()
        for name in cls.field_names():
            full = f"{prefix}.{name}" if prefix else name
            nested = cls.NESTED.get(name)
            if nested is not None:
                paths |= nested.all_field_paths(full)
            else:
                paths.add(full)
        return paths


# ---------------------------------------------------------------------------
# Config sections
# ---------------------------------------------------------------------------


@dataclass
class DataConfig(ConfigSpec):
    """The ``data:`` section: what data to load and how."""

    dtype: Any
    dataset_folder: Any
    resolution: Any = None
    to_dtype: Any = "image"
    file_path: Any = None
    local: Any = True
    frame_no: Any = None
    dynamic_range: Any = field(default_factory=lambda: [-60, 0])
    input_range: Any = None
    output_range: Any = None
    apodization: Any = None
    user: Any = None

    VALIDATORS: ClassVar[dict] = {
        "dtype": enum(*_DATA_TYPES),
        "dataset_folder": string,
        "resolution": optional(positive_float),
        "to_dtype": enum(*_DATA_TYPES),
        "file_path": optional(string_or_path),
        "local": boolean,
        "frame_no": optional(any_of(enum("all"), integer)),
        "dynamic_range": list_of_size_two,
        "input_range": optional(list_of_size_two),
        "output_range": optional(list_of_size_two),
        "apodization": optional(string),
        "user": optional(mapping),
    }


@dataclass
class PlotConfig(ConfigSpec):
    """The ``plot:`` section: UI / plotting settings."""

    save: Any = False
    plot_lib: Any = "opencv"
    fps: Any = 20
    tag: Any = None
    headless: Any = False
    selector: Any = None
    selector_metric: Any = "gcnr"
    fliplr: Any = False
    image_extension: Any = "png"
    video_extension: Any = "gif"

    VALIDATORS: ClassVar[dict] = {
        "save": boolean,
        "plot_lib": enum(*_ALLOWED_PLOT_LIBS),
        "fps": integer,
        "tag": optional(string),
        "headless": boolean,
        "selector": optional(enum("rectangle", "lasso")),
        "selector_metric": enum(*metrics_registry.registered_names()),
        "fliplr": boolean,
        "image_extension": enum("png", "jpg"),
        "video_extension": enum("mp4", "gif"),
    }


@dataclass
class PipelineConfig(ConfigSpec):
    """The ``pipeline:`` section: operations and JIT settings."""

    operations: Any = field(default_factory=lambda: ["identity"])
    with_batch_dim: Any = True
    jit_options: Any = "ops"
    jit_kwargs: Any = None
    name: Any = "pipeline"
    validate: Any = True

    VALIDATORS: ClassVar[dict] = {
        "operations": optional(operations_list),
        "with_batch_dim": boolean,
        "jit_options": optional(enum("ops", "pipeline")),
        "jit_kwargs": optional(mapping),
        "name": string,
        "validate": boolean,
    }


@dataclass
class ParametersConfig(ConfigSpec):
    """The ``parameters:`` section: flat scan/probe/custom parameters.

    This section is *open*: documented reconstruction parameters are validated,
    and arbitrary custom keys are accepted and passed through unchanged
    (consistent with :class:`zea.Parameters`).
    """

    xlims: Any = None
    zlims: Any = None
    ylims: Any = None
    selected_transmits: Any = None
    grid_size_x: Any = None
    grid_size_z: Any = None
    n_ch: Any = None
    n_ax: Any = None
    center_frequency: Any = None
    sampling_frequency: Any = None
    demodulation_frequency: Any = None
    f_number: Any = None
    apply_lens_correction: Any = False
    lens_thickness: Any = 1e-3
    lens_sound_speed: Any = 1000
    theta_range: Any = None
    phi_range: Any = None
    rho_range: Any = None
    fill_value: Any = 0.0
    resolution: Any = None

    ALLOW_EXTRA: ClassVar[bool] = True
    VALIDATORS: ClassVar[dict] = {
        "xlims": optional(list_of_size_two),
        "zlims": optional(list_of_size_two),
        "ylims": optional(list_of_size_two),
        "selected_transmits": optional(
            any_of(positive_integer, list_of_positive_integers, enum("all", "center"))
        ),
        "grid_size_x": optional(positive_integer),
        "grid_size_z": optional(positive_integer),
        "n_ch": optional(integer),
        "n_ax": optional(integer),
        "center_frequency": optional(any_number),
        "sampling_frequency": optional(any_number),
        "demodulation_frequency": optional(any_number),
        "f_number": optional(positive_float),
        "apply_lens_correction": boolean,
        "lens_thickness": positive_float,
        "lens_sound_speed": any_of(positive_float, positive_integer),
        "theta_range": optional(list_of_size_two),
        "phi_range": optional(list_of_size_two),
        "rho_range": optional(list_of_size_two),
        "fill_value": any_number,
        "resolution": optional(positive_float),
    }


@dataclass
class ConfigSchema(ConfigSpec):
    """The top-level config.

    This is *open*: arbitrary extra top-level sections (e.g. ``model:``) are
    accepted and passed through unchanged.  The deprecated ``scan:`` section is
    aliased to ``parameters:`` before validation (see
    :func:`zea.config._migrate_legacy_config`).
    """

    data: Any
    plot: Any = None
    pipeline: Any = None
    parameters: Any = None
    device: Any = "auto:1"
    hide_devices: Any = None
    git: Any = None

    ALLOW_EXTRA: ClassVar[bool] = True
    NESTED: ClassVar[dict] = {
        "data": DataConfig,
        "plot": PlotConfig,
        "pipeline": PipelineConfig,
        "parameters": ParametersConfig,
    }
    VALIDATORS: ClassVar[dict] = {
        "device": optional(
            any_of(
                enum("cpu", "gpu", "cuda"),
                regex(r"cuda:\d+"),
                regex(r"gpu:\d+"),
                regex(r"auto:-?\d+"),
            )
        ),
        "hide_devices": optional(any_of(list_of_positive_integers, positive_integer_and_zero)),
        "git": optional(string),
    }


def validate_config(config: Optional[dict]) -> dict:
    """Validate a config dict and return a plain dict with defaults filled in.

    This is the replacement for the previous ``config_schema.validate(...)``.
    """
    return ConfigSchema.from_dict(config).to_dict()
