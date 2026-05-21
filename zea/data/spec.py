from collections import defaultdict
from dataclasses import MISSING, dataclass, field, fields
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_pkg_version
from pathlib import Path
from typing import Any, List, Tuple

import h5py
import numpy as np

from zea import log

CONSISTENCY_DIMENSIONS = {"n_frames", "n_tx", "n_ax", "n_el", "n_ch", "n_spatial_ch"}

UNITS = {
    "m/s": "meters per second",
    "m": "meters",
    "Hz": "Hertz",
    "s": "seconds",
    "-": "unitless",
    "rad": "radians",
    "dB": "decibels",
    "#": "count",
    "%": "percent",
}


# Default unit/description for every SCHEMA leaf field.  Subclasses may
# override by defining their own FIELD_METADATA dict.
_DEFAULT_FIELD_UNIT = "-"
_DEFAULT_FIELD_DESCRIPTION = ""


def check_dtype(value: Any, expected_dtype: List[type]) -> None:
    """Check if the dtype of a value matches the expected dtype,
    allowing for compatible types.

    Works for numpy arrays, numpy scalars, and Python native types.
    """
    for dt in expected_dtype:
        try:
            expected_np_dtype = np.dtype(dt)
            is_numpy_dtype = True
        except TypeError:
            is_numpy_dtype = False

        if is_numpy_dtype:
            if hasattr(value, "dtype"):
                if np.issubdtype(value.dtype, expected_np_dtype):
                    return
            elif np.issubdtype(expected_np_dtype, np.character) and isinstance(value, (str, bytes)):
                return
        else:
            if isinstance(value, dt):
                return

    actual_type = (
        f"dtype {value.dtype}" if hasattr(value, "dtype") else f"Python {type(value).__name__}"
    )
    expected_dtypes_str = ", ".join(str(dt) for dt in expected_dtype)
    raise TypeError(
        f"Expected dtype compatible with one of ({expected_dtypes_str}), got {actual_type}. "
        f"Hint: wrap the value with the appropriate numpy type, "
        f"e.g. np.float32(...), np.str_(...), np.uint8(...)."
    )


def value_shape(value: Any) -> tuple:
    """Return the shape tuple for numpy arrays and scalar values."""
    if isinstance(value, np.ndarray):
        return value.shape
    return ()


def match_shape(value: Any, expected_shape: tuple) -> bool:
    """Check if the shape of a value matches the expected shape specification."""
    shape = value_shape(value)
    ellipsis_positions = [i for i, dim in enumerate(expected_shape) if dim == "..."]

    if len(ellipsis_positions) > 1:
        raise ValueError("Expected shape can contain at most one '...' wildcard")

    if not ellipsis_positions:
        if len(shape) != len(expected_shape):
            return False
        comparisons = zip(shape, expected_shape)
    else:
        ellipsis_pos = ellipsis_positions[0]
        prefix_expected = expected_shape[:ellipsis_pos]
        suffix_expected = expected_shape[ellipsis_pos + 1 :]

        # '...' matches any number of dimensions (including zero).
        min_required_dims = len(prefix_expected) + len(suffix_expected)
        if len(shape) < min_required_dims:
            return False

        prefix_shape = shape[: len(prefix_expected)]
        suffix_shape = shape[len(shape) - len(suffix_expected) :] if suffix_expected else ()
        comparisons = zip(
            prefix_shape + suffix_shape,
            prefix_expected + suffix_expected,
        )

    for dim_size, expected_dim in comparisons:
        if isinstance(expected_dim, str):
            continue
        if dim_size != expected_dim:
            return False

    return True


def find_matched_shape(value: Any, expected_shapes: List[tuple]) -> tuple | None:
    """Find the first expected shape specification that matches the shape of the value."""
    for expected_shape in expected_shapes:
        if match_shape(value, expected_shape):
            return expected_shape
    return None


class Spec:
    """Base class for data specifications with schema validation.

    Subclasses should define a SCHEMA class variable that specifies the expected dtype and shape
    for each field. The __post_init__ method will validate that the actual fields match the schema,
    including checking that dimensions with the same name have consistent sizes across fields.
    """

    SCHEMA: dict

    @staticmethod
    def _is_optional_dataclass_field(field_def: Any) -> bool:
        if field_def is None:
            return False
        return field_def.default is not MISSING or field_def.default_factory is not MISSING

    @classmethod
    def required_fields(cls) -> tuple[str, ...]:
        """Return the names of fields that have no default value."""
        return tuple(f.name for f in fields(cls) if not cls._is_optional_dataclass_field(f))

    @classmethod
    def fields(cls) -> tuple[str, ...]:
        """Return the names of all fields."""
        return tuple(f.name for f in fields(cls))

    @classmethod
    def optional_fields(cls) -> tuple[str, ...]:
        """Return the names of fields that have a default value."""
        return tuple(f.name for f in fields(cls) if cls._is_optional_dataclass_field(f))

    def warn_missing_optional_fields(self):
        """Warn about optional fields that were not provided."""
        _optional_fields = self.optional_fields()
        for field_name in self.SCHEMA.keys():
            if field_name in _optional_fields and getattr(self, field_name) is None:
                if hasattr(self, "FIELD_METADATA"):
                    meta = self.FIELD_METADATA.get(field_name, {})
                    description = meta.get("description", _DEFAULT_FIELD_DESCRIPTION)
                else:
                    description = _DEFAULT_FIELD_DESCRIPTION
                log.warning(
                    f"Optional {self.__class__.__name__} field '{field_name}' is not set. "
                    f"Description: {description} "
                    "Defaulted to None."
                )

    @staticmethod
    def _expected_shapes(shape_spec: Any) -> tuple[tuple, ...]:
        if shape_spec and isinstance(shape_spec[0], tuple):
            return tuple(shape_spec)
        return (shape_spec,)

    @staticmethod
    def _merge_dimension_info(
        dim_to_fields: defaultdict[str, set[str]],
        dim_to_sizes: defaultdict[str, set[int]],
        nested_dim_to_fields: defaultdict[str, set[str]],
        nested_dim_to_sizes: defaultdict[str, set[int]],
    ) -> None:
        for dim_name, nested_fields in nested_dim_to_fields.items():
            dim_to_fields[dim_name].update(nested_fields)
        for dim_name, nested_sizes in nested_dim_to_sizes.items():
            dim_to_sizes[dim_name].update(nested_sizes)

    @staticmethod
    def _track_named_dimensions(
        dim_to_fields: defaultdict[str, set[str]],
        dim_to_sizes: defaultdict[str, set[int]],
        field_path: str,
        matched_shape: tuple,
        shape: tuple,
    ) -> None:
        for i, dim_name in enumerate(matched_shape):
            if isinstance(dim_name, str) and dim_name in CONSISTENCY_DIMENSIONS:
                dim_to_fields[dim_name].add(field_path)
                dim_to_sizes[dim_name].add(shape[i])

    @staticmethod
    def _raise_if_shape_mismatch(
        field_name: str, value: Any, expected_shapes: tuple[tuple, ...]
    ) -> None:
        allowed_shapes = ", ".join(str(shape) for shape in expected_shapes)
        raise ValueError(
            f"{field_name} has shape {value_shape(value)}, expected one of: {allowed_shapes}"
        )

    def _validate_nested_field(
        self, field_name: str, nested_spec: "Spec", field_value: Any
    ) -> "Spec":
        """Validate a nested spec field, recursively validating its contents."""
        if isinstance(field_value, dict):
            field_value = nested_spec(**field_value)
            setattr(self, field_name, field_value)

        # Check that the nested spec field is now an instance of the expected Spec subclass
        # E.g. Segmentation if nested_spec is Map
        if not issubclass(type(field_value), nested_spec):
            raise TypeError(
                f"Expected field '{field_name}' to be {nested_spec}, got {type(field_value)}"
            )

        return field_value

    @staticmethod
    def _cast_native_to_numpy(value: Any, expected_dtype: list) -> Any:
        """Cast values to expected numpy dtypes when possible.

        For fields that expect a floating dtype, all floating-point inputs are
        accepted and normalized to the first floating dtype in ``expected_dtype``
        (typically ``np.float32``).
        """
        expected_np_dtypes = []
        for dt in expected_dtype:
            try:
                expected_np_dtypes.append(np.dtype(dt))
            except TypeError:
                continue

        expected_float_dtype = next(
            (dt for dt in expected_np_dtypes if np.issubdtype(dt, np.floating)),
            None,
        )

        # Keep native string/bytes values as-is instead of converting to numpy string scalars.
        if isinstance(value, (str, bytes)):
            return value

        if hasattr(value, "dtype"):
            value_dtype = np.dtype(value.dtype)

            if (
                expected_float_dtype is not None
                and np.issubdtype(value_dtype, np.floating)
                and value_dtype != expected_float_dtype
            ):
                return value.astype(expected_float_dtype, copy=False)

            return value

        for dt in expected_dtype:
            try:
                target_dtype = np.dtype(dt)
                return target_dtype.type(value)
            except (TypeError, ValueError, OverflowError):
                continue

        return value

    def _validate_and_track_primitive_field(
        self,
        field_name: str,
        field_info: dict,
        field_value: Any,
        dim_to_fields: defaultdict[str, set[str]],
        dim_to_sizes: defaultdict[str, set[int]],
    ) -> None:
        expected_dtype = field_info["dtype"]
        if not isinstance(expected_dtype, (list, tuple)):
            expected_dtype = [expected_dtype]
        expected_shapes = self._expected_shapes(field_info["shape"])

        # Auto-cast Python native types (str, int, float) to numpy equivalents
        field_value = self._cast_native_to_numpy(field_value, expected_dtype)
        setattr(self, field_name, field_value)

        try:
            check_dtype(field_value, expected_dtype)
        except TypeError as e:
            raise TypeError(f"{type(self).__name__}: field '{field_name}' has invalid dtype: {e}")

        matched_shape = find_matched_shape(field_value, expected_shapes)
        if matched_shape is None:
            self._raise_if_shape_mismatch(field_name, field_value, expected_shapes)

        self._track_named_dimensions(
            dim_to_fields=dim_to_fields,
            dim_to_sizes=dim_to_sizes,
            field_path=field_name,
            matched_shape=matched_shape,
            shape=value_shape(field_value),
        )

    @staticmethod
    def _raise_if_inconsistent_dimensions(
        dim_to_fields: defaultdict[str, set[str]],
        dim_to_sizes: defaultdict[str, set[int]],
    ) -> None:
        for dim_name, sizes in dim_to_sizes.items():
            if len(sizes) > 1:
                field_names = sorted(dim_to_fields[dim_name])
                raise ValueError(
                    f"Dimension '{dim_name}' has inconsistent sizes across "
                    f"fields {field_names}: {sorted(sizes)}"
                )

    def _collect_dimension_info(
        self, prefix: str = ""
    ) -> tuple[defaultdict[str, set[str]], defaultdict[str, set[int]]]:
        """Collect named dimension usage and observed sizes for this spec subtree."""
        dim_to_fields = defaultdict(set)
        dim_to_sizes = defaultdict(set)

        for field_name, field_info in self.SCHEMA.items():
            field_value = getattr(self, field_name)
            if field_value is None:
                continue

            nested_spec = field_info.get("spec")
            if nested_spec is not None:
                nested_dim_to_fields, nested_dim_to_sizes = field_value._collect_dimension_info(
                    prefix=f"{prefix}{field_name}."
                )
                self._merge_dimension_info(
                    dim_to_fields,
                    dim_to_sizes,
                    nested_dim_to_fields,
                    nested_dim_to_sizes,
                )
                continue

            expected_shapes = self._expected_shapes(field_info["shape"])

            matched_shape = find_matched_shape(field_value, expected_shapes)
            if matched_shape is None:
                # Child specs are already validated; skip defensively if no shape can be matched.
                continue

            self._track_named_dimensions(
                dim_to_fields=dim_to_fields,
                dim_to_sizes=dim_to_sizes,
                field_path=f"{prefix}{field_name}",
                matched_shape=matched_shape,
                shape=value_shape(field_value),
            )

        return dim_to_fields, dim_to_sizes

    def __post_init__(self):
        dim_to_fields = defaultdict(set)
        dim_to_sizes = defaultdict(set)
        dataclass_fields = {f.name: f for f in fields(self)}

        for field_name, field_info in self.SCHEMA.items():
            field_value = getattr(self, field_name)
            field_def = dataclass_fields.get(field_name)
            is_optional = self._is_optional_dataclass_field(field_def)

            if field_value is None:
                if not is_optional:
                    raise ValueError(f"Missing required field '{field_name}'")
                continue

            nested_spec = field_info.get("spec")
            if nested_spec is not None:
                try:
                    field_value = self._validate_nested_field(field_name, nested_spec, field_value)
                except (TypeError, ValueError) as e:
                    raise type(e)(f"In field '{field_name}': {e}") from e

                nested_dim_to_fields, nested_dim_to_sizes = field_value._collect_dimension_info(
                    prefix=f"{field_name}."
                )
                self._merge_dimension_info(
                    dim_to_fields,
                    dim_to_sizes,
                    nested_dim_to_fields,
                    nested_dim_to_sizes,
                )
                continue

            self._validate_and_track_primitive_field(
                field_name=field_name,
                field_info=field_info,
                field_value=field_value,
                dim_to_fields=dim_to_fields,
                dim_to_sizes=dim_to_sizes,
            )

        self._raise_if_inconsistent_dimensions(dim_to_fields, dim_to_sizes)

    @staticmethod
    def _is_string_value(value: Any) -> bool:
        """Return True for scalar/array values that should be stored as HDF5 strings."""
        if isinstance(value, (str, np.str_, bytes, np.bytes_)):
            return True

        if isinstance(value, np.ndarray):
            return value.dtype.kind in {"U", "S", "O"}

        return False

    @staticmethod
    def create_dataset(
        group: h5py.Group, field_name: str, value: Any, compression: str = "gzip"
    ) -> None:
        """Create a dataset in the given group for the specified field and value,
        handling string and scalar values appropriately."""
        dataset_is_scalar = np.isscalar(value) or value.ndim == 0
        compression = None if dataset_is_scalar else compression
        if Spec._is_string_value(value):
            string_dtype = h5py.string_dtype(encoding="utf-8")
            string_value = np.asarray(value, dtype=object)
            group.create_dataset(
                field_name,
                data=string_value,
                dtype=string_dtype,
                compression=compression,
            )
        else:
            group.create_dataset(field_name, data=value, compression=compression)

    def store_in_group(self, group: h5py.Group, compression: str = "gzip") -> None:
        """Store the data in the given group (e.g. hdf5 group)."""

        assert isinstance(group, h5py.Group), "group must be an h5py Group"

        field_metadata = getattr(self, "FIELD_METADATA", {})

        for field_name, field_info in self.SCHEMA.items():
            value = getattr(self, field_name)
            if value is None:
                continue

            nested_spec = field_info.get("spec")
            if nested_spec is not None:
                subgroup = group.create_group(field_name)
                value.store_in_group(subgroup, compression=compression)
            else:
                self.create_dataset(group, field_name, value, compression=compression)
                meta = field_metadata.get(field_name, {})
                group[field_name].attrs["unit"] = meta.get("unit", _DEFAULT_FIELD_UNIT)
                group[field_name].attrs["description"] = meta.get(
                    "description", _DEFAULT_FIELD_DESCRIPTION
                )

    def to_dict(self) -> dict[str, Any]:
        """Return this spec as a nested dictionary based on ``SCHEMA`` fields.

        Nested specs are converted recursively.
        """
        result = {}
        for field_name, field_info in self.SCHEMA.items():
            value = getattr(self, field_name)
            nested_spec = field_info.get("spec")

            if nested_spec is not None and value is not None:
                if isinstance(value, Spec):
                    result[field_name] = value.to_dict()
                elif isinstance(value, dict):
                    result[field_name] = {
                        k: v.to_dict() if isinstance(v, Spec) else v for k, v in value.items()
                    }
                else:
                    result[field_name] = value
            else:
                result[field_name] = value

        return result

    @classmethod
    def get_dtype(cls, field_name) -> Tuple[type, ...] | type:
        """Get the dtype of a field."""
        return cls.SCHEMA[field_name]["dtype"]


@dataclass
class Map(Spec):
    """Map data with per-pixel Cartesian coordinates.

    A map is a function from Cartesian space to some real values: every pixel at
    spatial index ``[f, i, j, ...]`` is assigned a 3-D position ``coordinates[f, i, j, ..., :]``
    = ``[x, y, z]`` in metres.

    The most flexible map spec, which can be used for any spatially aligned data product.
    See, for example, :func:`~zea.beamform.pixelgrid.cartesian_pixel_grid` or
    :func:`~zea.beamform.pixelgrid.polar_pixel_grid` to create a suitable coordinate array
    from your scan geometry.

    Args:
        values: The map values of shape ``(n_frames, z, x, y, n_ch)`` or ``(n_frames, z, x, y)``
            or ``(n_frames, z, x, n_ch)`` or ``(n_frames, z, x)`` and type uint8, float32,
            int16, or complex64.
        coordinates: Per-pixel Cartesian positions in metres, shape ``(*spatial_dims, 3)``
            where ``spatial_dims`` matches the spatial (non-channel) dimensions of ``values``.
            For non-channeled values the shape is ``(*values.shape, 3)``; for channeled values
            the shape is ``(*values.shape[:-1], 3)``.  The last axis holds ``[x, y, z]``.
        labels: The labels corresponding to the ``n_ch`` channels in the values.
            This is required when values have an n_ch dimension, and should be None otherwise.
            For IQ data, this would typically be ``["I", "Q"]``.
        description: An optional free-text description of the map.
        unit: An optional string specifying the physical unit of the map values,
            e.g. ``"m/s"``, ``"%"``, etc.
        min: The minimum value of the map.
        max: The maximum value of the map.
    """

    values: np.ndarray
    coordinates: np.ndarray | None = None
    labels: np.ndarray | None = None
    description: str | None = None
    unit: str | None = None
    min: float | None = None
    max: float | None = None

    SCHEMA = {
        "values": {
            "dtype": (np.uint8, np.float32, np.int16, np.complex64),
            "shape": (
                ("n_frames", "z", "x", "y", "n_spatial_ch"),
                ("n_frames", "z", "x", "y"),
                ("n_frames", "z", "x"),
            ),
        },
        "coordinates": {"dtype": np.float32, "shape": ("...", 3)},
        "labels": {"dtype": np.str_, "shape": ("n_spatial_ch",)},
        "description": {"dtype": str, "shape": ()},
        "unit": {"dtype": str, "shape": ()},
        "min": {"dtype": np.float32, "shape": ()},
        "max": {"dtype": np.float32, "shape": ()},
    }

    def __post_init__(self):
        super().__post_init__()

        if self.values.ndim == 5:
            assert self.labels is not None, (
                "labels must be provided when values have n_ch dimension"
            )

        if self.coordinates is not None:
            # coordinates.shape[-1] is guaranteed == 3 by the SCHEMA check above.
            # Validate that the spatial axes match values (with or without a trailing channel axis).
            coords_spatial = self.coordinates.shape[:-1]
            valid_spatial_shapes = {self.values.shape, self.values.shape[:-1]}
            if coords_spatial not in valid_spatial_shapes:
                raise ValueError(
                    f"{type(self).__name__}: coordinates shape {self.coordinates.shape} is "
                    f"incompatible with values shape {self.values.shape}. "
                    f"coordinates.shape[:-1] must equal values.shape "
                    f"({self.values.shape}) for non-channeled data, or "
                    f"values.shape[:-1] ({self.values.shape[:-1]}) for channeled data."
                )
            # Sanity-check units: clinical ultrasound scan regions are at most a few tens of
            # centimetres across, so any finite coordinate magnitude above 1 m almost certainly
            # indicates the array was supplied in millimetres rather than metres.
            max_abs = np.max(np.abs(self.coordinates[np.isfinite(self.coordinates)]), initial=0.0)
            if max_abs > 1.0:
                log.warning(
                    f"{type(self).__name__}: coordinates have a maximum absolute value of "
                    f"{max_abs:.4g}, which exceeds 1 m.  Ultrasound scan regions are "
                    "typically a few centimetres across.  Please verify that coordinates "
                    "are in metres, not millimetres."
                )
        else:
            log.warning(
                f"{type(self).__name__}: coordinates are not provided, please consider adding "
                "a coordinates field to ensure the map can be correctly displayed."
            )


@dataclass
class FloatMap(Map):
    """Map data with float32 pixel values and per-pixel Cartesian coordinates."""

    SCHEMA = {
        **Map.SCHEMA,
        "values": {
            **Map.SCHEMA["values"],
            "dtype": np.float32,
        },
    }


@dataclass
class BooleanMap(Map):
    """Map data with bool pixel values and per-pixel Cartesian coordinates."""

    SCHEMA = {
        **Map.SCHEMA,
        "values": {
            **Map.SCHEMA["values"],
            "dtype": np.bool_,
        },
    }


@dataclass
class UnsignedIntMap(Map):
    """Map data with uint8 pixel values and per-pixel Cartesian coordinates."""

    SCHEMA = {
        **Map.SCHEMA,
        "values": {
            **Map.SCHEMA["values"],
            "dtype": np.uint8,
        },
    }


@dataclass
class Segmentation(BooleanMap):
    """Segmentation data with per-pixel Cartesian coordinates.

    Args:
        values: The segmentation values of shape ``(n_frames, z, x, y, n_labels)`` and type bool.
        coordinates: Per-pixel Cartesian positions in metres, shape ``(n_frames, z, x, y, 3)``.
        labels: The labels corresponding to the segmentation values, where each unique value
            in the values corresponds to a label in this list of shape ``(n_labels,)`` and type str.
    """

    def __post_init__(self):
        assert self.values.ndim == 5, (
            "Segmentation values must have 5 dimensions (n_frames, z, x, y, n_labels)"
        )
        super().__post_init__()


@dataclass
class Image(Map):
    """Reconstructed (log-compressed) image data with per-pixel Cartesian coordinates.

    Args:
        values: The image values of shape ``(n_frames, z, x, y)`` or ``(n_frames, z, x)``
            and type uint8 or float32. For float32 values, the values should be in dB
            (between -inf and 0).
        coordinates: Per-pixel Cartesian positions in metres, shape ``(*values.shape, 3)``.
    """

    SCHEMA = {
        **Map.SCHEMA,
        "values": {
            "dtype": (np.float32, np.uint8),
            "shape": (
                ("n_frames", "x", "z", "y"),
                ("n_frames", "x", "z"),
            ),
        },
    }

    def __post_init__(self):
        super().__post_init__()

        # Check that image values are in dB scale (finite or -inf, and <= 0)
        if self.values.dtype == np.float32:
            if not np.all(np.isfinite(self.values) | np.isneginf(self.values)):
                raise ValueError("Image values must be finite or -inf (dB scale).")
            if not np.all(self.values <= 0):
                raise ValueError("Image values must be in dB scale <= 0 when using float32 dtype.")


@dataclass
class BeamformedData(FloatMap):
    """Beamformed (beamsummed) data with per-pixel Cartesian coordinates.

    Args:
        values: The beamformed data of shape ``(n_frames, z, x, n_ch)`` or
            ``(n_frames, z, x, y, n_ch)`` and type float32.
            n_ch is 1 for RF data or 2 for IQ data.
        coordinates: Per-pixel Cartesian positions in metres, shape
            ``(n_frames, z, x, 3)`` or ``(n_frames, z, x, y, 3)``.
        labels: The labels for the channel dimension, e.g. ``["RF"]`` or ``["I", "Q"]``.
            Auto-generated from n_ch if not provided.
    """

    SCHEMA = {
        **FloatMap.SCHEMA,
        "values": {
            "dtype": np.float32,
            "shape": (
                ("n_frames", "z", "x", "y", "n_ch"),
                ("n_frames", "z", "x", "n_ch"),
            ),
        },
        "labels": {"dtype": np.str_, "shape": ("n_ch",)},
    }

    def __post_init__(self):
        n_ch = self.values.shape[-1]
        if n_ch not in (1, 2):
            raise ValueError(
                f"Beamformed data must have n_ch ∈ {{1, 2}} (RF or IQ), "
                f"got n_ch={n_ch} (shape {self.values.shape})."
            )
        if self.labels is None:
            self.labels = (
                np.array(["RF"], dtype=np.str_)
                if n_ch == 1
                else np.array(["I", "Q"], dtype=np.str_)
            )
        super().__post_init__()


@dataclass
class EnvelopeData(FloatMap):
    """Envelope-detected data with per-pixel Cartesian coordinates.

    Args:
        values: The envelope data of shape ``(n_frames, x, z)`` or
            ``(n_frames, z, x, y)`` and type float32.
        coordinates: Per-pixel Cartesian positions in metres, shape ``(*values.shape, 3)``.
    """

    SCHEMA = {
        **FloatMap.SCHEMA,
        "values": {
            "dtype": np.float32,
            "shape": (
                ("n_frames", "z", "x", "y"),
                ("n_frames", "z", "x"),
            ),
        },
    }


@dataclass
class SosMap(FloatMap):
    """Speed-of-sound map data with per-pixel Cartesian coordinates.

    Args:
        values: The speed-of-sound map values in m/s of shape ``(n_frames, z, x, y)``
            and type float32.
        coordinates: Per-pixel Cartesian positions in metres, shape
            ``(n_frames, z, x, 3)`` or ``(n_frames, z, x, y, 3)``.
    """

    def __post_init__(self):
        super().__post_init__()

        if self.unit is not None and self.unit != "m/s":
            raise ValueError(f"Speed-of-sound map unit should be 'm/s', got '{self.unit}'")

        # Check sensible values for speed of sound
        if np.any(self.values < 300):
            log.warning(
                "Speed-of-sound map contains values below 300 m/s, which is unusually low. "
                "Please verify that the speed-of-sound values are correct and in m/s."
            )


@dataclass
class StrainPercentageMap(FloatMap):
    """Strain map data with per-pixel Cartesian coordinates.

    Args:
        values: The strain values in % of shape ``(n_frames, z, x, y)`` and type float32.
        coordinates: Per-pixel Cartesian positions in metres.
    """

    def __post_init__(self):
        super().__post_init__()

        if self.unit is not None and self.unit != "%":
            raise ValueError(f"Strain map unit should be '%', got '{self.unit}'")


@dataclass
class ShearWaveElastographyMap(FloatMap):
    """Shear-wave elastography data with per-pixel Cartesian coordinates.

    Args:
        values: The shear-wave elastography values in m/s of shape
            ``(n_frames, z, x, y)`` and type float32.
        coordinates: Per-pixel Cartesian positions in metres.
    """

    def __post_init__(self):
        super().__post_init__()

        if self.unit is not None and self.unit != "m/s":
            raise ValueError(f"SWE map unit should be 'm/s', got '{self.unit}'")


@dataclass
class TissueDopplerMap(FloatMap):
    """Tissue Doppler data with per-pixel Cartesian coordinates.

    Args:
        values: The tissue Doppler values in m/s of shape ``(n_frames, z, x, y)``
            and type float32.
        coordinates: Per-pixel Cartesian positions in metres.
    """

    def __post_init__(self):
        super().__post_init__()

        if self.unit is not None and self.unit != "m/s":
            raise ValueError(f"SWE map unit should be 'm/s', got '{self.unit}'")


@dataclass
class ColorDopplerMap(FloatMap):
    """Color Doppler (velocity) data with per-pixel Cartesian coordinates.

    Args:
        values: The color Doppler velocity values in m/s of shape
            ``(n_frames, z, x, y)`` and type float32. Positive values
            indicate flow towards the transducer, negative values
            indicate flow away from the transducer.
        coordinates: Per-pixel Cartesian positions in metres.
    """

    def __post_init__(self):
        super().__post_init__()

        if self.unit is not None and self.unit != "m/s":
            raise ValueError(f"SWE map unit should be 'm/s', got '{self.unit}'")


@dataclass(init=False)
class DataSpec(Spec):
    """Data group containing raw channels, derived pipeline products, and optional spatial maps.

    Pipeline data products (plain arrays):
        raw_data: Raw channel data of shape (n_frames, n_tx, n_ax, n_el, n_ch)
            and type float32 or int16.
        aligned_data: Time-of-flight corrected data of shape
            (n_frames, n_tx, n_ax, n_el, n_ch) and type float32 or int16.

    Spatial map data products (values + per-pixel coordinates):
        - beamformed_data: Beamformed (beamsummed) data and per-pixel coordinates.
        - envelope_data: Envelope-detected data and per-pixel coordinates.
        - image: Reconstructed image data and per-pixel coordinates.
        - segmentation: Segmentation data and per-pixel coordinates.
        - sos_map: Speed-of-sound map data and per-pixel coordinates.
        - strain_percentage_map: Strain map data and per-pixel coordinates.
        - shear_wave_elastography_map: Shear-wave elastography data and per-pixel coordinates.
        - tissue_doppler: Tissue Doppler data and per-pixel coordinates.
        - color_doppler: Color Doppler velocity data and per-pixel coordinates.
        - \\*\\*kwargs: Any other spatially aligned map data and per-pixel coordinates.

    At least one data field (pipeline or spatial map) must be provided.
    """

    # Pipeline data products (plain arrays)
    raw_data: np.ndarray | None = None
    aligned_data: np.ndarray | None = None
    # Spatial map data products (with coordinates metadata)
    beamformed_data: BeamformedData | dict | None = None
    envelope_data: EnvelopeData | dict | None = None
    image: Image | dict | None = None
    segmentation: Segmentation | dict | None = None
    sos_map: SosMap | dict | None = None
    strain_percentage_map: StrainPercentageMap | dict | None = None
    shear_wave_elastography_map: ShearWaveElastographyMap | dict | None = None
    tissue_doppler: TissueDopplerMap | dict | None = None
    color_doppler: ColorDopplerMap | dict | None = None

    SCHEMA = {
        # Pipeline data products
        "raw_data": {
            "dtype": (np.float32, np.int16),
            "shape": ("n_frames", "n_tx", "n_ax", "n_el", "n_ch"),
        },
        "aligned_data": {
            "dtype": (np.float32, np.int16),
            "shape": ("n_frames", "n_tx", "n_ax", "n_el", "n_ch"),
        },
        # Spatial map data products
        "beamformed_data": {"spec": BeamformedData},
        "envelope_data": {"spec": EnvelopeData},
        "image": {"spec": Image},
        "segmentation": {"spec": Segmentation},
        "sos_map": {"spec": SosMap},
        "strain_percentage_map": {"spec": StrainPercentageMap},
        "shear_wave_elastography_map": {"spec": ShearWaveElastographyMap},
        "tissue_doppler": {"spec": TissueDopplerMap},
        "color_doppler": {"spec": ColorDopplerMap},
    }

    FIELD_METADATA = {
        "raw_data": {"unit": "-", "description": "Raw channel data."},
        "aligned_data": {"unit": "-", "description": "Time-of-flight corrected data."},
    }

    def __init__(
        self,
        raw_data: np.ndarray | None = None,
        aligned_data: np.ndarray | None = None,
        beamformed_data: BeamformedData | dict | None = None,
        envelope_data: EnvelopeData | dict | None = None,
        image: Image | dict | None = None,
        segmentation: Segmentation | dict | None = None,
        sos_map: SosMap | dict | None = None,
        strain_percentage_map: StrainPercentageMap | dict | None = None,
        shear_wave_elastography_map: ShearWaveElastographyMap | dict | None = None,
        tissue_doppler: TissueDopplerMap | dict | None = None,
        color_doppler: ColorDopplerMap | dict | None = None,
        **extra_maps,
    ):
        self.raw_data = raw_data
        self.aligned_data = aligned_data
        self.beamformed_data = beamformed_data
        self.envelope_data = envelope_data
        self.image = image
        self.segmentation = segmentation
        self.sos_map = sos_map
        self.strain_percentage_map = strain_percentage_map
        self.shear_wave_elastography_map = shear_wave_elastography_map
        self.tissue_doppler = tissue_doppler
        self.color_doppler = color_doppler

        reserved_keys = set(self.SCHEMA) | set(self.__dataclass_fields__) | set(dir(Spec))
        for key, value in extra_maps.items():
            if key in reserved_keys:
                raise TypeError(f"Invalid custom data key '{key}': reserved name")
            if isinstance(value, np.ndarray):
                raise TypeError(
                    f"Custom data key '{key}' must be a spatial map "
                    f"(a dict with at least a 'values' key), not a flat array. "
                    f"Only 'raw_data' and 'aligned_data' are accepted as flat arrays. "
                    f"Wrap your data: {{'values': array, 'coordinates': coordinates_array}}."
                )
            setattr(self, key, value)

        # Add custom extra maps to the schema as generic Map specs, so they get validated.
        self._extra_map_keys = tuple(extra_maps.keys())
        if getattr(self, "_extra_map_keys", ()):
            self.SCHEMA = {
                **self.SCHEMA,
                **{key: {"spec": Map} for key in self._extra_map_keys},
            }

        self.__post_init__()

    def __post_init__(self):
        # Ensure at least one data field is present
        all_data_keys = [k for k in self.SCHEMA]
        has_any = any(getattr(self, k, None) is not None for k in all_data_keys)
        if not has_any:
            raise ValueError(
                "At least one data field must be provided. "
                f"Available fields: {', '.join(all_data_keys)}"
            )

        super().__post_init__()

        # n_ch must be 1 (RF) or 2 (IQ) for data types that carry a channel axis.
        _N_CH_FIELDS = ("raw_data", "aligned_data")
        for fname in _N_CH_FIELDS:
            arr = getattr(self, fname, None)
            if arr is not None and isinstance(arr, np.ndarray):
                n_ch = arr.shape[-1]
                if n_ch not in (1, 2):
                    raise ValueError(
                        f"'{fname}' must have n_ch ∈ {{1, 2}} (RF or IQ), "
                        f"got n_ch={n_ch} (shape {arr.shape})."
                    )


@dataclass
class ScanSpec(Spec):
    """Scan group with acquisition and transmit metadata.

    All fields are aligned with the data format specification.

    Args:
        probe_geometry: The probe geometry in meters of shape (n_el, 3),
            represented as (x, y, z) coordinates.
        sampling_frequency: The sampling frequency in Hz.
        center_frequency: The center frequency in Hz of the transmit pulse.
            Single scalar if all transmits share the same center frequency;
            otherwise an array of shape (n_tx,) with one frequency per transmit.
        demodulation_frequency: The frequency in Hz at which the data should
            be demodulated. Usually the same as center_frequency, but different
            when doing harmonic imaging. Single scalar if all transmits share
            the same center frequency; otherwise an array of shape (n_tx,) with
            one frequency per transmit.
        initial_times: The times in seconds when the A/D converter starts sampling
            of shape (n_tx,). This is the time between the first element firing
            and the first recorded sample.
        t0_delays: The transmit delays in seconds for each element of shape
            (n_tx, n_el). This is the time at which each element fires, shifted
            such that the first element fires at t=0.
        tx_apodizations: The apodization values that were applied to each
            element during transmit of shape (n_tx, n_el). This is a value
            between -1 and 1 that indicates how much each element contributed
            to the transmit beam, with 0 meaning no contribution and 1 meaning
            full contribution. Negative values indicate that the element was
            fired with opposite polarity.
        focus_distances: The transmit focus distances in meters of shape (n_tx,).
            This is the distance from the origin point on the transducer to
            where the beam comes to focus. For planewaves this is set to
            infinity or zero.
        transmit_origins: The transmit origins of the transmit beams in meters of
            shape (n_tx, 3). This is the (x, y, z) position from which the beam
            is transmitted.
        polar_angles: The polar angles in radians of the transmit beams of shape (n_tx,).
        time_to_next_transmit: The time in s between subsequent transmit events
            of shape (n_frames, n_tx).
        azimuth_angles: The azimuthal angles in radians of the transmit beams of
            shape (n_tx,).
        sound_speed: The speed of sound in meters per second.
        tgc_gain_curve: The time-gain-compensation that was applied to every
            sample in the raw_data of shape (n_ax,). Divide by this curve to
            undo the TGC.
        element_width: The width of the elements in the probe in meters.
        waveforms_one_way: One-way waveforms of shape (n_tx, .) as simulated
            by the Verasonics system. This is the waveform after being filtered
            by the transducer bandwidth once.
        waveforms_two_way: Two-way waveforms of shape (n_tx, .) as simulated
            by the Verasonics system. This is the waveform after being filtered
            by the transducer bandwidth twice.
    """

    probe_geometry: np.ndarray
    sampling_frequency: np.ndarray | float
    center_frequency: np.ndarray | float
    demodulation_frequency: np.ndarray | float
    initial_times: np.ndarray
    t0_delays: np.ndarray
    tx_apodizations: np.ndarray
    focus_distances: np.ndarray
    transmit_origins: np.ndarray
    polar_angles: np.ndarray
    time_to_next_transmit: np.ndarray = None
    azimuth_angles: np.ndarray = None
    sound_speed: np.ndarray | float | None = None
    tgc_gain_curve: np.ndarray | None = None
    element_width: np.ndarray | float | None = None
    waveforms_one_way: np.ndarray | None = None
    waveforms_two_way: np.ndarray | None = None

    SCHEMA = {
        "probe_geometry": {"dtype": np.float32, "shape": ("n_el", 3)},
        "sampling_frequency": {"dtype": np.float32, "shape": ()},
        "center_frequency": {"dtype": np.float32, "shape": ((), ("n_tx",))},
        "demodulation_frequency": {"dtype": np.float32, "shape": ((), ("n_tx",))},
        "initial_times": {"dtype": np.float32, "shape": ("n_tx",)},
        "t0_delays": {"dtype": np.float32, "shape": ("n_tx", "n_el")},
        "tx_apodizations": {"dtype": np.float32, "shape": ("n_tx", "n_el")},
        "focus_distances": {"dtype": np.float32, "shape": ("n_tx",)},
        "transmit_origins": {"dtype": np.float32, "shape": ("n_tx", 3)},
        "polar_angles": {"dtype": np.float32, "shape": ("n_tx",)},
        "time_to_next_transmit": {"dtype": np.float32, "shape": ("n_frames", "n_tx")},
        "azimuth_angles": {"dtype": np.float32, "shape": ("n_tx",)},
        "sound_speed": {"dtype": np.float32, "shape": ()},
        "tgc_gain_curve": {"dtype": np.float32, "shape": ("n_ax",)},
        "element_width": {"dtype": np.float32, "shape": ()},
        "waveforms_one_way": {
            "dtype": np.float32,
            "shape": ("n_tx", "n_samples_one_way"),
        },
        "waveforms_two_way": {
            "dtype": np.float32,
            "shape": ("n_tx", "n_samples_two_way"),
        },
    }

    FIELD_METADATA = {
        "probe_geometry": {"unit": "m", "description": "Probe geometry (x, y, z) per element."},
        "sampling_frequency": {"unit": "Hz", "description": "Sampling frequency."},
        "center_frequency": {
            "unit": "Hz",
            "description": "Center frequency of the transmit pulse.",
        },
        "demodulation_frequency": {"unit": "Hz", "description": "Demodulation frequency."},
        "initial_times": {"unit": "s", "description": "A/D converter start times per transmit."},
        "t0_delays": {"unit": "s", "description": "Transmit delays per element."},
        "tx_apodizations": {"unit": "-", "description": "Transmit apodization per element."},
        "focus_distances": {"unit": "m", "description": "Transmit focus distances."},
        "transmit_origins": {"unit": "m", "description": "Transmit beam origins (x, y, z)."},
        "polar_angles": {"unit": "rad", "description": "Polar angles of transmit beams."},
        "time_to_next_transmit": {"unit": "s", "description": "Time between transmit events."},
        "azimuth_angles": {"unit": "rad", "description": "Azimuthal angles of transmit beams."},
        "sound_speed": {"unit": "m/s", "description": "Speed of sound."},
        "tgc_gain_curve": {"unit": "-", "description": "Time-gain-compensation curve."},
        "element_width": {"unit": "m", "description": "Element width of the probe."},
        "waveforms_one_way": {"unit": "V", "description": "One-way transmit waveforms."},
        "waveforms_two_way": {"unit": "V", "description": "Two-way transmit waveforms."},
    }

    @property
    def n_tx(self) -> int:
        """Number of transmits."""
        return self.t0_delays.shape[0]

    @property
    def n_el(self) -> int:
        """Number of elements."""
        return self.t0_delays.shape[1]

    def __post_init__(self):
        super().__post_init__()

        if np.any(self.probe_geometry > 1.0) or np.any(self.probe_geometry < -1.0):
            log.warning(
                "Probe geometry values are unusually large, extending beyond +/- 1.0 meters. "
                "Please verify that the probe geometry values are correct and in meters."
            )
        if self.sampling_frequency <= 0:
            raise ValueError(f"Sampling frequency must be positive, got {self.sampling_frequency}")
        if np.any(self.center_frequency < 0):
            raise ValueError(f"Center frequency cannot be negative, got {self.center_frequency}")
        if np.any(self.demodulation_frequency < 0):
            raise ValueError(
                f"Demodulation frequency cannot be negative, got {self.demodulation_frequency}"
            )
        if np.any(self.t0_delays < 0):
            raise ValueError(f"Transmit delays cannot be negative, got {self.t0_delays}")
        if np.any(np.logical_and(self.focus_distances >= 1, self.focus_distances != np.inf)):
            log.warning(
                "Focus distances greater than or equal to 1 meter may be unusually large. "
                "Maybe you have to convert to meters?"
            )
        if np.any(self.transmit_origins > 1.0) or np.any(self.transmit_origins < -1.0):
            log.warning(
                "Transmit origin values are unusually large, extending beyond +/- 1.0 meters. "
                "Please verify that the transmit origin values are correct and in meters."
            )
        if np.any(self.polar_angles < -np.pi) or np.any(self.polar_angles > np.pi):
            raise ValueError(
                f"Polar angles should be between -pi and pi radians, got values between "
                f"{np.min(self.polar_angles)} and {np.max(self.polar_angles)}"
            )
        if self.azimuth_angles is not None and (
            np.any(self.azimuth_angles < -np.pi) or np.any(self.azimuth_angles > np.pi)
        ):
            raise ValueError(
                f"Azimuth angles should be between -pi and pi radians, got values between "
                f"{np.min(self.azimuth_angles)} and {np.max(self.azimuth_angles)}"
            )
        if self.sound_speed is not None and self.sound_speed <= 0:
            raise ValueError(f"Sound speed must be positive, got {self.sound_speed}")
        if self.tgc_gain_curve is not None and np.any(self.tgc_gain_curve < 0):
            raise ValueError(
                f"TGC gain curve values must be non-negative, got values between "
                f"{np.min(self.tgc_gain_curve)} and {np.max(self.tgc_gain_curve)}"
            )
        if self.element_width is not None and self.element_width <= 0:
            raise ValueError(f"Element width must be positive, got {self.element_width}")

        # Try to simplify the data by squeezing out any singleton dimensions,
        # e.g. if center_frequency is an array with all the same value
        if isinstance(self.center_frequency, np.ndarray) and self.center_frequency.ndim == 1:
            if np.all(self.center_frequency == self.center_frequency[0]):
                self.center_frequency = self.center_frequency[0]
        if (
            isinstance(self.demodulation_frequency, np.ndarray)
            and self.demodulation_frequency.ndim == 1
        ):
            if np.all(self.demodulation_frequency == self.demodulation_frequency[0]):
                self.demodulation_frequency = self.demodulation_frequency[0]

        self.warn_missing_optional_fields()


@dataclass
class Subject(Spec):
    """Subject metadata associated with the study.

    Args:
        id: Subject ID.
        type: Subject type, e.g. human, phantom, animal.
        age: Subject age in years.
        sex: Subject sex.
        fat: Subject fat percentage.
    """

    id: str | None = None
    type: str | None = None
    age: np.uint8 | None = None
    sex: str | None = None
    fat_percentage: np.float32 | None = None

    SCHEMA = {
        "id": {"dtype": str, "shape": ()},
        "type": {"dtype": str, "shape": ()},
        "age": {"dtype": np.uint8, "shape": ()},
        "sex": {"dtype": str, "shape": ()},
        "fat_percentage": {"dtype": np.float32, "shape": ()},
    }

    FIELD_METADATA = {
        "id": {"description": "Subject ID. Needed for subject-wise splits."},
    }

    def __post_init__(self):
        super().__post_init__()

        if self.id is not None and not self.id.strip():
            raise ValueError("Subject ID cannot be an empty string")

        self.warn_missing_optional_fields()

        if self.fat_percentage is not None and (
            self.fat_percentage < 0 or self.fat_percentage > 100
        ):
            raise ValueError(
                f"Subject fat percentage must be between 0 and 100, got {self.fat_percentage}"
            )


@dataclass
class Signal(Spec):
    """Base class for additional signals with timing and sampling-frequency metadata.

    Args:
        start_time_offset: Time offset in seconds between the first transmit event
            of the ultrasound acquisition and sample 0 of this data. Negative
            means this data starts before the first transmit event; positive
            means it starts after.
        sampling_frequency: Sampling frequency in Hz for the additional signal samples.
    """

    start_time_offset: np.ndarray | float
    sampling_frequency: np.ndarray | float

    SCHEMA = {
        "start_time_offset": {"dtype": np.float32, "shape": ()},
        "sampling_frequency": {"dtype": np.float32, "shape": ()},
    }

    FIELD_METADATA = {
        "start_time_offset": {
            "unit": "s",
            "description": (
                "Time offset between the first transmit event of the ultrasound "
                "acquisition and sample 0 of this data. Negative means this data "
                "starts before the first transmit event; positive means it starts "
                "after."
            ),
        },
        "sampling_frequency": {"unit": "Hz", "description": "Sampling frequency."},
    }

    def __post_init__(self):
        super().__post_init__()

        if self.sampling_frequency <= 0:
            raise ValueError(f"Sampling frequency must be positive, got {self.sampling_frequency}")


@dataclass
class ProbePose(Signal):
    """Sampled probe pose metadata at the tip of the transducer.

    The pose uses the coordinate convention x = lateral along the transducer,
    y = elevation (out of plane), and z = axial (depth).

    Args:
        translation: Position of the transducer tip in meters of shape (T, 3),
            ordered as (x, y, z).
        rotation: Orientation of the transducer tip of shape (T, 3) or (T, 4),
            interpreted according to ``rotation_representation``.
        rotation_representation: Rotation parameterization. Supported values are
            ``"euler_xyz"``, ``"quaternion_wxyz"``, and ``"quaternion_xyzw"``.
        start_time_offset: Time offset in seconds between the first transmit event
            of the ultrasound acquisition and sample 0 of this data.
        sampling_frequency: Sampling frequency in Hz for probe pose samples.
    """

    translation: np.ndarray
    rotation: np.ndarray
    rotation_representation: str

    SCHEMA = {
        "translation": {"dtype": np.float32, "shape": ("T", 3)},
        "rotation": {"dtype": np.float32, "shape": (("T", 3), ("T", 4))},
        "rotation_representation": {"dtype": str, "shape": ()},
        **Signal.SCHEMA,
    }

    FIELD_METADATA = {
        "translation": {
            "unit": "m",
            "description": (
                "Position of the transducer tip, ordered as (x, y, z), where x is "
                "lateral along the transducer, y is elevation (out of plane), and "
                "z is axial (depth)."
            ),
        },
        "rotation": {
            "unit": "-",
            "description": (
                "Orientation associated with the transducer-tip pose in the "
                "x-lateral, y-elevation, z-axial coordinate convention, interpreted "
                "according to rotation_representation."
            ),
        },
        "rotation_representation": {
            "unit": "-",
            "description": (
                "Rotation parameterization: one of euler_xyz, quaternion_wxyz, or quaternion_xyzw."
            ),
        },
        **Signal.FIELD_METADATA,
    }

    def __post_init__(self):
        super().__post_init__()

        valid_representations = {
            "euler_xyz": 3,
            "quaternion_wxyz": 4,
            "quaternion_xyzw": 4,
        }
        if self.translation.shape[0] != self.rotation.shape[0]:
            raise ValueError(
                "translation and rotation must have the same number of time samples, "
                f"got {self.translation.shape[0]} and {self.rotation.shape[0]}"
            )
        if self.rotation_representation not in valid_representations:
            valid = ", ".join(sorted(valid_representations))
            raise ValueError(
                f"rotation_representation must be one of {{{valid}}}, "
                f"got {self.rotation_representation!r}"
            )

        expected_width = valid_representations[self.rotation_representation]
        if self.rotation.shape[1] != expected_width:
            raise ValueError(
                "rotation shape does not match rotation_representation: "
                f"got {self.rotation.shape} for {self.rotation_representation!r}"
            )


@dataclass
class Signal1D(Signal):
    """One-dimensional sampled signal with timing metadata.

    Args:
        samples: Signal samples of shape (T) and type uint8 or float32 or int16 or complex64.
        start_time_offset: Time offset in seconds between the first transmit event
            of the ultrasound acquisition and sample 0 of this data.
        sampling_frequency: Sampling frequency in Hz for signal samples.
    """

    samples: np.ndarray

    SCHEMA = {
        "samples": {"dtype": (np.uint8, np.float32, np.int16, np.complex64), "shape": ("T",)},
        **Signal.SCHEMA,
    }

    FIELD_METADATA = {
        "samples": {"unit": "-", "description": "Signal samples."},
        **Signal.FIELD_METADATA,
    }


@dataclass
class SignalND(Signal):
    """N-dimensional sampled signal with timing metadata.

    Args:
        samples: Signal samples of shape (T, ...) and type uint8 or float32 or int16 or complex64.
        start_time_offset: Time offset in seconds between the first transmit event
            of the ultrasound acquisition and sample 0 of this data.
        sampling_frequency: Sampling frequency in Hz for signal samples.
    """

    samples: np.ndarray

    SCHEMA = {
        "samples": {"dtype": (np.uint8, np.float32, np.int16, np.complex64), "shape": ("T", "...")},
        **Signal.SCHEMA,
    }

    FIELD_METADATA = {
        "samples": {"unit": "-", "description": "Signal samples."},
        **Signal.FIELD_METADATA,
    }


@dataclass
class Annotations(Spec):
    """Frame-level annotations, either per frame or broadcast labels.

    Args:
        anatomy: Anatomy label.
        view: View label of shape (n_frames,).
        label: Pathology or classification label of shape (n_frames,).
        image_quality: Image quality label, e.g. low, mid, high.
    """

    anatomy: np.ndarray | str | None = None
    view: np.ndarray | None = None
    label: np.ndarray | None = None
    image_quality: np.ndarray | str | None = None

    SCHEMA = {
        "anatomy": {"dtype": np.str_, "shape": (("n_frames",), ())},
        "view": {"dtype": np.str_, "shape": ("n_frames",)},
        "label": {"dtype": np.str_, "shape": ("n_frames",)},
        "image_quality": {"dtype": np.str_, "shape": (("n_frames",), ())},
    }


@dataclass(init=False)
class MetadataSpec(Spec):
    """Metadata group with subject, acquisition context, annotations, and extra signals."""

    subject: Subject | dict = field(default_factory=Subject)
    credit: str | None = None
    probe_pose: ProbePose | dict | None = None
    voice_narration: Signal1D | dict | None = None
    ecg: Signal1D | dict | None = None
    text_report: str | None = None
    annotations: Annotations | dict | None = None

    SCHEMA = {
        "subject": {"spec": Subject},
        "credit": {"dtype": str, "shape": ()},
        "probe_pose": {"spec": ProbePose},
        "voice_narration": {"spec": Signal1D},
        "ecg": {"spec": Signal1D},
        "text_report": {"dtype": str, "shape": ()},
        "annotations": {"spec": Annotations},
    }

    FIELD_METADATA = {
        "credit": {"unit": "-", "description": "Credit or attribution for the dataset."},
        "probe_pose": {"unit": "-", "description": "Sampled probe pose at the transducer tip."},
        "voice_narration": {"unit": "-", "description": "Voice narration signal."},
        "ecg": {"unit": "-", "description": "Electrocardiogram signal."},
        "text_report": {"unit": "-", "description": "Free-text report associated with the study."},
        "annotations": {"unit": "-", "description": "Frame-level annotations."},
    }

    def __init__(
        self,
        subject: Subject | dict | None = None,
        credit: str | None = None,
        probe_pose: ProbePose | dict | None = None,
        voice_narration: Signal1D | dict | None = None,
        ecg: Signal1D | dict | None = None,
        text_report: str | None = None,
        annotations: Annotations | dict | None = None,
        **extra_signals,
    ):
        self.subject = subject
        self.credit = credit
        self.probe_pose = probe_pose
        self.voice_narration = voice_narration
        self.ecg = ecg
        self.text_report = text_report
        self.annotations = annotations

        reserved_keys = set(self.SCHEMA) | set(self.__dataclass_fields__) | set(dir(Spec))
        for key, value in extra_signals.items():
            if key in reserved_keys:
                raise TypeError(f"Invalid custom metadata key '{key}': reserved name")
            if isinstance(value, np.ndarray):
                raise TypeError(
                    f"Custom metadata key '{key}' must be a SignalND "
                    f"(a dict with 'samples', 'start_time_offset', and 'sampling_frequency'), "
                    f"not a flat array. "
                    f"Wrap your data: {{'samples': array, 'start_time_offset': 0.0, "
                    f"'sampling_frequency': fs}}."
                )
            setattr(self, key, value)

        # Add custom extra signals to the schema as generic SignalND specs, so they get validated.
        self._extra_signal_keys = tuple(extra_signals.keys())
        if getattr(self, "_extra_signal_keys", ()):
            self.SCHEMA = {
                **self.SCHEMA,
                **{key: {"spec": SignalND} for key in self._extra_signal_keys},
            }

        self.__post_init__()

    def __post_init__(self):
        super().__post_init__()

        self.warn_missing_optional_fields()


@dataclass
class MetricsSpec(Spec):
    """Metrics group for acquisition-level quality/performance metrics.

    Args:
        common_midpoint_phase_error: Common midpoint phase error in radians of
            shape (n_frames,) and type float32.
        coherence_factor: Coherence factor of shape (n_frames,) and type float32.
    """

    common_midpoint_phase_error: np.ndarray | None = None
    coherence_factor: np.ndarray | None = None

    SCHEMA = {
        "common_midpoint_phase_error": {
            "dtype": np.float32,
            "shape": ("n_frames",),
        },
        "coherence_factor": {"dtype": np.float32, "shape": ("n_frames",)},
    }


@dataclass
class FileSpec(Spec):
    """A dataset containing all the data, scan parameters, metadata,
    and metrics for a single acquisition.

    This class can be used to create a new dataset, which is validated upon initialization.
    Afterwards, it can be saved to disk as hdf5 file.

    Args:
        data: The data for the acquisition.
        scan: The scan parameters.
        metadata: Additional metadata about the acquisition.
        metrics: Metrics computed from the acquisition.
        probe_name: The name of the probe used to acquire the data.
        us_machine: The ultrasound machine used to acquire the data.

    Example:
        .. doctest::

            >>> from zea.data.spec import FileSpec
            >>> import numpy as np

            >>> dataset = FileSpec(
            ...     data={
            ...         "raw_data": np.zeros((2, 4, 64, 8, 1), dtype=np.float32),
            ...     },
            ...     scan={
            ...         "probe_geometry": np.zeros((8, 3), dtype=np.float32),
            ...         "sampling_frequency": np.float32(40e6),
            ...         "center_frequency": np.float32(5e6),
            ...         "demodulation_frequency": np.float32(5e6),
            ...         "initial_times": np.zeros(4, dtype=np.float32),
            ...         "t0_delays": np.zeros((4, 8), dtype=np.float32),
            ...         "tx_apodizations": np.ones((4, 8), dtype=np.float32),
            ...         "focus_distances": np.full(4, np.inf, dtype=np.float32),
            ...         "transmit_origins": np.zeros((4, 3), dtype=np.float32),
            ...         "polar_angles": np.zeros(4, dtype=np.float32),
            ...     },
            ... )
            >>> dataset.data.raw_data.shape
            (2, 4, 64, 8, 1)
    """

    data: DataSpec | dict
    scan: ScanSpec | dict | None = None
    metadata: MetadataSpec | dict = field(default_factory=MetadataSpec)
    metrics: MetricsSpec | dict = field(default_factory=MetricsSpec)
    probe_name: str | None = None
    us_machine: str | None = None
    description: str | None = None

    SCHEMA = {
        "data": {"spec": DataSpec},
        "scan": {"spec": ScanSpec},
        "metadata": {"spec": MetadataSpec},
        "metrics": {"spec": MetricsSpec},
        "probe_name": {"dtype": str, "shape": ()},
        "us_machine": {"dtype": str, "shape": ()},
        "description": {"dtype": str, "shape": ()},
    }

    def __post_init__(self):
        super().__post_init__()

        # scan is mandatory when raw channel data is present
        data = self.data
        has_raw = (isinstance(data, DataSpec) and data.raw_data is not None) or (
            isinstance(data, dict) and data.get("raw_data") is not None
        )
        if has_raw and self.scan is None:
            raise ValueError("'scan' is required when 'raw_data' is provided in the data.")

    def save(self, path: str, compression: str = "gzip") -> None:
        """Save the dataset to the specified path."""
        # Lazy import to avoid circular dependency (spec.py is imported by file.py)
        from zea import File

        try:
            _zea_version = _get_pkg_version("zea")
        except PackageNotFoundError:
            _zea_version = "dev"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with File(str(path), "w") as f:
            f.attrs["zea_version"] = _zea_version

            for group_name, schema in self.SCHEMA.items():
                if "spec" in schema:
                    value: Spec = getattr(self, group_name)
                    if value is None:
                        continue
                    group = f.create_group(group_name)
                    value.store_in_group(group, compression=compression)
                else:
                    value = getattr(self, group_name)
                    if value is not None:
                        f.attrs[group_name] = value

        log.info(f"File saved to {log.yellow(path)}")

    @classmethod
    def from_hdf5(cls, file: h5py.File) -> "FileSpec":
        """Load and validate a :class:`FileSpec` from an open HDF5 file.

        This reads all groups into memory and runs the full spec validation
        (dtype, shape, dimension consistency).  Legacy files are handled
        transparently: extra scalar fields in the scan group (``n_frames``,
        ``n_tx``, etc.) are ignored, and the ``probe`` root
        attribute is mapped to ``probe_name``.

        Args:
            file: An open ``h5py.File`` (or :class:`zea.File`).

        Returns:
            FileSpec: A fully validated spec object.
        """

        def _load_group_as_dict(group: h5py.Group) -> dict:
            result = {}
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Group):
                    result[key] = _load_group_as_dict(item)
                elif isinstance(item, h5py.Dataset):
                    if h5py.check_string_dtype(item.dtype) is not None:
                        val = item.asstr()[()]
                        # h5py returns object-dtype arrays for strings;
                        # convert back to np.str_ so spec dtype checks pass.
                        if isinstance(val, np.ndarray) and val.dtype == object:
                            val = val.astype(np.str_)
                        result[key] = val
                    else:
                        result[key] = item[()]
            return result

        kwargs: dict[str, Any] = {}

        # Load spec groups (data, scan, metadata, metrics)
        for group_name, schema in cls.SCHEMA.items():
            if "spec" in schema:
                if group_name in file:
                    kwargs[group_name] = _load_group_as_dict(file[group_name])
                # else: leave missing, will use default or raise if required
            else:
                # Scalar attrs (probe_name, us_machine, description)
                if group_name in file.attrs:
                    kwargs[group_name] = file.attrs[group_name]

        # ------------------------------------------------------------------
        # Legacy compatibility
        # ------------------------------------------------------------------

        # 1. Map legacy root attribute 'probe' → 'probe_name' by delegating
        #    to File.probe_name, which already checks both 'probe_name' and
        #    'probe' attrs in priority order.
        if "probe_name" not in kwargs:
            try:
                kwargs["probe_name"] = file.probe_name
            except AttributeError:
                log.warning(
                    "File '%s' has no 'probe_name' or 'probe' attribute; probe name will be None.",
                    file.filename,
                )

        # 2. Filter scan dict to only keys recognised by Scan.SCHEMA so
        #    that legacy scalar fields (n_frames, n_ax, n_el, n_tx, n_ch,
        #    bandwidth_percent, …) don't cause unexpected-keyword errors.
        if "scan" in kwargs and isinstance(kwargs["scan"], dict):
            scan_schema_keys = set(ScanSpec.SCHEMA.keys())
            kwargs["scan"] = {k: v for k, v in kwargs["scan"].items() if k in scan_schema_keys}

        # 3. Handle legacy flat `data/<key>` datasets.  In old files spatial
        #    maps (image, image_sc, envelope_data, …) were stored as plain
        #    arrays (n_frames, z, x) rather than groups with values +
        #    coordinates.  Wrap them as {"values": array} so DataSpec accepts
        #    them.  raw_data and aligned_data are valid as flat arrays and are
        #    left untouched.
        if "data" in kwargs and isinstance(kwargs["data"], dict):
            data_dict = kwargs["data"]
            for key in list(data_dict.keys()):
                if not isinstance(data_dict[key], np.ndarray):
                    continue
                schema_entry = DataSpec.SCHEMA.get(key)
                # raw_data / aligned_data are plain-array fields — skip them.
                if schema_entry is not None and "spec" not in schema_entry:
                    continue
                log.warning(
                    "Legacy flat dataset 'data/%s' has no spatial coordinates. "
                    "The array has been loaded as 'values'; coordinates information "
                    "was not stored in this file and will be None.",
                    key,
                )
                data_dict[key] = {"values": data_dict[key]}

        return cls(**kwargs)
