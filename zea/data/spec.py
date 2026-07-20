import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_pkg_version
from pathlib import Path
from typing import Any, ClassVar, List, NoReturn, Sequence, Tuple, cast

import h5py
import hdf5plugin
import numpy as np

from zea import log
from zea.internal.typing import Scalar

# Named dimensions whose sizes must agree wherever they appear.
CONSISTENCY_DIMENSIONS = {"n_frames", "n_tx", "n_ax", "n_el", "n_ch", "n_spatial_ch"}

# Subset that must only agree within a single spec, not across sibling data
# products: channel counts are independent between products (e.g. RF raw_data
# next to IQ beamformed_data), so they are not propagated across spec boundaries.
LOCAL_CONSISTENCY_DIMENSIONS = {"n_ch", "n_spatial_ch"}

UNITS = {
    "m/s": "meters per second",
    "m": "meters",
    "Hz": "Hertz",
    "s": "seconds",
    "V": "volts",
    "–": "unitless",
    "rad": "radians",
    "dB": "decibels",
    "dB/m/Hz": "decibels per meter per hertz",
    "#": "count",
    "%": "percent",
    "kg": "kilograms",
    "kg/m²": "kilograms per square meter",
}

# Blosc(zstd) + bit-shuffle: the one codec zea.data.chunk_reader can decode concurrently.
# Blosc2 compresses ~2% better but its binding holds the GIL, which costs ~30x on reads.
# Bit-shuffle beats byte-shuffle on int16 (only two byte-planes to separate); clevel 9 writes
# ~15x slower than 7 for ~1% more compression. Importing hdf5plugin also registers the filter.
DEFAULT_COMPRESSION = hdf5plugin.Blosc(cname="zstd", clevel=7, shuffle=hdf5plugin.Blosc.BITSHUFFLE)

# Threads Blosc uses on the blocks *within* one chunk. HDF5 runs the filter one chunk at a time
# and single-threaded, so this is most of the write throughput: ~4x (105 -> 453 MB/s). The gain
# reverses past ~8 (the blocks are small and go memory-bound), and writes are often already
# parallel per-file, so keep it modest. setdefault: an explicit env var wins.
BLOSC_NTHREADS = min(8, os.cpu_count() or 1)
os.environ.setdefault("BLOSC_NTHREADS", str(BLOSC_NTHREADS))

# Chunk size 1 on each of these dims: one frame per chunk, since a frame is what a read
# subsamples first. Dims not present on a field are ignored.
DEFAULT_CHUNK_AXES: tuple[str, ...] = ("n_frames",)

# Ceiling on the uncompressed bytes of one chunk. A chunk is the unit of *parallelism*:
# chunk_reader decodes each one in a single thread, so a whole-frame chunk (166 MB on a
# 149-transmit scan) has nothing to parallelise — 102 ms to read one frame against 13 ms at
# 8 MB. Too small costs round trips instead (149 chunks/frame = 1.3 s over HTTP). 8 MB sits
# mid-plateau of a broad optimum (~2-16 MB), the same one for local and cloud, and costs
# nothing on disk: the compression ratio is flat across chunk size.
MAX_CHUNK_BYTES = 8 << 20  # 8 MiB

# Paged file-space strategy: HDF5 allocates in fixed-size pages, which collects the
# metadata that a reader must walk on open (superblock, group and chunk B-trees) into
# few, adjacent pages instead of scattering it through the file. Over HTTP this cuts a
# cold open from 3 requests to 2 and ~0.16 s to ~0.05 s, at ~2% file size for 64 KiB
# pages (larger pages waste space and, past ~1 MiB, requests too).
#
# Requires HDF5 >= 1.10.1 on the reader, which the "v114" high bound below satisfies.
# The high bound is pinned rather than left as "latest": h5py 3.16 started bundling
# libhdf5 2.0, and "latest" resolves to whatever's bundled at write time. HDF5 2.0
# writes newer object-header message versions (e.g. datatype messages) that HDF5
# 1.14.x readers reject outright with "bad version number for datatype message" -
# confirmed by round-tripping a file through h5py 3.16/libhdf5 2.0 and reading it
# back with h5py 3.11/libhdf5 1.14.2. Capping at v114 keeps files readable by the
# 1.14.x installs still in the wild without limiting which h5py *version* writes them.
PAGED_LAYOUT = {
    "libver": ("earliest", "v114"),
    "fs_strategy": "page",
    "fs_page_size": 64 * 1024,
    "fs_persist": True,
}

# Default unit/description for every SCHEMA leaf field.  Subclasses may
# override by defining their own FIELD_METADATA dict.
_DEFAULT_FIELD_UNIT = "–"
_DEFAULT_FIELD_DESCRIPTION = ""


def check_dtype(value: Any, expected_dtype: List[type]) -> None:
    """Check if the dtype of a value matches the expected dtype,
    allowing for compatible types.

    Works for numpy arrays, numpy scalars, and Python native types.
    """
    for dt in expected_dtype:
        if isinstance(dt, type) and issubclass(dt, np.generic):
            expected_np_dtype = np.dtype(dt)
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


def find_matched_shape(value: Any, expected_shapes: Sequence[tuple]) -> tuple | None:
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

    SCHEMA: dict[str, Any]
    FIELD_METADATA: ClassVar[dict[str, Any]]
    __dataclass_fields__: ClassVar[dict[str, Any]]

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
        """Warn about optional fields that were not provided.

        Fields flagged as ``"rare"`` in ``FIELD_METADATA`` are skipped: they are
        almost never set, so warning about them on every write is just noise.
        """
        _optional_fields = self.optional_fields()
        for field_name in self.SCHEMA.keys():
            if field_name in _optional_fields and getattr(self, field_name) is None:
                if hasattr(self, "FIELD_METADATA"):
                    meta = self.FIELD_METADATA.get(field_name, {})
                    if meta.get("rare"):
                        continue
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
        dim_to_field_sizes: defaultdict[str, dict[str, int]],
        nested_dim_to_field_sizes: defaultdict[str, dict[str, int]],
    ) -> None:
        for dim_name, nested_field_sizes in nested_dim_to_field_sizes.items():
            # Channel dimensions are only consistent within a single spec, not
            # across data products, so they are not propagated to the parent.
            if dim_name in LOCAL_CONSISTENCY_DIMENSIONS:
                continue
            dim_to_field_sizes[dim_name].update(nested_field_sizes)

    @staticmethod
    def _track_named_dimensions(
        dim_to_field_sizes: defaultdict[str, dict[str, int]],
        field_path: str,
        matched_shape: tuple,
        shape: tuple,
    ) -> None:
        for i, dim_name in enumerate(matched_shape):
            if isinstance(dim_name, str) and dim_name in CONSISTENCY_DIMENSIONS:
                dim_to_field_sizes[dim_name][field_path] = shape[i]

    @staticmethod
    def _raise_if_shape_mismatch(
        field_name: str, value: Any, expected_shapes: tuple[tuple, ...]
    ) -> NoReturn:
        allowed_shapes = ", ".join(str(shape) for shape in expected_shapes)
        raise ValueError(
            f"{field_name} has shape {value_shape(value)}, expected one of: {allowed_shapes}"
        )

    def _validate_nested_field(
        self, field_name: str, nested_spec: "type[Spec]", field_value: Any
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
        # Keep None
        if value is None:
            return value

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

        # Auto-convert list/tuple to numpy array when a numpy dtype is expected.
        # Skip if list/tuple is itself a valid native type for this field.
        if isinstance(value, (list, tuple)) and expected_np_dtypes:
            for dt in expected_dtype:
                if (
                    isinstance(dt, type)
                    and not issubclass(dt, np.generic)
                    and isinstance(value, dt)
                ):
                    return value
            target = (
                expected_float_dtype if expected_float_dtype is not None else expected_np_dtypes[0]
            )
            return np.asarray(value, dtype=target)

        if hasattr(value, "dtype"):
            value_dtype = np.dtype(value.dtype)

            if (
                expected_float_dtype is not None
                and np.issubdtype(value_dtype, np.floating)
                and value_dtype != expected_float_dtype
            ):
                return value.astype(expected_float_dtype, copy=False)

            return value

        # If the spec expects a native Python type and the value already matches it,
        # keep it as-is instead of converting to a numpy scalar.
        for dt in expected_dtype:
            if isinstance(dt, type) and not issubclass(dt, np.generic) and isinstance(value, dt):
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
        dim_to_field_sizes: defaultdict[str, dict[str, int]],
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
            dim_to_field_sizes=dim_to_field_sizes,
            field_path=field_name,
            matched_shape=matched_shape,
            shape=value_shape(field_value),
        )

    @staticmethod
    def _format_inconsistent_dimension(dim_name: str, field_sizes: dict[str, int]) -> str:
        """Build an error message listing each field grouped by its size for a dimension."""
        sizes_to_fields = defaultdict(list)
        for field_path, size in field_sizes.items():
            sizes_to_fields[size].append(field_path)
        lines = [
            f"  size {size}: {', '.join(sorted(sizes_to_fields[size]))}"
            for size in sorted(sizes_to_fields)
        ]
        return f"Dimension '{dim_name}' has inconsistent sizes:\n" + "\n".join(lines)

    @classmethod
    def _raise_if_inconsistent_dimensions(
        cls,
        dim_to_field_sizes: defaultdict[str, dict[str, int]],
    ) -> None:
        for dim_name, field_sizes in dim_to_field_sizes.items():
            if len(set(field_sizes.values())) > 1:
                raise ValueError(cls._format_inconsistent_dimension(dim_name, field_sizes))

    def _collect_dimension_info(self, prefix: str = "") -> defaultdict[str, dict[str, int]]:
        """Collect the observed size of each named dimension per field in this spec subtree.

        Returns a mapping ``dim_name -> {field_path: size}``.
        """
        dim_to_field_sizes = defaultdict(dict)

        for field_name, field_info in self.SCHEMA.items():
            field_value = getattr(self, field_name)
            if field_value is None:
                continue

            nested_spec = field_info.get("spec")
            if nested_spec is not None:
                nested_dim_to_field_sizes = field_value._collect_dimension_info(
                    prefix=f"{prefix}{field_name}."
                )
                self._merge_dimension_info(dim_to_field_sizes, nested_dim_to_field_sizes)
                continue

            expected_shapes = self._expected_shapes(field_info["shape"])

            matched_shape = find_matched_shape(field_value, expected_shapes)
            if matched_shape is None:
                # Child specs are already validated; skip defensively if no shape can be matched.
                continue

            self._track_named_dimensions(
                dim_to_field_sizes=dim_to_field_sizes,
                field_path=f"{prefix}{field_name}",
                matched_shape=matched_shape,
                shape=value_shape(field_value),
            )

        return dim_to_field_sizes

    def __post_init__(self):
        dim_to_field_sizes = defaultdict(dict)
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

                nested_dim_to_field_sizes = field_value._collect_dimension_info(
                    prefix=f"{field_name}."
                )
                self._merge_dimension_info(dim_to_field_sizes, nested_dim_to_field_sizes)
                continue

            self._validate_and_track_primitive_field(
                field_name=field_name,
                field_info=field_info,
                field_value=field_value,
                dim_to_field_sizes=dim_to_field_sizes,
            )

        self._raise_if_inconsistent_dimensions(dim_to_field_sizes)

    @staticmethod
    def _is_string_value(value: Any) -> bool:
        """Return True for scalar/array values that should be stored as HDF5 strings."""
        if isinstance(value, (str, np.str_, bytes, np.bytes_)):
            return True

        if isinstance(value, np.ndarray):
            return value.dtype.kind in {"U", "S", "O"}

        return False

    @staticmethod
    def _resolve_chunks(
        value: Any,
        dim_names: tuple | None,
        chunk_axes: tuple[str, ...] | None,
        max_chunk_bytes: int | None = None,
    ) -> tuple | None:
        """Choose an HDF5 chunk shape aligned with common access patterns.

        ``chunk_axes`` names the dimensions to chunk with size 1 (default
        :data:`DEFAULT_CHUNK_AXES`, ``("n_frames",)``); every other axis is stored at
        full extent, so partial/streaming reads fetch only the requested frames
        instead of h5py's poorly-shaped auto-guess. Axes named in ``chunk_axes`` but
        not present on this field are ignored (e.g. an ``image`` without ``n_tx`` is
        chunked on ``n_frames`` only).

        The result is capped to ``max_chunk_bytes`` (default :data:`MAX_CHUNK_BYTES`) by
        splitting the outermost full axis — ``n_tx`` for ``raw_data`` — which keeps each
        chunk a contiguous run of the array. Only that axis is split: if one index along
        it already exceeds the budget, the chunk is left oversized rather than cutting
        into ``n_ax``/``n_el``, which are read whole anyway.

        Returns ``None`` (contiguous / h5py default) when ``chunk_axes`` is empty
        or ``None``, the value is not a ≥2-D array, or the field's dimension names
        are unknown.
        """
        if not chunk_axes or not isinstance(value, np.ndarray) or value.ndim < 2:
            return None
        if not (
            dim_names is not None
            and len(dim_names) == value.ndim
            and all(isinstance(d, str) for d in dim_names)
        ):
            return None
        mark = [d in chunk_axes for d in dim_names]
        # Require a mix: at least one chunk axis present *and* at least one full
        # axis, else the chunks would be scalar-sized (all ones).
        if not (any(mark) and not all(mark)):
            return None

        shape = cast(Tuple[int, ...], value.shape)
        chunks: list[int] = [1 if m else dim for m, dim in zip(mark, shape)]
        if max_chunk_bytes is None:
            max_chunk_bytes = MAX_CHUNK_BYTES
        if not max_chunk_bytes:  # 0 / None disables the cap: one full frame per chunk
            return tuple(chunks)

        split = mark.index(False)  # outermost axis kept at full extent
        # Bytes of one index along that axis, i.e. everything nested inside it.
        inner = math.prod(chunks[split + 1 :]) * value.dtype.itemsize
        if inner:
            chunks[split] = min(chunks[split], max(1, max_chunk_bytes // inner))
        return tuple(chunks)

    @staticmethod
    def create_dataset(
        group: h5py.Group,
        field_name: str,
        value: Any,
        compression: "str | Mapping | None" = DEFAULT_COMPRESSION,
        chunk_axes: tuple[str, ...] | None = DEFAULT_CHUNK_AXES,
        dim_names: tuple | None = None,
    ) -> None:
        """Create a dataset in the given group for the specified field and value,
        handling string and scalar values appropriately.

        ``compression`` may be an h5py filter name (e.g. ``"lzf"``, ``"gzip"``) or
        a mapping of ``create_dataset`` keyword arguments, such as an
        ``hdf5plugin`` filter object (``hdf5plugin.Blosc2(...)``) or an explicit
        ``{"compression": "gzip", "compression_opts": 4}``. Reading such a file
        back requires the corresponding filter to be available (for ``hdf5plugin``
        filters, ``import hdf5plugin`` in the reading process).

        When ``dim_names`` (the field's schema dimension names) is provided, the
        chunk shape is derived from ``chunk_axes`` to match common subsampling
        patterns; see :meth:`_resolve_chunks`.
        """
        dataset_is_scalar = np.isscalar(value) or value.ndim == 0
        chunks = None if dataset_is_scalar else Spec._resolve_chunks(value, dim_names, chunk_axes)
        # Filters are meaningless for scalars; strings only take named codecs
        # (plugin filters like Blosc reject variable-length string data).
        if dataset_is_scalar:
            comp_kwargs: dict = {}
        elif isinstance(compression, Mapping):
            comp_kwargs = dict(compression)
        elif compression is not None:
            comp_kwargs = {"compression": compression}
        else:
            comp_kwargs = {}
        if Spec._is_string_value(value):
            string_dtype = h5py.string_dtype(encoding="utf-8")
            string_value = np.asarray(value, dtype=object)
            # Only named codecs apply to strings, and never to scalar datasets
            # (h5py rejects chunk/filter options on scalars). Plugin filters
            # (Mapping) are skipped — they reject variable-length string data.
            use_str_codec = isinstance(compression, str) and not dataset_is_scalar
            string_comp = {"compression": compression} if use_str_codec else {}
            group.create_dataset(
                field_name,
                data=string_value,
                dtype=string_dtype,
                **string_comp,
            )
        else:
            group.create_dataset(field_name, data=value, chunks=chunks, **comp_kwargs)

    def store_in_group(
        self,
        group: h5py.Group,
        compression: "str | Mapping | None" = DEFAULT_COMPRESSION,
        chunk_axes: tuple[str, ...] | None = DEFAULT_CHUNK_AXES,
        warn_missing_optional_fields: bool = True,
    ) -> None:
        """Store the data in the given group (e.g. hdf5 group)."""

        assert isinstance(group, h5py.Group), "group must be an h5py Group"

        # Optional fields should only warn when persisting to disk, not on load.
        if warn_missing_optional_fields:
            self.warn_missing_optional_fields()

        field_metadata = getattr(self, "FIELD_METADATA", {})

        for field_name, field_info in self.SCHEMA.items():
            value = getattr(self, field_name)

            # We do not store fields with value None in the file.
            if value is None:
                continue

            nested_spec = field_info.get("spec")
            if nested_spec is not None:
                subgroup = group.create_group(field_name)
                value.store_in_group(
                    subgroup,
                    compression=compression,
                    chunk_axes=chunk_axes,
                    warn_missing_optional_fields=warn_missing_optional_fields,
                )
            else:
                self.create_dataset(
                    group,
                    field_name,
                    value,
                    compression=compression,
                    chunk_axes=chunk_axes,
                    dim_names=field_info.get("shape"),
                )
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

    def __repr__(self) -> str:
        parts = []
        for field_name, field_info in self.SCHEMA.items():
            value = getattr(self, field_name, None)
            if value is None:
                continue
            nested_spec = field_info.get("spec")
            if nested_spec is not None:
                parts.append(f"{field_name}={value!r}")
            elif isinstance(value, np.ndarray):
                parts.append(f"{field_name}=array({value.dtype} {value.shape})")
            else:
                parts.append(f"{field_name}={value!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


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
            The leading ``n_frames`` axis may be omitted to broadcast one coordinate grid
            across all frames.
        timestamps: Optional per-frame acquisition timestamps in seconds, shape ``(n_frames,)``,
            relative to frame 0.  Use this to record the time of each frame when the sampling
            is irregular.  Must start at 0 and be strictly increasing.  Requires
            ``start_time_offset`` to also be provided.
        start_time_offset: Time offset in seconds between the first transmit event of the
            ultrasound acquisition and frame 0 of this map.  Negative means frame 0 was
            acquired before the first transmit event; positive means it was acquired after.
            Required when ``timestamps`` is provided.
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
    timestamps: np.ndarray | None = None
    start_time_offset: np.ndarray | float | None = None
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
                ("n_frames", "z", "x", "n_spatial_ch"),
                ("n_frames", "z", "x"),
            ),
        },
        "coordinates": {"dtype": np.float32, "shape": ("...", 3)},
        "timestamps": {"dtype": np.float32, "shape": ("n_frames",)},
        "start_time_offset": {"dtype": np.float32, "shape": ()},
        "labels": {"dtype": np.str_, "shape": ("n_spatial_ch",)},
        "description": {"dtype": str, "shape": ()},
        "unit": {"dtype": str, "shape": ()},
        "min": {"dtype": np.float32, "shape": ()},
        "max": {"dtype": np.float32, "shape": ()},
    }

    FIELD_METADATA = {
        "values": {"unit": "–", "description": "Map pixel values."},
        "coordinates": {
            "unit": "m",
            "description": "Per-pixel Cartesian positions (x, y, z) in metres.",
        },
        "timestamps": {
            "unit": "s",
            "description": "Per-frame acquisition timestamps relative to frame 0.",
            "rare": True,
        },
        "start_time_offset": {
            "unit": "s",
            "description": (
                "Time offset between the first transmit event of the ultrasound "
                "acquisition and frame 0 of this map. Negative means frame 0 was "
                "acquired before the first transmit event; positive means it was "
                "acquired after."
            ),
            "rare": True,
        },
        "labels": {"unit": "–", "description": "Labels for each channel in values.", "rare": True},
        "description": {
            "unit": "–",
            "description": "Free-text description of the map contents.",
            "rare": True,
        },
        "unit": {
            "unit": "–",
            "description": "Physical unit of the map values, e.g. 'm/s', '%'.",
            "rare": True,
        },
        "min": {"unit": "–", "description": "Minimum value of the map.", "rare": True},
        "max": {"unit": "–", "description": "Maximum value of the map.", "rare": True},
    }

    def __post_init__(self):
        super().__post_init__()

        if (self.timestamps is None) != (self.start_time_offset is None):
            raise ValueError("Map.timestamps and Map.start_time_offset must be provided together.")
        if self.timestamps is not None:
            if not np.isclose(self.timestamps[0], 0.0):
                raise ValueError("Map.timestamps must start at 0.")
            if len(self.timestamps) > 1 and np.any(np.diff(self.timestamps) <= 0):
                raise ValueError("Map.timestamps must be strictly increasing.")

        if self.values.ndim == 5:
            assert self.labels is not None, (
                "labels must be provided when values have n_ch dimension"
            )

        if self.coordinates is not None:
            # coordinates.shape[-1] is guaranteed == 3 by the SCHEMA check above.
            # Validate that the spatial axes match values (with or without a trailing channel axis).
            coords_spatial = self.coordinates.shape[:-1]
            valid_spatial_shapes = {
                self.values.shape,
                self.values.shape[:-1],
            }
            # Also accept coordinates that omit the leading frame axis and
            # therefore broadcast across frames.
            if len(self.values.shape) > 1:
                valid_spatial_shapes.add(self.values.shape[1:])
            if len(self.values.shape[:-1]) > 1:
                valid_spatial_shapes.add(self.values.shape[1:-1])
            if coords_spatial not in valid_spatial_shapes:
                raise ValueError(
                    f"{type(self).__name__}: coordinates shape {self.coordinates.shape} is "
                    f"incompatible with values shape {self.values.shape}. "
                    f"coordinates.shape[:-1] must equal values.shape "
                    f"({self.values.shape}) for non-channeled data, or "
                    f"values.shape[:-1] ({self.values.shape[:-1]}) for channeled data, "
                    "with optional frame-axis broadcasting (leading n_frames omitted)."
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
        values: The segmentation values of shape ``(n_frames, z, x, y, n_labels)`` for 3D
            (volumetric) data or ``(n_frames, z, x, n_labels)`` for 2D data, with type bool.
        coordinates: Per-pixel Cartesian positions in metres, shape ``(*spatial_dims, 3)``
            where ``spatial_dims`` matches the spatial (non-label) dimensions of ``values``.
            The leading frame axis may be omitted to broadcast one coordinate grid
            across all frames.
        labels: The labels corresponding to the segmentation values, where each unique value
            in the values corresponds to a label in this list of shape ``(n_labels,)`` and type str.

    .. note::
        To indicate that certain frames have no segmentation, add an explicit
        ``"unannotated"`` entry to ``labels`` and set ``values[..., unannotated_idx]`` to
        ``True`` for those frames (with all other label channels set to ``False``).  This
        keeps the shape uniform across frames while clearly distinguishing genuinely
        annotated frames from frames that were not labelled.  For example::

            labels = np.array(["unannotated", "LV_endo", "LV_myo", "LA"])
            values = np.zeros((n_frames, H, W, 4), dtype=bool)
            # mark all frames as unannotated by default
            values[:, :, :, 0] = True
            # for annotated frames, set unannotated channel to False
            # and the appropriate label channel to True
            values[ed_idx, :, :, 0] = False
            values[ed_idx, :, :, 1:] = segmentation_mask  # shape (H, W, 3)
    """

    SCHEMA = {
        **BooleanMap.SCHEMA,
        "values": {
            **BooleanMap.SCHEMA["values"],
            "shape": (
                ("n_frames", "z", "x", "y", "n_spatial_ch"),
                ("n_frames", "z", "x", "n_spatial_ch"),
            ),
        },
    }

    def __post_init__(self):
        assert self.values.ndim in (4, 5), (
            "Segmentation values must have 4 or 5 dimensions: "
            "(n_frames, z, x, n_labels) for 2D or (n_frames, z, x, y, n_labels) for 3D, "
            f"got shape {self.values.shape}"
        )
        assert self.labels is not None, "Segmentation requires labels to be provided"
        super().__post_init__()


@dataclass
class Image(Map):
    """Reconstructed (log-compressed) image data with per-pixel Cartesian coordinates.

    Args:
        values: The image values of shape ``(n_frames, z, x, y)`` or ``(n_frames, z, x)``
            and type uint8 or float32. For float32 values, the values should be in dB
            (between -inf and 0).
        coordinates: Per-pixel Cartesian positions in metres, shape ``(*values.shape, 3)``.
            The leading frame axis may be omitted to broadcast one coordinate grid
            across all frames.
    """

    SCHEMA = {
        **Map.SCHEMA,
        "values": {
            "dtype": (np.float32, np.uint8),
            "shape": (
                ("n_frames", "z", "x", "y"),
                ("n_frames", "z", "x"),
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
class AlignedData(Spec):
    """Time-of-flight corrected data.

    Args:
        values: The aligned data of shape ``(n_frames, n_tx, n_ax, n_el, n_ch)``
            and type float32 or int16. n_ch is 1 for RF data or 2 for IQ data.
        labels: The labels for the channel dimension, e.g. ``["RF"]`` or ``["I", "Q"]``.
            Auto-generated from n_ch if not provided.
    """

    values: np.ndarray
    labels: np.ndarray | None = None

    SCHEMA = {
        "values": {
            "dtype": (np.float32, np.int16),
            "shape": ("n_frames", "n_tx", "n_ax", "n_el", "n_ch"),
        },
        "labels": {"dtype": np.str_, "shape": ("n_ch",)},
    }

    FIELD_METADATA = {
        "values": {"unit": "–", "description": "Time-of-flight corrected channel data."},
        "labels": {"unit": "–", "description": "Channel labels, e.g. 'RF' or ['I', 'Q']."},
    }

    def __post_init__(self):
        n_ch = self.values.shape[-1]
        if n_ch not in (1, 2):
            raise ValueError(
                f"Aligned data must have n_ch ∈ {{1, 2}} (RF or IQ), "
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
class BeamformedData(FloatMap):
    """Beamformed (beamsummed) data with per-pixel Cartesian coordinates.

    Args:
        values: The beamformed data of shape ``(n_frames, z, x, n_ch)`` or
            ``(n_frames, z, x, y, n_ch)`` and type float32.
            n_ch is 1 for RF data or 2 for IQ data.
        coordinates: Per-pixel Cartesian positions in metres, shape
            ``(n_frames, z, x, 3)`` or ``(n_frames, z, x, y, 3)``.
            The leading frame axis may be omitted to broadcast one coordinate grid
            across all frames.
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
        values: The envelope data of shape ``(n_frames, z, x)`` or
            ``(n_frames, z, x, y)`` and type float32.
        coordinates: Per-pixel Cartesian positions in metres, shape ``(*values.shape, 3)``.
            The leading frame axis may be omitted to broadcast one coordinate grid
            across all frames.
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
            The leading frame axis may be omitted to broadcast one coordinate grid
            across all frames.
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
class AttenuationMap(FloatMap):
    """Acoustic attenuation map with per-pixel Cartesian coordinates.

    Acoustic attenuation describes the loss of acoustic energy as the wave
    propagates through tissue (through absorption and scattering).  Attenuation
    is frequency dependent and is modelled here with the usual power law

    .. math::

        \\alpha(f) = \\alpha_0 \\, f^{\\gamma},

    where :math:`\\alpha_0` is the per-pixel attenuation coefficient stored in
    ``values`` and :math:`\\gamma` is the (scalar) power-law exponent stored in
    ``gamma``.  Reporting the coefficient normalized by frequency makes values
    comparable across systems and transmit frequencies.

    The coefficient is stored in the spec's base units of ``dB/m/Hz`` (rather
    than the common clinical ``dB/cm/MHz``; note ``1 dB/cm/MHz = 1e-4 dB/m/Hz``).
    Strictly, when :math:`\\gamma \\neq 1` the coefficient carries the exponent in
    its units (``dB/m/Hz**gamma``); the ``dB/m/Hz`` label reflects the linear
    (:math:`\\gamma = 1`) convention.

    The exponent is close to 1 for most soft tissue (attenuation is roughly
    linear in frequency), e.g. liver :math:`\\gamma \\approx 1.14`, breast
    :math:`\\gamma \\approx 1.5`, while water / low-loss viscous media follow
    :math:`\\gamma = 2`.  It defaults to ``1.0`` (linear), which reproduces the
    plain frequency-normalized "attenuation coefficient slope".

    Args:
        values: The attenuation coefficient :math:`\\alpha_0` in ``dB/m/Hz`` of
            shape ``(n_frames, z, x, y)`` and type float32.
        coordinates: Per-pixel Cartesian positions in metres, shape
            ``(n_frames, z, x, 3)`` or ``(n_frames, z, x, y, 3)``.
            The leading frame axis may be omitted to broadcast one coordinate grid
            across all frames.
        gamma: Scalar power-law exponent :math:`\\gamma` of the frequency
            dependence :math:`\\alpha(f) = \\alpha_0 f^{\\gamma}`.  Defaults to
            ``1.0`` (linear frequency dependence).
    """

    gamma: float = 1.0

    SCHEMA = {
        **FloatMap.SCHEMA,
        "gamma": {"dtype": np.float32, "shape": ()},
    }

    FIELD_METADATA = {
        **Map.FIELD_METADATA,
        "gamma": {
            "unit": "–",
            "description": (
                "Power-law exponent of the frequency dependence alpha(f) = alpha_0 * f**gamma. "
                "1.0 is linear (soft tissue ~1-1.5, e.g. liver ~1.14), 2.0 for water."
            ),
            "rare": True,
        },
    }

    def __post_init__(self):
        super().__post_init__()

        if self.unit is not None and self.unit != "dB/m/Hz":
            raise ValueError(f"Attenuation map unit should be 'dB/m/Hz', got '{self.unit}'")

        # Attenuation coefficients describe energy loss and are therefore non-negative.
        if np.any(self.values < 0):
            log.warning(
                "Attenuation map contains negative values, which is physically unexpected "
                "for an attenuation coefficient. Please verify the values are in dB/m/Hz."
            )

        # Guard against the coefficient being supplied in the common clinical unit
        # dB/cm/MHz (= 1e-4 dB/m/Hz) instead of the spec's dB/m/Hz base unit: even
        # highly attenuating media stay well below 1e-2 dB/m/Hz.
        max_abs = float(np.max(np.abs(self.values), initial=0.0))
        if max_abs > 1e-2:
            log.warning(
                f"Attenuation map has a maximum absolute value of {max_abs:.4g} dB/m/Hz, which "
                "is unusually high.  Please verify the values are in dB/m/Hz "
                "(1 dB/cm/MHz = 1e-4 dB/m/Hz), not dB/cm/MHz."
            )

        # The power-law exponent is positive; soft tissue is ~1, water is 2.
        if self.gamma is not None and (self.gamma <= 0 or self.gamma > 2.0):
            log.warning(
                f"Attenuation map gamma={self.gamma} is outside the physically typical range "
                "(0, 2]. Soft tissue is around 1.0-1.5 and water is 2.0."
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
            raise ValueError(f"Tissue Doppler map unit should be 'm/s', got '{self.unit}'")


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
            raise ValueError(f"Color Doppler map unit should be 'm/s', got '{self.unit}'")


@dataclass(init=False)
class DataSpec(Spec):
    """Data group containing raw channels, derived pipeline products, and optional spatial maps.

    Plain-array data products:
        raw_data: Raw channel data of shape (n_frames, n_tx, n_ax, n_el, n_ch)
            and type float32 or int16.

    Grouped data products (values + optional metadata):
        - aligned_data: Time-of-flight corrected data and optional labels.
        - beamformed_data: Beamformed (beamsummed) data and per-pixel coordinates.
        - envelope_data: Envelope-detected data and per-pixel coordinates.
        - image: Reconstructed image data and per-pixel coordinates.
        - segmentation: Segmentation data and per-pixel coordinates.
        - sos_map: Speed-of-sound map data and per-pixel coordinates.
        - attenuation_map: Acoustic attenuation map data and per-pixel coordinates.
        - strain_percentage_map: Strain map data and per-pixel coordinates.
        - shear_wave_elastography_map: Shear-wave elastography data and per-pixel coordinates.
        - tissue_doppler: Tissue Doppler data and per-pixel coordinates.
        - color_doppler: Color Doppler velocity data and per-pixel coordinates.
        - \\*\\*kwargs: Any other spatially aligned map data and per-pixel coordinates.

    At least one data field (plain-array or grouped) must be provided.
    """

    # Plain-array data products
    raw_data: np.ndarray | None = None
    # Grouped data products
    aligned_data: AlignedData | dict | None = None
    beamformed_data: BeamformedData | dict | None = None
    envelope_data: EnvelopeData | dict | None = None
    image: Image | dict | None = None
    segmentation: Segmentation | dict | None = None
    sos_map: SosMap | dict | None = None
    attenuation_map: AttenuationMap | dict | None = None
    strain_percentage_map: StrainPercentageMap | dict | None = None
    shear_wave_elastography_map: ShearWaveElastographyMap | dict | None = None
    tissue_doppler: TissueDopplerMap | dict | None = None
    color_doppler: ColorDopplerMap | dict | None = None

    SCHEMA = {
        # Plain-array data products
        "raw_data": {
            "dtype": (np.float32, np.int16),
            "shape": ("n_frames", "n_tx", "n_ax", "n_el", "n_ch"),
        },
        # Grouped data products
        "aligned_data": {"spec": AlignedData},
        "beamformed_data": {"spec": BeamformedData},
        "envelope_data": {"spec": EnvelopeData},
        "image": {"spec": Image},
        "segmentation": {"spec": Segmentation},
        "sos_map": {"spec": SosMap},
        "attenuation_map": {"spec": AttenuationMap},
        "strain_percentage_map": {"spec": StrainPercentageMap},
        "shear_wave_elastography_map": {"spec": ShearWaveElastographyMap},
        "tissue_doppler": {"spec": TissueDopplerMap},
        "color_doppler": {"spec": ColorDopplerMap},
    }

    FIELD_METADATA = {
        "raw_data": {"unit": "–", "description": "Raw channel data."},
        "aligned_data": {"description": "Time-of-flight corrected data.", "rare": True},
        "beamformed_data": {"description": "Beamformed data.", "rare": True},
        "envelope_data": {"description": "Envelope-detected data.", "rare": True},
        "image": {"description": "Reconstructed image data.", "rare": True},
        "segmentation": {"description": "Segmentation data.", "rare": True},
        "sos_map": {"description": "Speed-of-sound map data.", "rare": True},
        "attenuation_map": {"description": "Acoustic attenuation map data.", "rare": True},
        "strain_percentage_map": {"description": "Strain map data.", "rare": True},
        "shear_wave_elastography_map": {
            "description": "Shear-wave elastography data.",
            "rare": True,
        },
        "tissue_doppler": {"description": "Tissue Doppler data.", "rare": True},
        "color_doppler": {"description": "Color Doppler velocity data.", "rare": True},
    }

    def __init__(
        self,
        raw_data: np.ndarray | None = None,
        aligned_data: AlignedData | dict | None = None,
        beamformed_data: BeamformedData | dict | None = None,
        envelope_data: EnvelopeData | dict | None = None,
        image: Image | dict | None = None,
        segmentation: Segmentation | dict | None = None,
        sos_map: SosMap | dict | None = None,
        attenuation_map: AttenuationMap | dict | None = None,
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
        self.attenuation_map = attenuation_map
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
                    f"Only 'raw_data' is accepted as a flat array. "
                    f"Wrap your data: {{'values': array, 'coordinates': coordinates_array}}."
                )
            setattr(self, key, value)

        # Add custom extra maps to the schema as generic Map specs, so they get validated.
        self._extra_map_keys = tuple(extra_maps.keys())
        if getattr(self, "_extra_map_keys", ()):
            self.SCHEMA = cast(
                "dict[str, Any]",
                {**self.SCHEMA, **{str(key): {"spec": Map} for key in self._extra_map_keys}},
            )

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

        # n_ch must be 1 (RF) or 2 (IQ) for raw_data (checked for aligned_data by AlignedData).
        arr = getattr(self, "raw_data", None)
        if arr is not None and isinstance(arr, np.ndarray):
            n_ch = arr.shape[-1]
            if n_ch not in (1, 2):
                raise ValueError(
                    f"'raw_data' must have n_ch ∈ {{1, 2}} (RF or IQ), "
                    f"got n_ch={n_ch} (shape {arr.shape})."
                )


@dataclass
class ScanSpec(Spec):
    """Scan group with acquisition and transmit metadata.

    All fields are aligned with the data format specification.

    Args:
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
            This is the distance from the transmit origin on the transducer to
            where the beam comes to focus. The sign and magnitude encode the
            transmit type:

            - **positive finite**: focused transmit; the beam focuses at this
              distance in front of the array.
            - **negative finite**: diverging transmit; the (virtual) source
              lies this distance behind the array.
            - **infinite** (``np.inf``): plane wave. This is the preferred,
              canonical value for plane waves in zea. ``0.0`` is also accepted
              as a plane-wave marker (e.g. raw Verasonics data stores ``0``),
              but new data should use ``np.inf``.

            See :meth:`zea.Parameters.find_transmits` for how these values are
            used to classify transmits as ``"focused"``, ``"diverging"`` or
            ``"plane"``.
        transmit_origins: The transmit origins of the transmit beams in meters of
            shape (n_tx, 3). This is the (x, y, z) position from which the beam
            is transmitted.
        polar_angles: The polar angles in radians of the transmit beams of shape (n_tx,).
        time_to_next_transmit: The time in s between subsequent transmit events.
            Shape is either (n_frames, n_tx) or flat (n_frames * n_tx - 1,).
        azimuth_angles: The azimuthal angles in radians of the transmit beams of
            shape (n_tx,).
        sound_speed: The speed of sound in meters per second.
        tgc_gain_curve: The time-gain-compensation that was applied to every
            sample in the raw_data of shape (n_ax,). Divide by this curve to
            undo the TGC.
        waveforms_one_way: One-way waveforms of shape (n_tx, .) as simulated
            by the Verasonics system. This is the waveform after being filtered
            by the transducer bandwidth once.
        waveforms_two_way: Two-way waveforms of shape (n_tx, .) as simulated
            by the Verasonics system. This is the waveform after being filtered
            by the transducer bandwidth twice.
    """

    sampling_frequency: np.ndarray | float
    center_frequency: np.ndarray[Any, np.dtype[Any]] | float
    demodulation_frequency: np.ndarray[Any, np.dtype[Any]] | float
    initial_times: np.ndarray
    t0_delays: np.ndarray
    tx_apodizations: np.ndarray
    focus_distances: np.ndarray
    transmit_origins: np.ndarray
    polar_angles: np.ndarray
    time_to_next_transmit: np.ndarray | None = None
    azimuth_angles: np.ndarray | None = None
    sound_speed: np.ndarray | float | None = None
    tgc_gain_curve: np.ndarray | None = None
    waveforms_one_way: np.ndarray | None = None
    waveforms_two_way: np.ndarray | None = None

    SCHEMA = {
        "sampling_frequency": {"dtype": np.float32, "shape": ()},
        "center_frequency": {"dtype": np.float32, "shape": ((), ("n_tx",))},
        "demodulation_frequency": {"dtype": np.float32, "shape": ((), ("n_tx",))},
        "initial_times": {"dtype": np.float32, "shape": ("n_tx",)},
        "t0_delays": {"dtype": np.float32, "shape": ("n_tx", "n_el")},
        "tx_apodizations": {"dtype": np.float32, "shape": ("n_tx", "n_el")},
        "focus_distances": {"dtype": np.float32, "shape": ("n_tx",)},
        "transmit_origins": {"dtype": np.float32, "shape": ("n_tx", 3)},
        "polar_angles": {"dtype": np.float32, "shape": ("n_tx",)},
        "time_to_next_transmit": {
            "dtype": np.float32,
            "shape": (("n_frames", "n_tx"), ("n_timing_intervals",)),
        },
        "azimuth_angles": {"dtype": np.float32, "shape": ("n_tx",)},
        "sound_speed": {"dtype": np.float32, "shape": ()},
        "tgc_gain_curve": {"dtype": np.float32, "shape": ("n_ax",)},
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
        "sampling_frequency": {"unit": "Hz", "description": "Sampling frequency."},
        "center_frequency": {
            "unit": "Hz",
            "description": "Center frequency of the transmit pulse.",
        },
        "demodulation_frequency": {"unit": "Hz", "description": "Demodulation frequency."},
        "initial_times": {"unit": "s", "description": "A/D converter start times per transmit."},
        "t0_delays": {"unit": "s", "description": "Transmit delays per element."},
        "tx_apodizations": {
            "unit": "–",
            "description": (
                "Transmit apodization per element, in [-1, 1]. 0 = element did not "
                "contribute, 1 = full contribution, negative = fired with opposite polarity."
            ),
        },
        "focus_distances": {
            "unit": "m",
            "description": (
                "Transmit focus distances. Positive = focused, negative = diverging "
                "(virtual source behind the array), ``np.inf`` = plane wave (preferred; "
                "``0`` is also accepted)."
            ),
        },
        "transmit_origins": {"unit": "m", "description": "Transmit beam origins (x, y, z)."},
        "polar_angles": {"unit": "rad", "description": "Polar angles of transmit beams."},
        "time_to_next_transmit": {"unit": "s", "description": "Time between transmit events."},
        "azimuth_angles": {"unit": "rad", "description": "Azimuthal angles of transmit beams."},
        "sound_speed": {"unit": "m/s", "description": "Speed of sound."},
        "tgc_gain_curve": {
            "unit": "–",
            "description": "Time-gain-compensation curve.",
            "rare": True,
        },
        "waveforms_one_way": {
            "unit": "V",
            "description": "One-way transmit waveforms.",
            "rare": True,
        },
        "waveforms_two_way": {
            "unit": "V",
            "description": "Two-way transmit waveforms.",
            "rare": True,
        },
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

        # Try to simplify the data by squeezing out any singleton dimensions,
        # e.g. if center_frequency is an array with all the same value
        if isinstance(self.center_frequency, np.ndarray) and self.center_frequency.ndim == 1:
            cf = cast("np.ndarray[Any, np.dtype[Any]]", self.center_frequency)
            if np.all(cf == cf[0]):
                self.center_frequency = cf[0]
        if (
            isinstance(self.demodulation_frequency, np.ndarray)
            and self.demodulation_frequency.ndim == 1
        ):
            df = cast("np.ndarray[Any, np.dtype[Any]]", self.demodulation_frequency)
            if np.all(df == df[0]):
                self.demodulation_frequency = df[0]


@dataclass
class ProbeSpec(Spec):
    """Probe hardware specification.

    Stores static, physical characteristics of the transducer that are not
    captured by the per-acquisition :class:`ScanSpec`.  All fields are
    optional so that partial information can be recorded.

    Args:
        name: Probe model identifier (e.g. ``"verasonics_l11_4v"``).
        type: Probe geometry type: ``"linear"``, ``"phased"``, ``"curved"``, etc.
        probe_center_frequency: Probe nominal centre frequency in Hz. Named
            distinctly from :attr:`ScanSpec.center_frequency` (the per-acquisition
            transmit frequency) so the two never collide when merged into a single
            :class:`zea.Parameters` object.
        probe_bandwidth_percent: Fractional bandwidth as a percentage.
        probe_geometry: Element positions in metres, shape (n_el, 3) with columns
            (x, y, z).  :attr:`n_el` and :attr:`pitch` are computed
            automatically as read-only properties from this array.
        element_width: Width of a single transducer element in metres.
        element_height: Height (elevation aperture) of a single element in metres.
        lens_sound_speed: Speed of sound in the acoustic lens in m/s.
        lens_thickness: Thickness of the acoustic lens in metres.
    """

    name: str | None = None
    type: str | None = None
    probe_center_frequency: Scalar | None = None
    probe_bandwidth_percent: Scalar | None = None
    probe_geometry: np.ndarray | None = None
    element_width: Scalar | None = None
    element_height: Scalar | None = None
    lens_sound_speed: Scalar | None = None
    lens_thickness: Scalar | None = None

    SCHEMA = {
        "name": {"dtype": str, "shape": ()},
        "type": {"dtype": str, "shape": ()},
        "probe_center_frequency": {"dtype": np.float32, "shape": ()},
        "probe_bandwidth_percent": {"dtype": np.float32, "shape": ()},
        "probe_geometry": {"dtype": np.float32, "shape": ("n_el", 3)},
        "element_width": {"dtype": np.float32, "shape": ()},
        "element_height": {"dtype": np.float32, "shape": ()},
        "lens_sound_speed": {"dtype": np.float32, "shape": ()},
        "lens_thickness": {"dtype": np.float32, "shape": ()},
    }

    FIELD_METADATA = {
        "name": {"description": "Probe model name/identifier."},
        "type": {"description": "Probe geometry type (linear, phased, curved, ...)."},
        "probe_center_frequency": {
            "unit": "Hz",
            "description": "Probe nominal centre frequency.",
        },
        "probe_bandwidth_percent": {
            "unit": "%",
            "description": "Fractional bandwidth as a percentage.",
        },
        "probe_geometry": {
            "unit": "m",
            "description": "Element positions (x, y, z) per element, shape (n_el, 3).",
        },
        "element_width": {
            "unit": "m",
            "description": "Width of a single transducer element.",
        },
        "element_height": {
            "unit": "m",
            "description": "Height (elevation aperture) of a single transducer element.",
            "rare": True,
        },
        "lens_sound_speed": {
            "unit": "m/s",
            "description": "Speed of sound in the acoustic lens.",
            "rare": True,
        },
        "lens_thickness": {
            "unit": "m",
            "description": "Thickness of the acoustic lens.",
            "rare": True,
        },
    }

    @property
    def n_el(self) -> int | None:
        """Number of transducer elements, derived from :attr:`probe_geometry`."""
        if self.probe_geometry is not None:
            return int(self.probe_geometry.shape[0])
        return None

    def __post_init__(self):
        super().__post_init__()

        if self.probe_geometry is not None:
            if self.probe_geometry.ndim != 2 or self.probe_geometry.shape[1] != 3:
                raise ValueError(
                    f"ProbeSpec: probe_geometry must have shape (n_el, 3), "
                    f"got {self.probe_geometry.shape}"
                )
            if np.any(self.probe_geometry > 1.0) or np.any(self.probe_geometry < -1.0):
                log.warning(
                    "ProbeSpec probe_geometry values extend beyond \u00b11.0 m. "
                    "Please verify the values are in metres."
                )
        if self.probe_center_frequency is not None and self.probe_center_frequency <= 0:
            raise ValueError(
                "ProbeSpec: probe_center_frequency must be positive, got "
                f"{self.probe_center_frequency}"
            )
        if self.probe_bandwidth_percent is not None and self.probe_bandwidth_percent <= 0:
            raise ValueError(
                "ProbeSpec: probe_bandwidth_percent must be positive, "
                f"got {self.probe_bandwidth_percent}"
            )
        if self.element_width is not None and self.element_width <= 0:
            raise ValueError(f"ProbeSpec: element_width must be positive, got {self.element_width}")
        if self.element_height is not None and self.element_height <= 0:
            raise ValueError(
                f"ProbeSpec: element_height must be positive, got {self.element_height}"
            )
        if self.lens_sound_speed is not None and self.lens_sound_speed <= 0:
            raise ValueError(
                f"ProbeSpec: lens_sound_speed must be positive, got {self.lens_sound_speed}"
            )
        if self.lens_thickness is not None and self.lens_thickness < 0:
            raise ValueError(
                f"ProbeSpec: lens_thickness must be non-negative, got {self.lens_thickness}"
            )


@dataclass
class Subject(Spec):
    """Subject metadata associated with the study.

    Args:
        id: Subject ID.
        type: Subject type, e.g. human, phantom, animal.
        age: Subject age in years.
        sex: Subject sex.
        weight: Subject weight in kg.
        genetic_strain: Genetic strain of an animal subject, e.g. C57BL/6N.
        fat_percentage: Subject fat percentage.
        bmi: Subject body mass index in kg/m².
    """

    id: str | None = None
    type: str | None = None
    age: np.uint8 | None = None
    sex: str | None = None
    weight: np.float32 | None = None
    genetic_strain: str | None = None
    fat_percentage: np.float32 | None = None
    bmi: np.float32 | None = None

    SCHEMA = {
        "id": {"dtype": str, "shape": ()},
        "type": {"dtype": str, "shape": ()},
        "age": {"dtype": np.uint8, "shape": ()},
        "sex": {"dtype": str, "shape": ()},
        "weight": {"dtype": np.float32, "shape": ()},
        "genetic_strain": {"dtype": str, "shape": ()},
        "fat_percentage": {"dtype": np.float32, "shape": ()},
        "bmi": {"dtype": np.float32, "shape": ()},
    }

    FIELD_METADATA = {
        "id": {"unit": "–", "description": "Subject ID. Needed for subject-wise splits."},
        "type": {"unit": "–", "description": "Subject type, e.g. human, phantom, animal."},
        "age": {"unit": "–", "description": "Subject age in years.", "rare": True},
        "sex": {"unit": "–", "description": "Subject sex.", "rare": True},
        "weight": {"unit": "kg", "description": "Subject weight.", "rare": True},
        "genetic_strain": {
            "unit": "–",
            "description": "Genetic strain (inbred line) of an animal subject, e.g. C57BL/6N.",
            "rare": True,
        },
        "fat_percentage": {"unit": "%", "description": "Subject fat percentage.", "rare": True},
        "bmi": {"unit": "kg/m²", "description": "Subject body mass index.", "rare": True},
    }

    def __post_init__(self):
        super().__post_init__()

        if self.id is not None and not self.id.strip():
            raise ValueError("Subject ID cannot be an empty string")

        if self.fat_percentage is not None and (
            self.fat_percentage < 0 or self.fat_percentage > 100
        ):
            raise ValueError(
                f"Subject fat percentage must be between 0 and 100, got {self.fat_percentage}"
            )

        if self.genetic_strain is not None and not self.genetic_strain.strip():
            raise ValueError("Subject genetic_strain cannot be an empty string")

        if self.weight is not None:
            if not np.isfinite(self.weight):
                raise ValueError(f"Subject weight must be finite, got {self.weight}")
            if self.weight <= 0:
                raise ValueError(f"Subject weight must be positive, got {self.weight} kg")
            if self.weight > 1000:
                log.warning(
                    f"Subject weight was specified as {self.weight} kg."
                    "Please verify the value and that it is in kilograms, not grams."
                )

        if self.bmi is not None:
            if not np.isfinite(self.bmi):
                raise ValueError(f"Subject BMI must be finite, got {self.bmi}")
            if self.bmi <= 0 or self.bmi > 100:
                raise ValueError(f"Subject BMI must be between 0 and 100, got {self.bmi}")
            if self.bmi < 10 or self.bmi > 60:
                log.warning(
                    f"Subject BMI of {self.bmi} kg/m² is outside the typical clinical range "
                    "(10-60). Please verify the value and that it is in kg/m²."
                )


@dataclass
class Signal(Spec):
    """Base class for additional signals with timing metadata.

    Args:
        start_time_offset: Time offset in seconds between the first transmit event
            of the ultrasound acquisition and sample 0 of this data. Negative
            means this data starts before the first transmit event; positive
            means it starts after.
        sampling_frequency: Sampling frequency in Hz for uniformly sampled data.
        timestamps: Explicit sample timestamps in seconds of shape (T,), relative
            to sample 0. Must start at 0.

    Exactly one of ``sampling_frequency`` or ``timestamps`` must be provided.
    """

    start_time_offset: np.ndarray | float
    sampling_frequency: np.ndarray | float | None = field(default=None, kw_only=True)
    timestamps: np.ndarray | None = field(default=None, kw_only=True)

    SCHEMA = {
        "start_time_offset": {"dtype": np.float32, "shape": ()},
        "sampling_frequency": {"dtype": np.float32, "shape": ()},
        "timestamps": {"dtype": np.float32, "shape": ("T",)},
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
        "timestamps": {
            "unit": "s",
            "description": "Explicit sample timestamps relative to sample 0.",
        },
    }

    def __post_init__(self):
        super().__post_init__()

        if (self.sampling_frequency is None) == (self.timestamps is None):
            raise ValueError("Provide exactly one of 'sampling_frequency' or 'timestamps'.")
        if self.sampling_frequency is not None and self.sampling_frequency <= 0:
            raise ValueError(f"Sampling frequency must be positive, got {self.sampling_frequency}")
        if self.timestamps is not None:
            signal_samples = getattr(self, "samples", None)
            if signal_samples is None:
                signal_samples = getattr(self, "translation", None)
            if signal_samples is not None and self.timestamps.shape[0] != signal_samples.shape[0]:
                raise ValueError("Timestamps must have the same length as the signal samples.")
            if not np.isclose(self.timestamps[0], 0.0):
                raise ValueError("Sample timestamps must start at 0.")
            if np.any(np.diff(self.timestamps) <= 0):
                raise ValueError("Sample timestamps must be strictly increasing.")


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
        timestamps: Explicit probe pose timestamps in seconds of shape (T,),
            relative to sample 0.
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
            "unit": "–",
            "description": (
                "Orientation associated with the transducer-tip pose in the "
                "x-lateral, y-elevation, z-axial coordinate convention, interpreted "
                "according to rotation_representation."
            ),
        },
        "rotation_representation": {
            "unit": "–",
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
        timestamps: Explicit signal timestamps in seconds of shape (T,), relative
            to sample 0.

    Exactly one of ``sampling_frequency`` or ``timestamps`` must be provided.
    """

    samples: np.ndarray

    SCHEMA = {
        "samples": {"dtype": (np.uint8, np.float32, np.int16, np.complex64), "shape": ("T",)},
        **Signal.SCHEMA,
    }

    FIELD_METADATA = {
        "samples": {"unit": "–", "description": "Signal samples."},
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
        timestamps: Explicit signal timestamps in seconds of shape (T,), relative
            to sample 0.

    Exactly one of ``sampling_frequency`` or ``timestamps`` must be provided.
    """

    samples: np.ndarray

    SCHEMA = {
        "samples": {"dtype": (np.uint8, np.float32, np.int16, np.complex64), "shape": ("T", "...")},
        **Signal.SCHEMA,
    }

    FIELD_METADATA = {
        "samples": {"unit": "–", "description": "Signal samples."},
        **Signal.FIELD_METADATA,
    }


@dataclass
class Annotations(Spec):
    """Frame-level annotations, either per frame or broadcast labels.

    Args:
        anatomy (str): Anatomy label.
        view (str): View label.
        label (str): Pathology or classification label.
        image_quality (str): Image quality label, e.g. low, mid, high.
    """

    anatomy: np.ndarray | str | None = None
    view: np.ndarray | str | None = None
    label: np.ndarray | str | None = None
    image_quality: np.ndarray | str | None = None

    SCHEMA = {
        "anatomy": {"dtype": np.str_, "shape": (("n_frames",), ())},
        "view": {"dtype": np.str_, "shape": (("n_frames",), ())},
        "label": {"dtype": np.str_, "shape": (("n_frames",), ())},
        "image_quality": {"dtype": np.str_, "shape": (("n_frames",), ())},
    }

    FIELD_METADATA = {
        "anatomy": {"unit": "–", "description": "Anatomy label."},
        "view": {"unit": "–", "description": "View label."},
        "label": {"unit": "–", "description": "Pathology or classification label."},
        "image_quality": {"unit": "–", "description": "Image quality label.", "rare": True},
    }


@dataclass(init=False)
class MetadataSpec(Spec):
    """Metadata group with subject, acquisition context, annotations, and extra signals."""

    subject: Subject | dict | None = field(default_factory=Subject)
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
        "subject": {"unit": "–", "description": "Subject associated with the study."},
        "credit": {"unit": "–", "description": "Credit or attribution for the dataset."},
        "probe_pose": {
            "unit": "–",
            "description": "Sampled probe pose at the transducer tip.",
            "rare": True,
        },
        "voice_narration": {"unit": "–", "description": "Voice narration signal.", "rare": True},
        "ecg": {"unit": "–", "description": "Electrocardiogram signal.", "rare": True},
        "text_report": {
            "unit": "–",
            "description": "Free-text report associated with the study.",
            "rare": True,
        },
        "annotations": {"unit": "–", "description": "Frame-level annotations.", "rare": True},
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
            try:
                value = SignalND(**value)
            except TypeError as e:
                raise TypeError(
                    f"You are supplying a custom 'metadata' key '{key}'. We assume that is an "
                    "N-dimensional sampled signal with timing metadata (SignalND). "
                    "Wrap your data: {'samples': array, 'start_time_offset': 0.0, "
                    "'sampling_frequency': fs}, "
                    "or maybe you were looking for another field in 'metadata'?"
                ) from e
            setattr(self, key, value)

        # Add custom extra signals to the schema as generic SignalND specs, so they get validated.
        self._extra_signal_keys = tuple(extra_signals.keys())
        if getattr(self, "_extra_signal_keys", ()):
            extra = {str(key): {"spec": SignalND} for key in self._extra_signal_keys}
            self.SCHEMA = cast("dict[str, Any]", {**self.SCHEMA, **extra})

        self.__post_init__()

    def __post_init__(self):
        super().__post_init__()


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

    FIELD_METADATA = {
        "common_midpoint_phase_error": {
            "unit": "rad",
            "description": "Common midpoint phase error.",
            "rare": True,
        },
        "coherence_factor": {
            "unit": "–",
            "description": "Coherence factor; ratio of coherent to incoherent energy (0-1).",
            "rare": True,
        },
    }


@dataclass
class TrackSpec(Spec):
    """A single acquisition track with its own data and scan parameters.

    Used inside a multi-track :class:`FileSpec` where different transmit
    sequences coexist in the same acquisition.  The ``track_schedule`` on
    ``FileSpec`` specifies the global ordering of transmits across all tracks.

    For multi-track files a human-readable ``label`` is required on every
    track so that users can identify which track is which (e.g. ``"focused"``
    vs ``"planewave"``). Further information can be provided in the ``description``
    field of the parent :class:`FileSpec`, if necessary.
    Single-track files may omit the label.

    A track must carry at least one of ``data`` or ``scan``.  ``data`` may be
    left as ``None`` to describe a transmit-only track (one that records the
    transmit sequence via ``scan`` but stores no recorded data), but only when
    ``scan`` is provided and ``transmit_only=True`` is explicitly passed.

    A transmit-only track is useful when we want to store information about a
    transmit event without any corresponding receive data, for example a shear
    wave push pulse or therapeutic ultrasound.

    Args:
        data (DataSpec | dict | None): The data for this track. May be ``None``
            for a transmit-only track (e.g. to store a shear wave push pulse),
            but only if ``scan`` is provided and ``transmit_only=True``.
        scan (ScanSpec | dict | None): The scan parameters for this track. Required when raw_data is
            present in *data*, and required when *data* is ``None``.
        label (str | None): Short human-readable name for this track (e.g. ``"focused"``
            or ``"planewave"``).  Required when the parent :class:`FileSpec`
            contains more than one track.
        transmit_only (bool): Must be explicitly set to ``True`` to construct a
            transmit-only track (``data=None`` with ``scan`` provided).
    """

    data: DataSpec | dict | None = None
    scan: ScanSpec | dict | None = None
    label: str | None = None
    transmit_only: bool = False

    SCHEMA = {
        "data": {"spec": DataSpec},
        "scan": {"spec": ScanSpec},
        "label": {"dtype": str, "shape": ()},
        "transmit_only": {"dtype": np.bool_, "shape": ()},
    }

    FIELD_METADATA = {
        # label is enforced by FileSpec for multi-track (ValueError), and legitimately
        # absent for single-track files — warning here is never useful.
        "label": {"unit": "–", "description": "Short human-readable track name.", "rare": True},
        "transmit_only": {
            "unit": "–",
            "description": (
                "Whether this track records only the transmit sequence with no "
                "corresponding receive data (e.g. a shear wave push pulse or "
                "therapeutic ultrasound)."
            ),
            "rare": True,
        },
    }

    def __post_init__(self):
        super().__post_init__()

        if self.data is None and self.scan is None:
            raise ValueError(
                "A track must have at least one of 'data' or 'scan'. "
                "'data' may be None (a transmit-only track) only when 'scan' is provided "
                "and 'transmit_only=True' is explicitly set."
            )

        if self.data is None and self.scan is not None and not self.transmit_only:
            raise ValueError(
                "'data' is None but 'transmit_only' was not set to True. "
                "Pass 'transmit_only=True' to explicitly create a transmit-only track "
                "(one that records only the transmit sequence via 'scan', with no "
                "corresponding receive data, e.g. a shear wave push pulse or "
                "therapeutic ultrasound exposure)."
            )

        if self.transmit_only and self.data is not None:
            raise ValueError(
                "'transmit_only=True' was set but 'data' is not None. "
                "A transmit-only track must not carry data."
            )

        data = self.data
        has_raw = (isinstance(data, DataSpec) and data.raw_data is not None) or (
            isinstance(data, dict) and data.get("raw_data") is not None
        )
        if has_raw and self.scan is None:
            raise ValueError("'scan' is required when 'raw_data' is provided in track data.")

        if self.label is not None and not isinstance(self.label, str):
            raise TypeError(f"'label' must be a str, got {type(self.label)}")
        if self.label is not None and not self.label.strip():
            raise ValueError("'label' must not be an empty or whitespace-only string.")

    def store_in_group(
        self,
        group: "h5py.Group",
        compression: "str | Mapping | None" = DEFAULT_COMPRESSION,
        chunk_axes: tuple[str, ...] | None = DEFAULT_CHUNK_AXES,
        warn_missing_optional_fields: bool = True,
    ) -> None:
        """Store data, scan, and label in the HDF5 group."""
        super().store_in_group(
            group,
            compression=compression,
            chunk_axes=chunk_axes,
            warn_missing_optional_fields=warn_missing_optional_fields,
        )


@dataclass
class FileSpec(Spec):
    """A dataset containing all the data, scan parameters, metadata,
    and metrics for a single acquisition.

    A ``FileSpec`` always contains at least one track.  When ``data`` and
    ``scan`` are supplied at construction time they are transparently wrapped
    into a single :class:`TrackSpec`, so all existing call-sites continue to
    work unchanged.  For multi-track files pass ``tracks`` directly.

    Args:
        data: Data for a single-track acquisition (wrapped into ``tracks[0]``).
        scan: Scan parameters for a single-track acquisition.
        tracks: Explicit list of :class:`TrackSpec` objects (multi-track mode).
            Mutually exclusive with ``data``/``scan``.
        track_schedule: 1-D int32 array of length ``n_total_tx`` giving the
            track index for each global transmit event.
        metadata: Additional metadata about the acquisition.
        metrics: Metrics computed from the acquisition.
        probe: Physical probe specification (see :class:`ProbeSpec`).  The probe
            name is stored as ``probe.name``; use :attr:`zea.File.probe_name`
            to read it back from an HDF5 file.
        us_machine: The ultrasound machine used to acquire the data.
        description: Free-text description.
        custom: Optional list of :class:`~zea.data.file.CustomElement` objects holding
            data that does not fit the zea format.  These are written to a ``custom``
            group.

    Example:
        .. doctest::

            >>> from datetime import datetime, timezone
            >>> from zea.data.spec import FileSpec
            >>> import numpy as np

            >>> dataset = FileSpec(
            ...     data={
            ...         "raw_data": np.zeros((2, 4, 64, 8, 1), dtype=np.float32),
            ...     },
            ...     scan={
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
            ...     probe={"name": "test_probe", "probe_geometry": np.zeros((8, 3))},
            ...     acquisition_time=datetime.now(timezone.utc).isoformat(),
            ... )
            >>> dataset.data.raw_data.shape
            (2, 4, 64, 8, 1)
            >>> dataset.acquisition_time is not None
            True
    """

    # NOTE: data and scan are intentionally NOT dataclass fields — they are
    # accepted as constructor kwargs and folded into tracks[0] at init time.
    # @property accessors below provide backwards-compatible single-track access.
    tracks: list = field(default_factory=list)
    track_schedule: np.ndarray | None = None
    metadata: MetadataSpec | dict = field(default_factory=MetadataSpec)
    metrics: MetricsSpec | dict = field(default_factory=MetricsSpec)
    probe: ProbeSpec | dict | None = None
    us_machine: str | None = None
    description: str | None = None
    acquisition_time: str | None = None
    custom: list = field(default_factory=list)

    # tells the SCHEMA ↔ fields consistency test that 'tracks' and 'custom' are
    # intentionally absent from SCHEMA (list types don't fit the standard SCHEMA patterns)
    _SCHEMA_EXCLUDED_FIELDS = frozenset({"tracks", "custom"})

    SCHEMA = {
        "track_schedule": {"dtype": np.int32, "shape": ("n_total_tx",)},
        "metadata": {"spec": MetadataSpec},
        "metrics": {"spec": MetricsSpec},
        "probe": {"spec": ProbeSpec},
        "us_machine": {"dtype": str, "shape": ()},
        "description": {"dtype": str, "shape": ()},
        "acquisition_time": {"dtype": str, "shape": ()},
    }

    FIELD_METADATA = {
        "acquisition_time": {
            "description": (
                "UTC acquisition timestamp in ISO 8601 format "
                "(e.g. '2026-06-12T14:30:00+00:00'). "
                "Optional — not recorded when omitted."
            ),
        },
    }

    def __init__(
        self,
        data: "DataSpec | dict | None" = None,
        scan: "ScanSpec | dict | None" = None,
        tracks: "list | None" = None,
        track_schedule: "np.ndarray | None" = None,
        metadata: "MetadataSpec | dict | None" = None,
        metrics: "MetricsSpec | dict | None" = None,
        probe_name: "str | None" = None,
        probe: "ProbeSpec | dict | None" = None,
        us_machine: "str | None" = None,
        description: "str | None" = None,
        acquisition_time: "str | None" = None,
        custom: "list | None" = None,
    ):
        if data is not None or scan is not None:
            if tracks:
                raise ValueError(
                    "Provide either 'data'/'scan' (single-track shorthand) "
                    "or 'tracks' (multi-track), not both."
                )
            _implicit_track: "dict | None" = {"data": data, "scan": scan}
        else:
            _implicit_track = None

        if probe_name is not None:
            raise TypeError(
                "probe_name is not a FileSpec parameter. "
                "Use probe={'name': ...} to specify the probe name."
            )

        self.tracks = list(tracks) if tracks is not None else []
        self.track_schedule = track_schedule
        self.metadata = metadata if metadata is not None else MetadataSpec()
        self.metrics = metrics if metrics is not None else MetricsSpec()
        self.probe = probe
        self.us_machine = us_machine
        self.description = description
        self.acquisition_time = acquisition_time
        self.custom = list(custom) if custom else []

        self.__post_init__(_implicit_track)

    # ------------------------------------------------------------------
    # Backwards-compat read properties (single-track files only)
    # ------------------------------------------------------------------

    @property
    def data(self) -> "DataSpec":
        """Return the :class:`DataSpec` of the single track.

        Raises :exc:`AttributeError` when the file has more than one track —
        use ``spec.tracks[i].data`` instead.
        """
        if len(self.tracks) != 1:
            raise AttributeError(
                f"'data' is only available for single-track FileSpecs "
                f"({len(self.tracks)} tracks present). Use spec.tracks[i].data."
            )
        return self.tracks[0].data

    @property
    def scan(self) -> "ScanSpec | None":
        """Return the :class:`ScanSpec` of the single track.

        Raises :exc:`AttributeError` when the file has more than one track —
        use ``spec.tracks[i].scan`` instead.
        """
        if len(self.tracks) != 1:
            raise AttributeError(
                f"'scan' is only available for single-track FileSpecs "
                f"({len(self.tracks)} tracks present). Use spec.tracks[i].scan."
            )
        return self.tracks[0].scan

    def __post_init__(self, _implicit_track: "dict | None" = None):
        # Fold implicit data/scan into a TrackSpec if provided
        if _implicit_track is not None:
            self.tracks = [TrackSpec(**_implicit_track)]

        if not self.tracks:
            raise ValueError("A FileSpec must contain at least one track.")

        # Create TrackSpecs from dictionaries in the tracks list, if needed, and validate all tracks
        track_specs = []
        for i, t in enumerate(self.tracks):
            if isinstance(t, dict):
                try:
                    t = TrackSpec(**t)
                except (TypeError, ValueError) as e:
                    raise type(e)(f"In tracks[{i}]: {e}") from e
            elif not isinstance(t, TrackSpec):
                raise TypeError(f"tracks[{i}] must be a TrackSpec or dict, got {type(t)}")
            track_specs.append(t)
        self.tracks = track_specs

        # If any track contains raw_data, the file must define probe_geometry so
        # the acquisition can be beamformed.
        def _track_has_raw(track):
            d = track.data
            return (isinstance(d, DataSpec) and d.raw_data is not None) or (
                isinstance(d, dict) and d.get("raw_data") is not None
            )

        if any(_track_has_raw(t) for t in self.tracks):
            probe = self.probe
            probe_geometry = (
                probe.probe_geometry
                if isinstance(probe, ProbeSpec)
                else (probe.get("probe_geometry") if isinstance(probe, dict) else None)
            )
            if probe_geometry is None:
                raise ValueError(
                    "'probe_geometry' is required when 'raw_data' is provided in track data."
                )

        # For multi-track files every track must have a label so users can
        # identify tracks by name rather than relying on numeric indices.
        if len(self.tracks) > 1:
            missing = [i for i, t in enumerate(self.tracks) if not t.label]
            if missing:
                raise ValueError(
                    f"All tracks in a multi-track file must have a 'label'. "
                    f"Missing label for track(s) at index: {missing}. "
                    f"Provide a short descriptive name for each track, e.g. "
                    f"'focused' or 'planewave', so that "
                    f"File.get_track(label) and File.track_labels work correctly."
                )

        # Validate track_schedule indices are in range
        if self.track_schedule is not None:
            n_tracks = len(self.tracks)
            if not np.all((self.track_schedule >= 0) & (self.track_schedule < n_tracks)):
                raise ValueError(
                    f"All track_schedule indices must be in [0, {n_tracks - 1}], "
                    f"got min={self.track_schedule.min()}, max={self.track_schedule.max()}"
                )

        self._normalize_time_to_next_transmit()

        # Warn if multi-track frame counts differ without a schedule
        if len(self.tracks) > 1 and self.track_schedule is None:
            frame_counts = []
            for track in self.tracks:
                rd = (
                    track.data.raw_data
                    if isinstance(track.data, DataSpec)
                    else (track.data.get("raw_data") if isinstance(track.data, dict) else None)
                )
                if rd is not None and hasattr(rd, "shape"):
                    frame_counts.append(rd.shape[0])
            if len(set(frame_counts)) > 1:
                log.warning(
                    "Tracks have different numbers of frames "
                    f"({frame_counts}). Without a 'track_schedule' it is "
                    "ambiguous how frames correspond across tracks. Consider "
                    "passing 'track_schedule' to make the relationship explicit."
                )

        # Run base SCHEMA validation (metadata, metrics, scalars, track_schedule)
        super().__post_init__()

        # Per-frame metadata (e.g. annotations) describes one acquisition, so for
        # each shared dimension it must match at least one track - auxiliary tracks may
        # legitimately differ.
        if isinstance(self.metadata, MetadataSpec):
            meta_dim_field_sizes = self.metadata._collect_dimension_info("metadata.")
            per_track_dim_field_sizes = [
                track._collect_dimension_info(f"tracks[{i}].")
                for i, track in enumerate(self.tracks)
            ]
            for dim in CONSISTENCY_DIMENSIONS:
                # Nothing to validate if the metadata doesn't use this dimension.
                if dim not in meta_dim_field_sizes:
                    continue

                # Nor if no track carries it - there's nothing to match against.
                tracks_with_dim = [t[dim] for t in per_track_dim_field_sizes if dim in t]
                if not tracks_with_dim:
                    continue

                # Metadata is internally consistent, so meta_values is a single
                # size; it must match at least one track that carries this dim.
                meta_sizes = meta_dim_field_sizes[dim]
                meta_values = set(meta_sizes.values())
                if any(meta_values == set(track_sizes.values()) for track_sizes in tracks_with_dim):
                    continue
                # No track matched: report the metadata field(s) alongside every
                # track that carries the dimension.
                field_sizes = dict(meta_sizes)
                for track_sizes in tracks_with_dim:
                    field_sizes.update(track_sizes)
                raise ValueError(self._format_inconsistent_dimension(dim, field_sizes))

        # Validate custom elements are CustomElement instances
        if self.custom:
            from zea.data.file import CustomElement, _validate_custom_element_naming

            for i, element in enumerate(self.custom):
                if not isinstance(element, CustomElement):
                    raise TypeError(
                        f"custom[{i}] must be a CustomElement, got {type(element).__name__}."
                    )
                _validate_custom_element_naming(element, i)

    def _normalize_time_to_next_transmit(self) -> None:
        """Pad flat timing arrays and reshape to (n_frames * n_tx) by padding last
        frame with a zero."""
        for i, track in enumerate(self.tracks):
            raw_data = track.data.raw_data if track.data is not None else None
            scan = track.scan
            if raw_data is None or scan is None or scan.time_to_next_transmit is None:
                continue

            matrix_shape = raw_data.shape[:2]
            expected_flat_count = max(int(np.prod(matrix_shape)) - 1, 0)
            expected_flat_shape = (expected_flat_count,)
            t2nt = np.asarray(scan.time_to_next_transmit, dtype=np.float32)

            if t2nt.shape == matrix_shape:
                scan.time_to_next_transmit = t2nt
                continue

            if t2nt.shape != expected_flat_shape:
                raise ValueError(
                    f"tracks[{i}].scan.time_to_next_transmit has shape {t2nt.shape}, "
                    f"expected {matrix_shape} or flat length {expected_flat_count}."
                )

            if (
                len(self.tracks) > 1
                and self.track_schedule is not None
                and len(self.track_schedule) > 0
                and int(self.track_schedule[-1]) != i
            ):
                raise ValueError(
                    f"tracks[{i}].scan.time_to_next_transmit omits the final interval, "
                    "but this track is not the final track in track_schedule. "
                    f"Provide a full {matrix_shape} matrix for this track."
                )

            scan.time_to_next_transmit = np.pad(t2nt, (0, 1)).reshape(matrix_shape)

    def to_dict(self) -> dict:
        """Return this spec as a nested dictionary.

        Includes all :attr:`SCHEMA` fields plus the ``tracks`` list.
        """
        result = super().to_dict()
        result["tracks"] = [t.to_dict() for t in self.tracks]
        return result

    def save(
        self,
        path: str,
        compression: "str | Mapping | None" = DEFAULT_COMPRESSION,
        chunk_axes: tuple[str, ...] | None = DEFAULT_CHUNK_AXES,
        warn_missing_optional_fields: bool = True,
    ) -> None:
        """Save the dataset to the specified path."""
        try:
            _zea_version = _get_pkg_version("zea")
        except PackageNotFoundError:
            _zea_version = "dev"

        # HDF5 requires chunked storage for compression. With no chunk_axes we
        # emit contiguous datasets, so h5py would silently fall back to its
        # (poorly-shaped) auto-guess when compression is on — warn instead.
        if not chunk_axes and compression is not None:
            log.warning(
                f"chunk_axes is empty but compression={compression!r} is enabled; "
                "HDF5 requires chunking for compression, so h5py will auto-pick chunk "
                "shapes (often poor for partial/streamed reads). Pass compression=None "
                "for contiguous storage, or set chunk_axes to the dimensions to chunk."
            )

        _path = Path(path)
        _path.parent.mkdir(parents=True, exist_ok=True)

        subject_type = None
        if isinstance(self.metadata, MetadataSpec):
            subject = self.metadata.subject
            if isinstance(subject, Subject) and subject.type is not None:
                subject_type = str(subject.type).strip().casefold()

        is_human = subject_type == "human"

        if self.acquisition_time is not None:
            try:
                dt = datetime.fromisoformat(self.acquisition_time)
            except ValueError as e:
                raise ValueError(f"Invalid acquisition_time: {e}") from e
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            self.acquisition_time = dt.astimezone(timezone.utc).isoformat()
            if is_human:
                log.warning(
                    "PHI WARNING: 'acquisition_time' is set for a human subject. "
                    "Recording acquisition timestamps for human data constitutes "
                    "Protected Health Information (PHI) under HIPAA and similar "
                    "regulations. Ensure you have appropriate authorization and "
                    "de-identification measures in place before sharing this file."
                )
        # Write to a temporary file in the destination directory, then atomically
        # rename it into place.
        fd, tmp_name = tempfile.mkstemp(
            dir=str(_path.parent), prefix=f".{_path.stem}.tmp-", suffix=".hdf5"
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            self._write_hdf5(
                tmp_path, _zea_version, compression, chunk_axes, warn_missing_optional_fields
            )
            os.replace(tmp_path, _path)
        except BaseException:
            # Includes KeyboardInterrupt/SystemExit: clean up the partial temp file.
            tmp_path.unlink(missing_ok=True)
            raise

        log.info(f"File saved to {log.yellow(path)}")

    def _write_hdf5(
        self,
        path: Path,
        zea_version: str,
        compression: "str | Mapping | None",
        chunk_axes: tuple[str, ...] | None,
        warn_missing_optional_fields: bool,
    ) -> None:
        """Write all groups/datasets of this spec to a fresh HDF5 file at ``path``."""
        from zea import File

        with File(str(path), "w", **PAGED_LAYOUT) as f:
            f.attrs["zea_version"] = zea_version

            # Write scalar/array metadata fields (metadata, metrics, probe_name, etc.)
            for group_name, schema in self.SCHEMA.items():
                if "spec" in schema:
                    value: Spec = getattr(self, group_name)
                    if value is None:
                        continue
                    group = f.create_group(group_name)
                    value.store_in_group(
                        group,
                        compression=compression,
                        chunk_axes=chunk_axes,
                        warn_missing_optional_fields=warn_missing_optional_fields,
                    )
                else:
                    value = getattr(self, group_name)
                    if value is not None:
                        if group_name == "track_schedule":
                            # Array field — store as dataset, not attr
                            self.create_dataset(f, group_name, value, compression=compression)
                        else:
                            f.attrs[group_name] = value

            # Write tracks (always at least one)
            tracks_group = f.create_group("tracks")
            for i, track in enumerate(self.tracks):
                track_group = tracks_group.create_group(f"track_{i}")
                track.store_in_group(
                    track_group,
                    compression=compression,
                    chunk_axes=chunk_axes,
                    warn_missing_optional_fields=warn_missing_optional_fields,
                )

            # Write any custom (non-spec) elements into the 'custom' group, reusing the
            # standard Spec dataset-writing logic (scalar/string/compression handling).
            if self.custom:
                custom_group = f.create_group("custom")
                custom_group.attrs["description"] = (
                    "This group contains custom elements not in the zea format, added by the user."
                )
                for element in self.custom:
                    group = (
                        custom_group.require_group(element.group_name)
                        if element.group_name
                        else custom_group
                    )
                    self.create_dataset(
                        group, element.name, np.asarray(element.data), compression=compression
                    )
                    group[element.name].attrs["description"] = element.description
                    group[element.name].attrs["unit"] = element.unit
