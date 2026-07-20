"""
This module provides some utilities to edit zea data files, either individually or in bulk.

Each operation is available both as a Python function and as a ``zea data`` command line
subcommand. See the :doc:`CLI documentation </cli>` for the available operations and their
command-line usage.
"""

import functools
from collections.abc import Sequence
from copy import copy as shallow_copy
from pathlib import Path
from typing import TypeVar

import numpy as np
import tyro
from tqdm import tqdm

from zea import File, log
from zea.data.datasets import Dataset
from zea.data.spec import (
    CONSISTENCY_DIMENSIONS,
    FileSpec,
    ScanSpec,
    Spec,
    find_matched_shape,
)
from zea.internal.checks import _IMAGE_DATA_TYPES, _NON_IMAGE_DATA_TYPES
from zea.internal.preset_utils import HF_PREFIX, _hf_resolve_path

ALL_DATA_TYPES_EXCEPT_RAW = set(_IMAGE_DATA_TYPES + _NON_IMAGE_DATA_TYPES) - {"raw_data"}

SpecT = TypeVar("SpecT", bound=Spec)

# Data products stored log-compressed (in dB), which must be averaged in the linear domain.
_LOG_DOMAIN_FIELDS = frozenset({"image"})

# dB conversion factor: zea's ``log_compress`` uses ``20 * log10(x)`` (see
# ``zea.func.ultrasound.log_compress``), so the inverse is ``10 ** (dB / 20)``.
_DB_FACTOR = 20.0

OPERATION_NAMES = [
    "sum",
    "compound_frames",
    "compound_transmits",
    "resave",
    "extract",
    "summary",
    "copy",
]


def _iter_folder_io(input_path: str | Path, output_path: str | Path):
    """Yields ``(input_file, output_file)`` path pairs for a folder operation.

    Uses :class:`zea.Dataset` to iterate over every zea file in ``input_path``. The
    output folder mirrors the structure of the input folder.

    Args:
        input_path (Path): Path to a folder containing zea data files.
        output_path (Path): Path to the output folder.

    Yields:
        tuple[Path, Path]: Pairs of (input file, output file) paths.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    with Dataset(input_path, validate=False) as dataset:
        for file in dataset:
            yield file.path, output_path / file.path.relative_to(input_path)


def _supports_folders(operation):
    """Decorator that lets a single-file operation also accept a folder as input.

    When the decorated operation is called with a folder as ``input_path``, it is
    applied to every zea file in that folder (iterated with :class:`zea.Dataset`),
    writing the results to ``output_path`` and mirroring the input folder structure.
    A single file is processed as before.

    ``input_path`` may also be an ``hf://`` path (pointing at a single file or a
    folder in a Hugging Face dataset repo); it is downloaded via
    :func:`zea.internal.preset_utils._hf_resolve_path` before dispatching.

    Note that if a file operation does not need all the data, this might not be optimal since
    it doesn't stream the data and instead downloads the whole file.
    """

    @functools.wraps(operation)
    def wrapper(input_path, output_path, *args, **kwargs):
        input_path = str(input_path)
        if input_path.startswith(HF_PREFIX):
            input_path = _hf_resolve_path(input_path)
        if not Path(input_path).is_dir():
            return operation(input_path, output_path, *args, **kwargs)
        output_path = Path(output_path)
        if output_path.is_file():
            raise NotADirectoryError(
                f"Input {input_path} is a folder, so output {output_path} must be a "
                "folder, but it is an existing file."
            )
        for in_path, out_path in tqdm(list(_iter_folder_io(input_path, output_path))):
            operation(in_path, out_path, *args, **kwargs)
        return None

    return wrapper


def _data_arrays(track) -> dict:
    """Return every data-product array of a track, keyed by field name.

    Map-based products (``image``, ``beamformed_data``, ...) store their array in
    ``values``; plain products (``raw_data``) are the array itself. The array is
    returned together with the SCHEMA shape that describes it, so callers can
    resolve named dimensions against it.
    """
    arrays = {}
    if track.data is None:
        return arrays

    for field_name, field_info in track.data.SCHEMA.items():
        value = getattr(track.data, field_name)
        if value is None:
            continue
        nested = field_info.get("spec")
        if nested is None:
            arrays[field_name] = (value, field_info["shape"])
        else:
            arrays[field_name] = (value.values, type(value).SCHEMA["values"]["shape"])
    return arrays


def _set_data_array(track, field_name: str, array: np.ndarray) -> None:
    """Write ``array`` back into a track's data product (inverse of :func:`_data_arrays`)."""
    value = getattr(track.data, field_name)
    if isinstance(value, Spec):
        value.values = array
    else:
        setattr(track.data, field_name, array)


def _dim_axis(array: np.ndarray, shape_spec, dim_name: str) -> int | None:
    """The single axis of ``array`` carrying ``dim_name``, or None if it has none."""
    matched_shape = find_matched_shape(array, Spec._expected_shapes(shape_spec))
    if matched_shape is None:
        return None
    axes = _named_axes(matched_shape, array.ndim, dim_name)
    return axes[0] if axes else None


def _mean_over_axis(array: np.ndarray, axis: int, log_domain: bool = False) -> np.ndarray:
    """Average ``array`` along ``axis``, keeping the axis, in the domain the data lives in.

    float32 images are log-compressed (dB), so they are averaged in the linear domain;
    uint8 images are averaged in float and clipped back into range.
    """
    if log_domain and array.dtype == np.float32:
        # Undo the dB compression (10**(dB/20)), average in the linear domain, re-compress.
        linear = np.power(10.0, array / _DB_FACTOR)
        mean = np.mean(linear, axis=axis, keepdims=True)
        return (_DB_FACTOR * np.log10(mean)).astype(np.float32)
    if array.dtype == np.uint8:
        mean = np.mean(array.astype(np.float32), axis=axis, keepdims=True)
        return np.clip(mean, 0, 255).astype(np.uint8)
    return np.mean(array, axis=axis, keepdims=True).astype(array.dtype)


def _compound_named_dim(file_spec: FileSpec, dim_name: str) -> FileSpec:
    """Average every data product over ``dim_name``, collapsing that dimension to length 1.

    Data products are averaged; everything else that carries the dimension (scan
    parameters, timestamps, annotations, metrics) is reduced to its first entry, so
    the whole spec stays consistent. Which fields carry the dimension is decided by
    the SCHEMA, so products without it (e.g. ``image`` has no ``n_tx``) are left alone.
    """
    # Averages are computed from the unsliced spec, then written into the reduced one.
    means = []
    for track in file_spec.tracks:
        means.append(
            {
                field_name: _mean_over_axis(
                    array, axis, log_domain=field_name in _LOG_DOMAIN_FIELDS
                )
                for field_name, (array, shape_spec) in _data_arrays(track).items()
                if (axis := _dim_axis(array, shape_spec, dim_name)) is not None
            }
        )

    compounded = slice_spec_dims(file_spec, **{dim_name: [0]})

    for track, track_means in zip(compounded.tracks, means):
        for field_name, array in track_means.items():
            _set_data_array(track, field_name, array)

    compounded.__post_init__()
    return compounded


def sum_data(input_paths: Sequence[str | Path], output_path: str | Path, overwrite=False):
    """
    Sums multiple raw data files and saves the result to a new file.

    For images, this will actually average the images. If the images are uint8, it will average
    directly. If the images are float32, we assume they are in the log-domain and we will do the
    averaging in the linear domain.

    Args:
        input_paths (list[str, Path]): List of paths to the input raw data files. Each path
            may be a single file, a folder, or an ``hf://`` path; folders (local or ``hf://``)
            are expanded into all zea files they contain (using :class:`zea.Dataset`).
        output_path (Path): Path to the output file where the summed data will be saved.
        overwrite (bool, optional): Whether to overwrite the output file if it exists.
            Defaults to False.
    """

    _prepare_output_path(str(output_path), overwrite)

    with Dataset(input_paths, validate=False) as dataset:
        input_paths = [file.path for file in dataset]

    with File(input_paths[0]) as f:
        file_spec = f._to_file_spec()

    # Images are accumulated in the linear domain (float32 dB) or in float (uint8) and
    # converted back to an average once every file has been added.
    totals = []
    for track in file_spec.tracks:
        track_totals = {}
        for field_name, (array, _) in _data_arrays(track).items():
            if field_name in _LOG_DOMAIN_FIELDS:
                track_totals[field_name] = (
                    np.power(10.0, array / _DB_FACTOR)
                    if array.dtype == np.float32
                    else array.astype(np.float32)
                )
            elif np.issubdtype(array.dtype, np.integer):
                # Widen integer accumulators (raw_data may be int16) so summing many files
                # cannot overflow; the sum is validated and cast back to the input dtype below.
                track_totals[field_name] = array.astype(np.int64)
            else:
                track_totals[field_name] = array.copy()
        totals.append(track_totals)

    for path in input_paths[1:]:
        with File(path) as f:
            other = f._to_file_spec()

        if len(other.tracks) != len(file_spec.tracks):
            raise ValueError(
                f"{path} has {len(other.tracks)} track(s), but {input_paths[0]} has "
                f"{len(file_spec.tracks)}. Only files with matching tracks can be summed."
            )

        for i, (track, other_track) in enumerate(zip(file_spec.tracks, other.tracks)):
            _assert_scans_equal(track.scan, other_track.scan, f"tracks[{i}].scan")

            other_arrays = _data_arrays(other_track)
            extra_fields = set(other_arrays) - set(totals[i])
            if extra_fields:
                raise ValueError(
                    f"{path} has field(s) {sorted(extra_fields)} not present in "
                    f"{input_paths[0]}. Only files with matching data products can be summed."
                )
            for field_name, total in totals[i].items():
                if field_name not in other_arrays:
                    raise ValueError(
                        f"{path} is missing '{field_name}', present in {input_paths[0]}."
                    )
                array, _ = other_arrays[field_name]
                _assert_shapes_equal(total, array, field_name)
                if field_name in _LOG_DOMAIN_FIELDS:
                    total += (
                        np.power(10.0, array / _DB_FACTOR)
                        if array.dtype == np.float32
                        else array.astype(np.float32)
                    )
                else:
                    total += array

    n_files = len(input_paths)
    for track, track_totals in zip(file_spec.tracks, totals):
        field_dtypes = {name: array.dtype for name, (array, _) in _data_arrays(track).items()}
        for field_name, total in track_totals.items():
            dtype = field_dtypes[field_name]
            if field_name in _LOG_DOMAIN_FIELDS:
                if dtype == np.float32:
                    # Back to dB (20*log10), and clamp: log-compressed images are <= 0 dB.
                    total = np.minimum(_DB_FACTOR * np.log10(total / n_files), 0.0).astype(
                        np.float32
                    )
                else:
                    total = np.clip(total / n_files, 0, 255).astype(np.uint8)
            elif np.issubdtype(dtype, np.integer):
                # Cast the widened sum back to the input dtype, refusing silent overflow.
                info = np.iinfo(dtype)
                if total.min() < info.min or total.max() > info.max:
                    raise ValueError(
                        f"Summed '{field_name}' does not fit {np.dtype(dtype).name}: value "
                        f"range [{int(total.min())}, {int(total.max())}] exceeds "
                        f"[{info.min}, {info.max}]."
                    )
                total = total.astype(dtype)
            _set_data_array(track, field_name, total)

    file_spec.__post_init__()

    file_spec.save(str(output_path))


def _assert_shapes_equal(array0, array1, name="array"):
    # Raise explicitly (not ``assert``) so validation survives ``python -O``.
    shape0, shape1 = array0.shape, array1.shape
    if shape0 != shape1:
        raise ValueError(f"{name} shapes do not match. Got {shape0} and {shape1}.")


def _assert_scans_equal(scan, other_scan, name="scan"):
    """Check two ScanSpecs describe the same acquisition, field by field."""
    # Raise explicitly (not ``assert``) so validation survives ``python -O``.
    if scan is None or other_scan is None:
        if scan is not other_scan:
            raise ValueError(f"{name}: one file has scan parameters and the other does not.")
        return

    for field_name in scan.SCHEMA:
        value, other_value = getattr(scan, field_name), getattr(other_scan, field_name)
        if not np.array_equal(value, other_value):
            raise ValueError(
                f"{name}.{field_name} does not match across the summed files: "
                f"{value} vs {other_value}."
            )


@_supports_folders
def compound_frames(input_path: str | Path, output_path: str | Path, overwrite=False):
    """
    Compounds frames in a raw data file by averaging them.

    Args:
        input_path (str, Path): Path to the input raw data file, or a folder of files.
            Also accepts an ``hf://`` path (file or folder).
        output_path (Path): Path to the output file (or folder) where the compounded
            data will be saved.
        overwrite (bool, optional): Whether to overwrite the output file if it exists.
            Defaults to False.
    """

    _prepare_output_path(str(output_path), overwrite)

    with File(input_path) as f:
        file_spec = f._to_file_spec()

    file_spec = _compound_named_dim(file_spec, "n_frames")

    file_spec.save(str(output_path))


@_supports_folders
def compound_transmits(input_path: str | Path, output_path: str | Path, overwrite=False):
    """
    Compounds transmits in a raw data file by averaging them.

    Note:
        This function assumes that all transmits are identical. If this is not the case the
        function will result in incorrect scan parameters.

    Args:
        input_path (str, Path): Path to the input raw data file, or a folder of files.
            Also accepts an ``hf://`` path (file or folder).
        output_path (Path): Path to the output file (or folder) where the compounded
            data will be saved.
        overwrite (bool, optional): Whether to overwrite the output file if it exists.
            Defaults to False.
    """

    _prepare_output_path(str(output_path), overwrite)

    with File(input_path) as f:
        file_spec = f._to_file_spec()

    for i, track in enumerate(file_spec.tracks):
        if track.scan is not None and not _all_tx_are_identical(track.scan):
            log.warning(
                f"Not all transmits in track {i} are identical. Compounding transmits may "
                "lead to unexpected results."
            )

    file_spec = _compound_named_dim(file_spec, "n_tx")

    file_spec.save(str(output_path))


def _all_tx_are_identical(scan: ScanSpec):
    """Checks if all transmits in a ScanSpec are identical."""
    attributes_to_check = [
        scan.polar_angles,
        scan.azimuth_angles,
        scan.t0_delays,
        scan.tx_apodizations,
        scan.focus_distances,
        scan.transmit_origins,
        scan.initial_times,
    ]

    for attr in attributes_to_check:
        if attr is not None and not _check_all_identical(attr, axis=0):
            return False
    return True


def _check_all_identical(array, axis=0):
    """Checks if all elements along a given axis are identical."""
    first = array.take(0, axis=axis)
    return np.all(np.equal(array, first), axis=axis).all()


@_supports_folders
def resave(
    input_path: str | Path,
    output_path: str | Path,
    overwrite=False,
    **kwargs,
):
    """
    Resaves a zea data file to a new location.

    Args:
        input_path (str, Path): Path to the input zea data file, or a folder of files.
            Also accepts an ``hf://`` path (file or folder).
        output_path (Path): Path to the output file (or folder) where the data will be saved.
        overwrite (bool, optional): Whether to overwrite the output file if it exists.
            Defaults to False.
        chunk_axes (tuple, optional): Dimension names to chunk with size 1. Defaults
            to ``("n_frames",)`` — one full frame per chunk — so partial and streamed
            reads fetch only the requested frames; other axes stay at full extent. Use
            ``None``/``()`` for contiguous storage. See
            :meth:`zea.data.spec.Spec._resolve_chunks`.
    """

    _prepare_output_path(str(output_path), overwrite)

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist.")

    with File(input_path) as f:
        file_spec = f._to_file_spec()

    file_spec.save(str(output_path), **kwargs)


def _is_select_all(index) -> bool:
    """Whether ``index`` selects everything, so slicing along it is a no-op."""
    return isinstance(index, slice) and index == slice(None)


def _named_axes(matched_shape: tuple, ndim: int, dim_name: str) -> list[int]:
    """Axes of a value's actual shape that carry ``dim_name`` in its SCHEMA shape.

    Handles the ``"..."`` wildcard: dimensions before it are counted from the front
    of the value's shape, dimensions after it from the back.
    """
    if "..." not in matched_shape:
        return [axis for axis, dim in enumerate(matched_shape) if dim == dim_name]

    ellipsis_axis = matched_shape.index("...")
    prefix, suffix = matched_shape[:ellipsis_axis], matched_shape[ellipsis_axis + 1 :]
    return [axis for axis, dim in enumerate(prefix) if dim == dim_name] + [
        ndim - len(suffix) + axis for axis, dim in enumerate(suffix) if dim == dim_name
    ]


def _as_indices(index, size: int) -> np.ndarray:
    """Normalize any supported index into a 1-D integer array.

    A plain ``int`` is kept as a length-1 array so that the axis survives the
    selection: dropping it would break the SCHEMA shape of the field.
    """
    return np.atleast_1d(np.arange(size)[index])


def _slice_map_coordinates(spec: Spec, sliced: Spec, index) -> None:
    """Slice a :class:`~zea.data.spec.Map`'s coordinates along its frame axis, if it has one.

    Coordinates are optional and may either carry a leading ``n_frames`` axis or omit it
    and broadcast across frames. Only the former is sliced; the shapes accepted here are
    the frame-carrying ones from :meth:`Map.__post_init__`.
    """
    coordinates = getattr(spec, "coordinates", None)
    values = getattr(spec, "values", None)
    if index is None or coordinates is None or values is None:
        return

    if coordinates.shape[:-1] in (values.shape, values.shape[:-1]):
        indices = _as_indices(index, coordinates.shape[0])
        setattr(sliced, "coordinates", np.take(coordinates, indices, axis=0))


def _slice_spec_dims(spec: SpecT, dim_indices: dict) -> SpecT:
    """Recursive worker for :func:`slice_spec_dims` (no re-validation)."""
    sliced = shallow_copy(spec)
    sliced_timestamps = False

    for field_name, field_info in spec.SCHEMA.items():
        value = getattr(spec, field_name)
        if value is None:
            continue

        if field_info.get("spec") is not None:
            setattr(sliced, field_name, _slice_spec_dims(value, dim_indices))
            continue

        # Which of the SCHEMA's candidate shapes applies is decided by the value itself,
        # so e.g. a scalar center_frequency is left alone while a per-transmit one is sliced.
        matched_shape = find_matched_shape(value, Spec._expected_shapes(field_info["shape"]))
        if matched_shape is None:
            continue

        for dim_name, index in dim_indices.items():
            for axis in _named_axes(matched_shape, np.ndim(value), dim_name):
                # One axis at a time, so multiple index *lists* do not broadcast
                # against each other the way they would in a single fancy-index.
                value = np.take(value, _as_indices(index, value.shape[axis]), axis=axis)
                sliced_timestamps |= field_name == "timestamps"

        setattr(sliced, field_name, value)

    # Map.coordinates is declared as ("...", 3): it names no dimension, and its leading
    # frame axis is optional (coordinates may broadcast across frames). So whether it
    # carries n_frames has to be decided against values, the way Map.__post_init__ does.
    _slice_map_coordinates(spec, sliced, dim_indices.get("n_frames"))

    # Map.timestamps must start at zero, so re-anchor the selection and push the
    # dropped lead time into start_time_offset to keep the spec valid.
    start_time_offset = getattr(sliced, "start_time_offset", None)
    if sliced_timestamps and start_time_offset is not None:
        timestamps = getattr(sliced, "timestamps")
        setattr(sliced, "start_time_offset", start_time_offset + timestamps[0])
        setattr(sliced, "timestamps", timestamps - timestamps[0])

    # tracks sit outside SCHEMA (see FileSpec._SCHEMA_EXCLUDED_FIELDS), so recurse explicitly.
    if isinstance(sliced, FileSpec):
        sliced.tracks = [_slice_spec_dims(track, dim_indices) for track in sliced.tracks]

    return sliced


def slice_spec_dims(spec: SpecT, **dim_indices) -> SpecT:
    """Index a spec along named SCHEMA dimensions, wherever they occur.

    Every field whose ``SCHEMA`` shape mentions one of the given dimension names is
    indexed along that dimension, throughout the whole spec tree. So slicing
    ``n_frames`` also selects the matching entries of ``timestamps``, ``annotations``
    and ``metrics``, and slicing ``n_tx`` also selects the matching ``t0_delays``,
    ``tx_apodizations`` and ``time_to_next_transmit`` — without this function needing
    to know those fields exist.

    Fields that do not carry the dimension are passed through untouched, and arrays
    are shared with ``spec`` rather than copied where possible.

    Note:
        ``custom`` elements are opaque to the schema and are never sliced.

    Args:
        spec (Spec): The spec to slice. Not modified.
        **dim_indices: Index per named dimension, e.g. ``n_frames=[0, 2]``,
            ``n_tx=slice(0, 4)``. Values may be ints, lists, arrays, boolean masks
            or slices, and unlike a single fancy-index expression, several
            dimensions may be given as lists at the same time.

    Returns:
        Spec: A validated copy of ``spec`` with the requested dimensions indexed.

    Example:
        .. code-block:: python

            sliced = slice_spec_dims(file_spec, n_frames=[0, 2], n_tx=slice(0, 4))
    """
    unknown = set(dim_indices) - CONSISTENCY_DIMENSIONS
    if unknown:
        raise ValueError(
            f"Unknown dimension(s) {sorted(unknown)}. Only dimensions that the schema keeps "
            f"consistent across fields can be sliced: {sorted(CONSISTENCY_DIMENSIONS)}."
        )

    dim_indices = {dim: index for dim, index in dim_indices.items() if not _is_select_all(index)}
    if not dim_indices:
        return spec

    sliced = _slice_spec_dims(spec, dim_indices)

    # Re-run validation once, at the top: catches any dimension the slicing left
    # inconsistent across the tree.
    sliced.__post_init__()
    return sliced


@_supports_folders
def extract_frames_transmits(
    input_path: str | Path,
    output_path: str | Path,
    frame_indices=slice(None),
    transmit_indices=slice(None),
    overwrite=False,
    **kwargs,
):
    """
    extracts frames and transmits in a raw data file.

    Every field carrying an ``n_frames`` or ``n_tx`` dimension is sliced, as
    determined by the spec schema (see :func:`slice_spec_dims`), so scan parameters,
    annotations and metrics stay in sync with the extracted data.

    Args:
        input_path (str, Path): Path to the input raw data file, or a folder of files.
            Also accepts an ``hf://`` path (file or folder).
        output_path (Path): Path to the output file (or folder) where the extracted
            data will be saved.
        frame_indices (list, array-like, or slice): Indices of the frames to keep.
        transmit_indices (list, array-like, or slice): Indices of the transmits to keep.
        overwrite (bool, optional): Whether to overwrite the output file if it exists.
            Defaults to False.
        **kwargs: Passed to :meth:`zea.data.spec.FileSpec.save` (e.g. ``compression``,
            ``chunk_axes``).
    """

    # TODO: can be more efficient by only loading the requested frames and transmits
    # instead of loading all data and then slicing.
    with File(input_path) as f:
        file_spec = f._to_file_spec()

    if len(file_spec.tracks) > 1 and not (
        _is_select_all(frame_indices) and _is_select_all(transmit_indices)
    ):
        raise NotImplementedError(
            f"{input_path} has {len(file_spec.tracks)} tracks. Extracting frames or transmits "
            "from a multi-track file would require rebuilding 'track_schedule', which is "
            "ambiguous, so it is not supported."
        )

    file_spec = slice_spec_dims(file_spec, n_frames=frame_indices, n_tx=transmit_indices)

    # track_schedule is indexed by global transmit event ('n_total_tx'), not by 'n_tx',
    # so the schema-driven slicing above leaves it alone. A single-track schedule is all
    # zeros with one entry per transmit event, so it can simply be rebuilt. Multi-track
    # files only reach here on a no-op (select-all) extraction — see the guard above — so
    # their schedule encodes the genuine cross-track ordering and must be preserved as-is.
    if len(file_spec.tracks) == 1:
        data = file_spec.tracks[0].data
        raw_data = data.raw_data if data is not None else None
        if file_spec.track_schedule is not None and raw_data is not None:
            n_events = int(np.prod(raw_data.shape[:2]))
            file_spec.track_schedule = np.zeros(n_events, dtype=np.int32)

    _prepare_output_path(str(output_path), overwrite)
    file_spec.save(str(output_path), **kwargs)


def summary(input_path: str | Path):
    """Prints a summary of a zea data file to the console.

    Args:
        input_path (str, Path): Path to the zea data file. Also accepts an ``hf://`` path.
    """
    with File(input_path) as f:
        f.summary()


def copy(src: str | Path, dst: str | Path, key: str, mode: str | None = None):
    """Copies zea files to a new location using :meth:`zea.Dataset.copy`.

    Args:
        src (str, Path): Source path. Can be a single file, a list of files, a folder,
            or an ``hf://`` path.
        dst (Path): Destination folder path.
        key (str): Key to access in the HDF5 files. Use ``"all"`` or ``"*"`` to copy
            everything.
        mode (str, optional): HDF5 file mode for the destination files. Defaults to
            None, which lets :meth:`zea.Dataset.copy` auto-select the mode (``"a"`` for a
            single key, ``"w"`` when ``key`` is ``"all"``/``"*"``).
    """
    dataset = Dataset(src, validate=False)
    dataset.copy(dst, key, mode=mode)


def _delete_file_if_exists(path: Path):
    """Deletes a file if it exists."""
    if path.exists():
        path.unlink()


def _prepare_output_path(output_path: str, overwrite: bool):
    """Guard the save target, matching :func:`resave`: refuse to clobber unless asked.

    ``FileSpec.save`` overwrites atomically and has no ``overwrite`` flag of its own, so
    callers must enforce it here or an existing file is silently replaced.

    Also refuses to save to an ``hf://`` path, which is read-only.
    """
    if output_path.startswith(HF_PREFIX):
        raise ValueError(
            f"Cannot save to an 'hf://' path: {output_path}. 'hf://' paths are read-only; "
            "save to a local path instead."
        )
    if Path(output_path).exists() and not overwrite:
        raise FileExistsError(
            f"Output path {output_path} already exists. Use overwrite=True to overwrite."
        )
    if overwrite:
        _delete_file_if_exists(Path(output_path))


def _interpret_index(input_str):
    if "-" in input_str:
        start, end = map(int, input_str.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(x) for x in input_str.split(" ")]


def _interpret_indices(input_str_list):
    if input_str_list == "all" or input_str_list == ["all"]:
        return slice(None)

    if len(input_str_list) == 1 and "-" in input_str_list[0]:
        start, end = map(int, input_str_list[0].split("-"))
        return slice(start, end + 1)

    indices = []
    for part in input_str_list:
        indices.extend(_interpret_index(part))
    return indices


# ── Command line interface (tyro) ────────────────────────────────────────────
#
# The CLI subcommand dataclasses live in :mod:`zea.cli_args` (a light-import
# module) so that ``zea data …`` can be wired into the top-level ``zea`` CLI
# without importing this heavy module just to render ``--help``. The dataclasses
# there dispatch back to the operation functions above.


def main():
    """Parse command line arguments and run the requested data operation.

    Entry point for ``python -m zea.data``. This is equivalent to ``zea data``.
    """
    from zea.cli_args import DataCommand, _run_data_command

    args = tyro.cli(DataCommand)  # ty: ignore[no-matching-overload]
    _run_data_command(args)


if __name__ == "__main__":
    main()
