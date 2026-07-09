"""
This module provides some utilities to edit zea data files, either individually or in bulk.

Each operation is available both as a Python function and as a ``zea data`` command line
subcommand. See the :doc:`CLI documentation </cli>` for the available operations and their
command-line usage.
"""

import functools
from pathlib import Path

import numpy as np
import tyro
from tqdm import tqdm

from zea import Parameters
from zea.data.datasets import Dataset
from zea.data.file import File, load_file_all_data_types
from zea.data.spec import DEFAULT_COMPRESSION
from zea.internal.checks import _IMAGE_DATA_TYPES, _NON_IMAGE_DATA_TYPES
from zea.internal.core import DataTypes
from zea.log import logger

ALL_DATA_TYPES_EXCEPT_RAW = set(_IMAGE_DATA_TYPES + _NON_IMAGE_DATA_TYPES) - {"raw_data"}

OPERATION_NAMES = [
    "sum",
    "compound_frames",
    "compound_transmits",
    "resave",
    "extract",
    "summary",
    "copy",
]


def save_file(
    path,
    parameters: Parameters,
    raw_data: np.ndarray | None = None,
    aligned_data: dict | None = None,
    beamformed_data: dict | None = None,
    envelope_data: dict | None = None,
    image: dict | None = None,
    description=None,
    custom_maps: dict | None = None,
    custom_elements: list | None = None,
    metadata: dict | None = None,
    compression: str = DEFAULT_COMPRESSION,
    chunk_frames=False,
    **kwargs,
):
    """Saves data to a zea data file (h5py file).

    Args:
        path (str, pathlike): The path to the hdf5 file.
        parameters (Parameters): The parameters object containing acquisition and probe
            parameters.
        raw_data (np.ndarray): The data to save.
        aligned_data (np.ndarray, optional): Aligned data as a dict with ``"values"``
            and ``"extent"`` keys (validated as :class:`~zea.data.spec.AlignedData`).
        beamformed_data (dict, optional): Beamformed data as a dict with ``"values"`` and
            ``"extent"`` keys (validated as :class:`~zea.data.spec.BeamformedData`).
        envelope_data (dict, optional): Envelope-detected data as a dict with ``"values"``
            and ``"extent"`` keys (validated as :class:`~zea.data.spec.EnvelopeData`).
        image (dict, optional): Reconstructed (log-compressed) image data as a dict with
            ``"values"`` and ``"extent"`` keys (validated as :class:`~zea.data.spec.Image`).
        description (str, optional): A description for the dataset.
        custom_maps (dict, optional): Custom spatial map entries to include in the ``data`` group.
            Each key maps to a dict with ``"values"`` (np.ndarray, uint8) and ``"coordinates"``
            (np.ndarray, float32, shape ``(n_frames, ..., 3)``) fields, plus optional
            ``"description"`` and ``"unit"`` fields.  Example::

                custom_maps = {
                    "my_overlay": {
                        "values": values_array,      # (n_frames, z, x[, n_ch]), uint8
                        "coordinates": coords_array, # (n_frames, z, x, 3), float32
                    }
                }
        custom_elements (list, optional): List of :class:`~zea.data.file.CustomElement`
            objects holding data that does not fit the zea format. Stored in a ``custom``
            group and read back via :attr:`zea.File.custom`.
        metadata (dict, optional): Metadata to store in the ``metadata`` group, validated against
            :class:`~zea.data.spec.MetadataSpec`.  Standard keys include ``"subject"``,
            ``"credit"``, ``"annotations"``, ``"text_report"``, ``"ecg"``,
            ``"probe_pose"``, and ``"voice_narration"``.  Custom signal keys are also
            accepted and stored as :class:`~zea.data.spec.SignalND` entries.  Example::

                metadata = {
                    "credit": "My Lab, 2024",
                    "annotations": {"label": np.array(["healthy", "healthy"])},
                }
        compression (str, optional): The HDF5 compression filter to use. Defaults to ``"lzf"``.
        chunk_frames (bool, optional): Whether to store the data datasets with HDF5
            chunked storage, using one frame per chunk. Defaults to False.
    """

    data = {}
    for key, arr in [
        ("raw_data", raw_data),
        ("aligned_data", aligned_data),
        ("beamformed_data", beamformed_data),
        ("envelope_data", envelope_data),
        ("image", image),
    ]:
        if arr is not None:
            data[key] = arr

    if custom_maps:
        for key, map_dict in custom_maps.items():
            data[key] = map_dict

    File.create(
        path=path,
        data=data,
        scan=parameters.to_scan_dict(),
        metadata=metadata,
        probe=parameters.to_probe_dict(),
        description=description,
        custom=custom_elements,
        compression=compression,
        chunk_frames=chunk_frames,
        overwrite=True,
    )


def _iter_folder_io(input_path: Path, output_path: Path):
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
    """

    @functools.wraps(operation)
    def wrapper(input_path, output_path, *args, **kwargs):
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


def sum_data(input_paths: list[Path], output_path: Path, overwrite=False):
    """
    Sums multiple raw data files and saves the result to a new file.

    For images, this will actually average the images. If the images are uint8, it will average
    directly. If the images are float32, we assume they are in the log-domain and we will do the
    averaging in the linear domain.

    Args:
        input_paths (list[Path]): List of paths to the input raw data files. Each path
            may be a single file or a folder; folders are expanded into all zea files
            they contain (using :class:`zea.Dataset`).
        output_path (Path): Path to the output file where the summed data will be saved.
        overwrite (bool, optional): Whether to overwrite the output file if it exists.
            Defaults to False.
    """

    with Dataset(input_paths, validate=False) as dataset:
        input_paths = [file.path for file in dataset]

    data_dict, parameters = load_file_all_data_types(input_paths[0])
    with File(input_paths[0]) as f:
        description = f.description
        custom_elements = f.custom

    image_is_uint8 = (
        data_dict["image"] is not None
        and isinstance(data_dict["image"], dict)
        and data_dict["image"]["values"].dtype == np.uint8
    )
    image_is_float32 = (
        data_dict["image"] is not None
        and isinstance(data_dict["image"], dict)
        and data_dict["image"]["values"].dtype == np.float32
    )

    # Cast to float32 to avoid overflow
    if image_is_uint8:
        data_dict["image"]["values"] = data_dict["image"]["values"].astype(np.float32)

    for file in input_paths[1:]:
        new_data, new_parameters = load_file_all_data_types(file)

        if data_dict["raw_data"] is not None:
            _assert_shapes_equal(data_dict["raw_data"], new_data["raw_data"], "raw_data")
            data_dict["raw_data"] += new_data["raw_data"]

        if data_dict["aligned_data"] is not None:
            _assert_shapes_equal(
                data_dict["aligned_data"]["values"],
                new_data["aligned_data"]["values"],
                "aligned_data",
            )
            data_dict["aligned_data"]["values"] += new_data["aligned_data"]["values"]

        if data_dict["beamformed_data"] is not None:
            _assert_shapes_equal(
                data_dict["beamformed_data"]["values"],
                new_data["beamformed_data"]["values"],
                "beamformed_data",
            )
            data_dict["beamformed_data"]["values"] += new_data["beamformed_data"]["values"]

        if data_dict["envelope_data"] is not None:
            _assert_shapes_equal(
                data_dict["envelope_data"]["values"],
                new_data["envelope_data"]["values"],
                "envelope_data",
            )
            data_dict["envelope_data"]["values"] += new_data["envelope_data"]["values"]

        if data_dict["image"] is not None:
            _assert_shapes_equal(data_dict["image"]["values"], new_data["image"]["values"], "image")
            if image_is_float32:
                data_dict["image"]["values"] = np.log(
                    np.exp(new_data["image"]["values"]) + np.exp(data_dict["image"]["values"])
                )
            elif image_is_uint8:
                data_dict["image"]["values"] = (
                    new_data["image"]["values"] + data_dict["image"]["values"]
                )
            else:
                raise ValueError("image values must be uint8 or float32")

        assert parameters == new_parameters, "Scan parameters do not match."

    # Divide to get the mean; for uint8, keep float precision then clip and cast back
    if image_is_uint8:
        data_dict["image"]["values"] = np.clip(
            data_dict["image"]["values"] / len(input_paths), 0, 255
        ).astype(np.uint8)
    if image_is_float32:
        data_dict["image"]["values"] = np.minimum(
            data_dict["image"]["values"] - np.log(len(input_paths)), 0.0
        )

    if overwrite:
        _delete_file_if_exists(output_path)

    save_file(
        path=output_path,
        parameters=parameters,
        custom_elements=custom_elements,
        description=description,
        **data_dict,
    )


def _assert_shapes_equal(array0, array1, name="array"):
    shape0, shape1 = array0.shape, array1.shape
    assert shape0 == shape1, f"{name} shapes do not match. Got {shape0} and {shape1}."


@_supports_folders
def compound_frames(input_path: Path, output_path: Path, overwrite=False):
    """
    Compounds frames in a raw data file by averaging them.

    Args:
        input_path (Path): Path to the input raw data file, or a folder of files.
        output_path (Path): Path to the output file (or folder) where the compounded
            data will be saved.
        overwrite (bool, optional): Whether to overwrite the output file if it exists.
            Defaults to False.
    """

    data_dict, parameters = load_file_all_data_types(input_path)
    with File(input_path) as f:
        description = f.description
        custom_elements = f.custom

    # Assuming the first dimension is the frame dimension

    # Map-based data types store values in a dict; these need special handling.
    # "image_sc" is a deprecated, legacy-only data type: it is still handled here so
    # that legacy files containing it can be processed without crashing, but it is
    # dropped on save (save_file / File.create no longer write it).
    _MAP_KEYS = {"aligned_data", "beamformed_data", "envelope_data", "image_sc", "image"}
    _LOG_COMPOUND_KEYS = {"image", "image_sc"}

    compounded_data = {}
    for data_type in DataTypes:
        key = data_type.value
        if data_dict[key] is None:
            compounded_data[key] = None
            continue
        if key in _MAP_KEYS:
            values = data_dict[key]["values"]
            if key in _LOG_COMPOUND_KEYS and values.dtype == np.float32:
                values = np.log(np.mean(np.exp(values), axis=0, keepdims=True))
            elif values.dtype == np.uint8:
                values = np.clip(
                    np.mean(values.astype(np.float32), axis=0, keepdims=True), 0, 255
                ).astype(np.uint8)
            else:
                values = np.mean(values, axis=0, keepdims=True)
            compounded_data[key] = {**data_dict[key], "values": values}
        else:
            compounded_data[key] = np.mean(data_dict[key], axis=0, keepdims=True)

    parameters = _scan_reduce_frames(parameters, [0])

    if overwrite:
        _delete_file_if_exists(output_path)

    save_file(
        path=output_path,
        parameters=parameters,
        custom_elements=custom_elements,
        description=description,
        **compounded_data,  # ty: ignore[invalid-argument-type]
    )


@_supports_folders
def compound_transmits(input_path: Path, output_path: Path, overwrite=False):
    """
    Compounds transmits in a raw data file by averaging them.

    Note:
        This function assumes that all transmits are identical. If this is not the case the
        function will result in incorrect scan parameters.

    Args:
        input_path (Path): Path to the input raw data file, or a folder of files.
        output_path (Path): Path to the output file (or folder) where the compounded
            data will be saved.
        overwrite (bool, optional): Whether to overwrite the output file if it exists.
            Defaults to False.
    """

    data_dict, parameters = load_file_all_data_types(input_path)
    with File(input_path) as f:
        description = f.description
        custom_elements = f.custom

    if not _all_tx_are_identical(parameters):
        logger.warning(
            "Not all transmits are identical. Compounding transmits may lead to unexpected results."
        )

    # Assuming the second dimension is the transmit dimension
    if data_dict["raw_data"] is not None:
        data_dict["raw_data"] = np.mean(data_dict["raw_data"], axis=1, keepdims=True)
    if data_dict["aligned_data"] is not None:
        data_dict["aligned_data"]["values"] = np.mean(
            data_dict["aligned_data"]["values"], axis=1, keepdims=True
        )

    parameters.set_transmits([0])

    if overwrite:
        _delete_file_if_exists(output_path)

    save_file(
        path=output_path,
        parameters=parameters,
        custom_elements=custom_elements,
        description=description,
        **data_dict,
    )


def _all_tx_are_identical(parameters: Parameters):
    """Checks if all transmits in a Parameters object are identical."""
    attributes_to_check = [
        parameters.polar_angles,
        parameters.azimuth_angles,
        parameters.t0_delays,
        parameters.tx_apodizations,
        parameters.focus_distances,
        parameters.transmit_origins,
        parameters.initial_times,
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
    input_path: Path,
    output_path: Path,
    overwrite=False,
    enable_compression=True,
    chunk_frames=False,
):
    """
    Resaves a zea data file to a new location.

    Args:
        input_path (Path): Path to the input zea data file, or a folder of files.
        output_path (Path): Path to the output file (or folder) where the data will be saved.
        overwrite (bool, optional): Whether to overwrite the output file if it exists.
            Defaults to False.
        enable_compression (bool, optional): Whether to enable lzf compression for the
            datasets. Defaults to True.
        chunk_frames (bool, optional): Whether to store the data datasets with HDF5
            chunked storage, using one frame per chunk. Defaults to False.
    """

    data_dict, parameters = load_file_all_data_types(input_path)
    with File(input_path) as f:
        description = f.description
        custom_elements = f.custom
    parameters.set_transmits("all")

    if overwrite:
        _delete_file_if_exists(output_path)
    save_file(
        path=output_path,
        **data_dict,
        parameters=parameters,
        custom_elements=custom_elements,
        description=description,
        enable_compression=enable_compression,
        chunk_frames=chunk_frames,
    )


@_supports_folders
def extract_frames_transmits(
    input_path: Path,
    output_path: Path,
    frame_indices=slice(None),
    transmit_indices=slice(None),
    overwrite=False,
):
    """
    extracts frames and transmits in a raw data file.

    Note that the frame indices cannot both be lists. At least one of them must be a slice.
    Please refer to the documentation of :func:`zea.data.file.load_file_all_data_types` for more
    information on the supported index types.

    Args:
        input_path (Path): Path to the input raw data file, or a folder of files.
        output_path (Path): Path to the output file (or folder) where the extracted
            data will be saved.
        frame_indices (list, array-like, or slice): Indices of the frames to keep.
        transmit_indices (list, array-like, or slice): Indices of the transmits to keep.
        overwrite (bool, optional): Whether to overwrite the output file if it exists.
            Defaults to False.
    """
    indices = (frame_indices, transmit_indices)
    data_dict, parameters = load_file_all_data_types(input_path, indices=indices)

    with File(input_path) as f:
        description = f.description
        custom_elements = f.custom

    parameters = _scan_reduce_frames(parameters, frame_indices)

    if overwrite:
        _delete_file_if_exists(output_path)

    save_file(
        path=output_path,
        **data_dict,
        parameters=parameters,
        custom_elements=custom_elements,
        description=description,
    )


def summary(input_path: Path):
    """Prints a summary of a zea data file to the console.

    Args:
        input_path (Path): Path to the zea data file.
    """
    with File(input_path) as f:
        f.summary()


def copy(src: Path, dst: Path, key: str, mode: str | None = None):
    """Copies zea files to a new location using :meth:`zea.Dataset.copy`.

    Args:
        src (Path): Source path. Can be a single file, a list of files, or a folder.
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


def _scan_reduce_frames(parameters, frame_indices):
    transmit_indices = parameters.selected_transmits
    parameters.set_transmits("all")
    if parameters.time_to_next_transmit is not None:
        parameters.time_to_next_transmit = parameters.time_to_next_transmit[frame_indices]
    parameters.set_transmits(transmit_indices)
    return parameters


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
