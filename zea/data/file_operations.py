"""
This module provides some utilities to edit zea data files, either individually or in bulk.

Each operation is available both as a Python function and as a command line subcommand.
See the :ref:`CLI documentation <cli-file-operations>` for the available operations and
their command-line usage.
"""

import argparse
import functools
from pathlib import Path

import numpy as np
from tqdm import tqdm

from zea.data.data_format import generate_zea_dataset, load_additional_elements, load_description
from zea.data.datasets import Dataset
from zea.data.file import load_file_all_data_types
from zea.internal.checks import _IMAGE_DATA_TYPES, _NON_IMAGE_DATA_TYPES
from zea.internal.core import DataTypes
from zea.internal.parameters import MissingDependencyError
from zea.log import logger
from zea.probes import Probe
from zea.scan import Scan

ALL_DATA_TYPES_EXCEPT_RAW = set(_IMAGE_DATA_TYPES + _NON_IMAGE_DATA_TYPES) - {"raw_data"}

OPERATION_NAMES = [
    "sum",
    "compound_frames",
    "compound_transmits",
    "resave",
    "extract",
]


def _safe_getattr(obj, name):
    """Get ``obj.name``, returning ``None`` if it is missing or has unmet dependencies."""
    try:
        return getattr(obj, name, None)
    except MissingDependencyError:
        return None


def save_file(
    path,
    scan: Scan,
    probe: Probe,
    raw_data: np.ndarray = None,
    aligned_data: np.ndarray = None,
    beamformed_data: np.ndarray = None,
    envelope_data: np.ndarray = None,
    image: np.ndarray = None,
    image_sc: np.ndarray = None,
    additional_elements=None,
    description="",
    enable_compression=True,
    chunk_frames=False,
    **kwargs,
):
    """Saves data to a zea data file (h5py file).

    Args:
        path (str, pathlike): The path to the hdf5 file.
        raw_data (np.ndarray): The data to save.
        scan (Scan): The scan object containing the parameters of the acquisition.
        probe (Probe): The probe object containing the parameters of the probe.
        additional_elements (list of DatasetElement, optional): Additional elements to save in the
            file. Defaults to None.
        enable_compression (bool, optional): Whether to enable gzip compression for the
            datasets. Defaults to True.
        chunk_frames (bool, optional): Whether to store the data datasets with HDF5
            chunked storage, using one frame per chunk. Defaults to False.
    """

    generate_zea_dataset(
        path=path,
        enable_compression=enable_compression,
        chunk_frames=chunk_frames,
        raw_data=raw_data,
        aligned_data=aligned_data,
        beamformed_data=beamformed_data,
        image=image,
        image_sc=image_sc,
        envelope_data=envelope_data,
        probe_name="generic",
        probe_geometry=_safe_getattr(probe, "probe_geometry"),
        sampling_frequency=_safe_getattr(scan, "sampling_frequency"),
        center_frequency=_safe_getattr(scan, "center_frequency"),
        initial_times=_safe_getattr(scan, "initial_times"),
        t0_delays=_safe_getattr(scan, "t0_delays"),
        sound_speed=_safe_getattr(scan, "sound_speed"),
        focus_distances=_safe_getattr(scan, "focus_distances"),
        transmit_origins=_safe_getattr(scan, "transmit_origins"),
        polar_angles=_safe_getattr(scan, "polar_angles"),
        azimuth_angles=_safe_getattr(scan, "azimuth_angles"),
        tx_apodizations=_safe_getattr(scan, "tx_apodizations"),
        bandwidth_percent=_safe_getattr(scan, "bandwidth_percent"),
        time_to_next_transmit=_safe_getattr(scan, "time_to_next_transmit"),
        tgc_gain_curve=_safe_getattr(scan, "tgc_gain_curve"),
        element_width=_safe_getattr(scan, "element_width"),
        tx_waveform_indices=_safe_getattr(scan, "tx_waveform_indices"),
        waveforms_one_way=_safe_getattr(scan, "waveforms_one_way"),
        waveforms_two_way=_safe_getattr(scan, "waveforms_two_way"),
        description=description,
        additional_elements=additional_elements,
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

    data_dict, scan, probe = load_file_all_data_types(input_paths[0])
    description = load_description(input_paths[0])
    additional_elements = load_additional_elements(input_paths[0])

    for file in input_paths[1:]:
        new_data, new_scan, new_probe = load_file_all_data_types(file)

        if data_dict["raw_data"] is not None:
            _assert_shapes_equal(data_dict["raw_data"], new_data["raw_data"], "raw_data")
            data_dict["raw_data"] += new_data["raw_data"]

        if data_dict["aligned_data"] is not None:
            _assert_shapes_equal(
                data_dict["aligned_data"], new_data["aligned_data"], "aligned_data"
            )
            data_dict["aligned_data"] += new_data["aligned_data"]

        if data_dict["beamformed_data"] is not None:
            _assert_shapes_equal(
                data_dict["beamformed_data"], new_data["beamformed_data"], "beamformed_data"
            )
            data_dict["beamformed_data"] += new_data["beamformed_data"]

        if data_dict["envelope_data"] is not None:
            _assert_shapes_equal(
                data_dict["envelope_data"], new_data["envelope_data"], "envelope_data"
            )
            data_dict["envelope_data"] += new_data["envelope_data"]

        if data_dict["image"] is not None:
            _assert_shapes_equal(data_dict["image"], new_data["image"], "image")
            data_dict["image"] = np.log(np.exp(new_data["image"]) + np.exp(data_dict["image"]))

        if data_dict["image_sc"] is not None:
            _assert_shapes_equal(data_dict["image_sc"], new_data["image_sc"], "image_sc")
            data_dict["image_sc"] = np.log(
                np.exp(new_data["image_sc"]) + np.exp(data_dict["image_sc"])
            )
        assert scan == new_scan, "Scan parameters do not match."
        assert probe == new_probe, "Probe parameters do not match."

    if overwrite:
        _delete_file_if_exists(output_path)

    save_file(
        path=output_path,
        scan=scan,
        probe=probe,
        additional_elements=additional_elements,
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

    data_dict, scan, probe = load_file_all_data_types(input_path)
    additional_elements = load_additional_elements(input_path)
    description = load_description(input_path)

    # Assuming the first dimension is the frame dimension

    compounded_data = {}
    for data_type in DataTypes:
        key = data_type.value
        if data_dict[key] is None:
            compounded_data[key] = None
            continue
        if key == "image" or key == "image_sc":
            compounded_data[key] = np.log(np.mean(np.exp(data_dict[key]), axis=0, keepdims=True))
        else:
            compounded_data[key] = np.mean(data_dict[key], axis=0, keepdims=True)

    scan = _scan_reduce_frames(scan, [0])

    if overwrite:
        _delete_file_if_exists(output_path)

    save_file(
        path=output_path,
        scan=scan,
        probe=probe,
        additional_elements=additional_elements,
        description=description,
        **compounded_data,
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

    data_dict, scan, probe = load_file_all_data_types(input_path)
    additional_elements = load_additional_elements(input_path)
    description = load_description(input_path)

    if not _all_tx_are_identical(scan):
        logger.warning(
            "Not all transmits are identical. Compounding transmits may lead to unexpected results."
        )

    # Assuming the second dimension is the transmit dimension
    for key in ["raw_data", "aligned_data"]:
        if data_dict[key] is None:
            continue
        data_dict[key] = np.mean(data_dict[key], axis=1, keepdims=True)

    scan.set_transmits([0])

    if overwrite:
        _delete_file_if_exists(output_path)

    save_file(
        path=output_path,
        scan=scan,
        probe=probe,
        additional_elements=additional_elements,
        description=description,
        **data_dict,
    )


def _all_tx_are_identical(scan: Scan):
    """Checks if all transmits in a Scan object are identical."""
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
        enable_compression (bool, optional): Whether to enable gzip compression for the
            datasets. Defaults to True.
        chunk_frames (bool, optional): Whether to store the data datasets with HDF5
            chunked storage, using one frame per chunk. Defaults to False.
    """

    data_dict, scan, probe = load_file_all_data_types(input_path)
    additional_elements = load_additional_elements(input_path)
    description = load_description(input_path)
    scan.set_transmits("all")

    if overwrite:
        _delete_file_if_exists(output_path)
    save_file(
        path=output_path,
        **data_dict,
        scan=scan,
        probe=probe,
        additional_elements=additional_elements,
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
    data_dict, scan, probe = load_file_all_data_types(input_path, indices=indices)

    additional_elements = load_additional_elements(input_path)
    description = load_description(input_path)

    scan = _scan_reduce_frames(scan, frame_indices)

    if overwrite:
        _delete_file_if_exists(output_path)

    save_file(
        path=output_path,
        **data_dict,
        scan=scan,
        probe=probe,
        additional_elements=additional_elements,
        description=description,
    )


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
    if isinstance(input_str_list, str) and input_str_list == "all":
        return slice(None)

    if len(input_str_list) == 1 and "-" in input_str_list[0]:
        start, end = map(int, input_str_list[0].split("-"))
        return slice(start, end + 1)

    indices = []
    for part in input_str_list:
        indices.extend(_interpret_index(part))
    return indices


def _scan_reduce_frames(scan, frame_indices):
    transmit_indices = scan.selected_transmits
    scan.set_transmits("all")
    if scan.time_to_next_transmit is not None:
        scan.time_to_next_transmit = scan.time_to_next_transmit[frame_indices]
    scan.set_transmits(transmit_indices)
    return scan


def get_parser():
    """Command line argument parser with subcommands"""

    parser = argparse.ArgumentParser(
        description=(
            "Manipulate zea data files.\n\n"
            "All operations accept files; folder inputs are also supported. For "
            "file-to-file operations, each zea file in the input folder is processed "
            "and written to a mirrored path in the output folder."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    _add_parser_sum(subparsers)
    _add_parser_compound_frames(subparsers)
    _add_parser_compound_transmits(subparsers)
    _add_parser_resave(subparsers)
    _add_parser_extract(subparsers)

    return parser


def _add_parser_sum(subparsers):
    sum_parser = subparsers.add_parser("sum", help="Sum the raw data of multiple files or folders.")
    sum_parser.add_argument(
        "input_paths", type=Path, nargs="+", help="Paths to the input files or folders."
    )
    sum_parser.add_argument("output_path", type=Path, help="Output HDF5 file.")
    sum_parser.add_argument(
        "--overwrite", action="store_true", default=False, help="Overwrite existing output file."
    )


def _add_parser_compound_frames(subparsers):
    cf_parser = subparsers.add_parser("compound_frames", help="Compound frames to increase SNR.")
    cf_parser.add_argument("input_path", type=Path, help="Input HDF5 file or folder.")
    cf_parser.add_argument("output_path", type=Path, help="Output HDF5 file or folder.")
    cf_parser.add_argument(
        "--overwrite", action="store_true", default=False, help="Overwrite existing output file."
    )


def _add_parser_compound_transmits(subparsers):
    ct_parser = subparsers.add_parser(
        "compound_transmits", help="Compound transmits to increase SNR."
    )
    ct_parser.add_argument("input_path", type=Path, help="Input HDF5 file or folder.")
    ct_parser.add_argument("output_path", type=Path, help="Output HDF5 file or folder.")
    ct_parser.add_argument(
        "--overwrite", action="store_true", default=False, help="Overwrite existing output file."
    )


def _add_parser_resave(subparsers):
    resave_parser = subparsers.add_parser("resave", help="Resave a file to change format version.")
    resave_parser.add_argument("input_path", type=Path, help="Input HDF5 file or folder.")
    resave_parser.add_argument("output_path", type=Path, help="Output HDF5 file or folder.")
    resave_parser.add_argument(
        "--overwrite", action="store_true", default=False, help="Overwrite existing output file."
    )
    resave_parser.add_argument(
        "--chunk-frames",
        action="store_true",
        default=False,
        help="Store the data datasets with HDF5 chunked storage, one frame per chunk.",
    )
    resave_parser.add_argument(
        "--disable-compression",
        action="store_true",
        default=False,
        help="Disable gzip compression for the datasets.",
    )


def _add_parser_extract(subparsers):
    extract_parser = subparsers.add_parser("extract", help="Extract subset of frames or transmits.")
    extract_parser.add_argument("input_path", type=Path, help="Input HDF5 file or folder.")
    extract_parser.add_argument("output_path", type=Path, help="Output HDF5 file or folder.")
    extract_parser.add_argument(
        "--transmits",
        type=str,
        nargs="+",
        default="all",
        help="Target transmits. Can be a list of integers or ranges (e.g. 0-3 7).",
    )
    extract_parser.add_argument(
        "--frames",
        type=str,
        nargs="+",
        default="all",
        help="Target frames. Can be a list of integers or ranges (e.g. 0-3 7).",
    )
    extract_parser.add_argument(
        "--overwrite", action="store_true", default=False, help="Overwrite existing output file."
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # For folder operations the output is a directory; individual output files are
    # still guarded per file. Only block when the output is an existing file.
    if args.output_path.is_file() and not args.overwrite:
        logger.error(
            f"Output file {args.output_path} already exists. Use --overwrite to overwrite it."
        )
        exit(1)

    if args.operation == "compound_frames":
        compound_frames(
            input_path=args.input_path, output_path=args.output_path, overwrite=args.overwrite
        )
    elif args.operation == "compound_transmits":
        compound_transmits(
            input_path=args.input_path, output_path=args.output_path, overwrite=args.overwrite
        )
    elif args.operation == "resave":
        resave(
            input_path=args.input_path,
            output_path=args.output_path,
            overwrite=args.overwrite,
            enable_compression=not args.disable_compression,
            chunk_frames=args.chunk_frames,
        )
    elif args.operation == "extract":
        extract_frames_transmits(
            input_path=args.input_path,
            output_path=args.output_path,
            frame_indices=_interpret_indices(args.frames),
            transmit_indices=_interpret_indices(args.transmits),
            overwrite=args.overwrite,
        )
    elif args.operation == "sum":
        sum_data(
            input_paths=args.input_paths, output_path=args.output_path, overwrite=args.overwrite
        )
    else:
        raise ValueError(f"Unknown operation: {args.operation}")
