"""
This module provides some utilities to edit zea data files.

Available operations
--------------------

- `sum`: Sum multiple raw data files into one.

- `compound_frames`: Compound frames in a raw data file to increase SNR.

- `compound_transmits`: Compound transmits in a raw data file to increase SNR.

- `resave`: Resave a zea data file. This can be used to change the file format version.

- `extract`: extract frames and transmits in a raw data file.
"""

import argparse
from pathlib import Path

import numpy as np

from zea import Probe, Scan
from zea.data.data_format import load_additional_elements, load_description
from zea.data.file import File, load_file_all_data_types
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
]


def save_file(
    path,
    scan: Scan,
    probe: Probe,
    raw_data: np.ndarray = None,
    aligned_data: np.ndarray = None,
    beamformed_data: dict = None,
    envelope_data: dict = None,
    image: dict = None,
    image_sc: dict = None,
    description="",
    custom_maps: dict | None = None,
    metadata: dict | None = None,
    **kwargs,
):
    """Saves data to a zea data file (h5py file).

    Args:
        path (str, pathlike): The path to the hdf5 file.
        raw_data (np.ndarray): The data to save.
        scan (Scan): The scan object containing the parameters of the acquisition.
        probe (Probe): The probe object containing the parameters of the probe.
        description (str): A description for the dataset.
        beamformed_data (dict, optional): Beamformed data as a dict with ``"values"`` and
            ``"extent"`` keys (validated as :class:`~zea.data.spec.BeamformedData`).
        envelope_data (dict, optional): Envelope-detected data as a dict with ``"values"``
            and ``"extent"`` keys (validated as :class:`~zea.data.spec.EnvelopeData`).
        image (dict, optional): Reconstructed (log-compressed) image data as a dict with
            ``"values"`` and ``"extent"`` keys (validated as :class:`~zea.data.spec.Image`).
        custom_maps (dict, optional): Custom spatial map entries to include in the ``data`` group.
            Each key maps to a dict with ``"values"`` (np.ndarray, uint8) and ``"extent"``
            (np.ndarray, float32, shape ``(6,)``) fields, plus optional ``"labels"``,
            ``"description"``, and ``"unit"`` fields.  Example::

                custom_maps = {
                    "my_overlay": {
                        "values": values_array,  # (n_frames, x, z, y[, n_ch]), uint8
                        "extent": extent_array,  # (6,) float32
                    }
                }

        metadata (dict, optional): Metadata to store in the ``metadata`` group, validated against
            :class:`~zea.data.spec.MetadataSpec`.  Standard keys include ``"subject"``,
            ``"credit"``, ``"annotations"``, ``"text_report"``, ``"ecg"``,
            ``"probe_pose"``, and ``"voice_narration"``.  Custom signal keys are also
            accepted and stored as :class:`~zea.data.spec.SignalND` entries.  Example::

                metadata = {
                    "credit": "My Lab, 2024",
                    "annotations": {"label": np.array(["healthy", "healthy"])},
                }
    """

    data = {}
    for key, arr in [
        ("raw_data", raw_data),
        ("aligned_data", aligned_data),
        ("beamformed_data", beamformed_data),
        ("envelope_data", envelope_data),
        ("image", image),
        ("image_sc", image_sc),
    ]:
        if arr is not None:
            data[key] = arr

    if custom_maps:
        for key, map_dict in custom_maps.items():
            data[key] = map_dict

    scan_dict = {
        "probe_geometry": probe.probe_geometry,
        "sampling_frequency": np.float32(scan.sampling_frequency),
        "center_frequency": np.float32(scan.center_frequency),
        "demodulation_frequency": np.float32(scan.demodulation_frequency),
        "initial_times": scan.initial_times,
        "t0_delays": scan.t0_delays,
        "sound_speed": np.float32(scan.sound_speed) if scan.sound_speed is not None else None,
    }

    optional_scan = {
        "focus_distances": scan.focus_distances,
        "transmit_origins": scan.transmit_origins,
        "polar_angles": scan.polar_angles,
        "azimuth_angles": scan.azimuth_angles,
        "tx_apodizations": scan.tx_apodizations,
        "time_to_next_transmit": scan.time_to_next_transmit,
        "tgc_gain_curve": scan.tgc_gain_curve,
        "element_width": scan.element_width,
    }
    for key, val in optional_scan.items():
        if val is not None:
            scan_dict[key] = val

    # Filter out None values from scan_dict
    scan_dict = {k: v for k, v in scan_dict.items() if v is not None}

    f = File.create(
        path=path,
        data=data,
        scan=scan_dict if scan_dict else None,
        metadata=metadata or None,
        probe_name="generic",
        description=description or None,
        overwrite=True,
    )
    f.close()


def sum_data(input_paths: list[Path], output_path: Path, overwrite=False):
    """
    Sums multiple raw data files and saves the result to a new file.

    For images, this will actually average the images. If the images are uint8, it will average
    directly. If the images are float32, we assume they are in the log-domain and we will do the
    averaging in the linear domain.

    Args:
        input_paths (list[Path]): List of paths to the input raw data files.
        output_path (Path): Path to the output file where the summed data will be saved.
        overwrite (bool, optional): Whether to overwrite the output file if it exists. Defaults to
            False.
    """

    data_dict, scan, probe = load_file_all_data_types(input_paths[0])
    description = load_description(input_paths[0])
    additional_elements = load_additional_elements(input_paths[0])

    image_is_uint8 = (
        data_dict["image"] is not None
        and isinstance(data_dict["image"], dict)
        and data_dict["image"]["values"].dtype == np.uint8
    )
    image_sc_is_uint8 = (
        data_dict["image_sc"] is not None
        and isinstance(data_dict["image_sc"], dict)
        and data_dict["image_sc"]["values"].dtype == np.uint8
    )
    image_is_float32 = (
        data_dict["image"] is not None
        and isinstance(data_dict["image"], dict)
        and data_dict["image"]["values"].dtype == np.float32
    )
    image_sc_is_float32 = (
        data_dict["image_sc"] is not None
        and isinstance(data_dict["image_sc"], dict)
        and data_dict["image_sc"]["values"].dtype == np.float32
    )

    # Cast to float32 to avoid overflow
    if image_is_uint8:
        data_dict["image"]["values"] = data_dict["image"]["values"].astype(np.float32)
    if image_sc_is_uint8:
        data_dict["image_sc"]["values"] = data_dict["image_sc"]["values"].astype(np.float32)

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

        if data_dict["image_sc"] is not None:
            _assert_shapes_equal(
                data_dict["image_sc"]["values"],
                new_data["image_sc"]["values"],
                "image_sc",
            )
            if image_sc_is_float32:
                data_dict["image_sc"]["values"] = np.log(
                    np.exp(new_data["image_sc"]["values"]) + np.exp(data_dict["image_sc"]["values"])
                )
            elif image_sc_is_uint8:
                data_dict["image_sc"]["values"] = (
                    new_data["image_sc"]["values"] + data_dict["image_sc"]["values"]
                )
            else:
                raise ValueError("image_sc values must be uint8 or float32")

        assert scan == new_scan, "Scan parameters do not match."
        assert probe == new_probe, "Probe parameters do not match."

    # Divide to get the mean; for uint8, keep float precision then clip and cast back
    if image_is_uint8:
        data_dict["image"]["values"] = np.clip(
            data_dict["image"]["values"] / len(input_paths), 0, 255
        ).astype(np.uint8)
    if image_is_float32:
        data_dict["image"]["values"] = np.minimum(
            data_dict["image"]["values"] - np.log(len(input_paths)), 0.0
        )
    if image_sc_is_uint8:
        data_dict["image_sc"]["values"] = np.clip(
            data_dict["image_sc"]["values"] / len(input_paths), 0, 255
        ).astype(np.uint8)
    if image_sc_is_float32:
        data_dict["image_sc"]["values"] = np.minimum(
            data_dict["image_sc"]["values"] - np.log(len(input_paths)), 0.0
        )

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


def compound_frames(input_path: Path, output_path: Path, overwrite=False):
    """
    Compounds frames in a raw data file by averaging them.

    Args:
        input_path (Path): Path to the input raw data file.
        output_path (Path): Path to the output file where the compounded data will be saved.
        overwrite (bool, optional): Whether to overwrite the output file if it exists. Defaults to
            False.
    """

    data_dict, scan, probe = load_file_all_data_types(input_path)
    additional_elements = load_additional_elements(input_path)
    description = load_description(input_path)

    # Assuming the first dimension is the frame dimension

    # Map-based data types store values in a dict; these need special handling
    _MAP_KEYS = {"beamformed_data", "envelope_data", "image_sc", "image"}
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
        elif key in _LOG_COMPOUND_KEYS and data_dict[key]["values"].dtype == np.float32:
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


def compound_transmits(input_path: Path, output_path: Path, overwrite=False):
    """
    Compounds transmits in a raw data file by averaging them.

    Note
    ----
    This function assumes that all transmits are identical. If this is not the case the function
    will result in incorrect scan parameters.

    Args:
        input_path (Path): Path to the input raw data file.
        output_path (Path): Path to the output file where the compounded data will be saved.
        overwrite (bool, optional): Whether to overwrite the output file if it exists. Defaults to
        False.
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


def resave(input_path: Path, output_path: Path, overwrite=False):
    """
    Resaves a zea data file to a new location.

    Args:
        input_path (Path): Path to the input zea data file.
        output_path (Path): Path to the output file where the data will be saved.
        overwrite (bool, optional): Whether to overwrite the output file if it exists. Defaults to
            False.
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
    )


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
        input_path (Path): Path to the input raw data file.
        output_path (Path): Path to the output file where the extracted data will be saved.
        frame_indices (list, array-like, or slice): Indices of the frames to keep.
        transmit_indices (list, array-like, or slice): Indices of the transmits to keep.
        overwrite (bool, optional): Whether to overwrite the output file if it exists. Defaults to
            False.
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
        description="Manipulate zea data files.",
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
    sum_parser = subparsers.add_parser("sum", help="Sum the raw data of multiple files.")
    sum_parser.add_argument("input_paths", type=Path, nargs="+", help="Paths to the input files.")
    sum_parser.add_argument("output_path", type=Path, help="Output HDF5 file.")
    sum_parser.add_argument(
        "--overwrite", action="store_true", default=False, help="Overwrite existing output file."
    )


def _add_parser_compound_frames(subparsers):
    cf_parser = subparsers.add_parser("compound_frames", help="Compound frames to increase SNR.")
    cf_parser.add_argument("input_path", type=Path, help="Input HDF5 file.")
    cf_parser.add_argument("output_path", type=Path, help="Output HDF5 file.")
    cf_parser.add_argument(
        "--overwrite", action="store_true", default=False, help="Overwrite existing output file."
    )


def _add_parser_compound_transmits(subparsers):
    ct_parser = subparsers.add_parser(
        "compound_transmits", help="Compound transmits to increase SNR."
    )
    ct_parser.add_argument("input_path", type=Path, help="Input HDF5 file.")
    ct_parser.add_argument("output_path", type=Path, help="Output HDF5 file.")
    ct_parser.add_argument(
        "--overwrite", action="store_true", default=False, help="Overwrite existing output file."
    )


def _add_parser_resave(subparsers):
    resave_parser = subparsers.add_parser("resave", help="Resave a file to change format version.")
    resave_parser.add_argument("input_path", type=Path, help="Input HDF5 file.")
    resave_parser.add_argument("output_path", type=Path, help="Output HDF5 file.")
    resave_parser.add_argument(
        "--overwrite", action="store_true", default=False, help="Overwrite existing output file."
    )


def _add_parser_extract(subparsers):
    extract_parser = subparsers.add_parser("extract", help="Extract subset of frames or transmits.")
    extract_parser.add_argument("input_path", type=Path, help="Input HDF5 file.")
    extract_parser.add_argument("output_path", type=Path, help="Output HDF5 file.")
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

    if args.output_path.exists() and not args.overwrite:
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
        resave(input_path=args.input_path, output_path=args.output_path, overwrite=args.overwrite)
    elif args.operation == "extract":
        extract_frames_transmits(
            input_path=args.input_path,
            output_path=args.output_path,
            frame_indices=_interpret_indices(args.frames),
            transmit_indices=_interpret_indices(args.transmits),
            overwrite=args.overwrite,
        )
    else:
        sum_data(
            input_paths=args.input_paths, output_path=args.output_path, overwrite=args.overwrite
        )
