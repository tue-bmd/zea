"""Test the file operations module."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Generator

import h5py
import numpy as np
import pytest

from zea import Parameters
from zea.data.data_format import (
    load_additional_elements,
    load_description,
)
from zea.data.file import File, load_file_all_data_types, validate_file
from zea.data.file_operations import (
    compound_frames,
    compound_transmits,
    extract_frames_transmits,
    resave,
    save_file,
    sum_data,
)

from . import generate_dummy_scan, generate_example_dataset


@pytest.fixture
def tmp_hdf5_path(tmp_path) -> Generator[Path, None, None]:
    """Fixture to create a temporary HDF5 file."""
    yield Path(tmp_path, "test_case_dataset.hdf5")


def test_file_operations_sum(tmp_hdf5_path):
    """Tests the sum_data function by creating two example datasets,
    summing them and checking if the result is correct."""

    # Create two example datasets
    input_path1 = tmp_hdf5_path.parent / "test_case_dataset1.hdf5"
    input_path2 = tmp_hdf5_path.parent / "test_case_dataset2.hdf5"
    generate_example_dataset(input_path1, add_optional_dtypes=True, image_dtype=np.float32)
    generate_example_dataset(input_path2, add_optional_dtypes=True, image_dtype=np.float32)

    with File(input_path1) as f:
        data1 = f["data/raw_data"][:]
    with File(input_path2) as f:
        data2 = f["data/raw_data"][:]

    # Sum the datasets
    output_path = tmp_hdf5_path.parent / "summed_dataset.hdf5"

    sum_data([input_path1, input_path2], output_path)

    _assert_descriptions_and_additional_elements_equal(input_path1, output_path)

    # Load the summed dataset and check if the data is correct
    with File(output_path) as f:
        raw_data = f["data/raw_data"][:]
        assert raw_data[0, 0, 0, 0, 0] == data1[0, 0, 0, 0, 0] + data2[0, 0, 0, 0, 0]


def test_file_operations_extract(tmp_hdf5_path):
    """Tests the load_data function by creating an example dataset and
    loading a subset of the data."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "extracted_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path, add_optional_dtypes=True, image_dtype=np.float32)

    extract_frames_transmits(
        input_path, output_path, frame_indices=slice(1), transmit_indices=[0, 3]
    )
    data_dict, parameters = load_file_all_data_types(output_path)
    data_dict = SimpleNamespace(**data_dict)

    _assert_descriptions_and_additional_elements_equal(input_path, output_path)

    assert data_dict.raw_data.shape[0] == 1
    assert data_dict.raw_data.shape[1] == 2
    assert data_dict.aligned_data["values"].shape[0] == 1
    assert data_dict.aligned_data["values"].shape[1] == 2
    assert data_dict.beamformed_data["values"].shape[0] == 1

    _assert_beamformed_data_still_exists(output_path)
    _assert_descriptions_and_additional_elements_equal(input_path, output_path)


def test_file_operations_resave(tmp_hdf5_path):
    """Tests the resave operation by creating an example dataset and
    resaving it to a new file."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "resaved_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path, add_optional_dtypes=True, image_dtype=np.float32)

    resave(input_path, output_path)

    _assert_descriptions_and_additional_elements_equal(input_path, output_path)

    # Validate the resaved dataset
    validate_file(output_path)


def test_file_operations_compound_frames(tmp_hdf5_path):
    """Tests the compound_frames function by creating an example dataset and
    compounding frames."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "compounded_frames_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path, add_optional_dtypes=True, image_dtype=np.float32)

    compound_frames(input_path, output_path)

    _assert_descriptions_and_additional_elements_equal(input_path, output_path)

    data_dict, parameters = load_file_all_data_types(output_path)
    data_dict = SimpleNamespace(**data_dict)
    for dataset in vars(data_dict).values():
        if dataset is None:
            continue
        arr = dataset["values"] if isinstance(dataset, dict) else dataset
        assert arr.shape[0] == 1  # Only one frame should remain


def test_file_operations_compound_transmits(tmp_hdf5_path):
    """Tests the compound_transmits function by creating an example dataset and
    compounding transmits."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "compounded_transmits_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path, add_optional_dtypes=True, image_dtype=np.float32)

    compound_transmits(input_path, output_path)

    _assert_descriptions_and_additional_elements_equal(input_path, output_path)

    with File(output_path) as f:
        data = f["data/raw_data"][:]
        parameters = f.load_parameters()
    assert data.shape[1] == 1  # Only one transmit should remain
    assert parameters["initial_times"].shape[0] == 1
    assert parameters["t0_delays"].shape[0] == 1
    assert parameters["azimuth_angles"].shape[0] == 1
    assert parameters["tx_apodizations"].shape[0] == 1


def test_file_operations_cli_sum(tmp_hdf5_path):
    """Tests the sum_data function CLI by creating two example datasets,
    summing them and checking if the result is correct."""

    # Create two example datasets
    path1 = tmp_hdf5_path.parent / "test_case_dataset1.hdf5"
    path2 = tmp_hdf5_path.parent / "test_case_dataset2.hdf5"
    generate_example_dataset(path1, add_optional_dtypes=True, image_dtype=np.float32)
    generate_example_dataset(path2, add_optional_dtypes=True, image_dtype=np.float32)

    with File(path1) as f:
        data1 = f["data/raw_data"][:]
    with File(path2) as f:
        data2 = f["data/raw_data"][:]

    # Sum the datasets
    output_path = tmp_hdf5_path.parent / "summed_dataset.hdf5"

    os.system(
        "python -m zea.data.file_operations sum "
        + str(path1)
        + " "
        + str(path2)
        + " "
        + str(output_path)
    )

    # Load the summed dataset and check if the data is correct
    with File(output_path) as f:
        raw_data = f["data/raw_data"][:]
        assert raw_data[0, 0, 0, 0, 0] == data1[0, 0, 0, 0, 0] + data2[0, 0, 0, 0, 0]


def test_file_operations_cli_extract(tmp_hdf5_path):
    """Tests the load_data function CLI by creating an example dataset and
    loading a subset of the data."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "extracted_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path, add_optional_dtypes=True, image_dtype=np.float32)

    os.system(
        "python -m zea.data.file_operations extract "
        + str(input_path)
        + " "
        + str(output_path)
        + " --frames 0-1 --transmits 0 3 4"
    )

    data_dict, parameters = load_file_all_data_types(output_path)
    data_dict = SimpleNamespace(**data_dict)
    assert data_dict.raw_data.shape[0] == 2
    assert data_dict.raw_data.shape[1] == 3
    assert data_dict.aligned_data["values"].shape[0] == 2
    assert data_dict.aligned_data["values"].shape[1] == 3
    assert data_dict.beamformed_data["values"].shape[0] == 2


def test_file_operations_cli_resave(tmp_hdf5_path):
    """Tests the resave operation CLI by creating an example dataset and
    resaving it to a new file."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "resaved_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path, add_optional_dtypes=True, image_dtype=np.float32)

    os.system(
        "python -m zea.data.file_operations resave " + str(input_path) + " " + str(output_path)
    )

    # Validate the resaved dataset
    validate_file(output_path)


def test_file_operations_cli_compound_frames(tmp_hdf5_path):
    """Tests the compound_frames function CLI by creating an example dataset and
    compounding frames."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "compounded_frames_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path, add_optional_dtypes=True, image_dtype=np.float32)

    os.system(
        "python -m zea.data.file_operations compound_frames "
        + str(input_path)
        + " "
        + str(output_path)
    )

    data_dict, parameters = load_file_all_data_types(output_path)
    data_dict = SimpleNamespace(**data_dict)
    assert data_dict.raw_data.shape[0] == 1  # Only one frame should remain
    assert data_dict.aligned_data["values"].shape[0] == 1
    assert data_dict.beamformed_data["values"].shape[0] == 1


def test_file_operations_cli_compound_transmits(tmp_hdf5_path):
    """Tests the compound_transmits function CLI by creating an example dataset and
    compounding transmits."""

    input_path = tmp_hdf5_path.parent / "test_case_dataset.hdf5"
    output_path = tmp_hdf5_path.parent / "compounded_transmits_dataset.hdf5"

    # Create an example dataset
    generate_example_dataset(input_path, add_optional_dtypes=True, image_dtype=np.float32)

    os.system(
        "python -m zea.data.file_operations compound_transmits "
        + str(input_path)
        + " "
        + str(output_path)
    )

    data_dict, parameters = load_file_all_data_types(output_path)
    data_dict = SimpleNamespace(**data_dict)
    assert data_dict.raw_data.shape[1] == 1  # Only one transmit should remain
    assert data_dict.aligned_data["values"].shape[1] == 1


def test_file_operations_folder_resave(tmp_path):
    """Tests that resave works on a folder of files, mirroring the folder structure."""

    input_folder = tmp_path / "input"
    output_folder = tmp_path / "output"

    # Create a folder of example datasets, including a nested subfolder
    input_paths = [
        input_folder / "case_0.hdf5",
        input_folder / "case_1.hdf5",
        input_folder / "nested" / "case_2.hdf5",
    ]
    for path in input_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_example_dataset(path, add_optional_dtypes=True)

    resave(input_folder, output_folder)

    # Each input file should have a matching output file at the mirrored location
    for input_path in input_paths:
        output_path = output_folder / input_path.relative_to(input_folder)
        assert output_path.is_file()
        validate_file(output_path)
        _assert_descriptions_and_additional_elements_equal(input_path, output_path)


def test_file_operations_folder_compound_frames(tmp_path):
    """Tests that compound_frames works on a folder of files."""

    input_folder = tmp_path / "input"
    output_folder = tmp_path / "output"
    input_folder.mkdir()

    input_paths = [input_folder / "case_0.hdf5", input_folder / "case_1.hdf5"]
    for path in input_paths:
        generate_example_dataset(path, add_optional_dtypes=True)

    compound_frames(input_folder, output_folder)

    for input_path in input_paths:
        output_path = output_folder / input_path.name
        assert output_path.is_file()
        data_dict, _ = load_file_all_data_types(output_path)
        for dataset in data_dict.values():
            if dataset is not None:
                arr = dataset["values"] if isinstance(dataset, dict) else dataset
                assert arr.shape[0] == 1  # Only one frame should remain


def test_file_operations_folder_sum(tmp_path):
    """Tests that sum_data accepts a folder and sums all files it contains."""

    input_folder = tmp_path / "input"
    input_folder.mkdir()
    output_path = tmp_path / "summed.hdf5"

    input_paths = [input_folder / "case_0.hdf5", input_folder / "case_1.hdf5"]
    for path in input_paths:
        generate_example_dataset(path, add_optional_dtypes=True)

    with File(input_paths[0]) as f:
        data0 = f["data/raw_data"][:]
    with File(input_paths[1]) as f:
        data1 = f["data/raw_data"][:]

    sum_data(input_folder, output_path)

    with File(output_path) as f:
        raw_data = f["data/raw_data"][:]
        assert raw_data[0, 0, 0, 0, 0] == data0[0, 0, 0, 0, 0] + data1[0, 0, 0, 0, 0]


def _load_description_and_additional_elements(path: Path):
    description = load_description(path)
    additional_elements = load_additional_elements(path)
    return description, additional_elements


def _assert_descriptions_and_additional_elements_equal(path, other_path: Path):
    description, additional_elements = _load_description_and_additional_elements(path)
    other_description, other_additional_elements = _load_description_and_additional_elements(
        other_path
    )
    assert description == other_description
    assert len(additional_elements) == len(other_additional_elements)


def _assert_beamformed_data_still_exists(path: Path):
    with h5py.File(path, "r") as f:
        key = "tracks/track_0/data/beamformed_data" if "tracks" in f else "data/beamformed_data"
        assert key in f


def _make_file_with_distinct_demod_freq(tmp_path, demod_freq=5e6, center_freq=7e6):
    """Create a file via save_file with distinct demodulation / center frequencies."""

    n_tx, n_el, n_ax = 4, 16, 64
    scan_dict = generate_dummy_scan(n_tx=n_tx, n_el=n_el, center_frequency=center_freq)
    scan_dict["n_tx"] = n_tx
    scan_dict["n_ax"] = n_ax
    scan_dict["demodulation_frequency"] = np.float32(demod_freq)

    parameters = Parameters(**scan_dict)
    raw = np.zeros((2, n_tx, n_ax, n_el, 1), dtype=np.float32)

    path = tmp_path / "scan_demod.hdf5"
    save_file(path=path, parameters=parameters, raw_data=raw)
    return path, demod_freq, center_freq


def test_demodulation_frequency_saved_correctly(tmp_path):
    """save_file must store demodulation_frequency from scan.demodulation_frequency,
    not from scan.center_frequency."""
    path, demod_freq, center_freq = _make_file_with_distinct_demod_freq(
        tmp_path, demod_freq=5e6, center_freq=7e6
    )
    assert demod_freq != center_freq, "test requires distinct demod/center frequencies"

    with File(path) as f:
        stored = float(f["scan/demodulation_frequency"][()])

    assert stored == pytest.approx(demod_freq), (
        f"demodulation_frequency should be {demod_freq} Hz, got {stored} Hz"
    )
    assert stored != pytest.approx(center_freq), (
        "demodulation_frequency must not be equal to center_frequency"
    )


def test_sum_data_without_image(tmp_path):
    """sum_data must succeed on files that contain only raw_data (no image or
    image_sc), without raising TypeError from unconditional dict access."""
    input1 = tmp_path / "raw1.hdf5"
    input2 = tmp_path / "raw2.hdf5"
    output = tmp_path / "summed.hdf5"

    generate_example_dataset(input1, add_optional_dtypes=False)
    generate_example_dataset(input2, add_optional_dtypes=False)

    sum_data([input1, input2], output)
    assert output.exists()


def test_uint8_sum_no_truncation(tmp_path):
    """Averaging two uint8 images must not truncate the intermediate sum.
    Pixel value 200 in each file → sum 400 → if cast to uint8 before /2 wraps
    to 144/2 = 72 (wrong); correct answer is 400/2 = 200."""
    input1 = tmp_path / "img1.hdf5"
    input2 = tmp_path / "img2.hdf5"
    output = tmp_path / "summed_img.hdf5"

    grid = 16
    generate_example_dataset(
        input1,
        add_optional_dtypes=True,
        grid_size_z=grid,
        grid_size_x=grid,
        image_dtype=np.uint8,
    )
    generate_example_dataset(
        input2,
        add_optional_dtypes=True,
        grid_size_z=grid,
        grid_size_x=grid,
        image_dtype=np.uint8,
    )

    for p in (input1, input2):
        with h5py.File(p, "r+") as hf:
            key = "tracks/track_0/data/image/values" if "tracks" in hf else "data/image/values"
            hf[key][0, 0, 0] = 200

    sum_data([input1, input2], output)

    result, _ = load_file_all_data_types(output)
    pixel = result["image"]["values"][0, 0, 0]

    assert pixel == 200, f"Expected 200, got {pixel}"
    assert result["image"]["values"].dtype == np.uint8


def test_compound_frames_uint8_linear(tmp_path):
    """compound_frames must use linear averaging for uint8 images, not
    log(mean(exp(...))), which is semantically wrong for integer data."""
    input_path = tmp_path / "frames.hdf5"
    output_path = tmp_path / "compounded.hdf5"

    grid = 16
    n_frames = 4
    generate_example_dataset(
        input_path,
        add_optional_dtypes=True,
        n_frames=n_frames,
        grid_size_z=grid,
        grid_size_x=grid,
        image_dtype=np.uint8,
    )

    with h5py.File(input_path, "r+") as hf:
        key = "tracks/track_0/data/image/values" if "tracks" in hf else "data/image/values"
        hf[key][:] = 100

    compound_frames(input_path, output_path)

    result, _ = load_file_all_data_types(output_path)
    pixel = float(result["image"]["values"][0, 0, 0])

    assert pixel == pytest.approx(100, abs=1), f"Expected ~100, got {pixel}"


def test_save_file_from_parameters_round_trip(tmp_path):
    """Round-trip: generate a file, load its Parameters, save to a new file, validate.

    This is the canonical usage pattern for reprocessing: load parameters from an
    existing file.
    """
    src_path = tmp_path / "source.hdf5"
    dst_path = tmp_path / "output.hdf5"

    generate_example_dataset(src_path)

    # Load parameters and raw data from the source file
    with File(src_path) as f:
        parameters = f.load_parameters()
        raw_data = f.data.raw_data[:]

    scan = parameters.to_scan_dict()
    probe = parameters.to_probe_dict()

    File.create(
        path=dst_path,
        data={
            "raw_data": raw_data,
        },
        scan=scan,
        probe=probe,
        description="Test dataset for save_file round-trip",
    )

    # Output file must pass full spec validation
    validate_file(dst_path)

    # Data must round-trip exactly
    with File(dst_path) as f:
        loaded_raw_data = f.data.raw_data[:]
        loaded_parameters = f.load_parameters()

    np.testing.assert_array_equal(loaded_raw_data, raw_data)
    # The full parameters object must round-trip: every stored key/value pair
    # must be identical after a save + load cycle.
    assert loaded_parameters == parameters, "Parameters did not round-trip: " + repr(
        {
            k: (v, loaded_parameters._params.get(k))
            for k, v in parameters.items()
            if not np.array_equal(v, loaded_parameters._params.get(k))
        }
    )
