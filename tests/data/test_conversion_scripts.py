"""Test dataset conversion scripts"""

import csv
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import h5py
import imageio
import numpy as np
import pytest
import SimpleITK as sitk
import yaml

from zea.data.convert.images import convert_image_dataset
from zea.data.convert.utils import load_avi, unzip
from zea.io_lib import _SUPPORTED_IMG_TYPES

from .. import DEFAULT_TEST_SEED


@pytest.mark.parametrize(
    "dataset", ["echonet", "camus", "picmus"]
)  # Does not test 'echonetlvh' or 'verasonics' yet
@pytest.mark.heavy
def test_conversion_script(tmp_path_factory, dataset):
    """
    Function that given a dataset name creates some temporary data which is
    similar to the real dataset, runs the corresponding conversion script,
    and verifies the output.
    """
    base = tmp_path_factory.mktemp("base")
    src = base / "src"
    dst = base / "dst"

    create_test_data_for_dataset(dataset, src)
    subprocess.run(
        [sys.executable, "-m", "zea.data.convert", dataset, str(src), str(dst)],
        check=True,
    )
    verify_converted_test_dataset(dataset, dst)

    if dataset == "echonet":
        # For echonet we want to run it again, using the split.yaml file created in dst
        # to verify that the script can copy and verify integrity of existing split files
        # We also test no_hyperthreading with the H5Processor for good measure
        dst2 = tmp_path_factory.mktemp("dst2")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "zea.data.convert",
                dataset,
                str(src),
                str(dst2),
                "--split_path",
                str(dst),
                "--no_hyperthreading",
            ],
            check=True,
        )
        with open(dst / "split.yaml", "r") as f:
            split_content1 = yaml.safe_load(f)
        with open(dst2 / "split.yaml", "r") as f:
            split_content2 = yaml.safe_load(f)
        for split in split_content1.keys():
            assert set(split_content1[split]) == set(split_content2[split]), (
                "Split contents do not match after re-conversion"
            )


def create_test_data_for_dataset(dataset, src):
    """
    Selects the function that generates test data based on the provided dataset

    Args:
        dataset (str): string containing name of the dataset
        src (Path): path to the source directory where test data will be created

    Raises:
        ValueError: If the dataset name is unknown
    """
    if dataset == "echonet":
        create_echonet_test_data(src)
    elif dataset == "echonetlvh":
        create_echonetlvh_test_data(src)
    elif dataset == "camus":
        create_camus_test_data(src)
    elif dataset == "picmus":
        create_picmus_test_data(src)
    elif dataset == "verasonics":
        create_verasonics_test_data(src)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def verify_converted_test_dataset(dataset, dst):
    """
    Selects the function that reads the converted test dataset based on the provided dataset

    Args:
        dataset (str): string containing name of the dataset
        dst (Path): path to the destination directory where converted test data is located

    Raises:
        ValueError: If the dataset name is unknown
    """

    if dataset == "echonet":
        verify_converted_echonet_test_data(dst)
    elif dataset == "echonetlvh":
        verify_converted_echonetlvh_test_data(dst)
    elif dataset == "camus":
        verify_converted_camus_test_data(dst)
    elif dataset == "picmus":
        verify_converted_picmus_test_data(dst)
    elif dataset == "verasonics":
        verify_converted_verasonics_test_data(dst)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def create_echonet_test_data(src):
    """
    Creates test AVI files with random content in the expected directory
    structure for the EchoNet dataset. They should be defined such that
    the convert function splits them evenly into train/val/test/rejected sets
    and creates a split.yaml file.

    Args:
        src (Path): path to the source directory where test data will be created.

    """
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    os.mkdir(src)
    os.mkdir(src / "EchoNet-Dynamic")
    os.mkdir(src / "EchoNet-Dynamic" / "Videos")

    accepted_files = 10 * np.abs(rng.normal(size=(6, 112, 112)))

    # Create a file with missing bottom left corner
    missing_bottom_left = 10 * np.abs(rng.normal(size=(1, 112, 112)))
    rows_lower = np.linspace(78, 47, 21).astype(np.int32)
    rows_upper = np.linspace(67, 47, 21).astype(np.int32)
    for idx, row in enumerate(rows_lower):
        missing_bottom_left[0, rows_upper[idx] : row, idx] = 0

    # Create a file with missing bottom right corner
    missing_bottom_right = 10 * np.abs(rng.normal(size=(1, 112, 112)))
    cols = np.linspace(70, 111, 42).astype(np.int32)
    rows_bot = np.linspace(17, 57, 42).astype(np.int32)
    rows_top = np.linspace(17, 80, 42).astype(np.int32)
    for i, col in enumerate(cols):
        missing_bottom_right[0, rows_bot[i] : rows_top[i], col] = 0

    files = np.concatenate([accepted_files, missing_bottom_left, missing_bottom_right], axis=0)
    # Make a single avi file for each sample
    for i, file_data in enumerate(files):
        avi_path = src / "EchoNet-Dynamic" / "Videos" / f"video_{i}.avi"
        with imageio.get_writer(avi_path, fps=30, codec="ffv1") as writer:
            writer.append_data(file_data)


def create_echonetlvh_test_data(src):
    raise NotImplementedError("Test data generation for EchoNetLVH is not implemented yet.")


def create_camus_test_data(src):
    """
    Creates test data representing the CAMUS dataset.
    Makes a folder CAMUS_public with in it, database_nifti and database_split folders
    database_nifti folder:
        patient0001 folder:
            file.nii.gz (SimpleITK image with random data and metadata)
        patient0002 folder:
            ...
    database_split folder:
        can be empty

    Args:
        src (Path): path to the source directory where test data will be created.
    """
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    os.mkdir(src)
    os.mkdir(src / "CAMUS_public")
    os.mkdir(src / "CAMUS_public" / "database_nifti")
    os.mkdir(src / "CAMUS_public" / "database_split")

    data_folder = src / "CAMUS_public" / "database_nifti"
    for i in [50, 420, 470]:  # Patients to be put in train, val, test
        patient_folder = data_folder / f"patient{i:04d}"
        os.mkdir(patient_folder)
        filepath = patient_folder / f"patient{i:04d}_half_sequence.nii.gz"

        # Create some data that does not crash the
        # transform_sc_image_to_polar function in camus.py
        img = np.zeros((32, 32), dtype=float)
        active_cols = rng.choice(32, size=30, replace=False)
        active_cols.sort()
        for c in active_cols:
            start = rng.integers(0, 32 // 4)
            length = rng.integers(32 // 2, 32)
            end = min(32, start + length)
            img[start:end, c] = rng.uniform(0.2, 1.0, end - start)
        img_set = []
        for _ in range(10):
            noise = rng.normal(0, 0.02, (32, 32))
            img += noise
            img = np.clip(img, 0, None)
            img_set.append(img.copy())
        img = np.stack(img_set, axis=0)
        img[:, 0, :] = 0.0

        # Create SimpleITK image with metadata
        image = sitk.GetImageFromArray(img)
        image.SetOrigin((0.0, 0.0, 0.0))
        image.SetSpacing((1.0, 1.0, 1.0))
        image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        image.SetMetaData("PatientName", "John Doe")
        image.SetMetaData("Modality", "US")
        image.SetMetaData("StudyDate", "01011970")
        sitk.WriteImage(image, str(filepath))


def create_picmus_test_data(src):
    """
    Creates test hdf5 files ending in iq or rf with random content,
    representative of the subset of picmus files we process.
    These files must contain:
        ["US"]["US_DATASET0000"]["data"]["real"]
        ["US"]["US_DATASET0000"]["data"]["imag"]
        ["US"]["US_DATASET0000"]["modulation_frequency"][":"][0]
        ["US"]["US_DATASET0000"]["sampling_frequency"][":"][0]
        ["US"]["US_DATASET0000"]["probe_geometry"][":"]
        ["US"]["US_DATASET0000"]["sound_speed"][":"][0]
        ["US"]["US_DATASET0000"]["angles"][":"]

    Args:
        src (Path): path to the source directory where test data will be created.
    """
    os.mkdir(src)
    os.mkdir(src / "archive_to_download")
    os.mkdir(src / "archive_to_download" / "parent_folder")
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    for name in ["test1_iq.hdf5", "test2_rf.hdf5", "ignore_me.hdf5"]:
        file_path = src / "archive_to_download" / "parent_folder" / name
        with h5py.File(file_path, "w") as f:
            us_group = f.create_group("US")
            dataset_group = us_group.create_group("US_DATASET0000")
            data_group = dataset_group.create_group("data")
            n_tx = 5
            n_el = 32
            n_samples = 128
            real_part = rng.normal(size=(n_tx, n_el, n_samples)).astype(np.float32)
            imag_part = rng.normal(size=(n_tx, n_el, n_samples)).astype(np.float32)
            data_group.create_dataset("real", data=real_part)
            data_group.create_dataset("imag", data=imag_part)
            dataset_group.create_dataset(
                "modulation_frequency", data=np.array([5e6], dtype=np.float32)
            )
            dataset_group.create_dataset(
                "sampling_frequency", data=np.array([20e6], dtype=np.float32)
            )
            probe_geometry = rng.uniform(-0.01, 0.01, size=(3, n_el)).astype(np.float32)
            dataset_group.create_dataset("probe_geometry", data=probe_geometry)
            dataset_group.create_dataset("sound_speed", data=np.array([1540.0], dtype=np.float32))
            angles = np.linspace(-np.pi / 6, np.pi / 6, n_tx).astype(np.float32)
            dataset_group.create_dataset("angles", data=angles)
    assert len(list((src / "archive_to_download").rglob("*.hdf5"))) == 3, (
        "Failed to create test PICMUS hdf5 files."
    )


def create_verasonics_test_data(src):
    raise NotImplementedError("Test data generation for Verasonics is not implemented yet.")


def verify_converted_echonet_test_data(dst):
    """
    Verify that the converted EchoNet test dataset has the correct structure with hdf5 files
    in train/val/test/rejected folders for every original AVI file. The split.yaml file is
    already test in the test_conversion_script function.

    Args:
        dst (Path): path to the destination directory where converted test data is located.
    """
    # List all hdf5 files in the splits
    all_files = []
    for split in ["train", "val", "test", "rejected"]:
        split_dir = dst / split
        assert split_dir.exists(), f"Missing directory: {split_dir}"
        h5_files = list(split_dir.rglob("*.hdf5"))
        all_files.append(h5_files)
        # The rejected split should have video_6 and video_7 only
        if split == "rejected":
            rejected_filenames = [f.name for f in h5_files]
            assert set(rejected_filenames) == {"video_6.hdf5", "video_7.hdf5"}, (
                "Rejected split does not have the expected files"
            )

    # Verify that the set of hdf5 files is video_0.hdf5 to video_7.hdf5
    all_h5_files = [f.name for split_files in all_files for f in split_files]
    expected_files = [f"video_{i}.hdf5" for i in range(8)]
    assert set(all_h5_files) == set(expected_files), "Mismatch in converted hdf5 files"


def verify_converted_echonetlvh_test_data(dst):
    raise NotImplementedError("Verification for EchoNetLVH conversion is not implemented yet.")


def verify_converted_camus_test_data(dst):
    """
    Verify that all 3 created nifti files were converted to zea format and split correctly.

    Args:
        dst (Path): Path to the destination directory where converted test data is located.
    """
    splits = ["train", "val", "test"]
    expected_patients = {
        "train": ["patient0050_half_sequence.hdf5"],
        "val": ["patient0420_half_sequence.hdf5"],
        "test": ["patient0470_half_sequence.hdf5"],
    }
    for split in splits:
        split_dir = dst / split
        assert split_dir.exists(), f"Missing directory: {split_dir}"
        h5_files = list(split_dir.rglob("*.hdf5"))
        h5_filenames = [f.name for f in h5_files]
        assert set(h5_filenames) == set(expected_patients[split]), (
            f"Mismatch in converted hdf5 files for split {split}"
        )

        # Load the hdf5 file and check for expected datasets
        for h5_file in h5_files:
            with h5py.File(h5_file, "r") as f:
                assert "scan" in f, f"Missing 'scan' in {h5_file}"


def verify_converted_picmus_test_data(dst):
    """
    Verify that 2/3 of the created hdf5 files were converted to zea format.

    Args:
        dst (Path): Path to the destination directory where converted test data is located.
    """
    h5_files = list(dst.rglob("*.hdf5"))
    assert len(h5_files) == 2, "Expected 2 converted hdf5 files."

    # Check that the files contain data
    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            assert "data" in f, f"Missing 'data' in {h5_file}"
            assert "scan" in f, f"Missing 'scan' in {h5_file}"


def verify_converted_verasonics_test_data(dst):
    raise NotImplementedError("Verification for Verasonics conversion is not implemented yet.")


@pytest.mark.parametrize("image_type", _SUPPORTED_IMG_TYPES)
def test_convert_image_dataset(tmp_path_factory, image_type):
    """Test the convert_image_dataset function from zea.data.convert.images"""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    src = tmp_path_factory.mktemp("src")
    dst = tmp_path_factory.mktemp("dst")

    # Create a temporary directory structure with image files
    subdirs = ["dir1", "dir2/subdir"]
    for subdir in subdirs:
        dir_path = src / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            img_array = rng.integers(0, 256, (32, 32), dtype=np.uint8)
            img_path = dir_path / f"image_{i}{image_type}"
            imageio.imwrite(img_path, img_array)

    # Convert the image dataset
    convert_image_dataset(
        existing_dataset_root=str(src),
        new_dataset_root=str(dst),
        dataset_name="test_images",
    )

    # Verify that the converted dataset exists and has the expected structure
    for subdir in subdirs:
        new_dir_path = dst / subdir
        assert new_dir_path.exists()
        for i in range(5):
            h5_path = new_dir_path / f"image_{i}.hdf5"
            assert h5_path.exists()


def test_load_avi(tmp_path):
    """Test the load_avi function from zea.data.convert.utils"""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    # Create a temporary AVI file with known content
    avi_path = tmp_path / "test_video.avi"
    frames = [rng.integers(0, 256, (32, 32), dtype=np.uint8) for _ in range(10)]
    with imageio.get_writer(avi_path, fps=10, codec="ffv1") as writer:
        for frame in frames:
            writer.append_data(frame)

    # Load the AVI file using the function
    loaded_frames = load_avi(avi_path, mode="L")

    # Verify the shape and content
    assert loaded_frames.shape == (10, 32, 32)
    for i in range(10):
        np.testing.assert_allclose(loaded_frames[i], frames[i], atol=1)


@pytest.mark.parametrize(
    "dataset",
    [
        ("picmus", "picmus.zip", "archive_to_download"),
        ("camus", "CAMUS_public.zip", "CAMUS_public"),
        ("echonet", "EchoNet-Dynamic.zip", "EchoNet-Dynamic"),
        ("echonetlvh", "EchoNet-LVH.zip", "Batch1"),
    ],
)
def test_unzip(tmp_path, dataset):
    """Test the unzip function from zea.data.convert.utils for all dataset-name pairs"""
    dataset_name, zip_filename, folder_name = dataset
    # Create a dummy zip file
    zip_path = tmp_path / zip_filename
    with zipfile.ZipFile(zip_path, "w") as zipf:
        # Add a dummy file to the zip
        if dataset_name == "echonet":
            # Match unzip()’s expected structure: EchoNet-Dynamic/Videos/...
            zipf.writestr(f"{folder_name}/Videos/dummy.txt", "This is a test.")
        else:
            zipf.writestr(f"{folder_name}/dummy.txt", "This is a test.")

        if dataset_name == "echonetlvh":
            # EchoNetLVH dataset unzips into four folders and a csv file.
            zipf.writestr("Batch2/dummy.txt", "This is a test.")
            zipf.writestr("Batch3/dummy.txt", "This is a test.")
            zipf.writestr("Batch4/dummy.txt", "This is a test.")

            with open(Path(f"{tmp_path}/MeasurementsList.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "mean_value"])
                for i in range(3):
                    writer.writerow([i, i * 2])  # example data
            zipf.write(f"{tmp_path}/MeasurementsList.csv", "MeasurementsList.csv")

    # Call the unzip function twice:
    # Once to initialize from zip, once to initialize from folder
    unzip(tmp_path, dataset_name)
    unzip(tmp_path, dataset_name)

    # Verify that the folder was created and contains the dummy file
    if dataset_name == "echonet":
        extracted_folder = tmp_path / folder_name / "Videos"
    else:
        extracted_folder = tmp_path / folder_name

    assert extracted_folder.exists()
    assert (extracted_folder / "dummy.txt").exists()
