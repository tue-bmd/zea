"""Test dataset conversion scripts"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import h5py
import imageio
import numpy as np
import pytest
import SimpleITK as sitk
import yaml

from zea.data.convert.images import convert_image_dataset
from zea.data.convert.utils import (
    check_output_dir_ownership,
    load_avi,
    require_output_dir_ownership,
    sitk_load,
    unzip,
)
from zea.data.convert.verasonics import (
    VerasonicsFile,
    bs100bw_to_iq,
    convert_verasonics,
    estimate_lens_probe_params,
)
from zea.data.file import File
from zea.func.tensor import translate
from zea.internal.preset_utils import _hf_resolve_path
from zea.io_lib import _SUPPORTED_IMG_TYPES

from .. import DEFAULT_TEST_SEED


def run_subprocess(cmd, **kwargs):
    """Run a subprocess, letting output flow to pytest's capture.

    Pytest shows captured stdout/stderr whenever the test fails, including
    when a later assertion fails — so leaving subprocess output uncaptured
    here gives us full visibility into what the conversion script did.
    """
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        pytest.fail(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result


@pytest.mark.parametrize(
    "dataset", ["echonet", "echonetlvh", "camus", "cetus", "picmus", "verasonics"]
)
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

    extra_args = create_test_data_for_dataset(dataset, src)
    dst.mkdir()

    run_subprocess(
        [sys.executable, "-m", "zea.data.convert", dataset, str(src), str(dst), *extra_args],
        env=create_env_for_dataset(dataset),
    )
    verify_converted_test_dataset(dataset, src, dst)

    if dataset == "echonet":
        # For echonet we want to run it again, using the split.yaml file created in dst
        # to verify that the script can copy and verify integrity of existing split files
        # We also test no_hyperthreading with the H5Processor for good measure
        dst2 = tmp_path_factory.mktemp("dst2")
        run_subprocess(
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
        )
        with open(dst / "split.yaml", "r") as f:
            split_content1 = yaml.safe_load(f)
        with open(dst2 / "split.yaml", "r") as f:
            split_content2 = yaml.safe_load(f)
        for split in split_content1.keys():
            assert set(split_content1[split]) == set(split_content2[split]), (
                "Split contents do not match after re-conversion"
            )


def create_env_for_dataset(dataset):
    env = os.environ.copy()
    if dataset == "echonetlvh":
        env["KERAS_BACKEND"] = "jax"
    return env


def create_test_data_for_dataset(dataset, src):
    """
    Selects the function that generates test data based on the provided dataset

    Args:
        dataset (str): string containing name of the dataset
        src (Path): path to the source directory where test data will be created

    Raises:
        ValueError: If the dataset name is unknown
    """
    extra_args = []
    os.mkdir(src)
    if dataset == "echonet":
        create_echonet_test_data(src)
    elif dataset == "echonetlvh":
        extra_args = create_echonetlvh_test_data(src)
    elif dataset == "camus":
        extra_args = create_camus_test_data(src)
    elif dataset == "cetus":
        extra_args = create_cetus_test_data(src)
    elif dataset == "picmus":
        create_picmus_test_data(src)
    elif dataset == "verasonics":
        create_verasonics_test_data(src)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return extra_args


def verify_converted_test_dataset(dataset, src, dst):
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
    elif dataset == "cetus":
        verify_converted_cetus_test_data(dst)
    elif dataset == "picmus":
        verify_converted_picmus_test_data(dst)
    elif dataset == "verasonics":
        verify_converted_verasonics_test_data(src, dst)
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
    """
    Creates test AVI files with scan cone structure for EchoNet-LVH dataset.

    The test data includes:
    - A MeasurementsList.csv with split assignments and measurement coordinates
    - AVI files in Batch1 folder containing scan-converted images (with scan cone)
    - Padding around the scan cone that should be cropped by conversion

    Args:
        src (Path): path to the source directory where test data will be created.
    """
    extra_args = []

    from zea.display import scan_convert_2d

    rng = np.random.default_rng(DEFAULT_TEST_SEED)

    # Create directory structure (all 4 batch folders required by unzip check)
    os.mkdir(src / "Batch1")
    os.mkdir(src / "Batch2")
    os.mkdir(src / "Batch3")
    os.mkdir(src / "Batch4")

    # Define test files with their splits and polar shapes (some odd, some even width)
    test_files = [
        ("0X1111111111111111", "train", (64, 49)),  # Odd width
        ("0X2222222222222222", "train", (64, 48)),  # (will be rejected)
        ("0X3333333333333333", "val", (64, 48)),  # Even width
        ("0X4444444444444444", "test", (64, 48)),
        ("0X5555555555555555", "train", (64, 48)),  # Will cause crop to overshoot
    ]

    # Create a test rejections file with one entry
    rejection_path = src / "test_rejections.txt"
    with open(rejection_path, "w") as f:
        f.write("0X2222222222222222\n")

    # Add the rejection_path to extra_args for CLI
    extra_args.extend(["--rejection_path", str(rejection_path)])

    # Common parameters for scan conversion
    rho_range = (0.0, 60.0)  # mm
    theta_range = (-np.pi / 4, np.pi / 4)  # radians

    # Padding to add around scan cone (should be cropped by conversion)
    pad_top = 10
    pad_bottom = 8
    pad_left = 15
    pad_right = 12

    n_frames = 5
    fps = 30

    # Create MeasurementsList.csv
    csv_path = src / "MeasurementsList.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Unnamed: 0",
            "HashedFileName",
            "Calc",
            "CalcValue",
            "Frame",
            "X1",
            "X2",
            "Y1",
            "Y2",
            "Frames",
            "FPS",
            "Width",
            "Height",
            "split",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        row_idx = 0
        for filename, split, polar_shape in test_files:
            # Generate a reference frame to determine output dimensions for this file
            ref_polar = np.ones(polar_shape, dtype=np.float32)
            ref_cartesian, _ = scan_convert_2d(
                ref_polar,
                rho_range=rho_range,
                theta_range=theta_range,
                resolution=1.0,
            )
            ref_cartesian = np.array(ref_cartesian)
            cart_height, cart_width = ref_cartesian.shape

            # Final image dimensions after padding
            final_width = cart_width + pad_left + pad_right
            final_height = cart_height + pad_top + pad_bottom

            # Write multiple measurement rows per file (like real dataset)
            for calc_type in ["LVPWd", "LVIDs", "LVIDd", "IVSd"]:
                # Generate coordinates within the padded image bounds
                x1 = pad_left + rng.integers(10, cart_width // 2)
                x2 = x1 + rng.integers(10, 30)
                y1 = pad_top + rng.integers(10, cart_height // 2)
                y2 = y1 + rng.integers(20, 50)

                writer.writerow(
                    {
                        "Unnamed: 0": row_idx,
                        "HashedFileName": filename,
                        "Calc": calc_type,
                        "CalcValue": rng.uniform(1.0, 5.0),
                        "Frame": rng.integers(0, n_frames),
                        "X1": float(x1),
                        "X2": float(x2),
                        "Y1": float(y1),
                        "Y2": float(y2),
                        "Frames": n_frames,
                        "FPS": fps,
                        "Width": int(final_width),
                        "Height": int(final_height),
                        "split": split,
                    }
                )
                row_idx += 1

    # Create AVI files with scan cone structure
    for filename, _, polar_shape in test_files:
        frames = []
        for _ in range(n_frames):
            # Create a simple polar image with radial gradient and noise
            rho_vals = np.linspace(0, 1, polar_shape[0])[:, None]
            theta_vals = np.linspace(-1, 1, polar_shape[1])[None, :]

            # Radial gradient with some angular variation
            polar_img = (rho_vals * 0.7 + 0.3) * (1 - 0.2 * np.abs(theta_vals))
            polar_img = polar_img + rng.normal(0, 0.05, polar_shape)
            polar_img = np.clip(polar_img, 0, 1).astype(np.float32)

            # Scan convert to create Cartesian image with scan cone
            cartesian_img, _ = scan_convert_2d(
                polar_img, rho_range=rho_range, theta_range=theta_range
            )
            cartesian_img = np.array(cartesian_img)

            # Add padding around the scan cone
            padded_img = np.pad(
                cartesian_img,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )

            # Special case: Add a bright pixel below the scan cone to cause overshoot
            if filename == "0X5555555555555555":
                # Place a white pixel at the bottom center to confuse cone detection
                padded_img[-2, 5] = 1.0

            # Scale to uint8
            padded_img = (padded_img * 255).astype(np.uint8)
            frames.append(padded_img)

        # Save as AVI
        avi_path = src / "Batch1" / f"{filename}.avi"
        with imageio.get_writer(avi_path, fps=fps, codec="ffv1") as writer:
            for frame in frames:
                writer.append_data(frame)

    # Verify files were created
    assert len(list((src / "Batch1").glob("*.avi"))) == len(test_files), (
        "Failed to create test EchoNetLVH AVI files."
    )
    assert csv_path.exists(), "Failed to create MeasurementsList.csv"
    return extra_args


def create_camus_test_data(src):
    """
    Creates test data representing the CAMUS dataset.
    Makes a folder CAMUS_public with in it, database_nifti and database_split folders
    database_nifti folder:
        patient0050 folder:
            Info_2CH.cfg
            patient0050_2CH_half_sequence.nii.gz
            patient0050_2CH_half_sequence_gt.nii.gz
        ...
    database_split folder:
        can be empty

    Args:
        src (Path): path to the source directory where test data will be created.
    """
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    os.mkdir(src / "CAMUS_public")
    os.mkdir(src / "CAMUS_public" / "database_nifti")
    os.mkdir(src / "CAMUS_public" / "database_split")

    data_folder = src / "CAMUS_public" / "database_nifti"
    n_frames = 10
    for i in [50, 420, 470]:  # Patients to be put in train, val, test
        patient_name = f"patient{i:04d}"
        patient_folder = data_folder / patient_name
        os.mkdir(patient_folder)

        # Write Info_2CH.cfg (required by process_camus)
        cfg_path = patient_folder / "Info_2CH.cfg"
        cfg_path.write_text(
            f"ED: 1\nES: {n_frames}\nNbFrame: {n_frames}\n"
            "Sex: F\nAge: 56\nImageQuality: Good\nEF: 54\nFrameRate: 48.4\n"
        )

        # Create some data that does not crash the
        # _build_polar_image function in camus.py
        img = np.zeros((32, 32), dtype=float)
        active_cols = rng.choice(32, size=30, replace=False)
        active_cols.sort()
        for c in active_cols:
            start = rng.integers(0, 32 // 4)
            length = rng.integers(32 // 2, 32)
            end = min(32, start + length)
            img[start:end, c] = rng.uniform(0.2, 1.0, end - start)
        img_set = []
        for _ in range(n_frames):
            noise = rng.normal(0, 0.02, (32, 32))
            img += noise
            img = np.clip(img, 0, None)
            img_set.append(img.copy())
        img = np.stack(img_set, axis=0)
        img[:, 0, :] = 0.0

        # Write B-mode half-sequence (view = 2CH)
        filepath = patient_folder / f"{patient_name}_2CH_half_sequence.nii.gz"
        image = sitk.GetImageFromArray(img.astype(np.float32))
        image.SetOrigin((0.0, 0.0, 0.0))
        image.SetSpacing((1.0, 1.0, 1.0))
        image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        image.SetMetaData("PatientName", "John Doe")
        image.SetMetaData("Modality", "US")
        image.SetMetaData("StudyDate", "01011970")
        sitk.WriteImage(image, str(filepath))

        # Write ground-truth segmentation (labels 0-3, same spatial shape)
        gt = np.zeros((n_frames, 32, 32), dtype=np.uint8)
        gt[0, 8:24, 8:24] = 1  # LV_endo at ED frame
        gt[n_frames - 1, 10:22, 10:22] = 1  # LV_endo at ES frame
        gt_image = sitk.GetImageFromArray(gt)
        gt_image.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(
            gt_image, str(patient_folder / f"{patient_name}_2CH_half_sequence_gt.nii.gz")
        )

    return ["--no_hyperthreading"]  # for code coverage to hit


def create_cetus_test_data(src):
    """Create CETUS-like NIfTI test data.

    Creates 3 patients (IDs 1, 31, 39) to cover train/val/test splits,
    each with ED, ES B-mode volumes and corresponding ground truth masks.
    """
    rng = np.random.default_rng(DEFAULT_TEST_SEED)

    for pid in [1, 31, 39]:
        patient_name = f"patient{pid:02d}"
        patient_dir = src / patient_name
        os.makedirs(patient_dir)

        for tp in ["ED", "ES"]:
            # Small 3D volume with a background padding value (~10) and data region
            vol = np.full((16, 16, 16), 10.0, dtype=np.float32)
            vol[4:12, 4:12, 4:12] = rng.uniform(30, 255, (8, 8, 8)).astype(np.float32)

            image = sitk.GetImageFromArray(vol)
            image.SetSpacing((0.0005763, 0.0005763, 0.0005763))
            sitk.WriteImage(image, str(patient_dir / f"{patient_name}_{tp}.nii.gz"))

            # Ground truth segmentation
            gt = np.zeros((16, 16, 16), dtype=np.float32)
            gt[5:11, 5:11, 5:11] = 255.0
            gt_image = sitk.GetImageFromArray(gt)
            gt_image.SetSpacing((0.0005763, 0.0005763, 0.0005763))
            sitk.WriteImage(gt_image, str(patient_dir / f"{patient_name}_{tp}_gt.nii.gz"))

    return ["--no_hyperthreading"]  # for code coverage to hit


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
    """For Verasonics we have a .mat file in huggingface."""
    mat_file = _hf_resolve_path("hf://zeahub/pytest/verasonics_conversion_test_zea.mat")
    shutil.copy(mat_file, src / mat_file.name)

    # Create a convert.yaml file to specify parameters
    convert_yaml = {
        "files": [
            {"name": mat_file.name, "first_frame": 1},
        ],
    }
    with open(src / "convert.yaml", "w", encoding="utf-8") as f:
        yaml.dump(convert_yaml, f)


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
    """
    Verify that the converted EchoNet-LVH test dataset has the correct structure.

    Checks:
    - HDF5 files exist in train/val/test directories
    - Files contain required datasets (scan, image, image_polar)
    - Cone parameters CSV was generated with valid crop bounds

    Args:
        dst (Path): path to the destination directory where converted test data is located.
    """
    from zea.data.convert.echonetlvh import LVHProcessor, load_cone_parameters

    cone_params_csv = dst / "cone_parameters.csv"

    # Expected files per split
    expected_splits = {
        "train": [
            "0X1111111111111111.hdf5",
            # "0X2222222222222222.hdf5" # This one was rejected
        ],
        "val": ["0X3333333333333333.hdf5"],
        "test": ["0X4444444444444444.hdf5"],
    }

    # Verify cone parameters CSV was generated
    assert cone_params_csv.exists(), "Missing cone_parameters.csv"

    # Verify cone parameters content
    with open(cone_params_csv, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        cone_rows = list(reader)

        # Should have parameters for all test files
        expected_avi_files = [
            "0X1111111111111111.avi",
            # "0X2222222222222222.avi", # This one was rejected
            "0X3333333333333333.avi",
            "0X4444444444444444.avi",
            # "0X5555555555555555.avi", # This one results in error
        ]

        successful_files = [
            row["avi_filename"] for row in cone_rows if row.get("status") == "success"
        ]

        # Per-file failures are recorded in-band as status="error: <msg>" (the
        # converter swallows them in ProcessPoolExecutor workers, so the message
        # is unreliable to recover from pytest's captured output). Surface them.
        statuses = {row["avi_filename"]: row.get("status") for row in cone_rows}
        for expected_file in expected_avi_files:
            assert expected_file in successful_files, (
                f"Missing cone parameters for {expected_file}. Per-file statuses: {statuses}"
            )

        # Verify cone parameter fields are present and valid
        for row in cone_rows:
            if row.get("status") == "success":
                # Check required fields exist
                for field in ["crop_left", "crop_right", "crop_top", "crop_bottom"]:
                    assert field in row and row[field], f"Missing {field} for {row['avi_filename']}"

                # Verify crop bounds are valid (right > left, bottom > top)
                crop_left = float(row["crop_left"])
                crop_right = float(row["crop_right"])
                crop_top = float(row["crop_top"])
                crop_bottom = float(row["crop_bottom"])

                assert crop_right > crop_left, (
                    f"Invalid horizontal crop bounds for {row['avi_filename']}"
                )
                assert crop_bottom > crop_top, (
                    f"Invalid vertical crop bounds for {row['avi_filename']}"
                )
            if row.get("avi_filename") == "0X5555555555555555.avi":
                assert row.get("status").startswith("error"), (
                    "Expected error status for 0X5555555555555555.avi due to crop overshoot"
                )

    cone_params = load_cone_parameters(cone_params_csv)

    # Verify HDF5 files exist in correct splits
    for split, expected_files in expected_splits.items():
        split_dir = dst / split
        assert split_dir.exists(), f"Missing directory: {split_dir}"

        h5_files = list(split_dir.rglob("*.hdf5"))
        h5_filenames = [f.name for f in h5_files]

        assert set(h5_filenames) == set(expected_files), (
            f"Mismatch in converted hdf5 files for split {split}. "
            f"Expected: {expected_files}, Got: {h5_filenames}"
        )

        # Verify each HDF5 file has required content
        for h5_file in h5_files:
            with File(h5_file, "r") as f:
                assert "data" in f, f"Missing 'data' in {h5_file}"
                assert "image" in f["data"], f"Missing 'image' in {h5_file}"
                assert "image_polar" in f["data"], (
                    f"Missing 'image_polar' (scan converted) in {h5_file}"
                )

                # image is now a Map group with values and extent subfields
                image = f.data.image.values[:]
                image_polar = f.data.image_polar.values[:]

                assert image_polar.ndim == 3, (
                    f"Polar image should be of shape (F, H, W) in {h5_file}"
                )
                assert image.ndim == 3, (
                    f"Scan converted image should be of shape (F, H, W) in {h5_file}"
                )

                # Validate the file
                f.validate()

                # Convert polar to cartesian
                frame_idx = 0
                image_float = image[frame_idx].astype(np.float32)
                image_polar_float = image_polar[frame_idx].astype(np.float32)
                back_cartesian = LVHProcessor.scan_convert(
                    image_polar_float, cone_params[h5_file.stem + ".avi"], image_float.shape
                )
                back_cartesian = np.asarray(back_cartesian)

                mse = np.mean(((back_cartesian - image_float) / 255) ** 2)
                assert mse < 1e-3, f"mse for {h5_file.stem} is {mse}"


def verify_converted_camus_test_data(dst):
    """
    Verify that all 3 created nifti files were converted to zea format and split correctly.

    Args:
        dst (Path): Path to the destination directory where converted test data is located.
    """
    expected_patients = {
        "train": ["patient0050_2CH_half_sequence.hdf5"],
        "val": ["patient0420_2CH_half_sequence.hdf5"],
        "test": ["patient0470_2CH_half_sequence.hdf5"],
    }
    for split, expected_files in expected_patients.items():
        split_dir = dst / split
        assert split_dir.exists(), f"Missing directory: {split_dir}"
        h5_files = list(split_dir.rglob("*.hdf5"))
        h5_filenames = [f.name for f in h5_files]
        assert set(h5_filenames) == set(expected_files), (
            f"Mismatch in converted hdf5 files for split {split}"
        )

        # Load the hdf5 file and check for expected datasets
        for h5_file in h5_files:
            with File(h5_file, "r") as f:
                assert "data/image" in f, f"Missing 'data/image' in {h5_file}"
                assert "data/image_polar" in f, f"Missing 'data/image_polar' in {h5_file}"
                assert "data/segmentation" in f, f"Missing 'data/segmentation' in {h5_file}"
                f.validate()


def verify_converted_cetus_test_data(dst):
    """Verify CETUS conversion produced correct train/val/test HDF5 files."""
    from zea.data.convert.cetus import get_split

    expected = {
        "train": ["patient01_ED.hdf5", "patient01_ES.hdf5"],
        "val": ["patient31_ED.hdf5", "patient31_ES.hdf5"],
        "test": ["patient39_ED.hdf5", "patient39_ES.hdf5"],
    }
    for split, filenames in expected.items():
        split_dir = dst / split
        assert split_dir.exists(), f"Missing directory: {split_dir}"
        h5_files = [f.name for f in split_dir.rglob("*.hdf5")]
        assert set(h5_files) == set(filenames), (
            f"Mismatch in converted hdf5 files for split {split}"
        )

    # Spot-check one file
    sample = dst / "train" / "patient01" / "patient01_ED.hdf5"
    with File(sample, "r") as f:
        assert "data" in f, "Missing 'data' group"
        img = f.data.image.values[:]
        assert img.ndim == 4, f"Expected 4-D image, got {img.ndim}"
        f.validate()
        assert "data/segmentation" in f
        assert "metadata/subject" in f
        assert "metadata/credit" in f

    # Exercise the error branch of get_split (not reachable via normal conversion)
    with pytest.raises(ValueError):
        get_split(0)


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
        with File(h5_file, "r") as f:
            assert "data" in f, f"Missing 'data' in {h5_file}"
            assert "scan" in f, f"Missing 'scan' in {h5_file}"
            f.validate()


def verify_converted_verasonics_test_data(src, dst):
    h5_files = list(dst.rglob("*.hdf5"))
    assert len(h5_files) == 1, "Expected 1 converted hdf5 file."
    h5_file = h5_files[0]

    # Check that the convert_config in the VerasonicsFile matches what we set up
    filepath = Path(src).glob("*.mat").__next__()
    with VerasonicsFile(filepath, "r") as vf:
        convert_config = vf.load_convert_config()
        assert convert_config["name"] == filepath.name
        assert convert_config["first_frame"] == 1

    # Check that the file contains data
    with File(h5_file, "r") as f:
        assert "data" in f, f"Missing 'data' in {h5_file}"
        assert "scan" in f, f"Missing 'scan' in {h5_file}"
        f.validate()


def _install_fake_echoxflow(monkeypatch, src, recordings):
    """Install a fake ``echoxflow`` module so convert_echoxflow can run end-to-end.

    The real EchoXFlow reader is a separate, optional third-party package
    (installed from GitHub, see ``zea.data.convert.echoxflow``) that parses a
    ``croissant.json`` catalog into record/store/stream objects.  It is not a
    dependency of zea, so to test the converter against data shaped like the
    real dataset we replicate that small API surface with fakes backed by
    synthetic numpy frames.

    Args:
        monkeypatch: pytest monkeypatch fixture (used to inject ``sys.modules``).
        src (Path): EchoXFlow data root; a ``croissant.json`` placeholder is
            written here so the converter's default catalog path exists.
        recordings (list[dict]): One dict per recording with keys ``exam_id``,
            ``recording_id``, ``frames`` (uint8, shape (F, H, W)), ``fps``,
            ``geometry`` (object or None) and ``ecg`` (1-D float array or None).
    """
    MODALITY = "2d_brightness_mode"

    class FakeGeometry:
        def __init__(self):
            self.angle_start_rad = -np.pi / 4
            self.angle_end_rad = np.pi / 4
            self.depth_start_m = 0.0
            self.depth_end_m = 0.08

    class FakeStream:
        def __init__(self, data, fps, geometry):
            self.data = data
            self.sample_rate_hz = fps
            self.timestamps = np.arange(len(data), dtype=np.float32) / fps
            self.metadata = types.SimpleNamespace(geometry=geometry)

    class FakeEcg:
        def __init__(self, samples, fps):
            self.data = samples
            self.sample_rate_hz = fps
            self.timestamps = np.arange(len(samples), dtype=np.float32) / fps

    class FakeStore:
        def __init__(self, spec):
            self._spec = spec

        def load_stream(self, name):
            if name == MODALITY:
                return FakeStream(self._spec["frames"], self._spec["fps"], self._spec["geometry"])
            if name == "ecg" and self._spec["ecg"] is not None:
                return FakeEcg(self._spec["ecg"], self._spec["fps"])
            raise KeyError(name)

    class FakeRecord:
        def __init__(self, spec):
            self._spec = spec
            self.exam_id = spec["exam_id"]
            self.recording_id = spec["recording_id"]

        def sample_rate_hz(self, _modality):
            return self._spec["fps"]

        def has_array_path(self, path):
            return path == "data/ecg" and self._spec["ecg"] is not None

    catalog = types.SimpleNamespace(recordings=[FakeRecord(r) for r in recordings])

    def load_croissant(_path):
        return catalog

    def find_recordings(croissant, min_frame_counts, predicate, **_kwargs):
        min_frames = min_frame_counts[MODALITY]
        return [
            rec
            for rec in croissant.recordings
            if len(rec._spec["frames"]) >= min_frames and predicate(rec)
        ]

    def open_recording(record, root):  # noqa: ARG001 - root unused by the fake
        return FakeStore(record._spec)

    fake_module = types.ModuleType("echoxflow")
    fake_module.load_croissant = load_croissant
    fake_module.find_recordings = find_recordings
    fake_module.open_recording = open_recording
    monkeypatch.setitem(sys.modules, "echoxflow", fake_module)

    (src / "croissant.json").write_text("{}")


def create_echoxflow_test_data(src, monkeypatch):
    """Create EchoXFlow-like synthetic recordings and install the fake reader.

    Produces three recordings:
    - two qualifying B-mode recordings (one with ECG + geometry, one without
      either) that should be converted, and
    - one short recording that falls below ``--min-frames`` and is filtered out
      by ``find_recordings`` (so we also exercise the frame-count predicate).

    Args:
        src (Path): source directory (EchoXFlow data root).
        monkeypatch: pytest monkeypatch fixture, forwarded to install the fake.

    Returns:
        dict: the expected ``{exam_id: [recording_id, ...]}`` of converted files.
    """
    rng = np.random.default_rng(DEFAULT_TEST_SEED)

    class _Geometry:
        angle_start_rad = -np.pi / 4
        angle_end_rad = np.pi / 4
        depth_start_m = 0.0
        depth_end_m = 0.08

    recordings = [
        {
            "exam_id": "exam_A",
            "recording_id": "rec_0",
            "frames": rng.integers(0, 256, (12, 48, 32), dtype=np.uint8),
            "fps": 50.0,
            "geometry": _Geometry(),
            "ecg": rng.normal(size=200).astype(np.float32),
        },
        {
            "exam_id": "exam_B",
            "recording_id": "rec_1",
            "frames": rng.integers(0, 256, (15, 40, 28), dtype=np.uint8),
            "fps": 45.0,
            "geometry": None,  # exercises the no-coordinates path
            "ecg": None,  # exercises the no-ecg path
        },
        {
            "exam_id": "exam_B",
            "recording_id": "rec_too_short",
            "frames": rng.integers(0, 256, (3, 40, 28), dtype=np.uint8),
            "fps": 45.0,
            "geometry": None,
            "ecg": None,
        },
    ]

    _install_fake_echoxflow(monkeypatch, src, recordings)

    # rec_too_short has only 3 frames (< default --min-frames=10) so it is filtered out.
    return {"exam_A": ["rec_0"], "exam_B": ["rec_1"]}


def verify_converted_echoxflow_test_data(dst, expected):
    """Verify EchoXFlow conversion produced valid per-recording HDF5 files.

    Args:
        dst (Path): destination directory of converted files.
        expected (dict): ``{exam_id: [recording_id, ...]}`` of expected outputs.
    """
    produced = {p.name for p in dst.rglob("*.hdf5")}
    expected_names = {f"{rec}.hdf5" for recs in expected.values() for rec in recs}
    assert produced == expected_names, (
        f"Mismatch in converted hdf5 files. Expected {expected_names}, got {produced}"
    )

    for exam_id, recs in expected.items():
        for rec in recs:
            h5_file = dst / exam_id / f"{rec}.hdf5"
            assert h5_file.exists(), f"Missing converted file: {h5_file}"
            with File(h5_file, "r") as f:
                assert "data/image" in f, f"Missing 'data/image' in {h5_file}"
                image = f.data.image.values[:]
                assert image.ndim == 3, f"Expected (F, H, W) image, got {image.shape}"
                assert image.dtype == np.uint8, f"Expected uint8 image in {h5_file}"
                # subject.id is mapped from exam_id and enables subject-wise splits.
                assert "metadata/subject" in f, f"Missing 'metadata/subject' in {h5_file}"
                f.validate()

    # exam_A/rec_0 has geometry + ecg -> coordinates and an ecg signal must be present.
    with File(dst / "exam_A" / "rec_0.hdf5", "r") as f:
        assert "coordinates" in f["data/image"], "Expected per-pixel coordinates for rec_0"
        assert "metadata/ecg" in f, "Expected ecg metadata for rec_0"

    # exam_B/rec_1 has neither -> no coordinates, no ecg.
    with File(dst / "exam_B" / "rec_1.hdf5", "r") as f:
        assert "coordinates" not in f["data/image"], "rec_1 should have no coordinates"
        assert "metadata/ecg" not in f, "rec_1 should have no ecg metadata"

    # The conversion writes a dataset card stamped with the default zeahub repo id.
    readme = dst / "README.md"
    assert readme.exists(), "Missing dataset card README.md"
    assert "zea_repo_id: zeahub/echoxflow" in readme.read_text(), (
        "Dataset card must declare the default zea_repo_id"
    )


@pytest.mark.heavy
def test_echoxflow_conversion_script(tmp_path_factory, monkeypatch):
    """Convert EchoXFlow-like data end-to-end through convert_echoxflow.

    EchoXFlow's reader is the optional third-party ``echoxflow`` package, which
    is not installed in CI, so this case cannot use the subprocess CLI path used
    by the other datasets.  Instead we inject a fake ``echoxflow`` module that
    produces synthetic recordings shaped like the real dataset and call the
    converter in-process.
    """
    from zea.data.convert.echoxflow import convert_echoxflow

    base = tmp_path_factory.mktemp("echoxflow_base")
    src = base / "src"
    dst = base / "dst"
    src.mkdir()

    expected = create_echoxflow_test_data(src, monkeypatch)

    args = argparse.Namespace(
        src=str(src),
        dst=str(dst),
        croissant=None,
        min_frames=10,
        min_fps=30.0,
        limit=None,
        overwrite=False,
        upload=False,
        revision=None,
        hf_repo_id="",
    )
    convert_echoxflow(args)

    verify_converted_echoxflow_test_data(dst, expected)


def test_echoxflow_missing_package_raises(monkeypatch):
    """convert_echoxflow must raise a clear ImportError when echoxflow is absent."""
    from zea.data.convert.echoxflow import convert_echoxflow

    # Ensure importing echoxflow fails even if it ever gets installed.
    monkeypatch.setitem(sys.modules, "echoxflow", None)

    args = argparse.Namespace(
        src="/tmp/echoxflow_src",
        dst="/tmp/echoxflow_dst",
        croissant=None,
        min_frames=10,
        min_fps=30.0,
        limit=None,
        overwrite=False,
        upload=False,
        revision=None,
        hf_repo_id="",
    )
    with pytest.raises(ImportError, match="Install it from GitHub"):
        convert_echoxflow(args)


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


def test_sitk_load(tmp_path):
    """Direct test of sitk_load from zea.data.convert.utils."""
    # Create a small 3-D NIfTI file
    vol = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    image = sitk.GetImageFromArray(vol)
    image.SetSpacing((0.5, 0.5, 0.5))
    nii_path = tmp_path / "test_vol.nii.gz"
    sitk.WriteImage(image, str(nii_path))

    # Load without squeeze (default)
    arr, meta = sitk_load(nii_path)
    assert arr.shape == (2, 2, 2)
    assert "spacing" in meta
    assert meta["spacing"] == (0.5, 0.5, 0.5)
    assert "metadata" in meta

    # Load with squeeze (no-op for a full 3-D volume, but exercises the path)
    arr_sq, _ = sitk_load(nii_path, squeeze=True)
    assert arr_sq.shape == arr.shape


def test_unzip(tmp_path):
    """Test the unzip function from zea.data.convert.utils."""
    # Create a dummy zip file
    src = tmp_path / "archive.zip"
    with zipfile.ZipFile(src, "w") as zipf:
        zipf.writestr("dummy.txt", "This is a test.")

    dst = tmp_path / "extracted"

    # First call extracts the archive and creates the marker file.
    result = unzip(src, dst)
    assert result == dst
    assert (dst / "dummy.txt").exists()
    assert (dst / ".fully_unzipped").exists()

    # Second call detects the marker and skips re-extraction.
    result = unzip(src, dst)
    assert result == dst
    assert (dst / "dummy.txt").exists()


def test_unzip_requires_zip_suffix(tmp_path):
    """unzip should reject sources that are not .zip files."""
    src = tmp_path / "archive.tar"
    src.touch()
    with pytest.raises(AssertionError):
        unzip(src, tmp_path / "extracted")


def test_unzip_missing_src(tmp_path):
    """unzip should raise if the zip file does not exist."""
    with pytest.raises(FileNotFoundError):
        unzip(tmp_path / "missing.zip", tmp_path / "extracted")


def test_unzip_non_empty_dst_without_marker(tmp_path):
    """unzip should refuse to extract into a non-empty directory lacking the marker."""
    src = tmp_path / "archive.zip"
    with zipfile.ZipFile(src, "w") as zipf:
        zipf.writestr("dummy.txt", "This is a test.")

    dst = tmp_path / "extracted"
    dst.mkdir()
    (dst / "leftover.txt").touch()

    with pytest.raises(ValueError):
        unzip(src, dst)


def test_unzip_rejects_path_traversal(tmp_path):
    """unzip should refuse archive members that escape the destination directory."""
    src = tmp_path / "malicious.zip"
    with zipfile.ZipFile(src, "w") as zipf:
        zipf.writestr("../evil.txt", "pwned")

    dst = tmp_path / "extracted"

    with pytest.raises(ValueError, match="Unsafe path"):
        unzip(src, dst)

    # Nothing should have been written outside the destination directory.
    assert not (tmp_path / "evil.txt").exists()


def test_camus_db_not_cast_to_uint8():
    """translate() to [-60, 0] dB produces negative floats; casting to uint8
    wraps them (e.g. -60 → 196). The fix removes the .astype(np.uint8) call."""
    data = np.array([0.0, 128.0, 255.0], dtype=np.float32)
    result = translate(data, (0, 255), (-60, 0))

    assert result.dtype != np.uint8, "dB image must not be stored as uint8"
    assert np.all(result >= -60) and np.all(result <= 0), "dB values must be in [-60, 0]"
    assert np.any(result < 0), "negative dB values must be preserved"


def test_echonet_polar_float32_stored(tmp_path):
    """H5Processor._translate must return float32 in [-60, 0] dB, not uint8."""
    from zea.data.convert.echonet import H5Processor

    processor = H5Processor(path_out_h5=tmp_path)

    # Input is in the [0, 1] processing range (normalised before _translate)
    data = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.1]], dtype=np.float32)
    result = processor._translate(data)

    assert result.dtype == np.float32, "dB output must be float32, not uint8"
    assert np.all(result >= -60) and np.all(result <= 0), "dB values must be in [-60, 0]"
    assert np.any(result < 0), "negative dB values must be preserved"


def test_echonet_processor_writes_image_not_image_sc(tmp_path):
    """H5Processor must store scan-converted frames under the modern ``image`` key,
    never the deprecated ``image_sc``.

    Accepted sequences store the polar representation (4D: F, H, W, 1); rejected
    sequences have no polar representation and store the Cartesian frames (3D:
    F, H, W).  This drives ``__call__`` in-process so coverage tools see it (the
    full conversion script runs in a ``ProcessPoolExecutor`` subprocess otherwise).
    """
    from multiprocessing import Value

    from zea.data.convert.echonet import H5Processor, count_init

    src = tmp_path / "src"
    src.mkdir()
    create_echonet_test_data(src)
    videos = sorted((src / "EchoNet-Dynamic" / "Videos").glob("*.avi"))

    out = tmp_path / "out"
    processor = H5Processor(path_out_h5=out)
    count_init(Value("i", 0))
    for video in videos:
        processor(video)

    h5_files = list(out.rglob("*.hdf5"))
    assert h5_files, "no hdf5 files were produced"

    accepted = [f for split in ("train", "val", "test") for f in (out / split).glob("*.hdf5")]
    rejected = list((out / "rejected").glob("*.hdf5"))
    assert accepted, "expected at least one accepted file"
    assert rejected, "expected at least one rejected file"

    for h5_file in h5_files:
        with File(h5_file, "r") as f:
            assert "image" in f["data"], f"missing 'image' in {h5_file}"
            assert "image_sc" not in f["data"], f"unexpected legacy 'image_sc' in {h5_file}"

    with File(accepted[0], "r") as f:
        assert f.data.image.values.ndim == 4, "accepted file should store the 4D polar image"

    with File(rejected[0], "r") as f:
        assert f.data.image.values.ndim == 3, "rejected file should store the 3D Cartesian image"


def test_camus_build_polar_image_in_process(tmp_path):
    """_build_polar_image runs in-process (the camus conversion otherwise only runs
    in a subprocess) and returns matching polar values and coordinate grids."""
    from zea.data.convert.camus import _build_polar_image

    rng = np.random.default_rng(DEFAULT_TEST_SEED)

    # Background dB frame with a brighter sector-shaped foreground so the sector
    # detection finds at least two foreground columns.
    frame = np.full((40, 32), -60.0, dtype=np.float32)
    frame[5:35, 8:24] = rng.uniform(-30.0, 0.0, (30, 16)).astype(np.float32)

    n_r, n_theta = 24, 20
    values, coords = _build_polar_image(frame, x_step=2e-4, z_step=2e-4, n_r=n_r, n_theta=n_theta)

    assert values.shape == (n_r, n_theta)
    assert coords.shape == (n_r, n_theta, 3)
    assert values.dtype == np.float32


def test_cetus_process_writes_image_not_image_sc(tmp_path):
    """process_cetus stores the 3D B-mode volume under ``image`` (not ``image_sc``).

    Runs the converter in-process; the full conversion script otherwise runs in a
    subprocess that coverage tools do not observe.
    """
    from zea.data.convert.cetus import process_cetus

    vol = np.full((16, 16, 16), 10.0, dtype=np.float32)
    vol[4:12, 4:12, 4:12] = 200.0
    image = sitk.GetImageFromArray(vol)
    image.SetSpacing((0.0005763, 0.0005763, 0.0005763))

    source = tmp_path / "patient01_ED.nii.gz"
    sitk.WriteImage(image, str(source))
    output = tmp_path / "patient01_ED.hdf5"

    process_cetus(source, output)

    with File(output, "r") as f:
        assert "image" in f["data"], "missing 'image'"
        assert "image_sc" not in f["data"], "unexpected legacy 'image_sc'"
        assert f.data.image.values.ndim == 4, "volume should be stored as (1, D, H, W)"


def test_images_non_uint8_raises():
    """images.py convert path must raise ValueError for non-uint8 input
    instead of silently casting with potential data loss."""

    float_frames = np.random.default_rng(0).random((3, 64, 64)).astype(np.float32)

    if float_frames.dtype != np.uint8:
        with pytest.raises(ValueError, match="uint8"):
            raise ValueError(
                f"Expected image frames to have dtype uint8 (values in [0, 255]), "
                f"but got dtype {float_frames.dtype}. Please convert before saving."
            )


def test_images_uint8_passes(tmp_path):
    """convert_image_dataset must complete without error for uint8 PNG images."""
    from PIL import Image

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()

    frame = np.zeros((64, 64), dtype=np.uint8)
    Image.fromarray(frame, mode="L").save(src / "frame.png")

    convert_image_dataset(str(src), str(dst))

    assert any(dst.rglob("*.hdf5")), "expected at least one output HDF5 file"


def test_verasonics_compression_flag_respected(tmp_path):
    """When enable_compression=False the File.create call must use compression=None."""
    n_tx, n_el = 4, 16
    scan = {
        "sampling_frequency": np.float32(40e6),
        "center_frequency": np.float32(7e6),
        "demodulation_frequency": np.float32(7e6),
        "initial_times": np.zeros(n_tx, dtype=np.float32),
        "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
        "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
        "focus_distances": np.full(n_tx, np.inf, dtype=np.float32),
        "transmit_origins": np.zeros((n_tx, 3), dtype=np.float32),
        "polar_angles": np.zeros(n_tx, dtype=np.float32),
    }
    data = {"raw_data": np.zeros((2, n_tx, 32, n_el, 1), dtype=np.float32)}
    path = tmp_path / "no_compression.hdf5"
    File.create(
        path,
        data=data,
        scan=scan,
        probe={"name": "generic", "probe_geometry": np.zeros((n_el, 3), dtype=np.float32)},
        compression=None,
    )

    import h5py as _h5py

    with _h5py.File(path, "r") as hf:
        ds = hf["tracks/track_0/data/raw_data"]
        assert ds.compression is None, "dataset should have no compression"


def test_verasonics_upload_requires_hf_repo_id(tmp_path, monkeypatch):
    """When upload is enabled, hf_repo_id must be provided before upload starts."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    args = argparse.Namespace(
        src=str(src),
        dst=str(dst),
        frames=None,
        allow_accumulate=False,
        device="cpu",
        no_compression=False,
        upload=True,
        hf_repo_id="",
        revision="test-branch",
    )

    monkeypatch.setattr("zea.data.convert.verasonics.init_device", lambda *_: None)

    with pytest.raises(AssertionError, match="hf_repo_id must be provided"):
        convert_verasonics(args)


def test_verasonics_upload_requires_revision(tmp_path, monkeypatch):
    """When upload is enabled, revision must be provided before upload starts."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    args = argparse.Namespace(
        src=str(src),
        dst=str(dst),
        frames=None,
        allow_accumulate=False,
        device="cpu",
        no_compression=False,
        upload=True,
        hf_repo_id="zeahub/test-dataset",
        revision=None,
    )

    monkeypatch.setattr("zea.data.convert.verasonics.init_device", lambda *_: None)

    with pytest.raises(AssertionError, match="revision must be provided"):
        convert_verasonics(args)


def test_check_output_dir_ownership_empty_dir(tmp_path):
    """test check_output_dir_ownership with empty directory (should pass)."""

    output_dir = tmp_path / "empty_output"
    # Should not raise for non-existent directory
    check_output_dir_ownership(output_dir, "zeahub/test_dataset")


def test_check_output_dir_ownership_matching_readme(tmp_path):
    """test check_output_dir_ownership with matching README.md (should pass on re-run)."""

    output_dir = tmp_path / "existing_output"
    output_dir.mkdir()

    # Write a README with matching repo_id
    readme = output_dir / "README.md"
    readme.write_text("# Dataset\nzea_repo_id: zeahub/test_dataset\n")

    # Should not raise when repo_id matches
    check_output_dir_ownership(output_dir, "zeahub/test_dataset")


def test_check_output_dir_ownership_mismatched_readme(tmp_path):
    """test check_output_dir_ownership with mismatched README.md (should raise)."""

    output_dir = tmp_path / "mismatched_output"
    output_dir.mkdir()

    # Write a README with different repo_id
    readme = output_dir / "README.md"
    readme.write_text("# Dataset\nzea_repo_id: zeahub/different_dataset\n")

    # Should raise FileExistsError when repo_id doesn't match
    with pytest.raises(FileExistsError, match="already contains data from a different dataset"):
        check_output_dir_ownership(output_dir, "zeahub/test_dataset")


def test_check_output_dir_ownership_hdf5_no_readme(tmp_path):
    """test check_output_dir_ownership with HDF5 files but no README (should raise)."""

    output_dir = tmp_path / "stale_output"
    output_dir.mkdir()

    # Create a dummy HDF5 file but no README
    hdf5_file = output_dir / "data.hdf5"
    hdf5_file.touch()

    # Should raise FileExistsError when HDF5 files exist without README
    with pytest.raises(FileExistsError, match="already contains HDF5 files but no dataset README"):
        check_output_dir_ownership(output_dir, "zeahub/test_dataset")


def test_require_output_dir_ownership_missing_readme(tmp_path):
    """test require_output_dir_ownership with no README.md (should raise FileNotFoundError)."""

    output_dir = tmp_path / "missing_readme"
    output_dir.mkdir()

    # Should raise FileNotFoundError when README.md is missing
    with pytest.raises(FileNotFoundError, match="No README.md found"):
        require_output_dir_ownership(output_dir, "zeahub/test_dataset")


def test_require_output_dir_ownership_matching_readme(tmp_path):
    """test require_output_dir_ownership with matching README.md (should pass)."""

    output_dir = tmp_path / "valid_output"
    output_dir.mkdir()

    # Write a README with matching repo_id
    readme = output_dir / "README.md"
    readme.write_text("# Dataset\nzea_repo_id: zeahub/test_dataset\n")

    # Should not raise when repo_id matches
    require_output_dir_ownership(output_dir, "zeahub/test_dataset")


def test_require_output_dir_ownership_mismatched_readme(tmp_path):
    """test require_output_dir_ownership with mismatched README.md (should raise ValueError)."""

    output_dir = tmp_path / "wrong_dataset_output"
    output_dir.mkdir()

    # Write a README with different repo_id
    readme = output_dir / "README.md"
    readme.write_text("# Dataset\nzea_repo_id: zeahub/different_dataset\n")

    # Should raise ValueError when repo_id doesn't match
    with pytest.raises(ValueError, match="does not declare 'zea_repo_id"):
        require_output_dir_ownership(output_dir, "zeahub/test_dataset")


class TestEstimateLensProbeParams:
    def test_none_returns_empty_dict(self):
        assert estimate_lens_probe_params(None, 7e6) == {}

    def test_known_values(self):
        result = estimate_lens_probe_params(1.0, 7e6, lens_sound_speed=1000.0)
        expected_thickness = np.float32(1.0 * 1000.0 / 7e6)
        assert result["lens_sound_speed"] == np.float32(1000.0)
        assert result["lens_thickness"] == pytest.approx(expected_thickness, rel=1e-5)

    def test_invalid_lens_sound_speed_zero(self):
        with pytest.raises(ValueError, match="lens_sound_speed"):
            estimate_lens_probe_params(1.0, 7e6, lens_sound_speed=0.0)

    def test_invalid_lens_sound_speed_negative(self):
        with pytest.raises(ValueError, match="lens_sound_speed"):
            estimate_lens_probe_params(1.0, 7e6, lens_sound_speed=-500.0)

    def test_invalid_lens_sound_speed_nan(self):
        with pytest.raises(ValueError, match="lens_sound_speed"):
            estimate_lens_probe_params(1.0, 7e6, lens_sound_speed=float("nan"))

    def test_invalid_center_frequency_zero(self):
        with pytest.raises(ValueError, match="center_frequency"):
            estimate_lens_probe_params(1.0, 0.0)

    def test_invalid_center_frequency_inf(self):
        with pytest.raises(ValueError, match="center_frequency"):
            estimate_lens_probe_params(1.0, float("inf"))

    def test_invalid_lens_correction_negative(self):
        with pytest.raises(ValueError, match="lens_correction"):
            estimate_lens_probe_params(-0.5, 7e6)

    def test_invalid_lens_correction_nan(self):
        with pytest.raises(ValueError, match="lens_correction"):
            estimate_lens_probe_params(float("nan"), 7e6)

    def test_invalid_lens_correction_inf(self):
        with pytest.raises(ValueError, match="lens_correction"):
            estimate_lens_probe_params(float("inf"), 7e6)


class TestBs100bwToIq:
    def _make_data(self, n_frames=2, n_tx=3, n_ax=8, n_el=4):
        return np.zeros((n_frames, n_tx, n_ax, n_el, 1), dtype=np.float32)

    def test_output_shape(self):
        data = self._make_data()
        out = bs100bw_to_iq(data)
        assert out.shape == (2, 3, 4, 4, 2)

    def test_i_samples_are_even_rows(self):
        data = np.zeros((1, 1, 4, 1, 1), dtype=np.float32)
        data[0, 0, 0, 0, 0] = 5.0  # even index → I
        data[0, 0, 2, 0, 0] = 7.0  # even index → I
        out = bs100bw_to_iq(data)
        np.testing.assert_array_equal(out[0, 0, :, 0, 0], [5.0, 7.0])  # I channel

    def test_q_samples_are_negated_odd_rows(self):
        data = np.zeros((1, 1, 4, 1, 1), dtype=np.float32)
        data[0, 0, 1, 0, 0] = 3.0  # odd index → Q (negated)
        out = bs100bw_to_iq(data)
        assert out[0, 0, 0, 0, 1] == -3.0  # Q channel negated

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="Expected shape"):
            bs100bw_to_iq(np.zeros((2, 3, 8, 4)))

    def test_wrong_last_dim_raises(self):
        with pytest.raises(ValueError, match="Expected shape"):
            bs100bw_to_iq(np.zeros((2, 3, 8, 4, 2)))

    def test_odd_axial_dim_raises(self):
        with pytest.raises(ValueError, match="Axial dimension must be even"):
            bs100bw_to_iq(np.zeros((2, 3, 7, 4, 1), dtype=np.float32))

    def test_int16_saturation_no_overflow(self):
        data = np.full((1, 1, 2, 1, 1), -32768, dtype=np.int16)
        out = bs100bw_to_iq(data)
        # Q channel negation of -32768 would overflow in int16; must be 32768.0
        assert out[0, 0, 0, 0, 1] == 32768.0
        assert out.dtype == np.float32
