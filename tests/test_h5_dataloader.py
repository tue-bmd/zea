"""Test Tensorflow H5 Dataloader functions"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from usbmd.tensorflow_ultrasound.dataloader import (
    H5Generator,
    h5_dataset_from_directory,
)

CAMUS_DATASET_PATH = (
    "Z:/Ultrasound-BMd/data/USBMD_datasets/CAMUS/"
    "database_nifti/patient0001/patient0001_2CH_half_sequence.hdf5"
)


@pytest.mark.parametrize(
    "filename, dataset_name, n_frames, new_frames_dim",
    [
        ("dummy_data.hdf5", "data", 1, True),
        ("dummy_data.hdf5", "data", 3, True),
        ("dummy_data.hdf5", "data", 1, False),
        ("dummy_data.hdf5", "data", 3, False),
        (CAMUS_DATASET_PATH, "data/image_sc", 1, True),
        (CAMUS_DATASET_PATH, "data/image_sc", 3, True),
        (CAMUS_DATASET_PATH, "data/image_sc", 1, False),
        (CAMUS_DATASET_PATH, "data/image_sc", 3, False),
        (CAMUS_DATASET_PATH, "data/image_sc", 15, False),
    ],
)
def test_h5_generator(filename, dataset_name, n_frames, new_frames_dim):
    """Test the H5Generator class"""
    # Create a dummy  hdf5 file with some dummy data
    if filename == "dummy_data.hdf5":
        with h5py.File(filename, "w") as f:
            data = np.random.rand(100, 28, 28)
            f.create_dataset(dataset_name, data=data)
    else:
        if not Path(filename).exists():
            return

    # Create a H5Generator instance
    generator = H5Generator(n_frames=n_frames, new_frames_dim=new_frames_dim)
    generator.length(filename, dataset_name)
    batch_shape = next(generator(filename, dataset_name)).shape
    if new_frames_dim:
        assert batch_shape[-1] == n_frames, (
            f"Something went wrong as the last dimension of the batch shape {batch_shape[-1]}"
            " is not equal to the number of frames {n_frames}"
        )
    else:
        assert (batch_shape[-1] / n_frames) == (batch_shape[-1] // n_frames), (
            f"Something went wrong as the last dimension of the batch shape {batch_shape[-1]}"
            " is not divisible by the number of frames {n_frames}"
        )

    if filename == "dummy_data.hdf5":
        # clean up the dummy file
        Path("dummy_data.hdf5").unlink()


@pytest.mark.parametrize(
    "directory, key, n_frames, new_frames_dim, num_files, total_samples",
    [
        (Path(CAMUS_DATASET_PATH).parent, "data/image_sc", 1, True, 2, 18 + 20),
        ("fake_directory", "data", 1, True, 3, 9 * 3),
        (Path(CAMUS_DATASET_PATH).parent, "data/image_sc", 5, False, 2, 18 + 20),
        ("fake_directory", "data", 5, False, 3, 9 * 3),
    ],
)
def test_h5_dataset_from_directory(
    directory, key, n_frames, new_frames_dim, num_files, total_samples
):
    """Test the h5_dataset_from_directory function"""

    if directory == "fake_directory":
        # create a fake directory with some dummy data
        _remove_fake_directory()
        Path("fake_directory").mkdir()
        for i in range(num_files):
            with h5py.File(f"fake_directory/dummy_data_{i}.hdf5", "w") as f:
                data = np.random.rand(total_samples // num_files, 28, 28)
                f.create_dataset(key, data=data)
        expected_len_dataset = total_samples // num_files // n_frames * num_files
    elif directory == Path(CAMUS_DATASET_PATH).parent:
        expected_len_dataset = 18 // n_frames + 20 // n_frames
    else:
        raise ValueError("Invalid directory for testing")

    dataset = h5_dataset_from_directory(
        directory, key, n_frames=n_frames, new_frames_dim=new_frames_dim
    )
    batch_shape = next(iter(dataset)).shape

    if new_frames_dim:
        assert batch_shape[-1] == n_frames, (
            f"Something went wrong as the last dimension of the batch shape {batch_shape[-1]}"
            " is not equal to the number of frames {n_frames}"
        )
    else:
        assert (batch_shape[-2] / n_frames) == (batch_shape[-2] // n_frames), (
            "Something went wrong as the second to last dimension of "
            f"the batch shape {batch_shape[-2]} "
            f"is not divisible by the number of frames {n_frames}"
        )

    real_len_dataset = len(dataset)

    assert real_len_dataset == expected_len_dataset, (
        f"Something went wrong as the length of the dataset {real_len_dataset}"
        f" is not equal to the expected length {expected_len_dataset}"
    )

    if directory == "fake_directory":
        # clean up the fake directory
        _remove_fake_directory()


def _remove_fake_directory():
    if Path("fake_directory").exists():
        # clean up the fake directory
        files = Path("fake_directory").glob("*")
        for file in files:
            file.unlink()
        Path("fake_directory").rmdir()
