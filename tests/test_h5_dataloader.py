"""Test Tensorflow H5 Dataloader functions"""

import os
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from usbmd.backend.tensorflow.dataloader import H5Generator, h5_dataset_from_directory

DUMMY_DATASET_PATH = "dummy_data.hdf5"
CAMUS_DATASET_PATH = (
    "Z:/Ultrasound-BMd/data/USBMD_datasets/CAMUS/"
    "train/patient0001/patient0001_2CH_half_sequence.hdf5"
    if os.name == "nt"
    else "/mnt/z/Ultrasound-BMd/data/USBMD_datasets/CAMUS/"
    "train/patient0001/patient0001_2CH_half_sequence.hdf5"
)


@pytest.fixture
def create_dummy_hdf5():
    """Fixture to create and clean up a dummy hdf5 file."""
    with h5py.File(DUMMY_DATASET_PATH, "w") as f:
        data = np.random.rand(100, 28, 28)
        f.create_dataset("data", data=data)
    yield
    Path(DUMMY_DATASET_PATH).unlink()


def _get_h5_generator(filename, dataset_name, n_frames, insert_frame_axis, seed=None):
    if filename == CAMUS_DATASET_PATH:
        if not Path(filename).exists():
            return

    with h5py.File(filename, "r") as f:
        file_shapes = [f[dataset_name].shape]

    file_names = [filename]
    # Create a H5Generator instance
    generator = H5Generator(
        file_names=file_names,
        file_shapes=file_shapes,
        key=dataset_name,
        n_frames=n_frames,
        insert_frame_axis=insert_frame_axis,
        seed=seed,
    )
    return generator


@pytest.mark.parametrize(
    "filename, dataset_name, n_frames, insert_frame_axis",
    [
        (DUMMY_DATASET_PATH, "data", 1, True),
        (DUMMY_DATASET_PATH, "data", 3, True),
        (DUMMY_DATASET_PATH, "data", 1, False),
        (DUMMY_DATASET_PATH, "data", 3, False),
        (CAMUS_DATASET_PATH, "data/image_sc", 1, True),
        (CAMUS_DATASET_PATH, "data/image_sc", 3, True),
        (CAMUS_DATASET_PATH, "data/image_sc", 1, False),
        (CAMUS_DATASET_PATH, "data/image_sc", 3, False),
        (CAMUS_DATASET_PATH, "data/image_sc", 15, False),
    ],
)
def test_h5_generator(
    filename,
    dataset_name,
    n_frames,
    insert_frame_axis,
    create_dummy_hdf5,  # pytest fixture
):  # pylint: disable=unused-argument
    """Test the H5Generator class"""

    generator = _get_h5_generator(filename, dataset_name, n_frames, insert_frame_axis)

    batch_shape = next(generator()).shape
    if insert_frame_axis:
        assert batch_shape[-1] == n_frames, (
            f"Something went wrong as the last dimension of the batch shape {batch_shape[-1]}"
            " is not equal to the number of frames {n_frames}"
        )
    else:
        assert (batch_shape[-1] / n_frames) == (batch_shape[-1] // n_frames), (
            f"Something went wrong as the last dimension of the batch shape {batch_shape[-1]}"
            " is not divisible by the number of frames {n_frames}"
        )


def test_h5_generator_shuffle(
    create_dummy_hdf5,  # pytest fixture
):  # pylint: disable=unused-argument
    """Test the H5Generator class"""

    generator = _get_h5_generator(DUMMY_DATASET_PATH, "data", 10, False, seed=42)

    # Test shuffle
    indices = deepcopy(generator.indices)
    generator._shuffle()
    assert indices != generator.indices, "The generator indices were not shuffled"


@pytest.mark.parametrize(
    "directory, key, n_frames, insert_frame_axis, num_files, total_samples",
    [
        (Path(CAMUS_DATASET_PATH).parent, "data/image_sc", 1, True, 2, 18 + 20),
        ("fake_directory", "data", 1, True, 3, 9 * 3),
        (Path(CAMUS_DATASET_PATH).parent, "data/image_sc", 5, False, 2, 18 + 20),
        ("fake_directory", "data", 5, False, 3, 9 * 3),
    ],
)
def test_h5_dataset_from_directory(
    tmp_path, directory, key, n_frames, insert_frame_axis, num_files, total_samples
):
    """Test the h5_dataset_from_directory function.
    Uses the tmp_path fixture: https://docs.pytest.org/en/stable/how-to/tmp_path.html"""

    if directory == "fake_directory":
        # create a fake directory with some dummy data
        for i in range(num_files):
            with h5py.File(tmp_path / f"dummy_data_{i}.hdf5", "w") as f:
                data = np.random.rand(total_samples // num_files, 28, 28)
                f.create_dataset(key, data=data)
        expected_len_dataset = total_samples // num_files // n_frames * num_files
        directory = tmp_path
    elif directory == Path(CAMUS_DATASET_PATH).parent:
        expected_len_dataset = 18 // n_frames + 20 // n_frames
        if not Path(directory).exists():
            return
    else:
        raise ValueError("Invalid directory for testing")

    dataset = h5_dataset_from_directory(
        directory,
        key,
        n_frames=n_frames,
        insert_frame_axis=insert_frame_axis,
        search_file_tree_kwargs={"parallel": False},
    )
    batch_shape = next(iter(dataset)).shape

    if insert_frame_axis:
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
