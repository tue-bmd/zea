"""Test Tensorflow H5 dataloader functions"""

import hashlib
import pickle
from copy import deepcopy

import h5py
import keras
import numpy as np
import pytest
from keras import ops

from zea import log
from zea.backend.tensorflow.dataloader import make_dataloader
from zea.data.augmentations import RandomCircleInclusion
from zea.data.dataloader import MAX_RETRY_ATTEMPTS, H5Generator
from zea.data.datasets import Dataset
from zea.data.file import File
from zea.data.layers import Resizer
from zea.data.utils import json_loads
from zea.tools.hf import HFPath
from .. import DEFAULT_TEST_SEED

CAMUS_DATASET_PATH = HFPath("hf://zeahub/camus-sample")
CAMUS_FILE = CAMUS_DATASET_PATH / "val/patient0401/patient0401_4CH_half_sequence.hdf5"
DUMMY_IMAGE_SHAPE = (28, 28)


@pytest.fixture
def dummy_hdf5(tmp_path):
    """Fixture to create and clean up a dummy hdf5 file."""
    file_path = tmp_path / "dummy_data.hdf5"
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    with h5py.File(file_path, "w") as f:
        data = rng.standard_normal((100, *DUMMY_IMAGE_SHAPE))
        f.create_dataset("data", data=data)
    return file_path


@pytest.fixture
def multi_shape_dataset(tmp_path):
    """Fixture to create and clean up a dummy hdf5 file."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    with h5py.File(tmp_path / "dummy_data_1.hdf5", "w") as f:
        data = rng.standard_normal((1, 28, 28))
        f.create_dataset("data", data=data)
    with h5py.File(tmp_path / "dummy_data_2.hdf5", "w") as f:
        data = rng.standard_normal((1, 32, 32))
        f.create_dataset("data", data=data)
    return tmp_path


@pytest.fixture
def ndim_hdf5_dataset_path(tmp_path):
    """Fixture to create and clean up a dummy hdf5 dataset with
    files having data with n dimensions."""
    n_dims = 5
    n_files = 3
    n_samples = 10
    image_shape = [i + 20 for i in range(1, n_dims + 1)] + [1]

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    for i in range(n_files):
        with h5py.File(tmp_path / f"dummy_data_{i}.hdf5", "w") as f:
            data = rng.standard_normal((n_samples, *image_shape))
            f.create_dataset("data", data=data)
    return tmp_path


@pytest.fixture
def camus_dataset():
    """Fixture to return the path to the CAMUS dataset."""
    return CAMUS_DATASET_PATH


@pytest.fixture
def camus_file():
    """Fixture to return the path to the CAMUS dataset."""
    return CAMUS_FILE


def _get_h5_generator(file_path, key, n_frames, insert_frame_axis, seed=None, validate=True):
    file_paths = [file_path]
    # Create a H5Generator instance
    generator = H5Generator(
        file_paths=file_paths,
        key=key,
        n_frames=n_frames,
        insert_frame_axis=insert_frame_axis,
        seed=seed,
        validate=validate,
    )
    return generator


@pytest.mark.parametrize(
    "file_path, key, n_frames, insert_frame_axis",
    [
        ("dummy_hdf5", "data", 1, True),
        ("dummy_hdf5", "data", 3, True),
        ("dummy_hdf5", "data", 1, False),
        ("dummy_hdf5", "data", 3, False),
        ("camus_file", "data/image_sc", 1, True),
        ("camus_file", "data/image_sc", 3, True),
        ("camus_file", "data/image_sc", 1, False),
        ("camus_file", "data/image_sc", 3, False),
        ("camus_file", "data/image_sc", 15, False),
    ],
)
def test_h5_generator(file_path, key, n_frames, insert_frame_axis, request):
    """Test the H5Generator class"""

    validate = file_path != "dummy_hdf5"
    file_path = request.getfixturevalue(file_path)

    generator = _get_h5_generator(file_path, key, n_frames, insert_frame_axis, validate=validate)

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


def test_h5_generator_shuffle(dummy_hdf5):
    """Test the H5Generator class"""

    generator = _get_h5_generator(
        dummy_hdf5, "data", 10, False, seed=DEFAULT_TEST_SEED, validate=False
    )

    # Test shuffle
    shuffled_items = deepcopy(generator.shuffled_items)
    generator._shuffle()
    assert shuffled_items != generator.shuffled_items, "The generator indices were not shuffled"


@pytest.mark.parametrize(
    "directory, key, n_frames, insert_frame_axis, num_files, total_samples",
    [
        ("camus_dataset", "data/image_sc", 1, True, 6, 101),
        ("fake_directory", "data", 1, True, 3, 9 * 3),
        ("camus_dataset", "data/image_sc", 5, False, 6, 101),
        ("fake_directory", "data", 5, False, 3, 9 * 3),
    ],
)
def test_dataloader(
    tmp_path,
    directory,
    key,
    n_frames,
    insert_frame_axis,
    num_files,
    total_samples,
    request,
):
    """Test the dataloader.
    Uses the tmp_path fixture: https://docs.pytest.org/en/stable/how-to/tmp_path.html"""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    if directory == "fake_directory":
        # create a fake directory with some dummy data
        for i in range(num_files):
            with File(tmp_path / f"dummy_data_{i}.hdf5", "w") as f:
                data = rng.random((total_samples // num_files, 28, 28))
                f.create_dataset(key, data=data)
        directory = tmp_path
        image_range = (0, 1)
    elif directory == "camus_dataset":
        directory = request.getfixturevalue(directory)
        image_range = (-60, 0)
    else:
        raise ValueError("Invalid directory for testing")

    with Dataset(directory, key=key) as dataset_test:
        file_lengths = [len(file[key]) for file in dataset_test]

    expected_len_dataset = sum(
        [length // n_frames if not insert_frame_axis else length for length in file_lengths]
    )

    dataset = make_dataloader(
        directory,
        batch_size=1,
        key=key,
        n_frames=n_frames,
        insert_frame_axis=insert_frame_axis,
        shuffle=True,
        seed=DEFAULT_TEST_SEED,
        image_range=image_range,
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

    # Test shuffling
    shuffle_key = {}
    for i in range(2):
        shuffle_key[i] = ""
        for batch in iter(dataset):
            key = hashlib.md5(pickle.dumps(batch)).hexdigest()
            shuffle_key[i] += key

    assert shuffle_key[0] != shuffle_key[1], "The dataset was not shuffled"


@pytest.mark.parametrize(
    "directory, key, n_frames, insert_frame_axis, image_size, batch_size",
    [
        ("camus_dataset", "data/image_sc", 1, True, (20, 20), 2),
        ("dummy_hdf5", "data", 1, True, (20, 20), 2),
        ("camus_dataset", "data/image_sc", 5, False, (20, 20), 1),
        ("dummy_hdf5", "data", 5, False, (20, 20), 1),
    ],
)
def test_h5_dataset_return_filename(
    directory,
    key,
    n_frames,
    insert_frame_axis,
    image_size,
    batch_size,
    request,
):
    """Test the dataloader with return_filename=True."""

    validate = directory != "dummy_hdf5"
    directory = request.getfixturevalue(directory)

    dataset = make_dataloader(
        directory,
        key=key,
        image_size=image_size,
        n_frames=n_frames,
        insert_frame_axis=insert_frame_axis,
        shuffle=True,
        seed=DEFAULT_TEST_SEED,
        return_filename=True,
        resize_type="resize",
        batch_size=batch_size,
        validate=validate,
    )

    batch = next(iter(dataset))

    assert len(batch) == 2, "The batch should contain two elements: images and file names"

    _, file_dict = batch

    assert len(file_dict) == batch_size, (
        "The file_dict should contain the same number of elements as the batch size"
    )

    file_dict = file_dict[0]  # get the first file_dict of the batch
    file_dict = json_loads(file_dict.numpy())

    filename = file_dict["filename"]
    assert isinstance(filename, str), "The filename should be a string"
    fullpath = file_dict["fullpath"]
    assert isinstance(fullpath, str), "The fullpath should be a string"
    assert "indices" in file_dict, "The file_dict should contain indices"


@pytest.mark.parametrize(
    "directory, key, image_size, resize_type, batch_size",
    [
        ("camus_dataset", "data/image_sc", (20, 23), "resize", 1),
        ("dummy_hdf5", "data", (20, 23), "resize", 1),
        (
            "camus_dataset",
            "data/image_sc",
            (20, 23),
            "resize",
            1,
        ),
        ("dummy_hdf5", "data", (20, 23), "resize", 1),
        ("dummy_hdf5", "data", (20, 23), "center_crop", 3),
        ("dummy_hdf5", "data", (20, 23), "random_crop", 3),
        ("dummy_hdf5", "data", (20, 23), "random_crop", 1),
        ("dummy_hdf5", "data", (32, 32), "crop_or_pad", 1),
    ],
)
def test_h5_dataset_resize_types(directory, key, image_size, resize_type, batch_size, request):
    """Test the dataloader with different resize types."""

    validate = directory != "dummy_hdf5"
    directory = request.getfixturevalue(directory)

    dataset = make_dataloader(
        directory,
        key=key,
        image_size=image_size,
        n_frames=1,
        shuffle=True,
        batch_size=batch_size,
        seed=DEFAULT_TEST_SEED,
        return_filename=False,
        resize_type=resize_type,
        assert_image_range=False,
        validate=validate,
    )

    images = next(iter(dataset))

    expected_shape = (batch_size, *image_size)
    dataset_shape = images.shape[:-1]

    assert expected_shape == dataset_shape, (
        f"The images should be resized to {expected_shape}, but got {dataset_shape}"
    )


def test_crop_or_pad():
    """Test the resize_type="crop_or_pad" for to behave as expected"""
    resizer = Resizer(np.array(DUMMY_IMAGE_SHAPE) * 2, resize_type="crop_or_pad")
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    inp = rng.standard_normal((1, *DUMMY_IMAGE_SHAPE, 1))
    out = resizer(inp)

    assert ops.sum(keras.layers.CenterCrop(*DUMMY_IMAGE_SHAPE)(out) - inp) == 0.0, (
        "The center crop pad layer did not work as expected, probably a one-off padding issue"
    )


@pytest.mark.parametrize(
    (
        "key, n_frames, insert_frame_axis, additional_axes_iter, "
        "frame_axis, initial_frame_axis, frame_index_stride, "
        "resize_type, image_size, batch_size"
    ),
    [
        (
            "data",
            1,
            True,
            (1, 3),
            0,
            0,
            1,
            "resize",
            (20, 20),
            1,
        ),
        (
            "data",
            3,
            False,
            (2, 3),
            -1,
            0,
            2,
            "center_crop",
            (20, 20),
            2,
        ),
        (
            "data",
            5,
            True,
            (2, 3),
            -1,
            0,
            1,
            "random_crop",
            (20, 20),
            2,
        ),
    ],
)
def test_ndim_hdf5_dataset(
    ndim_hdf5_dataset_path,  # pytest fixture
    key,
    n_frames,
    insert_frame_axis,
    additional_axes_iter,
    frame_axis,
    initial_frame_axis,
    frame_index_stride,
    resize_type,
    image_size,
    batch_size,
):
    """Test the dataloader with an n-dimensional HDF5 dataset."""

    dataset = make_dataloader(
        ndim_hdf5_dataset_path,
        key=key,
        image_size=image_size,
        n_frames=n_frames,
        insert_frame_axis=insert_frame_axis,
        frame_axis=frame_axis,
        initial_frame_axis=initial_frame_axis,
        frame_index_stride=frame_index_stride,
        batch_size=batch_size,
        additional_axes_iter=additional_axes_iter,
        shuffle=True,
        seed=DEFAULT_TEST_SEED,
        return_filename=False,
        resize_type=resize_type,
        resize_axes=(-3, -1),
        validate=False,  # ndim_hdf5_dataset_path is not a zea dataset
    )

    next(iter(dataset))


@pytest.mark.parametrize(
    "mock_error_count, expected_retries, should_succeed",
    [
        (1, 1, True),  # One error, should succeed on retry
        (
            MAX_RETRY_ATTEMPTS - 1,
            MAX_RETRY_ATTEMPTS - 1,
            True,
        ),  # Two errors, should succeed on third try
        (
            MAX_RETRY_ATTEMPTS + 1,
            MAX_RETRY_ATTEMPTS,
            False,
        ),  # Too many errors, should fail after max retries
    ],
)
def test_h5_file_retry_count(
    mock_error_count, expected_retries, should_succeed, dummy_hdf5, monkeypatch
):
    """Test that the H5Generator correctly counts retries when files are temporarily unavailable."""

    generator = _get_h5_generator(dummy_hdf5, "data", 1, True, validate=False)

    # Store the original load method
    original_load_data = File.load_data
    error_count = [0]  # Use list to allow modification in closure

    # Create a mock load function that fails a specified number of times
    def mock_load_data(self, dtype, indices):
        if error_count[0] < mock_error_count:
            error_count[0] += 1
            log.debug(f"Simulating I/O error in File.load_data. Error count: {error_count[0]}")
            raise OSError(f"Simulated file access error (attempt {error_count[0]})")
        # After specified failures, call the original method
        return original_load_data(self, dtype, indices)

    # Apply the monkeypatch to the zea.file.File class method
    monkeypatch.setattr(File, "load_data", mock_load_data)

    if should_succeed:
        # Should succeed after retries
        batch = next(iter(generator))
        batch = ops.convert_to_numpy(batch)
        assert isinstance(batch, np.ndarray), "Failed to get valid data after retries"
    else:
        # Should fail after max retries
        with pytest.raises(ValueError) as exc_info:
            next(iter(generator))
        assert "Failed to complete operation" in str(exc_info.value)

    assert generator.retry_count == expected_retries, (
        f"Expected {expected_retries} retries but got {generator.retry_count}"
    )


@pytest.mark.usefixtures("dummy_hdf5")
def test_random_circle_inclusion_augmentation(dummy_hdf5):
    """Test RandomCircleInclusion augmentation with dataloader."""

    # 2D case: use as dataloader augmentation (must not return centers)
    augmentation = keras.Sequential(
        [
            RandomCircleInclusion(
                radius=5,
                fill_value=1.0,
                circle_axes=(1, 2),
                return_centers=True,
                with_batch_dim=True,
                seed=keras.random.SeedGenerator(DEFAULT_TEST_SEED),
            )
        ]
    )

    dataset = make_dataloader(
        dummy_hdf5,
        batch_size=4,
        key="data",
        image_size=(28, 28),
        resize_type="center_crop",
        n_frames=1,
        shuffle=False,
        seed=DEFAULT_TEST_SEED,
        augmentation=augmentation,
        validate=False,
    )

    images = next(iter(dataset))
    images_np = np.array(images)

    # Output shape should match input shape
    assert images_np.shape == (
        4,
        28,
        28,
        1,
    ), f"Output shape {images_np.shape} does not match expected (4, 28, 28, 1)"

    # Since input is random and augmentation sets a circle to fill_value=1.0,
    # there should be some pixels exactly 1.0
    assert np.any(np.isclose(images_np, 1.0)), (
        "Augmentation did not set any pixels to fill_value=1.0 as expected"
    )


def test_resize_with_different_shapes(multi_shape_dataset):
    """Test the dataloader class with different image shapes in a batch."""

    # Create a dataloader instance with different image shapes
    dataset = make_dataloader(
        multi_shape_dataset,
        key="data",
        image_size=(16, 16),
        resize_type="resize",
        n_frames=1,
        shuffle=False,
        seed=DEFAULT_TEST_SEED,
        validate=False,
        batch_size=2,
    )

    # Get the first batch
    images = next(iter(dataset))
    images_np = np.array(images)

    # Output shape should match input shape
    assert images_np.shape[-3:-1] == (
        16,
        16,
    ), f"Output shape {images_np.shape} does not match expected (16, 16)"
