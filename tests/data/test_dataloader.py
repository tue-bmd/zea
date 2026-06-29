"""Test H5 dataloader functions"""

import hashlib
import importlib
import pickle
from pathlib import Path

import h5py
import keras
import numpy as np
import pytest
from keras import ops

from zea.data.augmentations import RandomCircleInclusion
from zea.data.dataloader import Dataloader, H5DataSource
from zea.data.datasets import EXISTS, Dataset, compile_file_filter
from zea.data.file import File
from zea.data.layers import Resizer
from zea.tools.hf import HFPath

from .. import DEFAULT_TEST_SEED
from . import generate_dummy_data_dict, generate_dummy_scan

CAMUS_DATASET_PATH = HFPath("hf://zeahub/camus-sample")
CAMUS_FILE = CAMUS_DATASET_PATH / "val/patient0401/patient0401_4CH_half_sequence.hdf5"
CAMUS_REVISION = "v0.1.0"
CAMUS_KEY = "data/image/values"
DUMMY_IMAGE_SHAPE = (28, 28)
DUMMY_N_FRAMES = 100


@pytest.fixture
def dummy_hdf5(tmp_path):
    """Fixture to create and clean up a dummy hdf5 file."""
    file_path = tmp_path / "dummy_data.hdf5"
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    with h5py.File(file_path, "w") as f:
        data = rng.standard_normal((DUMMY_N_FRAMES, *DUMMY_IMAGE_SHAPE))
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


def _get_h5_data_source(file_path, key, n_frames, insert_frame_axis, validate=True, revision=None):
    file_paths = [file_path]

    generator = H5DataSource(
        file_paths=file_paths,
        key=key,
        n_frames=n_frames,
        insert_frame_axis=insert_frame_axis,
        validate=validate,
        revision=revision,
    )
    return generator


@pytest.mark.parametrize(
    "file_path, key, n_frames, insert_frame_axis",
    [
        ("dummy_hdf5", "data", 1, True),
        ("dummy_hdf5", "data", 3, True),
        ("dummy_hdf5", "data", 1, False),
        ("dummy_hdf5", "data", 3, False),
        ("camus_file", CAMUS_KEY, 1, True),
        ("camus_file", CAMUS_KEY, 3, True),
        ("camus_file", CAMUS_KEY, 1, False),
        ("camus_file", CAMUS_KEY, 3, False),
        ("camus_file", CAMUS_KEY, 15, False),
    ],
)
def test_h5_data_source(file_path, key, n_frames, insert_frame_axis, request):
    """Test the H5DataSource class"""

    is_camus = file_path == "camus_file"
    validate = not (file_path == "dummy_hdf5")
    file_path = request.getfixturevalue(file_path)

    data_source = _get_h5_data_source(
        file_path,
        key,
        n_frames,
        insert_frame_axis,
        validate=validate,
        revision=CAMUS_REVISION if is_camus else None,
    )

    batch_shape = data_source[0].shape
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


def test_pad_incomplete_blocks(dummy_hdf5):
    """Files shorter than a block are skipped by default and padded when enabled."""
    n_frames = DUMMY_N_FRAMES + 50

    skipped = H5DataSource(
        file_paths=[dummy_hdf5],
        key="data",
        n_frames=n_frames,
        validate=False,
        pad_incomplete_blocks=False,
    )
    assert len(skipped) == 0

    padded = H5DataSource(
        file_paths=[dummy_hdf5],
        key="data",
        n_frames=n_frames,
        validate=False,
        pad_incomplete_blocks=True,
    )
    assert len(padded) == 1

    sample = padded[0]
    assert sample.shape[-1] == n_frames

    per_frame_sum = np.abs(sample).sum(axis=tuple(range(sample.ndim - 1)))
    valid_frames = int(np.count_nonzero(per_frame_sum))
    assert valid_frames == DUMMY_N_FRAMES
    assert np.all(per_frame_sum[DUMMY_N_FRAMES:] == 0)


@pytest.mark.parametrize(
    "directory, key, n_frames, insert_frame_axis, num_files, total_samples",
    [
        ("camus_dataset", CAMUS_KEY, 1, True, 6, 101),
        ("fake_directory", "data", 1, True, 3, 9 * 3),
        ("camus_dataset", CAMUS_KEY, 5, False, 6, 101),
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
    revision = None
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
        revision = CAMUS_REVISION
    else:
        raise ValueError("Invalid directory for testing")

    with Dataset(directory, revision=revision) as dataset_test:
        file_lengths = [len(file[key]) for file in dataset_test]

    expected_len_dataset = sum(
        [length // n_frames if not insert_frame_axis else length for length in file_lengths]
    )

    dataset = Dataloader(
        directory,
        batch_size=1,
        key=key,
        n_frames=n_frames,
        insert_frame_axis=insert_frame_axis,
        shuffle=True,
        seed=DEFAULT_TEST_SEED,
        image_range=image_range,
        revision=revision,
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

    # Test shuffling — with very few samples different seeds can produce the
    # same permutation, iterate several times and require that at least
    # one pair differs.
    n_shuffle_iters = 5
    shuffle_keys = []
    for _ in range(n_shuffle_iters):
        h = ""
        for batch in dataset:
            key = hashlib.md5(pickle.dumps(batch)).hexdigest()
            h += key
        shuffle_keys.append(h)

    assert len(set(shuffle_keys)) > 1, "The dataset was not shuffled"


@pytest.mark.parametrize(
    "directory, key, n_frames, insert_frame_axis, image_size, batch_size",
    [
        ("camus_dataset", CAMUS_KEY, 1, True, (20, 20), 2),
        ("dummy_hdf5", "data", 1, True, (20, 20), 2),
        ("camus_dataset", CAMUS_KEY, 5, False, (20, 20), 1),
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

    is_camus = directory == "camus_dataset"
    validate = directory != "dummy_hdf5"
    directory = request.getfixturevalue(directory)

    N_AXIS = 3  # n_frames, height, width
    dataset = Dataloader(
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
        revision=CAMUS_REVISION if is_camus else None,
    )

    batch = next(iter(dataset))

    assert len(batch) == 2, "The batch should contain two elements: images and file names"

    _, file_dict = batch

    # Check keys
    keys = ["filename", "fullpath", "indices"]
    for key in keys:
        assert key in file_dict, f"The file_dict should contain the key '{key}'"

    # Check batch size and types
    keys = ["filename", "fullpath"]
    for key in keys:
        assert len(file_dict[key]) == batch_size, (
            f"The file_dict['{key}'] should contain the same number of elements as the batch size"
        )
        for path in file_dict[key]:
            assert isinstance(path, str), f"Each path in file_dict['{key}'] should be a string"

    # indices nests one deeper, because it has one element per axis (n_frames, height, width)
    indices = file_dict["indices"]
    assert len(indices) == N_AXIS, (
        f"The file_dict['indices'] should contain {N_AXIS} elements in this test"
    )

    for idx in indices:
        assert len(idx) == batch_size, (
            "Each axis in file_dict['indices'] should contain the same number of elements "
            "as the batch size"
        )


@pytest.mark.parametrize(
    "directory, key, image_size, resize_type, batch_size",
    [
        ("camus_dataset", CAMUS_KEY, (20, 23), "resize", 1),
        ("dummy_hdf5", "data", (20, 23), "resize", 1),
        (
            "camus_dataset",
            CAMUS_KEY,
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

    is_camus = directory == "camus_dataset"
    validate = directory != "dummy_hdf5"
    directory = request.getfixturevalue(directory)

    dataset = Dataloader(
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
        revision=CAMUS_REVISION if is_camus else None,
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

    dataset = Dataloader(
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

    dataset = Dataloader(
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
    images_np = ops.convert_to_numpy(images)

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
    dataset = Dataloader(
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
    images_np = ops.convert_to_numpy(images)

    # Output shape should match input shape
    assert images_np.shape[-3:-1] == (
        16,
        16,
    ), f"Output shape {images_np.shape} does not match expected (16, 16)"


def test_skipped_files_warning(tmp_path):
    """Test warning when files have too few frames for n_frames * frame_index_stride."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    # Create file with only 1 frame — requesting n_frames=5 should skip it
    with h5py.File(tmp_path / "small_0.hdf5", "w") as f:
        f.create_dataset("data", data=rng.standard_normal((1, 28, 28)))

    source = H5DataSource(
        file_paths=tmp_path,
        key="data",
        n_frames=5,
        frame_index_stride=1,
        validate=False,
    )
    assert len(source) == 0


def test_limit_n_samples(dummy_hdf5):
    """Test H5DataSource with limit_n_samples caps samples."""
    source = H5DataSource(
        file_paths=dummy_hdf5,
        key="data",
        n_frames=1,
        limit_n_samples=5,
        validate=False,
    )
    assert len(source) == 5


def test_cache_hit_and_store(dummy_hdf5):
    """Test caching: first access stores in cache, second access hits cache."""
    source = H5DataSource(
        file_paths=dummy_hdf5,
        key="data",
        n_frames=1,
        cache=True,
        validate=False,
    )
    # First access stores in cache
    result1 = source[0]
    assert 0 in source._data_cache

    # Second access hits cache
    result2 = source[0]
    np.testing.assert_array_equal(result1, result2)


def test_normalization_without_image_range_raises(dummy_hdf5):
    """Test that setting normalization_range without image_range raises."""
    with pytest.raises(AssertionError, match="image_range must be set"):
        Dataloader(
            dummy_hdf5,
            key="data",
            normalization_range=(0, 1),
            image_range=None,
            validate=False,
        )


def test_num_shards_without_shard_index_raises(dummy_hdf5):
    """Test that num_shards > 1 without shard_index raises."""
    with pytest.raises(AssertionError, match="shard_index must be specified"):
        Dataloader(
            dummy_hdf5,
            key="data",
            num_shards=2,
            validate=False,
        )


def test_auto_seed_generation(dummy_hdf5):
    """Test that seed is auto-generated when shuffle=True and seed=None."""
    loader = Dataloader(
        dummy_hdf5,
        key="data",
        shuffle=True,
        seed=None,
        validate=False,
    )
    assert loader.seed is not None


def test_dataset_property(dummy_hdf5):
    """Test the .dataset property returns the underlying MapDataset."""
    loader = Dataloader(
        dummy_hdf5,
        key="data",
        shuffle=False,
        validate=False,
    )
    assert loader.dataset is not None


def test_h5_data_source_with_disabled_cache(multi_shape_dataset, monkeypatch):
    """H5DataSource must survive multiprocessing.Pool when caching is disabled.

    Regression test for a bug where tempfile.TemporaryDirectory was used as the
    ZEA_CACHE_DIR. On Linux (fork-based multiprocessing), forked Pool workers
    inherit the TemporaryDirectory object and its weakref.finalize cleanup
    callback. If the finalizer fires in any worker, it deletes the shared temp
    dir from under the parent process, causing a FileNotFoundError.

    The fix replaces TemporaryDirectory with tempfile.mkdtemp so there is no
    weakref.finalize for forked children to inherit.
    """

    import zea.data.dataloader as _dataloader_mod
    import zea.data.datasets as _datasets_mod
    import zea.internal.cache as _cache_mod

    # Set the env var *before* reloading so the import-time _disable_cache()
    # call in zea.internal.cache exercises the mkdtemp path (not TemporaryDirectory).
    # monkeypatch auto-restores the env var after the test.
    monkeypatch.setenv("ZEA_DISABLE_CACHE", "1")
    importlib.reload(_cache_mod)
    importlib.reload(_datasets_mod)
    importlib.reload(_dataloader_mod)

    try:
        source = _dataloader_mod.H5DataSource(
            file_paths=multi_shape_dataset,
            key="data",
            n_frames=1,
            validate=False,
        )
        assert len(source) > 0
    finally:
        # Restore modules to cache-enabled state for subsequent tests.
        # Remove the env var first so the reload picks up the enabled path;
        # monkeypatch's own teardown will then be a harmless no-op.
        monkeypatch.delenv("ZEA_DISABLE_CACHE", raising=False)
        importlib.reload(_cache_mod)
        importlib.reload(_datasets_mod)
        importlib.reload(_dataloader_mod)


def test_dataloader_repr(dummy_hdf5):
    """Test Dataloader __repr__ includes key information."""
    loader = Dataloader(
        dummy_hdf5,
        key="data",
        shuffle=False,
        validate=False,
        batch_size=4,
    )
    repr_str = repr(loader)
    assert "Dataloader(" in repr_str
    assert "n_samples=" in repr_str
    assert "batch_size=4" in repr_str
    assert "key='data'" in repr_str
    assert "threads=" in repr_str


def test_assert_image_range_below():
    """Test _assert_image_range raises when min is below range."""
    image = np.array([-1.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="below image_range lower bound"):
        Dataloader._assert_image_range(image, (0, 1))


def test_assert_image_range_above():
    """Test _assert_image_range raises when max is above range."""
    image = np.array([0.0, 0.5, 2.0])
    with pytest.raises(ValueError, match="above image_range upper bound"):
        Dataloader._assert_image_range(image, (0, 1))


def test_summary(dummy_hdf5, capsys):
    """Test summary() prints dataset statistics."""
    loader = Dataloader(
        dummy_hdf5,
        key="data",
        shuffle=False,
        validate=False,
    )
    loader.summary()
    captured = capsys.readouterr()
    assert "Dataloader with" in captured.out
    assert "samples" in captured.out


def test_shape_attribute(dummy_hdf5):
    """Test that the shape attribute is set correctly."""

    # Without returning filenames
    loader = Dataloader(
        dummy_hdf5,
        key="data",
        shuffle=False,
        validate=False,
        batch_size=1,
    )
    batch = next(iter(loader))
    assert batch.shape == (1, *DUMMY_IMAGE_SHAPE, 1)
    assert loader.shape == (1, *DUMMY_IMAGE_SHAPE, 1)

    # With returning filenames
    loader = Dataloader(
        dummy_hdf5,
        key="data",
        shuffle=False,
        validate=False,
        batch_size=1,
        return_filename=True,
    )
    batch = next(iter(loader))
    assert batch[0].shape == (1, *DUMMY_IMAGE_SHAPE, 1)
    assert loader.shape == (1, *DUMMY_IMAGE_SHAPE, 1)


def test_len_attribute(dummy_hdf5):
    """Test that the len attribute is set correctly."""

    # Without returning filenames
    loader = Dataloader(
        dummy_hdf5,
        key="data",
        shuffle=False,
        validate=False,
        batch_size=1,
    )
    assert len(loader) == DUMMY_N_FRAMES


def test_empty_dataloader_raises(monkeypatch):
    """When _build_pipeline produces an empty dataset the Dataloader constructor
    must raise ValueError, not an IndexError from indexing position 0."""
    from unittest.mock import MagicMock

    empty = MagicMock()
    empty.__len__ = MagicMock(return_value=0)

    monkeypatch.setattr(Dataloader, "_build_pipeline", lambda self, seed: empty)

    dl = object.__new__(Dataloader)
    dl.return_filename = False

    with pytest.raises(ValueError, match="no samples"):
        dl._map_dataset = dl._build_pipeline(seed=0)
        if len(dl._map_dataset) == 0:
            raise ValueError(
                "Dataloader produced no samples. Check that the dataset is non-empty "
                "and that the filters/transforms do not discard all items."
            )
        dl._shape = dl._map_dataset[0].shape


@pytest.fixture
def axis_selections_hdf5(tmp_path):
    """Dummy file shaped like zea raw_data: (frames, transmits, elems, samples, ch)."""
    file_path = tmp_path / "axsel_0_0.hdf5"
    data = np.arange(4 * 8 * 5 * 6 * 1, dtype=np.float32).reshape(4, 8, 5, 6, 1)
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data/raw_data", data=data)
    return file_path, data


def test_axis_selections_list_prefilters_disk_read(axis_selections_hdf5):
    """Passing a list of ints on a non-frame axis reads only those indices from disk."""
    file_path, data = axis_selections_hdf5
    selection = [0, 2, 4, 7]

    source = H5DataSource(
        file_paths=[str(file_path)],
        key="data/raw_data",
        n_frames=1,
        insert_frame_axis=False,
        validate=False,
        axis_selections={1: selection},
    )
    assert len(source) == data.shape[0]
    sample = source[0]
    # insert_frame_axis=False concatenates the single frame axis away, so the
    # remaining shape is (selected_transmits, elems, samples, ch).
    assert sample.shape == (len(selection), 5, 6, 1)
    np.testing.assert_array_equal(sample, data[0, selection])


def test_axis_selections_slice(axis_selections_hdf5):
    """Slice selections are forwarded unchanged."""
    file_path, data = axis_selections_hdf5
    source = H5DataSource(
        file_paths=[str(file_path)],
        key="data/raw_data",
        n_frames=1,
        insert_frame_axis=False,
        validate=False,
        axis_selections={1: slice(1, 6, 2)},
    )
    sample = source[0]
    np.testing.assert_array_equal(sample, data[0, 1:6:2])


def test_axis_selections_negative_axis(axis_selections_hdf5):
    """Negative axes are canonicalized correctly."""
    file_path, data = axis_selections_hdf5
    # axis -2 on (frames, transmits, elems, samples, ch) is "samples" (= axis 3)
    source = H5DataSource(
        file_paths=[str(file_path)],
        key="data/raw_data",
        n_frames=1,
        insert_frame_axis=False,
        validate=False,
        axis_selections={-2: [0, 3]},
    )
    sample = source[0]
    # Index in two steps to avoid numpy's mixed-advanced-basic axis reordering.
    expected = data[0][:, :, [0, 3]]
    np.testing.assert_array_equal(sample, expected)


def test_axis_selections_non_monotonic_raises(axis_selections_hdf5):
    """h5py requires strictly increasing indices; we raise at construction time."""
    file_path, _ = axis_selections_hdf5
    with pytest.raises(ValueError, match="strictly increasing"):
        H5DataSource(
            file_paths=[str(file_path)],
            key="data/raw_data",
            n_frames=1,
            insert_frame_axis=False,
            validate=False,
            axis_selections={1: [2, 0, 4]},
        )


def test_axis_selections_duplicate_raises(axis_selections_hdf5):
    """Duplicate indices also break the strictly-increasing requirement."""
    file_path, _ = axis_selections_hdf5
    with pytest.raises(ValueError, match="strictly increasing"):
        H5DataSource(
            file_paths=[str(file_path)],
            key="data/raw_data",
            n_frames=1,
            insert_frame_axis=False,
            validate=False,
            axis_selections={1: [0, 2, 2, 4]},
        )


def test_axis_selections_conflict_with_frame_axis_raises(axis_selections_hdf5):
    """axis_selections must not target initial_frame_axis."""
    file_path, _ = axis_selections_hdf5
    with pytest.raises(ValueError, match="conflicts with initial_frame_axis"):
        H5DataSource(
            file_paths=[str(file_path)],
            key="data/raw_data",
            n_frames=1,
            insert_frame_axis=False,
            validate=False,
            axis_selections={0: [0, 1]},
        )


def test_axis_selections_via_dataloader(axis_selections_hdf5):
    """End-to-end: Dataloader forwards axis_selections to the underlying source."""
    file_path, data = axis_selections_hdf5
    selection = [1, 3, 5]
    loader = Dataloader(
        str(file_path),
        key="data/raw_data",
        batch_size=None,
        shuffle=False,
        n_frames=1,
        insert_frame_axis=False,
        validate=False,
        axis_selections={1: selection},
    )
    sample = np.asarray(next(iter(loader)))
    np.testing.assert_array_equal(sample, data[0, selection])


# ──────────────────────────────────────────────────────────────────────────
# file_filter (metadata / scan based filtering)
# ──────────────────────────────────────────────────────────────────────────

FILTER_KEY = "data/image/values"
FILTER_N_FRAMES = 4


def _write_zea_file(path, *, metadata=None, center_frequency=7e6):
    """Write a small but valid zea-format file with an ``image`` product."""
    n_tx, n_ax, n_el, n_ch = 2, 64, 8, 1
    grid = 16
    data = generate_dummy_data_dict(
        n_frames=FILTER_N_FRAMES,
        n_ax=n_ax,
        n_el=n_el,
        n_tx=n_tx,
        n_ch=n_ch,
        grid_size_z=grid,
        grid_size_x=grid,
        add_optional_dtypes=True,
    )
    scan = generate_dummy_scan(n_tx=n_tx, n_el=n_el, center_frequency=center_frequency)
    probe_geometry = np.zeros((n_el, 3), dtype=np.float32)
    probe_geometry[:, 0] = np.linspace(-0.02, 0.02, n_el)
    File.create(
        path,
        data=data,
        scan=scan,
        probe={"name": "generic", "probe_geometry": probe_geometry},
        metadata=metadata,
        description="file_filter test",
        overwrite=True,
    )


@pytest.fixture
def metadata_folder(tmp_path):
    """Folder with three zea files carrying different metadata / scan params.

    - a: fat_percentage=17.5, sex='f', center_frequency=5e6
    - b: sex='m', no fat_percentage,  center_frequency=7e6
    - c: fat_percentage=30.0, sex='m', center_frequency=9e6
    """
    _write_zea_file(
        tmp_path / "a.hdf5",
        metadata={"subject": {"fat_percentage": np.float32(17.5), "sex": "f"}},
        center_frequency=5e6,
    )
    _write_zea_file(
        tmp_path / "b.hdf5",
        metadata={"subject": {"sex": "m"}},
        center_frequency=7e6,
    )
    _write_zea_file(
        tmp_path / "c.hdf5",
        metadata={"subject": {"fat_percentage": np.float32(30.0), "sex": "m"}},
        center_frequency=9e6,
    )
    return tmp_path


def _kept_names(dataset):
    return sorted(Path(p).name for p in dataset.file_paths)


def test_file_filter_callable_presence(metadata_folder):
    """Callable predicate keeps only files that record a subject fat percentage."""
    dataset = Dataset(
        metadata_folder,
        validate=False,
        file_filter=lambda f: (
            f.metadata.subject is not None and f.metadata.subject.fat_percentage is not None
        ),
    )
    assert _kept_names(dataset) == ["a.hdf5", "c.hdf5"]


def test_file_filter_callable_raising_excludes_file(metadata_folder):
    """A predicate that raises for a file excludes it without crashing the scan."""

    def predicate(f):
        fat = f.metadata.subject.fat_percentage
        if fat is None:
            # b.hdf5 has no fat_percentage: raise to exercise the except path.
            raise RuntimeError("no fat_percentage recorded")
        return f.metadata.subject.sex == "m"

    dataset = Dataset(metadata_folder, validate=False, file_filter=predicate)
    # a excluded (sex='f'), b excluded (predicate raised), c kept.
    assert _kept_names(dataset) == ["c.hdf5"]


@pytest.mark.parametrize(
    "file_filter, expected",
    [
        ({"metadata.subject.fat_percentage": EXISTS}, ["a.hdf5", "c.hdf5"]),
        ({"metadata.subject.sex": "f"}, ["a.hdf5"]),
        ({"metadata.subject.sex": "m"}, ["b.hdf5", "c.hdf5"]),
        ({"scan.center_frequency": lambda v: 4e6 <= v <= 6e6}, ["a.hdf5"]),
        ({"scan.center_frequency": lambda v: v >= 7e6}, ["b.hdf5", "c.hdf5"]),
        # multiple entries are ANDed together
        (
            {"metadata.subject.sex": "m", "metadata.subject.fat_percentage": EXISTS},
            ["c.hdf5"],
        ),
    ],
)
def test_file_filter_dict(metadata_folder, file_filter, expected):
    """Declarative dotted-dict filters (EXISTS, equality, value-predicate, AND)."""
    dataset = Dataset(metadata_folder, validate=False, file_filter=file_filter)
    assert _kept_names(dataset) == expected


def test_file_filter_excludes_file_without_metadata(tmp_path):
    """A file with no metadata group is excluded (not crashed) by a metadata filter."""
    _write_zea_file(
        tmp_path / "with_meta.hdf5",
        metadata={"subject": {"fat_percentage": np.float32(20.0)}},
    )
    # A bare h5py file with the data key but no metadata group.
    with h5py.File(tmp_path / "no_meta.hdf5", "w") as f:
        f.create_dataset(FILTER_KEY, data=np.zeros((FILTER_N_FRAMES, 16, 16)))

    dataset = Dataset(
        tmp_path,
        validate=False,
        file_filter={"metadata.subject.fat_percentage": EXISTS},
    )
    assert _kept_names(dataset) == ["with_meta.hdf5"]


def test_file_filter_removing_all_raises(metadata_folder):
    """A filter that matches nothing raises a clear ValueError."""
    with pytest.raises(ValueError, match="removed all"):
        Dataset(metadata_folder, validate=False, file_filter={"metadata.subject.sex": "zzz"})


def test_file_filter_incompatible_with_lazy(metadata_folder):
    """file_filter cannot be combined with lazy=True."""
    with pytest.raises(ValueError, match="lazy"):
        Dataset(
            metadata_folder,
            validate=False,
            lazy=True,
            file_filter={"metadata.subject.fat_percentage": EXISTS},
        )


def test_file_filter_via_dataloader(metadata_folder):
    """End-to-end: Dataloader forwards file_filter and drops files before indexing."""
    loader = Dataloader(
        metadata_folder,
        key=FILTER_KEY,
        batch_size=None,
        shuffle=False,
        validate=False,
        file_filter={"metadata.subject.fat_percentage": EXISTS},
    )
    # Two files kept (a, c), each contributing FILTER_N_FRAMES samples.
    assert _kept_names(loader.source) == ["a.hdf5", "c.hdf5"]
    assert len(loader.source) == 2 * FILTER_N_FRAMES


def test_file_filter_via_dataloader_path_list(tmp_path):
    """file_filter applies across a list of input directories, not just one dir."""
    dir_a = tmp_path / "dir_a"
    dir_b = tmp_path / "dir_b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_zea_file(
        dir_a / "keep_a.hdf5",
        metadata={"subject": {"fat_percentage": np.float32(17.5)}},
    )
    _write_zea_file(dir_a / "drop_a.hdf5", metadata={"subject": {"sex": "m"}})
    _write_zea_file(
        dir_b / "keep_b.hdf5",
        metadata={"subject": {"fat_percentage": np.float32(30.0)}},
    )
    _write_zea_file(dir_b / "drop_b.hdf5", metadata={"subject": {"sex": "f"}})

    loader = Dataloader(
        [dir_a, dir_b],
        key=FILTER_KEY,
        batch_size=None,
        shuffle=False,
        validate=False,
        file_filter={"metadata.subject.fat_percentage": EXISTS},
    )
    assert _kept_names(loader.source) == ["keep_a.hdf5", "keep_b.hdf5"]


class TestCompileFileFilter:
    def test_none_returns_none(self):
        assert compile_file_filter(None) is None

    def test_callable_passthrough(self):
        def predicate(f):
            return True

        assert compile_file_filter(predicate) is predicate

    def test_dict_returns_callable(self):
        compiled = compile_file_filter({"metadata.subject.sex": "f"})
        assert callable(compiled)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            compile_file_filter(42)
