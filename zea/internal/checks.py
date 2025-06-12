"""Check functions for data types and shapes."""

import importlib.util

from zea.internal.core import DataTypes, ModTypes
from zea.internal.registry import checks_registry

_DATA_TYPES = [member.value for member in DataTypes]
_MOD_TYPES = [member.value for member in ModTypes]

_BACKENDS = [None, "torch", "tensorflow", "keras", "jax", "numpy"]

_ML_LIB_AVAILABLE = False
for lib in _BACKENDS:
    if importlib.util.find_spec(str(lib)):
        if lib == "torch":
            _ML_LIB_AVAILABLE = True
        if lib == "tensorflow":
            _ML_LIB_AVAILABLE = True

_REQUIRED_SCAN_KEYS = [
    "n_ax",
    "n_el",
    "n_tx",
    "probe_geometry",
    "sampling_frequency",
    "center_frequency",
    "t0_delays",
    "n_frames",
]

_IMAGE_DATA_TYPES = ["image", "image_sc", "envelope_data", "beamformed_data"]

_NON_IMAGE_DATA_TYPES = ["raw_data", "aligned_data"]


def get_check(data_type):
    """Get check function for data type.

    Args:
        data_type (str): data type to get check function for

    Raises:
        ValueError: if data type is not valid

    Returns:
        function: check function for data type
    """
    if data_type not in _DATA_TYPES:
        raise ValueError(f"Data type {data_type} not valid. Must be one of {_DATA_TYPES}")
    return checks_registry[data_type]


@checks_registry("raw_data")
def _check_raw_data(data=None, shape=None, with_batch_dim=None):
    """Check raw data shape.

    If data is provided, shape is derived from data.
    If shape is provided, data is ignored. Only supply one of data or shape.

    Args:
        data (np.ndarray, optional): raw data. Defaults to None.
            either data or shape must be provided.
        shape (tuple, optional): shape of the data. Defaults to None.
            either data or shape must be provided.
        with_batch_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 5 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape
        AssertionError: if data does not have expected number of channels
    """
    assert data is not None or shape is not None, "Either data or shape must be provided."
    if data is not None:
        shape = data.shape

    if with_batch_dim is None:
        with_batch_dim = len(shape) == 5

    if not with_batch_dim:
        assert len(shape) == 4, (
            f"raw data must be 4D, with expected shape [n_tx, n_ax, n_el, n_ch], got {shape}"
        )
    else:
        assert len(shape) == 5, (
            f"raw data must be 5D, with expected shape [n_fr, n_tx, n_ax, n_el, n_ch], got {shape}"
        )
    assert shape[-1] in [1, 2], (
        "raw data must have 1 or 2 channels, for RF or IQ data respectively, "
        f"got {shape[-1]} channels"
    )


@checks_registry("aligned_data")
def _check_aligned_data(data=None, shape=None, with_batch_dim=None):
    """Check aligned data shape.

    If data is provided, shape is derived from data.
    If shape is provided, data is ignored. Only supply one of data or shape.

    Args:
        data (np.ndarray, optional): aligned data. Defaults to None.
            either data or shape must be provided.
        shape (tuple, optional): shape of the data. Defaults to None.
            either data or shape must be provided.
        with_batch_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 5 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape
        AssertionError: if data does not have expected number of channels
    """
    assert data is not None or shape is not None, "Either data or shape must be provided."
    if data is not None:
        shape = data.shape

    if with_batch_dim is None:
        with_batch_dim = len(shape) == 5

    if not with_batch_dim:
        assert len(shape) == 4, (
            f"aligned data must be 4D, with expected shape [n_tx, n_ax, n_el, n_ch], got {shape}"
        )
    else:
        assert len(shape) == 5, (
            "aligned data must be 5D, with expected shape [n_fr, n_tx, n_ax, n_el, n_ch], "
            f"got {shape}"
        )
    assert shape[-1] in [1, 2], (
        "raw data must have 1 or 2 channels, for RF or IQ data respectively, "
        f"got {shape[-1]} channels"
    )


@checks_registry("beamformed_data")
def _check_beamformed_data(data=None, shape=None, with_batch_dim=None):
    """Check beamformed data shape.

    If data is provided, shape is derived from data.
    If shape is provided, data is ignored. Only supply one of data or shape.

    Args:
        data (np.ndarray, optional): beamformed data. Defaults to None.
            either data or shape must be provided.
        shape (tuple, optional): shape of the data. Defaults to None.
            either data or shape must be provided.
        with_batch_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 4 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape
        AssertionError: if data does not have expected number of channels
    """
    assert data is not None or shape is not None, "Either data or shape must be provided."
    if data is not None:
        shape = data.shape

    if with_batch_dim is None:
        with_batch_dim = len(shape) == 4

    if not with_batch_dim:
        assert len(shape) == 3, (
            f"beamformed data must be 3D, with expected shape [Ny, Nx, n_ch], got {shape}"
        )
    else:
        assert len(shape) == 4, (
            f"beamformed data must be 4D, with expected shape [n_fr, Ny, Nx, n_ch], got {shape}"
        )
    assert shape[-1] in [1, 2], (
        "beamformed data must have 1 or 2 channels, for RF or IQ data respectively, "
        f"got {shape[-1]} channels"
    )


@checks_registry("envelope_data")
def _check_envelope_data(data=None, shape=None, with_batch_dim=None):
    """Check envelope data shape.

    If data is provided, shape is derived from data.
    If shape is provided, data is ignored. Only supply one of data or shape.

    Args:
        data (np.ndarray, optional): envelope data. Defaults to None.
            either data or shape must be provided.
        shape (tuple, optional): shape of the data. Defaults to None.
            either data or shape must be provided.
        with_batch_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 4 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape
    """
    assert data is not None or shape is not None, "Either data or shape must be provided."
    if data is not None:
        shape = data.shape

    if with_batch_dim is None:
        with_batch_dim = len(shape) == 3

    if not with_batch_dim:
        assert len(shape) == 2, (
            f"envelope data must be 2D, with expected shape [Ny, Nx], got {shape}"
        )
    else:
        assert len(shape) == 3, (
            f"envelope data must be 3D, with expected shape [n_fr, Ny, Nx], got {shape}"
        )


@checks_registry("image")
def _check_image(data=None, shape=None, with_batch_dim=None):
    """Check image data shape.

    If data is provided, shape is derived from data.
    If shape is provided, data is ignored. Only supply one of data or shape.

    Args:
        data (np.ndarray, optional): image data. Defaults to None.
            either data or shape must be provided.
        shape (tuple, optional): shape of the data. Defaults to None.
            either data or shape must be provided.
        with_batch_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 4 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape.
    """
    assert data is not None or shape is not None, "Either data or shape must be provided."
    if data is not None:
        shape = data.shape

    if with_batch_dim is None:
        with_batch_dim = len(shape) == 3

    if not with_batch_dim:
        assert len(shape) == 2, f"image data must be 2D, with expected shape [Ny, Nx], got {shape}"
    else:
        assert len(shape) == 3, (
            f"image data must be 3D, with expected shape [n_fr, Ny, Nx], got {shape}"
        )


@checks_registry("image_sc")
def _check_image_sc(data=None, shape=None, with_batch_dim=None):
    """Check image data shape.

    If data is provided, shape is derived from data.
    If shape is provided, data is ignored. Only supply one of data or shape.

    Args:
        data (np.ndarray, optional): scan-converted data. Defaults to None.
            either data or shape must be provided.
        shape (tuple, optional): shape of the data. Defaults to None.
            either data or shape must be provided.
        with_batch_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 4 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape.
    """
    assert data is not None or shape is not None, "Either data or shape must be provided."
    if data is not None:
        shape = data.shape

    if with_batch_dim is None:
        with_batch_dim = len(shape) == 3

    if not with_batch_dim:
        assert len(shape) == 2, f"image data must be 2D, with expected shape [Ny, Nx], got {shape}"
    else:
        assert len(shape) == 3, (
            f"image data must be 3D, with expected shape [n_fr, Ny, Nx], got {shape}"
        )


def _assert_keys_and_axes(keys, axes):
    """Quick check to ensure that the keys and axes are lists of the same length,
    and that the keys are strings and the axes are integers."""

    if not isinstance(keys, list):
        keys = [keys]
    if not isinstance(axes, list):
        axes = [axes]
    if len(keys) != len(axes):
        raise ValueError("The number of keys and axes must match.")

    # assert that all keys are strings
    for key in keys:
        assert isinstance(key, str), "All keys must be strings."

    # assert that all axes are integers
    for axis in axes:
        assert isinstance(axis, int), "All axes must be integers."

    return keys, axes
