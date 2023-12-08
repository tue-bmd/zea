"""Check functions for data types and shapes.

- **Author(s)**     : Tristan Stevens
- **Date**          : October 30 2023
"""

from usbmd.registry import checks_registry

_DATA_TYPES = [
    "raw_data",
    "aligned_data",
    "beamformed_data",
    "envelope_data",
    "image",
    "image_sc",
]

_ML_LIBRARIES = [None, "torch", "tensorflow"]

_MOD_TYPES = [None, "rf", "iq"]

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

_IMAGE_DATA_TYPES = ["image", "image_sc"]

_NON_IMAGE_DATA_TYPES = ["raw_data", "beamformed_data", "aligned_data", "envelope_data"]


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
        raise ValueError(
            f"Data type {data_type} not valid. Must be one of {_DATA_TYPES}"
        )
    return checks_registry[data_type]


@checks_registry("raw_data")
def _check_raw_data(data, with_frame_dim=False):
    """Check raw data shape.

    Args:
        data (np.ndarray): raw data
        with_frame_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 5 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape
        AssertionError: if data does not have expected number of channels
    """
    if not with_frame_dim:
        assert len(data.shape) == 4, (
            "raw data must be 4D, with expected shape [n_tx, n_ax, n_el, n_ch], "
            f"got {data.shape}"
        )
    else:
        assert len(data.shape) == 5, (
            "raw data must be 5D, with expected shape [n_fr, n_tx, n_ax, n_el, n_ch], "
            f"got {data.shape}"
        )
    assert data.shape[-1] in [1, 2], (
        "raw data must have 1 or 2 channels, for RF or IQ data respectively, "
        f"got {data.shape[-1]} channels"
    )


@checks_registry("aligned_data")
def _check_aligned_data(data, with_frame_dim=False):
    """Check aligned data shape.

    Args:
        data (np.ndarray): aligned data
        with_frame_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 5 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape
        AssertionError: if data does not have expected number of channels
    """

    if not with_frame_dim:
        assert len(data.shape) == 4, (
            "aligned data must be 4D, with expected shape [n_tx, n_ax, n_el, n_ch], "
            f"got {data.shape}"
        )
    else:
        assert len(data.shape) == 5, (
            "aligned data must be 5D, with expected shape [n_fr, n_tx, n_ax, n_el, n_ch], "
            f"got {data.shape}"
        )
    assert data.shape[-1] in [1, 2], (
        "raw data must have 1 or 2 channels, for RF or IQ data respectively, "
        f"got {data.shape[-1]} channels"
    )


@checks_registry("beamformed_data")
def _check_beamformed_data(data, with_frame_dim=False):
    """Check beamformed data shape.

    Args:
        data (np.ndarray): beamformed data
        with_frame_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 4 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape
        AssertionError: if data does not have expected number of channels
    """
    if not with_frame_dim:
        assert len(data.shape) == 3, (
            "beamformed data must be 3D, with expected shape [Ny, Nx, n_ch], "
            f"got {data.shape}"
        )
    else:
        assert len(data.shape) == 4, (
            "beamformed data must be 4D, with expected shape [n_fr, Ny, Nx, n_ch], "
            f"got {data.shape}"
        )
    assert data.shape[-1] in [1, 2], (
        "raw data must have 1 or 2 channels, for RF or IQ data respectively, "
        f"got {data.shape[-1]} channels"
    )


@checks_registry("envelope_data")
def _check_envelope_data(data, with_frame_dim=False):
    """Check envelope data shape.

    Args:
        data (np.ndarray): envelope data
        with_frame_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 4 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape
    """
    if not with_frame_dim:
        assert len(data.shape) == 2, (
            "envelope data must be 2D, with expected shape [Ny, Nx], "
            f"got {data.shape}"
        )
    else:
        assert len(data.shape) == 3, (
            "envelope data must be 3D, with expected shape [n_fr, Ny, Nx], "
            f"got {data.shape}"
        )


@checks_registry("image")
def _check_image(data, with_frame_dim=False):
    """Check image data shape.

    Args:
        data (np.ndarray): image data
        with_frame_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 4 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape.
    """
    if not with_frame_dim:
        assert len(data.shape) == 2, (
            "image data must be 2D, with expected shape [Ny, Nx], " f"got {data.shape}"
        )
    else:
        assert len(data.shape) == 3, (
            "image data must be 3D, with expected shape [n_fr, Ny, Nx], "
            f"got {data.shape}"
        )


@checks_registry("image_sc")
def _check_image_sc(data, with_frame_dim=False):
    """Check image data shape.

    Args:
        data (np.ndarray): image data
        with_frame_dim (bool, optional): whether data has frame dimension at the start.
            Setting this to True requires the data to have 4 dimensions. Defaults to
            False.

    Raises:
        AssertionError: if data does not have expected shape.
    """
    if not with_frame_dim:
        assert len(data.shape) == 2, (
            "image data must be 2D, with expected shape [Ny, Nx], " f"got {data.shape}"
        )
    else:
        assert len(data.shape) == 3, (
            "image data must be 3D, with expected shape [n_fr, Ny, Nx], "
            f"got {data.shape}"
        )
