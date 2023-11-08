"""Check functions for data types and shapes.

- **Author(s)**     : Tristan Stevens
- **Date**          : October 30 2023
"""

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


def _check_raw_data(data):
    assert len(data.shape) == 4, (
        "raw data must be 4D, with expected shape [n_tx, n_el, n_ax, n_ch], "
        f"got {data.shape}"
    )
    assert data.shape[-1] in [1, 2], (
        "raw data must have 1 or 2 channels, for RF or IQ data respectively, "
        f"got {data.shape[-1]} channels"
    )


def _check_aligned_data(data):
    assert len(data.shape) == 4, (
        "aligned data must be 4D, with expected shape [n_tx, n_el, n_ax, n_ch], "
        f"got {data.shape}"
    )
    assert data.shape[-1] in [1, 2], (
        "raw data must have 1 or 2 channels, for RF or IQ data respectively, "
        f"got {data.shape[-1]} channels"
    )


def _check_beamformed_data(data):
    assert len(data.shape) == 3, (
        "beamformed data must be 3D, with expected shape [Ny, Nx, n_ch], "
        f"got {data.shape}"
    )
    assert data.shape[-1] in [1, 2], (
        "raw data must have 1 or 2 channels, for RF or IQ data respectively, "
        f"got {data.shape[-1]} channels"
    )


def _check_envelope_data(data):
    assert (
        len(data.shape) == 2
    ), f"envelope data must be 2D, with expected shape [Ny, Nx], got {data.shape}"


def _check_image(data):
    assert (
        len(data.shape) == 2
    ), f"image data must be 2D, with expected shape [Ny, Nx], got {data.shape}"


def _check_image_sc(data):
    assert (
        len(data.shape) == 2
    ), f"image data must be 2D, with expected shape [Ny, Nx], got {data.shape}"
