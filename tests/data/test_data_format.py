"""Test generating and validating zea data format."""

from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from zea.data.file import File, validate_file
from zea.data.file_operations import save_file
from zea.data.spec import ScanSpec

from . import generate_example_dataset

n_frames = 2
n_tx = 4
n_el = 16
n_ax = 128
n_ch = 1

_REQUIRED_SCAN_KEYS = ScanSpec.required_fields()

# Data dict for File.create
DATA = {
    "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
}

# Scan dict for File.create
SCAN = {
    "probe_geometry": np.zeros((n_el, 3), dtype=np.float32),
    "sampling_frequency": np.float32(30e6),
    "center_frequency": np.float32(6e6),
    "demodulation_frequency": np.float32(6e6),
    "initial_times": np.zeros((n_tx), dtype=np.float32),
    "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
    "sound_speed": np.float32(1540.0),
    "focus_distances": np.zeros((n_tx,), dtype=np.float32),
    "polar_angles": np.linspace(-np.pi / 2, np.pi / 2, n_tx, dtype=np.float32),
    "azimuth_angles": np.zeros((n_tx), np.float32),
    "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
    "time_to_next_transmit": np.ones((n_frames, n_tx), dtype=np.float32),
    "transmit_origins": np.zeros((n_tx, 3), dtype=np.float32),
}


@pytest.fixture
def tmp_hdf5_path(tmp_path) -> Generator[Path, None, None]:
    """Fixture to create a temporary HDF5 file."""
    yield Path(tmp_path, "test_case_dataset.hdf5")


@pytest.fixture
def example_dataset_path(tmp_hdf5_path):
    """Fixture to create a temporary dataset for testing."""
    generate_example_dataset(tmp_hdf5_path)
    yield tmp_hdf5_path


def test_example_dataset(example_dataset_path):
    """Tests the generate_example_dataset function by calling it and then
    validating it using the validate_file function.
    """
    # Validate the dataset
    validate_file(example_dataset_path)

    # Check if the dataset can be loaded correctly
    with File(example_dataset_path) as dataset:
        raw_data = dataset.data.raw_data[0]
        assert raw_data is not None, "Dataset not loaded correctly"


def test_create_basic(tmp_hdf5_path):
    """Tests basic File.create with data and scan dicts."""
    f = File.create(
        tmp_hdf5_path,
        data=DATA,
        scan=SCAN,
        probe_name="generic",
        description="Dataset parameters for testing",
        overwrite=True,
    )
    f.close()
    validate_file(tmp_hdf5_path)


@pytest.mark.parametrize(
    "key",
    list(SCAN.keys()),
)
def test_wrong_scan_shape(key, tmp_hdf5_path):
    """Tests if passing a scan parameter with the wrong shape raises an error.

    Args:
        key (str): The key to change in the scan dictionary.
    """
    wrong_scan = SCAN.copy()
    wrong_scan[key] = np.zeros((n_frames, n_tx + 7, n_el + 1), dtype=np.float32)
    with pytest.raises((AssertionError, ValueError, TypeError)):
        f = File.create(
            tmp_hdf5_path,
            data=DATA,
            scan=wrong_scan,
            probe_name="generic",
            description="Dataset parameters for testing",
            overwrite=True,
        )
        f.close()


@pytest.mark.parametrize(
    "key",
    [k for k in SCAN.keys() if k not in _REQUIRED_SCAN_KEYS],
)
def test_omit_optional_scan_key(key, tmp_hdf5_path):
    """Tests that omitting an optional scan key does not raise an error.

    Args:
        key (str): The optional key to omit from the scan dictionary.
    """
    reduced_scan = {k: v for k, v in SCAN.items() if k != key}
    f = File.create(
        tmp_hdf5_path,
        data=DATA,
        scan=reduced_scan,
        overwrite=True,
    )
    f.close()
    validate_file(tmp_hdf5_path)


@pytest.mark.parametrize(
    "key",
    _REQUIRED_SCAN_KEYS,
)
def test_omit_required_scan_key(key, tmp_hdf5_path):
    """Tests that omitting a required scan key raises a TypeError.

    Args:
        key (str): The required key to omit from the scan dictionary.
    """
    reduced_scan = {k: v for k, v in SCAN.items() if k != key}
    with pytest.raises(TypeError, match="missing"):
        File.create(
            tmp_hdf5_path,
            data=DATA,
            scan=reduced_scan,
            overwrite=True,
        )


def test_existing_path(tmp_hdf5_path):
    """Tests if passing a path that already exists raises an error."""
    # Ensure that the file exists
    tmp_hdf5_path.touch()

    with pytest.raises(FileExistsError):
        File.create(
            tmp_hdf5_path,
            data=DATA,
            scan=SCAN,
            probe_name="generic",
            description="Dataset parameters for testing",
        )


def test_overwrite(tmp_hdf5_path):
    """Tests that overwrite=True allows replacing an existing file."""
    tmp_hdf5_path.touch()

    f = File.create(
        tmp_hdf5_path,
        data=DATA,
        scan=SCAN,
        probe_name="generic",
        description="Dataset parameters for testing",
        overwrite=True,
    )
    f.close()
    validate_file(tmp_hdf5_path)


def test_image_only(tmp_hdf5_path):
    """Tests creating a file with only image_sc data (no scan)."""
    image_sc = {
        "values": np.zeros((n_frames, 256, 256), dtype=np.uint8),
        "coordinates": np.zeros((n_frames, 256, 256, 3), dtype=np.float32),
    }
    f = File.create(
        tmp_hdf5_path,
        data={"image_sc": image_sc},
        probe_name="generic",
        description="Image-only dataset",
        overwrite=True,
    )
    f.close()

    with File(tmp_hdf5_path) as dataset:
        assert dataset.data.image_sc.values.shape == (n_frames, 256, 256)


def test_custom_map(tmp_hdf5_path):
    """Tests creating a file with a custom map element in the data group."""
    import warnings

    custom_values = np.zeros((n_frames, 64, 64, 1), dtype=np.uint8)
    custom_coordinates = np.zeros((n_frames, 64, 64, 3), dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = File.create(
            tmp_hdf5_path,
            data={
                "raw_data": DATA["raw_data"],
                "my_custom_overlay": {
                    "values": custom_values,
                    "coordinates": custom_coordinates,
                    "description": "custom overlay map",
                    "unit": "a.u.",
                },
            },
            scan=SCAN,
            overwrite=True,
        )
    f.close()

    with File(tmp_hdf5_path) as f:
        assert "my_custom_overlay" in f["data"]
        np.testing.assert_array_equal(f.data.my_custom_overlay.values[:], custom_values)
        np.testing.assert_array_equal(f.data.my_custom_overlay.coordinates[:], custom_coordinates)


@pytest.fixture
def _scan_and_probe(tmp_path):
    """Return a minimal Scan + Probe pair by round-tripping through generate_example_dataset."""
    from zea.data.file import load_file

    path = tmp_path / "_scan_probe_helper.hdf5"
    generate_example_dataset(path, n_frames=n_frames, n_tx=n_tx, n_el=n_el, n_ax=n_ax)
    data_dict, scan, probe = load_file(path)
    return scan, probe


def test_save_file_custom_maps(tmp_hdf5_path, _scan_and_probe):
    """Tests that save_file correctly stores custom spatial maps in the data group."""
    import warnings

    scan, probe = _scan_and_probe
    custom_values = np.zeros((n_frames, 32, 32, 1), dtype=np.uint8)
    custom_coordinates = np.zeros((n_frames, 32, 32, 3), dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_file(
            path=tmp_hdf5_path,
            scan=scan,
            probe=probe,
            raw_data=np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
            custom_maps={
                "my_overlay": {
                    "values": custom_values,
                    "coordinates": custom_coordinates,
                }
            },
        )

    with File(tmp_hdf5_path) as f:
        assert "my_overlay" in f["data"]
        np.testing.assert_array_equal(f.data.my_overlay.values[:], custom_values)
        np.testing.assert_array_equal(f.data.my_overlay.coordinates[:], custom_coordinates)


def test_save_file_custom_metadata(tmp_hdf5_path, _scan_and_probe):
    """Tests that save_file correctly stores metadata in the metadata group."""
    scan, probe = _scan_and_probe

    save_file(
        path=tmp_hdf5_path,
        scan=scan,
        probe=probe,
        raw_data=np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
        metadata={
            "credit": "Test Lab, 2024",
            "text_report": "Normal acquisition.",
            "annotations": {
                "label": np.array(["healthy", "healthy"]),
            },
        },
    )

    with File(tmp_hdf5_path) as f:
        assert "metadata" in f
        assert f["metadata/credit"][()] == b"Test Lab, 2024"


def test_validate_input_data_docstring():
    """The docstring of validate_input_data must not mention dict as an accepted
    type; only ndarray inputs are supported on the deprecated path."""
    from zea.data.data_format import validate_input_data

    doc = validate_input_data.__doc__
    assert "If a dict" not in doc, "docstring must not advertise dict support"
    assert "ndarray" in doc, "docstring must say ndarray"


def test_bandwidth_percent_warns(tmp_path):
    """generate_zea_dataset must emit a UserWarning when bandwidth_percent is
    passed, so callers know the value will be dropped."""
    from zea.data.data_format import generate_zea_dataset

    n_tx, n_el = 4, 16
    raw = np.zeros((2, n_tx, 32, n_el, 1), dtype=np.float32)

    path = tmp_path / "bw.hdf5"
    with pytest.warns(UserWarning, match="bandwidth_percent"):
        generate_zea_dataset(
            path=str(path),
            raw_data=raw,
            probe_geometry=np.zeros((n_el, 3), dtype=np.float32),
            sampling_frequency=40e6,
            center_frequency=7e6,
            demodulation_frequency=7e6,
            initial_times=np.zeros(n_tx, dtype=np.float32),
            t0_delays=np.zeros((n_tx, n_el), dtype=np.float32),
            probe_name="generic",
            bandwidth_percent=60.0,
        )
