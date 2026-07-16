"""Test generating and validating zea data format."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Generator

import h5py
import numpy as np
import pytest

from zea.data.file import File, validate_file
from zea.data.spec import BLOSC_NTHREADS, MAX_CHUNK_BYTES, PAGED_LAYOUT, ScanSpec

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

# Probe dict for File.create (probe_geometry is required when raw_data is present)
PROBE = {
    "name": "generic",
    "probe_geometry": np.zeros((n_el, 3), dtype=np.float32),
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
    File.create(
        tmp_hdf5_path,
        data=DATA,
        scan=SCAN,
        probe=PROBE,
        description="Dataset parameters for testing",
        overwrite=True,
    )
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
        File.create(
            tmp_hdf5_path,
            data=DATA,
            scan=wrong_scan,
            probe=PROBE,
            description="Dataset parameters for testing",
            overwrite=True,
        )


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
    File.create(
        tmp_hdf5_path,
        data=DATA,
        scan=reduced_scan,
        probe=PROBE,
        overwrite=True,
    )
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


@pytest.mark.parametrize(
    "chunk_axes, expected",
    [
        # One chunk per (frame, transmit) plane, full n_ax/n_el/n_ch.
        (("n_frames", "n_tx"), lambda s: (1, 1) + s[2:]),
        # One full frame per chunk (the current default).
        (("n_frames",), lambda s: (1,) + s[1:]),
        # Disabled: contiguous storage (no chunking) — needs compression off.
        (None, lambda s: None),
        ((), lambda s: None),
    ],
)
def test_chunk_axes(chunk_axes, expected, tmp_hdf5_path):
    """chunk_axes controls which dimensions get HDF5 chunk size 1."""
    # Contiguous storage is only possible without compression.
    compression = None if not chunk_axes else "lzf"

    File.create(
        path=tmp_hdf5_path,
        chunk_axes=chunk_axes,
        compression=compression,
        data=DATA,
        scan=SCAN,
        probe=PROBE,
    )

    validate_file(tmp_hdf5_path)

    with File(tmp_hdf5_path) as file:
        raw_data = file["data/raw_data"]
        assert raw_data.chunks == expected(raw_data.shape)
        # Data must still be readable regardless of chunking.
        assert np.array_equal(raw_data[:], DATA["raw_data"])


def test_blosc_nthreads_is_set_for_writes():
    """Blosc threads within a chunk, which is ~4x of the write throughput.

    A regression here would be silent: the files stay correct, just slow to produce.
    """
    assert os.environ.get("BLOSC_NTHREADS") == str(BLOSC_NTHREADS)
    # Lower bound is 1 (a single-core host is a valid config); above 8 measured *slower*
    # (memory-bound).
    assert 1 <= BLOSC_NTHREADS <= 8, "more threads than this measured *slower* (memory-bound)"


def test_blosc_nthreads_does_not_override_the_user():
    """An explicit BLOSC_NTHREADS wins — zea only supplies a default.

    Run in a subprocess because the behaviour happens at import: reloading the module in-process
    rebuilds zea's dataclasses and breaks isinstance checks across the rest of the suite.
    """
    env = {**os.environ, "BLOSC_NTHREADS": "3"}
    proc = subprocess.run(
        [sys.executable, "-c", "import os, zea.data.spec; print(os.environ['BLOSC_NTHREADS'])"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert proc.stdout.strip().splitlines()[-1] == "3"  # last line: zea logs a backend banner


def test_default_write_is_blosc_and_per_frame(tmp_hdf5_path):
    """Default File.create uses Blosc(zstd) compression and one-frame-per-chunk."""
    import hdf5plugin

    File.create(path=tmp_hdf5_path, data=DATA, scan=SCAN, probe=PROBE)  # all defaults

    with File(tmp_hdf5_path) as file:
        raw_data = file["data/raw_data"]
        # per-frame default: chunk size 1 on n_frames, full on the rest
        assert raw_data.chunks == (1,) + raw_data.shape[1:]
        # Blosc filter is applied (h5py doesn't name external filters, so check id)
        dcpl = raw_data.id.get_create_plist()
        filter_ids = {dcpl.get_filter(i)[0] for i in range(dcpl.get_nfilters())}
        assert hdf5plugin.Blosc.filter_id in filter_ids
        assert np.array_equal(raw_data[:], DATA["raw_data"])


def test_large_frames_are_split_into_capped_chunks(tmp_hdf5_path):
    """A frame bigger than MAX_CHUNK_BYTES is split along n_tx, not stored as one chunk.

    A chunk decodes in a single thread, so a whole-frame chunk of a high-transmit scan has
    nothing to parallelise (~7x slower to read). The split must fall on ``n_tx``, leaving
    ``n_ax``/``n_el`` whole so each chunk stays a contiguous run of the array.
    """
    big_frames, big_tx, big_ax, big_el = 2, 64, 2048, 64
    raw = np.zeros((big_frames, big_tx, big_ax, big_el, 1), dtype=np.float32)
    assert raw[0].nbytes > MAX_CHUNK_BYTES, "the frame must exceed the cap to test the cap"

    scan = {
        **SCAN,
        "initial_times": np.zeros(big_tx, dtype=np.float32),
        "t0_delays": np.zeros((big_tx, big_el), dtype=np.float32),
        "focus_distances": np.zeros(big_tx, dtype=np.float32),
        "polar_angles": np.linspace(-np.pi / 2, np.pi / 2, big_tx, dtype=np.float32),
        "azimuth_angles": np.zeros(big_tx, dtype=np.float32),
        "tx_apodizations": np.ones((big_tx, big_el), dtype=np.float32),
        "time_to_next_transmit": np.ones((big_frames, big_tx), dtype=np.float32),
        "transmit_origins": np.zeros((big_tx, 3), dtype=np.float32),
    }
    probe = {**PROBE, "probe_geometry": np.zeros((big_el, 3), dtype=np.float32)}
    File.create(
        path=tmp_hdf5_path,
        data={"raw_data": raw},
        scan=scan,
        probe=probe,
        overwrite=True,
    )

    with File(tmp_hdf5_path) as file:
        raw_data = file["data/raw_data"]
        chunks = raw_data.chunks
        chunk_bytes = np.prod(chunks) * raw.dtype.itemsize

        assert chunk_bytes <= MAX_CHUNK_BYTES
        assert chunks[0] == 1  # still one frame per chunk
        assert 1 <= chunks[1] < big_tx  # split along n_tx ...
        assert chunks[2:] == (big_ax, big_el, 1)  # ... and only along n_tx
        assert np.array_equal(raw_data[:], raw)


def test_default_write_is_paged(tmp_hdf5_path):
    """Files are written with a paged file space, which speeds up streamed opens.

    Paging collects the metadata a reader walks on open into few adjacent pages, so a
    cold open over HTTP costs fewer round trips. It must not change what is stored: the
    file stays a plain HDF5 file, readable by h5py without any special handling.
    """
    File.create(path=tmp_hdf5_path, data=DATA, scan=SCAN, probe=PROBE)  # all defaults

    with h5py.File(tmp_hdf5_path, "r") as file:
        plist = file.id.get_create_plist()
        strategy = plist.get_file_space_strategy()[0]
        assert strategy == h5py.h5f.FSPACE_STRATEGY_PAGE
        assert plist.get_file_space_page_size() == PAGED_LAYOUT["fs_page_size"]
        # plain h5py (no zea key remapping): the file is an ordinary HDF5 file
        assert np.array_equal(file["tracks/track_0/data/raw_data"][:], DATA["raw_data"])


def test_existing_path(tmp_hdf5_path):
    """Tests if passing a path that already exists raises an error."""
    # Ensure that the file exists
    tmp_hdf5_path.touch()

    with pytest.raises(FileExistsError):
        File.create(
            tmp_hdf5_path,
            data=DATA,
            scan=SCAN,
            probe=PROBE,
            description="Dataset parameters for testing",
        )


def test_overwrite(tmp_hdf5_path):
    """Tests that overwrite=True allows replacing an existing file."""
    tmp_hdf5_path.touch()

    File.create(
        tmp_hdf5_path,
        data=DATA,
        scan=SCAN,
        probe=PROBE,
        description="Dataset parameters for testing",
        overwrite=True,
    )
    validate_file(tmp_hdf5_path)


def test_image_only(tmp_hdf5_path):
    """Tests creating a file with only image data (no scan)."""
    image = {
        "values": np.zeros((n_frames, 256, 256), dtype=np.uint8),
        "coordinates": np.zeros((n_frames, 256, 256, 3), dtype=np.float32),
    }
    File.create(
        tmp_hdf5_path,
        data={"image": image},
        probe=PROBE,
        description="Image-only dataset",
        overwrite=True,
    )

    with File(tmp_hdf5_path) as dataset:
        assert dataset.data.image.values.shape == (n_frames, 256, 256)


def test_custom_map(tmp_hdf5_path):
    """Tests creating a file with a custom map element in the data group."""
    import warnings

    custom_values = np.zeros((n_frames, 64, 64, 1), dtype=np.uint8)
    custom_coordinates = np.zeros((n_frames, 64, 64, 3), dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        File.create(
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
            probe=PROBE,
            overwrite=True,
        )

    with File(tmp_hdf5_path) as f:
        assert "my_custom_overlay" in f["data"]
        np.testing.assert_array_equal(f.data.my_custom_overlay.values[:], custom_values)
        np.testing.assert_array_equal(f.data.my_custom_overlay.coordinates[:], custom_coordinates)


@pytest.fixture
def _parameters(tmp_path):
    """Return a Parameters object loaded via File.load_parameters()."""
    path = tmp_path / "_parameters_helper.hdf5"
    generate_example_dataset(path, n_frames=n_frames, n_tx=n_tx, n_el=n_el, n_ax=n_ax)
    with File(path) as f:
        parameters = f.load_parameters()
    return parameters


def test_save_file_custom_maps(tmp_hdf5_path, _parameters):
    """Tests that saving correctly stores custom spatial maps in the data group."""
    import warnings

    custom_values = np.zeros((n_frames, 32, 32, 1), dtype=np.uint8)
    custom_coordinates = np.zeros((n_frames, 32, 32, 3), dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        File.create(
            path=tmp_hdf5_path,
            data={
                "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
                "my_overlay": {
                    "values": custom_values,
                    "coordinates": custom_coordinates,
                },
            },
            scan=_parameters.to_scan_dict(),
            probe=_parameters.to_probe_dict(),
            overwrite=True,
        )

    with File(tmp_hdf5_path) as f:
        assert "my_overlay" in f["data"]
        np.testing.assert_array_equal(f.data.my_overlay.values[:], custom_values)
        np.testing.assert_array_equal(f.data.my_overlay.coordinates[:], custom_coordinates)


def test_save_file_custom_metadata(tmp_hdf5_path, _parameters):
    """Tests that saving correctly stores metadata in the metadata group."""
    File.create(
        path=tmp_hdf5_path,
        data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
        scan=_parameters.to_scan_dict(),
        probe=_parameters.to_probe_dict(),
        overwrite=True,
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
