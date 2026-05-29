"""Tests for the probes module."""

import os
import tempfile

import numpy as np
import pytest

from zea.data.file import File
from zea.internal.registry import probe_registry
from zea.probes import Probe


@pytest.mark.parametrize("probe_name", probe_registry.registered_names())
def test_get_probe(probe_name):
    """Tests the Probe.from_name function by calling it on all registered probes and
    checking that it returns a probe object."""
    probe = Probe.from_name(probe_name)

    assert isinstance(probe, Probe), "Probe.from_name must return a Probe object"


def test_get_probe_error():
    """Tests the Probe.from_name function by calling it on a probe name that is not
    registered and checking that it raises a NotImplementedError."""
    with pytest.raises(NotImplementedError):
        Probe.from_name("nonexistent_probe")


@pytest.mark.parametrize("probe_name", probe_registry.registered_names())
def test_get_default_scan_paramters(probe_name):
    """Tests the Probe.from_name function by calling it on all registered probes and
    calling their get_parameters() method."""
    probe = Probe.from_name(probe_name)

    probe.get_parameters()

    assert isinstance(probe.probe_geometry, np.ndarray), "Element positions must be a numpy array"
    assert probe.probe_geometry.shape == (
        probe.n_el,
        3,
    ), "Element positions must be of shape (n_el, 3)"


def test_file_create_accepts_probe_object():
    """File.create should accept a Probe object for the probe argument."""
    n_frames, n_tx, n_el, n_ax = 1, 4, 128, 64
    probe = Probe.from_name("verasonics_l11_4v")
    raw = np.zeros((n_frames, n_tx, n_ax, n_el, 1), dtype=np.float32)
    scan = {
        "sampling_frequency": np.float32(40e6),
        "center_frequency": np.float32(6.25e6),
        "demodulation_frequency": np.float32(6.25e6),
        "initial_times": np.zeros(n_tx, dtype=np.float32),
        "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
        "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
        "focus_distances": np.full(n_tx, np.inf, dtype=np.float32),
        "transmit_origins": np.zeros((n_tx, 3), dtype=np.float32),
        "polar_angles": np.zeros(n_tx, dtype=np.float32),
        "time_to_next_transmit": np.ones((n_frames, n_tx), dtype=np.float32) * 1e-4,
    }
    fd, path = tempfile.mkstemp(suffix=".hdf5")
    os.close(fd)
    try:
        f = File.create(path, data={"raw_data": raw}, scan=scan, probe=probe, overwrite=True)
        assert f.probe.name == "verasonics_l11_4v"
        assert f.probe.n_el == 128
        f.close()
    finally:
        os.unlink(path)


def test_file_create_probe_wrong_type():
    """File.create should raise TypeError when probe is an unsupported type."""
    n_frames, n_tx, n_el, n_ax = 1, 4, 128, 64
    raw = np.zeros((n_frames, n_tx, n_ax, n_el, 1), dtype=np.float32)
    fd, path = tempfile.mkstemp(suffix=".hdf5")
    os.close(fd)
    try:
        with pytest.raises(TypeError, match="probe must be a Probe object or a dict"):
            File.create(
                path,
                data={"raw_data": raw},
                probe="verasonics_l11_4v",
                overwrite=True,
            )
    finally:
        os.unlink(path)
