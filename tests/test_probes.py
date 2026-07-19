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
        File.create(path, data={"raw_data": raw}, scan=scan, probe=probe, overwrite=True)
        with File(path) as f:
            assert f.probe.name == "verasonics_l11_4v"
            assert f.probe.n_el == 128
    finally:
        os.unlink(path)


def test_probe_repr():
    """Probe repr is a single-line constructor-style string with key fields."""
    probe = Probe.from_name("verasonics_l11_4v")
    r = repr(probe)
    assert r.startswith("Probe(")
    assert r.endswith(")")
    assert "\n" not in r
    assert "name=" in r
    assert "MHz" in r


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


def test_pitch_derived_from_probe_geometry():
    """If the probe geometry is provided, the pitch should be derived from it."""
    n_el = 4
    pitch_m = 0.3e-3
    xs = np.arange(n_el, dtype=np.float32) * pitch_m
    pg = np.zeros((n_el, 3), dtype=np.float32)
    pg[:, 0] = xs
    probe = Probe(probe_geometry=pg)
    assert probe.pitch == pytest.approx(pitch_m, rel=1e-4)

    probe = Probe()

    with pytest.raises(ValueError, match="Cannot compute pitch: probe_geometry is not set"):
        _ = probe.pitch

    pg = np.zeros((1, 3), dtype=np.float32)
    probe = Probe(probe_geometry=pg)
    with pytest.raises(ValueError, match="Cannot compute pitch: probe has fewer than 2 elements"):
        _ = probe.pitch


def test_verasonics_c5_2v_curved_probe():
    """The C5-2v built-in is a 128-element convex arc (curved, y=0, apex at +z)."""
    from zea.probes import create_curved_probe_geometry

    probe = Probe.from_name("verasonics_c5_2v")
    pg = np.asarray(probe.probe_geometry)

    assert probe.type == "curved"
    assert pg.shape == (128, 3)
    # Imaging plane is x-z; the array is genuinely curved (z varies), not flat.
    assert np.allclose(pg[:, 1], 0.0)
    assert np.ptp(pg[:, 2]) > 1e-3
    # Apex element faces +z (z ~= 0); peripheral elements curve back toward -z.
    assert pg[:, 2].max() == pytest.approx(0.0, abs=1e-4)
    assert pg[:, 2].min() < 0.0
    # Arc spacing (pitch) is uniform at 0.508 mm.
    spacing = np.linalg.norm(np.diff(pg, axis=0), axis=1)
    assert np.allclose(spacing, 0.508e-3, atol=1e-6)

    # The helper reproduces the same geometry from (n_el, pitch, radius).
    regenerated = create_curved_probe_geometry(n_el=128, pitch=0.508e-3, radius=49.57e-3)
    assert np.allclose(pg, regenerated)


def test_verasonics_p4_2v_phased_probe():
    """The P4-2v built-in is a 64-element flat phased array."""
    probe = Probe.from_name("verasonics_p4_2v")
    pg = np.asarray(probe.probe_geometry)

    assert probe.type == "phased"
    assert pg.shape == (64, 3)
    # Flat linear array: y and z are zero, x uniformly spaced at 0.30 mm.
    assert np.allclose(pg[:, 1:], 0.0)
    assert probe.pitch == pytest.approx(0.3e-3, rel=1e-4)
