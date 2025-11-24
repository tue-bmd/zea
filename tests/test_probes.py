"""Tests for the probes module."""

import numpy as np
import pytest

from zea import File
from zea.internal.registry import probe_registry
from zea.probes import Probe

from . import DEFAULT_TEST_SEED


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
        Probe.from_name("nonexistent_probe", fallback=False)


@pytest.mark.parametrize("probe_name", probe_registry.registered_names())
def test_get_default_scan_paramters(probe_name):
    """Tests the Probe.from_name function by calling it on all registered probes and
    calling their get_parameters() method."""
    probe = Probe.from_name(probe_name)

    probe.get_parameters()

    # Because generic probes do not have a geometry, we skip the test for them
    if probe_name == "generic":
        return

    assert isinstance(probe.probe_geometry, np.ndarray), "Element positions must be a numpy array"
    assert probe.probe_geometry.shape == (
        probe.n_el,
        3,
    ), "Element positions must be of shape (n_el, 3)"


def test_probe_from_file(tmp_path):
    """Test loading a probe from an HDF5 file"""

    # Use a known probe name from the registry
    probe_name = "verasonics_l11_4v"
    default_probe = Probe.from_name(probe_name)
    default_geometry = np.copy(default_probe.probe_geometry)

    # Create a different geometry for the test
    new_geometry = default_geometry + 1.0

    # Create HDF5 file with probe name and scan/probe_geometry
    file_path = tmp_path / "test_probe_geometry.hdf5"
    with File(file_path, "w") as f:
        f.attrs["probe"] = probe_name
        f.attrs["description"] = "Test file for probe geometry update"
        scan_grp = f.create_group("scan")
        scan_grp.create_dataset("probe_geometry", data=new_geometry)
        scan_grp.create_dataset("n_el", data=new_geometry.shape[0])

    with File(file_path, "r") as f:
        probe = f.probe()
        assert np.allclose(probe.probe_geometry, new_geometry)
        assert probe_registry.get_name(probe) == f.probe_name

    # Use a probe name that is not registered
    unknown_probe_name = "unknown_probe_xyz"

    # Create dummy probe_geometry
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    probe_geometry = rng.standard_normal((32, 3))

    # Create HDF5 file with unknown probe name and probe_geometry
    file_path = tmp_path / "unknown_probe.hdf5"
    with File(file_path, "w") as f:
        f.attrs["probe"] = unknown_probe_name
        scan_grp = f.create_group("scan")
        scan_grp.create_dataset("probe_geometry", data=probe_geometry)
        scan_grp.create_dataset("n_el", data=probe_geometry.shape[0])

    with File(file_path, "r") as f:
        probe = f.probe()
        assert np.allclose(probe.probe_geometry, probe_geometry)
        assert probe_registry.get_name(probe) == "generic"
