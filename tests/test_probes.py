"""Tests for the probes module."""

import numpy as np
import pytest

from usbmd.probes import Probe
from usbmd.registry import probe_registry


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

    assert isinstance(
        probe.probe_geometry, np.ndarray
    ), "Element positions must be a numpy array"
    assert probe.probe_geometry.shape == (
        probe.n_el,
        3,
    ), "Element positions must be of shape (n_el, 3)"
