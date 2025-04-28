"""Tests for the probes module."""

import numpy as np
import pytest

from usbmd.probes import Probe, get_probe
from usbmd.registry import probe_registry


@pytest.mark.parametrize("probe_name", probe_registry.registered_names())
def test_get_probe(probe_name):
    """Tests the get_probe function by calling it on all registered probes and
    checking that it returns a probe object."""
    probe = get_probe(probe_name)

    assert isinstance(probe, Probe), "get_probe must return a Probe object"


def test_get_probe_error():
    """Tests the get_probe function by calling it on a probe name that is not
    registered and checking that it raises a NotImplementedError."""
    with pytest.raises(NotImplementedError):
        get_probe("nonexistent_probe", fallback=False)


@pytest.mark.parametrize("probe_name", probe_registry.registered_names())
def test_get_default_scan_paramters(probe_name):
    """Tests the get_probe function by calling it on all registered probes and
    calling their get_parameters() method."""
    probe = get_probe(probe_name)

    probe.get_parameters()

    assert isinstance(
        probe.probe_geometry, np.ndarray
    ), "Element positions must be a numpy array"
    assert probe.probe_geometry.shape == (
        probe.n_el,
        3,
    ), "Element positions must be of shape (n_el, 3)"
