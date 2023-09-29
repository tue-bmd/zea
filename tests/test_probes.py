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
        get_probe("nonexistent_probe")


@pytest.mark.parametrize("probe_name", probe_registry.registered_names())
def test_get_default_scan_paramters(probe_name):
    """Tests the get_probe function by calling it on all registered probes and
    calling their get_default_scan_parameters method."""
    probe = get_probe(probe_name)

    probe.get_default_scan_parameters()


@pytest.mark.parametrize("probe_name", probe_registry.registered_names())
def test_probe_attributes(probe_name):
    """Tests the get_probe function by calling it on all registered probes and
    checking that
    1. the element positions are of the correct shape
    2. the probe type is either 'linear' or 'phased'
    """

    probe = get_probe(probe_name)

    assert isinstance(probe.ele_pos, np.ndarray), (
        "Element positions must be" " a numpy array"
    )
    assert probe.ele_pos.shape == (probe.N_el, 3), (
        "Element positions must be" " of shape (N_el, 3)"
    )
    # assert probe.bandwidth is not None, 'Probe must have a bandwidth'
    assert probe.probe_type in ("linear", "phased"), (
        "Probe type must be" ' either "linear" or' ' "phased"'
    )
