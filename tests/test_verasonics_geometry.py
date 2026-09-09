"""Tests for the Verasonics converter's probe-geometry ordering classifier."""

import numpy as np

from zea.data.convert.verasonics import classify_ordered_geometry
from zea.probes import create_curved_probe_geometry, create_probe_geometry


def test_classify_linear_array():
    """A uniform linear array is classified as ordered/linear."""
    geom = create_probe_geometry(n_el=64, pitch=0.3e-3)
    assert classify_ordered_geometry(geom) == (True, "linear")


def test_classify_curved_array():
    """A uniform convex arc (e.g. C5-2v) is ordered/arc, not linear."""
    geom = create_curved_probe_geometry(n_el=128, pitch=0.508e-3, radius=49.57e-3)
    is_ordered, kind = classify_ordered_geometry(geom)
    assert is_ordered is True
    assert kind == "arc"


def test_classify_unordered_geometry():
    """A shuffled / non-uniform element list is not a recognized ordered array."""
    rng = np.random.default_rng(0)
    geom = rng.normal(size=(64, 3)).astype(np.float32) * 1e-3
    assert classify_ordered_geometry(geom) == (False, "unknown")


def test_classify_two_elements_is_linear():
    """Two elements trivially define a linear step."""
    geom = np.array([[0.0, 0.0, 0.0], [0.3e-3, 0.0, 0.0]], dtype=np.float32)
    assert classify_ordered_geometry(geom) == (True, "linear")


def test_matrix_array_is_unknown():
    """A 2-D matrix array (raster-ordered) is not linear or a single arc."""
    xs, ys = np.meshgrid(np.arange(8), np.arange(8))
    geom = np.stack([xs.ravel() * 0.3e-3, ys.ravel() * 0.3e-3, np.zeros(64)], axis=-1).astype(
        np.float32
    )
    assert classify_ordered_geometry(geom) == (False, "unknown")
