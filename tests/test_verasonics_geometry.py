"""Tests for the Verasonics converter's probe-geometry classifier and steering angles."""

import numpy as np

from zea.data.convert.verasonics import classify_ordered_geometry, steer_to_spherical
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


def test_steer_to_spherical_keeps_xz_steering():
    """Without elevation steering the signed polar angle is kept and azimuth is zero."""
    theta = np.radians([-20.0, -10.0, 0.0, 10.0, 20.0])
    polar, azimuth = steer_to_spherical(theta, -0.0 * theta)
    np.testing.assert_allclose(polar, theta, atol=1e-12)
    np.testing.assert_allclose(azimuth, 0.0, atol=1e-12)


def test_steer_to_spherical_elevation_only():
    """``Steer = (0, 10 deg)`` tilts toward +y: polar 10 deg at azimuth 90 deg."""
    polar, azimuth = steer_to_spherical(np.array([0.0]), np.radians([10.0]))
    np.testing.assert_allclose(np.degrees(polar), 10.0)
    np.testing.assert_allclose(np.degrees(azimuth), 90.0)


def test_steer_to_spherical_direction():
    """The spherical angles point along the Verasonics steering direction."""
    theta, alpha = np.meshgrid(np.radians(np.linspace(-30, 30, 7)), np.radians([-20, 0, 7, 20]))
    theta, alpha = theta.ravel(), alpha.ravel()
    polar, azimuth = steer_to_spherical(theta, alpha)

    expected = np.stack(
        [np.sin(theta) * np.cos(alpha), np.sin(alpha), np.cos(theta) * np.cos(alpha)], axis=-1
    )
    direction = np.stack(
        [np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)],
        axis=-1,
    )
    np.testing.assert_allclose(direction, expected, atol=1e-12)
    assert np.all((azimuth > -np.pi / 2) & (azimuth <= np.pi / 2))
