"""Test for the fnumber_mask function."""

import keras
import numpy as np
import pytest
from keras import ops

from zea.beamform.beamformer import (
    fnum_window_fn_hann,
    fnum_window_fn_rect,
    fnum_window_fn_tukey,
    fnumber_mask,
)
from zea.beamform.geometry import compute_element_normals
from zea.beamform.pixelgrid import cartesian_pixel_grid


def _convex_probe_geometry(n_el=64, radius=40e-3, half_angle=0.5):
    """Convex arc with the apex element at the origin facing +z.

    Element i sits at arc-angle ``phi_i`` with position
    ``(R sin phi, 0, R (cos phi - 1))`` and true outward normal
    ``(sin phi, 0, cos phi)``.
    """
    phi = np.linspace(-half_angle, half_angle, n_el).astype(np.float32)
    positions = np.stack(
        [radius * np.sin(phi), np.zeros_like(phi), radius * (np.cos(phi) - 1.0)], axis=-1
    ).astype(np.float32)
    normals = np.stack([np.sin(phi), np.zeros_like(phi), np.cos(phi)], axis=-1).astype(np.float32)
    return positions, normals


@pytest.fixture
def probe_geometry():
    n_el = 5
    return np.stack(
        [np.linspace(-0.05, 0.05, n_el), np.zeros(n_el), np.zeros(n_el)], axis=-1
    ).astype(np.float32)


@pytest.fixture
def flatgrid():
    return (
        cartesian_pixel_grid(
            xlims=(-10e-3, 10e-3), zlims=(0, 20e-3), grid_size_x=65, grid_size_z=65
        )
        .reshape(-1, 3)
        .astype(np.float32)
    )


@pytest.mark.parametrize(
    "fnum_window_fn", [fnum_window_fn_hann, fnum_window_fn_rect, fnum_window_fn_tukey]
)
def test_fnumber_mask(probe_geometry, flatgrid, fnum_window_fn):
    """Runs the fnumber_mask function with different window functions."""
    mask = fnumber_mask(
        flatgrid, probe_geometry=probe_geometry, f_number=0.5, fnum_window_fn=fnum_window_fn
    )

    assert mask.shape == (flatgrid.shape[0], probe_geometry.shape[0], 1)

    mask_middle_element = ops.reshape(mask[:, probe_geometry.shape[0] // 2, 0], (65, 65))

    # Mask should not be zero in front of the element
    idx_x = 32
    idx_z = 32
    assert mask_middle_element[idx_z, idx_x] > 0.0

    # Mask should be zero all the way to the right of the element
    idx_x = 64
    idx_z = 1
    assert mask_middle_element[idx_z, idx_x] == 0.0

    # Mask should be zero just right of the f-number cone boundary
    idx_x = 32 + 16
    idx_z = 16
    assert mask_middle_element[idx_z, idx_x] == 0.0

    idx_x = 32 + 15
    idx_z = 16
    assert mask_middle_element[idx_z, idx_x] > 0.0


def test_compute_element_normals_flat_linear(probe_geometry):
    """A flat linear array must derive exactly +z for every element."""
    normals = ops.convert_to_numpy(compute_element_normals(probe_geometry))
    expected = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (probe_geometry.shape[0], 1))
    assert np.array_equal(normals, expected)


def test_compute_element_normals_single_element():
    """A lone element has no tangent; fall back to +z."""
    normals = ops.convert_to_numpy(compute_element_normals(np.zeros((1, 3), dtype=np.float32)))
    assert np.array_equal(normals, np.array([[0.0, 0.0, 1.0]], dtype=np.float32))


def test_compute_element_normals_convex_arc():
    """Interior elements of a convex arc recover the analytic radial normal."""
    positions, expected = _convex_probe_geometry(n_el=64, radius=40e-3, half_angle=0.5)
    normals = ops.convert_to_numpy(compute_element_normals(positions))
    # Central differences are exact in the interior; endpoints use a one-sided
    # difference so allow them a looser tolerance.
    np.testing.assert_allclose(normals[1:-1], expected[1:-1], atol=1e-4)
    np.testing.assert_allclose(normals[[0, -1]], expected[[0, -1]], atol=2e-2)


@pytest.mark.parametrize(
    "fnum_window_fn", [fnum_window_fn_hann, fnum_window_fn_rect, fnum_window_fn_tukey]
)
def test_fnumber_mask_flat_linear_is_legacy_noop(probe_geometry, flatgrid, fnum_window_fn):
    """On a flat array the per-element-normal mask equals the old global-+z mask.

    The reference is the legacy angle-relative-to-global-+z formula computed
    with the same backend ops, so any difference would be a real change in
    behaviour rather than a norm-implementation rounding difference.
    """
    rel = flatgrid[:, None] - probe_geometry[None]
    rel_norm = ops.linalg.norm(rel, axis=-1)
    alpha = ops.arccos(rel[..., 2] / (rel_norm + 1e-6))
    max_alpha = ops.arctan(1.0 / (2.0 * 0.5 + keras.backend.epsilon()))
    reference = ops.convert_to_numpy(fnum_window_fn(alpha / max_alpha))[..., None]

    mask = ops.convert_to_numpy(
        fnumber_mask(flatgrid, probe_geometry, f_number=0.5, fnum_window_fn=fnum_window_fn)
    )
    assert np.array_equal(mask, reference)


def test_fnumber_mask_convex_widens_field_of_view():
    """Per-element normals extend a convex array's field of view vs forcing +z.

    Because peripheral elements of a convex array physically look outward,
    steering each element's acceptance cone along its true normal illuminates
    sector-edge pixels that the global-+z cone blacks out. That shows up as a
    larger field of view (more pixels receiving any aperture), which is the
    "larger frame" a curved probe should produce.
    """
    # Odd element count so a real element sits exactly at arc-angle 0 (normal +z).
    positions, _ = _convex_probe_geometry(n_el=65, radius=40e-3, half_angle=0.5)
    z_normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (positions.shape[0], 1))

    # A grid spanning the full sector so the tilted elements' look-direction
    # pixels are actually present.
    grid = (
        cartesian_pixel_grid(
            xlims=(-30e-3, 30e-3), zlims=(1e-3, 55e-3), grid_size_x=160, grid_size_z=160
        )
        .reshape(-1, 3)
        .astype(np.float32)
    )

    mask_new = ops.convert_to_numpy(
        fnumber_mask(grid, positions, f_number=2.0, fnum_window_fn=fnum_window_fn_rect)
    )[..., 0]
    mask_old = ops.convert_to_numpy(
        fnumber_mask(
            grid,
            positions,
            f_number=2.0,
            fnum_window_fn=fnum_window_fn_rect,
            element_normals=z_normals,
        )
    )[..., 0]

    # Per-pixel aperture count -> field of view is the set of illuminated pixels.
    aperture_new = mask_new.sum(axis=1)
    aperture_old = mask_old.sum(axis=1)
    assert int((aperture_new > 0).sum()) > int((aperture_old > 0).sum())
    # Some sector-edge pixels are illuminated only once the normals are correct.
    assert np.any((aperture_new > 0) & (aperture_old == 0))

    # The apex element (arc-angle 0) faces +z, so its column is unchanged.
    apex = positions.shape[0] // 2
    np.testing.assert_allclose(mask_new[:, apex], mask_old[:, apex], atol=1e-6)
