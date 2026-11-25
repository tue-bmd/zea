"""Tests for the 3-D scan frustum detection in zea.tools.fit_scan_cone."""

import numpy as np
import pytest

from zea.tools.fit_scan_cone import detect_frustum_parameters


def make_frustum_mask(
    shape,
    apex,
    theta_range,
    phi_range,
    rho_max,
    spacing=(1.0, 1.0, 1.0),
    fan_of_sectors=False,
):
    """Boolean ``(z, x, y)`` mask of a scan frustum, as in a scan-converted volume.

    Args:
        shape: Volume shape ``(n_z, n_x, n_y)``.
        apex: Apex as a fractional voxel index ``(z, x, y)``; ``z`` may be negative.
        theta_range: Azimuth range in radians, along ``x``.
        phi_range: Elevation range in radians, along ``y``.
        rho_max: Radius of the far cap, in units of ``spacing``.
        spacing: Voxel size along ``(z, x, y)``.
        fan_of_sectors: If True, model a fan of 2-D sectors tilted in elevation, where
            ``theta`` is measured within each tilted plane. Its constant-depth cross-sections
            are pincushion-shaped instead of rectangular: narrowest in ``x`` at ``y = 0``.
    """
    z, x, y = [(np.arange(n) - a) * s for n, a, s in zip(shape, apex, spacing)]
    z, x, y = z[:, None, None], x[None, :, None], y[None, None, :]
    depth = np.maximum(z, 1e-9)
    in_plane_depth = np.sqrt(depth**2 + y**2) if fan_of_sectors else depth
    return (
        (z > 0)
        & (x >= in_plane_depth * np.tan(theta_range[0]))
        & (x <= in_plane_depth * np.tan(theta_range[1]))
        & (y >= depth * np.tan(phi_range[0]))
        & (y <= depth * np.tan(phi_range[1]))
        & (np.sqrt(z**2 + x**2 + y**2) <= rho_max)
    )


FRUSTUMS = {
    "symmetric": dict(
        shape=(60, 70, 64), apex=(0.0, 35.0, 32.0), theta=(-0.6, 0.6), phi=(-0.5, 0.5), rho=58
    ),
    "apex_above_volume": dict(
        shape=(60, 70, 64), apex=(-4.0, 35.0, 32.0), theta=(-0.6, 0.6), phi=(-0.5, 0.5), rho=62
    ),
    "asymmetric_off_centre": dict(
        shape=(60, 72, 64), apex=(-3.0, 40.5, 28.0), theta=(-0.7, 0.5), phi=(-0.4, 0.6), rho=60
    ),
    # Wide sector in a tight volume: the edges reach the lateral borders well before the cap.
    "clipped_by_borders": dict(
        shape=(64, 60, 56), apex=(-2.0, 30.0, 28.0), theta=(-0.8, 0.8), phi=(-0.75, 0.75), rho=70
    ),
}


@pytest.mark.parametrize("name", FRUSTUMS)
def test_detect_frustum_recovers_parameters(name):
    """The apex, angle ranges and depth of a synthetic frustum are recovered."""
    cfg = FRUSTUMS[name]
    mask = make_frustum_mask(cfg["shape"], cfg["apex"], cfg["theta"], cfg["phi"], cfg["rho"])

    result = detect_frustum_parameters(mask)

    np.testing.assert_allclose(result["apex"], cfg["apex"], atol=1.0)
    np.testing.assert_allclose(result["theta_range"], cfg["theta"], atol=np.radians(1.5))
    np.testing.assert_allclose(result["phi_range"], cfg["phi"], atol=np.radians(1.5))
    # rho_max is the farthest data voxel, which may lie below a volume that clips the cap.
    assert result["rho_range"][1] <= cfg["rho"] + 1e-6
    assert result["rho_range"][1] > cfg["rho"] - 2.0 or name == "clipped_by_borders"
    # The nearest data lies just below the apex, or in the first row if the apex is above it.
    assert result["rho_range"][0] < max(0.0, -cfg["apex"][0]) + 1.5
    assert max(result["edge_rms"]) < 1.0
    assert result["apex_depth_spread"] < 1.5
    assert result["sector_type"] == "pyramid"
    np.testing.assert_allclose(result["edge_widening"], 0.0, atol=0.15)


def test_detect_frustum_physical_spacing():
    """With anisotropic voxels, angles and distances are returned in physical units."""
    spacing = (0.5e-3, 0.8e-3, 0.7e-3)
    apex = (-3.0, 30.0, 32.0)
    theta, phi, rho = (-0.6, 0.6), (-0.5, 0.5), 0.035
    mask = make_frustum_mask((76, 60, 64), apex, theta, phi, rho, spacing=spacing)

    result = detect_frustum_parameters(mask, spacing=spacing)

    # The faces pass through the centres of the outermost voxels, on average half a lateral
    # voxel inside the true boundary. With wide, shallow voxels this moves the apex down by
    # about 0.5 * 0.8 / 0.5 / tan(0.6) = 1.2 voxels.
    np.testing.assert_allclose(result["apex"], apex, atol=1.5)
    np.testing.assert_allclose(result["theta_range"], theta, atol=np.radians(1.5))
    np.testing.assert_allclose(result["phi_range"], phi, atol=np.radians(1.5))
    np.testing.assert_allclose(result["rho_range"][1], rho, atol=2 * max(spacing))
    assert result["sector_type"] == "pyramid"


def test_detect_frustum_fan_of_sectors_gives_bounding_frustum():
    """For a fan of tilted 2-D sectors, the frustum spans the corner rays of the sector, and
    the sector is recognised as a fan."""
    theta, phi = (-0.6, 0.6), (-0.5, 0.5)
    mask = make_frustum_mask((60, 80, 64), (-2.0, 40.0, 32.0), theta, phi, 60, fan_of_sectors=True)

    result = detect_frustum_parameters(mask)

    # In frustum terms, the sector is widest in x at the elevation extremes.
    corner = np.arctan(np.tan(theta[1]) / np.cos(phi[1]))
    np.testing.assert_allclose(result["theta_range"], (-corner, corner), atol=np.radians(1.5))
    np.testing.assert_allclose(result["phi_range"], phi, atol=np.radians(1.5))
    # The azimuth edges widen as 1 / cos(phi), the elevation edges do not widen.
    assert result["sector_type"] == "fan"
    np.testing.assert_allclose(result["edge_widening"], (1.0, 0.0), atol=0.3)


def test_detect_frustum_other_sector_shapes_are_unknown():
    """A fan whose sectors lie in the elevation plane is neither of the known shapes."""
    mask = make_frustum_mask(
        (60, 80, 64), (-2.0, 40.0, 32.0), (-0.6, 0.6), (-0.5, 0.5), 60, fan_of_sectors=True
    )

    result = detect_frustum_parameters(mask.transpose(0, 2, 1))  # swap azimuth and elevation

    assert result["sector_type"] == "unknown"
    np.testing.assert_allclose(result["edge_widening"], (0.0, 1.0), atol=0.3)


def test_detect_frustum_ignores_stray_voxels():
    """Voxels outside the sector are dropped with the largest connected component."""
    cfg = FRUSTUMS["apex_above_volume"]
    mask = make_frustum_mask(cfg["shape"], cfg["apex"], cfg["theta"], cfg["phi"], cfg["rho"])
    clean = detect_frustum_parameters(mask)

    mask[2, 1, 1] = mask[30, 0, 0] = mask[5, 68, 2] = True  # far outside the sector
    noisy = detect_frustum_parameters(mask)

    for key in ("apex", "theta_range", "phi_range", "rho_range"):
        np.testing.assert_allclose(noisy[key], clean[key])


@pytest.mark.parametrize(
    "mask, match",
    [
        (np.zeros((20, 20, 20), dtype=bool), "no data voxels"),
        (np.ones((20, 20), dtype=bool), "3-D"),
        # A box has parallel sides: there is no apex.
        (
            np.pad(np.ones((30, 20, 20), dtype=bool), ((2, 2), (5, 5), (5, 5))),
            "Failed to detect scan frustum",
        ),
    ],
)
def test_detect_frustum_raises(mask, match):
    """Detection fails loudly on volumes that do not contain a frustum."""
    with pytest.raises(ValueError, match=match):
        detect_frustum_parameters(mask)


def test_detect_frustum_rejects_upside_down_volume():
    """The apex must be at the top of the first axis."""
    cfg = FRUSTUMS["symmetric"]
    mask = make_frustum_mask(cfg["shape"], cfg["apex"], cfg["theta"], cfg["phi"], cfg["rho"])
    with pytest.raises(ValueError):
        detect_frustum_parameters(mask[::-1])
