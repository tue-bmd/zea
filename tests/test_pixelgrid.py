"""Tests for the cartesian_pixel_grid function."""

import numpy as np
import pytest

from zea.beamform.pixelgrid import cartesian_pixel_grid

# --- 2D Grid Tests ---


def test_2d_grid_shape_with_grid_sizes():
    """Test 2D grid output shape when using grid_size_x and grid_size_z."""
    grid = cartesian_pixel_grid(
        xlims=(-0.02, 0.02),
        zlims=(0.0, 0.06),
        grid_size_x=128,
        grid_size_z=256,
    )
    assert grid.shape == (256, 128, 3)


def test_2d_grid_shape_with_spacings():
    """Test 2D grid output shape when using dx and dz."""
    dx, dz = 0.001, 0.0005
    xlims, zlims = (-0.02, 0.02), (0.0, 0.06)

    grid = cartesian_pixel_grid(
        xlims=xlims,
        zlims=zlims,
        dx=dx,
        dz=dz,
    )

    expected_nx = len(np.arange(xlims[0], xlims[1] + 1e-10, dx))
    expected_nz = len(np.arange(zlims[0], zlims[1] + 1e-10, dz))
    assert grid.shape == (expected_nz, expected_nx, 3)


def test_2d_grid_coordinate_bounds():
    """Test that 2D grid coordinates span the expected limits."""
    xlims, zlims = (-0.02, 0.02), (0.0, 0.06)

    grid = cartesian_pixel_grid(
        xlims=xlims,
        zlims=zlims,
        grid_size_x=128,
        grid_size_z=256,
    )

    # X coordinates
    assert np.isclose(grid[0, 0, 0], xlims[0], atol=1e-9)
    assert np.isclose(grid[0, -1, 0], xlims[1], atol=1e-9)

    # Z coordinates
    assert np.isclose(grid[0, 0, 2], zlims[0], atol=1e-9)
    assert np.isclose(grid[-1, 0, 2], zlims[1], atol=1e-9)

    # Y coordinates should all be zero
    assert np.allclose(grid[:, :, 1], 0.0)


def test_2d_grid_asymmetric_xlims():
    """Test 2D grid with asymmetric x limits."""
    xlims, zlims = (-0.01, 0.03), (0.005, 0.055)

    grid = cartesian_pixel_grid(
        xlims=xlims,
        zlims=zlims,
        grid_size_x=100,
        grid_size_z=200,
    )

    assert grid.shape == (200, 100, 3)
    assert np.isclose(grid[0, 0, 0], xlims[0], atol=1e-9)
    assert np.isclose(grid[0, -1, 0], xlims[1], atol=1e-9)
    assert np.isclose(grid[0, 0, 2], zlims[0], atol=1e-9)
    assert np.isclose(grid[-1, 0, 2], zlims[1], atol=1e-9)


def test_2d_grid_raises_on_both_sizes_and_spacings():
    """Test that providing both grid sizes and spacings raises error."""
    with pytest.raises(ValueError, match="either provide grid_size_x & grid_size_z"):
        cartesian_pixel_grid(
            xlims=(-0.02, 0.02),
            zlims=(0.0, 0.06),
            grid_size_x=128,
            grid_size_z=256,
            dx=0.001,
            dz=0.0005,
        )


def test_2d_grid_raises_on_neither_sizes_nor_spacings():
    """Test that providing neither grid sizes nor spacings raises error."""
    with pytest.raises(ValueError, match="either provide grid_size_x & grid_size_z"):
        cartesian_pixel_grid(
            xlims=(-0.02, 0.02),
            zlims=(0.0, 0.06),
        )


# --- 3D Grid Tests ---


def test_3d_grid_shape_with_grid_sizes():
    """Test 3D grid output shape when using grid sizes."""
    grid = cartesian_pixel_grid(
        xlims=(-0.02, 0.02),
        zlims=(0.0, 0.06),
        ylims=(-0.01, 0.01),
        grid_size_x=128,
        grid_size_y=64,
        grid_size_z=256,
    )
    assert grid.shape == (256, 128, 64, 3)


def test_3d_grid_shape_with_spacings():
    """Test 3D grid output shape when using spacings."""
    xlims, ylims, zlims = (-0.02, 0.02), (-0.01, 0.01), (0.0, 0.06)
    dx, dy, dz = 0.001, 0.001, 0.0005

    grid = cartesian_pixel_grid(
        xlims=xlims,
        zlims=zlims,
        ylims=ylims,
        dx=dx,
        dy=dy,
        dz=dz,
    )

    expected_nx = len(np.arange(xlims[0], xlims[1] + 1e-10, dx))
    expected_ny = len(np.arange(ylims[0], ylims[1] + 1e-10, dy))
    expected_nz = len(np.arange(zlims[0], zlims[1] + 1e-10, dz))
    assert grid.shape == (expected_nz, expected_nx, expected_ny, 3)


def test_3d_grid_coordinate_bounds():
    """Test that 3D grid coordinates span the expected limits."""
    xlims, ylims, zlims = (-0.02, 0.02), (-0.01, 0.01), (0.0, 0.06)

    grid = cartesian_pixel_grid(
        xlims=xlims,
        zlims=zlims,
        ylims=ylims,
        grid_size_x=128,
        grid_size_y=64,
        grid_size_z=256,
    )

    # X coordinates
    assert np.isclose(grid[0, 0, 0, 0], xlims[0], atol=1e-9)
    assert np.isclose(grid[0, -1, 0, 0], xlims[1], atol=1e-9)

    # Y coordinates
    assert np.isclose(grid[0, 0, 0, 1], ylims[0], atol=1e-9)
    assert np.isclose(grid[0, 0, -1, 1], ylims[1], atol=1e-9)

    # Z coordinates
    assert np.isclose(grid[0, 0, 0, 2], zlims[0], atol=1e-9)
    assert np.isclose(grid[-1, 0, 0, 2], zlims[1], atol=1e-9)


def test_3d_grid_raises_on_both_sizes_and_spacings():
    """Test that providing both grid sizes and spacings raises error."""
    with pytest.raises(ValueError, match="either provide grid_size_x/grid_size_y/grid_size_z"):
        cartesian_pixel_grid(
            xlims=(-0.02, 0.02),
            zlims=(0.0, 0.06),
            ylims=(-0.01, 0.01),
            grid_size_x=128,
            grid_size_y=64,
            grid_size_z=256,
            dx=0.001,
            dy=0.001,
            dz=0.0005,
        )


def test_3d_grid_raises_on_neither_sizes_nor_spacings():
    """Test that providing neither grid sizes nor spacings raises error."""
    with pytest.raises(ValueError, match="either provide grid_size_x/grid_size_y/grid_size_z"):
        cartesian_pixel_grid(
            xlims=(-0.02, 0.02),
            zlims=(0.0, 0.06),
            ylims=(-0.01, 0.01),
        )


def test_3d_grid_raises_on_partial_sizes():
    """Test that providing partial grid sizes for 3D raises error."""
    with pytest.raises(ValueError):
        cartesian_pixel_grid(
            xlims=(-0.02, 0.02),
            zlims=(0.0, 0.06),
            ylims=(-0.01, 0.01),
            grid_size_x=128,
            grid_size_z=256,
            # Missing grid_size_y
        )


# --- Coordinate Ordering Tests ---


def test_2d_x_increases_along_axis_1():
    """Test that x coordinates increase along axis 1 in 2D grid."""
    grid = cartesian_pixel_grid(
        xlims=(-0.02, 0.02),
        zlims=(0.0, 0.06),
        grid_size_x=64,
        grid_size_z=128,
    )
    x_coords = grid[0, :, 0]
    assert np.all(np.diff(x_coords) > 0)


def test_2d_z_increases_along_axis_0():
    """Test that z coordinates increase along axis 0 in 2D grid."""
    grid = cartesian_pixel_grid(
        xlims=(-0.02, 0.02),
        zlims=(0.0, 0.06),
        grid_size_x=64,
        grid_size_z=128,
    )
    z_coords = grid[:, 0, 2]
    assert np.all(np.diff(z_coords) > 0)


def test_3d_coordinates_increase_along_correct_axes():
    """Test that x/y/z coordinates increase along correct axes in 3D grid."""
    grid = cartesian_pixel_grid(
        xlims=(-0.02, 0.02),
        zlims=(0.0, 0.06),
        ylims=(-0.01, 0.01),
        grid_size_x=64,
        grid_size_y=32,
        grid_size_z=128,
    )

    # X should increase along axis 1
    assert np.all(np.diff(grid[0, :, 0, 0]) > 0)
    # Y should increase along axis 2
    assert np.all(np.diff(grid[0, 0, :, 1]) > 0)
    # Z should increase along axis 0
    assert np.all(np.diff(grid[:, 0, 0, 2]) > 0)


# --- Spacing Tests ---


def test_2d_uniform_spacing():
    """Test that x and z spacing is uniform in 2D grid."""
    grid = cartesian_pixel_grid(
        xlims=(-0.02, 0.02),
        zlims=(0.0, 0.06),
        grid_size_x=64,
        grid_size_z=128,
    )

    x_spacings = np.diff(grid[0, :, 0])
    z_spacings = np.diff(grid[:, 0, 2])

    assert np.allclose(x_spacings, x_spacings[0])
    assert np.allclose(z_spacings, z_spacings[0])


def test_3d_uniform_spacing():
    """Test that spacing is uniform along all axes in 3D grid."""
    grid = cartesian_pixel_grid(
        xlims=(-0.02, 0.02),
        zlims=(0.0, 0.06),
        ylims=(-0.01, 0.01),
        grid_size_x=64,
        grid_size_y=32,
        grid_size_z=128,
    )

    x_spacings = np.diff(grid[0, :, 0, 0])
    y_spacings = np.diff(grid[0, 0, :, 1])
    z_spacings = np.diff(grid[:, 0, 0, 2])

    assert np.allclose(x_spacings, x_spacings[0])
    assert np.allclose(y_spacings, y_spacings[0])
    assert np.allclose(z_spacings, z_spacings[0])


def test_2d_spacing_matches_dx_dz():
    """Test that spacing matches provided dx and dz."""
    dx, dz = 0.001, 0.0005

    grid = cartesian_pixel_grid(
        xlims=(-0.02, 0.02),
        zlims=(0.0, 0.06),
        dx=dx,
        dz=dz,
    )

    assert np.allclose(np.diff(grid[0, :, 0]), dx)
    assert np.allclose(np.diff(grid[:, 0, 2]), dz)


def test_3d_spacing_matches_dx_dy_dz():
    """Test that spacing matches provided dx, dy, and dz."""
    dx, dy, dz = 0.001, 0.0005, 0.00025

    grid = cartesian_pixel_grid(
        xlims=(-0.02, 0.02),
        zlims=(0.0, 0.06),
        ylims=(-0.01, 0.01),
        dx=dx,
        dy=dy,
        dz=dz,
    )

    assert np.allclose(np.diff(grid[0, :, 0, 0]), dx)
    assert np.allclose(np.diff(grid[0, 0, :, 1]), dy)
    assert np.allclose(np.diff(grid[:, 0, 0, 2]), dz)


# --- Edge Cases ---


@pytest.mark.parametrize(
    "xlims, expected_sign",
    [
        ((-0.05, -0.01), -1),  # Entirely negative
        ((0.01, 0.05), 1),  # Entirely positive
    ],
)
def test_2d_grid_various_xlim_signs(xlims, expected_sign):
    """Test 2D grid with various xlim sign combinations."""
    grid = cartesian_pixel_grid(
        xlims=xlims,
        zlims=(0.0, 0.06),
        grid_size_x=64,
        grid_size_z=128,
    )

    assert grid.shape == (128, 64, 3)
    if expected_sign < 0:
        assert np.all(grid[:, :, 0] < 0)
    else:
        assert np.all(grid[:, :, 0] > 0)


@pytest.mark.parametrize(
    "grid_size_x, grid_size_z",
    [
        (2, 3),  # Very small
        (512, 1024),  # Large
    ],
)
def test_2d_grid_various_sizes(grid_size_x, grid_size_z):
    """Test 2D grid with various grid sizes."""
    grid = cartesian_pixel_grid(
        xlims=(-0.02, 0.02),
        zlims=(0.0, 0.06),
        grid_size_x=grid_size_x,
        grid_size_z=grid_size_z,
    )
    assert grid.shape == (grid_size_z, grid_size_x, 3)


@pytest.mark.parametrize(
    "grid_size_x, grid_size_y, grid_size_z",
    [
        (2, 2, 3),  # Very small
        (64, 32, 128),  # Medium
    ],
)
def test_3d_grid_various_sizes(grid_size_x, grid_size_y, grid_size_z):
    """Test 3D grid with various grid sizes."""
    grid = cartesian_pixel_grid(
        xlims=(-0.02, 0.02),
        zlims=(0.0, 0.06),
        ylims=(-0.01, 0.01),
        grid_size_x=grid_size_x,
        grid_size_y=grid_size_y,
        grid_size_z=grid_size_z,
    )
    assert grid.shape == (grid_size_z, grid_size_x, grid_size_y, 3)
