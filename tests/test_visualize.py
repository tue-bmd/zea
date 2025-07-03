"""Tests for the zea.visualize module."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from zea.visualize import pad_or_crop_extent, plot_biplanes, plot_frustum_vertices, plot_quadrants


@pytest.fixture
def frustum_params():
    """Fixture to provide common parameters for frustum plotting."""
    return {
        "rho_range": (0, 1),
        "theta_range": (-0.7, 0.7),
        "phi_range": (-0.7, 0.7),
        "theta_plane": 0.3,
        "rho_plane": 0.5,
        "phi_plane": 0.5,
    }


@pytest.fixture
def quadrant_params():
    """Fixture to provide common parameters for quadrant plotting."""
    return {
        "nx": 70,
        "ny": 100,
        "nz": 50,
        "array": (np.mgrid[-1 : 1 : 1j * 70, -1 : 1 : 1j * 100, -1 : 1 : 1j * 50] ** 2).sum(0),
    }


def test_plot_frustum_vertices(frustum_params):
    """Tests the plot_frustum_vertices function."""
    fig, axs = plot_frustum_vertices(
        frustum_params["rho_range"],
        frustum_params["theta_range"],
        frustum_params["phi_range"],
        theta_plane=frustum_params["theta_plane"],
        rho_plane=frustum_params["rho_plane"],
        phi_plane=frustum_params["phi_plane"],
    )
    # Check that the function returns a figure and axes
    assert fig is not None, "Figure is None"
    assert axs is not None, "Axes are None"
    plt.close()


def test_plot_quadrants(quadrant_params):
    """Tests the plot_quadrants function."""
    nx, ny, nz = quadrant_params["nx"], quadrant_params["ny"], quadrant_params["nz"]
    array = quadrant_params["array"]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect(array.shape)
    plot_quadrants(ax, array, "x", cmap="viridis", slice_index=nx // 2)
    plot_quadrants(ax, array, "y", cmap="plasma", slice_index=ny // 2)
    plot_quadrants(ax, array, "z", cmap="inferno", slice_index=nz // 2)

    plt.close()


def test_plot_biplanes():
    """Tests the plot_biplanes function."""
    volume = np.random.rand(10, 10, 10)
    fig, ax = plot_biplanes(
        volume,
        cmap="gray",
        resolution=1.0,
        stride=1,
        slice_x=5,
        slice_y=5,
        slice_z=5,
    )
    assert fig is not None, "Figure is None"
    assert ax is not None, "Axis is None"
    plt.close()


def test_pad_or_crop_extent():
    """Tests the pad_or_crop_extent function."""

    # image_1.shape = (12, 10)
    image_1 = np.random.rand(12, 10)
    extent_1 = (-5, 5, -6, 6)  # (xmin, xmax, ymin, ymax)

    # image_2.shape = (14, 14)
    image_2 = np.pad(image_1, ((1, 1), (2, 2)), mode="constant", constant_values=0)
    extent_2 = np.array(extent_1) + np.array([-2, 2, -1, 1])

    image_1_to_2 = pad_or_crop_extent(image_1, extent_1, extent_2)
    image_2_to_1 = pad_or_crop_extent(image_2, extent_2, extent_1)

    np.testing.assert_array_equal(image_1_to_2, image_2)
    np.testing.assert_array_equal(image_2_to_1, image_1)
