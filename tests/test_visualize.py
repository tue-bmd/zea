"""Tests for the visualize module."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from zea.visualize import (
    pad_or_crop_extent,
    plot_biplanes,
    plot_frustum_vertices,
    plot_image_grid,
    plot_quadrants,
    set_mpl_style,
    visualize_matrix,
)

from . import DEFAULT_TEST_SEED


# Use non-interactive backend for testing
matplotlib.use("Agg")


def random_images(n, shape=(10, 10)):
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    return [rng.standard_normal(shape) for _ in range(n)]


def random_volume(shape=(20, 20, 20)):
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    return rng.standard_normal(shape)


def assert_is_figure(obj):
    assert isinstance(obj, plt.Figure)


def assert_is_ax(obj):
    assert hasattr(obj, "plot")  # crude check for Axes3D


def make_3d_ax():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


@pytest.mark.parametrize(
    "images,kwargs",
    [
        (random_images(4), {}),
        (random_images(4), {"titles": ["Image 1", "Image 2", "Image 3", "Image 4"]}),
        (random_images(2), {"cmap": ["viridis", "plasma"]}),
        (random_images(2), {"vmin": [0, 0.2], "vmax": [0.8, 1.0]}),
    ],
)
def test_plot_image_grid(images, kwargs):
    expected_len = len(images)
    fig, fig_contents = plot_image_grid(images, **kwargs)
    assert_is_figure(fig)
    assert isinstance(fig_contents, list)
    assert len(fig_contents) == expected_len
    plt.close(fig)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"slice_x": 10, "slice_y": 10, "slice_z": 10},
        {"slice_x": 10},
        {"resolution": 0.5, "slice_x": 10},
    ],
)
def test_plot_biplanes(kwargs):
    volume = random_volume()
    fig, ax = plot_biplanes(volume, **kwargs)
    assert_is_figure(fig)
    assert_is_ax(ax)
    plt.close(fig)


def test_plot_biplanes_no_slice_raises():
    volume = random_volume()
    with pytest.raises(AssertionError):
        plot_biplanes(volume)


def test_plot_biplanes_reuse_fig_ax():
    volume = random_volume()
    fig, ax = make_3d_ax()
    fig_out, ax_out = plot_biplanes(volume, slice_x=10, fig=fig, ax=ax)
    assert fig_out is fig
    assert ax_out is ax
    plt.close(fig)


@pytest.mark.parametrize(
    "coord,cmap,kwargs",
    [
        ("x", "gray", {"slice_index": 10}),
        ("y", "viridis", {"slice_index": 10}),
        ("z", "plasma", {"slice_index": 10}),
        ("x", "gray", {"slice_index": None}),
        ("z", "gray", {"slice_index": 10, "stride": 2}),
        ("x", "gray", {"slice_index": 10, "centroid": [15, 15, 15]}),
        ("y", "gray", {"slice_index": 10, "alpha": 0.5, "antialiased": False}),
    ],
)
def test_plot_quadrants_variants(coord, cmap, kwargs):
    volume = random_volume()
    fig, ax = make_3d_ax()
    ax_out = plot_quadrants(ax, volume, coord, cmap, **kwargs)
    assert ax_out is ax
    assert len(ax.collections) == 4
    plt.close(fig)


_frustum_args = dict(
    rho_range=[0.1, 10],
    theta_range=[-0.6, 0.6],
    phi_range=[-0.6, 0.6],
)


@pytest.mark.parametrize(
    "kwargs,lines_min",
    [
        ({"phi_plane": 0}, 1),
        ({"theta_plane": 0.2}, 1),
        ({"rho_plane": 5.0}, 1),
        ({"phi_plane": [0, 0.3], "theta_plane": [0.2, -0.2], "rho_plane": [2.0, 5.0, 8.0]}, 1),
        (
            {
                "phi_plane": 0,
                "theta_plane": 0.2,
                "rho_plane": 5.0,
                "frustum_style": {"color": "blue", "linewidth": 1.5, "alpha": 0.6},
                "phi_style": {"color": "red", "linestyle": "--", "linewidth": 2},
                "theta_style": {"color": "green", "linestyle": ":", "alpha": 0.7},
                "rho_style": {"color": "yellow", "linestyle": "-.", "linewidth": 2.5},
            },
            10,
        ),
        ({"phi_plane": 0, "num_points": 50}, 1),
    ],
)
def test_plot_frustum_vertices_variants(kwargs, lines_min):
    args = dict(_frustum_args)
    args.update(kwargs)
    fig, ax = plot_frustum_vertices(**args)
    assert_is_figure(fig)
    assert_is_ax(ax)
    assert len(ax.lines) >= lines_min
    plt.close(fig)


def test_plot_frustum_no_plane_raises():
    with pytest.raises(ValueError, match="At least one plane must be specified"):
        plot_frustum_vertices(**_frustum_args)


def test_plot_frustum_reuse_fig_ax():
    fig, ax = make_3d_ax()
    args = dict(_frustum_args)
    args["phi_plane"] = 0
    fig_out, ax_out = plot_frustum_vertices(**args, fig=fig, ax=ax)
    assert fig_out is fig
    assert ax_out is ax
    plt.close(fig)


def test_set_mpl_style_default():
    set_mpl_style()  # Should not raise


@pytest.mark.parametrize(
    "matrix,kwargs",
    [
        (np.random.default_rng(DEFAULT_TEST_SEED).random((5, 5)), {}),
        (
            np.random.default_rng(DEFAULT_TEST_SEED).random((3, 3)),
            {"font_color": "black", "cmap": "viridis"},
        ),
    ],
)
def test_visualize_matrix_variants(matrix, kwargs):
    fig = visualize_matrix(matrix, **kwargs)
    assert_is_figure(fig)
    plt.close(fig)


@pytest.mark.parametrize(
    "image,extent,target_extent,shape_cmp",
    [
        (
            np.ones((10, 10)),
            (0, 10, 0, 10),
            (-5, 15, -5, 15),
            lambda r, i: r.shape[0] > i.shape[0] and r.shape[1] > i.shape[1],
        ),
        (
            np.ones((20, 20)),
            (0, 20, 0, 20),
            (5, 15, 5, 15),
            lambda r, i: r.shape[0] < i.shape[0] and r.shape[1] < i.shape[1],
        ),
        (
            np.ones((10, 10)),
            (5, 15, 5, 15),
            (0, 20, 3, 17),
            lambda r, i: isinstance(r, np.ndarray) and r.ndim == 2,
        ),
        (
            np.ones((10, 10)),
            (0, 10, 0, 10),
            (0, 10, 0, 10),
            lambda r, i: r.shape == i.shape and np.all(r == i),
        ),
    ],
)
def test_pad_or_crop_extent_variants(image, extent, target_extent, shape_cmp):
    result = pad_or_crop_extent(image, extent, target_extent)
    assert shape_cmp(result, image)
