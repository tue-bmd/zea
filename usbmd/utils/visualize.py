"""Module with visualization functions for 2D and 3D ultrasound data.

- **Author(s)**     : Tristan Stevens
- **Date**          : 5/11/2024
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.ndimage import zoom
from usbmd.display import frustum_convert_rtp2xyz


def set_mpl_style(style: str = None) -> None:
    """Set the matplotlib style.

    Args:
        style (str, optional): Path to the matplotlib style file.
        Defaults to "usbmd_darkmode.mplstyle", which is the default
        darkmode style used throughout the USBMD toolbox.

    """
    if style is None:
        style = Path(__file__).parents[1] / "usbmd_darkmode.mplstyle"
    plt.style.use(style)


def plot_image_grid(
    images: List[np.ndarray],
    ncols: Optional[int] = None,
    cmap: Optional[Union[str, List[str]]] = "gray",
    vmin: Optional[Union[float, List[float]]] = None,
    vmax: Optional[Union[float, List[float]]] = None,
    titles: Optional[List[str]] = None,
    suptitle: Optional[str] = None,
    aspect: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    fig: Optional[plt.Figure] = None,
    fig_contents: Optional[List] = None,
    remove_axis: Optional[bool] = True,
    background_color: Optional[str] = "black",
    text_color: Optional[str] = "white",
    **kwargs,
) -> Tuple[plt.Figure, List]:
    """Plot a batch of images in a grid.

    Args:
        images (List[np.ndarray]): batch of images.
        ncols (int, optional): Number of columns. Defaults to None.
        cmap (str or list, optional): Colormap. Defaults to 'gray'.
            If list, cmap must be of same length as images and is set for each axis.
        vmin (float or list, optional): Minimum plot value. Defaults to None.
            if list vmin must be of same length as images and is set for each axis.
        vmax (float or list , optional): Maximum plot value. Defaults to None.
             if list vmax must be of same length as images and is set for each axis.
        titles (list, optional): List of titles for subplots. Defaults to None.
        suptitle (str, optional): Title for the plot. Defaults to None.
        aspect (optional): Aspect ratio for imshow.
        figsize (tuple, optional): Figure size. Defaults to None.
        fig (figure, optional): Matplotlib figure object. Defaults to None. Can
            be used to plot on an existing figure.
        fig_contents (list, optional): List of matplotlib image objects. Defaults to None.
        remove_axis (bool, optional): Whether to remove axis. Defaults to True. If
            False, the axis will be removed and the spines will be hidden, which allows
            for the labels to still be visible if plotted after the fact.
        background_color (str, optional): Background color. Defaults to None.
        **kwargs: arguments for plt.Figure.

    Returns:
        fig (figure): Matplotlib figure object
        fig_contents (list): List of matplotlib image objects.

    """
    if ncols is None:
        factors = [i for i in range(1, len(images) + 1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    nrows = int(len(images) / ncols) + int(len(images) % ncols)
    images = [images[i] if len(images) > i else None for i in range(nrows * ncols)]

    aspect_ratio = images[0].shape[1] / images[0].shape[0]
    if figsize is None:
        figsize = (ncols * 2, nrows * 2 / aspect_ratio)

    # either supply both fig and fig_contents or neither
    assert (fig is None) == (
        fig_contents is None
    ), "Supply both fig and fig_contents or neither"

    if fig is None:
        fig = plt.figure(figsize=figsize, **kwargs)
        axes = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.1)
        if background_color:
            fig.patch.set_facecolor(background_color)
        fig.set_tight_layout({"pad": 0.1})
    else:
        axes = fig.axes[: len(images)]

    if isinstance(cmap, str):
        cmap = [cmap] * len(images)
    else:
        if cmap is None:
            cmap = [None] * len(images)
        assert len(cmap) == len(
            images
        ), f"cmap must be a string or list of strings of length {len(images)}"

    if isinstance(vmin, (int, float)):
        vmin = [vmin] * len(images)
    else:
        if vmin is None:
            vmin = [None] * len(images)
        assert len(vmin) == len(
            images
        ), f"vmin must be a float or list of floats of length {len(images)}"

    if isinstance(vmax, (int, float)):
        vmax = [vmax] * len(images)
    else:
        if vmax is None:
            vmax = [None] * len(images)
        assert len(vmax) == len(
            images
        ), f"vmax must be a float or list of floats of length {len(images)}"

    if fig_contents is None:
        fig_contents = [None for _ in range(len(images))]
    for i, ax in enumerate(axes):
        image = images[i]
        if fig_contents[i] is None:
            im = ax.imshow(
                image, cmap=cmap[i], vmin=vmin[i], vmax=vmax[i], aspect=aspect
            )
            fig_contents[i] = im
        else:
            fig_contents[i].set_data(image)
        if remove_axis:
            ax.axis("off")
        else:
            for spine in ax.spines.values():
                # spine.set_visible(False)
                spine.set_color(background_color)
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
            )
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        if titles:
            ax.set_title(titles[i], color=text_color)

    if suptitle:
        fig.suptitle(suptitle, color=text_color)

    fig.set_tight_layout(False)
    # use bbox_inches="tight" for proper tight layout when saving
    return fig, fig_contents


def plot_quadrants(ax, array, fixed_coord, cmap, slice_index, stride=1, centroid=None):
    """
    For a given 3D array, plot a plane with fixed_coord using four individual quadrants.

    Args:
        ax (matplotlib.axes.Axes3DSubplot): The 3D axis to plot on.
        array (numpy.ndarray): The 3D array to be plotted.
        fixed_coord (str): The coordinate to be fixed ('x', 'y', or 'z').
        cmap (str): The colormap to be used for plotting.
        slice_index (int or None): The index of the slice to be plotted.
            If None, the middle slice is used.
        stride (int, optional): The stride step for plotting. Defaults to 1.
        centroid (tuple, optional): centroid around which to break the quadrants.
            If None, the middle of the image is used.

    Returns:
        matplotlib.axes.Axes3DSubplot: The axis with the plotted quadrants.
    """
    nx, ny, nz = array.shape
    index = {
        "x": (
            slice_index if slice_index is not None else nx // 2,
            slice(None),
            slice(None),
        ),
        "y": (
            slice(None),
            slice_index if slice_index is not None else ny // 2,
            slice(None),
        ),
        "z": (
            slice(None),
            slice(None),
            slice_index if slice_index is not None else nz // 2,
        ),
    }[fixed_coord]
    plane_data = array[index]

    if centroid is None:
        centroid = [x // 2 for x in array.shape]
    coords = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
    n0, n1 = (centroid[i] for i in coords[fixed_coord])
    quadrants = [
        plane_data[:n0, :n1],
        plane_data[:n0, n1:],
        plane_data[n0:, :n1],
        plane_data[n0:, n1:],
    ]

    min_val = np.nanmin(array)
    max_val = np.nanmax(array)

    cmap = plt.get_cmap(cmap)

    for i, quadrant in enumerate(quadrants):
        facecolors = cmap((quadrant - min_val) / (max_val - min_val))
        if fixed_coord == "x":
            Y, Z = np.mgrid[: quadrant.shape[0] + 1, : quadrant.shape[1] + 1]
            X = (slice_index if slice_index is not None else nx // 2) * np.ones_like(Y)
            Y_offset = (i // 2) * n0
            Z_offset = (i % 2) * n1
            ax.plot_surface(
                X,
                Y + Y_offset,
                Z + Z_offset,
                rstride=stride,
                cstride=stride,
                facecolors=facecolors,
                shade=False,
            )
        elif fixed_coord == "y":
            X, Z = np.mgrid[: quadrant.shape[0] + 1, : quadrant.shape[1] + 1]
            Y = (slice_index if slice_index is not None else ny // 2) * np.ones_like(X)
            X_offset = (i // 2) * n0
            Z_offset = (i % 2) * n1
            ax.plot_surface(
                X + X_offset,
                Y,
                Z + Z_offset,
                rstride=stride,
                cstride=stride,
                facecolors=facecolors,
                shade=False,
            )
        elif fixed_coord == "z":
            X, Y = np.mgrid[: quadrant.shape[0] + 1, : quadrant.shape[1] + 1]
            Z = (slice_index if slice_index is not None else nz // 2) * np.ones_like(X)
            X_offset = (i // 2) * n0
            Y_offset = (i % 2) * n1
            ax.plot_surface(
                X + X_offset,
                Y + Y_offset,
                Z,
                rstride=stride,
                cstride=stride,
                facecolors=facecolors,
                shade=False,
            )
    return ax


def plot_biplanes(
    volume,
    cmap="gray",
    resolution=1.0,
    stride=1,
    slice_x=None,
    slice_y=None,
    slice_z=None,
    show_axes=None,
    fig=None,
    ax=None,
):
    """
    Plot three intersecting planes from a 3D volume in 3D space.

    Also known as ultrasound biplane visualization.

    Args:
        volume (ndarray): 3D numpy array representing the volume to be plotted.
        cmap (str, optional): Colormap to be used for plotting. Defaults to "gray".
        resolution (float, optional): Resolution factor for the volume. Defaults to 1.0.
        stride (int, optional): Stride for plotting the quadrants. Defaults to 1.
        slice_x (int, optional): Index for the slice in the x-plane. Defaults to None.
        slice_y (int, optional): Index for the slice in the y-plane. Defaults to None.
        slice_z (int, optional): Index for the slice in the z-plane. Defaults to None.
        show_axes (dict, optional): Dictionary to specify axis labels and extents.
            Defaults to None.
        fig (matplotlib.figure.Figure, optional): Matplotlib figure object.
            Defaults to None. Can be used to reuse the figure in a loop.
        ax (matplotlib.axes.Axes3DSubplot, optional): Matplotlib 3D axes object.
            Defaults to None. Can be used to reuse the axes in a loop.

    Returns:
        tuple: A tuple containing the figure and axes objects (fig, ax).

    Raises:
        AssertionError: If none of slice_x, slice_y, or slice_z are provided.
    """

    assert (
        slice_x is not None or slice_y is not None or slice_z is not None
    ), "At least one slice index must be set."

    volume = zoom(volume, (resolution, resolution, resolution), order=1)

    # Adjust slice indices if resolution < 1
    if resolution < 1:
        if slice_x is not None:
            slice_x = int(slice_x * resolution)
        if slice_y is not None:
            slice_y = int(slice_y * resolution)
        if slice_z is not None:
            slice_z = int(slice_z * resolution)

    # volume is n_z, n_x, n_y -> n_x, n_y, n_z
    volume = np.transpose(volume, (1, 2, 0))
    volume = np.flip(volume, axis=2)  # Flip the z-axis

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(projection="3d")
        ax.set_box_aspect(volume.shape)
        # Remove background and axes faces
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    if slice_x is not None:
        plot_quadrants(ax, volume, "x", cmap=cmap, slice_index=slice_x, stride=stride)
    if slice_y is not None:
        plot_quadrants(ax, volume, "y", cmap=cmap, slice_index=slice_y, stride=stride)
    if slice_z is not None:
        plot_quadrants(ax, volume, "z", cmap=cmap, slice_index=slice_z, stride=stride)

    # Optionally show axes
    if show_axes:
        ax.set_xlabel(show_axes.get("x", ""))
        ax.set_ylabel(show_axes.get("y", ""))
        ax.set_zlabel(show_axes.get("z", ""))
        if "x_extent" in show_axes:
            ax.set_xticks(np.linspace(0, volume.shape[0], len(show_axes["x_extent"])))
            ax.set_xticklabels(show_axes["x_extent"])
        if "y_extent" in show_axes:
            ax.set_yticks(np.linspace(0, volume.shape[1], len(show_axes["y_extent"])))
            ax.set_yticklabels(show_axes["y_extent"])
        if "z_extent" in show_axes:
            ax.set_zticks(np.linspace(0, volume.shape[2], len(show_axes["z_extent"])))
            ax.set_zticklabels(show_axes["z_extent"])
    else:
        ax.set_axis_off()

    return fig, ax


def plot_frustum_vertices(
    rho_range,
    theta_range,
    phi_range,
    num_points=20,
    phi_plane=None,
    theta_plane=None,
    rho_plane=None,
    fig=None,
    ax=None,
    color_frustum="blue",
    phi_color="yellow",
    theta_color="green",
    rho_color="red",
):
    """
    Plots the vertices of a frustum in spherical coordinates and highlights specified planes.

    Args:
        rho_range (tuple): Range of rho values (min, max).
        theta_range (tuple): Range of theta values (min, max).
        phi_range (tuple): Range of phi values (min, max).
        num_points (int, optional): Number of points to generate along each edge. Defaults to 20.
        phi_plane (float or list, optional): Value(s) of phi at which to plot plane(s). Defaults to None.
        theta_plane (float or list, optional): Value(s) of theta at which to plot plane(s). Defaults to None.
        rho_plane (float or list, optional): Value(s) of rho at which to plot plane(s). Defaults to None.
        fig (matplotlib.figure.Figure, optional): Figure object to plot on.
            Defaults to None. Can be used to reuse the figure in a loop.
        ax (matplotlib.axes.Axes3DSubplot, optional): Axes object to plot on.
            Defaults to None. Can be used to reuse the axes in a loop.

    Returns:
        tuple: A tuple containing the figure and axes objects (fig, ax).

    Raises:
        ValueError: If no plane is specified (phi_plane, theta_plane, or rho_plane).
    """
    # Convert single values to lists
    phi_plane = [phi_plane] if isinstance(phi_plane, (int, float)) else phi_plane
    theta_plane = (
        [theta_plane] if isinstance(theta_plane, (int, float)) else theta_plane
    )
    rho_plane = [rho_plane] if isinstance(rho_plane, (int, float)) else rho_plane

    # Ensure at least one plane is specified
    if all(p is None for p in [phi_plane, theta_plane, rho_plane]):
        raise ValueError("At least one plane must be specified")

    # Define edges of the frustum
    edges = []

    # Edges along rho (vertical edges)
    for theta in theta_range:
        for phi in phi_range:
            edges.append(((rho_range[0], theta, phi), (rho_range[1], theta, phi)))

    # Edges along theta (near and far planes)
    for rho in rho_range:
        for phi in phi_range:
            edges.append(((rho, theta_range[0], phi), (rho, theta_range[1], phi)))

    # Edges along phi (near and far planes)
    for rho in rho_range:
        for theta in theta_range:
            edges.append(((rho, theta, phi_range[0]), (rho, theta, phi_range[1])))

    # Function to generate edge points
    def generate_edge_points(start, end, num_points):
        rho_points = np.linspace(start[0], end[0], num_points)
        theta_points = np.linspace(start[1], end[1], num_points)
        phi_points = np.linspace(start[2], end[2], num_points)
        return rho_points, theta_points, phi_points

    # Collect all points to determine axes limits
    all_points = []
    for edge in edges:
        rho_pts, theta_pts, phi_pts = generate_edge_points(edge[0], edge[1], num_points)
        x, y, z = frustum_convert_rtp2xyz(rho_pts, theta_pts, phi_pts)
        all_points.extend(zip(x, y, -z))  # Flip z-axis

    all_points = np.array(all_points)
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    z_min, z_max = np.min(all_points[:, 2]), np.max(all_points[:, 2])

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection="3d")

    def _plot_edges(edges, color, alpha=1.0, linestyle="-", **kwargs):
        for edge in edges:
            rho_pts, theta_pts, phi_pts = generate_edge_points(
                edge[0], edge[1], num_points
            )
            x, y, z = frustum_convert_rtp2xyz(rho_pts, theta_pts, phi_pts)
            ax.plot(x, y, -z, color=color, alpha=alpha, linestyle=linestyle, **kwargs)

    # Plot frustum edges
    _plot_edges(edges, color=color_frustum, alpha=1, lw=2)

    def get_plane_edges(plane_value, plane_type):
        """Generate edges for a specific plane type (phi, theta, or rho)"""
        if plane_type == "phi":
            return [
                (
                    (rho_range[0], theta_range[0], plane_value),
                    (rho_range[1], theta_range[0], plane_value),
                ),
                (
                    (rho_range[0], theta_range[1], plane_value),
                    (rho_range[1], theta_range[1], plane_value),
                ),
                (
                    (rho_range[0], theta_range[0], plane_value),
                    (rho_range[0], theta_range[1], plane_value),
                ),
                (
                    (rho_range[1], theta_range[0], plane_value),
                    (rho_range[1], theta_range[1], plane_value),
                ),
            ]
        elif plane_type == "theta":
            return [
                (
                    (rho_range[0], plane_value, phi_range[0]),
                    (rho_range[1], plane_value, phi_range[0]),
                ),
                (
                    (rho_range[0], plane_value, phi_range[1]),
                    (rho_range[1], plane_value, phi_range[1]),
                ),
                (
                    (rho_range[0], plane_value, phi_range[0]),
                    (rho_range[0], plane_value, phi_range[1]),
                ),
                (
                    (rho_range[1], plane_value, phi_range[0]),
                    (rho_range[1], plane_value, phi_range[1]),
                ),
            ]
        else:  # rho
            return [
                (
                    (plane_value, theta_range[0], phi_range[0]),
                    (plane_value, theta_range[1], phi_range[0]),
                ),
                (
                    (plane_value, theta_range[0], phi_range[1]),
                    (plane_value, theta_range[1], phi_range[1]),
                ),
                (
                    (plane_value, theta_range[0], phi_range[0]),
                    (plane_value, theta_range[0], phi_range[1]),
                ),
                (
                    (plane_value, theta_range[1], phi_range[0]),
                    (plane_value, theta_range[1], phi_range[1]),
                ),
            ]

    # Plot plane edges
    plane_configs = [
        (phi_plane, "phi", phi_color, "-"),
        (theta_plane, "theta", theta_color, "--"),
        (rho_plane, "rho", rho_color, "--"),
    ]

    for planes, plane_type, color, line in plane_configs:
        if planes is not None:
            for plane_value in planes:
                plane_edges = get_plane_edges(plane_value, plane_type)
                _plot_edges(plane_edges, color=color, linestyle=line)

    # Set axes properties
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    return fig, ax


def visualize_matrix(matrix, font_color="white", **kwargs):
    """
    Visualize a matrix with the values in each cell.
    """
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, **kwargs)
    fig.colorbar(cax)
    for (j, i), label in np.ndenumerate(matrix):
        ax.text(i, j, f"{label:.2f}", ha="center", va="center", color=font_color)
    return fig
