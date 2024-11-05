"""Module with visualization functions for 2D and 3D ultrasound data.

- **Author(s)**     : Tristan Stevens
- **Date**          : 5/11/2024
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

from usbmd.display import frustum_convert_rtp2xyz


def plot_quadrants(ax, array, fixed_coord, cmap, slice_index, stride=1):
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

    n0, n1 = plane_data.shape
    quadrants = [
        plane_data[: n0 // 2, : n1 // 2],
        plane_data[: n0 // 2, n1 // 2 :],
        plane_data[n0 // 2 :, : n1 // 2],
        plane_data[n0 // 2 :, n1 // 2 :],
    ]

    min_val = np.nanmin(array)
    max_val = np.nanmax(array)

    cmap = plt.get_cmap(cmap)

    for i, quadrant in enumerate(quadrants):
        facecolors = cmap((quadrant - min_val) / (max_val - min_val))
        if fixed_coord == "x":
            Y, Z = np.mgrid[0 : ny // 2, 0 : nz // 2]
            X = (slice_index if slice_index is not None else nx // 2) * np.ones_like(Y)
            Y_offset = (i // 2) * ny // 2
            Z_offset = (i % 2) * nz // 2
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
            X, Z = np.mgrid[0 : nx // 2, 0 : nz // 2]
            Y = (slice_index if slice_index is not None else ny // 2) * np.ones_like(X)
            X_offset = (i // 2) * nx // 2
            Z_offset = (i % 2) * nz // 2
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
            X, Y = np.mgrid[0 : nx // 2, 0 : ny // 2]
            Z = (slice_index if slice_index is not None else nz // 2) * np.ones_like(X)
            X_offset = (i // 2) * nx // 2
            Y_offset = (i % 2) * ny // 2
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
):
    """
    Plots the vertices of a frustum in spherical coordinates and highlights specified planes.

    Args:
        rho_range (tuple): Range of rho values (min, max).
        theta_range (tuple): Range of theta values (min, max).
        phi_range (tuple): Range of phi values (min, max).
        num_points (int, optional): Number of points to generate along each edge. Defaults to 20.
        phi_plane (float, optional): Value of phi at which to plot a plane. Defaults to None.
        theta_plane (float, optional): Value of theta at which to plot a plane. Defaults to None.
        rho_plane (float, optional): Value of rho at which to plot a plane. Defaults to None.
        fig (matplotlib.figure.Figure, optional): Figure object to plot on.
            Defaults to None. Can be used to reuse the figure in a loop.
        ax (matplotlib.axes.Axes3DSubplot, optional): Axes object to plot on.
            Defaults to None. Can be used to reuse the axes in a loop.

    Returns:
        tuple: A tuple containing the figure and axes objects (fig, ax).

    Raises:
        ValueError: If no plane is specified (phi_plane, theta_plane, or rho_plane).
    """
    # Ensure only one plane is specified
    planes = [phi_plane, theta_plane, rho_plane]
    if sum(p is not None for p in planes) == 0:
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
    all_rho = []
    all_theta = []
    all_phi = []

    for edge in edges:
        rho_pts, theta_pts, phi_pts = generate_edge_points(edge[0], edge[1], num_points)
        all_rho.extend(rho_pts)
        all_theta.extend(theta_pts)
        all_phi.extend(phi_pts)

    # Convert all points to Cartesian coordinates
    x_all, y_all, z_all = frustum_convert_rtp2xyz(all_rho, all_theta, all_phi)
    z_all = -z_all  # Flip the z-axis

    # Determine axes limits
    x_min, x_max = np.min(x_all), np.max(x_all)
    y_min, y_max = np.min(y_all), np.max(y_all)
    z_min, z_max = np.min(z_all), np.max(z_all)

    # Plot the edges
    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = fig.add_subplot(111, projection="3d")

    def _plot_edges(edges, color, alpha=1.0, linestyle="-"):
        for edge in edges:
            rho_pts, theta_pts, phi_pts = generate_edge_points(
                edge[0], edge[1], num_points
            )
            x_edge, y_edge, z_edge = frustum_convert_rtp2xyz(
                rho_pts, theta_pts, phi_pts
            )
            z_edge = -z_edge  # Flip the z-axis
            ax.plot(
                x_edge,
                y_edge,
                z_edge,
                color=color,
                alpha=alpha,
                linestyle=linestyle,
            )

    # Plot frustum edges
    _plot_edges(edges, color=(102 / 255, 153 / 255, 255 / 255), alpha=0.5)

    # Plot plane edges
    if phi_plane is not None:
        # Edges along rho and theta at phi=phi_plane
        plane_edges = [
            (
                (rho_range[0], theta_range[0], phi_plane),
                (rho_range[1], theta_range[0], phi_plane),
            ),
            (
                (rho_range[0], theta_range[1], phi_plane),
                (rho_range[1], theta_range[1], phi_plane),
            ),
            (
                (rho_range[0], theta_range[0], phi_plane),
                (rho_range[0], theta_range[1], phi_plane),
            ),
            (
                (rho_range[1], theta_range[0], phi_plane),
                (rho_range[1], theta_range[1], phi_plane),
            ),
        ]
        plane_color = "y"
        _plot_edges(plane_edges, color=plane_color, linestyle="--")
    if theta_plane is not None:
        # Edges along rho and phi at theta=theta_plane
        plane_edges = [
            (
                (rho_range[0], theta_plane, phi_range[0]),
                (rho_range[1], theta_plane, phi_range[0]),
            ),
            (
                (rho_range[0], theta_plane, phi_range[1]),
                (rho_range[1], theta_plane, phi_range[1]),
            ),
            (
                (rho_range[0], theta_plane, phi_range[0]),
                (rho_range[0], theta_plane, phi_range[1]),
            ),
            (
                (rho_range[1], theta_plane, phi_range[0]),
                (rho_range[1], theta_plane, phi_range[1]),
            ),
        ]
        plane_color = "g"
        _plot_edges(plane_edges, color=plane_color, linestyle="--")
    if rho_plane is not None:
        # Edges along theta and phi at rho=rho_plane
        plane_edges = [
            (
                (rho_plane, theta_range[0], phi_range[0]),
                (rho_plane, theta_range[1], phi_range[0]),
            ),
            (
                (rho_plane, theta_range[0], phi_range[1]),
                (rho_plane, theta_range[1], phi_range[1]),
            ),
            (
                (rho_plane, theta_range[0], phi_range[0]),
                (rho_plane, theta_range[0], phi_range[1]),
            ),
            (
                (rho_plane, theta_range[1], phi_range[0]),
                (rho_plane, theta_range[1], phi_range[1]),
            ),
        ]
        plane_color = "r"
        _plot_edges(plane_edges, color=plane_color, linestyle="--")

    # Set consistent axes limits
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    # Remove background and axes
    ax.set_axis_off()

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    return fig, ax
