"""Interactive region-of-interest (ROI) selection.

This module provides interactive tools for selecting regions of interest from 2D
arrays or images displayed with matplotlib. It is designed for ultrasound and image
processing workflows where manual or semi-automatic selection of regions is required.

Key features
------------
- Interactive selection with a rectangle or lasso tool, via matplotlib widgets.
- Selecting and confirming both happen in the plot window; no tkinter required.
- Cropping, masking and extracting the selected regions from images.
- Polygon and rectangle extraction, interpolation and mask reconstruction.
- Mask interpolation across the frames of a sequence, plus animation of the result.
- Metric computation (e.g. gCNR) between two selected patches.
- Reading and writing zea HDF5 files, storing the annotations as a
  :class:`~zea.data.spec.Segmentation` map alongside the images.

Command line interface
----------------------

The module is exposed through the ``zea`` CLI as ``zea tools select``::

    zea tools select                              # ask for the file paths on the terminal
    zea tools select frame.png other.png          # compare two images with gCNR
    zea tools select clip.mp4 --num-selections 3  # annotate a video and interpolate

Run ``zea tools select --help`` for all options. Any option that is omitted is asked
for interactively, so the command can be used without arguments as well.

Annotating a zea dataset
------------------------

Any zea file with image data (``data/image``) can be annotated directly, including
files on the Hugging Face Hub. For example, on a CAMUS recording::

    zea tools select \\
        hf://zeahub/camus/val/patient0409/patient0409_4CH_half_sequence.hdf5 \\
        --selector lasso --title lv_endo --num-selections 3 --fps 20

Draw the left-ventricle border in each of the three key frames and press ``enter`` to
keep it. The masks are interpolated across all frames and written to
``patient0409_4CH_half_sequence_lv_endo_annotations.hdf5`` (plus a ``.gif``
preview) in the working directory. The result is a regular zea file, so it reads back
like any other dataset::

    from zea import File

    with File("patient0409_4CH_half_sequence_lv_endo_annotations.hdf5") as file:
        images = file.data.image.values[:]              # (n_frames, H, W)
        masks = file.data.segmentation.values[..., 0]   # (n_frames, H, W), bool
        labels = file.data.segmentation.labels[:]       # ["lv_endo"]

Since the tool only produces images and segmentations, the warnings about the
acquisition fields it cannot fill in (scan parameters, probe geometry, …) are
suppressed when saving.

Python API
----------

.. doctest::

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from zea.tools.selection_tool import interactive_selector

    >>> image = np.zeros((100, 100))  # Load your 2D image array
    >>> fig, ax = plt.subplots()
    >>> _ = ax.imshow(image, cmap="gray")
    >>> patches, masks = interactive_selector(image, ax, selector="rectangle")  # doctest: +SKIP

"""

import re
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as pltPath
from matplotlib.widgets import LassoSelector, RectangleSelector
from PIL import Image, ImageDraw
from scipy.interpolate import interp1d
from skimage.measure import approximate_polygon, find_contours
from sklearn.metrics import pairwise_distances

from zea import log
from zea.func.tensor import translate
from zea.internal.preset_utils import HF_PREFIX, _hf_resolve_path
from zea.internal.viewer import get_matplotlib_figure_props, move_matplotlib_figure
from zea.io_lib import (
    _SUPPORTED_IMG_TYPES,
    _SUPPORTED_VID_TYPES,
    _SUPPORTED_ZEA_TYPES,
    load_image,
    load_video,
)
from zea.visualize import plot_rectangle_from_mask, plot_shape_from_mask

#: Selection tools that can be used to draw a region of interest.
SELECTORS = ("rectangle", "lasso")


def crop_array(array, value=None):
    """Crop an array to remove all rows and columns containing only a given value.

    Args:
        array (ndarray): 2D input array.
        value: Value that marks a row/column as empty. With the default (``None``)
            nothing matches and the array is returned unchanged.

    Returns:
        np.ndarray: The cropped 2D array.
    """
    array = np.array(array)
    assert array.ndim == 2, f"Array must be 2D, not {array.ndim}D."
    mask = np.all(np.equal(array, value), axis=1)  # ty: ignore[no-matching-overload]
    array = array[~mask]

    mask = np.all(np.equal(array, value), axis=0)  # ty: ignore[no-matching-overload]
    array = array[:, ~mask]
    return array


def interactive_selector(
    data,
    ax,
    selector: str = "rectangle",
    extent: list | None = None,
    verbose: bool = True,
    num_selections: int | None = None,
    confirm_selection: bool = True,
) -> tuple:
    """Interactively select part of an array displayed as an image with matplotlib.

    Args:
        data (ndarray): Input array, must be 2D.
        ax (matplotlib.axes.Axes): Existing matplotlib axis to select a region on.
        selector (str, optional): Type of selector, one of :data:`SELECTORS`.
            Defaults to ``"rectangle"``. ``"lasso"`` uses matplotlib's
            ``LassoSelector``, ``"rectangle"`` its ``RectangleSelector``.
        extent (list, optional): Extent of the axis the selection is made on. Used to
            transform coordinates back to pixel values. Defaults to None.
        verbose (bool, optional): Whether to log progress messages. Defaults to True.
        num_selections (int, optional): Number of selections to make. When omitted the
            user presses Enter in the plot window to signal they are done.
        confirm_selection (bool, optional): Whether to ask (in the plot window) to
            confirm the selection before returning. Defaults to True.

    Returns:
        tuple: ``(patches, masks)``, where ``patches`` is a list of the selected parts
        of ``data`` and ``masks`` a list of the corresponding boolean masks.
    """
    assert data.ndim == 2, f"Data must be 2D, not {data.ndim}D."
    assert selector in SELECTORS, f"Selector must be one of {SELECTORS}, not {selector!r}."

    x, y = np.meshgrid(np.arange(data.shape[1], dtype=int), np.arange(data.shape[0], dtype=int))
    pix = np.vstack((x.flatten(), y.flatten())).T

    def _translate_coordinates(x, y):
        if extent:
            x = translate(x, (extent[0], extent[1]), (0, data.shape[1]))
            y = translate(y, (extent[2], extent[3]), (0, data.shape[0]))
        return x, y

    def _onselect_lasso(verts):
        nonlocal select_idx
        if verbose:
            log.info(f"Selection {select_idx} done")
        select_idx += 1
        verts = np.array(verts)
        # if axis is drawn with extent argument, first translate coordinates to pixels
        verts = np.array(_translate_coordinates(*verts.T)).T
        p = pltPath(verts)
        ind = p.contains_points(pix, radius=1)
        mask.flat[ind] = True
        masks.append(np.copy(mask))
        mask.flat[ind] = False

    def _onselect_rectangle(start, end):
        nonlocal select_idx
        if verbose:
            log.info(f"Selection {select_idx} done")
        select_idx += 1
        # if axis is drawn with extent argument, first translate coordinates to pixels
        start.xdata, start.ydata = _translate_coordinates(start.xdata, start.ydata)
        end.xdata, end.ydata = _translate_coordinates(end.xdata, end.ydata)

        verts = np.array(
            [
                [start.xdata, start.ydata],
                [start.xdata, end.ydata],
                [end.xdata, end.ydata],
                [end.xdata, start.ydata],
            ],
            int,
        )
        p = pltPath(verts)
        ind = p.contains_points(pix, radius=1)
        mask.flat[ind] = True
        masks.append(np.copy(mask))
        mask.flat[ind] = False

    name_to_selector = {"lasso": LassoSelector, "rectangle": RectangleSelector}
    selector_cls = name_to_selector[selector]
    onselect_dict = {
        LassoSelector: _onselect_lasso,
        RectangleSelector: _onselect_rectangle,
    }
    kwargs_dict = {LassoSelector: {}, RectangleSelector: {"interactive": True}}

    # Selection state, shared with the callbacks above.
    mask = np.tile(False, data.shape)
    masks = []
    select_idx = 0

    def _execute_selector():
        """Run one round of selecting and return the patches it produced."""
        nonlocal mask, masks, select_idx
        mask = np.tile(False, data.shape)
        masks = []
        select_idx = 0

        widget = selector_cls(
            ax,
            onselect_dict[selector_cls],  # ty: ignore[invalid-argument-type]
            **kwargs_dict[selector_cls],  # ty: ignore[invalid-argument-type]
        )

        if num_selections:
            if verbose:
                log.info(f"...Plot will close after {num_selections} selections...")
            plt.show(block=False)
            figure = ax.get_figure()
            while select_idx < num_selections:
                if not plt.fignum_exists(figure.number):
                    log.warning(
                        f"Plot was closed after {select_idx} of {num_selections} selections."
                    )
                    break
                plt.pause(0.1)
        else:
            plt.show(block=False)
            wait_for_key(
                ax.get_figure(),
                f"Press {_keys(ACCEPT_KEYS)} in this window when you are done selecting.",
            )

        widget.disconnect_events()
        widget.set_visible(False)
        widget.update()

        return [crop_array(data * selected, value=0) for selected in masks]

    patches = _execute_selector()

    if not confirm_selection:
        return patches, masks

    while masks:
        for current_mask in masks:
            plot_mask(ax, current_mask, selector)
        plt.draw()

        if confirm_in_figure(ax.get_figure(), len(patches)):
            return patches, masks

        remove_masks_from_axs(ax)
        patches = _execute_selector()

    return patches, masks


def interactive_selector_with_plot_and_metric(
    data,
    ax=None,
    selector: str = "rectangle",
    metric: str | None = None,
    cmap: str = "gray",
    plot: bool = True,
    mask_plot: bool = False,
    selection_axis: int = 0,
    **kwargs,
):
    """Select two regions in one image and compare them across a list of images.

    The selection is made on a single image (``data[selection_axis]``) and the resulting
    masks are applied to every image in ``data``, so the same two regions are compared
    in each of them.

    Args:
        data (ndarray or list of ndarray): Input data.
        ax (matplotlib.axes.Axes or list, optional): Axis (or axes) corresponding to the
            input data. Defaults to None, in which case the data is plotted first to
            create the axes.
        selector (str, optional): Type of selection tool, one of :data:`SELECTORS`.
            Defaults to ``"rectangle"``.
        metric (str, optional): Name of a metric in :mod:`zea.metrics` to compute between
            the two patches (e.g. ``"gcnr"``). Defaults to None, i.e. no metric.
        cmap (str, optional): Colormap to display the data in. Defaults to ``"gray"``.
        plot (bool, optional): Whether to plot the selections / metrics on top of the
            axes. Defaults to True.
        mask_plot (bool, optional): Whether to also plot the masks in a separate figure.
            Can be useful to isolate the patches and see the selections more clearly.
            Defaults to False.
        selection_axis (int, optional): Index of the image the selection is made on.
            Defaults to 0.
        **kwargs: Forwarded to :func:`interactive_selector`.

    Returns:
        list: The computed metric scores, one per image in ``data``. Empty when
        ``metric`` is None.

    Raises:
        ValueError: If the user did not make exactly two selections. More or fewer
            patches don't make sense in this context.
    """
    if not isinstance(data, list):
        data = [data]

    if ax is None:
        _, ax = plt.subplots(1, len(data))
        for _data, _ax in zip(data, np.atleast_1d(ax)):
            _ax.imshow(_data, cmap=cmap, aspect="auto")

    if not isinstance(ax, Iterable):
        ax = [ax]

    # create selector for first axis only
    _, masks = interactive_selector(
        data[selection_axis], ax[selection_axis], selector, num_selections=2, **kwargs
    )

    if len(masks) != 2:
        raise ValueError("exactly 2 patches are required for using this wrapper function")

    # get patches for all data in data list using the selection made
    patches = []
    for image in data:
        patches.extend([crop_array(image * mask, value=0) for mask in masks])

    # compute metrics
    scores = []
    if metric:
        from zea.metrics import get_metric

        for i in range(len(data)):
            idx = i * len(masks)
            score = get_metric(metric)(patches[idx], patches[idx + 1])
            scores.append(score)
            log.info(f"{metric}: {score:.3f}")

    # plot on top of existing plot
    if plot:
        for i, _ax in enumerate(ax):
            for mask in masks:
                plot_mask(_ax, mask, selector)
            if i < len(scores):
                _ax.set_title(f"{_ax.get_title()}\n{metric}: {scores[i]:.3f}")
        plt.tight_layout()

    # plot patches and masks
    if mask_plot:
        fig, axs = plt.subplots(len(masks), 3)
        for i, (ax_new, patch, mask) in enumerate(zip(axs, patches, masks)):
            if i == 0:
                ax_base = ax_new[selection_axis]
                ax_base.imshow(data[selection_axis], cmap=cmap, aspect="auto")
            ax_new[1].imshow(patch, cmap=cmap, aspect="auto")
            ax_new[2].imshow(mask, aspect="auto")

            plot_mask(ax_base, mask, selector)

            for _ax in ax_new:
                _ax.axis("off")

        fig.tight_layout()

    return scores


def extract_rectangle_from_mask(image):
    """Find the corner points of the rectangle in a binary mask.

    Args:
        image (np.ndarray): 2D binary mask.

    Returns:
        tuple | None: ``((x1, y1), (x2, y2))`` with the corner points of the rectangle,
        or None when the mask is empty.
    """
    image = np.array(image)
    indices = np.argwhere(image == 1)
    if len(indices) == 0:
        return None
    top, left = indices.min(axis=0)
    bottom, right = indices.max(axis=0)
    return ((left, top), (right, bottom))


def reconstruct_mask_from_rectangle(corner_points, image_shape):
    """Reconstruct a binary mask from corner points of a rectangle.

    Args:
        corner_points (tuple): Tuple of the form ``((x1, y1), (x2, y2))``
            with the corner points of the rectangle.
        image_shape (tuple): Size of the image (height, width).

    Returns:
        np.ndarray: 2D boolean mask of shape (height, width).

    """
    image = np.zeros(image_shape, dtype=bool)
    x1, y1 = corner_points[0]
    x2, y2 = corner_points[1]
    image[y1 : y2 + 1, x1 : x2 + 1] = True
    return image


def interpolate_rectangles(rectangles, positions, frames):
    """Interpolate between an arbitrary number of rectangles.

    Args:
        rectangles (list): List with any number of rectangles as tuples of the form
            ``((x1, y1), (x2, y2))``. Its length must equal the number of positions.
        positions (np.ndarray): Frame index each rectangle sits on.
        frames (np.ndarray): Frame indices to interpolate onto.

    Returns:
        list: Interpolated rectangles as tuples of the form ``((x1, y1), (x2, y2))``,
        one per entry in ``frames``.
    """
    new_rectangles = []
    x1 = [rect[0][0] for rect in rectangles]
    x2 = [rect[1][0] for rect in rectangles]
    y1 = [rect[0][1] for rect in rectangles]
    y2 = [rect[1][1] for rect in rectangles]

    values_interp = []
    for values in [x1, x2, y1, y2]:
        values_interp.append(np.interp(frames, positions, values).astype(np.int32))

    x1, x2, y1, y2 = values_interp
    new_rectangles = [((x1[i], y1[i]), (x2[i], y2[i])) for i in range(len(x1))]
    return new_rectangles


def extract_polygon_from_mask(mask, tolerance: float = 0.01, verbose: bool = True):
    """Find the largest contour in a binary mask and fit a polygon to it.

    Polygon approximation will reduce the number of contour points, unless ``tolerance``
    is 0.

    Args:
        mask (np.ndarray): 2D binary mask.
        tolerance (float, optional): Approximation tolerance for the polygonal contour.
            Defaults to 0.01.
        verbose (bool, optional): Whether to warn when zero or multiple contours are
            found. Defaults to True.

    Returns:
        np.ndarray | None: Array of shape (N, 2) with the vertices of the polygon, or
        None when the mask contains no contour.
    """
    contours = find_contours(mask, 0.5, fully_connected="high")
    # return the largest contour
    if len(contours) > 1:
        contour_lengths = [len(contour) for contour in contours]
        contour = contours[np.argmax(contour_lengths)]
        if verbose:
            log.warning("Multiple contours found. Returning the largest contour.")
    elif len(contours) == 0:
        if verbose:
            log.warning("No contours found. Returning None.")
        return None
    else:
        contour = contours[0]
    poly = approximate_polygon(contour, tolerance)
    return poly


def reconstruct_mask_from_polygon(vertices, image_size):
    """Reconstruct a binary mask from a polygon.

    Fills in the region defined by the polygon contour.

    Args:
        vertices (np.ndarray): Vertices of the polygon as an array of shape (N, 2).
        image_size (tuple): Size of the image (height, width).

    Returns:
        np.ndarray: Array of shape (height, width) with the reconstructed mask.
    """
    # Create a path for the polygon
    mask = Image.new("L", (image_size[1], image_size[0]), 0)

    # Create a draw object
    draw = ImageDraw.Draw(mask)

    # Close the polygon by adding the first point to the end
    vertices = np.vstack((vertices, vertices[0]))

    # Draw the filled polygon on the mask
    polygon_coords = [(x, y) for y, x in vertices]
    draw.polygon(polygon_coords, outline=1, fill=1)

    # Convert the mask to a NumPy array
    mask_array = np.array(mask)
    return mask_array


def interpolate_polygons(polygon1, polygon2, t):
    """Interpolate between two polygons.

    Args:
        polygon1 (np.ndarray): First polygon as an array of shape (N, 2).
        polygon2 (np.ndarray): Second polygon as an array of shape (N, 2).
        t (float): Interpolation parameter, where ``0 <= t <= 1``.

    Returns:
        np.ndarray: Interpolated polygon as an array of shape (N, 2).

    Raises:
        ValueError: If the polygons do not have the same number of vertices.
    """
    # Ensure both polygons have the same number of vertices
    if polygon1.shape[0] != polygon2.shape[0]:
        raise ValueError("Both polygons must have the same number of vertices.")

    # Perform linear interpolation for each vertex
    interpolated_polygon = (1 - t) * polygon1 + t * polygon2

    return interpolated_polygon


def match_polygons(polygon1, polygon2):
    """Match two polygons by minimizing the total distance between their vertices.

    The vertices of the first polygon are shifted circularly to find the best match.
    The order of the vertices is preserved.

    Args:
        polygon1 (np.ndarray): First polygon as an array of shape (N, 2).
        polygon2 (np.ndarray): Second polygon as an array of shape (N, 2).

    Returns:
        tuple: ``(poly1, poly2)``, the matched polygons.
    """

    distances = pairwise_distances(polygon1, polygon2, metric="euclidean")

    min_total_distance = float("inf")
    best_shift = 0

    # Find the shift that minimizes the total distance.
    n, m = distances.shape
    for shift in range(n):
        total_distance = 0
        for i in range(n):
            total_distance += distances[i, (i + shift) % m]
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_shift = shift

    polygon1 = np.roll(polygon1, best_shift, axis=0)
    return polygon1, polygon2


def equalize_polygons(polygons, mode: str = "max"):
    """Make sure all polygons have the same number of vertices.

    Args:
        polygons (list): List with any number of polygons as arrays of shape (N, 2).
        mode (str, optional): Method for equalizing the number of vertices, either
            ``"max"`` (match the polygon with the most vertices, by interpolation) or
            ``"min"`` (match the polygon with the fewest vertices, by subsampling).
            Defaults to ``"max"``.

    Returns:
        list: The polygons, all with the same number of vertices.
    """
    assert mode in ("max", "min"), f"Mode must be either 'max' or 'min', not {mode}."
    sizes = [polygon.shape[0] for polygon in polygons]
    num_vertices = max(sizes) if mode == "max" else min(sizes)

    if num_vertices < 0.8 * max(sizes):
        log.warning(
            "Difference in number of vertices is large. "
            "Possibly due to large difference in polygon size."
        )

    if mode == "min":
        # subsample the contours
        trimmed_polygons = []
        for polygon in polygons:
            indices = np.linspace(0, len(polygon) - 1, num_vertices).astype(int)
            trimmed_polygons.append(polygon[indices])
        return trimmed_polygons

    # interpolate the contours
    interpolated_polygons = []
    for polygon in polygons:
        if polygon.shape[0] < num_vertices:
            indices = np.linspace(0, len(polygon) - 1, num_vertices)

            # create a function to interpolate the x and y coordinates separately
            f_x = interp1d(np.arange(len(polygon)), polygon[:, 0], kind="linear")
            f_y = interp1d(np.arange(len(polygon)), polygon[:, 1], kind="linear")

            # evaluate the functions at the interpolated indices
            interpolated_polygons.append(np.column_stack((f_x(indices), f_y(indices))))
        else:
            interpolated_polygons.append(polygon)
    return interpolated_polygons


def interpolate_masks(
    masks: list | np.ndarray,
    num_frames: int,
    rectangle: bool = False,
    positions: Sequence[int] | None = None,
) -> list:
    """Interpolate between an arbitrary number of masks.

    Args:
        masks (list or np.ndarray): At least two binary masks of equal shape.
        num_frames (int): Number of masks to interpolate to.
        rectangle (bool, optional): Whether the masks are rectangular, in which case the
            faster rectangle interpolation is used instead of polygon interpolation.
            Defaults to False.
        positions (Sequence[int], optional): Frame index each mask belongs to, strictly
            increasing. Defaults to None, i.e. spread the masks evenly over the frames.
            Frames outside the range hold on to the nearest mask.

    Returns:
        list: ``num_frames`` interpolated masks.
    """
    assert isinstance(masks, (list, np.ndarray)), "Masks must be a list of numpy arrays."
    assert num_frames > 1, "At least two frames are required for interpolation."
    number_of_masks = len(masks)
    assert number_of_masks > 1, "At least two masks are required for interpolation."
    mask_shape = masks[0].shape
    assert all(mask.shape == mask_shape for mask in masks), "All masks must have the same shape."

    if positions is None:
        frame_positions = np.linspace(0, num_frames - 1, number_of_masks)
    else:
        frame_positions = np.asarray(positions, dtype=float)
        assert len(frame_positions) == number_of_masks, "One position per mask is required."
        assert np.all(np.diff(frame_positions) > 0), "Positions must be strictly increasing."
        assert (
            np.all(frame_positions == np.floor(frame_positions))
            and frame_positions[0] >= 0
            and frame_positions[-1] < num_frames
        ), f"Positions must be integer frame indices in [0, {num_frames})."

    frames = np.arange(num_frames)

    if rectangle:
        # get the rectangles
        rectangles = []
        for mask in masks:
            rectangles.append(extract_rectangle_from_mask(mask))

        # np.interp holds the outer rectangles for frames outside the position range
        rectangles = interpolate_rectangles(rectangles, frame_positions, frames)

        # reconstruct the masks
        interpolated_masks = []
        for _rectangle in rectangles:
            interpolated_masks.append(reconstruct_mask_from_rectangle(_rectangle, mask_shape))
        return interpolated_masks
    # get the contours
    polygons = []
    for mask in masks:
        polygons.append(extract_polygon_from_mask(mask))

    # trim the polygons for equal number of vertices
    polygons = equalize_polygons(polygons)

    # roll each polygon onto the previous, already-fixed one; matching pairwise in
    # both directions would undo the alignment of the segment before it
    for i in range(number_of_masks - 1):
        polygons[i + 1], _ = match_polygons(polygons[i + 1], polygons[i])

    # interpolate the polygons, holding the outer ones outside the position range
    interpolated_polygons = []
    for frame in frames:
        segment = int(
            np.clip(
                np.searchsorted(frame_positions, frame, side="right") - 1,
                0,
                len(frame_positions) - 2,
            )
        )
        start, end = frame_positions[segment], frame_positions[segment + 1]
        t = float(np.clip((frame - start) / (end - start), 0.0, 1.0))
        interpolated_polygons.append(
            interpolate_polygons(polygons[segment], polygons[segment + 1], t)
        )

    # reconstruct the masks
    interpolated_masks = []
    for interpolated_polygon in interpolated_polygons:
        interpolated_masks.append(reconstruct_mask_from_polygon(interpolated_polygon, mask_shape))

    return interpolated_masks


def plot_mask(ax: matplotlib.axes.Axes, mask: np.ndarray, selector: str = "rectangle", **kwargs):
    """Draw a mask on an axis the way its selector drew it.

    Args:
        ax (matplotlib.axes.Axes): Axis to draw on.
        mask (np.ndarray): 2D boolean mask.
        selector (str, optional): One of :data:`SELECTORS`. ``"rectangle"`` draws the
            bounding box, anything else the mask's own outline. Defaults to
            ``"rectangle"``.
        **kwargs: Forwarded to the underlying plotting function. ``alpha`` defaults to
            0.5 so the image stays visible underneath.

    Returns:
        The matplotlib patch(es) that were added, or None for an empty rectangle mask.
    """
    kwargs.setdefault("alpha", 0.5)
    if selector == "rectangle":
        return plot_rectangle_from_mask(ax, mask, **kwargs)
    return plot_shape_from_mask(ax, mask, **kwargs)


def remove_masks_from_axs(axs: matplotlib.axes.Axes) -> None:
    """Remove all mask patches from the given axes object."""
    for obj in axs.findobj():
        if isinstance(obj, (PathPatch, Rectangle)):
            try:
                obj.remove()
            except Exception:
                pass


def update_imshow_with_mask(
    frame_no: int,
    axs: matplotlib.axes.Axes,
    imshow_obj: matplotlib.image.AxesImage,
    images: np.ndarray,
    masks: np.ndarray,
    selector: str,
    **kwargs,
) -> tuple:
    """Update an imshow object with one frame and overlay the corresponding mask.

    This function is designed for animation where each frame has one associated mask.
    It removes any existing masks from the axes before plotting the new one.

    Args:
        frame_no (int): The index of the frame to display.
        axs (matplotlib.axes.Axes): The axes object to display the image on.
        imshow_obj (matplotlib.image.AxesImage): The imshow object to update.
        images (numpy.ndarray): An array of images with shape (num_frames, height, width).
        masks (numpy.ndarray): An array of masks with shape (num_frames, height, width),
            where each mask corresponds to one frame in the images array.
        selector (str): The type of selector used, one of :data:`SELECTORS`. Rectangles
            are drawn as a bounding box, anything else as an arbitrary shape.
        **kwargs: Forwarded to the plotting function.

    Returns:
        tuple: The updated imshow object and the mask object (the matplotlib patch that
        was plotted).
    """
    imshow_obj.set_array(images[frame_no])
    remove_masks_from_axs(axs)
    mask_obj = plot_mask(axs, masks[frame_no], selector, **kwargs)
    return imshow_obj, mask_obj


# ── In-figure prompts ─────────────────────────────────────────────────────────
#
# Confirming happens in the plot window rather than through a dialog: it keeps the
# user's hands where the selecting happens, and needs no tkinter.

#: Keys that accept what is currently shown. Closing the window accepts too.
ACCEPT_KEYS = ("enter", "y")
#: Keys that discard the current selection and start over. Deliberately outside
#: matplotlib's default keymap ('r' is "reset view", 'q' closes the window, ...).
REDO_KEYS = ("n", "escape")

#: Style of the banner drawn under a figure, chosen to stand out against both a light
#: figure background and the grayscale images it sits under.
_BANNER_STYLE = {
    "ha": "center",
    "va": "bottom",
    "fontsize": "large",
    "fontweight": "bold",
    "color": "black",
    "bbox": {
        "boxstyle": "round,pad=0.5",
        "facecolor": "gold",
        "edgecolor": "black",
        "linewidth": 1.5,
    },
}


def show_status(fig, message: str, banner=None):
    """Show ``message`` in a highlighted banner under a figure, and paint it right away.

    Args:
        fig (matplotlib.figure.Figure): Figure to draw the banner on.
        message (str): Text to show.
        banner (matplotlib.text.Text, optional): Banner returned by an earlier call,
            which is replaced. Defaults to None, i.e. draw a new one.

    Returns:
        matplotlib.text.Text: The banner, to pass back in or to ``remove()``.
    """
    log.info(message)
    if banner is not None:
        banner.remove()
    banner = fig.text(0.5, 0.015, message, **_BANNER_STYLE)
    if plt.fignum_exists(fig.number):
        fig.canvas.draw_idle()
        plt.pause(0.001)  # let the backend paint before we go back to work
    return banner


def wait_for_key(
    fig, message: str, accept: Sequence[str] = ACCEPT_KEYS, redo: Sequence[str] = ()
) -> bool:
    """Show ``message`` under a figure and block until the user presses a listed key.

    Args:
        fig (matplotlib.figure.Figure): Figure to listen on and write the message under.
        message (str): Instruction shown to the user, e.g. which keys to press.
        accept (Sequence[str], optional): Keys that return True. Defaults to
            :data:`ACCEPT_KEYS`.
        redo (Sequence[str], optional): Keys that return False. Defaults to none, i.e.
            the prompt can only be accepted.

    Returns:
        bool: True when an ``accept`` key was pressed (or the window was closed), False
        for a ``redo`` key.
    """
    decision = {}

    def _on_key(event):
        if event.key in accept:
            decision["accept"] = True
        elif event.key in redo:
            decision["accept"] = False

    # Connect before the banner: showing it flushes pending events, which would
    # otherwise swallow a keypress that arrived first.
    cid = fig.canvas.mpl_connect("key_press_event", _on_key)
    banner = show_status(fig, message)
    try:
        while "accept" not in decision:
            # A closed window means the user is done; take what we have.
            if not plt.fignum_exists(fig.number):
                return True
            plt.pause(0.1)
    finally:
        fig.canvas.mpl_disconnect(cid)
        banner.remove()
        if plt.fignum_exists(fig.number):
            fig.canvas.draw_idle()
    return decision["accept"]


def confirm_in_figure(fig, num_selections: int) -> bool:
    """Ask, in the plot window, whether to keep the selection that is drawn on it.

    Args:
        fig (matplotlib.figure.Figure): Figure showing the selection.
        num_selections (int): Number of selections that were made.

    Returns:
        bool: True to keep the selection, False to redo it.
    """
    return wait_for_key(
        fig,
        f"{num_selections} selection(s) made. Press {_keys(ACCEPT_KEYS)} to keep, "
        f"{_keys(REDO_KEYS)} to redo.",
        accept=ACCEPT_KEYS,
        redo=REDO_KEYS,
    )


def _keys(keys: Sequence[str]) -> str:
    """Render key names for a prompt, e.g. ``"'enter'/'y'"``."""
    return "/".join(f"'{key}'" for key in keys)


# ── Terminal prompts ──────────────────────────────────────────────────────────
#
# These are only used for options that were not passed on the command line, so that
# ``zea tools select`` works both fully interactively and fully non-interactively.


def normalize_title(title: str) -> str:
    """Normalize a user supplied title to a snake_case name.

    The result is used both as a segmentation label and as part of the output filename,
    so anything outside ``[a-z0-9_-]`` is collapsed into underscores.

    Args:
        title (str): Raw title, e.g. ``"Left Ventricle"``.

    Returns:
        str: The normalized title, e.g. ``"left_ventricle"``.

    Raises:
        ValueError: If the title is empty (or contains nothing usable).
    """
    title = re.sub(r"[^a-z0-9_-]+", "_", title.strip().lower()).strip("_")
    if not title:
        raise ValueError("Title cannot be empty.")
    return title


def ask_for_title() -> str:
    """Ask the user for a title describing what is being selected."""
    log.info("What are you selecting?")
    while True:
        try:
            title = normalize_title(input("Enter a title for the selection: "))
            break
        except ValueError:
            log.error("Please enter a non-empty title")
    log.info(f"Title set to: {log.yellow(title)}")
    return title


def ask_for_selection_tool() -> str:
    """Ask the user which selection tool to use."""
    while True:
        selector = input(f"Which selection tool do you want to use? [{'/'.join(SELECTORS)}]: ")
        if selector in SELECTORS:
            return selector
        log.error(f"Please enter one of {SELECTORS}")


def ask_for_num_selections() -> int:
    """Ask the user how many key frames to annotate."""
    while True:
        try:
            num_selections = int(input("How many selections do you want to make? "))
            if num_selections < 1:
                raise ValueError
            return num_selections
        except ValueError:
            log.error("Please enter a positive integer")


def ask_save_animation_with_fps() -> int:
    """Ask the user for the frame rate to save the preview animation with."""
    while True:
        try:
            fps = int(input("Frames per second for the preview animation: "))
            if fps < 1:
                raise ValueError
            return fps
        except ValueError:
            log.error("Please enter a positive integer")


# ── Input handling ────────────────────────────────────────────────────────────

#: File types that hold a whole sequence, and are therefore annotated on their own.
_SEQUENCE_TYPES = tuple(suffix.lower() for suffix in _SUPPORTED_VID_TYPES + _SUPPORTED_ZEA_TYPES)
#: File types that hold a single image. ``_SUPPORTED_IMG_TYPES`` lists some suffixes
#: twice (``.png`` and ``.PNG``); matching on the lower-cased suffix covers every casing.
_IMAGE_TYPES = tuple(sorted({suffix.lower() for suffix in _SUPPORTED_IMG_TYPES}))


def _suffix(file: str | Path) -> str:
    """Lower-case suffix of a local path or an ``hf://`` URI."""
    return PurePosixPath(str(file)).suffix.lower()


def ask_for_files() -> list[str]:
    """Ask for the input file paths on the terminal, one per line.

    Only reached when no paths were passed on the command line. Typing a video, gif or
    zea file ends the loop right away, since a sequence is annotated on its own.

    Returns:
        list[str]: The chosen paths, local or ``hf://``.

    Raises:
        ValueError: If no file was given.
    """
    log.info(
        "Enter the path to each input file, one per line: as many images as you like, "
        "OR one video / gif / zea file. Leave empty to continue."
    )
    files: list[str] = []
    while True:
        answer = input("Path: ").strip().strip("'\"")
        if not answer:
            break
        if not answer.startswith(HF_PREFIX):
            local = Path(answer).expanduser()
            if not local.exists():
                log.error(f"{local} does not exist.")
                continue
            answer = str(local)
        files.append(answer)
        if _suffix(answer) in _SEQUENCE_TYPES:
            break

    if not files:
        raise ValueError("No files selected.")
    return files


class SourceMetadata(NamedTuple):
    """Small, cheap-to-copy fields carried over from a zea input file.

    The bulk arrays (raw data, beamformed data, ...) are deliberately left behind: the
    annotation file holds only the images that were annotated and their masks, so it
    stays small and can be written without streaming gigabytes back out.
    """

    #: Keyword arguments for :meth:`zea.File.create`, e.g. ``metadata``, ``probe``.
    file_fields: dict
    #: Extra fields for the copied image map, e.g. ``coordinates``, ``timestamps``.
    map_fields: dict


#: Map fields the tool writes itself, so they are never copied from a source file.
_OWN_MAP_FIELDS = frozenset({"values", "labels"})
#: Map fields describing the pixel values, which say nothing about a boolean mask.
_VALUE_MAP_FIELDS = frozenset({"unit", "min", "max", "description"})
#: File fields the annotation file writes itself (``description``) or cannot honour
#: without the acquisition it describes (``track_schedule``).
_OWN_FILE_FIELDS = frozenset({"track_schedule", "description"})


def _copyable_fields(schema, skip: frozenset) -> tuple[str, ...]:
    """Names in a spec ``SCHEMA`` that are worth copying from a source file.

    Derived from the spec rather than listed here, so fields added to
    :mod:`zea.data.spec` later are carried over without touching this module.
    """
    return tuple(name for name in schema if name not in skip)


def _select_track(file, track: str | int | None):
    """Return the data group to annotate, from a single- or multi-track file.

    Args:
        file (zea.File): The open file.
        track (str | int, optional): Label or index of the track to annotate. Only
            needed for files with more than one track.

    Returns:
        The track's data group.

    Raises:
        ValueError: If the file has several tracks and ``track`` does not name one.
    """
    labels = file.track_labels
    if len(labels) <= 1 and track is None:
        return file.data
    if track is None:
        raise ValueError(
            f"This file has {len(labels)} tracks, so --track is needed to say which one "
            f"to annotate. Available: {labels}."
        )
    if isinstance(track, int) or (isinstance(track, str) and track.isdigit()):
        return file.tracks[int(track)].data
    if track not in labels:
        raise ValueError(f"No track labelled {track!r} in this file. Available: {labels}.")
    return file.get_track(track).data


def _load_zea_file(
    path: str | Path, track: str | int | None = None
) -> tuple[np.ndarray, SourceMetadata]:
    """Read the image map of a zea HDF5 file, plus the metadata worth carrying over.

    Args:
        path (str | Path): Path to a zea file. Also accepts an ``hf://`` URI.
        track (str | int, optional): Label or index of the track to annotate, for files
            holding more than one.

    Returns:
        tuple: ``(values, source)`` with the image values and a :class:`SourceMetadata`
        holding everything small enough to copy into the annotation file.

    Raises:
        ValueError: If the track cannot be resolved, if the file has no image data, or
            if the images are not 2D.
    """
    from zea.data.file import File, load_dict_from_hdf5_group
    from zea.data.spec import FileSpec, Map

    path = str(path)
    if path.startswith(HF_PREFIX):
        path = _hf_resolve_path(path)

    with File(path) as file:
        data = _select_track(file, track)
        if "image" not in data.keys():
            raise ValueError(
                f"{path} has no 'data/image' group. The selection tool annotates images, "
                "so the file must contain (beamformed and log-compressed) image data. "
                "Use `zea process` to reconstruct images from raw data first."
            )
        image = data.image
        values = image.values[:]

        map_fields = {
            name: getattr(image, name)[()]
            for name in _copyable_fields(Map.SCHEMA, _OWN_MAP_FIELDS)
            if name in image.keys()
        }
        file_fields = {}
        for name in _copyable_fields(FileSpec.SCHEMA, _OWN_FILE_FIELDS):
            if name in file:
                file_fields[name] = load_dict_from_hdf5_group(file[name])
            elif name in file.attrs:
                file_fields[name] = file.attrs[name]

    if values.ndim != 3:
        raise ValueError(
            f"Expected 2D images of shape (n_frames, z, x) in {path}, got shape "
            f"{values.shape}. Volumetric data is not supported by the selection tool."
        )
    return values, SourceMetadata(file_fields, map_fields)


class SelectionInputs(NamedTuple):
    """The images to annotate, and where they came from."""

    #: The 2D images to annotate.
    images: list[np.ndarray]
    #: Name of the file each image came from.
    file_names: list[str]
    #: True when the images are consecutive frames of one recording, annotated by
    #: interpolating between key frames. False when they are separate images, compared
    #: with a metric.
    is_sequence: bool
    #: Fields carried over from a zea input file, so the saved annotations line up with
    #: (and describe) the source. None for images, videos and gifs.
    source: SourceMetadata | None = None


def load_input_files(
    files: Sequence[str | Path], track: str | int | None = None
) -> SelectionInputs:
    """Load a set of images, the frames of a single video / gif, or a zea file.

    Args:
        files (Sequence[str | Path]): Image files, or a single video / gif or zea HDF5
            file. zea files also accept an ``hf://`` URI.
        track (str | int, optional): Label or index of the track to annotate, for zea
            files holding more than one.

    Returns:
        SelectionInputs: The loaded images and where they came from.

    Raises:
        ValueError: If no files were given, if a file type is unsupported, or if a video
            / zea file was combined with other files.
    """
    # Kept as strings: `Path('hf://zeahub/camus')` collapses the double slash to
    # `hf:/zeahub/camus`, breaking the Hugging Face prefix checks downstream.
    files = [str(file) for file in files]
    if not files:
        raise ValueError("No input files given.")

    sequences = [file for file in files if _suffix(file) in _SEQUENCE_TYPES]
    if sequences:
        if len(files) > 1:
            raise ValueError(
                f"Select either a single video / zea file or one or more images, got "
                f"{len(files)} files including a sequence."
            )
        path = sequences[0]
        source = None
        if _suffix(path) in _SUPPORTED_ZEA_TYPES:
            values, source = _load_zea_file(path, track)
            frames = list(values)
        else:
            frames = list(load_video(path))
        name = PurePosixPath(path).name
        # A single frame cannot be interpolated, so treat it as a plain image.
        return SelectionInputs(frames, [name] * len(frames), len(frames) > 1, source)

    images, file_names = [], []
    for file in files:
        if _suffix(file) not in _IMAGE_TYPES:
            raise ValueError(
                f"Unsupported file type {PurePosixPath(file).suffix!r}. Supported types are "
                f"{', '.join(_IMAGE_TYPES + _SEQUENCE_TYPES)}."
            )
        images.append(load_image(file))
        file_names.append(PurePosixPath(file).name)
    return SelectionInputs(images, file_names, False)


# ── High level routines ───────────────────────────────────────────────────────


def compare_images(
    images: Sequence[np.ndarray],
    file_names: Sequence[str],
    selector: str = "rectangle",
    metric: str | None = "gcnr",
    confirm_selection: bool = True,
) -> list:
    """Select two regions in one image and compare them across all images.

    Every image is plotted in its own figure; the selection is made in the first one.
    Nothing is written to disk, so the comparison is held on screen until dismissed,
    the way sequence mode holds its preview open after saving.

    Args:
        images (Sequence[np.ndarray]): The images to compare.
        file_names (Sequence[str]): Names shown as the title of each figure.
        selector (str, optional): Type of selection tool. Defaults to ``"rectangle"``.
        metric (str, optional): Metric to compute between the two patches. Defaults to
            ``"gcnr"``.
        confirm_selection (bool, optional): Whether to confirm the selection and hold
            the comparison open, both in the plot window. Defaults to True.

    Returns:
        list: The computed metric scores, one per image.
    """
    axs, figures = [], []
    # Plot in reverse so that the figure the selection is made in ends up on top.
    for i, (image, file_name) in enumerate(zip(images[::-1], file_names[::-1])):
        fig, ax = plt.subplots()
        ax.imshow(image, cmap="gray")
        if i == len(images) - 1:
            ax.set_title(f"Make selection in this plot\n {file_name}")
        else:
            ax.set_title(file_name)
        ax.axis("off")
        axs.append(ax)
        figures.append(fig)

    scores = interactive_selector_with_plot_and_metric(
        list(images),
        axs[::-1],
        selector=selector,
        metric=metric,
        confirm_selection=confirm_selection,
    )

    if confirm_selection:
        wait_for_key(figures[-1], f"Press {_keys(ACCEPT_KEYS)} to close.")
    for fig in figures:
        plt.close(fig)
    return scores


def _select_key_frame_mask(image, axs, selector: str, confirm_selection: bool):
    """Select one non-empty mask on ``image``, retrying while the selection is empty.

    Returns:
        np.ndarray | None: The mask, or None when the user closed the window without
        selecting anything.
    """
    while True:
        _, mask = interactive_selector(
            image,
            axs,
            selector=selector,
            num_selections=1,
            confirm_selection=confirm_selection,
        )
        if not mask:
            return None
        if mask[0].sum() > 0:
            return mask[0]
        log.warning("Empty mask. Try again, make sure to make a decent selection...")


def annotate_sequence(
    images: Sequence[np.ndarray],
    selector: str = "rectangle",
    num_selections: int = 2,
    confirm_selection: bool = True,
) -> list[np.ndarray]:
    """Annotate evenly spaced key frames of a sequence and interpolate in between.

    Closing the window instead of selecting stops the annotating and keeps the key
    frames done so far, matching what closing the window means elsewhere in the tool.
    The masks stay tied to the key frames they were drawn on, and the frames past the
    last annotated key frame keep its mask.

    Args:
        images (Sequence[np.ndarray]): Frames of the sequence.
        selector (str, optional): Type of selection tool. Defaults to ``"rectangle"``.
        num_selections (int, optional): Number of key frames to annotate. Defaults to 2.
        confirm_selection (bool, optional): Whether to ask (in the plot window) to
            confirm each key frame's selection. Defaults to True.

    Returns:
        list[np.ndarray]: One interpolated mask per frame in ``images``.

    Raises:
        ValueError: If ``num_selections`` is not positive, or if no key frame was
            annotated at all.
    """
    assert len(images) > 1, "At least two frames are required to annotate a sequence."
    if num_selections < 1:
        raise ValueError(f"At least one key frame must be annotated, got {num_selections}.")
    num_selections = min(num_selections, len(images))

    key_frames = np.linspace(0, len(images) - 1, num_selections).astype(int)
    masks = []
    annotated_frames = []
    pos, size = None, None
    for idx in key_frames:
        image = images[idx]
        fig, axs = plt.subplots()
        fig.tight_layout()
        # set window size to what the user selected for the previous plot
        if pos is not None:
            move_matplotlib_figure(fig, pos, size)

        axs.imshow(image, cmap="gray")

        mask = _select_key_frame_mask(image, axs, selector, confirm_selection)
        if mask is None:
            log.warning(f"Stopping after {len(masks)} of {num_selections} key frames.")
            plt.close(fig)
            break

        pos, size = get_matplotlib_figure_props(fig)

        plot_mask(axs, mask, selector)
        plt.close(fig)
        masks.append(mask)
        annotated_frames.append(int(idx))

    if not masks:
        raise ValueError("No region was selected, so there is nothing to interpolate.")

    # interpolation needs at least two masks, so duplicate a single selection. Both
    # copies are the same, so spanning the sequence just holds it over every frame.
    if len(masks) == 1:
        masks.append(masks[0])
        annotated_frames = [0, len(images) - 1]

    return interpolate_masks(
        masks,
        num_frames=len(images),
        rectangle=(selector == "rectangle"),
        positions=annotated_frames,
    )


def preview_figure(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    selector: str = "rectangle",
    title: str = "",
    frame: int = 0,
) -> matplotlib.figure.Figure:
    """Open a figure showing one annotated frame, to report progress on.

    Args:
        images (Sequence[np.ndarray]): Frames of the sequence.
        masks (Sequence[np.ndarray]): One mask per frame.
        selector (str, optional): Type of selection tool the masks came from. Defaults
            to ``"rectangle"``.
        title (str, optional): Name of what was selected, shown above the frame.
        frame (int, optional): Which frame to show. Defaults to 0.

    Returns:
        matplotlib.figure.Figure: The (non-blocking) figure.
    """
    fig, ax = plt.subplots()
    ax.imshow(images[frame], cmap="gray")
    ax.axis("off")
    ax.set_title(f"{title} - frame {frame + 1} of {len(images)}".strip(" -"))
    plot_mask(ax, masks[frame], selector)
    fig.tight_layout()
    plt.show(block=False)
    return fig


def save_mask_animation(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    filename: str | Path,
    selector: str = "rectangle",
    fps: int = 20,
) -> Path:
    """Save an animation of the images with their masks overlaid.

    Args:
        images (Sequence[np.ndarray]): Frames of the sequence.
        masks (Sequence[np.ndarray]): One mask per frame.
        filename (str | Path): Output path of the gif.
        selector (str, optional): Type of selection tool the masks came from, which
            determines how they are drawn. Defaults to ``"rectangle"``.
        fps (int, optional): Frames per second of the animation. Defaults to 20.

    Returns:
        Path: The path the animation was written to.
    """
    assert len(images) == len(masks), (
        f"Number of images ({len(images)}) and masks ({len(masks)}) must match."
    )
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots()
    imshow_obj = axs.imshow(images[0], cmap="gray")

    ani = FuncAnimation(
        fig,
        update_imshow_with_mask,
        frames=len(images),
        fargs=(axs, imshow_obj, images, masks, selector),
        interval=1000 / fps,
    )
    ani.save(filename, writer="pillow")
    plt.close(fig)
    log.info(f"Successfully saved animation as {log.yellow(filename)}")
    return filename


def _with_extension(path: Path, extension: str) -> Path:
    """Give ``path`` this extension, appending rather than replacing it.

    ``Path.with_suffix`` would eat everything after the first dot, so a source named
    ``clip.720p.mp4`` would lose the title from its annotation filename.
    """
    return path if path.suffix == extension else path.with_name(path.name + extension)


def save_masks(
    masks: Sequence[np.ndarray] | np.ndarray,
    filename: str | Path,
    images: Sequence[np.ndarray] | np.ndarray,
    label: str = "roi",
    source: SourceMetadata | None = None,
    description: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Save annotations as a zea HDF5 file with an image and a segmentation map.

    The result is a regular zea file (see :class:`~zea.data.spec.FileSpec`) holding the
    annotated images under ``data/image`` and the masks as a single-label boolean
    :class:`~zea.data.spec.Segmentation` under ``data/segmentation``, so it can be read
    back with :class:`zea.File` like any other zea dataset.

    When the images came from a zea file, ``source`` carries its metadata over: the pixel
    coordinates, frame timing, probe, subject and credit information. The bulk arrays are
    not copied -- the annotation file describes the images that were annotated, not the
    acquisition they were reconstructed from.

    Since the selection tool only produces images and segmentations, the warnings about
    the acquisition fields it cannot fill in (scan parameters, ...) are suppressed.

    Args:
        masks (Sequence[np.ndarray] | np.ndarray): One boolean mask per image.
        filename (str | Path): Output path; the ``.hdf5`` suffix is enforced.
        images (Sequence[np.ndarray] | np.ndarray): The annotated images, of equal shape
            as the masks.
        label (str, optional): Name of the segmentation label. Defaults to ``"roi"``.
        source (SourceMetadata, optional): Fields carried over from a zea input file, as
            returned in :attr:`SelectionInputs.source`. Defaults to None.
        description (str, optional): Free-text description stored in the file.
        overwrite (bool, optional): Whether to overwrite an existing file. Defaults to
            False.

    Returns:
        Path: The path the file was written to.
    """
    from zea.data.file import File

    image_values = np.asarray(images)
    mask_values = np.asarray(masks, dtype=np.bool_)
    assert image_values.shape == mask_values.shape, (
        f"Images {image_values.shape} and masks {mask_values.shape} must have the same shape."
    )

    # zea images are uint8 or float32 (in dB); anything else is a plain array we
    # normalized ourselves, so store it as uint8.
    if image_values.dtype not in (np.uint8, np.float32):
        image_values = image_values.astype(np.uint8)

    filename = _with_extension(Path(filename), ".hdf5")
    filename.parent.mkdir(parents=True, exist_ok=True)

    file_fields = dict(source.file_fields) if source is not None else {}
    map_fields = dict(source.map_fields) if source is not None else {}

    image_map = {"values": image_values, **map_fields}
    segmentation_map = {
        # (n_frames, z, x) -> (n_frames, z, x, n_labels) with a single label
        "values": mask_values[..., None],
        "labels": np.array([label], dtype=np.str_),
        **{name: value for name, value in map_fields.items() if name not in _VALUE_MAP_FIELDS},
    }

    File.create(
        path=filename,
        data={"image": image_map, "segmentation": segmentation_map},
        description=description or f"Regions of interest ('{label}') from zea tools select.",
        overwrite=overwrite,
        ignore_warnings=True,
        **file_fields,
    )
    log.info(f"Successfully saved annotations to {log.yellow(filename)}")
    return filename


def _output_stem(source: str, title: str, output_dir: str | Path | None) -> Path:
    """Build the output path (without suffix) for the annotations of ``source``."""
    if output_dir is not None:
        directory = Path(output_dir)
    elif source.startswith(HF_PREFIX):
        # Remote inputs are read-only, so write next to where the tool was started.
        directory = Path.cwd()
    else:
        directory = Path(source).parent
    return directory / f"{PurePosixPath(source).stem}_{title}_annotations"


def _check_outputs_free(paths: Sequence[Path], overwrite: bool) -> None:
    """Raise unless every output path is free, mirroring the ``zea data`` guards."""
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"Output file(s) already exist: {', '.join(str(path) for path in existing)}. "
            "Use overwrite=True (--overwrite) to overwrite them."
        )


def run_selection_tool(
    files: Sequence[str | Path] | None = None,
    selector: str | None = None,
    title: str | None = None,
    num_selections: int | None = None,
    fps: int | None = None,
    metric: str | None = "gcnr",
    output_dir: str | Path | None = None,
    save_animation: bool = True,
    confirm_selection: bool = True,
    overwrite: bool = False,
    track: str | int | None = None,
):
    """Run the interactive selection tool.

    This is the entry point behind ``zea tools select``. Depending on the input it runs
    in one of two modes:

    - **Images**: two regions are selected in the first image and compared across all
      images using ``metric``.
    - **Sequence** (video, gif or a zea file with more than one frame):
      ``num_selections`` key frames are annotated, the masks are interpolated over all
      frames, written to a zea HDF5 file as a ``segmentation`` map next to the images,
      and optionally previewed as an animated gif.

    Any argument left as None is asked for interactively.

    Args:
        files (Sequence[str | Path], optional): Input images, or a single video / gif or
            zea HDF5 file (an ``hf://`` URI works too). Defaults to None, i.e. ask for
            the paths on the terminal.
        selector (str, optional): Type of selection tool, one of :data:`SELECTORS`.
        title (str, optional): Name of what is being selected. Used as the segmentation
            label and in the output filenames. Only used in sequence mode.
        num_selections (int, optional): Number of key frames to annotate. Only used in
            sequence mode.
        fps (int, optional): Frame rate of the preview animation. Only used in sequence
            mode, and only when ``save_animation`` is True.
        metric (str, optional): Metric to compute between the two patches. Only used in
            image mode. Defaults to ``"gcnr"``.
        output_dir (str | Path, optional): Directory to write the annotations and
            animation to. Defaults to the folder of the input file, or the working
            directory for ``hf://`` inputs.
        save_animation (bool, optional): Whether to save a preview gif in sequence mode.
            Defaults to True.
        confirm_selection (bool, optional): Whether to ask (in the plot window) to
            confirm each selection. Defaults to True.
        overwrite (bool, optional): Whether to overwrite existing output files. Checked
            before the annotating starts, so no work is lost. Defaults to False.
        track (str | int, optional): Label or index of the track to annotate, for zea
            files holding more than one.

    Returns:
        list: The metric scores in image mode, or the interpolated masks in sequence
        mode.

    Raises:
        FileExistsError: If an output file exists and ``overwrite`` is False.
    """
    files = list(files) if files else ask_for_files()
    inputs = load_input_files(files, track)
    images = inputs.images

    if selector is None:
        selector = ask_for_selection_tool()
    assert selector in SELECTORS, f"Selector must be one of {SELECTORS}, not {selector!r}."

    if not inputs.is_sequence:
        return compare_images(
            images,
            inputs.file_names,
            selector=selector,
            metric=metric,
            confirm_selection=confirm_selection,
        )

    log.info(f"Found sequence of {len(images)} frames.")
    if title is None:
        title = ask_for_title()
    else:
        title = normalize_title(title)
    if num_selections is None:
        num_selections = ask_for_num_selections()

    source = str(files[0])
    stem = _output_stem(source, title, output_dir)
    outputs = [_with_extension(stem, ".hdf5")]

    animation_path, animation_fps = None, 0
    if save_animation:
        animation_path = _with_extension(stem, ".gif")
        animation_fps = fps if fps is not None else ask_save_animation_with_fps()
        outputs.append(animation_path)

    _check_outputs_free(outputs, overwrite)

    masks = annotate_sequence(
        images,
        selector=selector,
        num_selections=num_selections,
        confirm_selection=confirm_selection,
    )

    # Report progress in a window, so the result and where it went stay visible.
    fig = preview_figure(images, masks, selector, title=title)
    status = show_status(fig, "Saving annotations...")

    save_masks(
        masks,
        stem,
        images=images,
        label=title,
        source=inputs.source,
        description=f"Regions of interest ('{title}') selected in {PurePosixPath(source).name}.",
        overwrite=overwrite,
    )

    if animation_path is not None:
        status = show_status(fig, "Saving preview animation...", status)
        save_mask_animation(images, masks, animation_path, selector=selector, fps=animation_fps)

    # The banner has to fit under the figure; save_masks and save_mask_animation
    # already log where they wrote to.
    status.remove()
    if confirm_selection:
        wait_for_key(fig, f"Saved. Press {_keys(ACCEPT_KEYS)} to close.")
    plt.close(fig)

    return masks


def main() -> None:
    """Entry point for ``python -m …``, equivalent to ``zea tools select``."""
    import tyro

    from zea.cli_args import _Select

    tyro.cli(_Select).run()


if __name__ == "__main__":
    main()
