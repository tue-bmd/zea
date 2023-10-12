"""Selection tools for interactively selecting part of an array
displayed as an image with matplotlib.

- **Author(s)**     : Tristan Stevens
- **Date**          : 24/02/2023
"""
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as pltPath
from matplotlib.widgets import LassoSelector, RectangleSelector
from PIL import Image, ImageDraw
from scipy.interpolate import interp1d
from skimage import measure
from skimage.measure import approximate_polygon, find_contours
from sklearn.metrics import pairwise_distances

from usbmd.utils.io import _SUPPORTED_VID_TYPES, filename_from_window_dialog, load_video
from usbmd.utils.metrics import get_metric
from usbmd.utils.utils import translate


def crop_array(array, value=None):
    """Crop an array to remove all rows and columns containing only a given value."""
    mask = np.all(np.equal(array, value), axis=1)
    array = array[~mask]

    mask = np.all(np.equal(array, value), axis=0)
    array = array[:, ~mask]
    return array


def interactive_selector(
    data, ax, selector="rectangle", extent=None, verbose=True, num_selections=None
):
    """Interactively select part of an array displayed as an image with matplotlib.

    Args:
        data (ndarray): input array.
        ax (plt.ax): existing matplotlib figure ax to select region on.
        selector (str, optional): type of selector. Defaults to 'rectangle'.
            For `lasso` use `LassoSelector`; for `rectangle`, use `RectangleSelector`.
        extent (list): extent of axis where selection is made. Used to transform
            coordinates back to pixel values. Defaults to None.
        verbose (bool): verbosity of print statements. Defaults to False.
        num_selections (int): number of selections to make. Defaults to None.

    Returns:
        patches (list): list of selected parts of data
        masks (list): list of boolean masks for selected parts of data
    """
    x, y = np.meshgrid(
        np.arange(data.shape[1], dtype=int), np.arange(data.shape[0], dtype=int)
    )
    pix = np.vstack((x.flatten(), y.flatten())).T

    mask = np.tile(False, data.shape)
    masks = []
    select_idx = 0

    def _translate_coordinates(x, y):
        if extent:
            x = translate(x, (extent[0], extent[1]), (0, data.shape[1]))
            y = translate(y, (extent[2], extent[3]), (0, data.shape[0]))
        return x, y

    def _onselect_lasso(verts):
        nonlocal select_idx
        if verbose:
            print(f"Selection {select_idx} done")
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
            print(f"Selection {select_idx} done")
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
    selector = name_to_selector[selector]
    onselect_dict = {
        LassoSelector: _onselect_lasso,
        RectangleSelector: _onselect_rectangle,
    }
    kwargs_dict = {LassoSelector: {}, RectangleSelector: {"interactive": True}}

    lasso = selector(ax, onselect_dict[selector], **kwargs_dict[selector])

    if num_selections:
        if verbose:
            print(f"...Plot will close after {num_selections} selections...")
        plt.show(block=False)
        while not select_idx >= num_selections:
            plt.pause(0.1)
    else:
        plt.show(block=False)
        input("Press Enter to continue (don't close plot)...\n")

    lasso.disconnect_events()
    lasso.set_visible(False)
    lasso.update()

    patches = []
    for mask in masks:
        patches.append(crop_array(data * mask, value=0))

    return patches, masks


def add_rectangle_from_mask(
    ax,
    mask,
    edgecolor="r",
    facecolor="none",
    linewidth=1,
    **kwargs,
):
    """add a rectangle box to axis from mask array.

    Args:
        ax (plt.ax): matplotlib axis
        mask (ndarray): numpy array with rectangle non-zero
            box defining the region of interest.
        edgecolor (str): color of the shape's edge
        facecolor (str): color of the shape's face
        linewidth (int): width of the shape's edge

    Returns:
        plt.ax: matplotlib axis with rectangle added
    """
    # Create a Rectangle patch
    y1, y2 = np.where(np.diff(mask, axis=0).sum(axis=1))[0]
    x1, x2 = np.where(np.diff(mask, axis=1).sum(axis=0))[0]
    rect = Rectangle(
        (x1, y1),
        (x2 - x1),
        (y2 - y1),
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=linewidth,
        **kwargs,
    )

    # Add the patch to the Axes
    rect_obj = ax.add_patch(rect)
    return rect_obj


def add_shape_from_mask(ax, mask, **kwargs):
    """add a shape to axis from mask array.

    Args:
        ax (plt.ax): matplotlib axis
        mask (ndarray): numpy array with non-zero
            shape defining the region of interest.
        edgecolor (str): color of the shape's edge
        facecolor (str): color of the shape's face
        linewidth (int): width of the shape's edge

    Returns:
        plt.ax: matplotlib axis with shape added
    """
    # Create a Path patch
    contours = measure.find_contours(mask, 0.5)
    patches = []
    for contour in contours:
        path = pltPath(contour[:, ::-1])
        patch = PathPatch(path, **kwargs)
        patches.append(ax.add_patch(patch))
    return patches


def interactive_selector_with_plot_and_metric(
    data,
    ax=None,
    selector="rectangle",
    metric=None,
    cmap="gray",
    plot=True,
    mask_plot=False,
    selection_axis=0,
    **kwargs,
):
    """Wrapper for interactive_selector to plot the selected regions.

    Args:
        data (ndarray or list of ndarray): input data.
        ax (plt.ax or list of plt.ax, optional): axis corresponding to input data.
            Defaults to None. In that case function plots data first to create axis.
        selector (str, optional): type of selection tool. Defaults to 'rectangle'.
        metric (str, optional): metric to compute. Defaults to None.
        cmap (str, optional): color map to display data in. Defaults to 'gray'.
        plot (bool, optional): whether to plot selections / metrics on top of axis.
            Defaults to True.
        mask_plot (bool, optional): whether to also plot the masks in a separate plot.
            Can be useful to isolate the patches and see the selections more clearly.
            Defaults to False.
        selection_axis (int, optional): axis on which to make selection. Defaults to 0.

    Raises:
        ValueError: Can only select two patches to compute metric with. More patches
            don't make sense in this context.
    """
    if not isinstance(data, list):
        data = [data]

    if ax is None:
        fig, ax = plt.subplots(1, len(data))
        for _data, _ax in zip(data, ax):
            _ax.imshow(_data, cmap=cmap, aspect="auto")

    if not isinstance(ax, Iterable):
        ax = [ax]

    # create selector for first axis only
    patches, masks = interactive_selector(
        data[selection_axis], ax[selection_axis], selector, num_selections=2, **kwargs
    )

    if len(patches) != 2:
        raise ValueError(
            "exactly 2 patches are required for using this wrapper function"
        )

    # get patches for all data in data list using the selection made
    patches = []
    for image in data:
        patches.extend([crop_array(image * mask, value=0) for mask in masks])

    # compute metrics
    scores = []
    if metric:
        for i in range(len(data)):
            idx = i * len(masks)
            score = get_metric(metric)(patches[idx], patches[idx + 1])
            scores.append(score)
            print(f"{metric}: {score:.3f}")

    # plot on top of existing plot
    if plot:
        for _ax, score in zip(ax, scores):
            title = _ax.get_title()
            _ax.set_title(title + "\n" + f"{metric}: {score:.3f}")
            for mask in masks:
                if selector == "rectangle":
                    add_rectangle_from_mask(_ax, mask, alpha=0.5)
                else:
                    add_shape_from_mask(_ax, mask, alpha=0.5)
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

            if selector == "rectangle":
                add_rectangle_from_mask(ax_base, mask)

            for _ax in ax_new:
                _ax.axis("off")

        fig.tight_layout()

    return scores


def extract_rectangle_from_mask(image):
    """Find corner points of rectangle in binary mask.
    Args:
        image (np.ndarray): 2D binary mask
    Returns:
        Tuple of the form ((x1, y1), (x2, y2)) with the corner points of the rectangle.
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
        corner_points (tuple): Tuple of the form ((x1, y1), (x2, y2))
            with the corner points of the rectangle.
        image_shape (tuple): Size of the image (height, width).
    Returns:
        np.ndarray (height, width): 2D boolean mask.
    """
    image = np.zeros(image_shape, dtype=bool)
    x1, y1 = corner_points[0]
    x2, y2 = corner_points[1]
    image[y1 : y2 + 1, x1 : x2 + 1] = True
    return image


def interpolate_rectangles(rectangles, x_indices, y_indices):
    """Interpolate between arbitrary number of rectangles.
    Args:
        rectangles (list): List with any number of rectangles as tuples of the form
            ((x1, y1), (x2, y2)). Size of the list must be equal to the number of x indices.
        x_indices (np.ndarray): Array with x indices for interpolation.
        y_indices (np.ndarray): Array with y indices for interpolation.
    Returns:
        List with interpolated rectangles as tuples of the form ((x1, y1), (x2, y2)).
            Size of the list is equal to the number of y indices.
    """
    new_rectangles = []
    x1 = [rect[0][0] for rect in rectangles]
    x2 = [rect[1][0] for rect in rectangles]
    y1 = [rect[0][1] for rect in rectangles]
    y2 = [rect[1][1] for rect in rectangles]

    values_interp = []
    for values in [x1, x2, y1, y2]:
        values_interp.append(np.interp(y_indices, x_indices, values).astype(np.int32))

    x1, x2, y1, y2 = values_interp
    new_rectangles = [((x1[i], y1[i]), (x2[i], y2[i])) for i in range(len(x1))]
    return new_rectangles


def extract_polygon_from_mask(mask, tolerance: float = 0.01):
    """Find contours in a binary mask and fit polygon.

    Polygon approximation will reduce contour points, unless tolerance is 0.

    Args:
        mask (np.ndarray): 2D binary mask
        tolerance (float): Approximation tolerance for polygonal contour
    Returns:
        Numpy array of shape (N, 2) with vertices of the polygon.
    """
    contours = find_contours(mask, 0.5, fully_connected="high")
    poly = approximate_polygon(contours[0], tolerance)
    return poly


def reconstruct_mask_from_polygon(vertices, image_size):
    """Reconstruct a binary mask from a polygon.

    Fills in regions defined by the polygon contour.
    Args:
        vertices (np.ndarray): Vertices of the polygon as an array of shape (N, 2).
        image_size (tuple): Size of the image (height, width).
    Returns:
        np.ndarray (height, width) with the reconstructed mask.
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
        t (float): Interpolation parameter, where 0 <= t <= 1.
    Returns:
        Interpolated polygon as an array of shape (N, 2).
    """
    # Ensure both polygons have the same number of vertices
    if polygon1.shape[0] != polygon2.shape[0]:
        raise ValueError("Both polygons must have the same number of vertices.")

    # Perform linear interpolation for each vertex
    interpolated_polygon = (1 - t) * polygon1 + t * polygon2

    return interpolated_polygon


def match_polygons(polygon1, polygon2):
    """Match two polygons by minimizing the total distance between vertices.

    The vertices of the first polygon are shifted circularly to find the best match.
    Order of vertices is preserved.

    Args:
        polygon1 (np.ndarray): First polygon as an array of shape (N, 2).
        polygon2 (np.ndarray): Second polygon as an array of shape (N, 2).
    Returns:
        Tuple of the form (poly1, poly2), where poly1 and poly2 are the matched polygons.
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


def equalize_polygons(polygons, mode="max"):
    """Make sure all polygons have the same number of vertices.

    Args:
        polygons (list): List with any number of polygons as arrays of shape (N, 2).
        mode (str): Method for equalizing the number of vertices. Either 'max' or 'min'.
            with 'max' the number of vertices is equal to the polygon with the most vertices.
            with 'min' the number of vertices is equal to the polygon with the least vertices.
    Returns:
        A tuple of the form (poly1, poly2, ...), where poly1, poly2, ...
            are the trimmed polygons with the same number of vertices as the
            polygon with the fewest / most vertices, depending on the mode.
    """
    assert mode in ["max", "min"], f"Mode must be either 'max' or 'min', not {mode}."
    if mode == "max":
        num_vertices = max(polygon.shape[0] for polygon in polygons)
    elif mode == "min":
        num_vertices = min(polygon.shape[0] for polygon in polygons)

    # give warning if difference in min / max vertices is large
    if num_vertices < 0.8 * max(polygon.shape[0] for polygon in polygons):
        warnings.warn(
            "Warning: difference in number of vertices is large. "
            "Possibly due to large difference in polygon size."
        )

    if mode == "min":
        trimmed_polygons = []
        for polygon in polygons:
            indices = np.linspace(0, len(polygon) - 1, num_vertices).astype(int)
            trimmed_polygons.append(polygon[indices])

        return trimmed_polygons
    elif mode == "max":
        # interpolate the contours
        interpolated_polygons = []
        for polygon in polygons:
            if polygon.shape[0] < num_vertices:
                # interp2d
                indices = np.linspace(0, len(polygon) - 1, num_vertices)

                # create a function to interpolate the x and y coordinates separately
                f_x = interp1d(np.arange(len(polygon)), polygon[:, 0], kind="linear")
                f_y = interp1d(np.arange(len(polygon)), polygon[:, 1], kind="linear")

                # evaluate the functions at the interpolated indices
                interpolated_polygons.append(
                    np.column_stack((f_x(indices), f_y(indices)))
                )
            else:
                interpolated_polygons.append(polygon)
        return interpolated_polygons


def interpolate_masks(masks: list, num_frames: int, rectangle: bool = False):
    """Interpolate between arbitrary number of masks."""
    assert isinstance(masks, list), "Masks must be a list of numpy arrays."
    assert num_frames > 1, "At least two frames are required for interpolation."
    number_of_masks = len(masks)
    assert number_of_masks > 1, "At least two masks are required for interpolation."
    mask_shape = masks[0].shape
    assert all(
        mask.shape == mask_shape for mask in masks
    ), "All masks must have the same shape."

    # distribute number of frames over number of masks
    num_frames_per_segment = [num_frames // (number_of_masks - 1)] * (
        number_of_masks - 1
    )
    if num_frames % num_frames_per_segment[0] != 0:
        # make sure that number of frames per mask adds up to total number of frames
        num_frames_per_segment[-1] += num_frames - sum(num_frames_per_segment)

    if rectangle:
        # get the rectangles
        rectangles = []
        for mask in masks:
            rectangles.append(extract_rectangle_from_mask(mask))

        rectangles = interpolate_rectangles(
            rectangles,
            np.linspace(0, num_frames - 1, len(rectangles)),
            np.arange(num_frames),
        )

        # reconstruct the masks
        interpolated_masks = []
        for _rectangle in rectangles:
            interpolated_masks.append(
                reconstruct_mask_from_rectangle(_rectangle, mask_shape)
            )
        return interpolated_masks

    # get the contours
    polygons = []
    for mask in masks:
        polygons.append(extract_polygon_from_mask(mask))

    # trim the polygons for equal number of vertices
    polygons = equalize_polygons(polygons)

    # match the polygons
    for i in range(number_of_masks - 1):
        polygons[i], polygons[i + 1] = match_polygons(polygons[i], polygons[i + 1])

    # interpolate the polygons
    interpolated_polygons = []
    for i in range(number_of_masks - 1):
        for t in np.linspace(0, 1, num_frames_per_segment[i]):
            interpolated_polygons.append(
                interpolate_polygons(polygons[i], polygons[i + 1], t)
            )

    # reconstruct the masks
    interpolated_masks = []
    for interpolated_polygon in interpolated_polygons:
        interpolated_masks.append(
            reconstruct_mask_from_polygon(interpolated_polygon, mask_shape)
        )

    return interpolated_masks



def interactive_selector_for_dataset():
    """To be added. UI for generating and saving masks for entire dataset.
    In an efficient and user friendly way.
    """
    raise NotImplementedError


def main():
    """Main function for interactive selector on multiple images."""
    print("Select as many images as you like, and close window to continue...")
    images = []
    file_names = []
    try:
        while True:
            file = filename_from_window_dialog("Choose image / video file")
            if file.suffix in [".png", ".jpg", ".jpeg"]:
                image = plt.imread(file)
                images.append(image)
                file_names.append(file.name)
                same_images = True
            elif file.suffix in _SUPPORTED_VID_TYPES:
                images.extend(load_video(file))
                same_images = False
                break
    except Exception as e:
        print("Error:", e)
        if len(images) == 0:
            sys.exit("Please select 1 or more images")

    while True:
        selector = input(
            "Which selection tool do you want to use? [rectangle/lasso]): "
        )
        if selector in ["rectangle", "lasso"]:
            break
        print("Please enter either 'rectangle' or 'lasso'")

    if same_images is True:
        figs, axs = [], []
        for i, (image, file_name) in enumerate(zip(images[::-1], file_names[::-1])):
            fig, ax = plt.subplots()
            ax.imshow(image, cmap="gray")
            if i == len(images) - 1:
                ax.set_title(f"Make selection in this plot\n {file_name}")
            else:
                ax.set_title(file_name)
            ax.axis("off")
            axs.append(ax)
            figs.append(fig)

        axs = axs[::-1]
        figs = figs[::-1]

        interactive_selector_with_plot_and_metric(
            images,
            axs,
            selector=selector,
            metric="gcnr",
        )

    else:
        if len(images) > 3:
            print(f"Found sequence of {len(images)} images. ")
            while True:
                num_selections = input("How many selections do you want to make? ")
                try:
                    num_selections = int(num_selections)
                    if num_selections < 1:
                        raise ValueError
                    break
                except ValueError:
                    print("Please enter a positive integer")
            selection_idx = np.linspace(0, len(images) - 1, int(num_selections)).astype(
                int
            )
            selection_images = [images[idx] for idx in selection_idx]
            selection_masks = []
            for image in selection_images:
                fig, axs = plt.subplots()
                axs.imshow(image, cmap="gray")
                _, mask = interactive_selector(
                    image, axs, selector=selector, num_selections=1
                )
                if selector == "rectangle":
                    add_rectangle_from_mask(axs, mask[0], alpha=0.5)
                else:
                    add_shape_from_mask(axs, mask[0], alpha=0.5)
                plt.close()
                selection_masks.append(mask[0])

        interpolated_masks = interpolate_masks(
            selection_masks, num_frames=len(images), rectangle=(selector == "rectangle")
        )

        fig, axs = plt.subplots()
        animation_img = axs.imshow(images[0], cmap="gray")

        if selector == "rectangle":
            masks = add_rectangle_from_mask(axs, interpolated_masks[0])
        else:
            masks = add_shape_from_mask(axs, interpolated_masks[0], alpha=0.5)

        def update(frame, animation_img, masks):
            animation_img.set_array(images[frame])
            for obj in axs.findobj():
                if isinstance(obj, (PathPatch, Rectangle)):
                    try:
                        obj.remove()
                    except:
                        pass
            if selector == "rectangle":
                masks = add_rectangle_from_mask(axs, interpolated_masks[frame])
            else:
                masks = add_shape_from_mask(axs, interpolated_masks[frame], alpha=0.5)
            return animation_img, masks

        while True:
            try:
                fps = int(input("Save animation as gif? Enter fps: "))
                break
            except ValueError:
                print("Please enter a positive integer")
        ani = FuncAnimation(
            fig,
            update,
            frames=len(images),
            fargs=(animation_img, masks),
            interval=1000 / fps,
        )
        filename = Path(file.parent.stem + "_" + f"{file.stem}_interpolated_masks.gif")
        ani.save(filename, writer="pillow")
        print(f"Succesfully saved animation as {filename}")

if __name__ == "__main__":
    main()
