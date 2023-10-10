"""Selection tools for interactively selecting part of an array
displayed as an image with matplotlib.

- **Author(s)**     : Tristan Stevens
- **Date**          : 24/02/2023
"""

import sys
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector, RectangleSelector
from skimage import measure

from usbmd.utils.metrics import get_metric
from usbmd.utils.utils import filename_from_window_dialog, translate


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
        p = Path(verts)
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
        p = Path(verts)
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
    ax, mask, edgecolor="r", facecolor="none", linewidth=1, **kwargs
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
        path = Path(contour[:, ::-1])
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
    mask_plot=True,
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
            Defaults to True.
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
                    add_rectangle_from_mask(_ax, mask)
                else:
                    add_shape_from_mask(_ax, mask)
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

def interactive_selector_for_dataset():
    """To be added. UI for generating and saving masks for entire dataset.
    In an efficient and user friendly way.
    """
    raise NotImplementedError

def main():
    """Main function for interactive selector on multiple images."""
    print("Select as many images as you like, and close window to continue...")
    images = []
    try:
        while True:
            file = filename_from_window_dialog("Choose image file")
            image = plt.imread(file)
            images.append(image)
    except:
        # pylint: disable=raise-missing-from
        if len(images) == 0:
            sys.exit("Please select 1 or more images")

    fig, axs = plt.subplots(1, len(images))
    if not isinstance(axs, Iterable):
        axs = [axs]
    for i, (ax, image) in enumerate(zip(axs, images)):
        ax.imshow(image, cmap="gray", aspect="auto")
        ax.set_title(f"image {i}")

    fig.tight_layout()

    while True:
        selector = input("Which selection tool do you want to use? [rectangle/lasso]): ")
        if selector in ["rectangle", "lasso"]:
            break
        print("Please enter either 'rectangle' or 'lasso'")

    interactive_selector_with_plot_and_metric(
        images, axs, selector=selector, metric="gcnr"
    )

    plt.show()


if __name__ == "__main__":
    main()
