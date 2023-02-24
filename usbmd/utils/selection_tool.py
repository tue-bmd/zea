"""Selection tools for interactively selecting part of an array
displayed as an image with matplotlib.
Author(s): Tristan Stevens
Date: 24/02/2023
"""

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector, RectangleSelector
from matplotlib.patches import Rectangle
import numpy as np

from usbmd.utils.metrics import get_metric
from usbmd.utils.utils import translate

def crop_array(array, value=None):
    """Crop an array to remove all rows and columns containing only a given value."""
    mask = np.all(np.equal(array, value), axis=1)
    array = array[~mask]

    mask = np.all(np.equal(array, value), axis=0)
    array = array[:, ~mask]
    return array


def interactive_selector(data, ax, selector='rectangle', extent=None):
    """Interactively select part of an array displayed as an image with matplotlib.

    Args:
        data (ndarray): input array.
        ax (plt.ax): existing matplotlib figure ax to select region on.
        selector (str, optional): type of selector. Defaults to 'rectangle'. For `lasso`,
            use `LassoSelector`; for `rectangle`, use `RectangleSelector`.

    Returns:
        patches (list): list of selected parts of data
        masks (list): list of boolean masks for selected parts of data
    """
    x, y = np.meshgrid(
        np.arange(data.shape[1], dtype=int),
        np.arange(data.shape[0], dtype=int))
    pix = np.vstack((x.flatten(), y.flatten())).T

    selected_data = np.zeros_like(data)
    mask = np.tile(False, data.shape)
    masks = []

    def _translate_coordinates(x, y):
        if extent:
            x = translate(x, (extent[0], extent[1]), (0, data.shape[1]))
            y = translate(y, (extent[2], extent[3]), (0, data.shape[0]))
        return x, y

    def _onselect_lasso(verts):
        verts = np.array(verts)
        # if axis is drawn with extent argument, first translate coordinates to pixels
        verts = np.array(_translate_coordinates(*verts.T)).T
        p = Path(verts)
        ind = p.contains_points(pix, radius=1)
        selected_data.flat[ind] = data.flat[ind]
        mask.flat[ind] = True
        masks.append(np.copy(mask))
        mask.flat[ind] = False

    def _onselect_rectangle(start, end):
        # if axis is drawn with extent argument, first translate coordinates to pixels
        start.xdata, start.ydata = _translate_coordinates(start.xdata, start.ydata)
        end.xdata, end.ydata = _translate_coordinates(end.xdata, end.ydata)

        verts = np.array([[start.xdata, start.ydata],
                          [start.xdata, end.ydata],
                          [end.xdata, end.ydata],
                          [end.xdata, start.ydata]], int)
        p = Path(verts)
        ind = p.contains_points(pix, radius=1)
        selected_data.flat[ind] = data.flat[ind]
        mask.flat[ind] = True
        masks.append(np.copy(mask))
        mask.flat[ind] = False

    name_to_selector = {'lasso': LassoSelector,
                        'rectangle': RectangleSelector}
    selector = name_to_selector[selector]
    onselect_dict = {LassoSelector: _onselect_lasso,
                     RectangleSelector: _onselect_rectangle}
    kwargs_dict = {LassoSelector: {},
                   RectangleSelector: {'interactive': True}}

    lasso = selector(ax, onselect_dict[selector], **kwargs_dict[selector])
    plt.show(block=True)

    lasso.disconnect_events()

    patches = []
    for mask in masks:
        patches.append(crop_array(selected_data * mask, value=0))

    return patches, masks

def interactive_selector_with_plot_and_metric(
    data, ax, selector='rectangle', metric=None, **kwargs):
    """Wrapper for interactive_selector to plot the selected regions."""
    patches, masks = interactive_selector(data, ax, selector, **kwargs)

    if metric:
        score = get_metric(metric)(patches[0], patches[1])
        print(f'{metric}: {score:.3f}')

    fig, axs = plt.subplots(len(masks), 3)
    for ax_new, patch, mask in zip(axs, patches, masks):
        ax_new[0].imshow(data)
        ax_new[1].imshow(patch)
        ax_new[2].imshow(mask)

        if selector == 'rectangle':
            # Create a Rectangle patch
            y1, y2 = np.where(np.diff(mask, axis=0).sum(axis=1))[0]
            x1, x2 = np.where(np.diff(mask, axis=1).sum(axis=0))[0]
            rect = Rectangle(
                (x1, y1), (x2 - x1), (y2 - y1), linewidth=1,
                edgecolor='r', facecolor="none")

            # Add the patch to the Axes
            ax_new[0].add_patch(rect)
        for _ax in ax_new:
            _ax.axis('off')
    if metric:
        fig.suptitle(f'{metric}: {score:.3f}')
    fig.tight_layout()
    plt.show(block=True)

if __name__ == '__main__':
    from skimage.data import coins

    data = coins()

    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(data)

    interactive_selector_with_plot_and_metric(
        data, ax, selector='rectangle', metric='gcnr')

    plt.show()
