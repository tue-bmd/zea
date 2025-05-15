"""
Example script for using the CarotidSegmenter model to generate masks for carotid ultrasound images.

For more information see original paper:
    - "Unsupervised domain adaptation method for segmenting cross-sectional CCA images"
    - https://doi.org/10.1016/j.cmpb.2022.107037
    - Author: Luuk van Knippenberg
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
from keras import ops

from usbmd import init_device, log, set_data_paths
from usbmd.backend.tensorflow.dataloader import h5_dataset_from_directory
from usbmd.models.carotid_segmenter import CarotidSegmenter
from usbmd.utils.selection_tool import add_shape_from_mask
from usbmd.utils.visualize import plot_image_grid, set_mpl_style

if __name__ == "__main__":
    # Set up data paths and device
    data_paths = set_data_paths()
    init_device()

    presets = list(CarotidSegmenter.presets.keys())
    log.info(f"Available built-in usbmd presets for CarotidSegmenter: {presets}")

    model = CarotidSegmenter.from_preset("carotid-segmenter")

    n_imgs = 4
    val_dataset = h5_dataset_from_directory(
        data_paths.data_root / "USBMD_datasets/2023_USBMD_carotid/HDF5",
        key="data/image",
        batch_size=n_imgs,
        shuffle=True,
        image_range=[-60, 0],
        normalization_range=[0, 1],
        seed=4,
    )

    batch = next(iter(val_dataset))

    masks = model(batch)
    masks = ops.squeeze(masks, axis=-1)
    masks_clipped = ops.where(masks > 0.5, 1, 0)
    masks_clipped = ops.convert_to_numpy(masks_clipped)

    set_mpl_style()

    # stack batch twice to get 2 rows
    batch_stacked = ops.concatenate([batch, batch])

    fig, _ = plot_image_grid(batch_stacked, vmin=0, vmax=1, ncols=n_imgs)
    axes = fig.axes[n_imgs : n_imgs * 2]
    for ax, mask in zip(axes, masks_clipped):
        add_shape_from_mask(ax, mask, color="red", alpha=0.5)

    path = "carotid_example.png"
    fig.savefig(
        path,
        pad_inches=0.2,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
    log.info(f"Saved to {log.yellow(path)}")
    del val_dataset  # weird tf datasets bug if not deleted
