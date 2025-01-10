"""
Example script for using the EchoNetDynamic model to generate masks for ultrasound images.

https://echonet.github.io/dynamic/

- **Author(s)**: Tristan Stevens
- **Date**: 20/12/2024
"""

import os

# NOTE: should be `tensorflow` for EchoNetDynamic
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

init_device()

import matplotlib.pyplot as plt
from keras import ops

from usbmd import init_device, log, set_data_paths
from usbmd.backend.tensorflow.dataloader import h5_dataset_from_directory
from usbmd.models.echonet import EchoNetDynamic
from usbmd.utils.selection_tool import add_shape_from_mask
from usbmd.utils.visualize import plot_image_grid, set_mpl_style

if __name__ == "__main__":
    # Set up data paths and device
    data_paths = set_data_paths()

    n_imgs = 16
    val_dataset = h5_dataset_from_directory(
        data_paths.data_root / "USBMD_datasets/CAMUS/val",
        key="data/image",
        batch_size=n_imgs,
        shuffle=True,
        image_size=[256, 256],
        resize_type="resize",
        image_range=[-60, 0],
        normalization_range=[-1, 1],
        seed=42,
    )

    presets = list(EchoNetDynamic.presets.keys())
    log.info(f"Available built-in usbmd presets for EchoNetDynamic: {presets}")

    model = EchoNetDynamic.from_preset("echonet-dynamic")

    batch = next(iter(val_dataset))

    masks = model(batch)
    masks = ops.squeeze(masks, axis=-1)
    masks = ops.convert_to_numpy(masks)

    set_mpl_style()

    fig, _ = plot_image_grid(batch)
    axes = fig.axes[:n_imgs]
    for ax, mask in zip(axes, masks):
        add_shape_from_mask(ax, mask, color="red", alpha=0.5)

    path = "echonet_example.png"
    fig.savefig(
        path,
        pad_inches=0.2,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
    log.info(f"Saved to {log.yellow(path)}")
    del val_dataset  # weird tf datasets bug if not deleted
