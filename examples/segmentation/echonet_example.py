"""
Example script for using the EchoNetDynamic model to generate masks for ultrasound images.

https://echonet.github.io/dynamic/

"""

import os

# NOTE: should be `tensorflow` or `jax` for EchoNetDynamic
# You'll need tf2jax installed to use JAX backend
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
from keras import ops

from usbmd import init_device, log, set_data_paths
from usbmd.backend.tensorflow.dataloader import make_dataloader
from usbmd.models.echonet import EchoNetDynamic
from usbmd.tools.selection_tool import add_shape_from_mask
from usbmd.utils import translate
from usbmd.visualize import plot_image_grid, set_mpl_style

if __name__ == "__main__":
    # Set up data paths and device
    data_paths = set_data_paths()
    init_device()

    presets = list(EchoNetDynamic.presets.keys())
    log.info(f"Available built-in usbmd presets for EchoNetDynamic: {presets}")

    model = EchoNetDynamic.from_preset("echonet-dynamic")

    n_imgs = 16
    val_dataset = make_dataloader(
        data_paths.data_root / "USBMD_datasets/echonet_v2025/val",
        key="data/image_sc",
        batch_size=n_imgs,
        shuffle=True,
        image_range=[-60, 0],
        normalization_range=[-1, 1],
        seed=42,
    )

    batch = next(iter(val_dataset))
    rgb_batch = ops.concatenate([batch, batch, batch], axis=-1)  # grayscale to RGB

    masks = model(rgb_batch)
    masks = ops.squeeze(masks, axis=-1)
    masks = ops.convert_to_numpy(masks)

    set_mpl_style()

    batch = translate(rgb_batch, [-1, 1], [0, 1])
    fig, _ = plot_image_grid(batch, vmin=0, vmax=1)
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
