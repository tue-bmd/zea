"""
TAESD model from: https://github.com/madebyollin/taesd

- **Author(s)**: Wessel van Nierop
- **Date**: 20/11/2024
"""

import os

# NOTE: should be `tensorflow` or `jax`
backend = "tensorflow"
os.environ["KERAS_BACKEND"] = backend

import matplotlib.pyplot as plt
from keras import ops

from usbmd import init_device, log, set_data_paths
from usbmd.backend.tensorflow.dataloader import h5_dataset_from_directory
from usbmd.models.taesd import TinyAutoencoder
from usbmd.utils import get_date_string
from usbmd.utils.visualize import plot_image_grid

if __name__ == "__main__":
    # Set up data paths and device
    data_paths = set_data_paths()
    data_root = data_paths["data_root"]
    init_device(backend=backend)

    n_imgs = 10
    val_dataset = h5_dataset_from_directory(
        data_root / "USBMD_datasets/CAMUS/val",
        key="data/image",
        batch_size=n_imgs,
        shuffle=True,
        image_size=[256, 256],
        resize_type="resize",
        image_range=[-60, 0],
        normalization_range=[-1, 1],
        seed=42,
    )

    presets = list(TinyAutoencoder.presets.keys())
    log.info(f"Available built-in usbmd presets for TAESD: {presets}")

    model = TinyAutoencoder.from_preset("taesdxl")
    # model = TinyAutoencoder.from_preset("hf://usbmd/taesdxl")
    # model = TinyAutoencoder.from_preset("/mnt/z/Ultrasound-BMd/pretrained/taesdxl")
    # model = TinyAutoencoder.from_preset("./test_model_savings")

    batch = next(iter(val_dataset))
    batch = ops.concatenate([batch, batch, batch], axis=-1)  # grayscale to RGB

    output = model(batch[..., 0][..., None])
    # model.save_to_preset("./test_model_savings")

    mse = ops.convert_to_numpy(ops.mean((output - batch) ** 2))
    print("MSE: ", mse)

    output = (output + 1) / 2
    batch = (batch + 1) / 2

    plot_list = ops.unstack(batch) + ops.unstack(output)
    fig, _ = plot_image_grid(plot_list, vmin=0, vmax=1, ncols=n_imgs)
    path = f"test_taesd_{get_date_string()}.png"
    fig.savefig(
        path,
        pad_inches=0.2,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
    log.info(f"Saved to {log.yellow(path)}")
    del val_dataset  # weird tf datasets bug if not deleted
