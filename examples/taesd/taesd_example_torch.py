"""
TAESD model from: https://github.com/madebyollin/taesd
run: `pip install diffusers`

- **Author(s)**: Wessel van Nierop
- **Date**: 20/11/2024
"""

import importlib
import sys

import matplotlib.pyplot as plt
import torch
from keras import ops

from usbmd import init_device, log, set_data_paths
from usbmd.backend.tensorflow.dataloader import h5_dataset_from_directory
from usbmd.utils import get_date_string
from usbmd.utils.visualize import plot_image_grid

if __name__ == "__main__":
    if importlib.util.find_spec("diffusers") is None:
        log.error(
            "diffusers not installed. Please install "
            "[diffusers](https://huggingface.co/docs/diffusers/en/index)"
        )
        sys.exit()
    from diffusers import AutoencoderTiny

    # Set up data paths and device
    data_paths = set_data_paths()
    data_root = data_paths["data_root"]
    device = init_device(backend="torch")

    n_imgs = 10
    dtype = torch.float32
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

    # Get model
    vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesdxl", torch_dtype=dtype  # or "madebyollin/taesd"
    ).to(device)

    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"Total number of parameters: {total_params}")

    # Get batch
    batch = next(iter(val_dataset))
    batch = ops.moveaxis(batch, -1, 1)  # channels last to channels first
    batch = ops.concatenate([batch, batch, batch], axis=1)  # grayscale to RGB
    batch = batch.to(device)
    batch = batch.to(dtype)

    with torch.no_grad():
        output = vae(batch).sample

    mse = torch.mean((output - batch) ** 2)
    print("MSE: ", mse.item())

    batch = ops.image.rgb_to_grayscale(batch, data_format="channels_first")
    output = ops.image.rgb_to_grayscale(output, data_format="channels_first")

    batch = ops.squeeze(batch, axis=1)
    output = ops.squeeze(output, axis=1)

    output = output.cpu()
    batch = batch.cpu()

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
