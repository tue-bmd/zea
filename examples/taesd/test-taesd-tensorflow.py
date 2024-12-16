"""
TAESD model from: https://github.com/madebyollin/taesd

Script by Wessel
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import matplotlib.pyplot as plt
from keras import ops

from usbmd import init_device, log, set_data_paths
from usbmd.backend.tensorflow.dataloader import h5_dataset_from_directory
from usbmd.models.taesd import TinyDecoder, TinyEncoder
from usbmd.utils import get_date_string
from usbmd.utils.visualize import plot_image_grid

if __name__ == "__main__":
    # Set up data paths and device
    data_paths = set_data_paths()
    data_root = data_paths["data_root"]
    init_device("tensorflow")

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

    # Get model
    encoder = TinyEncoder()
    decoder = TinyDecoder()

    batch = next(iter(val_dataset))
    batch = ops.concatenate([batch, batch, batch], axis=-1)  # grayscale to RGB

    encoded = encoder(batch)

    # NOTE: Here you can compress the encoding a little bit more by going
    # to uint8 like in the original model
    # https://github.com/huggingface/diffusers/blob/cd30820/src/diffusers/models/autoencoders/autoencoder_tiny.py?plain=1#L336-L342 # pylint: disable=line-too-long

    output = decoder(encoded)

    mse = ops.mean((output - batch) ** 2)
    print("MSE: ", mse.numpy())

    batch = ops.image.rgb_to_grayscale(batch, data_format="channels_last")
    output = ops.image.rgb_to_grayscale(output, data_format="channels_last")

    batch = ops.squeeze(batch, axis=-1)
    output = ops.squeeze(output, axis=-1)

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
