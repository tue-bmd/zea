"""
Example script for using a UNet model to inpaint ultrasound images (EchoNet dataset).

- **Author(s)**: Tristan Stevens
- **Date**: 23/01/2025
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
from keras import ops

from usbmd import init_device, log, set_data_paths
from usbmd.agent.masks import random_uniform_lines
from usbmd.backend.tensorflow.dataloader import h5_dataset_from_directory
from usbmd.models.lpips import LPIPS
from usbmd.models.unet import UNet
from usbmd.utils.visualize import plot_image_grid, set_mpl_style


def plot_unet_example(
    ground_truth,
    corrupted,
    inpainted,
    scores,
    save_path="unet_example.png",
):
    """Plot a grid comparing ground truth, corrupted, inpainted and error images."""
    error = ops.abs(ground_truth - inpainted)
    set_mpl_style()

    batch = ops.concatenate([ground_truth, corrupted, inpainted, error], axis=0)
    batch = ops.convert_to_numpy(batch)

    n_imgs = batch.shape[0] // 4
    cmaps = ["gray"] * (3 * n_imgs) + ["viridis"] * n_imgs

    fig, _ = plot_image_grid(
        batch, vmin=-1, vmax=1, ncols=n_imgs, remove_axis=False, cmap=cmaps
    )

    titles = ["Ground Truth", "Corrupted", "Inpainted", "Error"]
    for i, ax in enumerate(fig.axes[: len(titles) * n_imgs]):
        if i % n_imgs == 0:
            ax.set_ylabel(titles[i // n_imgs])

    # 2nd row of images is the inpainted images
    for ax, lpips_score in zip(fig.axes[n_imgs * 2 : 3 * n_imgs], scores):
        ax.text(
            0.95,
            0.95,
            f"LPIPS: {float(lpips_score):.4f}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            color="yellow",
        )

    fig.savefig(
        save_path,
        pad_inches=0.2,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
    return save_path


if __name__ == "__main__":
    # Set up data paths and device
    data_paths = set_data_paths()
    init_device()

    n_imgs = 8
    val_dataset = h5_dataset_from_directory(
        data_paths.data_root / "USBMD_datasets/echonet/val",
        key="data/image",
        batch_size=n_imgs,
        shuffle=True,
        image_size=[128, 128],
        resize_type="resize",
        image_range=[-60, 0],
        normalization_range=[-1, 1],
        seed=42,
        wrap_in_keras=True,
    )

    presets = list(UNet.presets.keys())
    log.info(f"Available built-in usbmd presets for UNet: {presets}")

    model = UNet.from_preset("unet-echonet-inpainter")

    ground_truth = next(iter(val_dataset))

    # set some columns to zero (75%)
    n_columns = ground_truth.shape[2]
    lines = random_uniform_lines(n_columns // 4, n_columns, n_imgs)

    batch = ground_truth * lines[:, None, :, None]

    inpainted_batch = model(batch)

    # compute lpips metric between ground truth and inpainted images
    lpips = LPIPS.from_preset("lpips")
    lpips_scores = lpips([ground_truth, inpainted_batch])
    lpips_scores = ops.convert_to_numpy(lpips_scores)

    saved_path = plot_unet_example(ground_truth, batch, inpainted_batch, lpips_scores)
    log.info(f"Saved to {log.yellow(saved_path)}")

    del val_dataset  # weird tf datasets bug if not deleted
