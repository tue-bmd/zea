"""
Example script for using the LPIPS model to compute perceptual similarity between images.

- **Author(s)**     : Tristan Stevens
- **Date**          : 20/01/2025
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import torch
from keras import ops

from usbmd import init_device, log, set_data_paths
from usbmd.backend.tensorflow.dataloader import h5_dataset_from_directory
from usbmd.models.lpips import LPIPS
from usbmd.utils.visualize import plot_image_grid, set_mpl_style

if __name__ == "__main__":
    TEST_WITH_TORCH = False  # set to True to test with torch variant
    # Set up data paths and device
    data_paths = set_data_paths()
    device = init_device()

    n_imgs = 9
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

    model = LPIPS.from_preset("lpips")

    batch = next(iter(val_dataset))
    batch = ops.tile(batch, [1, 1, 1, 3])  # to RGB (only required for torchmetrics)

    reference_image = batch[0]
    example_images = batch[1:]

    # Compute LPIPS similarity between reference image and example images

    # process one at a time (so we consume less memory)
    lpips_scores = []
    for example_image in example_images:
        score = model((reference_image, example_image))
        lpips_scores.append(score)
    lpips_scores = ops.concatenate(lpips_scores, axis=0)

    lpips_scores = ops.convert_to_numpy(lpips_scores)

    if TEST_WITH_TORCH:
        # test with torch variant to see if it is exactly the same
        # note that backend should be set to "torch" for this to work
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

        torch_lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg",
            normalize=False,
        )

        # channel first
        reference_image_torch = ops.transpose(reference_image, (2, 0, 1))
        example_images_torch = ops.transpose(example_images, (0, 3, 1, 2))

        reference_image_torch = ops.convert_to_numpy(reference_image_torch)
        example_images_torch = ops.convert_to_numpy(example_images_torch)
        reference_image_torch = torch.tensor(reference_image_torch).to("cpu")
        example_images_torch = torch.tensor(example_images_torch).to("cpu")

        torch_lpips = torch_lpips.to("cpu")

        # process in batches
        torch_lpips_scores = []
        for example_image in example_images_torch:
            score = torch_lpips(
                reference_image_torch[None, ...], example_image[None, ...]
            )
            torch_lpips_scores.append(score)
        torch_lpips_scores = ops.convert_to_numpy(torch_lpips_scores)

        # check if the scores are the same
        np.testing.assert_array_almost_equal(
            lpips_scores,
            torch_lpips_scores,
            decimal=4,
            err_msg=(
                "LPIPS scores are not the same as the torch variant. "
                "Please report this issue."
            ),
        )
        log.success("LPIPS scores are the same as the torch variant.")

    # plotting
    set_mpl_style()

    batch = ops.convert_to_numpy(batch)
    # to 255 range and uint8
    batch = ((batch + 1) / 2 * 255).astype(np.uint8)

    fig, _ = plot_image_grid(batch, remove_axis=False)
    axes = fig.axes[:n_imgs]

    # put a red border around the first image
    for spine in axes[0].spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(2)
        spine.set_linestyle("--")

    # put a green border around the most similar image
    most_similar_idx = lpips_scores.argmin()
    for spine in axes[most_similar_idx + 1].spines.values():
        spine.set_edgecolor("green")
        spine.set_linewidth(2)
        spine.set_linestyle("--")

    # also plot LPIPS scores on each image
    # as small text in the top right corner
    for ax, lpips_score in zip(axes[1:], lpips_scores):
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

    # title
    fig.suptitle(
        "Reference image (red border) and most similar image (green border)\n"
        "according to LPIPS similarity",
        fontsize=12,
    )

    path = "lpips_example.png"
    fig.savefig(
        path,
        pad_inches=0.2,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
    log.info(f"Saved to {log.yellow(path)}")
    del val_dataset  # weird tf datasets bug if not deleted
