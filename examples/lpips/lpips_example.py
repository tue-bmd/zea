"""
Example script for using the LPIPS model to compute perceptual similarity between images.

- **Author(s)**     : Tristan Stevens
- **Date**          : 20/01/2025
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import ops

from usbmd import init_device, log, set_data_paths
from usbmd.backend.tensorflow.dataloader import h5_dataset_from_directory
from usbmd.models.lpips import LPIPS
from usbmd.utils.visualize import plot_image_grid, set_mpl_style

if __name__ == "__main__":
    TEST_WITH_TORCH = False  # set to True to test with torch variant
    # Set up data paths and device
    data_paths = set_data_paths()
    init_device()

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

    model = LPIPS.from_preset("lpips")

    batch = next(iter(val_dataset))
    # to rgb
    batch = ops.tile(batch, [1, 1, 1, 3])

    reference_image = batch[0]
    example_images = batch[1:]

    # Compute LPIPS similarity between reference image and example images
    reference_images = ops.expand_dims(reference_image, axis=0)
    reference_images = ops.tile(reference_images, [n_imgs - 1, 1, 1, 1])
    lpips_scores = model((reference_images, example_images))
    lpips_scores = ops.convert_to_numpy(lpips_scores)

    if TEST_WITH_TORCH:
        assert keras.backend.backend() == "torch", "This test requires torch backend."
        # test with torch variant to see if it is exactly the same
        # note that backend should be set to "torch" for this to work
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

        torch_lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=False, reduction="mean"
        )

        # channel first
        reference_images_torch = ops.transpose(reference_images, (0, 3, 1, 2))
        example_images_torch = ops.transpose(example_images, (0, 3, 1, 2))

        torch_lpips.to(reference_images_torch.device)
        torch_lpips_scores = torch_lpips(
            reference_images_torch,
            example_images_torch,
        )
        torch_lpips_scores = ops.convert_to_numpy(torch_lpips_scores)

        # check if the scores are the same
        assert np.mean(lpips_scores) == torch_lpips_scores
        log.success("LPIPS scores are the same as the torch variant.")

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
