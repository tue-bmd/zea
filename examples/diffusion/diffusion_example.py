"""
Example Diffusion Model to generate synthetic echocardiograms

- **Author(s)**: Tristan Stevens
- **Date**: 24/04/2025
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
from keras import ops

from usbmd import init_device, log, set_data_paths
from usbmd.backend.tensorflow.dataloader import h5_dataset_from_directory
from usbmd.models.diffusion import DiffusionModel
from usbmd.ops_v2 import Pipeline
from usbmd.utils.visualize import plot_image_grid, set_mpl_style

if __name__ == "__main__":
    # Set up data paths and device
    data_paths = set_data_paths()
    init_device()

    n_imgs = 16
    val_dataset = h5_dataset_from_directory(
        data_paths.data_root / "USBMD_datasets/echonet/val",
        key="data/image",
        batch_size=n_imgs,
        shuffle=True,
        image_size=[112, 112],
        resize_type="resize",
        image_range=[-60, 0],
        normalization_range=[-1, 1],
        seed=42,
    )

    presets = list(DiffusionModel.presets.keys())
    log.info(f"Available built-in usbmd presets for DiffusionModel: {presets}")

    model = DiffusionModel.from_preset("diffusion-echonet-dynamic")

    samples = model.sample(n_samples=n_imgs, n_steps=90, verbose=True)

    batch = next(iter(val_dataset))

    concatenated = ops.concatenate([batch, samples], axis=0)
    concatenated = ops.squeeze(concatenated, axis=-1)

    # process (scan convert) images
    pipeline = Pipeline.from_config({"operations": ["scan_convert"]})
    parameters = {
        "theta_range": [-0.78, 0.78],  # [-45, 45] in radians
        "rho_range": [0, 1],
    }
    parameters = pipeline.prepare_parameters(**parameters)
    processed_batch = pipeline(data=concatenated, **parameters)["data"]

    # plotting
    set_mpl_style()
    fig, _ = plot_image_grid(processed_batch, vmin=-1, vmax=1, remove_axis=False)
    axes = fig.axes

    axes[0].set_ylabel("Data")
    axes[n_imgs].set_ylabel("Samples")

    path = "diffusion_echonet_example.png"
    fig.savefig(
        path,
        pad_inches=0.2,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
    log.info(f"Saved to {log.yellow(path)}")
    del val_dataset  # weird tf datasets bug if not deleted
