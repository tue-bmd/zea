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

from usbmd import init_device, log, make_dataloader, set_data_paths
from usbmd.agent.selection import EquispacedLines
from usbmd.models.diffusion import DiffusionModel
from usbmd.models.echonet import INFERENCE_SIZE
from usbmd.ops import Pipeline, ScanConvert
from usbmd.utils.visualize import plot_image_grid, set_mpl_style

if __name__ == "__main__":
    ## Setup
    data_paths = set_data_paths(local=False)
    init_device()

    img_size = INFERENCE_SIZE  # 112

    ## Dataset
    n_imgs = 8
    val_dataset = make_dataloader(
        data_paths.data_root / "USBMD_datasets/echonet/val",
        key="data/image",
        batch_size=n_imgs,
        shuffle=True,
        image_size=[img_size, img_size],
        resize_type="resize",
        image_range=[-60, 0],
        normalization_range=[-1, 1],
        assert_image_range=False,
        seed=42,
    )
    batch = next(iter(val_dataset))

    ## Measurements
    line_thickness = 2
    factor = 2
    agent = EquispacedLines(
        n_actions=img_size // line_thickness // factor,
        n_possible_actions=img_size // line_thickness,
        img_width=img_size,
        img_height=112,
        batch_size=n_imgs,
    )

    mask = agent.sample()
    mask = ops.expand_dims(mask, axis=-1)

    measurements = ops.where(mask, batch, -1.0)

    ## Diffusion Model
    presets = list(DiffusionModel.presets.keys())
    log.info(f"Available built-in usbmd presets for DiffusionModel: {presets}")

    model = DiffusionModel.from_preset(
        "diffusion-echonet-dynamic",
        guidance="dps",
        operator={"name": "inpainting", "params": {"min_val": -1}},
    )

    ## Prior sampling
    prior_samples = model.sample(n_samples=n_imgs, n_steps=90, verbose=True)

    ## Posterior sampling
    posterior_samples = model.posterior_sample(
        measurements=measurements,
        mask=mask,
        n_steps=90,
        omega=3.0,
    )

    ## Post processing (ScanConvert)
    concatenated = ops.concatenate(
        [prior_samples, batch, measurements, posterior_samples], axis=0
    )
    concatenated = ops.squeeze(concatenated, axis=-1)

    pipeline = Pipeline([ScanConvert(order=1, jit_compile=False)])
    parameters = {
        "theta_range": [-0.78, 0.78],  # [-45, 45] in radians
        "rho_range": [0, 1],
    }
    parameters = pipeline.prepare_parameters(**parameters)
    processed_batch = pipeline(data=concatenated, **parameters)["data"]

    ## Plotting
    set_mpl_style()
    fig, _ = plot_image_grid(
        processed_batch, vmin=-1, vmax=1, remove_axis=False, ncols=len(measurements)
    )
    axes = fig.axes

    titles = ["p(x)", "x", "y", "p(x|y)"]

    for i, title in enumerate(titles):
        axes[i * n_imgs].set_ylabel(title)

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
