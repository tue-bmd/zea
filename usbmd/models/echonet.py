"""Echonet-Dynamic segmentation model for cardiac ultrasound segmentation.

- **Author(s)**     : Tristan Stevens, adapted from https://echonet.github.io/dynamic/
- **Date**          : 20/11/2023
"""

import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import tqdm
import wget
from eval.generate_masks import read_images_evaluation_folder, read_masks
from matplotlib.animation import FuncAnimation

from usbmd.utils.io_lib import folder_from_window_dialog, load_video
from usbmd.utils.selection_tool import (
    add_shape_from_mask,
    extract_polygon_from_mask,
    reconstruct_mask_from_polygon,
    update_imshow_with_mask,
)

INFERENCE_SIZE = 112

SEGMENTATION_WEIGHTS_URL = (
    "https://github.com/douyang/EchoNetDynamic/releases"
    "/download/v1.0.0/deeplabv3_resnet50_random.pt"
)
EJECTION_FRACTION_WEIGHTS_URL = (
    "https://github.com/douyang/EchoNetDynamic/releases"
    "/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt"
)


class EchoNet:
    """EchoNet-Dynamic segmentation model for cardiac ultrasound segmentation.

    Original paper and code: https://echonet.github.io/dynamic/

    This class extracts useful parts of the original code and wraps it in a
    easy to use class.

    """

    def __init__(self, weights_folder=None, device=None, mean=None, std=None):
        self.file_path = self._download_weights(weights_folder=weights_folder)
        self.mean = mean
        self.std = std

        model = torchvision.models.segmentation.deeplabv3_resnet50(
            pretrained=False, aux_loss=False
        )
        model.classifier[-1] = torch.nn.Conv2d(
            model.classifier[-1].in_channels,
            1,
            kernel_size=model.classifier[-1].kernel_size,
        )

        if torch.cuda.is_available() and device != "cpu":
            print("CUDA is available, original weights loaded for EchoNet")
            self.device = torch.device("cuda")
            self.model = torch.nn.DataParallel(model)
            self.model.to(self.device)
            checkpoint = torch.load(self.file_path)
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            print("CUDA is not available, cpu weights loaded for EchoNet")
            self.device = torch.device("cpu")
            checkpoint = torch.load(
                self.file_path,
                map_location="cpu",
            )
            state_dict_cpu = {k[7:]: v for (k, v) in checkpoint["state_dict"].items()}
            self.model.load_state_dict(state_dict_cpu)

        self.model.eval()

    def __call__(self, x, batch_size=1):
        """Call the model on the input x.
        Args:
            x (np.ndarray / torch.tensor): input array of shape:
                (num_frames, num_channels, height, width)
            batch_size (int): batch size to use for inference
        Returns:
            torch.tensor: boolean segmentation mask of shape:
                (num_frames, height, width)
        """
        num_frames, num_channels, height, width = x.shape
        x = torch.tensor(x, dtype=torch.float32)
        x = self._normalize(x)

        x = torch.nn.functional.interpolate(
            x,
            size=(INFERENCE_SIZE, INFERENCE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        # if input array is grayscale, convert to RGB
        if num_channels == 1:
            x = x.repeat(1, 3, 1, 1)

        with torch.no_grad():
            x = x.to(self.device)
            y = torch.concat(
                [
                    self.model(x[i : (i + batch_size), :, :, :])["out"]
                    for i in range(0, num_frames, batch_size)
                ]
            )
            # interpolate back to original size
            y = torch.nn.functional.interpolate(
                y, size=(height, width), mode="bilinear", align_corners=False
            )
            # binary mask above 0
            y = y > 0

        return y

    def _normalize(self, x):
        """Normalize the input x using the mean and std of the training set."""
        if self.mean is None:
            mean = torch.mean(x)
        else:
            mean = self.mean
        if self.std is None:
            std = torch.std(x)
        else:
            std = self.std
        return (x - mean) / std

    def _download_weights(self, weights_folder=None):
        """Download the weights from the EchoNet github repository."""
        if weights_folder is None:
            weights_folder = "./echo_net_weights"

        weights_folder = Path(weights_folder)
        url = SEGMENTATION_WEIGHTS_URL

        if not Path(weights_folder).exists():
            print(f"Creating folder at {weights_folder} to store weights")
            Path(weights_folder).mkdir()

        assert weights_folder.is_dir(), (
            f"weights_folder {weights_folder} is not a directory. "
            "Please specify the path to the folder containing the weights"
        )

        file_path = weights_folder / Path(url).name
        if not file_path.is_file():
            print(
                "Downloading Segmentation Weights, ",
                url,
                " to ",
                file_path,
            )
            filename = wget.download(url, out=str(weights_folder))

            assert Path(filename).name == Path(url).name, (
                f"Downloaded file {Path(filename).name} does not match expected filename "
                f"{Path(url).name}"
            )
            assert len(list(weights_folder.glob("*.pt"))) != 0, (
                f"No .pt files found in {weights_folder}. "
                "Please make sure the correct weights are downloaded."
            )

        else:
            print(f"EchoNet weights found in {file_path}")
        return file_path


def pngs_to_mask():
    # load model
    model = EchoNet()

    # Define the file paths and image range
    image_folder = Path(
        "C:/Users/s154329/Projects/deep_generative/results/ultrasound/cambridge_invivo_comparison_full/noisy/"
    )
    start_image = 1106
    end_image = 1165

    # Create a list to store the images
    images = []

    # Loop through the image range and read the images
    print(f"Reading images from {image_folder}")
    for i in range(start_image, end_image + 1):
        image_path = image_folder / f"{i}.png"
        image = plt.imread(image_path)
        images.append(image)

    images = np.stack(images)
    images = images[:, None, ...]

    # Pass the images through the model and generate the masks
    masks = model(images, batch_size=10)
    masks = masks.squeeze().cpu().numpy()
    images = images.squeeze()
    if image_folder.stem == "sgm":
        images[images < 0.1] = 0

    fig, ax = plt.subplots()
    # black figure
    fig.set_facecolor("black")

    plt.axis("off")
    imshow_obj = ax.imshow(images[0], cmap="gray")
    fig.tight_layout()

    fps = 50
    ani = FuncAnimation(
        fig,
        update_imshow_with_mask,
        frames=len(images),
        fargs=(ax, imshow_obj, images, masks, "lasso"),
        interval=1000 / fps,
    )

    filename = Path(
        image_folder.parent
        / (
            "animations/"
            + image_folder.stem
            + "_echonet_"
            + f"{start_image}_{end_image}.gif"
        )
    )
    ani.save(filename, writer="pillow")
    print(f"Succesfully saved animation as {filename}")

    plt.show()
    # image = imageio.imread(
    #     "C:/Users/s154329/Projects/deep_generative/results/cambridge_invivo_comparison_full/sgm/421.png"
    # )
    # image = image[None, None, ...]

    # mask = model(image)
    # mask = mask.squeeze().cpu().numpy()

    # fig, ax = plt.subplots()
    # ax.imshow(image.squeeze(), cmap="gray")
    # from usbmd.utils.selection_tool import add_shape_from_mask

    # add_shape_from_mask(ax, mask, color="red", alpha=0.5)
    # ax.axis("off")
    # plt.show()


def plot_masks_echonet(images, masks, save_path):
    fig, ax = plt.subplots()
    # black figure
    fig.set_facecolor("black")

    plt.axis("off")
    imshow_obj = ax.imshow(images[0], cmap="gray")
    fig.tight_layout()

    fps = 50
    ani = FuncAnimation(
        fig,
        update_imshow_with_mask,
        frames=len(images),
        fargs=(ax, imshow_obj, images, masks, "lasso"),
        interval=1000 / fps,
    )
    plt.close()
    ani.save(save_path)
    print(f"Succesfully saved animation as {save_path}")


if __name__ == "__main__":
    model = EchoNet()

    # User enter path
    path = input("Enter the path to the image: ")

    image = plt.imread(path)
    image = image[None, None, ...]

    mask = model(image)
    mask = mask.squeeze().cpu().numpy()
