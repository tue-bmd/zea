import zipfile
from pathlib import Path

import imageio
import numpy as np
from PIL import Image

from zea import log


def load_avi(file_path, mode="L"):
    """Load a .avi file and return a numpy array of frames.

    Args:
        filename (str): The path to the video file.
        mode (str, optional): Color mode: "L" (grayscale) or "RGB".
            Defaults to "L".

    Returns:
        numpy.ndarray: Array of frames (num_frames, H, W) or (num_frames, H, W, C)
    """
    frames = []
    with imageio.get_reader(file_path) as reader:
        for frame in reader:
            img = Image.fromarray(frame)
            img = img.convert(mode)
            img = np.array(img)
            frames.append(img)
    return np.stack(frames)


def unzip(src: str | Path, dataset: str) -> Path:
    """
    Checks if data folder exist in src.
    Otherwise, unzip dataset.zip in src.

    Args:
        src (str | Path): The source directory containing the zip file or unzipped folder.
        dataset (str): The name of the dataset to unzip.
            Options are "picmus", "camus", "echonet", "echonetlvh".

    Returns:
        Path: The path to the unzipped dataset directory.
    """
    src = Path(src)
    if dataset == "picmus":
        zip_name = "picmus.zip"
        folder_name = "archive_to_download"
        unzip_dir = src / folder_name
    elif dataset == "camus":
        zip_name = "CAMUS_public.zip"
        folder_name = "CAMUS_public"
        unzip_dir = src / folder_name
    elif dataset == "echonet":
        zip_name = "EchoNet-Dynamic.zip"
        folder_name = "EchoNet-Dynamic"
        unzip_dir = src / folder_name / "Videos"
    elif dataset == "echonetlvh":
        zip_name = "EchoNet-LVH.zip"
        folder_name = "Batch1"
        unzip_dir = src
    else:
        raise ValueError(f"Dataset {dataset} not recognized for unzip.")

    if (src / folder_name).exists():
        if dataset == "echonetlvh":
            # EchoNetLVH dataset unzips into four folders. Check they all exist.
            assert (src / "Batch2").exists(), f"Missing Batch2 folder in {src}."
            assert (src / "Batch3").exists(), f"Missing Batch3 folder in {src}."
            assert (src / "Batch4").exists(), f"Missing Batch4 folder in {src}."
            assert (src / "MeasurementsList.csv").exists(), (
                f"Missing MeasurementsList.csv in {src}."
            )
            log.info(f"Found Batch1, Batch2, Batch3, Batch4 and MeasurementsList.csv in {src}.")
        return unzip_dir

    zip_path = src / zip_name
    if not zip_path.exists():
        raise FileNotFoundError(f"Could not find {zip_name} or {folder_name} folder in {src}.")

    log.info(f"Unzipping {zip_path} to {src}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(src)
    log.info("Unzipping completed.")
    log.info(f"Starting conversion from {src / folder_name}.")
    return unzip_dir
