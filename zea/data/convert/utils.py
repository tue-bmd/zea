import imageio
from PIL import Image
import numpy as np
from pathlib import Path
import sys

from zea import log


def load_avi(file_path, mode="L"):
    """
    Load all frames from an AVI file and return them as a single stacked NumPy array.
    
    Parameters:
        file_path (str | os.PathLike): Path to the AVI file to read.
        mode (str): PIL image mode to convert each frame to (default "L" for 8-bit grayscale).
    
    Returns:
        numpy.ndarray: Stacked array of frames with shape (n_frames, height, width) for single-channel modes
        or (n_frames, height, width, channels) for multi-channel modes. Array values are image pixel values (typically uint8).
    """
    frames = []
    with imageio.get_reader(file_path) as reader:
        for frame in reader:
            img = Image.fromarray(frame)
            img = img.convert(mode)
            img = np.array(img)
            frames.append(img)
    frames = np.stack(frames)
    return frames


def unzip(src: str | Path, dataset: str) -> Path:
    """
    Ensure the specified dataset is available under `src` by verifying the expected folder structure or extracting the corresponding zip archive.
    
    Parameters:
    	src (str | Path): Directory that should contain the dataset folder or the dataset zip file.
    	dataset (str): Dataset identifier; must be one of: "picmus", "camus", "echonet", "echonetlvh".
    
    Returns:
    	Path: The directory to use for the dataset:
    		- picmus: src/archive_to_download
    		- camus: src/CAMUS_public
    		- echonet: src/EchoNet-Dynamic/Videos
    		- echonetlvh: src
    
    Raises:
    	SystemExit: If the dataset is unrecognized or the required zip/folder is missing.
    	AssertionError: For "echonetlvh" if any of Batch2, Batch3, Batch4, or MeasurementsList.csv are missing.
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
        log.error(f"Dataset {dataset} not recognized for unzip.")
        sys.exit(1)

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
        log.error(f"Could not find {zip_name} or {folder_name} folder in {src}.")
        sys.exit()

    import zipfile

    log.info(f"Unzipping {zip_path} to {src}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(src)
    log.info("Unzipping completed.")
    log.info(f"Starting conversion from {src / folder_name}.")
    return unzip_dir