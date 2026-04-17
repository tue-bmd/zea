import json
import os
import urllib.request
import zipfile
from pathlib import Path

import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

from zea import log

# Girder API base URL shared by CAMUS and CETUS collections
GIRDER_API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"


def sitk_load(filepath: str | Path, squeeze: bool = False):
    """Load a NIfTI/medical image using SimpleITK and return the array and metadata.

    Args:
        filepath: Path to the image file.
        squeeze: If True, squeeze singleton dimensions from the array.
            Defaults to False.

    Returns:
        Tuple of:
            - Image array. Shape depends on the input and the ``squeeze`` parameter.
            - Dictionary of metadata: ``origin``, ``spacing``, ``direction``, ``size``,
              ``dimension``, and a ``metadata`` sub-dict with all image metadata keys.
    """
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is not installed. "
            "Please install it with `pip install SimpleITK` to use this function."
        ) from exc

    image = sitk.ReadImage(str(filepath))

    all_metadata = {}
    for k in image.GetMetaDataKeys():
        all_metadata[k] = image.GetMetaData(k)

    metadata = {
        "origin": image.GetOrigin(),
        "spacing": image.GetSpacing(),
        "direction": image.GetDirection(),
        "size": image.GetSize(),
        "dimension": image.GetDimension(),
        "metadata": all_metadata,
    }

    im_array = sitk.GetArrayFromImage(image)
    if squeeze:
        im_array = np.squeeze(im_array)
    return im_array, metadata


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


def download_from_girder(  # pragma: no cover
    collection_id: str,
    destination: str | Path,
    dataset_name: str,
    patients: list[int] | None = None,
    top_folder_name: str = "dataset",
) -> Path:
    """Download a dataset from the Girder server.

    Navigates the Girder collection to find patient folders and downloads
    all files for each patient. Existing files are skipped.

    Args:
        collection_id: Girder collection ID for the dataset.
        destination: Directory where the dataset will be downloaded.
        dataset_name: Human-readable name used in log messages
            (e.g. ``"CAMUS"`` or ``"CETUS"``).
        patients: Optional list of patient IDs to download.
            If None, all patients in the collection are downloaded.
        top_folder_name: Name of the top-level folder inside the collection
            that contains patient subfolders. Defaults to ``"dataset"``.

    Returns:
        Path to the downloaded dataset directory.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    timeout = int(os.getenv("ZEA_DOWNLOAD_TIMEOUT", "60"))

    # Get top-level folders in the collection
    url = f"{GIRDER_API}/folder?parentType=collection&parentId={collection_id}&limit=50"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        folders = json.loads(resp.read())

    dataset_folder_id = None
    for folder in folders:
        if folder["name"] == top_folder_name:
            dataset_folder_id = folder["_id"]
            break

    if dataset_folder_id is None:
        raise RuntimeError(
            f"Could not find '{top_folder_name}' folder in {dataset_name} collection."
        )

    # Get patient folders (paginated — some datasets have >50 patients)
    patient_folders = []
    offset = 0
    page_size = 50
    while True:
        url = (
            f"{GIRDER_API}/folder?parentType=folder&parentId={dataset_folder_id}"
            f"&limit={page_size}&offset={offset}"
        )
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            page = json.loads(resp.read())
        if not page:
            break
        patient_folders.extend(page)
        if len(page) < page_size:
            break
        offset += page_size

    if patients is not None:
        patient_set = set(patients)
        patient_folders = [
            pf for pf in patient_folders if int(pf["name"].removeprefix("patient")) in patient_set
        ]

    log.info(f"Downloading {len(patient_folders)} patients from {dataset_name} dataset...")

    for pf in tqdm(patient_folders, desc="Downloading patients"):
        patient_name = pf["name"]
        patient_dir = destination / patient_name
        patient_dir.mkdir(parents=True, exist_ok=True)

        # Get items (files) in the patient folder
        url = f"{GIRDER_API}/item?folderId={pf['_id']}&limit=50"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            items = json.loads(resp.read())

        for item in items:
            file_path = patient_dir / item["name"]
            if file_path.exists():
                log.debug(f"File {file_path} already exists when downloading. Skipping.")
                continue

            download_url = f"{GIRDER_API}/item/{item['_id']}/download"
            log.debug(f"Downloading {item['name']}...")
            with urllib.request.urlopen(download_url, timeout=timeout) as resp:
                file_path.write_bytes(resp.read())

    log.info(f"{dataset_name} dataset downloaded to {destination}")
    return destination
