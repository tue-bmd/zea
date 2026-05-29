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

    # CAMUS special cases: Girder download produces a database_nifti sub-folder,
    # or the user may have extracted patient* folders directly into src.
    if dataset == "camus":
        if (src / "database_nifti").exists():
            log.info(f"Found database_nifti folder in {src}.")
            return src / "database_nifti"
        if any(src.glob("patient*")):
            log.info(f"Found patient folders directly in {src}.")
            return src

    zip_path = src / zip_name
    if not zip_path.exists():
        raise FileNotFoundError(f"Could not find {zip_name} or {folder_name} folder in {src}.")

    log.info(f"Unzipping {zip_path} to {src}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(src)
    log.info("Unzipping completed.")
    log.info(f"Starting conversion from {src / folder_name}.")
    return unzip_dir


def download_file(url: str, destination: str | Path) -> Path:  # pragma: no cover
    """Download a file from a URL to a local path.

    Skips the download if the file already exists at *destination*.
    Shows a :mod:`tqdm` progress bar based on the ``content-length``
    header when available.

    Uses the ``ZEA_DOWNLOAD_TIMEOUT`` environment variable (default 600 s)
    as the socket timeout.

    Args:
        url: URL to download from.
        destination: Full file path where the downloaded content will be saved.
            The parent directory is created if it does not exist.

    Returns:
        Path to the (possibly pre-existing) downloaded file.
    """
    destination = Path(destination)
    if destination.exists():
        log.info(f"File already exists: {destination.name}. Skipping download.")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    timeout = int(os.getenv("ZEA_DOWNLOAD_TIMEOUT", "600"))
    filename = destination.name
    temp_path = destination.with_name(f"{destination.name}.part")

    if temp_path.exists():
        temp_path.unlink()

    log.info(f"Downloading {filename} ...")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            total_header = response.headers.get("content-length")
            total = int(total_header) if total_header is not None else None
            bytes_written = 0
            with (
                open(temp_path, "wb") as f,
                tqdm(total=total or None, unit="B", unit_scale=True, desc=filename) as progress,
            ):
                while chunk := response.read(8192):
                    f.write(chunk)
                    bytes_written += len(chunk)
                    progress.update(len(chunk))
                f.flush()
                os.fsync(f.fileno())

        if total is not None and bytes_written != total:
            raise IOError(
                f"Downloaded size mismatch for {filename}: "
                f"expected {total} bytes, got {bytes_written}."
            )

        temp_path.replace(destination)
    finally:
        if temp_path.exists() and not destination.exists():
            temp_path.unlink(missing_ok=True)

    log.info(f"Downloaded {filename} to {destination.parent}")
    return destination


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


# ---------------------------------------------------------------------------
# HuggingFace Hub helpers
# ---------------------------------------------------------------------------


def check_output_dir_ownership(folder: "str | Path", repo_id: str) -> None:
    """Raise if *folder* already contains data from a different dataset.

    The check is based on the ``zea_repo_id`` field written into the dataset
    card (``README.md``) by each converter.  A directory is considered *owned*
    by a specific dataset when its README.md contains ``zea_repo_id: <repo_id>``.

    * **Empty or non-existent directory** → passes (first-time run).
    * **Directory with matching README.md** → passes (re-run of same dataset).
    * **Directory with mismatched README.md** → raises :class:`FileExistsError`.
    * **Directory with HDF5 files but no README.md** → raises :class:`FileExistsError`.

    Args:
        folder: Output directory to inspect.
        repo_id: Expected dataset repository ID, e.g. ``"zeahub/picmus"``.

    Raises:
        FileExistsError: If the directory belongs to a different dataset.
    """
    folder = Path(folder)
    readme = folder / "README.md"

    if not folder.exists():
        return  # fresh directory — OK

    if readme.exists():
        if f"zea_repo_id: {repo_id}" not in readme.read_text():
            raise FileExistsError(
                f"Output directory '{folder}' already contains data from a different dataset "
                f"(README.md does not declare 'zea_repo_id: {repo_id}'). "
                "Use a separate output directory for each dataset."
            )
        return  # correct dataset — OK (re-run)

    # No README.md yet — fail only if HDF5 files are present (stale/foreign data)
    if any(folder.rglob("*.hdf5")):
        raise FileExistsError(
            f"Output directory '{folder}' already contains HDF5 files but no dataset "
            "README.md.  Use a separate, empty output directory for each dataset, "
            "or delete this directory to start fresh."
        )


def require_output_dir_ownership(folder: "str | Path", repo_id: str) -> None:
    """Raise if *folder* does not contain a verified dataset card for *repo_id*.

    Used as a pre-flight check before uploading to HuggingFace Hub to prevent
    accidentally uploading files from a different dataset.

    Args:
        folder: Directory to check.
        repo_id: Expected dataset repository ID, e.g. ``"zeahub/picmus"``.

    Raises:
        FileNotFoundError: If no README.md is found.
        ValueError: If the README.md does not match *repo_id*.
    """
    folder = Path(folder)
    readme = folder / "README.md"

    if not readme.exists():
        raise FileNotFoundError(
            f"No README.md found in '{folder}'. Run the conversion step before uploading."
        )
    if f"zea_repo_id: {repo_id}" not in readme.read_text():
        raise ValueError(
            f"'{folder}/README.md' does not declare 'zea_repo_id: {repo_id}'. "
            f"This directory does not appear to contain the '{repo_id}' dataset. "
            "Make sure you are uploading the correct directory."
        )


def write_dataset_card(folder: str | Path, card_content: str) -> Path:  # pragma: no cover
    """Write a HuggingFace dataset card (``README.md``) into *folder*.

    Args:
        folder: Directory where ``README.md`` will be written.
        card_content: Markdown content for the dataset card.

    Returns:
        Path to the written ``README.md`` file.
    """
    folder = Path(folder)
    card_path = folder / "README.md"
    card_path.write_text(card_content)
    return card_path


def upload_dataset_to_hf(  # pragma: no cover
    folder: str | Path,
    repo_id: str,
    revision: str,
    file_glob: str = "*.hdf5",
    commit_message: str | None = None,
) -> None:
    """Upload a converted dataset to a HuggingFace Hub revision branch.

    Upload to the ``main`` branch is intentionally blocked.  After uploading
    to a named revision branch, verify the data manually and then merge the
    branch into ``main`` on the Hugging Face Hub.

    Args:
        folder: Root folder containing the files to upload.
        repo_id: Hugging Face Hub repository ID (e.g. ``"zeahub/picmus"``).
        revision: Target branch name.  Must not be ``"main"``.
        file_glob: Glob pattern for files to include in the size summary.
            Defaults to ``"*.hdf5"``.
        commit_message: Commit message.  Defaults to
            ``"Upload <repo_id> (zea format) to <revision>"``.

    Raises:
        ValueError: If *revision* is ``"main"``.
        FileNotFoundError: If no files matching *file_glob* are found
            under *folder*.
    """
    from huggingface_hub import HfApi, login

    if revision == "main":
        raise ValueError(
            "Upload to 'main' is intentionally blocked. "
            "Upload to a named revision branch instead, then merge into main "
            "manually after verifying the upload on the Hub."
        )

    folder = Path(folder)
    files = sorted(folder.rglob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files matching '{file_glob}' found in {folder}")

    total_size_mb = sum(f.stat().st_size for f in files) / 1e6

    if commit_message is None:
        commit_message = f"Upload {repo_id} (zea format) to {revision}"

    log.info("")
    log.info("=" * 60)
    log.info("  HuggingFace upload summary")
    log.info("=" * 60)
    log.info(f"  Repository : {repo_id}")
    log.info(f"  Branch     : {revision}")
    log.info(f"  Source     : {folder}")
    log.info(f"  Files      : {len(files)}")
    log.info(f"  Total size : {total_size_mb:.1f} MB")
    log.info("=" * 60)
    log.info("")

    answer = input("Proceed with upload? [y/N] ").strip().lower()
    if answer != "y":
        log.info("Upload cancelled.")
        return

    login(new_session=False)
    api = HfApi()

    # Check if the revision (branch) exists; if not, prompt to create it.
    try:
        refs = api.list_repo_refs(repo_id=repo_id, repo_type="dataset")
        branch_names = {b.name for b in refs.branches}
        if revision not in branch_names:
            create = (
                input(
                    f"Revision (branch) '{revision}' does not exist on {repo_id}. Create it? [y/N] "
                )
                .strip()
                .lower()
            )
            if create != "y":
                log.info("Upload cancelled — revision not created.")
                return
            api.create_branch(repo_id=repo_id, branch=revision, repo_type="dataset")
            log.info("Created branch '%s' on %s.", revision, repo_id)
    except Exception as exc:
        log.warning("Could not verify revision existence: %s", exc)

    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        commit_message=commit_message,
    )
    log.info(f"Uploaded to https://huggingface.co/datasets/{repo_id}/tree/{revision}")
