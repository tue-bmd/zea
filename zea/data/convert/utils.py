import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
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

    Decoding and colour conversion are done with OpenCV, which releases the GIL,
    so calling this from a thread pool actually parallelises across files (unlike a
    per-frame PIL loop, which is GIL-bound). The "L"/"RGB" conversions use the same
    ITU-R 601 luma coefficients as PIL, so results match to within rounding.

    Args:
        file_path (str | Path): The path to the video file.
        mode (str, optional): Color mode: "L" (grayscale) or "RGB".
            Defaults to "L".

    Returns:
        numpy.ndarray: Array of frames (num_frames, H, W) or (num_frames, H, W, C)
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for loading video files. "
            "Please install it with 'pip install opencv-python' or "
            "'pip install opencv-python-headless'."
        ) from exc

    if mode not in ("L", "RGB"):
        raise ValueError(f"Unsupported mode {mode!r}, expected 'L' or 'RGB'.")

    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video file {file_path}")

    frames = []
    try:
        while True:
            ok, frame = cap.read()  # OpenCV decodes to BGR
            if not ok:
                break
            if mode == "L":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise OSError(f"No frames decoded from video file {file_path}")

    return np.stack(frames)


def unzip(src: Path, dst: Path) -> Path:
    """Unzips a .zip file to a directory.

    Will check if the unzip has already been fully completed by checking for a file named
    ".fully_unzipped" in the destination directory.

    Args:
        src (Path): Path to the .zip file to unzip.
        dst (Path): Path to the directory where the files will be unzipped.

    Returns:
        Path: Path to the unzipped directory.
    """

    assert src.suffix == ".zip", f"Source path {src} is not a .zip file."

    already_unzipped_filepath = dst / ".fully_unzipped"

    if already_unzipped_filepath.exists():
        log.info("Files already fully unzipped. Skipping unzipping.")
        return dst

    if dst.exists() and dst.is_dir() and len(list(dst.iterdir())) > 0:
        raise ValueError(
            f"Destination directory {dst} is not empty, but the file {already_unzipped_filepath} "
            "does not exist. Maybe the previous unzip attempt failed. Please remove the directory "
            "and try again."
        )

    if not src.exists():
        raise FileNotFoundError(f"Zip file {src} does not exist.")

    log.info(f"Unzipping {src} to {dst}...")
    dst_root = os.path.realpath(dst)
    with zipfile.ZipFile(src, "r") as zip_ref:
        for member in tqdm(zip_ref.namelist(), desc="Extracting files"):
            target = os.path.realpath(os.path.join(dst_root, member))
            if os.path.commonpath([dst_root, target]) != dst_root:
                raise ValueError(f"Unsafe path in zip archive: {member}")
            zip_ref.extract(member, dst)
    log.info("Unzipping completed.")

    # Create file to indicate all files have been unzipped
    already_unzipped_filepath.touch()

    return dst


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
    allow_patterns: "list[str] | None" = None,
) -> None:
    """Upload a converted dataset to a HuggingFace Hub revision branch.

    Uses :meth:`huggingface_hub.HfApi.upload_large_folder`, the resumable,
    chunked, multi-commit uploader meant for large datasets (many or large
    files).  Upload to the ``main`` branch is intentionally blocked.  After
    uploading to a named revision branch, verify the data manually and then
    merge the branch into ``main`` on the Hugging Face Hub.

    Args:
        folder: Root folder containing the files to upload.
        repo_id: Hugging Face Hub repository ID (e.g. ``"zeahub/picmus"``).
        revision: Target branch name.  Must not be ``"main"``.
        file_glob: Glob pattern for files to include in the size summary.
            Defaults to ``"*.hdf5"``.
        commit_message: Message used only when creating the *revision* branch
            (``upload_large_folder`` generates its own per-commit messages).
            Defaults to ``"Upload <repo_id> (zea format) to <revision>"``.
        allow_patterns: Optional list of glob patterns limiting which files in
            *folder* are uploaded.  When ``None`` (default) the whole folder is
            uploaded.  Use this to scope an upload to specific files.

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

    login()
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

    # upload_large_folder is the resumable, chunked, multi-commit uploader meant
    # for big datasets (many/large files). It manages its own commit messages, so
    # commit_message only affects the branch-creation path above, not the upload.
    api.upload_large_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=allow_patterns,
    )
    log.info(f"Uploaded to https://huggingface.co/datasets/{repo_id}/tree/{revision}")
