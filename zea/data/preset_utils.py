"""Preset utils for zea datasets hosted on Hugging Face.

See https://huggingface.co/zeahub/
"""

from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, login
from huggingface_hub.utils import (
    EntryNotFoundError,
    HFValidationError,
    RepositoryNotFoundError,
)

from zea.internal.cache import ZEA_CACHE_DIR

HF_DATASETS_DIR = ZEA_CACHE_DIR / "huggingface" / "datasets"
HF_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

HF_SCHEME = "hf"
HF_PREFIX = "hf://"


def _hf_parse_path(hf_path: str):
    """Parse hf://repo_id[/subpath] into (repo_id, subpath or None)."""
    if not hf_path.startswith(HF_PREFIX):
        raise ValueError(f"Invalid hf_path: {hf_path}. It must start with '{HF_PREFIX}'.")
    path = hf_path.removeprefix(HF_PREFIX)
    parts = path.split("/")
    repo_id = "/".join(parts[:2])
    subpath = "/".join(parts[2:]) if len(parts) > 2 else None
    return repo_id, subpath


def _hf_list_files(repo_id):
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
    except (RepositoryNotFoundError, HFValidationError, EntryNotFoundError):
        login(new_session=False)
        files = list_repo_files(repo_id, repo_type="dataset")
    return files


def _hf_download(repo_id, filename, cache_dir=HF_DATASETS_DIR):
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        repo_type="dataset",
    )


def _get_snapshot_dir_from_downloaded_file(downloaded_file_path: str | Path) -> Path:
    """Extract the snapshot directory from a downloaded file's path.

    HF Hub downloads to: cache_dir/datasets--org--repo/snapshots/{hash}/path/to/filename
    This navigates up to find the {hash} directory (the snapshot directory).
    """
    file_path = Path(downloaded_file_path)

    # Navigate up the path until we find the snapshots directory
    current = file_path.parent
    while current.name != "snapshots" and current.parent != current:
        current = current.parent

    if current.name == "snapshots":
        # Return the snapshot hash directory (first subdirectory of snapshots)
        snapshot_dirs = [d for d in current.iterdir() if d.is_dir()]
        if snapshot_dirs:
            # Return the most recent snapshot directory
            return max(snapshot_dirs, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(f"Could not find snapshot directory for {downloaded_file_path}")


def _download_files_in_path(
    repo_id: str, files: list, path_filter: str = None, cache_dir=HF_DATASETS_DIR
) -> list[str]:
    """Download all files matching the path filter."""
    downloaded_files = []
    for f in files:
        if path_filter is None or f.startswith(path_filter):
            downloaded_path = _hf_download(repo_id, f, cache_dir)
            downloaded_files.append(downloaded_path)

    return downloaded_files


def _hf_resolve_path(hf_path: str, cache_dir=HF_DATASETS_DIR) -> Path:
    """Resolve a Hugging Face path to a local cache directory path.

    Downloads files from a HuggingFace dataset repository and returns
    the local path where they are cached. Handles:
    - hf://org/repo/subdir/ - Downloads all files in subdirectory
    - hf://org/repo/file.h5 - Downloads specific file
    - hf://org/repo - Downloads all files in repo
    """
    repo_id, subpath = _hf_parse_path(hf_path)
    files = _hf_list_files(repo_id)

    if subpath:
        # Directory case
        if any(f.startswith(subpath + "/") for f in files):
            downloaded_files = _download_files_in_path(repo_id, files, subpath + "/", cache_dir)
            if not downloaded_files:
                raise FileNotFoundError(f"No files found in directory {subpath}")

            snapshot_dir = _get_snapshot_dir_from_downloaded_file(downloaded_files[0])
            return snapshot_dir / subpath

        # File case
        elif subpath in files:
            downloaded_file = _hf_download(repo_id, subpath, cache_dir)
            return Path(downloaded_file)
        else:
            raise FileNotFoundError(f"{subpath} not found in {repo_id}")
    else:
        # All files in repo
        downloaded_files = _download_files_in_path(repo_id, files, None, cache_dir)
        if not downloaded_files:
            raise FileNotFoundError(f"No files found in repository {repo_id}")

        return _get_snapshot_dir_from_downloaded_file(downloaded_files[0])
