"""Preset utils for zea datasets hosted on Hugging Face.

See https://huggingface.co/zeahub/
"""

import os
from pathlib import Path

from huggingface_hub import RepoFile, hf_hub_download, list_repo_files, list_repo_tree, login
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


def _hf_login() -> None:
    """Authenticate using a token from the environment, if available.

    Reads ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) and only logs in when a
    token is present. This avoids ``login()`` falling back to an interactive
    prompt in headless environments; cached credentials or anonymous access are
    used when no token is set.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token, skip_if_logged_in=True)


def _hf_parse_path(hf_path: str):
    """Parse hf://repo_id[/subpath] into (repo_id, subpath or None)."""
    if not hf_path.startswith(HF_PREFIX):
        raise ValueError(f"Invalid hf_path: {hf_path}. It must start with '{HF_PREFIX}'.")
    path = hf_path.removeprefix(HF_PREFIX)
    parts = path.split("/")
    repo_id = "/".join(parts[:2])
    subpath = "/".join(parts[2:]) if len(parts) > 2 else None
    return repo_id, subpath


def _hf_list_files(repo_id, repo_type="dataset", **kwargs):
    try:
        files = list_repo_files(repo_id, repo_type=repo_type, **kwargs)
    except (RepositoryNotFoundError, HFValidationError, EntryNotFoundError):
        _hf_login()
        files = list_repo_files(repo_id, repo_type=repo_type, **kwargs)
    return files


def _hf_download(repo_id, filename, cache_dir=HF_DATASETS_DIR, repo_type="dataset", **kwargs):
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        repo_type=repo_type,
        **kwargs,
    )


def _get_snapshot_dir_from_downloaded_file(downloaded_file_path: str | Path) -> Path:
    """Extract the snapshot directory from a downloaded file's path.

    HF Hub downloads to: cache_dir/datasets--org--repo/snapshots/{hash}/path/to/filename
    This navigates up to find the {hash} directory (the snapshot directory).
    """
    current = Path(downloaded_file_path).parent
    while current.parent != current:
        if current.parent.name == "snapshots":
            return current
        current = current.parent
    raise FileNotFoundError(f"Could not find snapshot directory for {downloaded_file_path}")


def _download_files_in_path(
    repo_id: str,
    files: list,
    path_filter: str | None = None,
    cache_dir=HF_DATASETS_DIR,
    repo_type="dataset",
    **kwargs,
) -> list[str]:
    """Download all files matching the path filter."""
    downloaded_files = []
    for f in files:
        if path_filter is None or f.startswith(path_filter):
            downloaded_path = _hf_download(
                repo_id,
                f,
                cache_dir=cache_dir,
                repo_type=repo_type,
                **kwargs,
            )
            downloaded_files.append(downloaded_path)

    return downloaded_files


_HF_H5_EXTENSIONS = (".hdf5", ".h5")


def _hf_list_h5_files(hf_path: str, **kwargs) -> list[tuple[str, int]]:
    """List HDF5 files with sizes for an HF path (no download).

    Returns a list of ``(filename_relative_to_repo_root, size_bytes)`` tuples.
    Only .h5 / .hdf5 files are included; other repo files are ignored.

    Handles:
    - hf://org/repo           — all .h5/.hdf5 files in the repo
    - hf://org/repo/subdir    — all .h5/.hdf5 files under subdir/
    - hf://org/repo/file.h5   — [(file.h5, size)] if it exists as a single file
    """
    repo_id, subpath = _hf_parse_path(hf_path)
    try:
        entries = {
            e.path: e.size
            for e in list_repo_tree(repo_id, recursive=True, repo_type="dataset", **kwargs)
            if isinstance(e, RepoFile)
        }
    except (RepositoryNotFoundError, HFValidationError, EntryNotFoundError):
        _hf_login()
        entries = {
            e.path: e.size
            for e in list_repo_tree(repo_id, recursive=True, repo_type="dataset", **kwargs)
            if isinstance(e, RepoFile)
        }

    all_files = list(entries)

    if subpath and any(f == subpath for f in all_files):
        matched = [subpath]
    elif subpath:
        prefix = subpath.rstrip("/") + "/"
        matched = [f for f in all_files if f.startswith(prefix) and f.endswith(_HF_H5_EXTENSIONS)]
    else:
        matched = [f for f in all_files if f.endswith(_HF_H5_EXTENSIONS)]

    return [(f, entries[f]) for f in matched]


def _hf_resolve_path(
    hf_path: str, cache_dir=HF_DATASETS_DIR, repo_type="dataset", **kwargs
) -> Path:
    """Resolve a Hugging Face path to a local cache directory path.

    Downloads files from a HuggingFace dataset repository and returns
    the local path where they are cached. Handles:
    - hf://org/repo/subdir/ - Downloads all files in subdirectory
    - hf://org/repo/file.h5 - Downloads specific file
    - hf://org/repo - Downloads all files in repo
    """
    repo_id, subpath = _hf_parse_path(hf_path)
    files = _hf_list_files(
        repo_id,
        repo_type=repo_type,
        **kwargs,
    )

    if subpath:
        # Directory case
        if any(f.startswith(subpath + "/") for f in files):
            downloaded_files = _download_files_in_path(
                repo_id,
                files,
                subpath + "/",
                cache_dir=cache_dir,
                repo_type=repo_type,
                **kwargs,
            )
            if not downloaded_files:
                raise FileNotFoundError(f"No files found in directory {subpath}")

            snapshot_dir = _get_snapshot_dir_from_downloaded_file(downloaded_files[0])
            return snapshot_dir / subpath

        # File case
        elif subpath in files:
            downloaded_file = _hf_download(
                repo_id,
                subpath,
                cache_dir=cache_dir,
                repo_type=repo_type,
                **kwargs,
            )
            return Path(downloaded_file)
        else:
            raise FileNotFoundError(f"{subpath} not found in {repo_id}")
    else:
        # All files in repo
        downloaded_files = _download_files_in_path(
            repo_id,
            files,
            None,
            cache_dir=cache_dir,
            repo_type=repo_type,
            **kwargs,
        )
        if not downloaded_files:
            raise FileNotFoundError(f"No files found in repository {repo_id}")

        return _get_snapshot_dir_from_downloaded_file(downloaded_files[0])
