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

    Note that we also support streaming, so this should not be used that often!
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


# Maps huggingface_hub ``repo_type`` values to the path prefix that
# :class:`~huggingface_hub.HfFileSystem` expects.
_HF_FS_PREFIX = {"dataset": "datasets/", "model": "", "space": "spaces/"}


# This file object only serves h5py's metadata reads; chunk_reader fetches the array chunks
# itself. The paged layout keeps that metadata to ~0.26 MB in one request, so the block just has
# to cover it. Keep it aligned with chunk_cache.CachedFile's block size. Overridable via File().
_HF_STREAM_CACHE_TYPE = "blockcache"
_HF_STREAM_BLOCK_SIZE = 1024 * 1024  # 1 MiB


# Host serving the file bytes. Range requests for chunk reads go straight here rather than
# through :class:`~huggingface_hub.HfFileSystem`, which issues them one at a time.
_HF_HOST = "https://huggingface.co"

# The ``resolve`` endpoint of a repo type, as it appears in a download URL.
_HF_URL_PREFIX = {"dataset": "datasets/", "model": "", "space": "spaces/"}


def _hf_stream_url(
    hf_path: str,
    revision: str | None = None,
    repo_type: str = "dataset",
    **kwargs,
) -> str:
    """The HTTPS URL the bytes of an ``hf://`` file live at.

    :func:`_hf_stream_open` streams through :class:`~huggingface_hub.HfFileSystem`, which
    is a *sync* filesystem: its ``cat_ranges`` fetches ranges one after another. Concurrent
    chunk reads (:mod:`zea.data.chunk_reader`) therefore address this URL directly through
    fsspec's async HTTP filesystem instead — measured at one round trip for 16 ranges,
    against sixteen through ``HfFileSystem``.

    Args:
        hf_path (str): An ``hf://org/repo/path/to/file`` path to a single file.
        revision (str, optional): Branch, tag or commit hash. Defaults to the repository
            default branch.
        repo_type (str, optional): One of ``"dataset"``, ``"model"`` or ``"space"``.
        **kwargs: Ignored (accepts the same kwargs as :func:`_hf_stream_open`).

    Returns:
        str: The ``resolve`` URL of the file.
    """
    repo_id, subpath = _hf_parse_path(hf_path)
    if not subpath:
        raise ValueError(f"Expected an 'hf://' path to a single file, got '{hf_path}'.")
    prefix = _HF_URL_PREFIX.get(repo_type)
    if prefix is None:  # "model" maps to "", so test membership, not truthiness
        raise ValueError(
            f"Unsupported repo_type '{repo_type}'. Expected one of {list(_HF_URL_PREFIX)}."
        )
    return f"{_HF_HOST}/{prefix}{repo_id}/resolve/{revision or 'main'}/{subpath}"


def _hf_stream_open(
    hf_path: str,
    revision: str | None = None,
    repo_type: str = "dataset",
    block_size: int | None = None,
    cache_type: str | None = None,
    **kwargs,
):
    """Open a single Hugging Face file lazily for HTTP range-request streaming.

    Unlike :func:`_hf_resolve_path`, this does **not** download the whole file.
    It returns an open fsspec file object backed by
    :class:`~huggingface_hub.HfFileSystem`; only the byte ranges actually read
    (e.g. via ``h5py`` slicing) are fetched over HTTP.

    Args:
        hf_path (str): A ``hf://org/repo/path/to/file`` path pointing at a
            single file (not a repo root or directory).
        revision (str, optional): Branch, tag, or commit hash. Defaults to the
            repository default branch.
        repo_type (str, optional): One of ``"dataset"``, ``"model"`` or
            ``"space"``. Defaults to ``"dataset"``.
        block_size (int, optional): Block size in bytes for the fsspec cache.
            Larger blocks coalesce more chunk reads per HTTP request (faster for
            whole-frame reads, more over-fetch for sparse reads). Defaults to
            :data:`_HF_STREAM_BLOCK_SIZE`.
        cache_type (str, optional): fsspec cache strategy. Defaults to
            :data:`_HF_STREAM_CACHE_TYPE` (``"blockcache"``), which caches touched
            blocks so the many small chunks of a frame share a few requests.
        **kwargs: Forwarded to :meth:`HfFileSystem.open`.

    Returns:
        An open, seekable binary file object. The caller is responsible for
        closing it.
    """
    from huggingface_hub import HfFileSystem

    repo_id, subpath = _hf_parse_path(hf_path)
    if not subpath:
        raise ValueError(
            f"Streaming requires an 'hf://' path to a single file, got '{hf_path}'. "
            "Point at a specific '.hdf5'/'.h5' file, or pass stream=False to download."
        )

    prefix = _HF_FS_PREFIX.get(repo_type)
    if prefix is None:
        raise ValueError(
            f"Unsupported repo_type '{repo_type}'. Expected one of {list(_HF_FS_PREFIX)}."
        )
    ref = f"@{revision}" if revision else ""
    fs_path = f"{prefix}{repo_id}{ref}/{subpath}"

    if block_size is None:
        block_size = _HF_STREAM_BLOCK_SIZE
    if cache_type is None:
        cache_type = _HF_STREAM_CACHE_TYPE

    open_kwargs = {"cache_type": cache_type, "block_size": block_size, **kwargs}
    try:
        return HfFileSystem().open(fs_path, "rb", **open_kwargs)
    except (RepositoryNotFoundError, HFValidationError, EntryNotFoundError):
        _hf_login()
        return HfFileSystem().open(fs_path, "rb", **open_kwargs)
