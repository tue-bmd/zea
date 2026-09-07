"""Shared Hugging Face plumbing for zea presets (models and datasets).

Both halves of zea resolve ``hf://`` handles: :mod:`zea.models.preset_utils` loads Keras
presets from model repos, and the data stack (:mod:`zea.data.datasets`,
:mod:`zea.data.file`, :mod:`zea.data.file_operations`) loads HDF5 files from dataset
repos. Everything they have in common lives here — handle parsing, authentication,
repository listings, downloads and streaming — so the two sides cannot drift apart.

See https://huggingface.co/zeahub/
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import RepoFile, hf_hub_download, list_repo_tree, login
from huggingface_hub.utils import (
    EntryNotFoundError,
    HFValidationError,
    RepositoryNotFoundError,
)

from zea import log
from zea.internal.cache import ZEA_CACHE_DIR

HF_SCHEME = "hf"
HF_PREFIX = "hf://"

HF_DATASETS_DIR = ZEA_CACHE_DIR / "huggingface" / "datasets"
HF_MODELS_DIR = ZEA_CACHE_DIR / "huggingface" / "models"
for _cache_dir in (HF_DATASETS_DIR, HF_MODELS_DIR):
    _cache_dir.mkdir(parents=True, exist_ok=True)

# Default local cache directory per huggingface_hub ``repo_type``, so a model download
# never lands in the dataset cache (and vice versa) when no ``cache_dir`` is given.
_HF_CACHE_DIRS = {"dataset": HF_DATASETS_DIR, "model": HF_MODELS_DIR}

# Maps huggingface_hub ``repo_type`` values to the path prefix used by both
# :class:`~huggingface_hub.HfFileSystem` and the ``resolve`` download URLs.
_HF_REPO_TYPE_PREFIX = {"dataset": "datasets/", "model": "", "space": "spaces/"}


# ``login()`` writes the token file, and concurrent downloads can all fail
# authentication at once, so only one of them gets to log in.
_LOGIN_LOCK = threading.Lock()


def _hf_login() -> None:
    """Authenticate using a token from the environment, if available.

    Reads ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) and only logs in when a
    token is present. This avoids ``login()`` falling back to an interactive
    prompt in headless environments; cached credentials or anonymous access are
    used when no token is set.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        with _LOGIN_LOCK:
            login(token=token, skip_if_logged_in=True)


def _hf_parse_path(hf_path: str):
    """Parse hf://repo_id[/subpath] into (repo_id, subpath or None)."""
    if not hf_path.startswith(HF_PREFIX):
        raise ValueError(f"Invalid hf_path: {hf_path}. It must start with '{HF_PREFIX}'.")
    path = hf_path.removeprefix(HF_PREFIX).rstrip("/")
    parts = path.split("/")
    repo_id = "/".join(parts[:2])
    subpath = "/".join(parts[2:]) if len(parts) > 2 else None
    return repo_id, subpath


def _hf_repo_type_prefix(repo_type: str) -> str:
    """The path prefix for a ``repo_type``, validated."""
    prefix = _HF_REPO_TYPE_PREFIX.get(repo_type)
    if prefix is None:  # "model" maps to "", so test membership, not truthiness
        raise ValueError(
            f"Unsupported repo_type '{repo_type}'. Expected one of {list(_HF_REPO_TYPE_PREFIX)}."
        )
    return prefix


# The hub answers 404 for repos an anonymous client may not see, so "not found" can
# really mean "not authenticated".
_HF_LOGIN_RETRY_ERRORS = (RepositoryNotFoundError, HFValidationError, EntryNotFoundError)

# A missing *file* in a repo we can reach is a genuine 404, and `check_file_exists`
# needs it reported quickly.
_HF_DOWNLOAD_RETRY_ERRORS = (RepositoryNotFoundError,)


def _hf_call(func, *args, retry_on=_HF_LOGIN_RETRY_ERRORS, **kwargs):
    """Call a hub function, retrying it once after a login attempt.

    :func:`_hf_login` is a no-op without a token and never prompts interactively, so a
    failure it cannot fix is raised again by the retry.
    """
    try:
        return func(*args, **kwargs)
    except retry_on:
        _hf_login()
        return func(*args, **kwargs)


_DEFAULT_HF_CACHE_TTL = 300.0


def _hf_cache_ttl() -> float:
    """How long hub answers may be reused, from ``ZEA_HF_CACHE_TTL`` (seconds)."""
    value = os.environ.get("ZEA_HF_CACHE_TTL")
    if value is None:
        return _DEFAULT_HF_CACHE_TTL
    try:
        return float(value)
    except ValueError:
        log.warning(
            f"Ignoring ZEA_HF_CACHE_TTL={value!r}: expected a number of seconds. "
            f"Falling back to {_DEFAULT_HF_CACHE_TTL}."
        )
        return _DEFAULT_HF_CACHE_TTL


# A single operation asks for the same listing or file repeatedly (a dataset scan lists
# the repo once per path it inspects), so hub answers are reused briefly.
_HF_CACHE_TTL = _hf_cache_ttl()


class _TTLCache:
    """Minimal thread-safe cache whose entries expire after ``ttl`` seconds."""

    def __init__(self, ttl: float):
        self.ttl = ttl
        self._entries: dict = {}
        self._lock = threading.Lock()

    def get_or_call(self, key, func, valid=None):
        """Return the value cached under ``key``, else call ``func()`` and cache it.

        Args:
            key: Hashable cache key, or ``None`` to bypass the cache entirely.
            func (callable): Zero-argument callable producing the value.
            valid (callable, optional): Predicate re-checked on a cache hit; a value
                it rejects (e.g. a downloaded file that has since been deleted) is
                dropped and recomputed.
        """
        if self.ttl <= 0 or key is None:
            return func()

        with self._lock:
            entry = self._entries.get(key)
        if entry is not None:
            expires_at, value = entry
            if expires_at > time.monotonic() and (valid is None or valid(value)):
                return value

        value = func()
        now = time.monotonic()
        with self._lock:
            # The only place expired entries are dropped, so they cannot pile up.
            self._entries = {k: v for k, v in self._entries.items() if v[0] > now}
            self._entries[key] = (now + self.ttl, value)
        return value

    def clear(self):
        """Drop all entries."""
        with self._lock:
            self._entries.clear()


_LISTING_CACHE = _TTLCache(_HF_CACHE_TTL)
_DOWNLOAD_CACHE = _TTLCache(_HF_CACHE_TTL)


def _hf_clear_caches() -> None:
    """Drop the memoized repository listings and resolved download paths."""
    _LISTING_CACHE.clear()
    _DOWNLOAD_CACHE.clear()


def _cache_key(*parts, **kwargs):
    """Build a hashable cache key, or ``None`` if the arguments cannot be hashed."""
    key = tuple(parts) + tuple(sorted(kwargs.items()))
    try:
        hash(key)
    except TypeError:
        return None
    return key


def _hf_repo_files(repo_id, repo_type="dataset", **kwargs) -> dict:
    """Map every file in a repo to its :class:`~huggingface_hub.RepoFile` (memoized).

    ``list_repo_tree`` already carries the sizes and content ids that ``list_repo_files``
    throws away, so one listing serves the callers that only want names
    (:func:`_hf_list_files`), the ones that need sizes (:func:`_hf_list_h5_files`) and
    the ones that need a content id (:func:`_hf_content_id`).

    The returned mapping is shared between callers and must not be mutated.
    """

    def _list():
        entries = _hf_call(list_repo_tree, repo_id, recursive=True, repo_type=repo_type, **kwargs)
        return {entry.path: entry for entry in entries if isinstance(entry, RepoFile)}

    return _LISTING_CACHE.get_or_call(_cache_key(repo_id, repo_type, **kwargs), _list)


def _hf_list_files(repo_id, repo_type="dataset", **kwargs) -> list:
    """List the paths of all files in a Hugging Face repository."""
    return list(_hf_repo_files(repo_id, repo_type=repo_type, **kwargs))


def _hf_download(repo_id, filename, cache_dir=None, repo_type="dataset", **kwargs):
    """Download a single file from a repo and return its local path (memoized).

    Args:
        repo_id (str): The ``{org}/{repo}`` identifier.
        filename (str): Path of the file inside the repository.
        cache_dir (str or Path, optional): Local cache directory. Defaults to the zea
            cache for ``repo_type``.
        repo_type (str, optional): One of ``"dataset"``, ``"model"`` or ``"space"``.
        **kwargs: Forwarded to :func:`~huggingface_hub.hf_hub_download`.
    """
    if cache_dir is None:
        cache_dir = _HF_CACHE_DIRS.get(repo_type, HF_DATASETS_DIR)

    def _download():
        return _hf_call(
            hf_hub_download,
            retry_on=_HF_DOWNLOAD_RETRY_ERRORS,
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            repo_type=repo_type,
            **kwargs,
        )

    if kwargs.get("force_download"):
        # An explicit re-download is exactly what a memoized path would skip.
        return _download()

    key = _cache_key(repo_id, filename, str(cache_dir), repo_type, **kwargs)
    return _DOWNLOAD_CACHE.get_or_call(key, _download, valid=os.path.exists)


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


# Same default huggingface_hub uses for snapshot downloads.
_HF_DOWNLOAD_WORKERS = 8


def _download_files_in_path(
    repo_id: str,
    files: list,
    path_filter: str | None = None,
    cache_dir=None,
    repo_type="dataset",
    **kwargs,
) -> list[str]:
    """Download all files matching the path filter, concurrently.

    Returns the local paths in the same order as the matching entries of ``files``.
    """
    matched = [f for f in files if path_filter is None or f.startswith(path_filter)]

    def _download(filename):
        return _hf_download(repo_id, filename, cache_dir=cache_dir, repo_type=repo_type, **kwargs)

    if len(matched) <= 1:
        return [_download(filename) for filename in matched]

    with ThreadPoolExecutor(max_workers=min(_HF_DOWNLOAD_WORKERS, len(matched))) as pool:
        # ``map`` keeps the input order and raises the first failure, though unlike the
        # sequential loop it lets the downloads already in flight finish first.
        return list(pool.map(_download, matched))


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
    entries = _hf_repo_files(repo_id, repo_type="dataset", **kwargs)

    if subpath and subpath in entries:
        matched = [subpath]
    elif subpath:
        prefix = subpath + "/"
        matched = [f for f in entries if f.startswith(prefix) and f.endswith(_HF_H5_EXTENSIONS)]
    else:
        matched = [f for f in entries if f.endswith(_HF_H5_EXTENSIONS)]

    return [(f, entries[f].size) for f in matched]


def _hf_content_id(hf_path: str, **kwargs) -> str | None:
    """Content id of a single ``hf://`` file, or ``None`` if the repo has no such file.

    The LFS sha256 when the file is stored in LFS (as data files are), else its git blob
    id. Both name the *content*, so they change on re-upload and differ between revisions
    -- which is what makes them usable as a cache key, unlike the mutable path. Served
    from the memoized repo listing, so it costs no extra request.
    """
    repo_id, subpath = _hf_parse_path(hf_path)
    entry = _hf_repo_files(repo_id, repo_type="dataset", **kwargs).get(subpath)
    if entry is None:
        return None
    lfs = getattr(entry, "lfs", None)
    return getattr(lfs, "sha256", None) or entry.blob_id


def _hf_resolve_path(hf_path: str, cache_dir=None, repo_type="dataset", **kwargs) -> Path:
    """Resolve a Hugging Face path to a local cache directory path.

    Downloads files from a HuggingFace dataset repository and returns
    the local path where they are cached. Handles:
    - hf://org/repo/subdir/ - Downloads all files in subdirectory
    - hf://org/repo/file.h5 - Downloads specific file
    - hf://org/repo - Downloads all files in repo

    Note that we also support streaming, so this should not be used that often!
    """
    repo_id, subpath = _hf_parse_path(hf_path)
    files = _hf_list_files(repo_id, repo_type=repo_type, **kwargs)

    if subpath:
        prefix = subpath + "/"
        # Directory case
        if any(f.startswith(prefix) for f in files):
            downloaded_files = _download_files_in_path(
                repo_id,
                files,
                prefix,
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


# This file object only serves h5py's metadata reads; chunk_reader fetches the array chunks
# itself. The paged layout keeps that metadata to ~0.26 MB in one request, so the block just has
# to cover it. Keep it aligned with chunk_cache.CachedFile's block size. Overridable via File().
_HF_STREAM_CACHE_TYPE = "blockcache"
_HF_STREAM_BLOCK_SIZE = 1024 * 1024  # 1 MiB


# Host serving the file bytes. Range requests for chunk reads go straight here rather than
# through :class:`~huggingface_hub.HfFileSystem`, which issues them one at a time.
_HF_HOST = "https://huggingface.co"


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
    prefix = _hf_repo_type_prefix(repo_type)
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

    prefix = _hf_repo_type_prefix(repo_type)
    ref = f"@{revision}" if revision else ""
    fs_path = f"{prefix}{repo_id}{ref}/{subpath}"

    if block_size is None:
        block_size = _HF_STREAM_BLOCK_SIZE
    if cache_type is None:
        cache_type = _HF_STREAM_CACHE_TYPE

    open_kwargs = {"cache_type": cache_type, "block_size": block_size, **kwargs}
    # We never pass a token, so `HfFileSystem` resolves one per request: the retry
    # picks up a login even though fsspec may hand back the same cached instance.
    return _hf_call(lambda: HfFileSystem().open(fs_path, "rb", **open_kwargs))
