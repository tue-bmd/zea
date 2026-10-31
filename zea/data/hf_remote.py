"""
zea.data.hf_remote
==================

Resolve streamed ``hf://`` files to their CDN URL once, instead of on every read.

Each request to a Hub ``/resolve/`` URL counts toward the Hub rate limit (see
https://huggingface.co/docs/hub/rate-limits). The Hub answers it with a redirect to a signed
CDN URL, and requests to that CDN URL are not rate limited. Before this module, every chunk
read went through ``/resolve/`` (about 8 requests per raw-data frame), so multi-worker
training quickly hit HTTP 429.

We now do what :func:`huggingface_hub.hf_hub_download` does: one HEAD request per file via
:func:`huggingface_hub.get_hf_file_metadata`, then range requests directly to the returned
``location``, without the ``Authorization`` header (the token is for the Hub, not the CDN).

- The signed URL is treated as opaque. It is reused for :data:`URL_TTL` seconds and
  resolved again when it expires or the CDN rejects it.
- Re-resolving uses the commit of the first resolve, so the file cannot change while open.
- Resolved URLs are cached in memory and on disk, so all threads and processes on a machine
  (e.g. one process per GPU) share them. The disk cache files are only readable by the user.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from zea import log
from zea.internal.cache import ZEA_CACHE_DIR
from zea.internal.utils import atomic_write

#: Seconds to reuse a resolved URL. The Hub's signed URLs currently expire after about an
#: hour. If that changes, a rejected URL is resolved again anyway.
URL_TTL = 30 * 60

#: HTTP statuses the CDN returns for an expired or invalid URL.
REFUSED_STATUS = frozenset({401, 403, 404, 410})

_CACHE_DIR = Path(ZEA_CACHE_DIR) / "hf_locations"

#: Byte budget of the in-memory blocks shared by all open handles of the same file (see
#: :class:`_SharedBlocks`).
SHARED_BLOCK_BYTES = 256 << 20  # 256 MiB

_memory: dict[tuple, "Resolved"] = {}
_memory_lock = threading.Lock()
_key_locks: dict[tuple, threading.Lock] = {}


@dataclass(frozen=True)
class Resolved:
    """The CDN URL and metadata of one ``hf://`` file, as returned by the Hub."""

    url: str
    size: int
    etag: str | None
    commit: str | None
    #: Send the Hub token with requests to ``url``. Only true if ``url`` is on the Hub itself.
    send_auth: bool
    resolved_at: float

    @property
    def fresh(self) -> bool:
        return time.time() - self.resolved_at < URL_TTL


def _token_id(token: str | None) -> str:
    """Hash of the token, so cached URLs are kept per user."""
    return hashlib.sha256(token.encode()).hexdigest()[:16] if token else "anonymous"


def _origin(url: str) -> tuple[str, str]:
    parts = urlsplit(url)
    return parts.scheme, parts.netloc


def _disk_path(key: tuple) -> Path:
    return _CACHE_DIR / f"{hashlib.sha256(repr(key).encode()).hexdigest()}.json"


def _load(key: tuple) -> Resolved | None:
    try:
        resolved = Resolved(**json.loads(_disk_path(key).read_text()))
    except (OSError, ValueError, TypeError):
        return None
    return resolved if resolved.fresh else None


def _store(key: tuple, resolved: Resolved) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # atomic_write goes through mkstemp, so the file is private to this user (0600).
        with atomic_write(_disk_path(key), suffix=".part") as tmp:
            tmp.write_text(json.dumps(asdict(resolved)))
    except OSError as exc:  # the disk cache is optional
        log.debug(f"Could not cache the location of a streamed file: {exc}")


def resolve(
    hf_path: str,
    revision: str | None = None,
    repo_type: str = "dataset",
    refresh: bool = False,
) -> Resolved:
    """Return the CDN URL of an ``hf://`` file, from cache when possible.

    Args:
        hf_path (str): An ``hf://org/repo/path/to/file`` path to a single file.
        revision (str, optional): Branch, tag or commit hash. Defaults to the default branch.
        repo_type (str, optional): One of ``"dataset"``, ``"model"`` or ``"space"``.
        refresh (bool, optional): Skip the cache and ask the Hub again, e.g. after the CDN
            rejected the cached URL.

    Returns:
        Resolved: The CDN URL, size, etag and commit of the file.
    """
    from huggingface_hub import constants, get_hf_file_metadata, get_token, hf_hub_url

    from zea.internal.preset_utils import _hf_call, _hf_parse_path

    repo_id, subpath = _hf_parse_path(hf_path)
    if not subpath:
        raise ValueError(f"Expected an 'hf://' path to a single file, got '{hf_path}'.")
    token = get_token()
    key = (constants.ENDPOINT, repo_type, repo_id, revision, subpath, _token_id(token))

    with _memory_lock:
        lock = _key_locks.setdefault(key, threading.Lock())
    # One lock per file, so dataloader threads opening the same file send only one request.
    with lock:
        if not refresh:
            with _memory_lock:
                cached = _memory.get(key)
            if cached is None or not cached.fresh:
                cached = _load(key)
            if cached is not None:
                with _memory_lock:
                    _memory[key] = cached
                return cached

        url = hf_hub_url(repo_id, subpath, repo_type=repo_type, revision=revision)
        meta = _hf_call(get_hf_file_metadata, url, retry_on_errors=True)
        if meta.size is None:
            raise OSError(f"The Hub did not report a size for '{hf_path}'; cannot stream it.")
        resolved = Resolved(
            url=meta.location,
            size=int(meta.size),
            etag=meta.etag,
            commit=meta.commit_hash,
            send_auth=_origin(meta.location) == _origin(constants.ENDPOINT),
            resolved_at=time.time(),
        )
        with _memory_lock:
            _memory[key] = resolved
        _store(key, resolved)
        return resolved


def clear() -> None:
    """Clear all cached URLs (in memory and on disk) and shared file blocks."""
    import shutil

    with _memory_lock:
        _memory.clear()
    _shared_blocks.clear()
    shutil.rmtree(_CACHE_DIR, ignore_errors=True)


class SignedURL:
    """The current CDN URL of one open file. Refreshed when it expires or is rejected.

    Shared by the h5py file object (:class:`RemoteFile`) and the chunk reader
    (:class:`~zea.data.chunk_reader.HTTPFetcher`) of the same file.
    """

    def __init__(self, hf_path: str, revision: str | None = None, repo_type: str = "dataset"):
        self.hf_path = hf_path
        self.repo_type = repo_type
        self._lock = threading.Lock()
        self._resolved = resolve(hf_path, revision=revision, repo_type=repo_type)
        # Pin later refreshes to this commit, so the file cannot change while open.
        self._revision = self._resolved.commit or revision

    @property
    def size(self) -> int:
        return self._resolved.size

    @property
    def etag(self) -> str | None:
        return self._resolved.etag

    def _headers(self, resolved: Resolved) -> dict[str, str]:
        if not resolved.send_auth:
            return {}
        from huggingface_hub.utils import build_hf_headers

        return build_hf_headers()

    def current(self) -> tuple[str, dict[str, str]]:
        """Return ``(url, headers)``, refreshing the URL first if it is older than
        :data:`URL_TTL`. May make a request, so do not call it from an event loop.
        """
        resolved = self._resolved
        if not resolved.fresh:
            self.refresh(resolved.url)
            resolved = self._resolved
        return resolved.url, self._headers(resolved)

    def refresh(self, stale_url: str) -> bool:
        """Resolve again after ``stale_url`` was rejected or expired. Returns whether there is
        a new URL.

        Checks the shared cache first to prevent multiple handles of the same file sending
        multiple requests.
        """
        with self._lock:
            if self._resolved.url != stale_url:
                return True
            resolved = resolve(self.hf_path, revision=self._revision, repo_type=self.repo_type)
            if resolved == self._resolved:
                resolved = resolve(
                    self.hf_path, revision=self._revision, repo_type=self.repo_type, refresh=True
                )
            if self._resolved.etag and resolved.etag and resolved.etag != self._resolved.etag:
                raise OSError(
                    f"'{self.hf_path}' changed on the Hub while it was open "
                    f"(etag {self._resolved.etag} -> {resolved.etag}). Reopen the file."
                )
            changed = resolved.url != stale_url
            self._resolved = resolved
            return changed


class _SharedBlocks:
    """LRU of file blocks, shared by every open handle of the same file in this process.

    h5py holds its global lock while it reads file metadata, so a metadata read that waits on
    the network stalls every thread. Dataloader threads each open their own handle of the
    same file, so without sharing, every thread re-fetches the same metadata under that lock.
    Keyed by etag, so a block is never served for different content.
    """

    def __init__(self, max_bytes: int):
        from collections import OrderedDict

        self.max_bytes = max_bytes
        self._blocks: "OrderedDict[tuple, bytes]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: tuple) -> bytes | None:
        with self._lock:
            data = self._blocks.get(key)
            if data is not None:
                self._blocks.move_to_end(key)
            return data

    def put(self, key: tuple, data: bytes) -> None:
        with self._lock:
            if key in self._blocks:
                return
            self._blocks[key] = data
            self._bytes += len(data)
            while self._bytes > self.max_bytes and self._blocks:
                _, evicted = self._blocks.popitem(last=False)
                self._bytes -= len(evicted)

    def clear(self) -> None:
        with self._lock:
            self._blocks.clear()
            self._bytes = 0


_shared_blocks = _SharedBlocks(SHARED_BLOCK_BYTES)


class RemoteFile:
    """Read-only file object over a :class:`SignedURL`, passed to h5py.

    Uses an fsspec block cache so h5py's many small metadata reads become a few requests, and
    shares fetched blocks with other handles of the same file (:class:`_SharedBlocks`).
    Requests go through :class:`~zea.data.chunk_reader.HTTPFetcher` for retries and URL refresh.
    """

    def __init__(self, location: SignedURL, block_size: int, cache_type: str = "blockcache"):
        import fsspec.caching

        from zea.data.chunk_reader import HTTPFetcher

        self.location = location
        self.size = location.size
        self.blocksize = block_size
        #: fsspec-style file info. The chunk cache uses the etag as its key.
        self.details = {"name": location.hf_path, "size": self.size, "etag": location.etag}
        self._fetcher = HTTPFetcher(location)
        self._cache = fsspec.caching.caches[cache_type](block_size, self._fetch, self.size)
        self._pos = 0
        self._closed = False
        # zea files keep their HDF5 metadata at the start. Fetch that block now, before h5py
        # opens the file and takes its global lock, so threads opening different files fetch
        # in parallel instead of one after another.
        if self.size:
            self._fetch(0, min(block_size, self.size))

    def _fetch(self, start: int, end: int) -> bytes:
        key = (self.location.etag, start, end)
        if key[0] is not None and (hit := _shared_blocks.get(key)) is not None:
            return hit
        data = self._fetcher.read_range(start, end - start)
        if key[0] is not None:
            _shared_blocks.put(key, data)
        return data

    def read(self, length: int = -1) -> bytes:
        if length is None or length < 0:
            length = self.size - self._pos
        end = min(self.size, self._pos + length)
        if end <= self._pos:
            return b""
        data = self._cache._fetch(self._pos, end)
        self._pos += len(data)
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: self.size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


def open_remote(
    hf_path: str,
    revision: str | None = None,
    repo_type: str = "dataset",
    block_size: int | None = None,
    cache_type: str | None = None,
) -> RemoteFile:
    """Open an ``hf://`` file for streaming reads."""
    from zea.internal.preset_utils import _HF_STREAM_BLOCK_SIZE, _HF_STREAM_CACHE_TYPE

    location = SignedURL(hf_path, revision=revision, repo_type=repo_type)
    return RemoteFile(
        location,
        block_size=block_size or _HF_STREAM_BLOCK_SIZE,
        cache_type=cache_type or _HF_STREAM_CACHE_TYPE,
    )
