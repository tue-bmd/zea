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
  resolved again when it expires or the CDN rejects it. A branch or tag can point at new
  content after an upload, so opening a file by one reuses a cached URL only as long as the
  repo listing is cached (``ZEA_HF_CACHE_TTL``).
- Re-resolving uses the commit of the first resolve, so the file cannot change while open.
- Resolved URLs are cached in memory and on disk, so all threads and processes on a machine
  (e.g. one process per GPU) share them. The disk cache files are only readable by the user.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from zea import log
from zea.internal.cache import ZEA_CACHE_DIR
from zea.internal.utils import load_json_cache, store_json_cache

#: Seconds to reuse a resolved URL. The Hub's signed URLs currently expire after about an
#: hour. If that changes, a rejected URL is resolved again anyway.
URL_TTL = 30 * 60

#: Seconds after a refresh that did not give a new URL (it failed, or the URL is the same,
#: e.g. a Hub URL) during which a request rejected with that URL does not ask the Hub again.
REFRESH_COOLDOWN = 30.0

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


def _reusable(resolved: Resolved | None, revision: str | None) -> bool:
    """Whether a cached ``resolved`` may be used to open the file at ``revision``.

    A commit cannot change, so its URL is reused until it expires. A branch or tag is reused
    no longer than the repo listing, so the content id and the bytes read stay in step.
    """
    if resolved is None or not resolved.fresh:
        return False
    if revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision):
        return True
    from zea.internal.preset_utils import _HF_CACHE_TTL

    return time.time() - resolved.resolved_at < _HF_CACHE_TTL


def _origin(url: str) -> tuple[str, str]:
    parts = urlsplit(url)
    return parts.scheme, parts.netloc


def _disk_path(key: tuple) -> Path:
    return _CACHE_DIR / f"{hashlib.sha256(repr(key).encode()).hexdigest()}.json"


def _load(key: tuple) -> Resolved | None:
    data = load_json_cache(_disk_path(key), URL_TTL)
    if not isinstance(data, dict):
        return None
    try:
        return Resolved(**data)
    except TypeError:  # written by another version of zea
        return None


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

    from zea.internal.preset_utils import _hf_call, _hf_parse_path, _hf_token_id

    repo_id, subpath = _hf_parse_path(hf_path)
    if not subpath:
        raise ValueError(f"Expected an 'hf://' path to a single file, got '{hf_path}'.")
    token = get_token()
    key = (constants.ENDPOINT, repo_type, repo_id, revision, subpath, _hf_token_id(token))

    with _memory_lock:
        lock = _key_locks.setdefault(key, threading.Lock())
    # One lock per file, so dataloader threads opening the same file send only one request.
    with lock:
        if not refresh:
            with _memory_lock:
                cached = _memory.get(key)
            if not _reusable(cached, revision):
                cached = _load(key)
            if cached is not None and _reusable(cached, revision):
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
        store_json_cache(_disk_path(key), asdict(resolved))
        return resolved


def clear() -> None:
    """Clear all cached URLs (in memory and on disk) and shared file blocks."""
    import shutil

    with _memory_lock:
        _memory.clear()
    _shared_blocks.clear()
    shutil.rmtree(_CACHE_DIR, ignore_errors=True)


class FileChangedError(OSError):
    """The content of a file changed on the Hub while it was open."""


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
        #: ``(stale_url, time)`` of the last refresh, see :data:`REFRESH_COOLDOWN`.
        self._last_refresh: tuple[str, float] | None = None

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

        The refresh is early (signed URLs last longer than :data:`URL_TTL`), so if it fails
        the current URL is used. If that has expired too, the CDN rejects it and
        :meth:`refresh` is called again.
        """
        resolved = self._resolved
        if not resolved.fresh:
            try:
                self.refresh(resolved.url)
            except FileChangedError:
                raise
            except Exception as exc:  # noqa: BLE001 — any failure to reach the Hub
                log.debug(f"Could not refresh the URL of '{self.hf_path}', reusing it: {exc}")
            resolved = self._resolved
        return resolved.url, self._headers(resolved)

    def refresh(self, stale_url: str) -> bool:
        """Resolve again after ``stale_url`` was rejected or expired. Returns whether there is
        a new URL.

        Checks the shared cache first to prevent multiple handles of the same file sending
        multiple requests, and asks the Hub at most once per :data:`REFRESH_COOLDOWN` for the
        same ``stale_url``, so many rejected requests at once send one request.
        """
        with self._lock:
            if self._resolved.url != stale_url:
                return True
            last = self._last_refresh
            if last is not None and last[0] == stale_url:
                if time.time() - last[1] < REFRESH_COOLDOWN:
                    return False
            # Recorded before asking, so a failed request also holds off the next one.
            self._last_refresh = (stale_url, time.time())
            resolved = resolve(self.hf_path, revision=self._revision, repo_type=self.repo_type)
            if resolved == self._resolved:
                resolved = resolve(
                    self.hf_path, revision=self._revision, repo_type=self.repo_type, refresh=True
                )
            if self._resolved.etag and resolved.etag and resolved.etag != self._resolved.etag:
                raise FileChangedError(
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

    Reads whole blocks, so h5py's many small metadata reads become a few requests, and keeps
    them in :class:`_SharedBlocks`, shared with other handles of the same file. With
    ``keep_blocks=False`` it reads exactly the requested bytes instead. Requests go through
    :class:`~zea.data.chunk_reader.HTTPFetcher` for retries and URL refresh.
    """

    def __init__(self, location: SignedURL, block_size: int, keep_blocks: bool = True):
        from zea.data.chunk_reader import HTTPFetcher

        self.location = location
        self.size = location.size
        self.blocksize = block_size
        #: fsspec-style file info. The chunk cache uses the etag as its key.
        self.details = {"name": location.hf_path, "size": self.size, "etag": location.etag}
        self._fetcher = HTTPFetcher(location)
        #: Whether reads keep the blocks they fetch, so reading one ahead pays off.
        self.keeps_blocks = keep_blocks
        # Without an etag, blocks of other handles may be other content: keep them per handle.
        self._blocks_key = location.etag or object()
        self._pos = 0
        self._closed = False

    def _block(self, start: int) -> bytes:
        end = min(start + self.blocksize, self.size)
        key = (self._blocks_key, start, end)
        if (hit := _shared_blocks.get(key)) is not None:
            return hit
        data = self._fetcher.read_range(start, end - start)
        _shared_blocks.put(key, data)
        return data

    def read_at(self, offset: int, size: int) -> bytes:
        """Read ``size`` bytes at ``offset`` without moving the file position. Thread-safe."""
        end = min(self.size, offset + size)
        if end <= offset:
            return b""
        if not self.keeps_blocks:
            return self._fetcher.read_range(offset, end - offset)
        out = bytearray()
        pos = offset
        while pos < end:
            start = pos - pos % self.blocksize
            block = self._block(start)
            out += block[pos - start : end - start]
            pos = start + len(block)
        return bytes(out)

    def read(self, length: int = -1) -> bytes:
        if length is None or length < 0:
            length = self.size - self._pos
        data = self.read_at(self._pos, length)
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
    """Open an ``hf://`` file for streaming reads.

    ``cache_type="none"`` reads exactly the requested bytes; any other fsspec cache type
    reads and keeps whole blocks (see :class:`RemoteFile`).
    """
    import fsspec.caching

    from zea.internal.preset_utils import _HF_STREAM_BLOCK_SIZE

    known = {name for name in fsspec.caching.caches if name}
    if cache_type is not None and cache_type not in known:
        raise ValueError(f"Unknown cache_type {cache_type!r}; expected one of {sorted(known)}.")
    location = SignedURL(hf_path, revision=revision, repo_type=repo_type)
    return RemoteFile(
        location,
        block_size=block_size or _HF_STREAM_BLOCK_SIZE,
        keep_blocks=cache_type != "none",
    )
