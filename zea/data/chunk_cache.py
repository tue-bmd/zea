"""
zea.data.chunk_cache
====================

On-disk cache of compressed HDF5 chunks fetched over the network.

Streaming re-fetches the same bytes every time: reading the same 5 frames of an ``hf://``
file three times costs the same ~0.9 s three times. The chunks are immutable and we already
fetch at chunk granularity, so they cache exactly.

Keyed by the file's **content hash** (HF hands us ``lfs.sha256`` on open, for free) rather
than its URL, because an ``hf://`` resolve URL points at a mutable ref: re-uploading a file
changes what ``/resolve/main/`` returns while the URL stays the same. Keying on content means
a re-upload simply misses, instead of silently serving stale bytes.

Compressed bytes are stored, not decoded arrays: it is the network that costs, and decoding is
fast and parallel (see :mod:`zea.data.chunk_reader`).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from zea import log
from zea.internal.cache import ZEA_CACHE_DIR

#: Cache is enabled unless ZEA_CHUNK_CACHE=0.
ENABLED = os.environ.get("ZEA_CHUNK_CACHE", "1") != "0"

#: Byte budget for the whole cache. Overrun is pruned oldest-first (by access time).
MAX_BYTES = int(os.environ.get("ZEA_CHUNK_CACHE_SIZE", 10 << 30))  # 10 GiB

#: Prune this often (in writes) rather than on every one — the check has to stat the tree.
PRUNE_EVERY = 64


class ChunkCache:
    """Compressed chunks of one file, on disk, addressed by ``(offset, size)``."""

    def __init__(self, content_id: str, root: str | os.PathLike | None = None):
        self._root = root
        self.dir = Path(root or ZEA_CACHE_DIR) / "chunks" / content_id[:2] / content_id
        self._writes = 0

    def _path(self, offset: int, size: int) -> Path:
        return self.dir / f"{offset}-{size}"

    def get(self, offset: int, size: int) -> bytes | None:
        """The cached bytes of this chunk, or ``None`` on a miss."""
        path = self._path(offset, size)
        try:
            data = path.read_bytes()
        except OSError:  # miss, or pruned out from under us mid-read
            return None
        # A short read means a torn or truncated file: treat it as a miss rather than
        # handing corrupt bytes to the decoder.
        if len(data) != size:
            return None
        try:
            # Mark the hit ourselves. Reading a file is not enough to make eviction correct:
            # Linux mounts with `relatime` refresh atime at most once a day, so an LRU that
            # trusted it would quietly evict by *write* order instead — dropping exactly the
            # chunks a training run keeps re-reading. A utime is nothing against the read.
            os.utime(path, None)
        except OSError:
            pass
        return data

    def put(self, offset: int, size: int, data: bytes) -> None:
        """Store this chunk. Failure to cache is never an error — it just costs a re-fetch."""
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            # Write to a temporary and rename: a reader (possibly another dataloader worker)
            # must never observe a half-written chunk. os.replace is atomic within a filesystem.
            fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".part")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                os.replace(tmp, self._path(offset, size))
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

            self._writes += 1
            if self._writes % PRUNE_EVERY == 0:
                prune(self._root)
        except OSError as exc:  # full disk, read-only cache dir, ...
            log.debug(f"Could not cache chunk: {exc}")


class CachedFile:
    """A read-only file object that serves h5py's metadata reads from the chunk cache.

    Opening a streamed file re-reads its HDF5 metadata (superblock, group and chunk B-trees)
    over the network every time — and HF's CDN intermittently stalls ~3 s on the first range
    request to a cold object, which is then most of the cost of an open. Those metadata bytes
    are as cacheable as the chunks are, so they go in the same cache.

    Reads are served in aligned blocks, since h5py's metadata reads are many and small; a
    block is the unit that gets stored. Chunks and blocks share one keyspace safely: a key is
    ``(offset, size)`` into an immutable file, so the same key always means the same bytes.

    ``block_size`` defaults to the streamed file object's own block size, so a cached open and
    an uncached one fetch the same ranges — a smaller block here would split each of the
    underlying reads into several requests.
    """

    def __init__(self, fileobj, cache: "ChunkCache", block_size: int | None = None):
        from zea.internal.preset_utils import _HF_STREAM_BLOCK_SIZE

        self._file = fileobj
        self._cache = cache
        self._block = block_size or getattr(fileobj, "blocksize", None) or _HF_STREAM_BLOCK_SIZE
        self._pos = 0
        self.size = fileobj.size
        #: Passed through: this is where the chunk fetcher reads the file's content hash from.
        self.details = getattr(fileobj, "details", {})

    def _block_bytes(self, index: int) -> bytes:
        offset = index * self._block
        size = min(self._block, self.size - offset)
        hit = self._cache.get(offset, size)
        if hit is not None:
            return hit
        self._file.seek(offset)
        data = self._file.read(size)
        self._cache.put(offset, size, data)
        return data

    def read(self, length: int = -1) -> bytes:
        if length < 0:
            length = self.size - self._pos
        length = max(0, min(length, self.size - self._pos))
        if not length:
            return b""

        out = bytearray()
        pos = self._pos
        while len(out) < length:
            index = pos // self._block
            within = pos - index * self._block
            block = self._block_bytes(index)
            take = block[within : within + (length - len(out))]
            if not take:
                break  # short block: end of file
            out += take
            pos += len(take)
        self._pos = pos
        return bytes(out)

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
        self._file.close()

    @property
    def closed(self) -> bool:
        return self._file.closed


def cache_for(details: dict, root: str | os.PathLike | None = None) -> ChunkCache | None:
    """A cache for the file described by ``details`` (an fsspec ``info`` mapping).

    ``None`` when caching is off, or when nothing in ``details`` identifies the *content*.
    A weaker key (the path, say) would go stale on re-upload, and a stale chunk cache is worse
    than no chunk cache: it corrupts reads silently.
    """
    if not ENABLED:
        return None

    lfs = details.get("lfs")
    content_id = None
    if isinstance(lfs, dict):
        content_id = lfs.get("sha256")
    content_id = content_id or details.get("blob_id")  # non-LFS files
    if not content_id:
        return None
    return ChunkCache(str(content_id), root)


def prune(root: str | os.PathLike | None = None, max_bytes: int = MAX_BYTES) -> int:
    """Delete least-recently-used chunks until the cache fits in ``max_bytes``.

    Returns the bytes freed. Uses access time, so chunks a training run keeps re-reading
    survive while one-off reads age out.
    """
    base = Path(root or ZEA_CACHE_DIR) / "chunks"
    if not base.is_dir():
        return 0

    entries = []
    total = 0
    for path in base.rglob("*"):
        try:
            stat = path.stat()
        except OSError:
            continue
        if not path.is_file() or path.suffix == ".part":
            continue
        entries.append((stat.st_atime, stat.st_size, path))
        total += stat.st_size

    if total <= max_bytes:
        return 0

    freed = 0
    for _atime, size, path in sorted(entries):  # oldest access first
        try:
            path.unlink()
        except OSError:
            continue
        freed += size
        if total - freed <= max_bytes:
            break
    log.debug(f"Pruned {freed / 1e6:.0f} MB from the chunk cache")
    return freed


def clear(root: str | os.PathLike | None = None) -> None:
    """Remove every cached chunk."""
    import shutil

    shutil.rmtree(Path(root or ZEA_CACHE_DIR) / "chunks", ignore_errors=True)
