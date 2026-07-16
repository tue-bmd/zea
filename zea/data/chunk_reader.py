"""
zea.data.chunk_reader
=====================

Concurrent chunk reads for zea HDF5 files, bypassing h5py's serial read path.

h5py reads one chunk at a time and decodes them under a global lock, so N chunks cost N
decodes back to back — and, over HTTP, N round trips. But h5py *does* hand us the chunk
manifest (``get_chunk_info_by_coord``: byte offset, size, filter mask), so we can fetch the
compressed bytes ourselves — concurrently when remote, from the file descriptor when local —
and decode them in a thread pool (Blosc and zlib release the GIL). Measured on a 201 MB read
of 16 chunks: 31 ms against h5py's 291 ms locally, 126 ms against 863 ms over HTTP.

A pure optimisation, and treated as one: anything the fast path does not fully understand (an
unknown codec, a contiguous dataset, an exotic selection) falls back to ``dset[selection]``,
and h5py stays the reader for everything else in the file. Wired in through
:class:`~zea.data.file.ChunkedDataset`, so ``file.data.raw_data[0:8]`` gets it for free.

Two details carry most of the win and are easy to lose in a refactor:

* Bytes are read with ``os.pread``, **not** ``read_direct_chunk`` — that takes h5py's global
  lock, serialising the fetch and copying every chunk an extra time.
* Chunks are decompressed **straight into the output array**. Decoding to a temporary and
  copying it in costs more than the decode itself (121 ms of copy against 26 ms of decode)
  and the copy is serial, so it caps everything.
"""

from __future__ import annotations

import os
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import repeat
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import h5py
import numpy as np

#: Called with the byte size of each chunk as it arrives. Runs on worker threads.
Ticker = Callable[[int], None] | None

# HDF5 filter ids we can decode in-process, via numcodecs and the stdlib. Wider than zea's own
# default so foreign files also get the concurrent path, but it stops at codecs that would cost
# a dependency: Blosc2 and Bitshuffle were measured and rejected — their bindings hold the GIL,
# so they cannot decode concurrently however well they compress. Anything not listed (notably
# lzf) falls back to h5py: correct, just serial.
BLOSC = 32001  # zea's default codec
LZ4 = 32004
ZSTD = 32015
GZIP = 1
SHUFFLE = 2
DECODABLE = (BLOSC, LZ4, ZSTD, GZIP, SHUFFLE)


# Reads below this are served by h5py: the thread hand-off costs more than it saves.
MIN_BYTES = 1 << 20  # 1 MiB

# Ceiling on compressed bytes held in memory at once. Bounded by bytes and not by chunk
# *count*, because chunk sizes vary hugely across files (12 MB to 166 MB).
MAX_BYTES_IN_FLIGHT = 512 << 20  # 512 MiB

# Decode threads. Blosc and zlib release the GIL, so these scale with cores; but the work
# is memory-bandwidth-bound long before it is core-bound, hence the cap.
MAX_WORKERS = min(16, (os.cpu_count() or 4))


class _Unsupported(Exception):
    """The fast path does not understand this selection; use h5py instead."""


# --------------------------------------------------------------------------- #
# Fetchers: where a chunk's compressed bytes come from
# --------------------------------------------------------------------------- #
class Fetcher:
    """Source of raw (still-compressed) chunk bytes for one open file.

    The two backends want opposite things. A local file wants ``per_chunk``: one chunk at a
    time from inside a decode worker, so the next read overlaps the last decode. HTTP wants
    the ranges batched into one call so they go out *together* — N ranges for one round trip,
    which a chunk-at-a-time fetch would throw away.
    """

    #: Whether fetching one chunk on its own is cheap (see above).
    per_chunk = False

    #: Progress reporting for reads through this file (set by :class:`~zea.data.file.File`).
    progress: "bool | Ticker" = False

    #: Human-readable file identifier (the local path or ``hf://`` source url), used in
    #: fallback messages instead of a dataset's internal HDF5 path (set by :func:`fetcher_for`).
    source: "str | None" = None

    def fetch(self, ranges: Sequence[tuple[int, int]], on_bytes: Ticker = None) -> list[bytes]:
        """Return the bytes of each ``(offset, size)`` range, in order.

        ``on_bytes`` is called with the size of each range as it arrives (for progress
        reporting). It runs on whichever thread completed the fetch, so it must be
        thread-safe.
        """
        raise NotImplementedError

    def pending_bytes(self, ranges: Sequence[tuple[int, int]]) -> int:
        """Bytes of ``ranges`` that :meth:`fetch` would actually have to stream or download.

        Defaults to the full total: only :class:`HTTPFetcher` can know better, by checking
        its on-disk cache. Used solely to decide whether a progress bar has anything to show —
        a read served entirely from cache does not stream or download, so it gets no bar.
        """
        return sum(size for _, size in ranges)

    def close(self) -> None:
        """Release whatever the fetcher holds open."""


class LocalFetcher(Fetcher):
    """Reads chunk bytes from the file descriptor.

    ``os.pread`` is positional, so it needs no seek and no lock: reading a raw OS descriptor
    touches none of h5py/HDF5's state, so the decode workers each read the descriptor
    themselves, overlapping I/O with decoding (31 ms against 46 ms for a 16-chunk read).

    ``os.pread`` is Unix-only, though. On Windows it is absent, so we fall back to a
    lock-guarded ``lseek`` + ``read``: the workers share the fd's file position, so the reads
    serialise there, but decoding still overlaps across chunks. (``read_direct_chunk`` under a
    lock would serve here too, but it is slower and drags in HDF5's chunk API for no gain.)
    """

    per_chunk = True

    def __init__(self, path: str | os.PathLike):
        self._fd: int | None = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_BINARY", 0))
        # No lock needed when os.pread is available (its positional read is thread-safe).
        self._lock = None if hasattr(os, "pread") else threading.Lock()

    def _pread(self, offset: int, size: int) -> bytes:
        """Read ``size`` bytes at ``offset``, thread-safely, on any platform."""
        assert self._fd is not None  # fetch() guards against a closed file before calling
        if self._lock is None:
            return os.pread(self._fd, size, offset)
        with self._lock:  # Windows: seek+read is not atomic, so guard the shared position.
            os.lseek(self._fd, offset, os.SEEK_SET)
            return os.read(self._fd, size)

    def fetch(self, ranges: Sequence[tuple[int, int]], on_bytes: Ticker = None) -> list[bytes]:
        if self._fd is None:
            raise ValueError("Cannot read chunks: the file has been closed.")
        out = []
        for offset, size in ranges:
            out.append(self._pread(int(offset), int(size)))
            if on_bytes is not None:
                on_bytes(int(size))
        return out

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class HTTPFetcher(Fetcher):
    """Reads chunk bytes over HTTP range requests, all of them concurrently.

    Deliberately fsspec's **async** ``HTTPFileSystem`` and not ``HfFileSystem``, whose
    ``cat_ranges`` is serial: the same 16 ranges took 2745 ms through ``HfFileSystem`` against
    177 ms here — sixteen round trips against one. The whole remote win rests on this.

    The ranges are issued as individual ``_cat_file`` coroutines gathered on fsspec's event
    loop, rather than through ``cat_ranges``. That is both faster and steadier — ``cat_ranges``
    does its own batching and periodically stalls (measured at 20 ms/request: 32 ranges in
    1.06 s against 0.06 s here) — and it is what makes per-chunk progress possible at all,
    since ``cat_ranges`` only returns once every range is done.
    """

    def __init__(self, url: str, token: str | None = None, cache=None):
        import fsspec

        self.url = url
        self.cache = cache  # zea.data.chunk_cache.ChunkCache, or None
        headers = {"Authorization": f"Bearer {token}"} if token else None
        self._fs = fsspec.filesystem(
            "http", client_kwargs={"headers": headers} if headers else None
        )

    def fetch(self, ranges: Sequence[tuple[int, int]], on_bytes: Ticker = None) -> list[bytes]:
        import asyncio

        from fsspec.asyn import sync

        out: list[bytes | None] = [None] * len(ranges)

        # Serve what the cache has, and only go to the network for the rest.
        misses = []
        for index, (offset, size) in enumerate(ranges):
            hit = self.cache.get(int(offset), int(size)) if self.cache is not None else None
            if hit is not None:
                out[index] = hit
                if on_bytes is not None:
                    on_bytes(int(size))  # tick, so the bar still reaches 100%
            else:
                misses.append((index, int(offset), int(size)))

        if not misses:
            return cast(list[bytes], out)

        async def one(index: int, offset: int, size: int) -> None:
            data = await self._fs._cat_file(self.url, start=offset, end=offset + size)
            out[index] = data
            if self.cache is not None:
                self.cache.put(offset, size, data)
            if on_bytes is not None:
                on_bytes(size)

        async def gather() -> None:
            await asyncio.gather(*(one(*miss) for miss in misses))

        sync(self._fs.loop, gather)
        return cast(list[bytes], out)

    def pending_bytes(self, ranges: Sequence[tuple[int, int]]) -> int:
        if self.cache is None:
            return super().pending_bytes(ranges)
        return sum(
            size for offset, size in ranges if self.cache.get(int(offset), int(size)) is None
        )


def fetcher_for(file: h5py.File) -> Fetcher | None:
    """The fetcher for an open :class:`~zea.data.file.File`, or ``None`` if it has none.

    A file zea streamed from ``hf://`` reads over HTTP; a file on disk reads from its
    descriptor. Anything else (an in-memory file, a driver we do not recognise) has no
    fast path, and its datasets fall back to h5py.
    """
    from zea.data.chunk_cache import cache_for
    from zea.internal.preset_utils import HF_PREFIX, _hf_stream_url

    source = getattr(file, "_source_name", None)
    if source is not None and str(source).startswith(HF_PREFIX):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        url = _hf_stream_url(str(source), **getattr(file, "_hf_kwargs", {}))
        # The streamed file object already carries HF's metadata (fetched on open), so the
        # content hash that keys the cache costs no extra request.
        details = getattr(getattr(file, "_stream_fileobj", None), "details", None) or {}
        cache = cache_for(details) if getattr(file, "_cache_chunks", True) else None
        fetcher = HTTPFetcher(url, token, cache)
        fetcher.source = str(source)  # the hf:// path, not the resolved streaming url
        return fetcher

    if getattr(file, "_stream_fileobj", None) is not None:
        return None  # streamed from somewhere we cannot issue range requests against

    try:
        path = file.filename
    except (ValueError, RuntimeError):
        return None
    if not path or not os.path.isfile(path):
        return None
    fetcher = LocalFetcher(path)
    fetcher.source = str(path)
    return fetcher


# --------------------------------------------------------------------------- #
# Codecs
# --------------------------------------------------------------------------- #
def filter_ids(dset: h5py.Dataset) -> list[int]:
    """Filter ids of the dataset's pipeline, in the order HDF5 applied them on write."""
    plist = dset.id.get_create_plist()
    return [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]


def _decode_lz4(raw: bytes) -> bytes:
    """Reverse the HDF5 LZ4 filter (32004), whose framing is its own.

    Header of 8-byte big-endian total size and 4-byte big-endian block size, then blocks each
    prefixed with their 4-byte big-endian compressed size. ``numcodecs.lz4`` wants a 4-byte
    *little-endian* length header instead of a size argument, so each block is handed to it
    with that prepended — cheaper than depending on the ``lz4`` package for one call.
    """
    from numcodecs import lz4

    total = int.from_bytes(raw[0:8], "big")
    block_size = int.from_bytes(raw[8:12], "big") or total
    out, pos, remaining = [], 12, total
    while remaining > 0:
        size = int.from_bytes(raw[pos : pos + 4], "big")
        pos += 4
        wanted = min(block_size, remaining)
        block = raw[pos : pos + size]
        pos += size
        if size == wanted:
            out.append(block)  # stored raw: this block did not compress
        else:
            out.append(lz4.decompress(wanted.to_bytes(4, "little") + block))
        remaining -= wanted
    return b"".join(out)


def _decode(raw: bytes, filters: list[int], filter_mask: int, itemsize: int) -> bytes:
    """Reverse the filter pipeline of one chunk.

    ``filter_mask`` has bit *i* set when HDF5 **skipped** filter *i* for this chunk, which it
    does whenever the filter failed to shrink it — incompressible data is stored raw. Applying
    the codec regardless would decode such chunks to garbage, silently.
    """
    from numcodecs import blosc, shuffle, zstd

    buf = raw
    for i in reversed(range(len(filters))):
        if filter_mask & (1 << i):
            continue  # HDF5 stored this chunk without applying filter i
        fid = filters[i]
        if fid == BLOSC:
            buf = blosc.decompress(buf)  # codec params live in the blosc header
        elif fid == ZSTD:
            buf = zstd.decompress(buf)
        elif fid == LZ4:
            buf = _decode_lz4(buf)
        elif fid == GZIP:
            buf = zlib.decompress(buf)
        elif fid == SHUFFLE:
            buf = np.asarray(shuffle.Shuffle(elementsize=itemsize).decode(buf)).tobytes()
        else:  # unreachable: eligible() rejects unknown filters
            raise _Unsupported(f"no decoder for HDF5 filter {fid}")
    return buf


def eligible(dset: h5py.Dataset, fetcher: Fetcher | None) -> bool:
    """Whether ``dset`` can be read through the fast path at all.

    Cheap and conservative: chunked storage, a decodable filter pipeline, a fetcher, and a
    plain numeric dtype (no vlen strings, no compound types — h5py handles those, and they
    are never the bulk arrays this exists for).
    """
    if fetcher is None or dset.chunks is None:
        return False
    if dset.dtype.hasobject or dset.dtype.fields is not None:
        return False
    return all(fid in DECODABLE for fid in filter_ids(dset))


def _needs_resave(dset: h5py.Dataset, fetcher: Fetcher | None) -> bool:
    """Whether this dataset misses the fast path for a reason resaving would fix.

    True means it is plainly typed and has a fetcher — but is either stored without chunking, or
    compressed with a codec we cannot decode in-process (``lzf`` above all: zea's default before
    Blosc). Both cost the concurrent read path — locally as well as over HTTP — and both are
    fixed by resaving through :meth:`File.create`, which chunks and Blosc-compresses. A strided
    selection or an exotic dtype is a different matter and stays out of this.
    """
    if fetcher is None:
        return False
    if dset.dtype.hasobject or dset.dtype.fields is not None:
        return False
    if dset.chunks is None:
        return True
    return not all(fid in DECODABLE for fid in filter_ids(dset))


def _human_bytes(n: int) -> str:
    """``n`` bytes as a short string, e.g. ``'128 MB'`` or ``'2.3 GB'``."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _fallback_note(
    dset: h5py.Dataset, fetcher: Fetcher | None, kind: str, cause: str, fix: str
) -> None:
    """Nudge, once per dataset, when a read misses the concurrent path and goes to h5py.

    Serial reads happen locally too, not only when streaming, so this fires whether or not
    progress was requested. ``cause`` completes "Reading '{name}' {cause}" and ``fix`` says how
    to avoid it; ``kind`` scopes the once-only dedupe so the resave and selection notes do not
    silence each other. The name is the file, not the dataset's internal HDF5 path — that is
    what a user would actually act on. Anything that could break here is swallowed; a message
    is never worth failing a read over.
    """
    try:
        from zea import log

        name = fetcher.source if fetcher is not None else dset.name
        size = _human_bytes(dset.nbytes)
        log.warning_once(
            f"Reading '{name}' ({size}) {cause} — falling back to a serial h5py read "
            f"(slower). {fix}",
            key=(dset.name, kind),
        )
    except Exception:  # noqa: BLE001 — messaging must never break a read
        pass


def _resave_note(dset: h5py.Dataset, fetcher: Fetcher | None) -> None:
    """Note a read that misses the concurrent path for a reason resaving would fix (see
    :func:`_needs_resave`): the storage layout, not the selection."""
    if not _needs_resave(dset, fetcher):
        return
    if dset.chunks is None:
        cause = "stored without chunking, which zea cannot read concurrently"
    else:
        codec = getattr(dset, "compression", None)
        named = f"compressed with {codec}" if codec else "compressed with a codec"
        cause = f"{named}, which zea cannot decode concurrently"
    _fallback_note(
        dset,
        fetcher,
        kind="resave",
        cause=cause,
        fix="Re-save with the `zea data resave` CLI (or zea.File.create) to read it "
        "concurrently, with a progress bar.",
    )


def _selection_note(dset: h5py.Dataset, fetcher: Fetcher | None) -> None:
    """Note a read that misses the concurrent path *only* because of its selection.

    Reached after :func:`eligible` has already passed, so the dataset itself is fast-path
    material — the selection is the sole blocker, and unlike the layout, resaving will not help.
    """
    _fallback_note(
        dset,
        fetcher,
        kind="selection",
        cause=(
            "with a selection zea cannot map to chunks (e.g. a strided slice, a boolean mask, "
            "or an unsorted or repeated index list)"
        ),
        fix="Use a contiguous slice or a single increasing index list to read it concurrently.",
    )


# --------------------------------------------------------------------------- #
# Selection -> chunks
# --------------------------------------------------------------------------- #
def _normalize(selection: Any, shape: tuple[int, ...]) -> list[tuple[np.ndarray, bool]]:
    """Per axis, the selected indices and whether the axis survives into the output.

    Accepts what h5py accepts *and* we can map back to chunks: ints, unit-step slices, and
    increasing index lists (h5py requires those to be increasing too). Steps, boolean masks
    and other exotica raise :class:`_Unsupported` and go to h5py.
    """
    if not isinstance(selection, tuple):
        selection = (selection,)

    # Compare by identity: an array entry (a valid fancy index) makes ``== Ellipsis``
    # return an array, so ``count``/``in``/``index`` would raise on the ambiguous truth value.
    ellipsis_at = [i for i, item in enumerate(selection) if item is Ellipsis]
    if len(ellipsis_at) > 1:
        raise IndexError("An index can only have a single ellipsis ('...').")
    if ellipsis_at:
        at = ellipsis_at[0]
        fill = len(shape) - (len(selection) - 1)
        selection = selection[:at] + (slice(None),) * fill + selection[at + 1 :]
    if len(selection) > len(shape):
        raise IndexError(f"Too many indices for array with {len(shape)} dimensions.")
    selection = selection + (slice(None),) * (len(shape) - len(selection))

    axes: list[tuple[np.ndarray, bool]] = []
    for index, size in zip(selection, shape):
        if isinstance(index, (int, np.integer)):
            position = int(index) + size if int(index) < 0 else int(index)
            if not 0 <= position < size:
                raise IndexError(f"Index {index} is out of bounds for axis of size {size}.")
            axes.append((np.array([position]), False))  # axis dropped from the output
        elif isinstance(index, slice):
            start, stop, step = index.indices(size)
            if step != 1:
                raise _Unsupported("strided slice")
            axes.append((np.arange(start, max(start, stop)), True))
        elif isinstance(index, (list, np.ndarray)):
            values = np.asarray(index)
            if values.dtype == bool or values.ndim != 1 or values.size == 0:
                raise _Unsupported("boolean or non-1d index")
            if not np.issubdtype(values.dtype, np.integer):
                # h5py rejects float/other indices; astype(intp) would silently truncate
                # them, so hand back to h5py to preserve its behaviour.
                raise _Unsupported("non-integer index")
            values = values.astype(np.intp)
            values = np.where(values < 0, values + size, values)
            if np.any(values < 0) or np.any(values >= size):
                raise IndexError(f"Index out of bounds for axis of size {size}.")
            if np.any(np.diff(values) <= 0):
                raise _Unsupported("unsorted or repeated index list")
            axes.append((values, True))
        else:
            raise _Unsupported(f"index of type {type(index).__name__}")

    # h5py allows a fancy index on at most one axis and raises otherwise. We *could* serve
    # the outer product here, but the contract is to return exactly what h5py returns —
    # errors included — so hand it back and let it raise.
    if sum(isinstance(index, (list, np.ndarray)) for index in selection) > 1:
        raise _Unsupported("fancy indexing on more than one axis")
    return axes


def _blocks(indices: np.ndarray, start: int, size: int):
    """Where a chunk's slot on one axis lands in the output, and what it takes from it.

    Returns ``(out, src, whole)``: the slice of output positions this chunk fills, the
    slice-or-index into the chunk they come from, and whether that is the chunk's full
    extent (which decides whether we can decompress straight into the output).
    """
    lo = int(np.searchsorted(indices, start, side="left"))
    hi = int(np.searchsorted(indices, start + size, side="left"))
    if lo == hi:
        return None
    wanted = indices[lo:hi] - start
    contiguous = int(wanted[-1] - wanted[0]) == hi - lo - 1
    if contiguous:
        src = slice(int(wanted[0]), int(wanted[-1]) + 1)
        whole = int(wanted[0]) == 0 and hi - lo == size
    else:
        src = wanted
        whole = False
    return slice(lo, hi), src, whole


def read(
    dset: h5py.Dataset,
    selection: Any,
    fetcher: Fetcher | None,
    progress: bool | Ticker = False,
) -> np.ndarray:
    """Read ``selection`` from ``dset``, concurrently, falling back to h5py when unsure.

    The contract is equality: this returns exactly what ``dset[selection]`` returns.

    Any read that misses the concurrent path — for its storage layout
    (unchunked or a codec zea cannot decode) or its selection — logs a one-time note
    naming the cause and the fix, regardless of ``progress``, since the serial
    fallback is slower on disk too (see :func:`_resave_note`, :func:`_selection_note`).

    Args:
        dset (h5py.Dataset): The dataset to read from.
        selection: Any NumPy-style index. Ints, unit-step slices and increasing index
            lists take the fast path; anything else is handed to h5py.
        fetcher (Fetcher, optional): Source of the chunk bytes for this file (see
            :func:`fetcher_for`). ``None`` disables the fast path.
        progress (bool | callable): ``True`` shows a tqdm bar over the compressed bytes;
            a callable is invoked with each chunk's size as it arrives. Reads served by
            h5py (an lzf file, a strided selection) report no per-chunk progress: h5py
            fetches the whole selection in one opaque call, so there is nothing to observe.

    Returns:
        np.ndarray: The selected data.
    """
    if fetcher is None or not eligible(dset, fetcher):
        _resave_note(dset, fetcher)
        return dset[selection]
    try:
        axes = _normalize(selection, dset.shape)
    except _Unsupported:
        _selection_note(dset, fetcher)
        return dset[selection]

    itemsize = dset.dtype.itemsize
    n_selected = int(np.prod([len(indices) for indices, _ in axes]))
    if n_selected * itemsize < MIN_BYTES:
        return dset[selection]

    chunks = dset.chunks
    out = np.empty(
        tuple(len(indices) for indices, keep in axes if keep),
        dtype=dset.dtype,
    )

    # Every chunk that the selection touches, with the output region it fills.
    tasks = []
    grids = [sorted({int(i) // size for i in indices}) for (indices, _), size in zip(axes, chunks)]
    for cell in _product(grids):
        starts = tuple(index * size for index, size in zip(cell, chunks))
        mapped = [
            _blocks(indices, start, size) for (indices, _), start, size in zip(axes, starts, chunks)
        ]
        if any(block is None for block in mapped):
            continue
        target = tuple(
            block[0] for block, (_, keep) in zip(mapped, axes) if keep
        )  # int axes collapse away
        source = tuple(block[1] for block in mapped)
        whole = all(block[2] for block in mapped)
        info = dset.id.get_chunk_info_by_coord(starts)
        tasks.append((info, target, source, whole))

    if not tasks:
        return out

    filters = filter_ids(dset)
    blosc_only = filters == [BLOSC]
    n_elem = int(np.prod(chunks))

    def place(task, raw):
        info, target, source, whole = task
        mask = int(info.filter_mask)
        view = out[target]
        if blosc_only and not mask and whole and view.flags.c_contiguous and view.size == n_elem:
            # The chunk *is* this region of the output: decode straight into it, no temporary
            # and no (serial) copy back.
            from numcodecs import blosc

            blosc.decompress(raw, dest=view)
            return
        buf = _decode(raw, filters, mask, itemsize)
        block = np.frombuffer(buf, dtype=dset.dtype, count=n_elem).reshape(chunks)
        # ``source`` has one entry per *dataset* axis, so it keeps a length-1 axis wherever the
        # selection used an int, while ``view`` has dropped it. Same elements in the same order
        # (_normalize allows at most one advanced index, which does not move axes), so reshape.
        view[...] = block[source].reshape(view.shape)

    def fetch_and_place(task, on_bytes):
        info = task[0]
        (raw,) = fetcher.fetch([(int(info.byte_offset), int(info.size))], on_bytes)
        place(task, raw)

    # The manifest gives every chunk's compressed size up front, so the bar knows its true
    # total before a single byte moves — it measures the bytes actually transferred, not a
    # guess. Ticks arrive from worker threads; tqdm.update is thread-safe.
    total = sum(int(task[0].size) for task in tasks)
    all_ranges = [(int(task[0].byte_offset), int(task[0].size)) for task in tasks]
    with _ticker(progress, total, all_ranges, fetcher) as on_bytes:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            if fetcher.per_chunk:
                # Each worker reads its own chunk, so the next read overlaps the last decode.
                # Only MAX_WORKERS chunks are ever in flight, so there is nothing to bound.
                list(pool.map(fetch_and_place, tasks, repeat(on_bytes)))
            else:
                # Remote: the ranges go out together (that is the win), then we decode them.
                # Bounded by *bytes*, since chunk sizes vary hugely across files.
                for batch in _batched(tasks, MAX_BYTES_IN_FLIGHT):
                    ranges = [(int(t[0].byte_offset), int(t[0].size)) for t in batch]
                    raws = fetcher.fetch(ranges, on_bytes)
                    list(pool.map(place, batch, raws))

    return out


@contextmanager
def _ticker(
    progress: bool | Ticker, total: int, ranges: Sequence[tuple[int, int]], fetcher: Fetcher
):
    """The per-chunk callback for ``progress``, and the tqdm bar behind it if there is one.

    A callable gets every tick, cache hits included — it wants the true per-chunk arrivals for
    its own bookkeeping, not a curated view. The bar is display only, so it is skipped
    entirely when nothing is actually pending (:meth:`Fetcher.pending_bytes`): a read served
    wholly from cache neither streams nor downloads, so there is nothing to show progress on.
    """
    if not progress:
        yield None
        return
    if callable(progress):
        yield progress
        return
    if fetcher.pending_bytes(ranges) <= 0:
        yield None
        return

    from tqdm.auto import tqdm  # tqdm.auto: renders as a widget in notebooks, text elsewhere

    name = Path(fetcher.source).name if fetcher.source else "data"
    with tqdm(total=total, unit="B", unit_scale=True, desc=f"streaming {name}") as bar:
        yield bar.update


def _product(grids: list[list[int]]):
    """Cartesian product of the touched chunk indices per axis."""
    import itertools

    return itertools.product(*grids)


def _batched(tasks: list, budget: int):
    """Group tasks so that no batch fetches more than ``budget`` compressed bytes."""
    batch: list = []
    total = 0
    for task in tasks:
        size = int(task[0].size)
        if batch and total + size > budget:
            yield batch
            batch, total = [], 0
        batch.append(task)
        total += size
    if batch:
        yield batch


ReadFn = Callable[[h5py.Dataset, Any, Fetcher | None], np.ndarray]
