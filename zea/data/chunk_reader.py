"""
zea.data.chunk_reader
=====================

Concurrent chunk reads for zea HDF5 files, bypassing h5py's serial read path.

h5py reads one chunk at a time, and it decodes them under a global lock (the HDF5 C
library is not concurrency-safe), so a read of N chunks costs N decodes back to back —
and, over HTTP, N round trips. Neither scales with the data.

But h5py *does* hand us the chunk manifest: ``get_chunk_info_by_coord`` gives the byte
offset, byte size and filter mask of any chunk. With that we can fetch the compressed
bytes ourselves — concurrently when the file is remote, straight from the file descriptor
when it is local — and decode them in a thread pool (Blosc and zlib release the GIL). The
result is bit-identical to what h5py returns; it just arrives sooner. Measured on a
201 MB read of 16 chunks: **31 ms against h5py's 291 ms** locally, and **126 ms against
863 ms** over HTTP at 20 ms/request.

This is a pure optimisation, and treated as one: anything the fast path does not fully
understand — an unknown codec, a contiguous dataset, an exotic selection — falls back to
plain ``dset[selection]``. h5py stays the reader for everything else in the file
(parameters, metadata, strings, attributes).

Wired in through :class:`~zea.data.file.ChunkedDataset`, so callers get it for free::

    with File("scan.hdf5") as file:
        file.data.raw_data[0:8]  # 8 chunks, fetched and decoded concurrently

Two details carry most of the win, and both are easy to lose in a refactor:

* The bytes are read with ``os.pread``, **not** ``h5py``'s ``read_direct_chunk`` — that
  call takes the same global lock, so it would serialise the fetch and copy every chunk an
  extra time.
* Chunks are decompressed **straight into the output array**, not into a temporary that is
  copied in afterwards. That copy is serial and costs more than the decode itself (121 ms
  of copying against 26 ms of decoding, for a 16-chunk read).
"""

from __future__ import annotations

import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

import h5py
import numpy as np

# HDF5 filter ids we can decode in-process. A dataset using anything else (notably lzf,
# zea's pre-0.1.3 default) falls back to h5py, which decodes it natively.
BLOSC = 32001  # zea's default codec since 0.1.3
GZIP = 1
SHUFFLE = 2
DECODABLE = (BLOSC, GZIP, SHUFFLE)

# Reads below this are served by h5py: handing a chunk to a worker thread costs more than
# it saves when there is barely any data to decode.
MIN_BYTES = 1 << 20  # 1 MiB

# Ceiling on compressed bytes held in memory at once. Chunk sizes vary hugely across zea
# files (12 MB for a plane-wave scan, 166 MB for a 149-transmit carotid one), so bounding
# the *count* of concurrent chunks would bound nothing: we bound their bytes.
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

    Two shapes of fetch, because the two backends want opposite things:

    * ``per_chunk``: one chunk at a time, cheaply, from inside a decode worker — so the
      read of the next chunk overlaps the decode of the last one. This is how a local file
      wants to be read (a ``pread`` is nearly free to issue).
    * batched: every range in one call, so they can go out *together*. This is how HTTP
      wants to be read — the whole point is that N ranges cost one round trip, which a
      chunk-at-a-time fetch would throw away.
    """

    #: Whether fetching one chunk on its own is cheap (see above).
    per_chunk = False

    def fetch(self, ranges: Sequence[tuple[int, int]]) -> list[bytes]:
        """Return the bytes of each ``(offset, size)`` range, in order."""
        raise NotImplementedError

    def close(self) -> None:
        """Release whatever the fetcher holds open."""


class LocalFetcher(Fetcher):
    """Reads chunk bytes from the file descriptor.

    ``os.pread`` is positional, so it needs no seek and no lock: the decode workers can
    each read the descriptor themselves, which overlaps I/O with decoding (measured: 31 ms
    against 46 ms for a 16-chunk read, where the two phases run one after the other).
    Going through ``h5py.Dataset.id.read_direct_chunk`` would do neither — it takes h5py's
    global lock, serialising the reads, and copies each chunk an extra time.
    """

    per_chunk = True

    def __init__(self, path: str | os.PathLike):
        self._fd: int | None = os.open(os.fspath(path), os.O_RDONLY)

    def fetch(self, ranges: Sequence[tuple[int, int]]) -> list[bytes]:
        if self._fd is None:
            raise ValueError("Cannot read chunks: the file has been closed.")
        return [os.pread(self._fd, int(size), int(offset)) for offset, size in ranges]

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class HTTPFetcher(Fetcher):
    """Reads chunk bytes over HTTP range requests, all of them concurrently.

    Deliberately uses fsspec's **async** ``HTTPFileSystem`` rather than ``HfFileSystem``:
    the latter is a sync filesystem whose ``cat_ranges`` issues the ranges one after the
    other. Measured against the same file, 16 ranges took 2745 ms through ``HfFileSystem``
    and 177 ms through ``HTTPFileSystem`` — one round trip instead of sixteen. The whole
    remote win rests on this, so the choice is not incidental.
    """

    def __init__(self, url: str, token: str | None = None):
        import fsspec

        self.url = url
        headers = {"Authorization": f"Bearer {token}"} if token else None
        self._fs = fsspec.filesystem(
            "http", client_kwargs={"headers": headers} if headers else None
        )

    def fetch(self, ranges: Sequence[tuple[int, int]]) -> list[bytes]:
        starts = [offset for offset, _ in ranges]
        ends = [offset + size for offset, size in ranges]
        return self._fs.cat_ranges([self.url] * len(ranges), starts, ends)


def fetcher_for(file: h5py.File) -> Fetcher | None:
    """The fetcher for an open :class:`~zea.data.file.File`, or ``None`` if it has none.

    A file zea streamed from ``hf://`` reads over HTTP; a file on disk reads from its
    descriptor. Anything else (an in-memory file, a driver we do not recognise) has no
    fast path, and its datasets fall back to h5py.
    """
    from zea.internal.preset_utils import HF_PREFIX, _hf_stream_url

    source = getattr(file, "_source_name", None)
    if source is not None and str(source).startswith(HF_PREFIX):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        return HTTPFetcher(_hf_stream_url(str(source), **getattr(file, "_hf_kwargs", {})), token)

    if getattr(file, "_stream_fileobj", None) is not None:
        return None  # streamed from somewhere we cannot issue range requests against

    try:
        path = file.filename
    except (ValueError, RuntimeError):
        return None
    if not path or not os.path.isfile(path):
        return None
    return LocalFetcher(path)


# --------------------------------------------------------------------------- #
# Codecs
# --------------------------------------------------------------------------- #
def filter_ids(dset: h5py.Dataset) -> list[int]:
    """Filter ids of the dataset's pipeline, in the order HDF5 applied them on write."""
    plist = dset.id.get_create_plist()
    return [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]


def _decode(raw: bytes, filters: list[int], filter_mask: int, itemsize: int) -> bytes:
    """Reverse the filter pipeline of one chunk.

    ``filter_mask`` has bit *i* set when HDF5 **skipped** filter *i* for this chunk, which
    it does whenever the filter failed to shrink it (incompressible data is stored raw).
    Honouring the mask per chunk is what makes this correct where the Zarr path was not:
    Zarr applies the codec to every chunk unconditionally and decoded such chunks to
    garbage, so whole arrays had to be excluded from a virtual reference. Here the mask is
    handed to us, so the bug cannot happen.
    """
    from numcodecs import blosc, shuffle

    buf = raw
    for i in reversed(range(len(filters))):
        if filter_mask & (1 << i):
            continue  # HDF5 stored this chunk without applying filter i
        fid = filters[i]
        if fid == BLOSC:
            buf = blosc.decompress(buf)  # codec params live in the blosc header
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

    if selection.count(Ellipsis) > 1:
        raise IndexError("An index can only have a single ellipsis ('...').")
    if Ellipsis in selection:
        at = selection.index(Ellipsis)
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


def read(dset: h5py.Dataset, selection: Any, fetcher: Fetcher | None) -> np.ndarray:
    """Read ``selection`` from ``dset``, concurrently, falling back to h5py when unsure.

    The contract is equality: this returns exactly what ``dset[selection]`` returns.

    Args:
        dset (h5py.Dataset): The dataset to read from.
        selection: Any NumPy-style index. Ints, unit-step slices and increasing index
            lists take the fast path; anything else is handed to h5py.
        fetcher (Fetcher, optional): Source of the chunk bytes for this file (see
            :func:`fetcher_for`). ``None`` disables the fast path.

    Returns:
        np.ndarray: The selected data.
    """
    if fetcher is None or not eligible(dset, fetcher):
        return dset[selection]
    try:
        axes = _normalize(selection, dset.shape)
    except _Unsupported:
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
            # The chunk *is* this region of the output: decode straight into it. No
            # temporary, no copy, and the page faults spread across the worker threads.
            from numcodecs import blosc

            blosc.decompress(raw, dest=view)
            return
        buf = _decode(raw, filters, mask, itemsize)
        block = np.frombuffer(buf, dtype=dset.dtype, count=n_elem).reshape(chunks)
        # ``source`` keeps one entry per *dataset* axis, so the selected block still has an
        # axis (of length 1) wherever the selection used an int — while ``view`` has dropped
        # it. Same elements, same order, so reshape it onto the destination. At most one
        # axis of ``source`` is an index array (more is rejected in _normalize), and a lone
        # advanced index does not move axes around, so the order really is preserved.
        view[...] = block[source].reshape(view.shape)

    def fetch_and_place(task):
        info = task[0]
        (raw,) = fetcher.fetch([(int(info.byte_offset), int(info.size))])
        place(task, raw)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        if fetcher.per_chunk:
            # Each worker reads its own chunk, so the next read overlaps the last decode.
            # Nothing to bound here: only MAX_WORKERS chunks are ever in flight.
            list(pool.map(fetch_and_place, tasks))
        else:
            # Remote: the ranges must go out together (that is the whole win), so fetch a
            # batch and then decode it. Batches are bounded by *bytes*, not by chunk count:
            # one file's chunks are 12 MB and another's are 166 MB, and 16 of the latter in
            # flight would be 2.6 GB.
            for batch in _batched(tasks, MAX_BYTES_IN_FLIGHT):
                ranges = [(int(t[0].byte_offset), int(t[0].size)) for t in batch]
                raws = fetcher.fetch(ranges)
                list(pool.map(place, batch, raws))

    return out


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
