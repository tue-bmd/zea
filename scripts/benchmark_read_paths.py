"""Benchmark the read paths for a zea ``raw_data`` cube, local and over HTTP.

This is the evidence behind `plan-direct-chunk.md`: **how does reading N frames compare
across the ways zea could get those bytes?**

  1. ``h5py``            plain ``h5py.File`` / ``zea.File`` (streams over HTTP for ``hf://``,
                         with the fsspec blockcache). Reads chunks one at a time, under a
                         global lock.
  2. ``direct``          what zea now does (:mod:`zea.data.chunk_reader`): pull the chunk
                         manifest out of h5py (``get_chunk_info``), fetch the raw chunk
                         bytes ourselves — concurrently when remote — and decode them in a
                         thread pool, straight into the output array. Reproduced
                         standalone here so the comparison does not depend on zea internals.
  3. ``virtual``         the VirtualiZarr path (manifest + Zarr + obstore) that this
                         replaced, kept as the yardstick it was measured against. Needs
                         ``pip install virtualizarr zarr obstore xarray``; the rows are
                         skipped when it is not installed.

The sweep over ``--n-frames`` is the point: with zea's per-frame chunking, one frame is
one chunk, so it separates the regime where a read touches a *single* chunk (nothing to
parallelise — only the open cost differs) from the regime where it touches *many* (where
concurrent fetch + parallel decode is the whole argument).

Open and read are timed **separately**: the open is the HDF5 metadata walk (~2-3 HTTP
round trips) that a published manifest avoids entirely, and it is a constant, not a
function of ``n_frames`` — mixing the two hides which lever is doing the work.

Run::

    # local file (synthesised if omitted), sweeping 1..16 chunks
    python scripts/benchmark_read_paths.py --n-frames 1 2 4 8 16

    # a real local file
    python scripts/benchmark_read_paths.py --local /data/scan.hdf5

    # cloud, against real HF latency
    python scripts/benchmark_read_paths.py --hf hf://wesselvannierop/streaming/file.hdf5

    # both, plus the fsspec concurrency probe that Phase 3 of the plan hangs on
    python scripts/benchmark_read_paths.py --local scan.hdf5 --hf hf://... --probe-concurrency

Every path is checked against plain h5py for bit-exact equality (an ``equal`` column):
the fast path must be a pure optimisation, never a semantic change.

There are three benchmarks, and they answer different questions:

  * ``--local``  warm page cache (we cannot drop it without root), so it measures **decode**
    throughput, not disk. That is the question worth asking locally.
  * ``--serve``  the same file over a local HTTP server with a fixed latency per request and
    a **request counter**. This is the honest cloud comparison: real HF is bandwidth-bound
    and noisy by a factor of 3-4, which drowns the thing the paths actually differ in — how
    many round trips a read costs, and whether they overlap.
  * ``--hf``     real HuggingFace, as a sanity check that the served numbers hold up. Network
    noise is one-sided (it only ever *adds* time), so we report the **best** of ``--repeats``;
    a median there mostly reports the CDN's mood. Each repeat re-opens the file cold.
"""

from __future__ import annotations

import argparse
import http.server
import itertools
import os
import socketserver
import statistics
import threading
import time
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the Blosc HDF5 filter for the plain-h5py path)
import numpy as np

warnings.filterwarnings("ignore")

import zea  # noqa: E402  (after the warnings filter, which zea's imports trip)
from zea.internal.preset_utils import HF_PREFIX, _hf_parse_path  # noqa: E402

RAW_KEYS = ("tracks/track_0/data/raw_data", "data/raw_data")
HF_HOST = "https://huggingface.co"

# HDF5 filter ids we can decode in-process. Everything else falls back to h5py.
BLOSC_ID = 32001
GZIP_ID = 1
SHUFFLE_ID = 2
LZF_ID = 32000  # no in-process decoder (and no Zarr codec) -> fallback


# --------------------------------------------------------------------------- #
# The proposed path: chunk manifest from h5py, bytes fetched by us, decode in threads
# --------------------------------------------------------------------------- #
def _filters(dset: h5py.Dataset) -> list[int]:
    """Filter ids of the dataset's pipeline, in the order HDF5 applied them."""
    plist = dset.id.get_create_plist()
    return [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]


def decodable(dset: h5py.Dataset) -> bool:
    """Can we decode this dataset's chunks ourselves (else: fall back to h5py)?"""
    if dset.chunks is None:
        return False
    return all(fid in (BLOSC_ID, GZIP_ID, SHUFFLE_ID) for fid in _filters(dset))


def _decode(raw: bytes, filters: list[int], filter_mask: int, itemsize: int) -> bytes:
    """Reverse the HDF5 filter pipeline for one chunk.

    ``filter_mask`` has bit *i* set when HDF5 *skipped* filter *i* for this chunk — which
    it does when the filter did not shrink the chunk (incompressible data), storing it raw.
    Zarr cannot express that (it is the gotcha the virtual path had to detect and exclude
    whole arrays for); here it is handed to us per chunk, so we simply skip that filter.
    """
    from numcodecs import blosc
    from numcodecs import shuffle as nc_shuffle

    buf = raw
    for i in reversed(range(len(filters))):
        if filter_mask & (1 << i):
            continue  # HDF5 stored this chunk without applying filter i
        fid = filters[i]
        if fid == BLOSC_ID:
            buf = blosc.decompress(buf)  # params live in the blosc header
        elif fid == GZIP_ID:
            buf = zlib.decompress(buf)
        elif fid == SHUFFLE_ID:
            buf = np.asarray(nc_shuffle.Shuffle(elementsize=itemsize).decode(buf)).tobytes()
        else:
            raise ValueError(f"no in-process decoder for HDF5 filter {fid}")
    return buf


def _chunk_coords(dset: h5py.Dataset, n_frames: int) -> list[tuple[int, ...]]:
    """Chunk-grid coordinates covering frames ``[0, n_frames)`` (full extent elsewhere).

    A chunk is addressed by the dataset coordinate of its first element, which is what
    ``get_chunk_info_by_coord`` / ``read_direct_chunk`` take.
    """
    chunks, shape = dset.chunks, dset.shape
    n_covered = -(-n_frames // chunks[0]) * chunks[0]  # round up to the chunk grid
    axis0 = range(0, min(n_covered, shape[0]), chunks[0])
    others = [range(0, s, c) for s, c in zip(shape[1:], chunks[1:])]
    return list(itertools.product(axis0, *others))


def _scatter(dset: h5py.Dataset, coords, out: np.ndarray, workers: int, fetch):
    """Decode every chunk straight into its place in ``out``, in parallel.

    Two things make this fast, and both are easy to get wrong:

    * **No assemble copy.** ``blosc.decompress(raw, dest=view)`` writes the chunk directly
      into the output array. Decoding to a temporary and copying it in afterwards costs more
      than the decode itself (measured: 121 ms of copy vs 26 ms of decode for 16 chunks), and
      the copy is serial, so it caps the whole thing.
    * **Threads actually scale**, because Blosc releases the GIL — and because the bytes come
      from ``fetch``, not from h5py, whose global lock would serialise the workers.

    The in-place write needs the destination to be one contiguous block: true for zea's
    layouts (each axis is chunked at 1 or full extent, so a chunk is a contiguous run),
    but not in general — a non-contiguous or cropped chunk falls back to decode-and-copy.
    """
    chunks, shape = dset.chunks, dset.shape
    filters, itemsize = _filters(dset), dset.dtype.itemsize
    blosc_only = filters == [BLOSC_ID]
    n_elem = int(np.prod(chunks))

    def one(coord):
        raw, mask = fetch(coord)
        target = tuple(slice(o, min(o + c, s)) for o, c, s in zip(coord, chunks, shape))
        view = out[target]
        whole = view.shape == chunks  # not cropped at the array edge
        if blosc_only and not mask and whole and view.flags.c_contiguous:
            from numcodecs import blosc

            blosc.decompress(raw, dest=view)
            return
        buf = _decode(raw, filters, mask, itemsize)
        block = np.frombuffer(buf, dtype=dset.dtype, count=n_elem).reshape(chunks)
        view[...] = block[tuple(slice(0, s) for s in view.shape)]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, coords))


def _output(dset, coords, n_frames):
    """Buffer covering the touched chunks; the caller slices the requested frames out of it."""
    n_covered = min(max(c[0] for c in coords) + dset.chunks[0], dset.shape[0])
    return np.empty((n_covered, *dset.shape[1:]), dtype=dset.dtype)


def direct_read_local(dset: h5py.Dataset, path, n_frames: int, workers: int = 8) -> np.ndarray:
    """Local: ``pread`` the chunk bytes at their offsets, decode into the output in threads.

    We read the file descriptor ourselves rather than calling ``read_direct_chunk``: h5py
    serialises every call on its global lock, so going through it would make the fetch
    serial *and* pay for an extra bytes copy per chunk.
    """
    coords = _chunk_coords(dset, n_frames)
    infos = {coord: dset.id.get_chunk_info_by_coord(coord) for coord in coords}
    out = _output(dset, coords, n_frames)
    fd = os.open(str(path), os.O_RDONLY)
    try:

        def fetch(coord):
            info = infos[coord]
            return os.pread(fd, info.size, info.byte_offset), int(info.filter_mask)

        _scatter(dset, coords, out, workers, fetch)
    finally:
        os.close(fd)
    return out[:n_frames]


def direct_read_remote(dset, url, fs, n_frames: int, workers: int = 8) -> np.ndarray:
    """Remote: chunk offsets from the (already open) h5py handle, then ONE concurrent
    ``cat_ranges`` for all of them, then decode into the output in threads.

    This is the read h5py structurally cannot do: its global lock serialises every call, so
    N chunks cost N round trips however many threads you give it.
    """
    coords = _chunk_coords(dset, n_frames)
    infos = [dset.id.get_chunk_info_by_coord(coord) for coord in coords]
    starts = [int(i.byte_offset) for i in infos]
    ends = [int(i.byte_offset + i.size) for i in infos]
    raw_chunks = fs.cat_ranges([url] * len(starts), starts, ends)  # one concurrent batch
    by_coord = {
        coord: (raw, int(info.filter_mask)) for coord, raw, info in zip(coords, raw_chunks, infos)
    }
    out = _output(dset, coords, n_frames)
    _scatter(dset, coords, out, workers, by_coord.__getitem__)
    return out[:n_frames]


# --------------------------------------------------------------------------- #
# Remote plumbing
# --------------------------------------------------------------------------- #
def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def resolve_url(hf_path: str, revision: str | None = None) -> str:
    """The HF *resolve* URL our range requests go to (same one virtual.py pins chunks at)."""
    repo_id, subpath = _hf_parse_path(hf_path)
    if not subpath:
        raise ValueError(f"Expected an hf:// path to a single file, got '{hf_path}'.")
    return f"{HF_HOST}/datasets/{repo_id}/resolve/{revision or 'main'}/{subpath}"


def http_fs():
    """A fresh async HTTP filesystem (skip_instance_cache: no warm state between repeats)."""
    import fsspec

    headers = {}
    token = hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return fsspec.filesystem(
        "http",
        skip_instance_cache=True,
        client_kwargs={"headers": headers} if headers else None,
    )


def raw_key(f: h5py.File) -> str:
    for key in RAW_KEYS:
        if key in f:
            return key
    raise KeyError(f"No raw_data in this file (looked for {RAW_KEYS}).")


# --------------------------------------------------------------------------- #
# Timed read paths — each returns (open_s, read_s, array)
# --------------------------------------------------------------------------- #
def t_h5py_local(path, n):
    t0 = time.perf_counter()
    with h5py.File(path, "r") as f:
        ds = f[raw_key(f)]
        _ = ds.shape
        t1 = time.perf_counter()
        arr = ds[0:n]
        return t1 - t0, time.perf_counter() - t1, arr


def t_direct_local(path, n, workers):
    t0 = time.perf_counter()
    with h5py.File(path, "r") as f:
        ds = f[raw_key(f)]
        _ = ds.shape
        t1 = time.perf_counter()
        arr = direct_read_local(ds, path, n, workers)
        return t1 - t0, time.perf_counter() - t1, arr


def t_virtual(path, n):
    """On-the-fly manifest: the open pays for one HDF5 metadata pass, the read goes to Zarr."""
    import zarr
    from virtualizarr.parsers import HDFParser

    url = Path(path).absolute().as_uri()
    t0 = time.perf_counter()
    with h5py.File(path, "r") as f:
        group = raw_key(f).rsplit("/", 1)[0]
    store = HDFParser(group=group)(url, _obstore_registry(url))
    array = zarr.open_group(store, mode="r")["raw_data"]
    t1 = time.perf_counter()
    arr = np.asarray(array[0:n])
    return t1 - t0, time.perf_counter() - t1, arr


def t_zea_stream(hf_path, n):
    t0 = time.perf_counter()
    with zea.File(hf_path, mode="r") as f:  # stream=True by default for hf://
        ds = f[raw_key(f)]
        _ = ds.shape
        t1 = time.perf_counter()
        arr = ds[0:n]
        return t1 - t0, time.perf_counter() - t1, arr


def t_direct_remote(hf_path, n, workers, revision=None):
    """Open with the streaming h5py handle (metadata only), then fetch chunks ourselves."""
    url = resolve_url(hf_path, revision)
    t0 = time.perf_counter()
    with zea.File(hf_path, mode="r") as f:
        ds = f[raw_key(f)]
        _ = ds.shape
        t1 = time.perf_counter()
        arr = direct_read_remote(ds, url, http_fs(), n, workers)
        return t1 - t0, time.perf_counter() - t1, arr


# --------------------------------------------------------------------------- #
# Counting, latency-injecting HTTP server: the deterministic stand-in for the cloud.
#
# Real HF is bandwidth-bound (~150 MB/s here) and noisy by a factor of 3-4, which drowns
# the thing the paths actually differ in: **how many round trips** a read costs. Serving
# the same file locally with a fixed latency per request and a request counter isolates
# that, and it is reproducible.
# --------------------------------------------------------------------------- #
class _Counter:
    def __init__(self):
        self.n = 0
        self.lock = threading.Lock()

    def bump(self):
        with self.lock:
            self.n += 1

    def value(self):
        with self.lock:
            return self.n


def make_server(directory, counter, latency):
    """Range-capable HTTP server that counts requests and sleeps ``latency`` on each."""

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _path(self):
            return os.path.join(directory, self.path.lstrip("/").split("?")[0])

        def _respond(self, body=None, size=0, start=None, end=None):
            if start is None:
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
            else:
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_HEAD(self):
            counter.bump()
            time.sleep(latency)
            try:
                size = os.path.getsize(self._path())
            except OSError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            counter.bump()
            time.sleep(latency)
            try:
                size = os.path.getsize(self._path())
            except OSError:
                self.send_error(404)
                return
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                start_s, _, end_s = rng[6:].partition("-")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else size - 1
                with open(self._path(), "rb") as handle:
                    handle.seek(start)
                    body = handle.read(end - start + 1)
                self._respond(body, size, start, start + len(body) - 1)
            else:
                with open(self._path(), "rb") as handle:
                    self._respond(handle.read())

    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/"


def _obstore_registry(base_url):
    """Registry pointing Zarr/obstore at our test server (virtual.py hard-codes the HF host)."""
    from obstore.store import HTTPStore, LocalStore

    try:
        from obspec_utils.registry import ObjectStoreRegistry
    except ImportError:
        from virtualizarr.registry import ObjectStoreRegistry

    stores = {"file://": LocalStore()}
    if base_url.startswith("http"):
        # allow_http: obstore refuses plain (non-TLS) HTTP by default.
        stores[base_url] = HTTPStore.from_url(base_url, client_options={"allow_http": True})
    return ObjectStoreRegistry(stores)


def build_refs(local_path, url, out_json):
    """Kerchunk refs for one file, with the chunks pointed at ``url`` (what publishing does)."""
    from virtualizarr.parsers import HDFParser

    with h5py.File(local_path, "r") as f:
        group = raw_key(f).rsplit("/", 1)[0]
    store = HDFParser(group=group)(Path(local_path).absolute().as_uri(), _obstore_registry(url))
    dataset = store.to_virtual_dataset().vz.rename_paths(lambda _p: url)
    dataset.vz.to_kerchunk(str(out_json), format="json")
    return out_json


def t_http_h5py(url, n, counter):
    """h5py over fsspec HTTP with zea's streaming defaults (blockcache, 8 MiB blocks)."""
    import fsspec

    fs = fsspec.filesystem("http", skip_instance_cache=True)
    n0, t0 = counter.value(), time.perf_counter()
    fileobj = fs.open(url, block_size=8 * 1024 * 1024, cache_type="blockcache")
    with h5py.File(fileobj, "r") as f:
        ds = f[raw_key(f)]
        _ = ds.shape
        open_reqs, t1 = counter.value() - n0, time.perf_counter()
        n1 = counter.value()
        arr = ds[0:n]
        return open_reqs, t1 - t0, counter.value() - n1, time.perf_counter() - t1, arr


def t_http_direct(url, n, counter, workers):
    """The proposed path over HTTP: h5py open for the manifest, then our concurrent ranges."""
    import fsspec

    fs = fsspec.filesystem("http", skip_instance_cache=True)
    n0, t0 = counter.value(), time.perf_counter()
    fileobj = fs.open(url, block_size=8 * 1024 * 1024, cache_type="blockcache")
    with h5py.File(fileobj, "r") as f:
        ds = f[raw_key(f)]
        _ = ds.shape
        open_reqs, t1 = counter.value() - n0, time.perf_counter()
        n1 = counter.value()
        arr = direct_read_remote(
            ds, url, fsspec.filesystem("http", skip_instance_cache=True), n, workers
        )
        return open_reqs, t1 - t0, counter.value() - n1, time.perf_counter() - t1, arr


def t_http_virtual(refs_path, url, n, counter):
    """The published-reference path: the open is a local JSON parse, so it costs 0 requests."""
    import zarr
    from virtualizarr.parsers import KerchunkJSONParser

    n0, t0 = counter.value(), time.perf_counter()
    store = KerchunkJSONParser()(Path(refs_path).absolute().as_uri(), _obstore_registry(url))
    array = zarr.open_group(store, mode="r")["raw_data"]
    _ = array.shape
    open_reqs, t1 = counter.value() - n0, time.perf_counter()
    n1 = counter.value()
    arr = array[0:n]
    return open_reqs, t1 - t0, counter.value() - n1, time.perf_counter() - t1, arr


def bench_served(path, n_frames_list, repeats, workers, latency):
    """Deterministic 'cloud': same file over HTTP with a fixed per-request latency."""
    directory, name = os.path.dirname(os.path.abspath(path)), os.path.basename(path)
    counter = _Counter()
    httpd, base = make_server(directory, counter, latency)
    url = base + name

    with h5py.File(path, "r") as f:
        ds = f[raw_key(f)]
        n_frames_list = [n for n in n_frames_list if n <= ds.shape[0]]
        references = {n: ds[0:n] for n in n_frames_list}

    refs = build_refs(path, url, Path(directory) / "_bench_refs.json")

    line = (
        f"  {'path':<24}{'open req':>9}{'open ms':>9}{'read req':>9}{'read ms':>9}"
        f"{'total ms':>9}{'equal':>7}"
    )
    print(f"\n{'=' * len(line)}")
    print(f"  SERVED  {name}  (local HTTP, {latency * 1e3:.0f} ms/request, best of {repeats})")
    print("=" * len(line))

    try:
        for n in n_frames_list:
            ref, n_chunks = references[n], len(_chunk_coords_for(path, n))
            print(f"\n  n_frames = {n}   ({ref.nbytes / 1e6:.0f} MB, {n_chunks} chunks)")
            print(line)
            paths = {
                "h5py stream": lambda n=n: t_http_h5py(url, n, counter),
                "direct + conc. ranges": lambda n=n: t_http_direct(url, n, counter, workers),
                "virtual (published ref)": lambda n=n: t_http_virtual(refs, url, n, counter),
            }
            for label, fn in paths.items():
                trials, arr = [], None
                try:
                    for _ in range(repeats):
                        open_req, open_s, read_req, read_s, arr = fn()
                        trials.append((open_s + read_s, open_req, open_s, read_req, read_s))
                except Exception as exc:  # noqa: BLE001
                    print(f"  {label:<24}{'FAILED':>9}  {type(exc).__name__}: {exc}")
                    continue
                _, open_req, open_s, read_req, read_s = min(trials)  # best of N
                equal = np.array_equal(arr, ref)
                print(
                    f"  {label:<24}{open_req:>9}{open_s * 1e3:>9.0f}{read_req:>9}"
                    f"{read_s * 1e3:>9.0f}{(open_s + read_s) * 1e3:>9.0f}{'Y' if equal else 'N':>7}"
                )
    finally:
        httpd.shutdown()
        Path(refs).unlink(missing_ok=True)


def _chunk_coords_for(path, n):
    with h5py.File(path, "r") as f:
        return _chunk_coords(f[raw_key(f)], n)


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def have_virtualizarr() -> bool:
    """The virtual path is no longer a zea dependency: its rows are optional."""
    try:
        import virtualizarr  # noqa: F401
        import zarr  # noqa: F401

        return True
    except ImportError:
        return False


def run(label, fn, n, reference, repeats, rows, stat="median"):
    """Time ``fn`` ``repeats`` times and check the result against h5py.

    Local reads are stable, so the median is representative. Real-cloud reads are not:
    network noise is one-sided (it only ever *adds* time), so there the best run is the
    honest estimate of what the path costs — a median mostly reports the CDN's mood.
    """
    opens, reads = [], []
    arr = None
    try:
        for _ in range(repeats):
            open_s, read_s, arr = fn(n)
            opens.append(open_s)
            reads.append(read_s)
    except Exception as exc:  # noqa: BLE001 — a failing path is a result, not a crash
        print(f"  {label:<24}{'FAILED':>10}  {type(exc).__name__}: {exc}")
        return

    pick = min if stat == "best" else statistics.median
    open_s, read_s = pick(opens), pick(reads)
    equal = reference is None or np.array_equal(arr, reference)
    rows.append((n, label, open_s, read_s, equal))
    print(
        f"  {label:<24}{open_s * 1e3:>9.0f}{read_s * 1e3:>9.0f}"
        f"{(open_s + read_s) * 1e3:>9.0f}{'Y' if equal else 'N':>7}"
    )


def describe(dset, n_frames_list):
    """Print the chunk geometry — the thing that decides whether any of this can parallelise."""
    chunks = dset.chunks
    chunk_mb = np.prod(chunks) * dset.dtype.itemsize / 1e6
    per_frame = int(np.prod(chunks[1:]) and np.prod(dset.shape[1:]) // np.prod(chunks[1:]))
    print(
        f"raw_data {dset.shape} {dset.dtype} = {dset.nbytes / 1e6:.0f} MB\n"
        f"chunks   {chunks} = {chunk_mb:.1f} MB each, {dset.id.get_num_chunks()} total, "
        f"{per_frame} per frame\n"
        f"filters  {_filters(dset)} (in-process decodable: {decodable(dset)})\n"
        f"reading  {n_frames_list} frames "
        f"= {[len(_chunk_coords(dset, n)) for n in n_frames_list]} chunks"
    )


def header(where):
    line = f"  {'path':<24}{'open ms':>9}{'read ms':>9}{'total ms':>9}{'equal':>7}"
    print(f"\n{'=' * len(line)}\n  {where}\n{'=' * len(line)}")
    return line


def bench_local(path, n_frames_list, repeats, workers):
    with h5py.File(path, "r") as f:
        ds = f[raw_key(f)]
        describe(ds, n_frames_list)
        n_frames_list = [n for n in n_frames_list if n <= ds.shape[0]]
        references = {n: ds[0:n] for n in n_frames_list}

    rows = []
    line = header(f"LOCAL  {path}  (warm page cache)")
    for n in n_frames_list:
        print(f"\n  n_frames = {n}   ({references[n].nbytes / 1e6:.0f} MB)")
        print(line)
        ref = references[n]
        run("h5py", lambda n=n: t_h5py_local(path, n), n, ref, repeats, rows)
        run(
            "direct + par. decode",
            lambda n=n: t_direct_local(path, n, workers),
            n,
            ref,
            repeats,
            rows,
        )
        if have_virtualizarr():
            run("virtual (on the fly)", lambda n=n: t_virtual(path, n), n, ref, repeats, rows)
    return rows


def bench_cloud(hf_path, n_frames_list, repeats, workers):
    with zea.File(hf_path, mode="r") as f:
        ds = f[raw_key(f)]
        describe(ds, n_frames_list)
        n_frames_list = [n for n in n_frames_list if n <= ds.shape[0]]
        references = {n: ds[0:n] for n in n_frames_list}

    rows = []
    line = header(f"CLOUD  {hf_path}  (real HF, best of {repeats})")
    for n in n_frames_list:
        print(f"\n  n_frames = {n}   ({references[n].nbytes / 1e6:.0f} MB)")
        print(line)
        ref, best = references[n], "best"
        run(
            "zea.File (h5py stream)",
            lambda n=n: t_zea_stream(hf_path, n),
            n,
            ref,
            repeats,
            rows,
            best,
        )
        run(
            "direct + conc. ranges",
            lambda n=n: t_direct_remote(hf_path, n, workers),
            n,
            ref,
            repeats,
            rows,
            best,
        )
    return rows


def probe_concurrency(hf_path, n_ranges=16, revision=None):
    """Does ``cat_ranges`` actually go out concurrently? The remote win depends on it.

    ``plan-direct-chunk.md`` Phase 3 flags this as the open question: ``HfFileSystem`` is a
    *sync* fsspec filesystem, so its ``cat_ranges`` may serialise, whereas ``HTTPFileSystem``
    is async. Time both against the same file: serial would be ~N x the single-range time.
    """
    from huggingface_hub import HfFileSystem

    url = resolve_url(hf_path, revision)
    repo_id, subpath = _hf_parse_path(hf_path)
    fs_path = f"datasets/{repo_id}/{subpath}"
    starts = [i * 1_000_000 for i in range(n_ranges)]
    ends = [s + 65_536 for s in starts]

    print(f"\n== cat_ranges concurrency probe ({n_ranges} ranges of 64 KiB) ==")

    fs = http_fs()
    t0 = time.perf_counter()
    fs.cat_ranges([url], [starts[0]], [ends[0]])
    one = time.perf_counter() - t0

    t0 = time.perf_counter()
    fs.cat_ranges([url] * n_ranges, starts, ends)
    http_s = time.perf_counter() - t0

    hf_s = float("nan")
    try:
        hfs = HfFileSystem(token=hf_token(), skip_instance_cache=True)
        t0 = time.perf_counter()
        hfs.cat_ranges([fs_path] * n_ranges, starts, ends)
        hf_s = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        print(f"  HfFileSystem: FAILED ({type(exc).__name__}: {exc})")

    print(f"  1 range (HTTPFileSystem):        {one * 1e3:>7.0f} ms   <- one round trip")
    print(f"  {n_ranges} ranges (HTTPFileSystem, async): {http_s * 1e3:>7.0f} ms")
    print(f"  {n_ranges} ranges (HfFileSystem, sync):   {hf_s * 1e3:>7.0f} ms")
    print(
        f"  => concurrent if ~{one * 1e3:.0f} ms, serial if ~{one * n_ranges * 1e3:.0f} ms. "
        "The direct path should issue its ranges through whichever is concurrent."
    )


# --------------------------------------------------------------------------- #
def synth_file(path, n_frames, n_tx, n_ax, n_el):
    """A zea file with today's defaults (Blosc-zstd, one chunk per frame, paged)."""
    rng = np.random.default_rng(0)
    z = np.linspace(0, 1, n_ax)[:, None]
    base = np.exp(-2.5 * z) * np.sin(2 * np.pi * 40 * z)  # compressible, not pure noise
    raw = np.empty((n_frames, n_tx, n_ax, n_el, 1), np.float32)
    for fi in range(n_frames):
        for ti in range(n_tx):
            speckle = np.cumsum(rng.standard_normal((n_ax, n_el)), axis=0)
            speckle /= np.abs(speckle).max() + 1e-9
            raw[fi, ti, :, :, 0] = base * (1 + 0.7 * speckle)

    zea.File.create(
        str(path),
        data={"raw_data": raw},
        scan={
            "sampling_frequency": np.float32(40e6),
            "center_frequency": np.float32(7e6),
            "demodulation_frequency": np.float32(7e6),
            "initial_times": np.zeros(n_tx, np.float32),
            "t0_delays": np.zeros((n_tx, n_el), np.float32),
            "sound_speed": np.float32(1540.0),
            "tx_apodizations": np.ones((n_tx, n_el), np.float32),
            "focus_distances": np.full(n_tx, np.inf, np.float32),
            "transmit_origins": np.zeros((n_tx, 3), np.float32),
            "polar_angles": np.zeros(n_tx, np.float32),
            "azimuth_angles": np.zeros(n_tx, np.float32),
        },
        probe={"name": "generic", "probe_geometry": np.zeros((n_el, 3), np.float32)},
        overwrite=True,
        ignore_warnings=True,
    )
    return path


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--local", default=None, help="Local zea file. Synthesised when omitted.")
    p.add_argument("--hf", default=None, help="hf:// path to one zea file (cloud benchmark).")
    p.add_argument("--n-frames", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--workers", type=int, default=8, help="Decode/fetch threads (direct path).")
    p.add_argument("--probe-concurrency", action="store_true", help="Time cat_ranges (needs --hf).")
    p.add_argument("--no-local", action="store_true", help="Skip the local benchmark.")
    p.add_argument(
        "--serve",
        action="store_true",
        help="Also serve the local file over HTTP with injected latency: a deterministic "
        "cloud, where requests are counted and bandwidth noise cannot hide round trips.",
    )
    p.add_argument("--latency", type=float, default=0.02, help="Seconds/request in --serve mode.")
    # Synthetic file geometry (only used when --local is omitted).
    p.add_argument("--synth-frames", type=int, default=16)
    p.add_argument("--synth-tx", type=int, default=24)
    p.add_argument("--synth-ax", type=int, default=2048)
    p.add_argument("--synth-el", type=int, default=64)
    args = p.parse_args()

    if not args.no_local or args.serve:
        path = args.local
        if path is None:
            path = Path(os.environ.get("TMPDIR", "/tmp")) / "zea_bench_raw.hdf5"
            if not Path(path).exists():
                print(f"No --local given; synthesising {path} ...")
                synth_file(path, args.synth_frames, args.synth_tx, args.synth_ax, args.synth_el)
        if not args.no_local:
            bench_local(path, args.n_frames, args.repeats, args.workers)
        if args.serve:
            bench_served(path, args.n_frames, args.repeats, args.workers, args.latency)

    if args.hf:
        if not args.hf.startswith(HF_PREFIX):
            raise SystemExit(f"--hf must be an 'hf://' path, got '{args.hf}'.")
        bench_cloud(args.hf, args.n_frames, args.repeats, args.workers)
        if args.probe_concurrency:
            probe_concurrency(args.hf)


if __name__ == "__main__":
    main()
