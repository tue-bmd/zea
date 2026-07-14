"""Pick zea's HDF5 codec, compression level and chunk size, on the real read path.

This is the benchmark behind three defaults in :mod:`zea.data.spec`:

  * ``DEFAULT_COMPRESSION`` — which codec, and at which ``clevel``,
  * ``DEFAULT_CHUNK_AXES`` — how much data goes in one chunk,
  * and, by implication, whether a max-transmits-per-chunk cap is worth having.

It measures them **through the path zea actually reads with** — the concurrent
chunk reader (``zea.data.chunk_reader``), not h5py — because that is what changed:
chunks are now fetched by us and decoded in a thread pool, so a chunk is no longer
just a unit of I/O, it is the unit of *parallelism*. One 166 MB chunk decodes in one
thread no matter how many cores are free, which turns chunk size into a first-order
knob rather than a detail. h5py is still timed alongside, as the baseline.

Four numbers decide it:

  * **compression ratio** — disk (and cloud egress), the thing a slow codec buys;
  * **write throughput** — paid once per file, so it is a tiebreak, not a criterion;
  * **local read** — timed **warm** *and* **cold** (page cache dropped). Cold is the
    honest one for zea-sized data: a 25 GB scan does not sit in RAM, so a read is
    bounded by ``stored_bytes / disk_bandwidth + decode``, and compressing harder
    can make reads *faster* by moving fewer bytes;
  * **cloud read** — over an HTTP server with a fixed latency per request, which
    counts round trips deterministically (real HF is bandwidth-bound and noisy by
    3-4x, which drowns exactly the effect chunk size has).

All codecs are lossless; every configuration is asserted bit-equal to the source.

Usage::

    pip install hdf5plugin
    python scripts/benchmark_compression.py                       # synthetic
    python scripts/benchmark_compression.py --from-file scan.hdf5 --n-frames 4
    python scripts/benchmark_compression.py --from-file scan.hdf5 --only chunks
    python scripts/benchmark_compression.py --from-file scan.hdf5 --latency 0.02

``--from-file`` uses real data (strongly preferred — compressibility is a property of
the data, and synthetic RF only approximates it); ``--n-frames`` caps how much of it is
loaded, since these files run to tens of GB.

Note: files written with hdf5plugin filters need ``import hdf5plugin`` in the *reading*
process too (zea imports it for you; other tools must be told).
"""

from __future__ import annotations

import argparse
import http.server
import logging
import os
import socketserver
import tempfile
import threading
import time
from pathlib import Path

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401  (registers the external filters on import)

    HAVE_HDF5PLUGIN = True
except ImportError:
    HAVE_HDF5PLUGIN = False

import zea
from zea.data.chunk_reader import HTTPFetcher, LocalFetcher
from zea.data.chunk_reader import read as direct_read
from zea.data.spec import PAGED_LAYOUT

logging.getLogger("zea").setLevel(logging.WARNING)

# raw_data is (n_frames, n_tx, n_ax, n_el, n_ch). zea subsamples frames and transmits, so
# those are the axes a chunk may be split along; the rest of a plane is always read whole.
FRAME_AXIS, TX_AXIS = 0, 1

DSET = "data/raw_data"

# Chunk shapes are given as transmits-per-chunk (one frame per chunk always: the frame axis
# is what a read subsamples first). ``None`` means "the whole frame" — zea's current default,
# DEFAULT_CHUNK_AXES = ("n_frames",).
DEFAULT_TX_PER_CHUNK = (1, 4, 8, 16, 32, None)


# --------------------------------------------------------------------------- #
# Source data
# --------------------------------------------------------------------------- #
def synth_rf_data(shape, dtype, seed=0):
    """Synthetic RF-like data with realistic compressibility.

    Pure noise is incompressible and would make every codec look alike, so this builds
    structure: a depth-attenuated carrier (the pulse) under smooth element apodisation and
    correlated speckle, plus a little noise and gentle frame-to-frame motion. That gives it
    the spatial correlation and bounded entropy real channel data has — but it is still an
    approximation, which is why ``--from-file`` is preferred for a real decision.
    """
    n_frames, n_tx, n_ax, n_el, n_ch = shape
    rng = np.random.default_rng(seed)

    z = np.linspace(0.0, 1.0, n_ax)[:, None]
    envelope = np.exp(-2.5 * z)
    carrier = np.sin(2.0 * np.pi * 40.0 * z)
    el = np.linspace(-1.0, 1.0, n_el)[None, :]
    apod = 0.5 + 0.5 * np.cos(np.pi * el)
    base = (envelope * carrier) * apod

    out = np.empty(shape, dtype=np.float32)
    for fi in range(n_frames):
        motion = 1.0 + 0.03 * fi
        for ti in range(n_tx):
            speckle = np.cumsum(rng.standard_normal((n_ax, n_el)), axis=0)
            speckle -= speckle.mean(axis=0, keepdims=True)
            speckle /= np.abs(speckle).max() + 1e-9
            plane = base * motion * (1.0 + 0.7 * speckle)
            plane += 0.02 * rng.standard_normal((n_ax, n_el))
            for ci in range(n_ch):
                out[fi, ti, :, :, ci] = plane if ci == 0 else plane[::-1]

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        peak = 0.9 * min(abs(info.min), info.max)
        out = np.clip(out / (np.abs(out).max() + 1e-9) * peak, info.min, info.max)
    return out.astype(dtype)


def to_int16(raw):
    """Quantise float RF to int16, scaled to fill the range.

    New acquisitions store ``raw_data`` as int16 (what the scanner digitises), while the
    other fields stay float32 — and the codec question is dtype-dependent: shuffle groups
    bytes by significance, so it has 4 byte-planes to separate in a float32 and only 2 in an
    int16, and the exponent bits that make float32 RF compressible are simply not there.
    Casting the *same* real data lets both be compared without needing a native int16 file.
    """
    if np.issubdtype(raw.dtype, np.integer):
        return raw
    peak = 0.9 * np.iinfo(np.int16).max
    scaled = raw / (np.abs(raw).max() + 1e-9) * peak
    return np.clip(scaled, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)


def load_source(args):
    """Return ``(raw_data, description)`` for the chosen source."""
    if args.from_file:
        print(f"Loading {args.from_file} ...")
        with zea.File(args.from_file, mode="r") as f:
            raw = f.data.raw_data[: args.n_frames]
        note = ""
        if args.cast == "int16":
            raw, note = to_int16(raw), " (cast to int16)"
        return raw, f"file={args.from_file}{note}"

    shape = (args.n_frames, args.n_tx, args.n_ax, args.n_el, args.n_ch)
    dtype = np.float32 if args.dtype == "float32" else np.int16
    print("Generating synthetic RF data ...")
    return synth_rf_data(shape, dtype, seed=args.seed), f"synthetic dtype={args.dtype}"


# --------------------------------------------------------------------------- #
# Codecs
# --------------------------------------------------------------------------- #
def codec_configs(all_codecs: bool):
    """``(label, compression_kwargs)`` pairs, as ``h5py.create_dataset`` takes them.

    Every codec here decodes **in-process** through :mod:`zea.data.chunk_reader` (Blosc,
    Zstd, LZ4, gzip), so they are timed on the same concurrent read path and the comparison
    is about the codec rather than about the reader. The exception is lzf, which has no
    decoder outside h5py — kept precisely to show what that costs, since it was zea's default
    before 0.1.3.

    Blosc2 and Bitshuffle are **not** here. Both were implemented, measured and rejected:
    their Python bindings hold the GIL, so they cannot decode concurrently (Blosc2 scaled
    1.1x across the thread pool where Blosc scaled 7.2x), and no compression ratio recovers
    a ~30x loss of read throughput. Re-adding them means re-adding a dependency *and* a
    decoder, so they are left out rather than left in as a tempting-looking row.
    """
    configs = [("none", {})]
    if not HAVE_HDF5PLUGIN:
        return configs + [("lzf (pre-0.1.3)", {"compression": "lzf"})]

    B = hdf5plugin.Blosc

    def blosc(cname, clevel, shuffle=B.SHUFFLE):
        return dict(B(cname=cname, clevel=clevel, shuffle=shuffle))

    # The two questions that are actually open: where on the zstd clevel curve to sit, and
    # whether to shuffle by byte or by bit. The shuffle is not a detail — it decides how much
    # structure zstd can find, and it is dtype-dependent (a float32 has 4 byte-planes to
    # separate, an int16 only 2), so both are swept against both dtypes.
    configs += [(f"blosc-zstd-{c}", blosc("zstd", c)) for c in (1, 3, 5, 7, 9)]
    configs += [
        (f"blosc-zstd-{c}-bitshuf", blosc("zstd", c, B.BITSHUFFLE)) for c in (1, 3, 5, 7, 9)
    ]
    configs += [
        ("blosc-lz4-5", blosc("lz4", 5)),
        ("blosc-lz4-9", blosc("lz4", 9)),
        ("zstd-3 (no shuffle)", dict(hdf5plugin.Zstd(clevel=3))),
        ("lzf (pre-0.1.3, h5py-only)", {"compression": "lzf"}),
    ]
    if all_codecs:
        configs += [
            ("blosc-blosclz-5", blosc("blosclz", 5)),
            ("blosc-lz4hc-5", blosc("lz4hc", 5)),
            ("blosc-zlib-5", blosc("zlib", 5)),
            ("lz4 (hdf5)", dict(hdf5plugin.LZ4())),
            ("gzip-4", {"compression": "gzip", "compression_opts": 4}),
        ]
    return configs


DEFAULT_CODEC = "blosc-zstd-5"  # zea's current default; the chunk sweep holds it fixed


def codec_by_label(label, all_codecs=True):
    for name, kwargs in codec_configs(all_codecs):
        if name == label:
            return kwargs
    raise SystemExit(f"Unknown codec {label!r}. Options: {[c for c, _ in codec_configs(True)]}")


# --------------------------------------------------------------------------- #
# Write / read
# --------------------------------------------------------------------------- #
def chunk_shape(raw, tx_per_chunk):
    """``(1, tx_per_chunk, n_ax, n_el, n_ch)`` — one frame per chunk, split over transmits.

    ``tx_per_chunk=None`` is the whole frame in one chunk (zea's default today). Written with
    plain h5py because ``zea.File.create`` cannot *express* a partial-transmit chunk: its
    ``chunk_axes`` marks an axis as size-1-or-full, so "8 transmits" has no spelling yet. That
    gap is the thing this sweep is here to price.
    """
    n_tx = raw.shape[TX_AXIS]
    tx = n_tx if tx_per_chunk is None else min(tx_per_chunk, n_tx)
    return (1, tx) + raw.shape[2:]


def write_file(path, raw, chunks, compression):
    """Write ``raw`` as a zea-shaped, paged HDF5 file. Returns seconds elapsed."""
    t0 = time.perf_counter()
    with h5py.File(path, "w", **PAGED_LAYOUT) as f:
        f.create_dataset(DSET, data=raw, chunks=chunks, **compression)
    return time.perf_counter() - t0


def drop_cache(path):
    """Evict the file from the page cache, so the next read really hits the disk.

    ``POSIX_FADV_DONTNEED`` drops the file's clean pages; no root needed (unlike
    ``/proc/sys/vm/drop_caches``). Warm and cold reads are different questions and zea-sized
    files are almost always cold, so both are reported.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def time_read(path, selection, *, cold=False, use_h5py=False, repeats=3):
    """Best-of-N seconds for ``dset[selection]`` through the direct reader (or h5py).

    Cold reads are *not* repeated-best: dropping the cache each time is the measurement, and
    taking the min of several would just be the min of several cold reads. They are noisier;
    that is inherent.
    """
    best, result = float("inf"), None
    for _ in range(1 if cold else repeats):
        if cold:
            drop_cache(path)
        fetcher = None if use_h5py else LocalFetcher(path)
        try:
            t0 = time.perf_counter()
            with h5py.File(path, "r") as f:
                dset = f[DSET]
                result = dset[selection] if use_h5py else direct_read(dset, selection, fetcher)
            best = min(best, time.perf_counter() - t0)
        finally:
            if fetcher is not None:
                fetcher.close()
    return best, result


# --------------------------------------------------------------------------- #
# The cloud stand-in: an HTTP server that counts requests and adds latency to each
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
    """Range-capable HTTP server that counts requests and sleeps ``latency`` on each.

    Serving the file locally holds bandwidth constant and makes *round trips* the only thing
    that varies, which is what chunk size changes over the network: N chunks fetched together
    cost one round trip, N chunks fetched by h5py cost N.
    """

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _file(self):
            return os.path.join(directory, self.path.lstrip("/").split("?")[0])

        def do_HEAD(self):
            counter.bump()
            time.sleep(latency)
            try:
                size = os.path.getsize(self._file())
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
                size = os.path.getsize(self._file())
            except OSError:
                self.send_error(404)
                return
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                start_s, _, end_s = rng[6:].partition("-")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else size - 1
                with open(self._file(), "rb") as handle:
                    handle.seek(start)
                    body = handle.read(end - start + 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{start + len(body) - 1}/{size}")
            else:
                with open(self._file(), "rb") as handle:
                    body = handle.read()
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(body)

    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/"


def time_remote_read(path, url, counter, selection):
    """``(requests, seconds)`` for a read whose bytes come over HTTP.

    The chunk *manifest* is taken from the local handle: a remote open costs the same ~2
    requests whatever the chunk size, and folding it in would only add a constant. What is
    being timed is the fetch+decode of the chunks themselves.
    """
    fetcher = HTTPFetcher(url)
    with h5py.File(path, "r") as f:
        dset = f[DSET]
        n0, t0 = counter.value(), time.perf_counter()
        result = direct_read(dset, selection, fetcher)
        return counter.value() - n0, time.perf_counter() - t0, result


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #
def sweep_codecs(raw, outdir, args):
    """Codec x clevel, at a fixed chunk size. Answers: which codec, and how hard."""
    raw_mb = raw.nbytes / 1e6
    chunks = chunk_shape(raw, args.codec_sweep_tx)
    chunk_mb = np.prod(chunks) * raw.dtype.itemsize / 1e6
    one_frame = 0  # a single-frame read: the interactive latency users feel

    print(f"\n{'=' * 96}\nCODECS  (chunk = {chunks}, {chunk_mb:.1f} MB raw)\n{'=' * 96}")
    header = (
        f"{'codec':<22}{'ratio':>7}{'stored MB':>11}{'write MB/s':>12}"
        f"{'warm MB/s':>11}{'cold MB/s':>11}{'1 frame ms':>12}"
    )
    print(header + "\n" + "-" * len(header))

    rows = []
    for label, compression in codec_configs(args.all_codecs):
        path = str(Path(outdir) / f"codec_{label.split()[0]}.hdf5")
        try:
            write_s = write_file(path, raw, chunks, compression)
            with h5py.File(path, "r") as f:
                stored = f[DSET].id.get_storage_size()

            warm_s, got = time_read(path, slice(None), repeats=args.repeats)
            assert np.array_equal(got, raw), f"{label}: round-trip mismatch"
            cold_s, _ = time_read(path, slice(None), cold=True)
            frame_s, _ = time_read(path, one_frame, repeats=args.repeats)

            ratio, stored_mb = raw.nbytes / stored, stored / 1e6
            rows.append((label, ratio, stored_mb, raw_mb / cold_s, frame_s))
            print(
                f"{label:<22}{ratio:>7.2f}{stored_mb:>11.0f}{raw_mb / write_s:>12.0f}"
                f"{raw_mb / warm_s:>11.0f}{raw_mb / cold_s:>11.0f}{frame_s * 1e3:>12.0f}"
            )
        except Exception as e:  # noqa: BLE001 — report and carry on to the next codec
            print(f"{label:<22}  FAILED: {type(e).__name__}: {e}")
        finally:
            Path(path).unlink(missing_ok=True)
    return rows


def sweep_chunks(raw, outdir, args):
    """Chunk size, at a fixed codec. Answers: how much data belongs in one chunk."""
    raw_mb = raw.nbytes / 1e6
    n_frames, n_tx = raw.shape[FRAME_AXIS], raw.shape[TX_AXIS]
    compression = codec_by_label(args.codec)
    counter = _Counter()
    httpd, base = make_server(outdir, counter, args.latency)

    print(f"\n{'=' * 108}")
    print(f"CHUNK SIZE  (codec = {args.codec}, cloud = {args.latency * 1e3:.0f} ms/request)")
    print("=" * 108)
    header = (
        f"{'tx/chunk':>9}{'chunk MB':>10}{'chunks/frame':>14}{'stored MB':>11}"
        f"{'warm MB/s':>11}{'cold MB/s':>11}{'1 frame ms':>12}{'h5py 1f ms':>12}"
        f"{'cloud req':>11}{'cloud ms':>10}"
    )
    print(header + "\n" + "-" * len(header))

    rows = []
    try:
        for tx in args.tx_per_chunk:
            if tx is not None and tx > n_tx:
                continue
            chunks = chunk_shape(raw, tx)
            name = f"chunk_{tx or 'full'}.hdf5"
            path = str(Path(outdir) / name)
            label = "full" if tx is None or tx == n_tx else str(tx)
            try:
                write_file(path, raw, chunks, compression)
                with h5py.File(path, "r") as f:
                    stored = f[DSET].id.get_storage_size()

                warm_s, got = time_read(path, slice(None), repeats=args.repeats)
                assert np.array_equal(got, raw), f"tx/chunk={label}: round-trip mismatch"
                cold_s, _ = time_read(path, slice(None), cold=True)
                frame_s, _ = time_read(path, 0, repeats=args.repeats)
                h5py_s, _ = time_read(path, 0, use_h5py=True, repeats=args.repeats)

                reqs, cloud_s, got = time_remote_read(path, base + name, counter, 0)
                assert np.array_equal(got, raw[0]), f"tx/chunk={label}: remote mismatch"

                chunk_mb = float(np.prod(chunks)) * raw.dtype.itemsize / 1e6
                per_frame = int(np.ceil(n_tx / chunks[TX_AXIS]))
                rows.append((label, chunk_mb, stored / 1e6, frame_s, cloud_s, reqs))
                print(
                    f"{label:>9}{chunk_mb:>10.1f}{per_frame:>14}{stored / 1e6:>11.0f}"
                    f"{raw_mb / warm_s:>11.0f}{raw_mb / cold_s:>11.0f}{frame_s * 1e3:>12.0f}"
                    f"{h5py_s * 1e3:>12.0f}{reqs:>11}{cloud_s * 1e3:>10.0f}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"{label:>9}  FAILED: {type(e).__name__}: {e}")
            finally:
                Path(path).unlink(missing_ok=True)
    finally:
        httpd.shutdown()

    print(
        f"\n('1 frame' reads frame 0 = {raw[0].nbytes / 1e6:.0f} MB raw, over {n_frames} frames "
        f"x {n_tx} tx. 'cloud req' is the round trips that read costs.)"
    )
    return rows


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--from-file", default=None, help="Real zea file (local path or hf://...).")
    p.add_argument("--n-frames", type=int, default=6, help="Frames to load/write.")
    p.add_argument("--n-tx", type=int, default=64, help="Transmits (synthetic only).")
    p.add_argument("--n-ax", type=int, default=2176, help="Axial samples (synthetic only).")
    p.add_argument("--n-el", type=int, default=128, help="Elements (synthetic only).")
    p.add_argument("--n-ch", type=int, default=1, help="Channels (synthetic only).")
    p.add_argument("--dtype", choices=["float32", "int16"], default="float32")
    p.add_argument(
        "--cast",
        choices=["none", "int16"],
        default="none",
        help="Quantise a float32 --from-file to int16 (what new acquisitions store).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--only", choices=["codecs", "chunks", "both"], default="both", help="Which sweep to run."
    )
    p.add_argument("--all-codecs", action="store_true", help="Also try the long-shot codecs.")
    p.add_argument("--codec", default=DEFAULT_CODEC, help="Codec held fixed in the chunk sweep.")
    p.add_argument(
        "--codec-sweep-tx",
        type=int,
        default=16,
        help="Transmits per chunk held fixed in the codec sweep.",
    )
    p.add_argument(
        "--tx-per-chunk",
        type=lambda s: None if s == "full" else int(s),
        nargs="+",
        default=list(DEFAULT_TX_PER_CHUNK),
        help="Chunk sizes to sweep, in transmits ('full' = one frame per chunk).",
    )
    p.add_argument("--latency", type=float, default=0.02, help="Seconds per HTTP request.")
    p.add_argument("--repeats", type=int, default=3, help="Warm-read repeats (best of).")
    p.add_argument("--outdir", default=None, help="Where to write (default: a temp dir).")
    args = p.parse_args()

    if not HAVE_HDF5PLUGIN:
        print("WARNING: `hdf5plugin` is not installed — install it to compare Blosc/zstd.\n")

    raw, source = load_source(args)
    print(f"Source: {source}")
    print(f"raw_data: shape={raw.shape} dtype={raw.dtype} raw={raw.nbytes / 1e6:.0f} MB")

    outdir = args.outdir or tempfile.mkdtemp(prefix="zea_bench_")
    os.makedirs(outdir, exist_ok=True)
    if args.only in ("codecs", "both"):
        sweep_codecs(raw, outdir, args)
    if args.only in ("chunks", "both"):
        sweep_chunks(raw, outdir, args)


if __name__ == "__main__":
    main()
