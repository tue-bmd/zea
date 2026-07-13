"""Benchmark HDF5 compression filters for zea ultrasound files.

Compares h5py's built-in filters (lzf, gzip) against the external HDF5 filter
plugins shipped by ``hdf5plugin`` (Blosc, Blosc2, Zstd, LZ4, BitShuffle), writing
through ``zea.File.create`` so the files are real zea files with zea's own
chunking. It measures the four things that matter for zea data:

  * compression ratio (raw bytes / stored bytes of ``raw_data``)
  * write throughput
  * full-read throughput
  * partial-read time when subsampling frames *and* transmits — zea's real
    access pattern, which is why zea chunks per ``(frame, transmit)`` plane.

All codecs here are lossless; round-trip equality is asserted for each.

Install the plugins first (you do NOT need to build the HDFGroup C repo)::

    pip install hdf5plugin

Benchmark on synthetic data (default) or on a real zea file — the real file is
loaded and re-saved under each codec::

    python scripts/benchmark_compression.py
    python scripts/benchmark_compression.py --dtype int16 --n-tx 32
    python scripts/benchmark_compression.py --from-file my_scan.hdf5 --n-frames 4
    python scripts/benchmark_compression.py --from-file hf://zeahub/... --n-frames 2

``--n-frames`` caps how many frames are loaded/written (keeps size and time
sane); ``--from-file`` keeps all transmits so scan parameters stay consistent.

Note: files written with hdf5plugin filters require ``import hdf5plugin`` to be
read back anywhere (the filter must be registered in the reading process too).
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    import hdf5plugin  # noqa: F401  (registers the external filters on import)

    HAVE_HDF5PLUGIN = True
except ImportError:
    HAVE_HDF5PLUGIN = False

import h5py

import zea

# Quiet zea's per-file "File saved" info logs so the benchmark table stays clean.
logging.getLogger("zea").setLevel(logging.WARNING)

# raw_data axis order is (n_frames, n_tx, n_ax, n_el, n_ch); zea subsamples the
# first two, so partial reads select over axis 0 (frames) and axis 1 (transmits).
FRAME_AXIS, TX_AXIS = 0, 1


def synth_rf_data(shape, dtype, seed=0):
    """Generate synthetic RF-like data with realistic compressibility.

    Pure noise is incompressible and would make every codec look identical, so
    we build structured data: a depth-attenuated oscillating carrier (the pulse)
    modulated by smooth per-element apodisation and correlated speckle, plus a
    little additive noise and small frame-to-frame motion. This has the spatial
    correlation and limited entropy that real channel data has, so compression
    ratios are representative rather than ~1.0.
    """
    n_frames, n_tx, n_ax, n_el, n_ch = shape
    rng = np.random.default_rng(seed)

    z = np.linspace(0.0, 1.0, n_ax)[:, None]  # (n_ax, 1) axial/depth
    envelope = np.exp(-2.5 * z)  # depth-dependent attenuation
    carrier = np.sin(2.0 * np.pi * 40.0 * z)  # transducer carrier oscillation
    el = np.linspace(-1.0, 1.0, n_el)[None, :]  # (1, n_el)
    apod = 0.5 + 0.5 * np.cos(np.pi * el)  # element apodisation
    base = (envelope * carrier) * apod  # (n_ax, n_el)

    out = np.empty(shape, dtype=np.float32)
    for fi in range(n_frames):
        motion = 1.0 + 0.03 * fi  # gentle frame-to-frame change
        for ti in range(n_tx):
            speckle = rng.standard_normal((n_ax, n_el))
            speckle = np.cumsum(speckle, axis=0)  # smooth along depth
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
    return out.astype(dtype)


def dummy_scan(n_tx, n_el):
    """Minimal but valid scan parameters for a synthetic zea file."""
    return {
        "sampling_frequency": np.float32(40e6),
        "center_frequency": np.float32(7e6),
        "demodulation_frequency": np.float32(7e6),
        "initial_times": np.zeros(n_tx, dtype=np.float32),
        "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
        "sound_speed": np.float32(1540.0),
        "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
        "focus_distances": np.full(n_tx, np.inf, dtype=np.float32),
        "transmit_origins": np.zeros((n_tx, 3), dtype=np.float32),
        "polar_angles": np.zeros(n_tx, dtype=np.float32),
        "azimuth_angles": np.zeros(n_tx, dtype=np.float32),
    }


def dummy_probe(n_el):
    geometry = np.zeros((n_el, 3), dtype=np.float32)
    geometry[:, 0] = np.linspace(-0.02, 0.02, n_el)
    return {"name": "generic", "probe_geometry": geometry}


def codec_configs():
    """List of (label, compression) accepted by ``zea.File.create``. hdf5plugin
    entries are added only when the package is importable."""
    configs = [
        ("none", None),
        ("lzf (current)", "lzf"),
        ("gzip-4", {"compression": "gzip", "compression_opts": 4}),
    ]
    if not HAVE_HDF5PLUGIN:
        return configs

    B = hdf5plugin.Blosc
    B2 = hdf5plugin.Blosc2
    configs += [
        ("blosc-lz4-shuffle", dict(hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=B.SHUFFLE))),
        ("blosc-zstd-shuffle", dict(hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=B.SHUFFLE))),
        ("blosc2-zstd-bitshuffle", dict(B2(cname="zstd", clevel=5, filters=B2.BITSHUFFLE))),
        ("blosc2-lz4-bitshuffle", dict(B2(cname="lz4", clevel=5, filters=B2.BITSHUFFLE))),
        ("zstd-3", dict(hdf5plugin.Zstd(clevel=3))),
        ("bitshuffle-lz4", dict(hdf5plugin.Bitshuffle(cname="lz4"))),
    ]
    return configs


def load_source(args):
    """Return (raw_data, scan, probe, description) for the chosen data source."""
    if args.from_file:
        print(f"Loading real data from {args.from_file} ...")
        with zea.File(args.from_file, mode="r") as f:
            params = f.load_parameters()
            raw = f.data.raw_data[: args.n_frames]  # cap frames (streams if hf://)
        scan = params.to_scan_dict()
        probe = params.to_probe_dict()
        return raw, scan, probe, f"file={args.from_file}"

    shape = (args.n_frames, args.n_tx, args.n_ax, args.n_el, args.n_ch)
    dtype = np.float32 if args.dtype == "float32" else np.int16
    print("Generating synthetic RF data ...")
    raw = synth_rf_data(shape, dtype, seed=args.seed)
    n_tx, n_el = raw.shape[TX_AXIS], raw.shape[3]
    return raw, dummy_scan(n_tx, n_el), dummy_probe(n_el), f"synthetic dtype={dtype.__name__}"


def _raw_dataset_name(f):
    for candidate in ("tracks/track_0/data/raw_data", "data/raw_data"):
        if candidate in f:
            return candidate
    raise KeyError("raw_data dataset not found in file")


def _time_full_read(path, name, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        with h5py.File(path, "r") as f:
            _ = f[name][:]
        best = min(best, time.perf_counter() - t0)
    return best


def _time_partial_read(path, name, fsel, tsel, repeats=3):
    """Read only the selected (frame, transmit) planes.

    h5py allows fancy indexing on a single axis at a time, so we select each
    frame with a scalar index and the transmits with a list — this fetches only
    the requested planes from disk (mirroring zea's frame/transmit subsampling).
    """
    tsel = list(map(int, tsel))
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        with h5py.File(path, "r") as f:
            ds = f[name]
            _ = [ds[int(fi), tsel] for fi in fsel]
        best = min(best, time.perf_counter() - t0)
    return best


def benchmark(raw, scan, probe, outdir, keep_files):
    raw_mb = raw.nbytes / 1e6
    n_frames, n_tx = raw.shape[FRAME_AXIS], raw.shape[TX_AXIS]

    # Fixed random subsample of frames + transmits (zea's partial-read pattern).
    rng = np.random.default_rng(1234)
    fsel = np.sort(rng.choice(n_frames, max(1, n_frames // 4), replace=False))
    tsel = np.sort(rng.choice(n_tx, max(1, n_tx // 4), replace=False))
    partial_mb = raw[fsel][:, tsel].nbytes / 1e6

    print(f"\nraw_data: shape={raw.shape} dtype={raw.dtype} raw={raw_mb:.1f} MB")
    print(
        f"Partial read: {len(fsel)}/{n_frames} frames x {len(tsel)}/{n_tx} tx "
        f"= {partial_mb:.1f} MB\n"
    )

    header = (
        f"{'codec':<24}{'ratio':>7}{'stored MB':>11}{'write MB/s':>12}"
        f"{'read MB/s':>11}{'partial ms':>12}{'ok':>4}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for label, compression in codec_configs():
        path = Path(outdir) / f"bench_{label.replace(' ', '_').replace('/', '')}.hdf5"
        try:
            t0 = time.perf_counter()
            zea.File.create(
                str(path),
                data={"raw_data": raw},
                scan=scan,
                probe=probe,
                compression=compression,
                overwrite=True,
                ignore_warnings=True,
            )
            write_s = time.perf_counter() - t0

            with h5py.File(path, "r") as f:
                name = _raw_dataset_name(f)
                stored = f[name].id.get_storage_size()
                ok = np.array_equal(f[name][:], raw)
            stored_mb = stored / 1e6
            ratio = raw.nbytes / stored if stored else float("nan")

            full_s = _time_full_read(path, name)
            partial_s = _time_partial_read(path, name, fsel, tsel)

            rows.append((label, ratio, stored_mb, raw_mb / write_s, raw_mb / full_s, partial_s, ok))
            print(
                f"{label:<24}{ratio:>7.2f}{stored_mb:>11.1f}{raw_mb / write_s:>12.0f}"
                f"{raw_mb / full_s:>11.0f}{partial_s * 1e3:>12.1f}{'Y' if ok else 'N':>4}"
            )
        except Exception as e:  # noqa: BLE001 — report and continue to next codec
            print(f"{label:<24}  FAILED: {type(e).__name__}: {e}")
        finally:
            if not keep_files and path.exists():
                path.unlink()

    if rows:
        best_ratio = max(rows, key=lambda r: r[1])
        best_partial = min(rows, key=lambda r: r[5])
        print(f"\nBest ratio:        {best_ratio[0]} ({best_ratio[1]:.2f}x)")
        print(f"Fastest partial:   {best_partial[0]} ({best_partial[5] * 1e3:.1f} ms)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--from-file", default=None, help="Real zea file (local path or hf://...).")
    p.add_argument("--n-frames", type=int, default=6, help="Frames to write (caps --from-file).")
    p.add_argument("--n-tx", type=int, default=16, help="Transmits (synthetic only).")
    p.add_argument("--n-ax", type=int, default=2048, help="Axial samples (synthetic only).")
    p.add_argument("--n-el", type=int, default=128, help="Elements (synthetic only).")
    p.add_argument("--n-ch", type=int, default=1, help="Channels (synthetic only).")
    p.add_argument("--dtype", choices=["float32", "int16"], default="float32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--outdir", default=None, help="Where to write temp files (default: a temp dir)."
    )
    p.add_argument("--keep-files", action="store_true", help="Keep the .hdf5 files for inspection.")
    args = p.parse_args()

    if not HAVE_HDF5PLUGIN:
        print(
            "WARNING: `hdf5plugin` is not installed — only lzf/gzip will be benchmarked.\n"
            "         Install it with `pip install hdf5plugin` to compare Blosc/Zstd/etc.\n"
        )

    raw, scan, probe, source = load_source(args)
    print(f"Source: {source}")

    outdir = args.outdir or tempfile.mkdtemp(prefix="zea_bench_")
    os.makedirs(outdir, exist_ok=True)
    try:
        benchmark(raw, scan, probe, outdir, args.keep_files)
    finally:
        if args.keep_files:
            print(f"\nFiles kept in: {outdir}")


if __name__ == "__main__":
    main()
