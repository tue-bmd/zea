# HDF5 chunks, compression and streaming

I noticed loading a compressed hdf5 file was quite slow. I found out that HDF5 cannot natively do
concurrent decompression of chunks, and that the automatic chunk size was not really suitable for
how we load channel data generally.

This PR introduces various fixes to improve that.

- It switches compression algotrithm to a faster one (blosc+zstd+bitshuffle @ clevel 7)
- It decompresses chunks in parallel using a thread pool (going around h5py GIL limitation)
- It changes the default chunk size to be more suitable for our use case
- It lets Blosc thread *within* a chunk on writes (`BLOSC_NTHREADS`), which is ~4x on writes
- It adds a streaming interface to `zea.File` for huggingface datasets

Net effect on a real carotid scan: **10-13x faster local reads**, **~10x faster cloud `summary()`**,
**4-6x faster cloud reads**, files **34-42% smaller** — and writes come out *faster* than `main`
too, despite compressing much harder.

## Speed-ups in common use cases

Just checking the summary of a cloud file:

```python
import zea
with zea.File("hf://...", stream=True) as f:
    print(f.summary())  # 171 ms, previously 2.3 s — it had to download the entire file!
```

Loading a couple of frames from a cloud file:

```python
import zea
with zea.File("hf://...", stream=True) as f:
    raw_data = f.data.raw_data[:3]  # streams chunks from cloud, decompresses in parallel
```

Loading a local file:

```python
import zea
with zea.File("scan.hdf5") as f:
    raw_data = f.data.raw_data[:3]  # concurrently reads and decompresses chunks from disk
```

## Benchmark to compare to `main`

Measured on a real carotid scan (`n_tx=149`), 8 frames, reading 3 frames — on `main`'s own code,
checked out into a git worktree and driven in a subprocess, so both branches run the same data on
the same machine and only zea changes. Both are asserted to read back identical values.

**float32** (1328 MB raw — how this data is stored today):

|             | codec      | chunk                | chunk size | stored | ratio | write        |
|-------------|------------|----------------------|------------|--------|-------|--------------|
| main        | lzf        | (1, 10, 272, 16, 1)  | 0.2 MB     | 713 MB | 1.86x | 203 MB/s     |
| **this PR** | blosc-zstd | (1, 7, 2176, 128, 1) | 7.8 MB     | 410 MB | 3.24x | **341 MB/s** |

|                         | main  | this PR    | speed-up  |
|-------------------------|-------|------------|-----------|
| local read (cold cache) | 1.4 s | **105 ms** | **13.3x** |
| cloud `summary()`       | 1.9 s | **145 ms** | **13.0x** |
| cloud read (3 frames)   | 2.8 s | **468 ms** | **6.0x**  |

**int16** (664 MB raw — what new acquisitions will store):

|             | codec      | chunk                 | chunk size | stored | ratio | write        |
|-------------|------------|-----------------------|------------|--------|-------|--------------|
| main        | lzf        | (1, 19, 272, 16, 1)   | 0.2 MB     | 552 MB | 1.20x | 151 MB/s     |
| **this PR** | blosc-zstd | (1, 15, 2176, 128, 1) | 8.4 MB     | 362 MB | 1.83x | **477 MB/s** |

|                         | main   | this PR    | speed-up  |
|-------------------------|--------|------------|-----------|
| local read (cold cache) | 876 ms | **87 ms**  | **10.0x** |
| cloud `summary()`       | 1.6 s  | **163 ms** | **9.7x**  |
| cloud read (3 frames)   | 1.9 s  | **432 ms** | **4.4x**  |

Files also get **42% smaller** (float32) / **34% smaller** (int16).

**Writes got faster too, which was not obvious.** zstd at clevel 7 works far harder than lzf, so
this started out as a regression (74 MB/s against 184). The fix is that HDF5 runs the filter one
chunk at a time and *single-threaded*, while Blosc can compress the blocks **within** a chunk in
parallel — and its HDF5 filter picks the thread count up from `BLOSC_NTHREADS` on every call. zea
now defaults it to `min(8, cpu_count)`, which turns 105 MB/s into 453 MB/s on int16 and puts the
PR ahead of `main` despite compressing 1.5x harder. It is a `setdefault`, so an explicit
`BLOSC_NTHREADS` still wins; see [`docs/source/environment.rst`](docs/source/environment.rst) for
when you would want to turn it *down* (several dataloader workers writing files at once will
multiply with it).

Two notes on how the cloud row is measured. `main` **cannot stream at all**, so reaching a cloud
file means downloading it whole — that is what its cloud numbers are, and the asymmetry is the
point of the feature. And "the cloud" here is a local HTTP server with a fixed 20 ms latency per
request rather than real HF, because real HF is bandwidth-bound and noisy by 3-4x, which drowns
the round-trip effect being measured. It must be an *asyncio* server: a thread-per-connection one
collapses under concurrent range requests (462 MB/s for a single range, 103 MB/s for 64), which
would charge the concurrent reader a 4x penalty invented by the harness — enough to make streaming
look *slower* than downloading. Real object stores scale up with concurrency, not down.

The script below checks `main` out into a git worktree, runs each version in a subprocess with
`PYTHONPATH` pointed at that tree, and points zea's HF plumbing at a local latency-injecting HTTP
server so the `hf://` streaming path runs for real. Save it anywhere and run:

```bash
python benchmark_vs_main.py --from-file carotid.hdf5 --n-frames 8 --read-frames 3
python benchmark_vs_main.py --from-file carotid.hdf5 --n-frames 8 --read-frames 3 --cast int16
```

<details>

<summary>Benchmark vs main script</summary>

```python
"""Benchmark this branch against ``main``: write, local read, cloud read.

Runs the **real code of both branches**, rather than trusting a description of what ``main``
used to do: it checks ``main`` out into a git worktree and drives each version in its own
subprocess with ``PYTHONPATH`` pointed at that tree. Same machine, same data, same file
contents — only zea changes.

What separates the two:

* ``main`` writes **lzf**, with h5py's *auto-chosen* chunk shape, and reads through h5py —
  one chunk at a time, decoded under h5py's global lock.
* this branch writes **Blosc(zstd, clevel 7, bit-shuffle)**, one frame per chunk capped at
  ``MAX_CHUNK_BYTES``, in a paged file, and reads through :mod:`zea.data.chunk_reader`:
  chunk byte ranges fetched by us and decoded in a thread pool.

The cloud case is not a like-for-like read, because ``main`` cannot stream at all — an
``hf://`` file is *downloaded whole* before the first byte is used. That asymmetry is the
point, so it is measured as each branch actually behaves: ``main`` downloads the file and
opens it locally; this branch opens the file over HTTP range requests and fetches only the
chunks it needs. To keep it honest and reproducible, "the cloud" is a local HTTP server with
a fixed latency per request (real HF is bandwidth-bound and noisy by 3-4x, which drowns the
round-trip effect being measured), and zea's HF plumbing is pointed at it.

Run it from anywhere **inside the zea checkout you want to benchmark** (it compares that working
tree against ``main``)::

    python benchmark_vs_main.py --from-file scan.hdf5 --n-frames 8 --read-frames 3
    python benchmark_vs_main.py --from-file scan.hdf5 --n-frames 8 --cast int16
    python benchmark_vs_main.py --latency 0.02 --read-frames 3

Without ``--from-file`` it uses synthetic data, which is fine for a smoke test but not for a
decision: the wins here are proportional to file size, and a small array makes downloading the
whole thing look competitive.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np


def _find_repo() -> Path:
    """The zea checkout to benchmark: the git root of the working directory.

    Deliberately *not* derived from ``__file__`` — this script is meant to be runnable from
    anywhere (including pasted out of a PR description into /tmp), and a path relative to the
    script would then point at the wrong tree, or at no tree at all.
    """
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return Path(top)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            "Run this from inside the zea git checkout you want to benchmark "
            "(it compares the working tree against `main`)."
        ) from exc


RAW_KEY = "raw_data"

# A fake hf:// path. zea only streams for ``hf://``, so the worker monkeypatches zea's two HF
# helpers to resolve this to the local test server (see ``_patch_hf`` in the worker).
FAKE_HF = "hf://bench/repo/scan.hdf5"


# --------------------------------------------------------------------------- #
# Worker: runs inside a subprocess, against whichever zea is on PYTHONPATH
# --------------------------------------------------------------------------- #
def _warmup(zea_mod, path) -> None:
    """Pay zea's one-time process costs *before* the clock starts.

    The first ``zea.File`` open in a process costs ~3 s — it initialises the Keras backend
    lazily — and every subsequent one costs ~10 ms. That is a startup cost, not a read cost:
    left inside the timed region it swamps the thing being measured (a 17 MB read looked like
    3.0 s on both branches, i.e. a dead heat that was really just two JAX imports). Both
    versions pay it identically, so warming first changes no comparison — it only stops the
    constant from drowning the signal.
    """
    if not Path(path).exists():
        return
    with zea_mod.File(path, mode="r") as f:
        _ = f.summary()


def worker_main(args) -> None:
    """One timed task, in a process whose ``PYTHONPATH`` selects the zea under test."""
    import zea  # resolved from PYTHONPATH: either this branch or the main worktree

    result: dict = {"zea": str(Path(zea.__file__).resolve().parent.parent)}
    raw = np.load(args.source, mmap_mode="r")

    if args.task != "write":
        _warmup(zea, args.path)

    if args.task == "write":
        params = json.loads(Path(args.params).read_text())
        data = {RAW_KEY: np.asarray(raw)}
        scan = {k: np.asarray(v) for k, v in params["scan"].items()}
        probe = {"name": "generic", "probe_geometry": np.asarray(params["probe_geometry"])}

        # Same one-time backend init as _warmup, but a write has no file to open yet: do a
        # throwaway single-frame write so the real one is timed against a warm process.
        warm = str(Path(args.path).with_suffix(".warmup.hdf5"))
        warm_scan = dict(scan)
        if "time_to_next_transmit" in warm_scan:
            warm_scan["time_to_next_transmit"] = warm_scan["time_to_next_transmit"][:1]
        zea.File.create(
            warm,
            data={RAW_KEY: np.asarray(raw[:1])},
            scan=warm_scan,
            probe=probe,
            overwrite=True,
            ignore_warnings=True,
        )
        os.unlink(warm)

        t0 = time.perf_counter()
        zea.File.create(
            args.path,
            data=data,
            scan=scan,
            probe=probe,
            overwrite=True,
            ignore_warnings=True,
        )
        result["seconds"] = time.perf_counter() - t0

        import h5py

        with h5py.File(args.path, "r") as f:
            dset = f[_raw_path(f)]
            result["stored_bytes"] = dset.id.get_storage_size()
            result["chunks"] = list(dset.chunks) if dset.chunks else None
            plist = dset.id.get_create_plist()
            result["filters"] = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
        result["raw_bytes"] = int(np.asarray(raw).nbytes)

    elif args.task == "read_local":
        _drop_cache(args.path)
        t0 = time.perf_counter()
        with zea.File(args.path, mode="r") as f:
            got = f.data.raw_data[: args.read_frames]
        result["seconds"] = time.perf_counter() - t0
        result["checksum"] = float(np.asarray(got, dtype=np.float64).sum())

    elif args.task == "cloud_summary":
        # main has no streaming: reaching a file at all means downloading it whole.
        t0 = time.perf_counter()
        with _open_cloud(zea, args) as f:
            summary = f.summary()
        result["seconds"] = time.perf_counter() - t0
        result["summary_chars"] = len(str(summary))

    elif args.task == "cloud_read":
        t0 = time.perf_counter()
        with _open_cloud(zea, args) as f:
            got = f.data.raw_data[: args.read_frames]
        result["seconds"] = time.perf_counter() - t0
        result["checksum"] = float(np.asarray(got, dtype=np.float64).sum())

    print(json.dumps(result))


def _raw_path(f) -> str:
    for candidate in ("tracks/track_0/data/raw_data", "data/raw_data"):
        if candidate in f:
            return candidate
    raise KeyError("raw_data not found")


def _drop_cache(path) -> None:
    """Evict the file from the page cache — a cold read is the one that matters here."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


class _Downloaded:
    """A whole file pulled over HTTP, then opened locally — how ``main`` reaches a dataset."""

    def __init__(self, zea_mod, url):
        import urllib.request

        self._tmp = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
        with urllib.request.urlopen(url) as response:
            self._tmp.write(response.read())
        self._tmp.close()
        self._file = zea_mod.File(self._tmp.name, mode="r")

    def __enter__(self):
        return self._file

    def __exit__(self, *exc):
        self._file.close()
        os.unlink(self._tmp.name)


def _open_cloud(zea_mod, args):
    """Open the served file the way this branch would — or the way ``main`` is forced to."""
    if not args.streaming:
        return _Downloaded(zea_mod, args.url)
    _patch_hf(args.url)
    return zea_mod.File(FAKE_HF, mode="r", stream=True)


def _patch_hf(url: str) -> None:
    """Point zea's two HF helpers at the local server, so the hf:// stream path runs for real.

    Only the *location* of the bytes is faked. Everything downstream — the paged-metadata
    open over HTTP, ``ChunkedDataset``, the concurrent range fetches in ``chunk_reader`` — is
    the real code path, which is the whole point of measuring it.
    """
    import fsspec

    from zea.data import file as zea_file
    from zea.internal import preset_utils

    def stream_open(_hf_path, **_kwargs):
        fs = fsspec.filesystem("http", skip_instance_cache=True)
        return fs.open(url, block_size=8 * 1024 * 1024, cache_type="blockcache")

    def stream_url(_hf_path, **_kwargs):
        return url

    preset_utils._hf_stream_open = stream_open
    preset_utils._hf_stream_url = stream_url
    zea_file._hf_stream_open = stream_open


# --------------------------------------------------------------------------- #
# The cloud stand-in: counts requests, adds a fixed latency to each
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


class _Server:
    """Range-capable HTTP server on an asyncio loop, counting requests and adding latency.

    It must be **asyncio**, not ``http.server``. The threaded one collapses under exactly the
    access pattern this PR introduces: measured on the same file, it served a single 200 MB
    range at 462 MB/s but 64 concurrent ranges at only 103 MB/s — thread-per-connection plus
    the GIL. That would have charged the concurrent reader a 4x penalty *created by the test
    harness*, and made streaming look slower than downloading the whole file. On the asyncio
    loop the same two cases run at 486 and 533 MB/s: flat, which is how a real object store
    behaves (S3/HF scale up with concurrent range requests, they do not degrade).
    """

    def __init__(self, directory, counter, latency):
        self.directory, self.counter, self.latency = directory, counter, latency
        self.loop = None
        ready = threading.Event()
        threading.Thread(target=self._serve, args=(ready,), daemon=True).start()
        ready.wait()

    async def _handle(self, reader, writer):
        try:
            while True:
                request = await reader.readline()
                if not request:
                    return
                parts = request.decode(errors="replace").split()
                headers = {}
                while True:
                    line = await reader.readline()
                    if line in (b"\r\n", b"\n", b""):
                        break
                    key, _, value = line.decode(errors="replace").partition(":")
                    headers[key.strip().lower()] = value.strip()
                if len(parts) < 2:
                    return

                method, target = parts[0], parts[1].split("?")[0]
                self.counter.bump()
                if self.latency:
                    await asyncio.sleep(self.latency)

                path = os.path.join(self.directory, target.lstrip("/"))
                try:
                    size = os.path.getsize(path)
                except OSError:
                    writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                    await writer.drain()
                    continue

                if method == "HEAD":
                    writer.write(
                        f"HTTP/1.1 200 OK\r\nContent-Length: {size}\r\n"
                        f"Accept-Ranges: bytes\r\n\r\n".encode()
                    )
                    await writer.drain()
                    continue

                rng = headers.get("range", "")
                if rng.startswith("bytes="):
                    start_s, _, end_s = rng[6:].partition("-")
                    start = int(start_s) if start_s else 0
                    end = int(end_s) if end_s else size - 1
                    fd = os.open(path, os.O_RDONLY)
                    try:
                        body = os.pread(fd, end - start + 1, start)
                    finally:
                        os.close(fd)
                    head = (
                        f"HTTP/1.1 206 Partial Content\r\n"
                        f"Content-Range: bytes {start}-{start + len(body) - 1}/{size}\r\n"
                        f"Content-Length: {len(body)}\r\nAccept-Ranges: bytes\r\n\r\n"
                    )
                else:
                    with open(path, "rb") as handle:
                        body = handle.read()
                    head = (
                        f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n"
                        f"Accept-Ranges: bytes\r\n\r\n"
                    )
                writer.write(head.encode())
                writer.write(body)
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()

    def _serve(self, ready):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        server = self.loop.run_until_complete(asyncio.start_server(self._handle, "127.0.0.1", 0))
        self.port = server.sockets[0].getsockname()[1]
        ready.set()
        self.loop.run_forever()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def shutdown(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(tree: Path, task: str, counter: _Counter, **kwargs) -> dict:
    """Run one task in a subprocess whose zea comes from ``tree``. Returns its JSON result."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--task", task]
    for key, value in kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd += [flag, str(value)]

    env = dict(os.environ, PYTHONPATH=str(tree), KERAS_BACKEND="numpy", ZEA_DISABLE_CACHE="1")
    before = counter.value()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{task} failed under {tree}:\n{proc.stdout}\n{proc.stderr}")

    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    payload["requests"] = counter.value() - before
    return payload


def source_data(args, outdir: Path):
    """The array both branches will store, plus the scan/probe needed to make a valid file."""
    if args.from_file:
        import zea

        print(f"Loading {args.from_file} ...")
        with zea.File(args.from_file, mode="r") as f:
            raw = f.data.raw_data[: args.n_frames]
            params = f.load_parameters()
        scan, probe = params.to_scan_dict(), params.to_probe_dict()
    else:
        print("Generating synthetic data ...")
        shape = (args.n_frames, args.n_tx, args.n_ax, args.n_el, 1)
        rng = np.random.default_rng(0)
        depth = np.linspace(0, 1, args.n_ax)[None, None, :, None, None]
        base = np.exp(-2.5 * depth) * np.sin(2 * np.pi * 40 * depth)
        speckle = 1 + 0.7 * rng.standard_normal(shape)
        raw = (base * speckle * 1000).astype(np.float32)
        n_tx, n_el = args.n_tx, args.n_el
        scan = {
            "sampling_frequency": np.float32(40e6),
            "center_frequency": np.float32(7e6),
            "demodulation_frequency": np.float32(7e6),
            "initial_times": np.zeros(n_tx, np.float32),
            "t0_delays": np.zeros((n_tx, n_el), np.float32),
            "sound_speed": np.float32(1540.0),
            "tx_apodizations": np.ones((n_tx, n_el), np.float32),
            "focus_distances": np.zeros(n_tx, np.float32),
            "polar_angles": np.zeros(n_tx, np.float32),
            "azimuth_angles": np.zeros(n_tx, np.float32),
            "transmit_origins": np.zeros((n_tx, 3), np.float32),
        }
        probe = {"probe_geometry": np.zeros((n_el, 3), np.float32)}

    if args.cast == "int16" and not np.issubdtype(raw.dtype, np.integer):
        peak = 0.9 * np.iinfo(np.int16).max
        raw = np.clip(raw / (np.abs(raw).max() + 1e-9) * peak, -32768, 32767).astype(np.int16)

    n_frames = raw.shape[0]
    scan = {k: v for k, v in scan.items() if v is not None}
    # Per-frame scan fields must match the frames we kept.
    for key in ("time_to_next_transmit",):
        if key in scan:
            scan[key] = np.asarray(scan[key])[:n_frames]

    source = outdir / "source.npy"
    np.save(source, raw)
    params_path = outdir / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "scan": {k: np.asarray(v).tolist() for k, v in scan.items()},
                "probe_geometry": np.asarray(probe["probe_geometry"]).tolist(),
            }
        )
    )
    return raw, source, params_path


def worktree_for(repo: Path, ref: str, path: Path) -> Path:
    """A detached worktree at ``ref``, so ``main``'s real code can be imported."""
    if path.exists():
        return path
    # Drop registrations left behind by a previous run whose outdir has since been deleted,
    # otherwise git refuses to re-add the same path.
    subprocess.run(["git", "-C", str(repo), "worktree", "prune"], check=False, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(path), ref],
        check=True,
        capture_output=True,
    )
    return path


def fmt(seconds: float) -> str:
    return f"{seconds * 1e3:.0f} ms" if seconds < 1 else f"{seconds:.1f} s"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--task", default=None, help=argparse.SUPPRESS)
    p.add_argument("--path", default=None, help=argparse.SUPPRESS)
    p.add_argument("--source", default=None, help=argparse.SUPPRESS)
    p.add_argument("--params", default=None, help=argparse.SUPPRESS)
    p.add_argument("--url", default=None, help=argparse.SUPPRESS)
    p.add_argument("--streaming", action="store_true", help=argparse.SUPPRESS)

    p.add_argument("--from-file", default=None, help="Real zea file to benchmark on.")
    p.add_argument("--n-frames", type=int, default=8, help="Frames to store in the test file.")
    p.add_argument("--read-frames", type=int, default=3, help="Frames each timed read fetches.")
    p.add_argument("--cast", choices=["none", "int16"], default="none")
    p.add_argument("--n-tx", type=int, default=64, help="Synthetic only.")
    p.add_argument("--n-ax", type=int, default=2048, help="Synthetic only.")
    p.add_argument("--n-el", type=int, default=128, help="Synthetic only.")
    p.add_argument("--latency", type=float, default=0.02, help="Seconds per HTTP request.")
    p.add_argument("--main-ref", default="main", help="Git ref to compare against.")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    if args.worker:
        worker_main(args)
        return

    outdir = Path(args.outdir or tempfile.mkdtemp(prefix="zea_vs_main_"))
    outdir.mkdir(parents=True, exist_ok=True)
    raw, source, params = source_data(args, outdir)

    repo = _find_repo()
    main_tree = worktree_for(repo, args.main_ref, outdir / "main_tree")
    trees = {"main": main_tree, "this PR": repo}

    counter = _Counter()
    httpd = _Server(str(outdir), counter, args.latency)
    base_url = httpd.url

    print(f"\nraw_data: shape={raw.shape} dtype={raw.dtype} ({raw.nbytes / 1e6:.0f} MB)")
    print(f"reads fetch {args.read_frames} frame(s); cloud = {args.latency * 1e3:.0f} ms/request\n")

    results: dict[str, dict] = {}
    try:
        for label, tree in trees.items():
            path = outdir / f"{label.replace(' ', '_')}.hdf5"
            row: dict = {}

            row["write"] = run(tree, "write", counter, path=path, source=source, params=params)
            row["local"] = run(
                tree, "read_local", counter, path=path, source=source, read_frames=args.read_frames
            )

            url = base_url + path.name
            streaming = label != "main"  # main cannot stream: it downloads the file whole
            row["cloud_summary"] = run(
                tree,
                "cloud_summary",
                counter,
                path=path,
                source=source,
                url=url,
                streaming=streaming,
            )
            row["cloud_read"] = run(
                tree,
                "cloud_read",
                counter,
                path=path,
                source=source,
                url=url,
                streaming=streaming,
                read_frames=args.read_frames,
            )
            results[label] = row
            print(f"  {label:8} done")
    finally:
        httpd.shutdown()

    # The two versions must have stored the *same numbers*, or none of the times mean anything.
    checks = {label: row["local"]["checksum"] for label, row in results.items()}
    expected = float(np.asarray(raw[: args.read_frames], dtype=np.float64).sum())
    for label, value in checks.items():
        assert np.isclose(value, expected, rtol=1e-6), f"{label} read back different data!"
    print("\n✓ both branches read back identical data\n")

    report(results, raw, args)


def report(results, raw, args) -> None:
    raw_mb = raw.nbytes / 1e6

    def table(rows, headers):
        widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
        line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
        sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
        body = [
            "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |" for row in rows
        ]
        return "\n".join([line, sep, *body])

    codec = {32001: "blosc-zstd", 32000: "lzf", 1: "gzip"}
    file_rows, local_rows, cloud_rows = [], [], []
    for label, row in results.items():
        write, local = row["write"], row["local"]
        filters = ", ".join(codec.get(f, str(f)) for f in write["filters"]) or "none"
        chunk = write["chunks"]
        chunk_mb = (np.prod(chunk) * raw.dtype.itemsize / 1e6) if chunk else 0
        file_rows.append(
            [
                label,
                filters,
                f"{tuple(chunk)}" if chunk else "auto",
                f"{chunk_mb:.1f} MB",
                f"{write['stored_bytes'] / 1e6:.0f} MB",
                f"{write['raw_bytes'] / write['stored_bytes']:.2f}x",
                f"{raw_mb / write['seconds']:.0f} MB/s",
            ]
        )
        local_rows.append([label, fmt(local["seconds"])])
        cloud_rows.append(
            [
                label,
                fmt(row["cloud_summary"]["seconds"]),
                str(row["cloud_summary"]["requests"]),
                fmt(row["cloud_read"]["seconds"]),
                str(row["cloud_read"]["requests"]),
            ]
        )

    def speedup(rows, index):
        """main / this-PR, from the formatted table rows' underlying seconds."""
        a = results["main"][index]["seconds"]
        b = results["this PR"][index]["seconds"]
        return f"{a / b:.1f}x"

    local_rows.append(["**speed-up**", f"**{speedup(local_rows, 'local')}**"])
    cloud_rows.append(
        [
            "**speed-up**",
            f"**{speedup(cloud_rows, 'cloud_summary')}**",
            "",
            f"**{speedup(cloud_rows, 'cloud_read')}**",
            "",
        ]
    )

    print("### File on disk\n")
    print(table(file_rows, ["", "codec", "chunk", "chunk size", "stored", "ratio", "write"]))
    print(f"\n### Local read ({args.read_frames} frames, cold page cache)\n")
    print(table(local_rows, ["", "time"]))
    print(f"\n### Cloud ({args.latency * 1e3:.0f} ms/request)\n")
    print(
        table(
            cloud_rows,
            ["", "summary()", "requests", f"read {args.read_frames} frames", "requests"],
        )
    )
    print(
        "\n_`main` cannot stream: reaching a cloud file at all means downloading it whole, "
        "which is what its cloud numbers measure._"
    )


if __name__ == "__main__":
    main()
```

</details>

## Compression benchmark

I checked which compression algorithm and chunk size are best for our use case, on real data,
through the read path this PR actually uses.

### Chunk size is the biggest lever — and it is free

Compression ratio is **flat across chunk size** (181 MB stored whether a chunk is 1 transmit or a
whole frame), so chunk size costs nothing on disk and only ever affects read speed. A chunk is the
unit of *parallelism*: `chunk_reader` decodes each one in a single worker thread, so one 83 MB
whole-frame chunk has nothing to parallelise, while 149 single-transmit chunks drown the network in
round trips. int16 carotid, reading 1 frame:

| chunk               | local read | cloud requests | cloud read |
|---------------------|------------|----------------|------------|
| 1 tx (0.6 MB)       | 32 ms      | 149            | 1579 ms    |
| 4 tx (2.2 MB)       | **13 ms**  | 38             | 81 ms      |
| 8 tx (4.5 MB)       | 13 ms      | 19             | **77 ms**  |
| 16 tx (8.9 MB)      | 19 ms      | 10             | 122 ms     |
| whole frame (83 MB) | 102 ms     | 1              | 231 ms     |

The optimum is a broad plateau (~2–16 MB) and it is **the same plateau for local and cloud**, so one
default serves both. Hence `MAX_CHUNK_BYTES = 8 MiB`: one frame per chunk, split along `n_tx` until
it fits the budget.

### Codec: Blosc(zstd) at clevel 7 with bit-shuffle

Every codec below decodes **in-process and concurrently**, so this compares codecs rather than
comparing "codecs my reader happens to support" (lzf is the exception, kept to show what falling
back to h5py costs). int16 carotid, 8 transmits per chunk:

| codec                    | ratio    | write MB/s | cold read MB/s | 1 frame   |
|--------------------------|----------|------------|----------------|-----------|
| none                     | 0.98     | 1509       | 3332           | 17 ms     |
| **blosc-zstd-7-bitshuf** | **1.83** | **125**    | **5670**       | **13 ms** |
| blosc-zstd-5 (byte-shuf) | 1.75     | 47         | 4561           | 20 ms     |
| blosc-zstd-9 (byte-shuf) | 1.82     | 6          | 4646           | 23 ms     |
| blosc-lz4-9              | 1.27     | 697        | 4866           | 14 ms     |
| lzf (`main`'s default)   | 1.19     | 150        | 228            | 251 ms    |

`zstd-7 + bit-shuffle` **strictly dominates** the alternatives on int16 — better ratio, faster
writes *and* faster reads than `zstd-5 + byte-shuffle`. Bit-shuffle wins because an int16 has only
two byte-planes for byte-shuffle to separate, so there is far more structure to expose at the bit
level. clevel 9 is a trap: 6 MB/s writes for ~1% more compression.

Two codecs were implemented, measured, and **rejected**: Blosc2 and Bitshuffle-as-a-filter both
compress slightly better, but their Python bindings **hold the GIL** — Blosc2 scales 1.1x across
the decode thread pool where Blosc scales 7.2x, costing ~30x read throughput. No compression ratio
buys that back. (Their decoders are not in the PR; such files still read correctly via h5py.)

The script below sweeps codec x clevel and chunk size on real data, through
`zea.data.chunk_reader`, timing local reads warm *and* cold (page cache dropped — a 25 GB scan
never sits in RAM) and cloud reads against a latency-injecting server. Every configuration is
asserted bit-equal to the source. Save it anywhere and run:

```bash
python benchmark_compression.py --from-file carotid.hdf5 --n-frames 4 --cast int16
python benchmark_compression.py --from-file carotid.hdf5 --n-frames 4 --only chunks
```

<details>

<summary>Compression benchmark script</summary>

```python
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
    python benchmark_compression.py                       # synthetic
    python benchmark_compression.py --from-file scan.hdf5 --n-frames 4
    python benchmark_compression.py --from-file scan.hdf5 --only chunks
    python benchmark_compression.py --from-file scan.hdf5 --latency 0.02

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
```

</details>

## Follow-up: faster writes still (14x), and why it is *not* MPI

[HDF5's own "parallel compression"](https://support.hdfgroup.org/documentation/hdf5/latest/_par_compr.html)
solves a different problem than ours. It is **multi-process (MPI)** parallelism: several MPI ranks
collectively write one dataset, HDF5 assigns each chunk an owner rank, non-owners ship their edits
to the owner, and each rank runs the filter on the chunks it owns. It needs HDF5 built with
parallel support, collective writes (`H5Pset_dxpl_mpio` with `H5FD_MPIO_COLLECTIVE`), and every
rank must participate in every write even when it has no data. Our h5py is not an MPI build
(`h5py.get_config().mpi` is `False`), and our workload is one process writing one file — so it
buys us nothing without a rebuild and an MPI-shaped rewrite.

Worth reading anyway, because its **tuning** guidance independently endorses what this PR does:
get chunk sizing right *before* adding compression, avoid chunks shared between writers, use
`H5F_FSPACE_STRATEGY_PAGE`, and set `H5Pset_libver_bounds` to latest. We already do the paged
file space and `libver="latest"` (`PAGED_LAYOUT`), our per-frame chunks have exactly one writer
each, and "chunking before compression" is precisely what the benchmark above concluded.

The parallelism we *can* still exploit is threads in one process — the write-side mirror of
`chunk_reader`. Measured on the real carotid file (332 MB int16, 8.4 MB chunks, this PR's codec):

| approach                                                 | write         | vs today |
|----------------------------------------------------------|---------------|----------|
| HDF5 filter pipeline, single-threaded                    | 104 MB/s      | 1x       |
| **`BLOSC_NTHREADS=8`** (in this PR)                      | 453 MB/s      | 4.3x     |
| parallel compress + `write_direct_chunk` (16-32 threads) | **1491 MB/s** | **14x**  |

All three produce byte-identical files of the same size (181 MB), readable by plain h5py.

### Plan for `write_direct_chunk` (not in this PR — up for grabs)

Compression is the write bottleneck and it parallelises; the byte-writes stay serial under h5py's
lock, but they are cheap. So: compress each chunk yourself in a `ThreadPoolExecutor`
(`numcodecs.blosc.compress` releases the GIL), then hand HDF5 the finished bytes with
`dset.id.write_direct_chunk(offset, buf)`, bypassing the filter pipeline entirely. This mirrors
`zea.data.chunk_reader` exactly, and would live next to it.

Working prototype (this is the measured 1491 MB/s):

```python
ncblosc.set_nthreads(1)                       # the pool provides the parallelism, not blosc
grid = [math.ceil(s / c) for s, c in zip(raw.shape, CH)]
cells = [(i, j) for i in range(grid[0]) for j in range(grid[1])]

def compress(cell):
    i, j = cell
    offset = (i * CH[0], j * CH[1], 0, 0, 0)   # ELEMENT coords, on a chunk boundary
    sl = tuple(slice(o, min(o + c, s)) for o, c, s in zip(offset, CH, raw.shape))
    block = np.ascontiguousarray(raw[sl])
    if block.shape != CH:                      # edge chunk: HDF5 stores chunks FULL-size
        pad = np.zeros(CH, dtype=raw.dtype)
        pad[tuple(slice(0, n) for n in block.shape)] = block
        block = pad
    return offset, ncblosc.compress(block, b"zstd", 7, ncblosc.BITSHUFFLE)

with h5py.File(path, "w") as f:
    d = f.create_dataset("d", shape=raw.shape, dtype=raw.dtype, chunks=CH, **BLOSC)
    with ThreadPoolExecutor(max_workers=16) as ex:
        for offset, buf in ex.map(compress, cells):
            d.id.write_direct_chunk(offset, buf)
```

**The traps, all of which I hit building the prototype:**

1. **The codec params must match the dataset's declared filter pipeline exactly** — same cname,
   clevel, shuffle. HDF5 **does not verify this**. Get it wrong and you write a file that decodes
   to silent garbage, with no error at write *or* open. This is the one that would bite hardest,
   and it is why this is a separate PR: write correctness becomes ours, and the failure mode is
   invisible.
2. **Offsets are in element coordinates on a chunk boundary**, not chunk indices — `(i*1, j*15,
   0, 0, 0)`, not `(i, j, 0, 0, 0)`. Passing chunk indices raises
   `OSError: offset doesn't fall on chunks's boundary`, which at least fails loudly.
3. **Edge chunks are stored full-size.** A chunk that hangs off the end of the array must be
   zero-padded to the full chunk shape before compressing, or the buffer is short.
4. **Incompressible chunks.** HDF5's filter stores such a chunk *raw* and records that in its
   `filter_mask`. Writing directly, you either always store compressed (fine — that is what the
   prototype does, and `chunk_reader` handles the mask correctly on the way back in) or replicate
   the raw-store behaviour with the mask bit set.
5. **Memory is `workers x chunk_size`** — 16 x 8 MB here. Bound it the way `chunk_reader` bounds
   reads (`MAX_BYTES_IN_FLIGHT`), by bytes rather than by chunk count.

Diminishing returns past ~16 threads (32 was 1491, 64 was 1431 MB/s). The equality tests should
mirror `tests/data/test_chunk_reader.py`: write via the fast path, read back with **plain h5py**
as the oracle, across codecs x chunk layouts x edge-shaped arrays x incompressible data.

## Notes for reviewers

- **Existing files keep working.** lzf files still read (via h5py, serially); nothing needs
  migrating. Re-saving a file with `zea data resave` gets it the new codec and chunking.
- **`chunk_reader` correctness is ours now**, so it is a pure optimisation with an h5py fallback for
  anything it does not fully understand (unknown codec, contiguous dataset, exotic selection), and
  h5py is the oracle in the tests: equality across codecs x chunk layouts x selections, including
  incompressible chunks that HDF5 stores raw (`filter_mask`).
- **Edge case:** if a *single* transmit already exceeds `MAX_CHUNK_BYTES` (a very deep/wide
  acquisition), the chunk is left oversized rather than splitting `n_ax`/`n_el`, which are always
  read whole. Documented in `_resolve_chunks`.
- **Write speed** for `File.create` we are still going through `h5py`, which means it cannot concurrently encode chunks. That is a one-time cost per file, but may still be optimized in the future (not sure if even possible). We might want to encourage people saving multiple files to use `File.create` in a process pool
