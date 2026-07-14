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
