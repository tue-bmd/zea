"""Phase-0 spike: is VirtualiZarr a win for zea's HF-hosted HDF5 files?

THROWAWAY validation script (not production). It answers the go/no-go questions
from the plan by exercising the whole path end to end:

  A. Feasibility  — can VirtualiZarr virtualize a *Blosc*-compressed zea file with
     per-(frame,tx) plane chunking, and read raw_data back bit-identically?
     (gzip is tried too, as the documented fallback codec.)
  B. Serialization — write kerchunk references and re-open them via the client
     path (KerchunkJSONParser) *without* re-parsing the HDF5. Records which
     formats need the `kerchunk` package.
  C. Cloud read   — over a local HTTP server with injected per-request latency,
     compare the current path (fsspec-http + h5py streaming) against the virtual
     path (obstore-http + Zarr manifest store) on **HTTP request count** and
     **cold partial-read latency**.

Run:  python scripts/spike_virtualizarr.py
Deps: pip install virtualizarr "zarr>=3" numcodecs fsspec aiohttp   (hdf5plugin already in zea)
      `kerchunk` is only needed for parquet references (generation-side).
"""

from __future__ import annotations

import http.server
import os
import socketserver
import tempfile
import threading
import time
import warnings

import h5py
import hdf5plugin
import numpy as np

warnings.filterwarnings("ignore")

import zea  # noqa: E402  (import after warnings filter)

DATA_GROUP = "tracks/track_0/data"
RAW = f"{DATA_GROUP}/raw_data"


# --------------------------------------------------------------------------- #
# Synthetic data + zea file writing
# --------------------------------------------------------------------------- #
def synth_rf(shape, seed=0):
    """Small structured RF-like array (compressible, not pure noise)."""
    n_frames, n_tx, n_ax, n_el, n_ch = shape
    rng = np.random.default_rng(seed)
    z = np.linspace(0, 1, n_ax)[:, None]
    base = np.exp(-2.5 * z) * np.sin(2 * np.pi * 40 * z)
    out = np.empty(shape, np.float32)
    for fi in range(n_frames):
        for ti in range(n_tx):
            sp = np.cumsum(rng.standard_normal((n_ax, n_el)), axis=0)
            sp -= sp.mean(0, keepdims=True)
            sp /= np.abs(sp).max() + 1e-9
            out[fi, ti, :, :, 0] = base * (1 + 0.7 * sp)
    return out


def dummy_scan(n_tx, n_el):
    return {
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
    }


def write_zea_file(path, data, compression, chunk_axes=("n_frames", "n_tx")):
    n_tx, n_el = data.shape[1], data.shape[3]
    geom = np.zeros((n_el, 3), np.float32)
    zea.File.create(
        path,
        data={"raw_data": data},
        scan=dummy_scan(n_tx, n_el),
        probe={"name": "generic", "probe_geometry": geom},
        compression=compression,
        chunk_axes=chunk_axes,
        overwrite=True,
        ignore_warnings=True,
    )


# --------------------------------------------------------------------------- #
# VirtualiZarr helpers
# --------------------------------------------------------------------------- #
def make_local_registry():
    from obstore.store import LocalStore
    from virtualizarr.registry import ObjectStoreRegistry

    return ObjectStoreRegistry({"file://": LocalStore()})


def virtual_store_from_hdf(path, registry):
    """Parse the HDF5 data group -> zarr-readable ManifestStore (native, no kerchunk)."""
    from virtualizarr.parsers import HDFParser

    parser = HDFParser(group=DATA_GROUP)
    return parser("file://" + os.path.abspath(path), registry)


def read_subset_zarr(store, fsel, tsel):
    """Read only the selected (frame, transmit) chunks via zarr orthogonal indexing."""
    import zarr

    arr = zarr.open_group(store, mode="r")["raw_data"]
    return arr.oindex[list(fsel), list(tsel)]  # orthogonal: outer product of the two axes


# --------------------------------------------------------------------------- #
# Counting / latency-injecting HTTP server (Range-capable)
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
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _file(self):
            return os.path.join(directory, self.path.lstrip("/"))

        def do_HEAD(self):
            counter.bump()
            if latency:
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
            if latency:
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
                length = end - start + 1
                with open(self._file(), "rb") as fh:  # seek: don't read whole file
                    fh.seek(start)
                    chunk = fh.read(length)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(len(chunk)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self.wfile.write(chunk)
            else:
                with open(self._file(), "rb") as fh:
                    body = fh.read()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self.wfile.write(body)

    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


# --------------------------------------------------------------------------- #
# Read paths over HTTP — each returns (open_reqs, open_t, read_reqs, read_t, array),
# separating cold-open (HDF5 metadata walk vs local refs) from the data read.
# --------------------------------------------------------------------------- #
def timed_h5py_http(url, counter, fsel, tsel):
    import fsspec

    n0, t0 = counter.value(), time.perf_counter()
    fs = fsspec.filesystem("http")
    fo = fs.open(url, block_size=4 * 1024 * 1024, cache_type="blockcache")
    f = h5py.File(fo, "r")
    ds = f[RAW]
    _ = ds.shape  # force metadata resolution
    open_reqs, open_t = counter.value() - n0, time.perf_counter() - t0

    n0, t0 = counter.value(), time.perf_counter()
    arr = np.stack([ds[int(fi), list(tsel)] for fi in fsel])  # only selected tx chunks
    read_reqs, read_t = counter.value() - n0, time.perf_counter() - t0
    f.close()
    fo.close()
    return open_reqs, open_t, read_reqs, read_t, arr


def timed_virtual_http(refs_path, http_base, counter, fsel, tsel):
    import zarr
    from obstore.store import HTTPStore, LocalStore
    from virtualizarr.parsers import KerchunkJSONParser
    from virtualizarr.registry import ObjectStoreRegistry

    registry = ObjectStoreRegistry(
        {
            "file://": LocalStore(),
            # allow_http: obstore/reqwest refuses plain HTTP (non-TLS) by default.
            http_base: HTTPStore.from_url(http_base, client_options={"allow_http": True}),
        }
    )
    n0, t0 = counter.value(), time.perf_counter()
    store = KerchunkJSONParser()("file://" + os.path.abspath(refs_path), registry)
    arr = zarr.open_group(store, mode="r")["raw_data"]
    _ = arr.shape  # metadata comes from local refs -> ~0 HTTP
    open_reqs, open_t = counter.value() - n0, time.perf_counter() - t0

    n0, t0 = counter.value(), time.perf_counter()
    got = arr.oindex[list(fsel), list(tsel)]  # only the selected (frame, tx) chunks
    read_reqs, read_t = counter.value() - n0, time.perf_counter() - t0
    return open_reqs, open_t, read_reqs, read_t, got


def http_registry(http_base):
    from obstore.store import HTTPStore, LocalStore
    from virtualizarr.registry import ObjectStoreRegistry

    return ObjectStoreRegistry(
        {
            "file://": LocalStore(),
            http_base: HTTPStore.from_url(http_base, client_options={"allow_http": True}),
        }
    )


def build_combined_refs(paths, http_urls, out_json):
    """Concat per-file virtual datasets along a new 'file' dim; refs point at http_urls."""
    import xarray as xr

    reg = make_local_registry()
    vdss = []
    for p, u in zip(paths, http_urls):
        store = virtual_store_from_hdf(p, reg)
        vdss.append(store.to_virtual_dataset().vz.rename_paths(lambda _p, u=u: u))
    xr.concat(vdss, dim="file").vz.to_kerchunk(out_json, format="json")
    return out_json


# --------------------------------------------------------------------------- #
# Main spike
# --------------------------------------------------------------------------- #
def main():
    outdir = tempfile.mkdtemp(prefix="zea_spike_")
    shape = (4, 12, 2048, 64, 1)
    data = synth_rf(shape)
    fsel, tsel = [0, 2], [1, 5, 9]  # partial read: 2 frames x 3 transmits
    ref = np.stack([data[fi][tsel] for fi in fsel])
    print(
        f"data {shape} = {data.nbytes / 1e6:.1f} MB; partial read {len(fsel)}x{len(tsel)} planes\n"
    )

    verdict = {}

    # ---- A. Feasibility (Blosc + gzip) ----
    print("== A. Feasibility: virtualize + read back locally ==")
    registry = make_local_registry()
    codecs = {
        "blosc-zstd-shuffle": dict(
            hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        ),
        "gzip-4": {"compression": "gzip", "compression_opts": 4},
    }
    blosc_path = None
    for label, comp in codecs.items():
        path = os.path.join(outdir, f"{label}.hdf5")
        write_zea_file(path, data, comp)
        with h5py.File(path) as f:
            chunks = f[RAW].chunks
            stored = f[RAW].id.get_storage_size()
        try:
            store = virtual_store_from_hdf(path, registry)
            got = read_subset_zarr(store, fsel, tsel)
            ok = np.array_equal(got, ref)
        except Exception as e:  # noqa: BLE001
            ok, got = False, None
            print(f"  {label:<20} VIRTUALIZE FAILED: {type(e).__name__}: {e}")
        if got is not None:
            ratio = data.nbytes / stored
            print(f"  {label:<20} chunks={chunks} ratio={ratio:5.2f}x  read-equal={ok}")
        verdict[f"virtualize:{label}"] = ok
        if label.startswith("blosc") and ok:
            blosc_path = path

    # ---- B. Serialization / client refs path ----
    print("\n== B. Serialize kerchunk refs + re-open via client path (no HDF parse) ==")
    chosen = blosc_path or os.path.join(outdir, "gzip-4.hdf5")
    store = virtual_store_from_hdf(chosen, registry)
    vds = store.to_virtual_dataset()
    refs_json = os.path.join(outdir, "refs.json")
    vds.vz.to_kerchunk(refs_json, format="json")
    print(f"  json refs written: {os.path.getsize(refs_json)} bytes (no kerchunk needed)")
    try:
        vds.vz.to_kerchunk(os.path.join(outdir, "refs.parquet"), format="parquet")
        print("  parquet refs: OK")
        verdict["parquet_needs_kerchunk"] = False
    except Exception as e:  # noqa: BLE001
        print(f"  parquet refs: needs `kerchunk` ({type(e).__name__}) -> generation-side only")
        verdict["parquet_needs_kerchunk"] = True
    from virtualizarr.parsers import KerchunkJSONParser

    store2 = KerchunkJSONParser()("file://" + os.path.abspath(refs_json), registry)
    ok_refs = np.array_equal(read_subset_zarr(store2, fsel, tsel), ref)
    print(f"  re-open via refs, read-equal={ok_refs}")
    verdict["client_refs_read"] = ok_refs

    # ---- C. Cloud read: separate COLD-OPEN (metadata walk) from data read ----
    # Bigger many-chunk file + injected latency: this is the regime where avoiding
    # the HDF5 metadata B-tree walk over HTTP should matter.
    print("\n== C. Over local HTTP (bigger 192-chunk file, injected 20 ms/request) ==")
    big_shape = (8, 24, 2048, 64, 1)  # 8*24 = 192 (frame, tx) chunks
    big = synth_rf(big_shape)
    big_path = os.path.join(outdir, "big.hdf5")
    write_zea_file(big_path, big, codecs["blosc-zstd-shuffle"])
    bfsel, btsel = [0, 3, 6], [2, 7, 11, 18]  # read 3 frames x 4 tx = 12 planes
    bref = big[np.ix_(bfsel, btsel)]

    latency = 0.020
    counter = _Counter()
    httpd, port = make_server(outdir, counter, latency)
    http_base = f"http://127.0.0.1:{port}/"
    url = http_base + "big.hdf5"
    try:
        # generate refs once (local parse) and rewrite chunk paths to the http url
        store_big = virtual_store_from_hdf(big_path, make_local_registry())
        vds_http = store_big.to_virtual_dataset().vz.rename_paths(lambda _p: url)
        refs_http = os.path.join(outdir, "refs_big_http.json")
        vds_http.vz.to_kerchunk(refs_http, format="json")

        # warm the obstore/tokio runtime so timings below reflect steady state
        from obstore.store import HTTPStore

        HTTPStore.from_url(http_base, client_options={"allow_http": True}).get_range(
            "big.hdf5", start=0, length=16
        )

        ho, hot, hr, hrt, got_h5 = timed_h5py_http(url, counter, bfsel, btsel)
        vo, vot, vr, vrt, got_v = timed_virtual_http(refs_http, http_base, counter, bfsel, btsel)

        print(f"  {'path':<22}{'open req':>9}{'open s':>8}{'read req':>9}{'read s':>8}{'equal':>7}")
        print(
            f"  {'h5py streaming':<22}{ho:>9}{hot:>8.2f}{hr:>9}{hrt:>8.2f}"
            f"{str(np.array_equal(got_h5, bref)):>7}"
        )
        print(
            f"  {'virtual (zarr refs)':<22}{vo:>9}{vot:>8.2f}{vr:>9}{vrt:>8.2f}"
            f"{str(np.array_equal(got_v, bref)):>7}"
        )
        verdict["h5py_open_req"], verdict["virtual_open_req"] = ho, vo
        verdict["h5py_read_req"], verdict["virtual_read_req"] = hr, vr
        verdict["h5py_total_s"] = hot + hrt
        verdict["virtual_total_s"] = vot + vrt
    finally:
        httpd.shutdown()

    # ---- D. Cross-file cold-open: open N files vs ONE combined reference ----
    print("\n== D. Cross-file cold-open: open N files (h5py) vs 1 combined ref (virtual) ==")
    import fsspec
    import zarr
    from virtualizarr.parsers import KerchunkJSONParser

    n_files = 8
    counter = _Counter()
    httpd, port = make_server(outdir, counter, latency)
    http_base = f"http://127.0.0.1:{port}/"
    try:
        paths, urls = [], []
        for i in range(n_files):
            p = os.path.join(outdir, f"cf{i}.hdf5")
            write_zea_file(p, synth_rf((2, 12, 1024, 64, 1), seed=i), codecs["blosc-zstd-shuffle"])
            paths.append(p)
            urls.append(http_base + f"cf{i}.hdf5")

        n0, t0 = counter.value(), time.perf_counter()
        fs = fsspec.filesystem("http")
        for u in urls:
            fo = fs.open(u, block_size=4 * 1024 * 1024, cache_type="blockcache")
            fh = h5py.File(fo, "r")
            _ = fh[RAW].shape
            fh.close()
            fo.close()
        xf_h5_req, xf_h5_t = counter.value() - n0, time.perf_counter() - t0

        refs = build_combined_refs(paths, urls, os.path.join(outdir, "combined.json"))
        n0, t0 = counter.value(), time.perf_counter()
        store = KerchunkJSONParser()("file://" + os.path.abspath(refs), http_registry(http_base))
        combined_shape = zarr.open_group(store, mode="r")["raw_data"].shape
        xf_v_req, xf_v_t = counter.value() - n0, time.perf_counter() - t0

        print(f"  {n_files} files -> combined virtual raw_data shape {combined_shape}")
        print(f"  {'h5py (open each file)':<26}{xf_h5_req:>6} req{xf_h5_t:>7.2f}s")
        print(f"  {'virtual (1 combined ref)':<26}{xf_v_req:>6} req{xf_v_t:>7.2f}s")
        verdict["xfile_h5py_req"], verdict["xfile_virtual_req"] = xf_h5_req, xf_v_req
    finally:
        httpd.shutdown()

    # ---- E. Data-read vs chunk layout (coalescing by design) ----
    print("\n== E. Data-read vs chunk layout: read 3 frames x 4 tx ==")
    efsel, etsel = [0, 3, 6], [2, 7, 11, 18]
    for layout, caxes in [("per-(frame,tx)", ("n_frames", "n_tx")), ("per-frame", ("n_frames",))]:
        big = synth_rf((8, 24, 2048, 64, 1))
        p = os.path.join(outdir, f"E_{layout}.hdf5")
        write_zea_file(p, big, codecs["blosc-zstd-shuffle"], chunk_axes=caxes)
        with h5py.File(p) as f:
            ch = f[RAW].chunks
        counter = _Counter()
        httpd, port = make_server(outdir, counter, latency)
        http_base = f"http://127.0.0.1:{port}/"
        url = http_base + os.path.basename(p)
        try:
            store_big = virtual_store_from_hdf(p, make_local_registry())
            vds_http = store_big.to_virtual_dataset().vz.rename_paths(lambda _p, u=url: u)
            refs = os.path.join(outdir, f"E_{layout}.json")
            vds_http.vz.to_kerchunk(refs, format="json")
            _, _, hr, hrt, gh = timed_h5py_http(url, counter, efsel, etsel)
            _, _, vr, vrt, gv = timed_virtual_http(refs, http_base, counter, efsel, etsel)
            print(
                f"  {layout:<15} chunks={ch}\n"
                f"      h5py: {hr:>3} req {hrt:5.2f}s   |   virtual: {vr:>3} req {vrt:5.2f}s"
                f"   equal={np.array_equal(gh, gv)}"
            )
        finally:
            httpd.shutdown()

    # ---- F. Concurrency probe: does obstore parallelize many range GETs? ----
    print("\n== F. Concurrency: obstore.get_ranges(8) under 20 ms/request latency ==")
    counter = _Counter()
    httpd, port = make_server(outdir, counter, latency)
    http_base = f"http://127.0.0.1:{port}/"
    try:
        from obstore.store import HTTPStore

        st = HTTPStore.from_url(http_base, client_options={"allow_http": True})
        fname = "E_per-frame.hdf5"
        starts = [i * 200_000 for i in range(8)]
        t0 = time.perf_counter()
        st.get_ranges(fname, starts=starts, ends=[s + 65536 for s in starts])
        dt = time.perf_counter() - t0
        print(
            f"  8 ranges in {dt:.2f}s  ("
            f"~{latency:.2f}s if concurrent, ~{8 * latency:.2f}s if serial)"
        )
        verdict["get_ranges_8_s"] = dt
    finally:
        httpd.shutdown()

    # ---- Summary ----
    print("\n== GO / NO-GO ==")
    go = verdict.get("virtualize:blosc-zstd-shuffle") and verdict.get("client_refs_read")
    print(f"  Blosc virtualization + client refs read: {'PASS' if go else 'FAIL'}")
    print(
        f"  Cold-open requests  h5py={verdict.get('h5py_open_req')}  "
        f"virtual={verdict.get('virtual_open_req')}   "
        f"(data-read req h5py={verdict.get('h5py_read_req')} "
        f"virtual={verdict.get('virtual_read_req')})"
    )
    print(
        f"  Single-file total  h5py={verdict.get('h5py_total_s'):.2f}s  "
        f"virtual={verdict.get('virtual_total_s'):.2f}s"
    )
    print(
        f"  Cross-file cold-open ({verdict.get('xfile_h5py_req', 0)} vs "
        f"{verdict.get('xfile_virtual_req', 0)} requests) -> virtual scales with #files"
    )
    print(f"  obstore.get_ranges(8) concurrency: {verdict.get('get_ranges_8_s', 0):.2f}s")
    print(f"  parquet refs need kerchunk (gen-side): {verdict.get('parquet_needs_kerchunk')}")
    print(f"\n  => {'GO with Blosc' if go else 'NO-GO / investigate'}  (files in {outdir})")


if __name__ == "__main__":
    main()
