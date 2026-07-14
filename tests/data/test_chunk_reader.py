"""Tests for :mod:`zea.data.chunk_reader`.

The fast path is a pure optimisation, so **h5py is the oracle**: whatever
``dataset[selection]`` returns, the concurrent reader must return exactly that — same
values, dtype, shape, and the same exceptions. Nearly every test here is that comparison,
run across the codecs, chunk layouts and selection kinds where it could plausibly diverge.

The cases that earn their keep:

* **filter-masked chunks** — incompressible data that HDF5 stored *raw*, which the Zarr
  path decoded to garbage. Exercised by the ``int16`` noise dataset.
* **fallbacks** — lzf, contiguous datasets and strided slices have no fast path and must
  quietly fall through to h5py.
* **concurrency** — the remote win is real only if N chunks cost ~1 round trip, so we
  count requests against a latency-injecting server rather than trusting the wall clock.
"""

import http.server
import socketserver
import threading
import time

import hdf5plugin
import numpy as np
import pytest

from zea.data.chunk_reader import (
    MIN_BYTES,
    HTTPFetcher,
    LocalFetcher,
    eligible,
    fetcher_for,
    read,
)
from zea.data.file import ChunkedDataset, File

RAW = "tracks/track_0/data/raw_data"

BLOSC = dict(hdf5plugin.Blosc(cname="zstd", clevel=7, shuffle=hdf5plugin.Blosc.BITSHUFFLE))
BLOSC_BYTESHUF = dict(hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
ZSTD = dict(hdf5plugin.Zstd(clevel=3))
LZ4 = dict(hdf5plugin.LZ4())
GZIP = {"compression": "gzip", "compression_opts": 4}
GZIP_SHUFFLE = {"compression": "gzip", "compression_opts": 4, "shuffle": True}

# Big enough that reads clear MIN_BYTES and actually take the fast path.
N_FRAMES, N_TX, N_AX, N_EL = 6, 4, 900, 32


def _scan(n_tx=N_TX, n_el=N_EL):
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
    }


def _structured():
    """Compressible, RF-like: every codec shrinks it, so chunks are stored filtered."""
    depth = np.linspace(0, 1, N_AX)[None, None, :, None, None]
    base = np.exp(-2.5 * depth) * np.sin(2 * np.pi * 40 * depth)
    return (base * np.ones((N_FRAMES, N_TX, N_AX, N_EL, 1), np.float32)).astype(np.float32)


def _incompressible():
    """Full-range noise: HDF5 gives up and stores chunks *raw*, setting their filter mask.

    The range matters. Noise confined to a few thousand counts leaves the high byte of each
    int16 nearly constant, and Blosc's shuffle finds exactly that — it compresses, no chunk
    is stored raw, and the filter-mask path silently stops being exercised. Spanning the
    full dtype defeats it, which ``test_incompressible_chunks_are_stored_raw`` enforces.
    """
    rng = np.random.default_rng(0)
    return rng.integers(-32768, 32767, (N_FRAMES, N_TX, N_AX, N_EL, 1)).astype(np.int16)


def _write(path, raw, compression=BLOSC, chunk_axes=("n_frames",)):
    File.create(
        path,
        data={"raw_data": raw},
        scan=_scan(),
        probe={"name": "generic", "probe_geometry": np.zeros((N_EL, 3), np.float32)},
        compression=compression,
        chunk_axes=chunk_axes,
        overwrite=True,
        ignore_warnings=True,
    )
    return path


SELECTIONS = [
    0,
    3,
    -1,
    slice(None),
    slice(0, 4),
    slice(2, 5),
    slice(1, 2),
    (0, 0),
    (slice(0, 3), slice(1, 3)),
    ([0, 2, 5],),
    (Ellipsis,),
    (0, Ellipsis),
    (slice(None), 0),
    (2, 3, slice(10, 20)),
    (slice(0, 3), slice(None), slice(5, 9)),
    (slice(None, None, 2),),  # strided: falls back to h5py
]


@pytest.fixture(scope="module")
def structured_file(tmp_path_factory):
    return _write(tmp_path_factory.mktemp("cr") / "structured.hdf5", _structured())


class TestEqualityWithH5py:
    """h5py is the oracle: the fast path must not change a single value."""

    @pytest.mark.parametrize("selection", SELECTIONS)
    def test_selection_matches_h5py(self, structured_file, selection):
        with File(structured_file) as file:
            oracle = file[RAW]  # the bare h5py.Dataset
            fast = file.data.raw_data  # ChunkedDataset
            assert isinstance(fast, ChunkedDataset)

            want, got = oracle[selection], fast[selection]
            np.testing.assert_array_equal(got, want)
            assert got.dtype == want.dtype
            assert got.shape == want.shape

    @pytest.mark.parametrize(
        "codec",
        [BLOSC, BLOSC_BYTESHUF, ZSTD, LZ4, GZIP, GZIP_SHUFFLE, "lzf", None],
        ids=[
            "blosc",
            "blosc+byteshuffle",
            "zstd",
            "lz4",
            "gzip",
            "gzip+shuffle",
            "lzf",
            "none",
        ],
    )
    @pytest.mark.parametrize(
        "chunk_axes",
        [("n_frames",), ("n_frames", "n_tx"), None],
        ids=["per-frame", "per-tx", "contiguous"],
    )
    @pytest.mark.parametrize(
        "data", [_structured, _incompressible], ids=["structured", "incompressible"]
    )
    def test_codecs_and_layouts(self, tmp_path, codec, chunk_axes, data):
        """Across every codec x layout x data combination, including the ones that fall back."""
        path = _write(tmp_path / "m.hdf5", data(), compression=codec, chunk_axes=chunk_axes)
        with File(path) as file:
            oracle, fast = file[RAW], file.data.raw_data
            for selection in SELECTIONS:
                np.testing.assert_array_equal(fast[selection], oracle[selection])

    @pytest.mark.parametrize("codec", [BLOSC, ZSTD, LZ4], ids=["blosc", "zstd", "lz4"])
    def test_chunk_spanning_many_codec_blocks(self, tmp_path, codec):
        """A chunk far larger than the codec's internal block size must still decode exactly.

        Codecs are block-structured underneath, and a decoder that mishandles that boundary
        fails in a nasty direction: it returns the *right number of bytes*, so nothing raises
        — it just returns the wrong ones, and only once a chunk outgrows a single block. A
        small-chunk test would pass while every real file (chunks are capped at
        ``MAX_CHUNK_BYTES``, i.e. megabytes) silently decoded to garbage. This pins the
        multi-block case explicitly, since that is the only size that actually ships.
        """
        raw = _structured()
        path = _write(tmp_path / "big.hdf5", raw, compression=codec, chunk_axes=("n_frames",))
        with File(path) as file:
            chunk_bytes = np.prod(file[RAW].chunks) * raw.dtype.itemsize
            assert chunk_bytes > (256 << 10), "chunk must span several codec blocks"
            assert eligible(file[RAW], fetcher_for(file)), "must be on the fast path to test it"
            np.testing.assert_array_equal(file.data.raw_data[:], raw)

    def test_incompressible_chunks_are_stored_raw(self, tmp_path):
        """Guards the premise of the test above: this data really does trip filter masks.

        HDF5 stores a chunk raw when the filter fails to shrink it, and records that in the
        chunk's filter mask. This is the case Zarr could not express — it decoded such
        chunks to garbage — so if this data ever became compressible, the test that covers
        the mask handling would silently stop covering anything.
        """
        path = _write(tmp_path / "noise.hdf5", _incompressible())
        with File(path) as file:
            dsid = file[RAW].id
            masks = [dsid.get_chunk_info(i).filter_mask for i in range(dsid.get_num_chunks())]
        assert any(masks), "expected HDF5 to store incompressible chunks unfiltered"


class TestFallback:
    """Anything the fast path does not fully understand must go to h5py, not guess."""

    def test_lzf_is_not_eligible(self, tmp_path):
        path = _write(tmp_path / "lzf.hdf5", _structured(), compression="lzf")
        with File(path) as file:
            assert not eligible(file[RAW], file._chunk_fetcher)

    def test_contiguous_is_not_eligible(self, tmp_path):
        path = _write(tmp_path / "flat.hdf5", _structured(), compression=None, chunk_axes=None)
        with File(path) as file:
            assert file[RAW].chunks is None
            assert not eligible(file[RAW], file._chunk_fetcher)

    def test_no_fetcher_still_reads(self, structured_file):
        """Without a fetcher there is no fast path at all — but reads must still work."""
        with File(structured_file) as file:
            np.testing.assert_array_equal(read(file[RAW], slice(0, 2), None), file[RAW][0:2])

    def test_multi_axis_fancy_index_raises_like_h5py(self, structured_file):
        """h5py allows one fancy axis and raises otherwise; we must raise identically."""
        with File(structured_file) as file:
            with pytest.raises(TypeError):
                file[RAW][[0, 1], [0, 1]]
            with pytest.raises(TypeError):
                file.data.raw_data[[0, 1], [0, 1]]

    def test_small_read_below_threshold(self, structured_file):
        """Under MIN_BYTES h5py serves the read; the values are the same either way."""
        with File(structured_file) as file:
            selection = (0, 0, slice(0, 4))
            got = file.data.raw_data[selection]
            assert got.nbytes < MIN_BYTES
            np.testing.assert_array_equal(got, file[RAW][selection])


@pytest.fixture
def force_fast_path(monkeypatch):
    """Drop MIN_BYTES so even small reads go through the reader instead of falling back.

    Without this, the small arrays these tests use would quietly be served by h5py — and a
    test that never enters the fast path cannot catch a bug in it.
    """
    monkeypatch.setattr("zea.data.chunk_reader.MIN_BYTES", 0)


class TestChunkMapping:
    """Where a chunk lands in the output — the part we own, and the part that had a bug."""

    @pytest.mark.parametrize(
        "selection",
        [
            ([0, 2, 5], 1),  # fancy axis + int axis: the regression below
            (slice(1, 4), 0),
            (0, [0, 2]),
            ([1, 3], slice(0, 2)),
            (slice(2, 5), 2, slice(3, 9)),
        ],
    )
    def test_int_axis_collapses_out_of_the_output(
        self, structured_file, force_fast_path, selection
    ):
        """An int axis is dropped from the output, but not from the chunk it is read out of.

        The decoded chunk keeps an axis of length 1 wherever the selection used an int, so
        it has more dimensions than the region it must be written into. Assigning it
        straight across only happens to work while the extra axes are *leading* ones — put
        an int after a slice or a fancy index and it breaks. Fuzzing found it; this pins it.
        """
        with File(structured_file) as file:
            got, want = file.data.raw_data[selection], file[RAW][selection]
            np.testing.assert_array_equal(got, want)
            assert got.shape == want.shape

    @pytest.mark.parametrize("chunk_axes", [("n_frames",), ("n_frames", "n_tx")])
    def test_fuzz_against_h5py(self, tmp_path, force_fast_path, chunk_axes):
        """Random selections, h5py as the oracle. Seeded, so a failure is reproducible."""
        import random

        path = _write(tmp_path / "fuzz.hdf5", _structured(), chunk_axes=chunk_axes)
        shape = (N_FRAMES, N_TX, N_AX, N_EL, 1)
        rng = random.Random(0)

        def random_selection():
            selection = []
            for size in shape[: rng.randint(1, len(shape))]:
                roll = rng.random()
                if roll < 0.3:
                    selection.append(rng.randrange(-size, size))
                elif roll < 0.75:
                    start = rng.randrange(0, size)
                    selection.append(slice(start, rng.randrange(start + 1, size + 1)))
                else:
                    count = rng.randrange(1, min(size, 4) + 1)
                    selection.append(sorted(rng.sample(range(size), count)))
            # More than one fancy axis is h5py's error to raise, not ours to compare.
            if sum(isinstance(part, list) for part in selection) > 1:
                return None
            return tuple(selection)

        with File(path) as file:
            oracle, fast = file[RAW], file.data.raw_data
            checked = 0
            for _ in range(200):
                selection = random_selection()
                if selection is None:
                    continue
                want, got = oracle[selection], fast[selection]
                np.testing.assert_array_equal(got, want, err_msg=f"selection={selection}")
                assert got.shape == want.shape
                checked += 1
            assert checked > 100, "the fuzz should actually be exercising the reader"


class TestFetchers:
    def test_local_fetcher_returns_chunk_bytes(self, structured_file):
        with File(structured_file) as file:
            dsid = file[RAW].id
            info = dsid.get_chunk_info(0)
            fetcher = LocalFetcher(structured_file)
            try:
                (raw,) = fetcher.fetch([(int(info.byte_offset), int(info.size))])
            finally:
                fetcher.close()
            # read_direct_chunk is the (slower) reference for the very same bytes.
            assert raw == dsid.read_direct_chunk((0, 0, 0, 0, 0))[1]

    def test_local_file_gets_a_local_fetcher(self, structured_file):
        with File(structured_file) as file:
            assert isinstance(fetcher_for(file), LocalFetcher)

    def test_fetcher_closed_with_file(self, structured_file):
        file = File(structured_file)
        assert file._chunk_fetcher is not None
        file.close()
        assert file._fetcher is None


# --------------------------------------------------------------------------- #
# Remote: the win is round trips, so count them rather than timing them.
# --------------------------------------------------------------------------- #
class _CountingServer:
    """Range-capable HTTP server that counts requests and delays each one."""

    def __init__(self, directory, latency=0.02):
        self.count = 0
        lock = threading.Lock()
        outer = self

        class Handler(http.server.SimpleHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(directory), **kwargs)

            def log_message(self, *args):
                pass

            def send_head(self):
                with lock:
                    outer.count += 1
                time.sleep(latency)
                path = self.translate_path(self.path)
                with open(path, "rb") as handle:
                    body = handle.read()
                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes="):
                    start_s, _, end_s = rng[6:].partition("-")
                    start = int(start_s) if start_s else 0
                    end = int(end_s) if end_s else len(body) - 1
                    part = body[start : end + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
                    self.send_header("Content-Length", str(len(part)))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    return _Body(part)
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                return _Body(body)

        self._httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.latency = latency
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}/"
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def close(self):
        self._httpd.shutdown()


class _Body:
    """Minimal file-like wrapper so SimpleHTTPRequestHandler can copy our bytes out."""

    def __init__(self, data):
        self._data = data

    def read(self, *args):
        data, self._data = self._data, b""
        return data

    def close(self):
        pass


class TestRemote:
    def test_http_fetcher_matches_local_bytes(self, structured_file):
        server = _CountingServer(structured_file.parent, latency=0)
        try:
            fetcher = HTTPFetcher(server.url + structured_file.name)
            with File(structured_file) as file:
                info = file[RAW].id.get_chunk_info(0)
                ranges = [(int(info.byte_offset), int(info.size))]
                remote = fetcher.fetch(ranges)
                local = LocalFetcher(structured_file)
                try:
                    assert remote == local.fetch(ranges)
                finally:
                    local.close()
        finally:
            server.close()

    def test_reads_over_http_match_h5py(self, structured_file):
        """The whole path end to end: chunk offsets from h5py, bytes over HTTP, decode."""
        server = _CountingServer(structured_file.parent, latency=0)
        try:
            fetcher = HTTPFetcher(server.url + structured_file.name)
            with File(structured_file) as file:
                for selection in (slice(0, 4), 2, (slice(0, 2), slice(1, 3))):
                    got = read(file[RAW], selection, fetcher)
                    np.testing.assert_array_equal(got, file[RAW][selection])
        finally:
            server.close()

    def test_chunks_are_fetched_concurrently(self, structured_file):
        """The point of the remote path: N chunks cost ~1 round trip, not N.

        Timed against an injected per-request latency, because that is the only thing that
        distinguishes concurrent range requests from serial ones — the request *count* is
        the same either way.
        """
        latency = 0.05
        server = _CountingServer(structured_file.parent, latency=latency)
        try:
            fetcher = HTTPFetcher(server.url + structured_file.name)
            with File(structured_file) as file:
                n_chunks = 4
                before = server.count
                start = time.perf_counter()
                got = read(file[RAW], slice(0, n_chunks), fetcher)
                elapsed = time.perf_counter() - start
                requests = server.count - before

                np.testing.assert_array_equal(got, file[RAW][0:n_chunks])
                assert requests >= n_chunks, "expected one range request per chunk"
                # Serial would cost n_chunks * latency; concurrent costs about one.
                assert elapsed < n_chunks * latency, (
                    f"{n_chunks} chunks took {elapsed:.3f}s, "
                    f"which is serial ({n_chunks * latency:.3f}s at {latency}s/request)"
                )
        finally:
            server.close()
