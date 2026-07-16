"""Tests for :mod:`zea.data.chunk_reader`.

The fast path is a pure optimisation, so **h5py is the oracle**: whatever
``dataset[selection]`` returns, the concurrent reader must return exactly that — values,
dtype, shape and exceptions. Most tests here are that comparison, across the codecs, chunk
layouts and selections where it could plausibly diverge. The cases that earn their keep:

* **filter-masked chunks** — incompressible data HDF5 stored *raw* (the ``int16`` noise set).
* **fallbacks** — lzf, contiguous datasets and strided slices must fall through to h5py.
* **concurrency** — N chunks must cost ~1 round trip, counted against a latency-injecting
  server rather than trusted from the wall clock.
"""

import http.server
import logging
import os
import socketserver
import threading
import time
from unittest.mock import MagicMock

import hdf5plugin
import numpy as np
import pytest

from zea.data.chunk_reader import (
    MIN_BYTES,
    HTTPFetcher,
    LocalFetcher,
    _display_path,
    _human_bytes,
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

        A decoder that mishandles a block boundary returns the *right number of bytes* — so
        nothing raises, it just returns the wrong ones, and only once a chunk outgrows one
        block. A small-chunk test would pass while every real file (chunks run to megabytes)
        decoded to garbage.
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

        HDF5 stores a chunk raw when the filter fails to shrink it. If this data ever became
        compressible, the test covering the mask handling would silently cover nothing.
        """
        path = _write(tmp_path / "noise.hdf5", _incompressible())
        with File(path) as file:
            dsid = file[RAW].id
            masks = [dsid.get_chunk_info(i).filter_mask for i in range(dsid.get_num_chunks())]
        assert any(masks), "expected HDF5 to store incompressible chunks unfiltered"


class TestProgress:
    """Progress is reported per chunk, and counts the bytes that actually move."""

    def test_ticks_once_per_chunk_with_the_compressed_size(self, structured_file):
        """The bar's total comes from the chunk manifest, so it must match the bytes fetched.

        Ticking the *uncompressed* size would be the easy mistake: the bar would then race
        ahead of the download and finish early.
        """
        ticks = []
        with File(structured_file) as file:
            dset = file[RAW]
            got = read(dset, slice(0, 3), file._chunk_fetcher, progress=ticks.append)
            stored = [dset.id.get_chunk_info(i).size for i in range(3)]

            np.testing.assert_array_equal(got, dset[0:3])
            assert len(ticks) == 3, "expected one tick per chunk"
            assert sum(ticks) == sum(stored), "ticks must total the compressed bytes"

    def test_progress_does_not_change_the_data(self, structured_file):
        with File(structured_file) as file:
            dset, fetcher = file[RAW], file._chunk_fetcher
            with_bar = read(dset, slice(0, 3), fetcher, progress=True)
            without = read(dset, slice(0, 3), fetcher, progress=False)
            np.testing.assert_array_equal(with_bar, without)
            np.testing.assert_array_equal(with_bar, dset[0:3])

    def test_file_level_flag_reaches_the_read(self, structured_file):
        """``File(progress=...)`` and ``file.progress = ...`` both drive ChunkedDataset.

        Reads 3 frames, not 2: below ``MIN_BYTES`` the read is served by h5py and reports
        nothing at all — correctly, but it would make this test pass for the wrong reason.
        """
        ticks = []
        with File(structured_file, progress=ticks.append) as file:
            assert file.progress is not None
            file.data.raw_data[0:3]
        assert ticks, "the file-level flag never reached the reader"

        later = []
        with File(structured_file) as file:
            file.progress = later.append  # set after opening
            file.data.raw_data[0:3]
        assert later

    def test_fallback_reads_report_nothing(self, tmp_path):
        """lzf has no fast path, so there is no progress to report — and no crash either.

        h5py fetches the whole selection in one opaque call, so a bar could only jump 0->100%.
        Asking for one must still return the right data.
        """
        ticks = []
        path = _write(tmp_path / "lzf.hdf5", _structured(), compression="lzf")
        with File(path) as file:
            dset = file[RAW]
            got = read(dset, slice(0, 3), file._chunk_fetcher, progress=ticks.append)
            np.testing.assert_array_equal(got, dset[0:3])
        assert ticks == []


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


class TestFallbackNotes:
    """A read that misses the concurrent path nudges once, naming the cause and the fix.

    The serial fallback is slower on disk as well as over HTTP, so the note fires regardless
    of ``progress`` — but it must never fire on a read the fast path actually served, and never
    twice for the same dataset.
    """

    @pytest.fixture
    def notes(self, caplog):
        """Capture ``zea.log`` (``propagate=False``) and isolate the once-only dedupe set."""
        from zea import log

        caplog.set_level(logging.WARNING)
        log.logger.addHandler(caplog.handler)
        saved = set(log._warned_locations)
        log._warned_locations.clear()
        try:
            yield caplog
        finally:
            log.logger.removeHandler(caplog.handler)
            log._warned_locations.clear()
            log._warned_locations.update(saved)

    def test_lzf_nudges_to_resave_even_without_progress(self, tmp_path, notes):
        path = _write(tmp_path / "lzf.hdf5", _structured(), compression="lzf")
        with File(path) as file:
            read(file[RAW], slice(None), file._chunk_fetcher, progress=False)
        assert len(notes.records) == 1
        assert "lzf" in notes.text and "Re-save" in notes.text

    def test_contiguous_nudges_to_resave(self, tmp_path, notes):
        path = _write(tmp_path / "flat.hdf5", _structured(), compression=None, chunk_axes=None)
        with File(path) as file:
            read(file[RAW], slice(None), file._chunk_fetcher, progress=False)
        assert len(notes.records) == 1
        assert "without chunking" in notes.text and "Re-save" in notes.text

    def test_strided_selection_nudges_about_indexing_not_resave(self, tmp_path, notes):
        path = _write(tmp_path / "blosc.hdf5", _structured())  # fast-path-eligible dataset
        with File(path) as file:
            read(file[RAW], slice(None, None, 2), file._chunk_fetcher, progress=False)
        assert len(notes.records) == 1
        assert "selection" in notes.text and "Re-save" not in notes.text

    def test_fast_path_read_is_quiet(self, tmp_path, notes):
        path = _write(tmp_path / "blosc.hdf5", _structured())
        with File(path) as file:
            read(file[RAW], slice(0, 4), file._chunk_fetcher, progress=False)
        assert notes.records == []

    def test_note_fires_only_once_per_dataset(self, tmp_path, notes):
        path = _write(tmp_path / "lzf.hdf5", _structured(), compression="lzf")
        with File(path) as file:
            for _ in range(3):
                read(file[RAW], slice(None), file._chunk_fetcher, progress=False)
        assert len(notes.records) == 1

    def test_lzf_over_http_still_nudges_when_progress_is_requested(self, tmp_path, notes):
        """lzf has no fast path even when streamed, so asking for progress gets none of it —
        the read still must not go quiet about why."""
        path = _write(tmp_path / "lzf.hdf5", _structured(), compression="lzf")
        server = _CountingServer(tmp_path)
        try:
            fetcher = HTTPFetcher(server.url + path.name)
            with File(path) as file:
                got = read(file[RAW], slice(None), fetcher, progress=True)
                np.testing.assert_array_equal(got, file[RAW][:])
        finally:
            server.close()
        assert len(notes.records) == 1
        assert "lzf" in notes.text and "Re-save" in notes.text

    def test_message_names_the_file_not_the_dataset_path(self, tmp_path, notes):
        path = _write(tmp_path / "lzf.hdf5", _structured(), compression="lzf")
        with File(path) as file:
            read(file[RAW], slice(None), file._chunk_fetcher, progress=False)
        assert str(path) in notes.text
        assert RAW not in notes.text

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


class TestHumanBytes:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (0, "0 B"),
            (1023, "1023 B"),
            (1024, "1.0 KB"),
            (1536, "1.5 KB"),
            (1024**2, "1.0 MB"),
            (1024**3, "1.0 GB"),
            (1024**4, "1.0 TB"),
            (1024**5, "1024.0 TB"),
        ],
    )
    def test_units(self, n, expected):
        assert _human_bytes(n) == expected


class TestDisplayPath:
    def test_strips_track_prefix_for_single_track_file(self, structured_file):
        with File(structured_file) as file:
            assert _display_path(file[RAW]) == "/data/raw_data"

    def test_keeps_track_prefix_for_multi_track_file(self, tmp_path):
        path = tmp_path / "multi_track.hdf5"

        def _track(label):
            return {"data": {"raw_data": _structured()}, "scan": _scan(), "label": label}

        File.create(
            path,
            tracks=[_track("a"), _track("b")],
            probe={"name": "generic", "probe_geometry": np.zeros((N_EL, 3), np.float32)},
            ignore_warnings=True,
        )
        with File(path) as file:
            dset = file["tracks/track_0/data/raw_data"]
            assert _display_path(dset) == "/tracks/track_0/data/raw_data"


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

    def test_pread_path_is_lock_free(self, structured_file):
        """Where os.pread exists (Unix), its positional read is thread-safe, so no lock."""
        if not hasattr(os, "pread"):
            pytest.skip("platform has no os.pread")
        fetcher = LocalFetcher(structured_file)
        try:
            assert fetcher._lock is None
        finally:
            fetcher.close()

    def test_windows_path_guards_with_a_lock(self, structured_file, monkeypatch):
        """Without os.pread (Windows), seek+read shares the fd position and must be locked."""
        monkeypatch.delattr(os, "pread", raising=False)
        fetcher = LocalFetcher(structured_file)
        try:
            assert fetcher._lock is not None
        finally:
            fetcher.close()

    @pytest.mark.skipif(not hasattr(os, "O_BINARY"), reason="Windows-only file mode")
    def test_windows_opens_local_chunks_in_binary_mode(self, structured_file, monkeypatch):
        """Compressed chunk bytes must not undergo Windows text-mode translation."""
        opened = {}
        monkeypatch.setattr(os, "open", lambda path, flags: opened.setdefault("flags", flags) or 1)
        monkeypatch.setattr(os, "close", lambda fd: None)

        fetcher = LocalFetcher(structured_file)
        try:
            assert opened["flags"] & os.O_BINARY
        finally:
            fetcher.close()

    def test_windows_fallback_matches_h5py(self, structured_file, monkeypatch):
        """The lock-guarded lseek+read fallback must return exactly what h5py returns, even
        under the concurrent decode pool (a whole-file read fans out across every chunk)."""
        monkeypatch.delattr(os, "pread", raising=False)
        with File(structured_file) as file:
            dset = file[RAW]
            fetcher = LocalFetcher(structured_file)
            assert fetcher._lock is not None  # confirm the Windows branch is the one exercised
            try:
                got = read(dset, slice(None), fetcher, progress=False)
            finally:
                fetcher.close()
            np.testing.assert_array_equal(got, dset[:])


# --------------------------------------------------------------------------- #
# Remote: the win is round trips, so count them rather than timing them.
# --------------------------------------------------------------------------- #
class _CountingServer:
    """Range-capable HTTP server that counts requests and delays each one."""

    def __init__(self, directory, latency=0.02):
        self.count = 0
        #: Highest number of requests in flight at once — the real evidence of concurrency.
        self.peak = 0
        self._active = 0
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
                    outer._active += 1
                    outer.peak = max(outer.peak, outer._active)
                try:
                    time.sleep(latency)  # hold the request open so overlaps are observable
                finally:
                    with lock:
                        outer._active -= 1
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

        Asserted on *peak concurrency* — the maximum number of requests the server held open
        at once — which distinguishes concurrent range requests from serial ones directly,
        instead of the fragile wall-clock threshold. Elapsed time is kept as benchmark data.
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
                elapsed = time.perf_counter() - start  # benchmark only, not asserted
                requests = server.count - before

                np.testing.assert_array_equal(got, file[RAW][0:n_chunks])
                assert requests >= n_chunks, "expected one range request per chunk"
                assert server.peak > 1, (
                    f"requests were served serially (peak concurrency {server.peak}); "
                    f"expected overlapping range requests. Elapsed {elapsed:.3f}s."
                )
        finally:
            server.close()

    def test_fully_cached_read_draws_no_progress_bar(self, structured_file, tmp_path, monkeypatch):
        """A read served entirely from the on-disk chunk cache neither streams nor downloads,
        so ``progress=True`` must not draw a bar for it (it still would for a partial hit,
        which is fine — real bytes are moving there)."""
        from zea.data.chunk_cache import ChunkCache

        server = _CountingServer(structured_file.parent, latency=0)
        try:
            cache = ChunkCache("test-content-id", root=tmp_path / "cache")
            fetcher = HTTPFetcher(server.url + structured_file.name, cache=cache)
            with File(structured_file) as file:
                dset = file[RAW]
                read(dset, slice(None), fetcher, progress=False)  # warms the cache

                tqdm_mock = MagicMock()
                monkeypatch.setattr("tqdm.auto.tqdm", tqdm_mock)
                got = read(dset, slice(None), fetcher, progress=True)
                np.testing.assert_array_equal(got, dset[:])
            tqdm_mock.assert_not_called()
        finally:
            server.close()
