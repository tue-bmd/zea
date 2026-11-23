"""Tests for zea.data.hf_remote: resolving streamed hf:// files to their CDN URL once."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from zea.data import chunk_reader, hf_remote
from zea.data.file import File

from .test_chunk_reader import RAW, _CountingServer, _structured, _write

HF_PATH = "hf://org/repo/data/x.hdf5"


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_remote, "_CACHE_DIR", tmp_path / "locations")
    hf_remote.clear()
    yield
    hf_remote.clear()


@pytest.fixture
def hub(monkeypatch):
    """Fake ``get_hf_file_metadata``: records each HEAD and answers with ``hub.location``."""
    state = SimpleNamespace(
        heads=[], location="https://cdn.example/signed?1", etag="sha-1", size=1234
    )

    def fake_metadata(url, **kwargs):
        state.heads.append(url)
        return SimpleNamespace(
            location=state.location, etag=state.etag, commit_hash="c0ffee", size=state.size
        )

    monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", fake_metadata)
    return state


class TestResolve:
    def test_resolves_once(self, hub):
        first = hf_remote.resolve(HF_PATH)
        second = hf_remote.resolve(HF_PATH)
        assert first == second
        assert len(hub.heads) == 1
        assert first.size == 1234 and first.etag == "sha-1" and first.commit == "c0ffee"

    def test_shared_between_processes_through_disk(self, hub):
        """A second process (simulated by clearing memory) reuses the URL on disk."""
        hf_remote.resolve(HF_PATH)
        hf_remote._memory.clear()
        hf_remote.resolve(HF_PATH)
        assert len(hub.heads) == 1

    def test_disk_cache_is_private(self, hub):
        hf_remote.resolve(HF_PATH)
        (path,) = hf_remote._CACHE_DIR.iterdir()
        assert path.stat().st_mode & 0o077 == 0
        assert json.loads(path.read_text())["url"] == hub.location

    def test_expired_url_is_resolved_again(self, hub, monkeypatch):
        hf_remote.resolve(HF_PATH)
        monkeypatch.setattr(hf_remote, "URL_TTL", 0)
        hf_remote.resolve(HF_PATH)
        assert len(hub.heads) == 2

    def test_branch_reuse_is_capped_by_the_listing_ttl(self, hub, monkeypatch):
        """A branch can move after an upload, so its URL is reused no longer than the listing.
        A commit cannot move, so its URL is reused until it expires."""
        from zea.internal import preset_utils

        commit = "c0ffee" * 6 + "c0ff"
        monkeypatch.setattr(preset_utils, "_HF_CACHE_TTL", 0)
        hf_remote.resolve(HF_PATH)
        hf_remote.resolve(HF_PATH)
        assert len(hub.heads) == 2
        hf_remote.resolve(HF_PATH, revision=commit)
        hf_remote.resolve(HF_PATH, revision=commit)
        assert len(hub.heads) == 3

    def test_clearing_the_hub_caches_drops_urls(self, hub):
        """An upload clears the hub caches, so the next open sees the new file."""
        from zea.internal.preset_utils import _hf_clear_caches

        hf_remote.resolve(HF_PATH)
        _hf_clear_caches()
        hf_remote.resolve(HF_PATH)
        assert len(hub.heads) == 2

    def test_refresh_skips_the_cache(self, hub):
        hf_remote.resolve(HF_PATH)
        hf_remote.resolve(HF_PATH, refresh=True)
        assert len(hub.heads) == 2

    def test_keyed_on_revision(self, hub):
        hf_remote.resolve(HF_PATH)
        hf_remote.resolve(HF_PATH, revision="v1")
        assert len(hub.heads) == 2
        assert "/resolve/v1/" in hub.heads[1]

    def test_no_token_sent_to_the_cdn(self, hub):
        assert not hf_remote.resolve(HF_PATH).send_auth

    def test_token_kept_for_the_hub_itself(self, hub):
        from huggingface_hub import constants

        hub.location = f"{constants.ENDPOINT}/datasets/org/repo/resolve/main/data/x.hdf5"
        assert hf_remote.resolve(HF_PATH).send_auth

    def test_requires_a_file(self, hub):
        with pytest.raises(ValueError, match="single file"):
            hf_remote.resolve("hf://org/repo")


class TestSignedURL:
    def test_refresh_is_pinned_to_the_first_commit(self, hub):
        location = hf_remote.SignedURL(HF_PATH)
        hub.location = "https://cdn.example/signed?2"
        assert location.refresh("https://cdn.example/signed?1")
        assert "/resolve/c0ffee/" in hub.heads[-1]
        assert location.current()[0] == "https://cdn.example/signed?2"

    def test_refresh_happens_once_for_concurrent_failures(self, hub):
        location = hf_remote.SignedURL(HF_PATH)
        hub.location = "https://cdn.example/signed?2"
        assert location.refresh("https://cdn.example/signed?1")
        assert location.refresh("https://cdn.example/signed?1")  # already replaced
        assert len(hub.heads) == 2

    def test_unchanged_url_is_not_refreshed_again_right_away(self, hub):
        """Many requests rejected at once with a URL the Hub hands out again (e.g. a Hub URL)
        send one request to the Hub, not one each."""
        location = hf_remote.SignedURL(HF_PATH)
        for _ in range(5):
            assert not location.refresh("https://cdn.example/signed?1")
        assert len(hub.heads) == 2

    def test_failed_refresh_is_not_retried_right_away(self, hub, monkeypatch):
        location = hf_remote.SignedURL(HF_PATH)

        def rate_limited(url, **kwargs):
            hub.heads.append(url)
            raise OSError("429")

        monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", rate_limited)
        with pytest.raises(OSError, match="429"):
            location.refresh("https://cdn.example/signed?1")
        assert not location.refresh("https://cdn.example/signed?1")
        assert len(hub.heads) == 2

    def test_failed_early_refresh_keeps_the_current_url(self, hub, monkeypatch):
        """Signed URLs outlive URL_TTL, so a Hub outage at that point must not fail reads."""
        location = hf_remote.SignedURL(HF_PATH)
        monkeypatch.setattr(hf_remote, "URL_TTL", 0)

        def unavailable(url, **kwargs):
            raise RuntimeError("503")

        monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", unavailable)
        assert location.current() == ("https://cdn.example/signed?1", {})

    def test_content_change_while_open_raises(self, hub):
        location = hf_remote.SignedURL(HF_PATH)
        hub.etag = "sha-2"
        with pytest.raises(OSError, match="changed on the Hub"):
            location.refresh(location.current()[0])

    def test_stale_url_is_refreshed_before_use(self, hub, monkeypatch):
        location = hf_remote.SignedURL(HF_PATH)
        monkeypatch.setattr(hf_remote, "URL_TTL", 0)
        hub.location = "https://cdn.example/signed?2"
        assert location.current() == ("https://cdn.example/signed?2", {})


class TestStreamedFile:
    """End to end: ``zea.File("hf://...")`` against a local server standing in for the CDN."""

    @pytest.fixture(autouse=True)
    def _fast_path(self, monkeypatch):
        """The test file is small; send its reads through the chunk reader anyway."""
        monkeypatch.setattr(chunk_reader, "MIN_BYTES", 0)

    @pytest.fixture
    def cdn(self, tmp_path, hub):
        path = _write(tmp_path / "x.hdf5", _structured())
        server = _CountingServer(tmp_path, latency=0)
        hub.location = server.url + path.name
        hub.size = path.stat().st_size
        yield SimpleNamespace(server=server, path=path, hub=hub)
        server.close()

    def test_reads_match_the_local_file(self, cdn):
        with File(HF_PATH, cache=False, progress=False) as remote, File(cdn.path) as local:
            np.testing.assert_array_equal(remote[RAW][0:3], local[RAW][0:3])
            got = chunk_reader.read(remote[RAW], slice(0, 4), remote._chunk_fetcher)
            np.testing.assert_array_equal(got, local[RAW][0:4])

    def test_one_hub_request_for_open_and_reads(self, cdn):
        with File(HF_PATH, cache=False, progress=False) as remote:
            for i in range(4):
                chunk_reader.read(remote[RAW], i, remote._chunk_fetcher)
        assert len(cdn.hub.heads) == 1  # every read after that went to the CDN
        assert cdn.server.count > 1

    def test_expired_url_mid_read_is_refreshed(self, cdn):
        with File(HF_PATH, cache=False, progress=False) as remote, File(cdn.path) as local:
            fetcher = remote._chunk_fetcher
            # The CDN stops accepting the URL; the hub hands out a new, working one.
            fetcher.location._resolved = hf_remote.Resolved(
                **{**fetcher.location._resolved.__dict__, "url": cdn.server.url + "gone.hdf5"}
            )
            got = chunk_reader.read(remote[RAW], slice(0, 2), fetcher)
            np.testing.assert_array_equal(got, local[RAW][0:2])
        assert len(cdn.hub.heads) == 2

    def test_handles_share_metadata_blocks(self, cdn):
        """A second handle of the same file (another dataloader thread) fetches nothing."""
        with File(HF_PATH, cache=False, progress=False) as first:
            first[RAW].shape
        before = cdn.server.count
        with File(HF_PATH, cache=False, progress=False) as second:
            second[RAW].shape
        assert cdn.server.count == before

    def test_shared_blocks_are_bounded(self):
        blocks = hf_remote._SharedBlocks(max_bytes=10)
        for i in range(4):
            blocks.put(("etag", i), b"xxxx")
        assert blocks.get(("etag", 0)) is None  # evicted, oldest first
        assert blocks.get(("etag", 3)) == b"xxxx"

    def test_resolved_location_is_shared_with_the_chunk_reader(self, cdn):
        with File(HF_PATH, cache=False, progress=False) as remote:
            assert remote._chunk_fetcher.location is remote._stream_fileobj.location


def test_merged_requests_are_counted_once(tmp_path, hub):
    """The network byte counter sees the bytes actually fetched, not one count per chunk."""
    from zea.data import chunk_cache

    path = _write(tmp_path / "x.hdf5", _structured())
    server = _CountingServer(tmp_path, latency=0)
    try:
        fetcher = chunk_reader.HTTPFetcher(server.url + path.name)
        before = chunk_cache.network_bytes()
        fetcher.fetch([(0, 100), (100, 100)])
        assert chunk_cache.network_bytes() - before == 200
        assert server.count == 1
    finally:
        server.close()


@pytest.fixture
def small_cdn(tmp_path, hub, monkeypatch):
    """A streamed file whose frames are each below MIN_BYTES, with a private chunk cache."""
    from zea.data import chunk_cache

    monkeypatch.setattr(chunk_cache, "ZEA_CACHE_DIR", tmp_path / "zea-cache")
    path = _write(tmp_path / "x.hdf5", _structured())
    server = _CountingServer(tmp_path, latency=0)
    hub.location = server.url + path.name
    hub.size = path.stat().st_size
    yield SimpleNamespace(server=server, path=path)
    server.close()


def test_small_reads_share_blocks(small_cdn):
    """Frame reads below MIN_BYTES are served from the file's blocks, not one request each."""
    with File(HF_PATH, progress=False) as remote, File(small_cdn.path) as local:
        assert remote[RAW][0].nbytes < chunk_reader.MIN_BYTES
        before = small_cdn.server.count
        for i in range(remote[RAW].shape[0]):
            got = chunk_reader.read(remote[RAW], i, remote._chunk_fetcher)
            np.testing.assert_array_equal(got, local[RAW][i])
    # The whole test file fits in the block fetched on open.
    assert small_cdn.server.count == before


def test_open_uses_cached_first_block(small_cdn):
    """Opening a file whose first block is in the chunk cache sends no request for it,
    e.g. in a second dataloader worker process (simulated by clearing the memory caches)."""
    with File(HF_PATH, progress=False) as remote:
        remote[RAW].shape
    hf_remote._shared_blocks.clear()
    before = small_cdn.server.count
    with File(HF_PATH, progress=False) as remote:
        remote[RAW].shape
    assert small_cdn.server.count == before


def test_small_reads_of_large_chunks_stay_one_request(small_cdn):
    """A small read whose chunk spans several blocks is one request, not one per block."""
    with File(HF_PATH, progress=False, block_size=1024) as remote, File(small_cdn.path) as local:
        dset = remote[RAW]
        assert dset.id.get_chunk_info(0).size > 4 * 1024
        before = small_cdn.server.count
        got = chunk_reader.read(dset, 0, remote._chunk_fetcher)
        np.testing.assert_array_equal(got, local[RAW][0])
        assert small_cdn.server.count - before == dset.id.get_num_chunks() // dset.shape[0]


@pytest.mark.parametrize("cache_type", ["blockcache", "none"])
def test_reads_match_the_file(small_cdn, cache_type):
    expected = small_cdn.path.read_bytes()
    stream = hf_remote.open_remote(HF_PATH, block_size=4096, cache_type=cache_type)
    for offset, size in [(0, 1), (100, 5000), (4095, 2), (len(expected) - 10, 50)]:
        assert stream.read_at(offset, size) == expected[offset : offset + size]
    stream.seek(4000)
    assert stream.read(200) == expected[4000:4200] and stream.tell() == 4200


@pytest.mark.parametrize("cache_type", ["blockcache", "none"])
def test_stream_survives_exhausted_retries(tmp_path, hub, monkeypatch, cache_type):
    """A read that outlasts the retries raises an ``OSError`` and keeps no block, so
    the next read through the same stream gets the right bytes."""
    monkeypatch.setattr(chunk_reader, "MAX_RETRIES", 1)
    monkeypatch.setattr(chunk_reader, "BASE_BACKOFF", 0.01)
    path = _write(tmp_path / "x.hdf5", _structured())
    server = _CountingServer(tmp_path, latency=0, reject_first=2, reject_status=503)
    hub.location, hub.size = server.url + path.name, path.stat().st_size
    try:
        stream = hf_remote.open_remote(HF_PATH, block_size=4096, cache_type=cache_type)
        with pytest.raises(OSError):
            stream.read_at(100, 50)
        assert stream.read_at(100, 50) == path.read_bytes()[100:150]
        assert server.count == 3
    finally:
        server.close()


def test_read_shares_the_blocks_of_read_at(small_cdn):
    """h5py's reads and the chunk reader's positional reads use the same blocks."""
    stream = hf_remote.open_remote(HF_PATH, block_size=4096)
    stream.read_at(0, 10)
    before = small_cdn.server.count
    stream.seek(100)
    stream.read(1000)
    assert small_cdn.server.count == before


def test_unknown_cache_type_raises(small_cdn):
    with pytest.raises(ValueError, match="cache_type"):
        hf_remote.open_remote(HF_PATH, cache_type="nope")


def test_no_prefetch_when_nothing_keeps_the_block(small_cdn, monkeypatch):
    """With cache_type="none" and no chunk cache, prefetching the first block on open would
    be a request nothing reuses."""
    calls = []
    original = hf_remote.RemoteFile.read_at
    monkeypatch.setattr(
        hf_remote.RemoteFile, "read_at", lambda self, *a: calls.append(a) or original(self, *a)
    )
    with File(HF_PATH, progress=False, cache=False, cache_type="none") as remote:
        remote[RAW].shape
    assert (0, 1) not in calls  # h5py's own reads go through read_at too
    calls.clear()
    with File(HF_PATH, progress=False, cache=False) as remote:
        remote[RAW].shape
    assert calls[0] == (0, 1)


def test_small_reads_draw_no_progress_bar(small_cdn, monkeypatch):
    """A read below MIN_BYTES is one request at most, so it gets no bar (h5py, which served
    these reads before, drew none either)."""
    from unittest.mock import MagicMock

    tqdm_mock = MagicMock()
    monkeypatch.setattr("tqdm.auto.tqdm", tqdm_mock)
    with File(HF_PATH, progress=False) as remote:
        for _ in range(2):
            chunk_reader.read(remote[RAW], 0, remote._chunk_fetcher, progress=True)
    tqdm_mock.assert_not_called()


def test_small_reads_fetch_their_blocks_concurrently(tmp_path, hub):
    """A small read whose chunks lie in several blocks fetches those blocks together."""
    path = _write(tmp_path / "x.hdf5", _structured())
    server = _CountingServer(tmp_path, latency=0.05)
    hub.location, hub.size = server.url + path.name, path.stat().st_size
    try:
        block = 64 << 10
        with File(HF_PATH, progress=False, cache=False, block_size=block) as remote:
            dset = remote[RAW]
            chunks = [dset.id.get_chunk_info(i) for i in (3, 4)]
            assert all(chunk.size < block for chunk in chunks)
            assert len({chunk.byte_offset // block for chunk in chunks}) == 2
            server.peak, before = 0, server.count
            got = chunk_reader.read(dset, slice(3, 5), remote._chunk_fetcher)
            with File(path) as local:
                np.testing.assert_array_equal(got, local[RAW][3:5])
        assert server.count - before == 2 and server.peak == 2
    finally:
        server.close()


def test_small_reads_use_blocks_only_if_the_stream_keeps_them(small_cdn):
    """With cache_type="none", serving small reads from blocks would fetch every chunk on
    its own, so they go through the merged requests of fetch instead."""
    with File(HF_PATH, progress=False, cache=False, cache_type="none") as remote:
        assert remote._chunk_fetcher.blocks is None
    with File(HF_PATH, progress=False, cache=False) as remote:
        assert remote._chunk_fetcher.blocks is remote._stream_fileobj
