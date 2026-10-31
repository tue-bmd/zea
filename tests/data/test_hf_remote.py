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
