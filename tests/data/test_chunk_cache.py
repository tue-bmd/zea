"""Tests for :mod:`zea.data.chunk_cache`.

A stale or torn chunk is worse than no cache at all: it corrupts a read silently, with no
error anywhere. So most of these tests are about *not* returning bytes — on a short file, on
a different content hash, on a size that does not match.
"""

import os

import numpy as np
import pytest

from zea.data.chunk_cache import ChunkCache, cache_for, clear, prune

RAW = "tracks/track_0/data/raw_data"


@pytest.fixture
def root(tmp_path):
    return tmp_path / "cache"


class TestGetPut:
    def test_roundtrip(self, root):
        cache = ChunkCache("deadbeef", root)
        assert cache.get(0, 4) is None
        cache.put(0, 4, b"data")
        assert cache.get(0, 4) == b"data"

    def test_offset_and_size_are_both_part_of_the_key(self, root):
        cache = ChunkCache("deadbeef", root)
        cache.put(0, 4, b"aaaa")
        cache.put(8, 4, b"bbbb")
        assert cache.get(0, 4) == b"aaaa"
        assert cache.get(8, 4) == b"bbbb"
        assert cache.get(4, 4) is None

    def test_short_file_is_a_miss_not_a_corrupt_read(self, root):
        """A torn write (killed mid-put, full disk) must not be served to the decoder.

        The bytes would decompress to garbage or raise deep inside a codec, far from the
        cause. A miss just costs a re-fetch.
        """
        cache = ChunkCache("deadbeef", root)
        cache.put(0, 16, b"0123456789abcdef")
        cache._path(0, 16).write_bytes(b"trunc")  # simulate the torn file
        assert cache.get(0, 16) is None

    def test_put_failure_is_not_an_error(self, root, monkeypatch):
        """Caching is an optimisation: if the disk is full or read-only, the read still works."""
        cache = ChunkCache("deadbeef", root)

        def boom(*args, **kwargs):
            raise OSError("no space left on device")

        monkeypatch.setattr("tempfile.mkstemp", boom)
        cache.put(0, 4, b"data")  # must not raise
        assert cache.get(0, 4) is None


class TestKey:
    def test_content_hash_keys_the_cache(self, root):
        """Keyed by content, so a re-upload misses instead of serving the previous file.

        An ``hf://`` resolve URL points at a mutable ref (``/resolve/main/``): the same URL
        returns different bytes after a re-upload. Keying on the URL would serve stale chunks.
        """
        old = cache_for({"lfs": {"sha256": "1" * 64}}, root)
        new = cache_for({"lfs": {"sha256": "2" * 64}}, root)
        old.put(0, 4, b"OLD!")

        assert new.get(0, 4) is None, "a re-uploaded file must not hit the old file's chunks"
        assert old.get(0, 4) == b"OLD!"

    def test_falls_back_to_blob_id(self, root):
        assert cache_for({"blob_id": "abc"}, root) is not None

    def test_no_content_id_means_no_cache(self, root):
        """Without something identifying the content, we decline to cache rather than guess."""
        assert cache_for({"name": "f.hdf5", "size": 10}, root) is None

    def test_disabled_by_env(self, root, monkeypatch):
        monkeypatch.setattr("zea.data.chunk_cache.ENABLED", False)
        assert cache_for({"lfs": {"sha256": "1" * 64}}, root) is None


class TestPrune:
    def test_evicts_least_recently_used_until_under_budget(self, root):
        cache = ChunkCache("deadbeef", root)
        for i in range(10):
            cache.put(i * 100, 100, os.urandom(100))

        kept = cache._path(900, 100)
        os.utime(kept, (2**31, 2**31))  # far future access time: must survive

        prune(root, max_bytes=300)

        left = [p for p in (root / "chunks").rglob("*") if p.is_file()]
        assert sum(p.stat().st_size for p in left) <= 300
        assert kept.exists(), "the most recently used chunk should survive eviction"

    def test_a_hit_protects_a_chunk_from_eviction(self, root):
        """Reading a chunk must count as *using* it, or the LRU evicts the hot data.

        ``get`` refreshes the access time explicitly: filesystems mounted ``relatime`` (the
        Linux default) only update atime about once a day, so eviction would otherwise fall
        back to write order and drop precisely the chunks a training run keeps re-reading.
        """
        cache = ChunkCache("deadbeef", root)
        for i in range(6):
            cache.put(i * 100, 100, os.urandom(100))
            os.utime(cache._path(i * 100, 100), (1000 + i, 1000 + i))  # oldest written first

        assert cache.get(0, 100) is not None  # re-read the oldest chunk

        prune(root, max_bytes=300)
        assert cache.get(0, 100) is not None, "a chunk that was just read was evicted"

    def test_prune_of_missing_cache_is_a_no_op(self, root):
        assert prune(root, max_bytes=0) == 0

    def test_clear(self, root):
        ChunkCache("deadbeef", root).put(0, 4, b"data")
        clear(root)
        assert not (root / "chunks").exists()


class TestWithFetcher:
    """The cache must actually spare the network, and return exactly the same bytes."""

    def test_second_read_hits_disk_and_issues_no_requests(self, tmp_path, monkeypatch):
        from tests.data.test_chunk_reader import _CountingServer, _structured, _write
        from zea.data.chunk_reader import HTTPFetcher, read
        from zea.data.file import File

        path = _write(tmp_path / "cached.hdf5", _structured())
        server = _CountingServer(path.parent, latency=0.0)
        try:
            cache = ChunkCache("contenthash", tmp_path / "cache")
            url = server.url + path.name

            with File(path) as file:
                dset = file[RAW]
                expected = dset[0:3]

                cold = HTTPFetcher(url, cache=cache)
                before = server.count
                first = read(dset, slice(0, 3), cold)
                requests_cold = server.count - before

                warm = HTTPFetcher(url, cache=cache)  # fresh fetcher, same cache
                before = server.count
                second = read(dset, slice(0, 3), warm)
                requests_warm = server.count - before

            assert requests_cold > 0, "the first read must go to the network"
            assert requests_warm == 0, "the second read must be served entirely from the cache"
            np.testing.assert_array_equal(first, expected)
            np.testing.assert_array_equal(second, expected)
        finally:
            server.close()
