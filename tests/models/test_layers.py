"""Tests for the shared building blocks in :mod:`zea.models.layers`."""

import math

import numpy as np
import pytest
from keras import ops

from zea.models.layers import DownBlock, ResidualBlock, UpBlock, sinusoidal_embedding


def test_sinusoidal_embedding_matches_the_closed_form():
    """The embedding is sin/cos of a log-spaced frequency sweep."""
    # Modest frequencies: float32 sin/cos of the very fast ones is all round-off.
    x = np.array([[[[0.25]]]], dtype="float32")
    min_freq, max_freq, dims = 0.5, 4.0, 8

    embedding = ops.convert_to_numpy(sinusoidal_embedding(x, min_freq, max_freq, dims))

    frequencies = np.exp(np.linspace(np.log(min_freq), np.log(max_freq), dims // 2))
    angular_speeds = 2.0 * math.pi * frequencies * 0.25
    expected = np.concatenate([np.sin(angular_speeds), np.cos(angular_speeds)])
    assert embedding.shape == (1, 1, 1, dims)
    np.testing.assert_allclose(embedding[0, 0, 0], expected, rtol=1e-5, atol=1e-5)


class TestResidualBlock:
    """The residual block keeps the resolution and sets the channel count."""

    def test_projects_the_skip_when_the_width_changes(self, rng):
        x = rng.standard_normal((2, 8, 8, 4)).astype("float32")

        out = ResidualBlock(16)(x)

        assert out.shape == (2, 8, 8, 16)

    def test_reuses_the_input_as_skip_when_the_width_matches(self, rng):
        x = rng.standard_normal((2, 8, 8, 16)).astype("float32")

        out = ResidualBlock(16)(x)

        assert out.shape == (2, 8, 8, 16)

    @pytest.mark.parametrize("normalization", ["batch_norm", "group_norm"])
    def test_supports_both_normalizations(self, rng, normalization):
        x = rng.standard_normal((2, 8, 8, 32)).astype("float32")

        out = ResidualBlock(32, normalization=normalization)(x)

        assert out.shape == (2, 8, 8, 32)


def test_down_block_halves_the_resolution_and_records_skips(rng):
    """Each residual block in a down block leaves one skip behind."""
    x = rng.standard_normal((2, 8, 8, 4)).astype("float32")
    skips = []

    out = DownBlock(16, block_depth=2)([x, skips])

    assert out.shape == (2, 4, 4, 16)
    assert [tuple(skip.shape) for skip in skips] == [(2, 8, 8, 16), (2, 8, 8, 16)]


def test_up_block_doubles_the_resolution_and_consumes_skips(rng):
    """An up block undoes exactly what the matching down block did."""
    x = rng.standard_normal((2, 8, 8, 4)).astype("float32")
    skips = []
    down = DownBlock(16, block_depth=2)([x, skips])

    out = UpBlock(16, block_depth=2)([down, skips])

    assert out.shape == (2, 8, 8, 16)
    assert skips == []
