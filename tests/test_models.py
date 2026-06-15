"""Tests for zea models."""

import numpy as np
import pytest

from zea.models.speckle2self import Speckle2Self

from . import DEFAULT_TEST_SEED

BATCH_SIZE = 2
IMAGE_SHAPE = (512, 512, 1)


@pytest.fixture
def rng():
    """Random number generator for reproducible tests."""
    return np.random.default_rng(DEFAULT_TEST_SEED)


@pytest.fixture
def speckle2self_model():
    """Speckle2Self model without pretrained weights."""
    return Speckle2Self()


def test_speckle2self_call_nchw(speckle2self_model, rng):
    """Test Speckle2Self forward pass with (N, 1, H, W) input."""
    x = rng.random((BATCH_SIZE, *IMAGE_SHAPE)).astype("float32")
    out = speckle2self_model(x)
    assert out.shape == (BATCH_SIZE, *IMAGE_SHAPE)
