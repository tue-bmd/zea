"""Tests for zea models."""

import unittest.mock

import numpy as np
import pytest

from zea.models.echonet import EchoNetDynamic
from zea.models.speckle2self import Speckle2Self
from zea.models.taesd import TinyAutoencoder

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


class TestTinyAutoencoder:
    """Tests for ``TinyAutoencoder`` (TAESD) without pretrained weights."""

    @pytest.fixture
    def model(self):
        """TinyAutoencoder instantiated in the default (tensorflow) backend, no weights loaded."""
        return TinyAutoencoder()

    def test_raises_for_unsupported_backend(self):
        """Constructor raises NotImplementedError when the backend is not tensorflow or jax."""
        with unittest.mock.patch("keras.backend.backend", return_value="torch"):
            with pytest.raises(NotImplementedError):
                TinyAutoencoder()

    def test_encode_raises_without_loaded_weights(self, model, rng):
        """encode() raises ValueError when called before loading weights via from_preset()."""
        x = rng.random((BATCH_SIZE, 64, 64, 3)).astype("float32")
        with pytest.raises(ValueError, match="from_preset"):
            model.encode(x)

    def test_call_raises_without_loaded_weights(self, model, rng):
        """Forward pass raises ValueError when called before loading weights via from_preset()."""
        x = rng.random((BATCH_SIZE, 64, 64, 3)).astype("float32")
        with pytest.raises(ValueError, match="from_preset"):
            model(x)


class TestEchoNetDynamic:
    """Tests for ``EchoNetDynamic`` without pretrained weights."""

    @pytest.fixture
    def model(self):
        """EchoNetDynamic instantiated in the default (tensorflow) backend, no weights loaded."""
        return EchoNetDynamic()

    def test_raises_for_unsupported_backend(self):
        """Constructor raises NotImplementedError when the backend is not tensorflow or jax."""
        with unittest.mock.patch("keras.backend.backend", return_value="torch"):
            with pytest.raises(NotImplementedError):
                EchoNetDynamic()

    def test_call_raises_without_loaded_weights(self, model, rng):
        """call() raises ValueError when called before loading weights via from_preset()."""
        x = rng.random((BATCH_SIZE, 112, 112, 1)).astype("float32")
        with pytest.raises(ValueError, match="from_preset"):
            model(x)
