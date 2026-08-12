"""Tests for the Tiny Autoencoder (TAESD) model.

The published preset ships a TensorFlow SavedModel per half of the autoencoder,
which zea loads through :meth:`TinyBase._load_layer`. These tests replace that
loader with a stub so the wiring around it -- fetching the preset files,
grayscale handling, and encode/decode -- can be tested without a download.
"""

import unittest.mock
from pathlib import Path

import keras
import pytest

from zea.models.taesd import TinyAutoencoder, TinyBase, TinyDecoder, TinyEncoder

BATCH_SIZE = 2

pytestmark = pytest.mark.tensorflow


class StubSavedModel(keras.layers.Layer):
    """Stands in for the TensorFlow SavedModel that a TAESD preset ships."""

    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.conv = keras.layers.Conv2D(out_channels, 1)

    def call(self, x):
        out = self.conv(x)
        # A real SavedModel is loaded as a TFSMLayer, which returns a dict of outputs.
        if keras.backend.backend() == "tensorflow":
            return {"output_0": out}
        return out


@pytest.fixture
def taesd_preset(local_preset):
    """A local TAESD preset directory holding the files both halves ask for."""
    preset = Path(local_preset(TinyAutoencoder))
    for half in ("encoder", "decoder"):
        (preset / half / "variables").mkdir(parents=True)
        for name in (
            "variables/variables.data-00000-of-00001",
            "variables/variables.index",
            "saved_model.pb",
            "fingerprint.pb",
        ):
            (preset / half / name).touch()
    return str(preset)


@pytest.fixture
def stub_load_layer(monkeypatch):
    """Replaces the SavedModel loader with a stub, and records what it was given."""
    loaded = []

    def _load_layer(self, path):
        loaded.append(str(path))
        # The encoder maps to a 4-channel latent, the decoder back to RGB.
        return StubSavedModel(4 if isinstance(self, TinyEncoder) else 3)

    monkeypatch.setattr(TinyBase, "_load_layer", _load_layer)
    return loaded


@pytest.fixture
def model():
    """TinyAutoencoder without weights loaded."""
    return TinyAutoencoder()


class TestWithoutWeights:
    """A TinyAutoencoder that was constructed directly has no network yet."""

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

    def test_halves_reject_an_unknown_type(self):
        """``TinyBase`` only knows about an encoder and a decoder."""
        with pytest.raises(AssertionError, match="encoder"):
            TinyBase(tiny_type="something-else")

    @pytest.mark.parametrize("cls", [TinyEncoder, TinyDecoder])
    def test_half_raises_for_unsupported_backend(self, cls):
        """Both halves refuse a backend other than tensorflow or jax."""
        with unittest.mock.patch("keras.backend.backend", return_value="torch"):
            with pytest.raises(NotImplementedError):
                cls()

    @pytest.mark.parametrize("cls", [TinyEncoder, TinyDecoder])
    def test_half_raises_without_loaded_weights(self, cls, rng):
        """Each half also refuses to run before its SavedModel is loaded."""
        x = rng.random((1, 32, 32, 3)).astype("float32")
        with pytest.raises(ValueError, match="from_preset"):
            cls()(x)


class TestFromPreset:
    """A TinyAutoencoder loaded from a (stubbed) preset."""

    def test_loads_both_halves_from_the_preset(self, taesd_preset, stub_load_layer):
        """Each half is loaded from its own subdirectory of the preset."""
        model = TinyAutoencoder.from_preset(taesd_preset)

        assert model.encoder.network is not None
        assert model.decoder.network is not None
        assert [path.rsplit("/", 1)[-1] for path in stub_load_layer] == ["encoder", "decoder"]

    def test_roundtrip_keeps_rgb_channels(self, taesd_preset, stub_load_layer, rng):
        """An RGB image comes back out as an RGB image."""
        model = TinyAutoencoder.from_preset(taesd_preset)
        x = rng.random((BATCH_SIZE, 32, 32, 3)).astype("float32")

        assert model(x).shape == (BATCH_SIZE, 32, 32, 3)

    def test_roundtrip_keeps_grayscale_single_channel(self, taesd_preset, stub_load_layer, rng):
        """A grayscale image is broadcast to RGB for the encoder and folded back on decode."""
        model = TinyAutoencoder.from_preset(taesd_preset)
        x = rng.random((BATCH_SIZE, 32, 32, 1)).astype("float32")

        assert model(x).shape == (BATCH_SIZE, 32, 32, 1)

    def test_channel_handling_does_not_leak_between_calls(self, taesd_preset, stub_load_layer, rng):
        """A grayscale image must not make the next RGB image come back grayscale."""
        model = TinyAutoencoder.from_preset(taesd_preset)
        grayscale = rng.random((1, 32, 32, 1)).astype("float32")
        rgb = rng.random((1, 32, 32, 3)).astype("float32")

        assert model(grayscale).shape == (1, 32, 32, 1)
        assert model(rgb).shape == (1, 32, 32, 3)

    def test_encode_downsamples_to_the_latent(self, taesd_preset, stub_load_layer, rng):
        """encode() returns the latent, decode() maps it back to an image."""
        model = TinyAutoencoder.from_preset(taesd_preset)
        x = rng.random((BATCH_SIZE, 32, 32, 3)).astype("float32")

        latent = model.encode(x)
        assert latent.shape == (BATCH_SIZE, 32, 32, 4)
        assert model.decode(latent).shape == (BATCH_SIZE, 32, 32, 3)
