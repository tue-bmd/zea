"""Tests for the EchoNet-Dynamic segmentation model.

The published preset ships a TensorFlow SavedModel, which zea loads through
:meth:`EchoNetDynamic._load_layer`. These tests replace that loader with a stub
so the preprocessing around it -- resizing to the inference size, broadcasting
grayscale to three channels, and resizing back -- can be tested offline.
"""

import unittest.mock
from pathlib import Path

import keras
import numpy as np
import pytest

from zea.models.echonet import INFERENCE_SIZE, EchoNetDynamic

BATCH_SIZE = 2

pytestmark = pytest.mark.tensorflow


class StubSavedModel(keras.layers.Layer):
    """Stands in for the TensorFlow SavedModel that the EchoNet preset ships."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = keras.layers.Conv2D(1, 1)
        self.seen_shapes = []

    def call(self, x):
        self.seen_shapes.append(tuple(x.shape))
        out = self.conv(x)
        # A real SavedModel is loaded as a TFSMLayer, which returns a dict of outputs.
        if keras.backend.backend() == "tensorflow":
            return {"segmentation": out}
        return out


@pytest.fixture
def echonet_preset(local_preset):
    """A local EchoNet preset directory holding the files the model asks for."""
    preset = Path(local_preset(EchoNetDynamic))
    (preset / "variables").mkdir()
    for name in (
        "variables/variables.data-00000-of-00001",
        "variables/variables.index",
        "saved_model.pb",
        "fingerprint.pb",
    ):
        (preset / name).touch()
    return str(preset)


@pytest.fixture
def stub_load_layer(monkeypatch):
    """Replaces the SavedModel loader with a stub, and records what it was given."""
    loaded = []

    def _load_layer(self, path):
        loaded.append(str(path))
        return StubSavedModel()

    monkeypatch.setattr(EchoNetDynamic, "_load_layer", _load_layer)
    return loaded


@pytest.fixture
def model():
    """EchoNetDynamic without weights loaded."""
    return EchoNetDynamic()


class TestWithoutWeights:
    """An EchoNetDynamic constructed directly has no network yet."""

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


class TestFromPreset:
    """An EchoNetDynamic loaded from a (stubbed) preset."""

    def test_loads_the_saved_model_from_the_preset(self, echonet_preset, stub_load_layer):
        """The SavedModel is loaded from the preset root."""
        model = EchoNetDynamic.from_preset(echonet_preset)

        assert model.network is not None
        assert stub_load_layer == [echonet_preset]

    @pytest.mark.parametrize("channels", [1, 3])
    def test_segmentation_keeps_the_input_resolution(
        self, echonet_preset, stub_load_layer, rng, channels
    ):
        """Inference runs at 112x112, but the mask comes back at the input resolution."""
        model = EchoNetDynamic.from_preset(echonet_preset)
        x = rng.random((BATCH_SIZE, 64, 80, channels)).astype("float32")

        out = model(x)

        assert out.shape == (BATCH_SIZE, 64, 80, 1)

    def test_input_is_resized_to_the_inference_size(self, echonet_preset, stub_load_layer, rng):
        """Whatever the input resolution, the network sees ``INFERENCE_SIZE`` inputs."""
        model = EchoNetDynamic.from_preset(echonet_preset)

        model(rng.random((1, 40, 50, 1)).astype("float32"))

        assert set(model.network.seen_shapes) == {(1, INFERENCE_SIZE, INFERENCE_SIZE, 3)}

    @pytest.mark.parametrize(
        ("shape", "match"),
        [
            ((112, 112, 1), "4 dimensions"),
            ((1, 112, 112, 2), "1 or 3 channels"),
        ],
    )
    def test_rejects_bad_input(self, echonet_preset, stub_load_layer, shape, match):
        """Only 4D input with one or three channels is accepted."""
        model = EchoNetDynamic.from_preset(echonet_preset)
        with pytest.raises(AssertionError, match=match):
            model(np.zeros(shape, dtype="float32"))
