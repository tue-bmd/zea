"""Tests for the nnU-Net based left ventricle / myocardium segmentation model.

The preset ships an ONNX file that is run through ONNX Runtime. These tests
stand a stub session in its place, so the zea side of the model -- refusing to
run before the weights are loaded, fetching the file, and handing the input to
the session -- is covered without a download.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from zea.models.lv_segmentation import INFERENCE_SIZE, AugmentedCamusSeg


class StubOnnxSession:
    """Minimal stand-in for ``onnxruntime.InferenceSession``."""

    def __init__(self, path, output=None):
        self.path = path
        self.output = output
        self.seen = []

    class _IO:
        def __init__(self, name):
            self.name = name

    def get_inputs(self):
        return [self._IO("input")]

    def get_outputs(self):
        return [self._IO("output")]

    def run(self, output_names, feed):
        self.seen.append(feed["input"])
        if self.output is not None:
            return [self.output]
        batch = feed["input"].shape[0]
        return [np.zeros((batch, 3, INFERENCE_SIZE, INFERENCE_SIZE), dtype="float32")]


@pytest.fixture
def onnx_preset(local_preset):
    """A local preset directory holding a (dummy) ``model.onnx``."""
    preset = local_preset(AugmentedCamusSeg)
    (Path(preset) / "model.onnx").write_bytes(b"not really an onnx file")
    return preset


@pytest.fixture
def stub_onnxruntime(monkeypatch):
    """Replaces ``onnxruntime.InferenceSession`` with :class:`StubOnnxSession`."""
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "InferenceSession", StubOnnxSession)


def test_call_raises_without_loaded_weights():
    """Inference before ``custom_load_weights()`` is refused with a clear message."""
    model = AugmentedCamusSeg()
    with pytest.raises(ValueError, match="weights not loaded"):
        model(np.zeros((1, 1, INFERENCE_SIZE, INFERENCE_SIZE), dtype="float32"))


def test_custom_load_weights_reports_missing_onnxruntime(monkeypatch, onnx_preset):
    """Without onnxruntime installed the user gets an install hint, not an ImportError."""
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    model = AugmentedCamusSeg()

    with pytest.raises(ImportError, match="pip install onnxruntime"):
        model.custom_load_weights(onnx_preset)


def test_from_preset_opens_the_onnx_file(onnx_preset, stub_onnxruntime):
    """The ONNX session is created from the ``model.onnx`` inside the preset."""
    model = AugmentedCamusSeg.from_preset(onnx_preset)

    assert model.onnx_sess.path.endswith("model.onnx")


def test_call_returns_the_session_output(onnx_preset, stub_onnxruntime, rng):
    """The model returns three logit channels: background, LV and myocardium."""
    model = AugmentedCamusSeg.from_preset(onnx_preset)
    x = rng.random((2, 1, INFERENCE_SIZE, INFERENCE_SIZE))

    out = model(x)

    assert out.shape == (2, 3, INFERENCE_SIZE, INFERENCE_SIZE)
    # The session is fed float32, whatever dtype the caller passed in.
    assert model.onnx_sess.seen[0].dtype == np.float32
