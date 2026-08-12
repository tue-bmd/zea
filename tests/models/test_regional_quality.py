"""Tests for the MobileNetV2 regional image quality model.

The preset ships an ONNX file plus a bias-correction ``.npy``. These tests use a
stub ONNX session so the zea side -- input normalization, the slope/intercept
debiasing, and loading both preset files -- is covered without a download.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from zea.models.regional_quality import (
    QUALITY_CLASSES,
    REGION_LABELS,
    MobileNetv2RegionalQuality,
)

N_REGIONS = len(REGION_LABELS)
SLOPE, INTERCEPT = 2.0, 1.0


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
        return [np.zeros((batch, N_REGIONS), dtype="float32")]


@pytest.fixture
def quality_preset(local_preset):
    """A local preset directory with a dummy ONNX file and a bias correction."""
    preset = local_preset(MobileNetv2RegionalQuality)
    (Path(preset) / "model.onnx").write_bytes(b"not really an onnx file")
    np.save(
        Path(preset) / "slope_intercept_bias_correction.npy",
        np.array([SLOPE, INTERCEPT], dtype="float32"),
    )
    return preset


@pytest.fixture
def stub_onnxruntime(monkeypatch):
    """Replaces ``onnxruntime.InferenceSession`` with :class:`StubOnnxSession`."""
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "InferenceSession", StubOnnxSession)


def test_region_labels_and_classes_are_consistent():
    """The model scores eight myocardial regions on a five-point scale."""
    assert len(REGION_LABELS) == 8
    assert QUALITY_CLASSES == ["not visible", "poor", "ok", "good", "excellent"]


class TestPreprocessInput:
    """``preprocess_input`` rescales whatever range it is given to [0, 255]."""

    def test_rescales_to_the_full_range(self, rng):
        model = MobileNetv2RegionalQuality()
        x = rng.random((1, 1, 8, 8)) * 4 - 2  # roughly [-2, 2]

        out = model.preprocess_input(x)

        assert out.dtype == np.float32
        np.testing.assert_allclose([out.min(), out.max()], [0.0, 255.0], atol=1e-4)

    def test_constant_input_becomes_zeros(self):
        """A flat image has no range to stretch, so it maps to zeros instead of NaNs."""
        model = MobileNetv2RegionalQuality()

        out = model.preprocess_input(np.full((1, 1, 4, 4), 7.0))

        np.testing.assert_array_equal(out, np.zeros((1, 1, 4, 4), dtype="float32"))


def test_call_raises_without_loaded_weights():
    """Inference before ``custom_load_weights()`` is refused with a clear message."""
    model = MobileNetv2RegionalQuality()
    with pytest.raises(ValueError, match="weights not loaded"):
        model(np.zeros((1, 1, 256, 256), dtype="float32"))


def test_custom_load_weights_reports_missing_onnxruntime(monkeypatch, quality_preset):
    """Without onnxruntime installed the user gets an install hint, not an ImportError."""
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    model = MobileNetv2RegionalQuality()

    with pytest.raises(ImportError, match="pip install onnxruntime"):
        model.custom_load_weights(quality_preset)


def test_from_preset_loads_both_assets(quality_preset, stub_onnxruntime):
    """Both the ONNX file and the bias correction come out of the preset."""
    model = MobileNetv2RegionalQuality.from_preset(quality_preset)

    assert model.onnx_sess.path.endswith("model.onnx")
    np.testing.assert_array_equal(model.slope_intercept, [SLOPE, INTERCEPT])


def test_scores_are_debiased_with_slope_and_intercept(quality_preset, stub_onnxruntime, rng):
    """The raw network scores are mapped back with ``(score - intercept) / slope``."""
    raw = rng.random((2, N_REGIONS)).astype("float32")
    model = MobileNetv2RegionalQuality.from_preset(quality_preset)
    model.onnx_sess.output = raw

    out = model(rng.random((2, 1, 64, 64)))

    assert out.shape == (2, N_REGIONS)
    np.testing.assert_allclose(out, (raw - INTERCEPT) / SLOPE, rtol=1e-6)


def test_input_is_normalized_before_the_session(quality_preset, stub_onnxruntime, rng):
    """The session sees the [0, 255] normalized image, not the raw input."""
    model = MobileNetv2RegionalQuality.from_preset(quality_preset)

    model(rng.random((1, 1, 32, 32)) * 0.01)

    fed = model.onnx_sess.seen[0]
    np.testing.assert_allclose([fed.min(), fed.max()], [0.0, 255.0], atol=1e-4)
