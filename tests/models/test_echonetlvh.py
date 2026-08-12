"""Tests for the EchoNet-LVH landmark model.

What is specific to :class:`EchoNetLVH` is the landmark math around the network:
turning heatmaps into coordinates and drawing them onto an image. The
DeepLabV3+ backbone it wraps is slow to build and is covered by
``test_deeplabv3.py``, so it is replaced here by a stub with the same shape
behaviour.
"""

import sys

import keras
import numpy as np
import pytest
from keras import layers, ops

import zea.models.echonetlvh as echonetlvh_module
from zea.models.echonetlvh import EchoNetLVH

NETWORK_SIZE = 224
NUM_LANDMARKS = 4


class StubBackbone(keras.layers.Layer):
    """A stand-in for ``DeeplabV3Plus``: any image in, ``num_classes`` heatmaps out."""

    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(num_classes, 1)

    def call(self, x):
        return self.conv(x)


def stub_backbone(image_shape, num_classes, pretrained_weights=None):
    """Drop-in replacement for ``zea.models.echonetlvh.DeeplabV3Plus``."""
    return StubBackbone(num_classes)


@pytest.fixture
def model(monkeypatch):
    """An EchoNetLVH whose backbone is stubbed out."""
    monkeypatch.setattr(echonetlvh_module, "DeeplabV3Plus", stub_backbone)
    return EchoNetLVH()


def delta_heatmap(shape, peaks):
    """A batch of heatmaps that are zero everywhere except at ``peaks``."""
    heatmap = np.zeros(shape, dtype="float32")
    for index, (row, col) in enumerate(peaks):
        heatmap[index, row, col] = 1.0
    return heatmap


class TestCall:
    """The forward pass runs at 224x224 and reports back at the input resolution."""

    @pytest.mark.parametrize("channels", [1, 3])
    def test_output_keeps_the_input_resolution(self, model, rng, channels):
        x = rng.random((2, 32, 40, channels)).astype("float32") * 255

        logits = model(x)

        assert logits.shape == (2, 32, 40, NUM_LANDMARKS)

    def test_rejects_non_image_input(self, model, rng):
        with pytest.raises(AssertionError):
            model(rng.random((32, 40, 1)).astype("float32"))


class TestExpectedCoordinate:
    """``expected_coordinate`` is a differentiable argmax over a heatmap."""

    def test_finds_the_peak_of_a_delta_heatmap(self, model):
        """With a single hot pixel the centre of mass is that pixel, as (x, y)."""
        mask = delta_heatmap((1, NETWORK_SIZE, NETWORK_SIZE), [(3, 5)])

        coordinate = model.expected_coordinate(mask)

        np.testing.assert_allclose(ops.convert_to_numpy(coordinate), [[5.0, 3.0]], atol=1e-4)

    def test_accepts_a_grid_for_another_resolution(self, model):
        """A heatmap at another resolution needs a matching coordinate grid."""
        grid = ops.stack(ops.cast(ops.convert_to_tensor(np.indices((8, 10))), "float32"), axis=-1)
        mask = delta_heatmap((1, 8, 10), [(3, 5)])

        coordinate = model.expected_coordinate(mask, grid)

        np.testing.assert_allclose(ops.convert_to_numpy(coordinate), [[5.0, 3.0]], atol=1e-4)

    def test_averages_two_equal_peaks(self, model):
        """Two equally hot pixels put the expected coordinate halfway between them."""
        grid = ops.stack(ops.cast(ops.convert_to_tensor(np.indices((8, 8))), "float32"), axis=-1)
        mask = np.zeros((1, 8, 8), dtype="float32")
        mask[0, 2, 1] = mask[0, 4, 3] = 1.0

        coordinate = model.expected_coordinate(mask, grid)

        np.testing.assert_allclose(ops.convert_to_numpy(coordinate), [[2.0, 3.0]], atol=1e-4)

    def test_negative_values_are_ignored(self, model):
        """Negative logits are clipped away rather than pulling the centre of mass."""
        grid = ops.stack(ops.cast(ops.convert_to_tensor(np.indices((8, 8))), "float32"), axis=-1)
        mask = np.full((1, 8, 8), -1.0, dtype="float32")
        mask[0, 6, 2] = 1.0

        coordinate = model.expected_coordinate(mask, grid)

        np.testing.assert_allclose(ops.convert_to_numpy(coordinate), [[2.0, 6.0]], atol=1e-4)


def test_extract_key_points_returns_row_col_indices(model):
    """One landmark per channel, as ``(row, col)`` indices into the heatmap."""
    logits = np.zeros((2, 16, 20, NUM_LANDMARKS), dtype="float32")
    for channel in range(NUM_LANDMARKS):
        logits[0, 1 + channel, 2 + channel, channel] = 1.0
        logits[1, 5, 7, channel] = 1.0

    key_points = ops.convert_to_numpy(model.extract_key_points_as_indices(logits))

    assert key_points.shape == (2, NUM_LANDMARKS, 2)
    np.testing.assert_allclose(key_points[0], [[1, 2], [2, 3], [3, 4], [4, 5]], atol=1e-4)
    np.testing.assert_allclose(key_points[1], [[5, 7]] * NUM_LANDMARKS, atol=1e-4)


class TestOverlay:
    """``overlay_labels_on_image`` draws the landmarks and their connections."""

    @pytest.fixture
    def label(self, rng):
        return rng.random((NETWORK_SIZE, NETWORK_SIZE, NUM_LANDMARKS)).astype("float32")

    @pytest.mark.parametrize(
        "image_shape",
        [
            (NETWORK_SIZE, NETWORK_SIZE),
            (NETWORK_SIZE, NETWORK_SIZE, 1),
            (NETWORK_SIZE, NETWORK_SIZE, 3),
        ],
        ids=["grayscale", "single_channel", "rgb"],
    )
    def test_always_returns_a_normalized_rgb_image(self, model, label, image_shape):
        image = np.full(image_shape, 0.5, dtype="float32")

        out = model.overlay_labels_on_image(image, label)

        assert out.shape == (NETWORK_SIZE, NETWORK_SIZE, 3)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_leaves_the_image_untouched_where_nothing_is_drawn(self, model):
        """Only the landmark heatmaps and connecting lines are blended in."""
        image = np.full((NETWORK_SIZE, NETWORK_SIZE), 0.5, dtype="float32")
        label = np.zeros((NETWORK_SIZE, NETWORK_SIZE, NUM_LANDMARKS), dtype="float32")
        label[10, 10, :] = 1.0  # all four landmarks in the same spot

        out = model.overlay_labels_on_image(image, label)

        assert out[100, 100].tolist() == [0.5, 0.5, 0.5]
        assert not np.allclose(out[10, 10], 0.5)

    def test_reports_a_missing_opencv(self, model, monkeypatch, label):
        """OpenCV is an optional dependency, so its absence gets an install hint."""
        monkeypatch.setitem(sys.modules, "cv2", None)
        image = np.zeros((NETWORK_SIZE, NETWORK_SIZE), dtype="float32")

        with pytest.raises(ImportError, match="OpenCV is required"):
            model.overlay_labels_on_image(image, label)


def test_visualize_logits_keeps_the_input_resolution(model, rng):
    """The visualization is produced at 224x224 and resized back to the input size."""
    images = rng.random((2, 32, 40, 1)).astype("float32") * 255
    logits = rng.random((2, 32, 40, NUM_LANDMARKS)).astype("float32")

    out = model.visualize_logits(images, logits)

    assert out.shape == (2, 32, 40, 3)
    assert float(ops.min(out)) >= 0.0 and float(ops.max(out)) <= 1.0
