"""Tests for the carotid segmentation model.

The architecture is built by zea itself, so it is exercised with random weights;
only the trained weights live in the preset.
"""

import numpy as np
import pytest
from keras import ops

from zea.models.carotid_segmenter import INFERENCE_SIZE, CarotidSegmenter
from zea.models.presets import carotid_segmenter_presets


@pytest.fixture(scope="module")
def model():
    """Carotid segmenter with random weights (module-scoped: building it is the slow part)."""
    return CarotidSegmenter()


def test_network_matches_the_published_parameter_count(model):
    """The architecture still matches the preset the trained weights were saved from."""
    expected = carotid_segmenter_presets["carotid-segmenter"]["metadata"]["params"]
    assert model.network.count_params() == expected


def test_network_input_and_output_shapes(model):
    """The U-Net maps a 256x256 grayscale image onto a single-channel mask."""
    assert model.network.input_shape == (None, INFERENCE_SIZE, INFERENCE_SIZE, 1)
    assert model.network.output_shape == (None, INFERENCE_SIZE, INFERENCE_SIZE, 1)


def test_mask_keeps_the_input_resolution(model, rng):
    """Inference runs at 256x256, but the mask comes back at the input resolution."""
    x = rng.random((1, 64, 96, 1)).astype("float32")

    out = model(x)

    assert out.shape == (1, 64, 96, 1)


def test_mask_is_a_probability(model, rng):
    """The final layer is a sigmoid, so the mask is in [0, 1]."""
    out = model(rng.random((1, 32, 32, 1)).astype("float32"))

    assert float(ops.min(out)) >= 0.0
    assert float(ops.max(out)) <= 1.0


@pytest.mark.parametrize(
    ("shape", "match"),
    [
        ((INFERENCE_SIZE, INFERENCE_SIZE, 1), "4 dimensions"),
        ((1, INFERENCE_SIZE, INFERENCE_SIZE, 3), "1 channel"),
    ],
)
def test_rejects_bad_input(model, shape, match):
    """Only 4D single-channel input is accepted."""
    with pytest.raises(AssertionError, match=match):
        model(np.zeros(shape, dtype="float32"))


def test_config_roundtrip(model):
    """``input_shape`` and ``input_range`` survive a config roundtrip."""
    config = model.get_config()
    assert config["input_shape"] == (INFERENCE_SIZE, INFERENCE_SIZE, 1)
    assert config["input_range"] == (0, 1)

    restored = CarotidSegmenter.from_config(config)
    assert restored.get_config() == config
