"""Tests for the Speckle2Self speckle reduction model."""

import numpy as np
import pytest
from keras import ops

from zea.models.speckle2self import Speckle2Self

BATCH_SIZE = 2
IMAGE_SHAPE = (512, 512, 1)


@pytest.fixture
def speckle2self_model():
    """Speckle2Self model without pretrained weights."""
    return Speckle2Self()


def test_speckle2self_call_nchw(speckle2self_model, rng):
    """Test Speckle2Self forward pass with (N, 1, H, W) input."""
    x = rng.random((BATCH_SIZE, *IMAGE_SHAPE)).astype("float32")
    out = speckle2self_model(x)
    assert out.shape == (BATCH_SIZE, *IMAGE_SHAPE)


def test_speckle2self_resizes_back_to_the_input_size(speckle2self_model, rng):
    """Inference happens at a fixed size, but the output matches the input again."""
    x = rng.random((1, 64, 96, 1)).astype("float32")
    out = speckle2self_model(x)
    assert out.shape == (1, 64, 96, 1)
    # The model normalizes internally and clips its output to [0, 1].
    assert float(ops.min(out)) >= 0.0
    assert float(ops.max(out)) <= 1.0


@pytest.mark.parametrize(
    ("shape", "match"),
    [
        ((64, 64, 1), "4 dimensions"),
        ((1, 64, 64, 3), "1 channel"),
    ],
)
def test_speckle2self_rejects_bad_input(speckle2self_model, shape, match):
    """Only 4D single-channel input is accepted."""
    with pytest.raises(AssertionError, match=match):
        speckle2self_model(np.zeros(shape, dtype="float32"))
