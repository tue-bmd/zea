"""Tests for the DeepLabV3+ segmentation backbone.

The backbone wraps a ResNet50, so it is built once at a small input size and
shared between the tests. Its building blocks are checked on their own, which
keeps their shape arithmetic readable without another backbone build.
"""

import numpy as np
import pytest

from zea.models.deeplabv3 import (
    DeeplabV3Plus,
    DilatedSpatialPyramidPooling,
    convolution_block,
)

IMAGE_SIZE = 64
NUM_CLASSES = 4


@pytest.fixture(scope="module")
def model():
    """A DeepLabV3+ at a small input size (module-scoped: ResNet50 is slow to build)."""
    return DeeplabV3Plus((IMAGE_SIZE, IMAGE_SIZE, 1), NUM_CLASSES)


@pytest.mark.parametrize(
    ("num_filters", "kernel_size", "dilation_rate"),
    [(256, 3, 1), (48, 1, 1), (256, 3, 6)],
)
def test_convolution_block_keeps_the_spatial_size(rng, num_filters, kernel_size, dilation_rate):
    """``padding="same"`` means only the channel count changes."""
    x = rng.standard_normal((1, 16, 16, 8)).astype("float32")

    out = convolution_block(
        x, num_filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate
    )

    assert out.shape == (1, 16, 16, num_filters)
    # Conv -> BatchNorm -> ReLU, so nothing negative survives.
    assert np.all(np.asarray(out) >= 0)


def test_dilated_spatial_pyramid_pooling_merges_five_branches(rng):
    """ASPP concatenates five branches and projects them back to 256 channels."""
    x = rng.standard_normal((1, 8, 8, 16)).astype("float32")

    out = DilatedSpatialPyramidPooling(x)

    assert out.shape == (1, 8, 8, 256)


def test_output_shape(model):
    """The segmentation map has one channel per class at the input resolution."""
    assert model.input_shape == (None, IMAGE_SIZE, IMAGE_SIZE, 1)
    assert model.output_shape == (None, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES)


def test_forward_pass(model, rng):
    """A grayscale image is repeated to three channels for the ResNet50 backbone."""
    x = rng.standard_normal((2, IMAGE_SIZE, IMAGE_SIZE, 1)).astype("float32")

    out = model(x)

    assert out.shape == (2, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES)
