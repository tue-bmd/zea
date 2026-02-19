"""Tests for UNet architectures."""

import numpy as np
import pytest

from zea.models.unet import get_time_conditional_unetwork, get_unetwork

from . import DEFAULT_TEST_SEED

IMAGE_SHAPE = (32, 32, 1)
BATCH_SIZE = 2

WIDTHS_AND_DEPTHS = [
    ([16, 32], 2),
    ([8, 16, 32], 2),
    ([16, 32], 3),
]


@pytest.fixture
def rng():
    """Random number generator for reproducible tests."""
    return np.random.default_rng(DEFAULT_TEST_SEED)


@pytest.fixture(params=WIDTHS_AND_DEPTHS, ids=lambda p: f"w{len(p[0])}_d{p[1]}")
def unet_model(request):
    """Basic UNet model parametrized over widths and block_depth."""
    widths, block_depth = request.param
    return get_unetwork(IMAGE_SHAPE, widths, block_depth)


@pytest.fixture(params=WIDTHS_AND_DEPTHS, ids=lambda p: f"w{len(p[0])}_d{p[1]}")
def time_conditional_unet_model(request):
    """Time-conditional UNet model parametrized over widths and block_depth."""
    widths, block_depth = request.param
    return get_time_conditional_unetwork(IMAGE_SHAPE, widths, block_depth)


def test_unetwork_output_shape(unet_model, rng):
    """Test that the UNet produces the correct output shape."""
    x = rng.standard_normal((BATCH_SIZE, *IMAGE_SHAPE)).astype("float32")
    y = unet_model(x)
    assert y.shape == (BATCH_SIZE, *IMAGE_SHAPE)


def test_unetwork_invalid_image_shape():
    """Test that an invalid image shape raises an error."""
    with pytest.raises(AssertionError, match="image_shape must be a tuple"):
        get_unetwork((32, 32), [16, 32], 2)


def test_time_conditional_unetwork_output_shape(time_conditional_unet_model, rng):
    """Test that the time-conditional UNet produces the correct output shape."""
    x = rng.standard_normal((BATCH_SIZE, *IMAGE_SHAPE)).astype("float32")
    noise_variances = rng.standard_normal((BATCH_SIZE, 1, 1, 1)).astype("float32")
    y = time_conditional_unet_model([x, noise_variances])
    assert y.shape == (BATCH_SIZE, *IMAGE_SHAPE)


def test_time_conditional_unetwork_default_widths():
    """Test that default widths are used when none are provided."""
    model = get_time_conditional_unetwork(IMAGE_SHAPE, widths=None, block_depth=None)
    assert model is not None


def test_time_conditional_unetwork_invalid_embedding_dims():
    """Test that odd embedding_dims raises an error."""
    with pytest.raises(AssertionError, match="embedding_dims must be even"):
        get_time_conditional_unetwork(IMAGE_SHAPE, [16, 32], 2, embedding_dims=33)


def test_time_conditional_unetwork_custom_embedding(rng):
    """Test time-conditional UNet with custom embedding parameters."""
    model = get_time_conditional_unetwork(
        IMAGE_SHAPE,
        [16, 32],
        2,
        embedding_min_frequency=0.5,
        embedding_max_frequency=500.0,
        embedding_dims=16,
    )
    x = rng.standard_normal((BATCH_SIZE, *IMAGE_SHAPE)).astype("float32")
    noise_variances = rng.standard_normal((BATCH_SIZE, 1, 1, 1)).astype("float32")
    y = model([x, noise_variances])
    assert y.shape == (BATCH_SIZE, *IMAGE_SHAPE)
