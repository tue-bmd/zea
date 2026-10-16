"""Tests for UNet architectures."""

import numpy as np
import pytest
from keras import ops

from zea.models.unet import (
    UNet,
    UNetTimeConditional,
    get_time_conditional_unetwork,
    get_unetwork,
)

IMAGE_SHAPE = (32, 32, 1)
BATCH_SIZE = 2

WIDTHS_AND_DEPTHS = [
    ([16, 32], 2),
    ([8, 16, 32], 2),
    ([16, 32], 3),
]


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


def test_time_conditional_unetwork_group_norm(rng):
    """The residual blocks can normalize per group instead of per batch."""
    model = get_time_conditional_unetwork(IMAGE_SHAPE, [32, 64], 2, normalization="group_norm")
    x = rng.standard_normal((BATCH_SIZE, *IMAGE_SHAPE)).astype("float32")
    noise_variances = rng.standard_normal((BATCH_SIZE, 1, 1, 1)).astype("float32")

    assert model([x, noise_variances]).shape == (BATCH_SIZE, *IMAGE_SHAPE)


class TestUNet:
    """The registered model wrapping :func:`get_unetwork`."""

    @pytest.fixture
    def model(self):
        return UNet(input_shape=IMAGE_SHAPE, widths=[16, 32], block_depth=2, input_range=(0, 1))

    def test_output_shape(self, model, rng):
        x = rng.standard_normal((BATCH_SIZE, *IMAGE_SHAPE)).astype("float32")

        assert model(x).shape == (BATCH_SIZE, *IMAGE_SHAPE)

    def test_untrained_model_predicts_zeros(self, model, rng):
        """The output convolution is zero-initialized."""
        x = rng.standard_normal((1, *IMAGE_SHAPE)).astype("float32")

        np.testing.assert_allclose(ops.convert_to_numpy(model(x)), 0.0)

    def test_config_roundtrip(self, model):
        config = model.get_config()
        restored = UNet.from_config(config)

        assert restored.get_config() == config
        assert config["input_shape"] == IMAGE_SHAPE
        assert config["input_range"] == (0, 1)


class TestUNetTimeConditional:
    """The registered model wrapping :func:`get_time_conditional_unetwork`."""

    @pytest.fixture
    def model(self):
        return UNetTimeConditional(
            image_shape=IMAGE_SHAPE,
            widths=[16, 32],
            block_depth=2,
            image_range=(0, 1),
            embedding_dims=16,
        )

    def test_output_shape(self, model, rng):
        x = rng.standard_normal((BATCH_SIZE, *IMAGE_SHAPE)).astype("float32")
        noise_variances = rng.random((BATCH_SIZE, 1, 1, 1)).astype("float32")

        assert model([x, noise_variances]).shape == (BATCH_SIZE, *IMAGE_SHAPE)

    def test_config_roundtrip(self, model):
        config = model.get_config()
        restored = UNetTimeConditional.from_config(config)

        assert restored.get_config() == config
        assert config["embedding_dims"] == 16
