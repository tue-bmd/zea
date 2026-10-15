"""Tests for the Diffusion Transformer (DiT) backend.

The point of this module is that DiT is a drop-in replacement for the
time-conditional UNet, so the tests are about the shape contract and the
``adaLN-Zero`` initialization rather than about the transformer internals.
"""

import numpy as np
import pytest
from keras import ops

from zea.models.dit import (
    DiTTimeConditional,
    Unpatchify,
    get_time_conditional_dit_network,
    modulate,
)

IMAGE_SHAPE = (16, 16, 1)
BATCH_SIZE = 2
TINY_DIT = dict(patch_size=4, hidden_size=16, depth=2, num_heads=2, embedding_dims=8)


@pytest.fixture
def network():
    """A small DiT network."""
    return get_time_conditional_dit_network(IMAGE_SHAPE, **TINY_DIT)


def test_modulate_applies_a_per_token_shift_and_scale():
    """``x * (1 + scale) + shift``, broadcast over the token axis."""
    x = np.ones((1, 3, 2), dtype="float32")
    scale = np.array([[1.0, 0.0]], dtype="float32")
    shift = np.array([[0.0, 5.0]], dtype="float32")

    out = ops.convert_to_numpy(modulate(x, shift, scale))

    np.testing.assert_allclose(out, np.tile([[2.0, 6.0]], (1, 3, 1)))


def test_unpatchify_reassembles_patches_in_row_major_order():
    """Tokens come back as image patches, left to right and top to bottom."""
    grid, patch, channels = 2, 2, 1
    tokens = np.arange(grid * grid, dtype="float32").reshape(1, grid * grid, 1)
    tokens = np.tile(tokens, (1, 1, patch * patch * channels))

    image = ops.convert_to_numpy(Unpatchify(grid, grid, patch, channels)(tokens))

    assert image.shape == (1, 4, 4, 1)
    # Each 2x2 patch is filled with its own token index.
    np.testing.assert_array_equal(image[0, :2, :2, 0], np.zeros((2, 2)))
    np.testing.assert_array_equal(image[0, :2, 2:, 0], np.ones((2, 2)))
    np.testing.assert_array_equal(image[0, 2:, :2, 0], np.full((2, 2), 2.0))
    np.testing.assert_array_equal(image[0, 2:, 2:, 0], np.full((2, 2), 3.0))


def test_network_matches_the_unet_call_signature(network, rng):
    """``[noisy_images, time]`` in, an image of the same shape out."""
    x = rng.standard_normal((BATCH_SIZE, *IMAGE_SHAPE)).astype("float32")
    t = rng.random((BATCH_SIZE, 1, 1, 1)).astype("float32")

    assert network([x, t]).shape == (BATCH_SIZE, *IMAGE_SHAPE)


def test_untrained_network_predicts_zeros(network, rng):
    """adaLN-Zero: every block starts as the identity and the head at zero."""
    x = rng.standard_normal((1, *IMAGE_SHAPE)).astype("float32")
    t = rng.random((1, 1, 1, 1)).astype("float32")

    np.testing.assert_allclose(ops.convert_to_numpy(network([x, t])), 0.0, atol=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"image_shape": (16, 16)}, "must be a tuple"),
        ({"patch_size": 5}, "divisible by"),
        ({"num_heads": 5}, "divisible by num_heads"),
        ({"embedding_dims": 9}, "must be even"),
    ],
)
def test_rejects_inconsistent_hyperparameters(kwargs, match):
    """The size constraints between patches, heads and embeddings are checked upfront."""
    settings = {"image_shape": IMAGE_SHAPE, **TINY_DIT, **kwargs}
    with pytest.raises(AssertionError, match=match):
        get_time_conditional_dit_network(**settings)


class TestDiTTimeConditional:
    """The registered model wrapping the network."""

    @pytest.fixture
    def model(self):
        return DiTTimeConditional(image_shape=IMAGE_SHAPE, **TINY_DIT)

    def test_output_shape(self, model, rng):
        x = rng.standard_normal((BATCH_SIZE, *IMAGE_SHAPE)).astype("float32")
        t = rng.random((BATCH_SIZE, 1, 1, 1)).astype("float32")

        assert model([x, t]).shape == (BATCH_SIZE, *IMAGE_SHAPE)

    def test_config_roundtrip(self, model):
        config = model.get_config()
        restored = DiTTimeConditional.from_config(config)

        assert restored.get_config() == config
        assert config["patch_size"] == TINY_DIT["patch_size"]
        assert config["depth"] == TINY_DIT["depth"]
