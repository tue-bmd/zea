"""Tests for the dense (MLP) models."""

import numpy as np
import pytest
from keras import ops

from zea.models.dense import (
    DenseNet,
    DenseTimeConditionalNet,
    get_dense_network,
    get_time_conditional_dense_network,
)

INPUT_DIM = 5
OUTPUT_DIM = 3
WIDTHS = [8, 8]
BATCH_SIZE = 2


class TestDenseNet:
    """A plain feedforward network."""

    @pytest.fixture
    def model(self):
        return DenseNet(input_dim=INPUT_DIM, widths=WIDTHS, output_dim=OUTPUT_DIM)

    def test_output_shape(self, model, rng):
        x = rng.standard_normal((BATCH_SIZE, INPUT_DIM)).astype("float32")

        assert model(x).shape == (BATCH_SIZE, OUTPUT_DIM)

    def test_output_layer_starts_at_zero(self, model, rng):
        """The final layer is zero-initialized, so an untrained net predicts zeros."""
        x = rng.standard_normal((BATCH_SIZE, INPUT_DIM)).astype("float32")

        np.testing.assert_allclose(ops.convert_to_numpy(model(x)), 0.0)

    def test_config_roundtrip(self, model, rng):
        config = model.get_config()
        restored = DenseNet.from_config(config)

        assert restored.get_config() == config
        x = rng.standard_normal((1, INPUT_DIM)).astype("float32")
        assert restored(x).shape == model(x).shape


class TestDenseTimeConditionalNet:
    """The same network, conditioned on a scalar diffusion time."""

    @pytest.fixture
    def model(self):
        return DenseTimeConditionalNet(
            input_dim=INPUT_DIM, widths=WIDTHS, output_dim=OUTPUT_DIM, embedding_dims=8
        )

    def test_output_shape(self, model, rng):
        x = rng.standard_normal((BATCH_SIZE, INPUT_DIM)).astype("float32")
        t = rng.random((BATCH_SIZE, 1)).astype("float32")

        assert model([x, t]).shape == (BATCH_SIZE, OUTPUT_DIM)

    def test_config_roundtrip(self, model, rng):
        config = model.get_config()
        restored = DenseTimeConditionalNet.from_config(config)

        assert restored.get_config() == config
        assert config["embedding_dims"] == 8

    def test_time_embedding_widens_the_first_layer(self):
        """The sinusoidal time embedding is concatenated onto the input."""
        network = get_time_conditional_dense_network(
            INPUT_DIM, WIDTHS, OUTPUT_DIM, embedding_dims=16
        )
        first_dense = next(layer for layer in network.layers if layer.__class__.__name__ == "Dense")

        assert first_dense.input.shape[-1] == INPUT_DIM + 16


def test_networks_are_named_for_introspection():
    """The functional networks carry stable names, which presets rely on."""
    assert get_dense_network(INPUT_DIM, WIDTHS, OUTPUT_DIM).name == "dense_net"
    assert (
        get_time_conditional_dense_network(INPUT_DIM, WIDTHS, OUTPUT_DIM).name
        == "dense_time_conditional_net"
    )
