"""Tests for the unfolded convolutional ISTA (LISTA) model."""

import numpy as np
import pytest

from zea.models.lista import LISTA, Prox

from .. import backend_equality_check

BATCH_SIZE = 2


class TestLISTA:
    """Tests for the unfolded convolutional ISTA model."""

    def test_output_shape_matches_input(self, rng):
        """The model maps an image onto an image of the same shape."""
        model = LISTA(input_shape=(16, 16, 1), folds=3)
        x = rng.random((BATCH_SIZE, 16, 16, 1)).astype("float32")
        assert model(x).shape == (BATCH_SIZE, 16, 16, 1)

    def test_upsampling_scales_the_output(self, rng):
        """``upsampling`` scales the spatial dimensions of the output."""
        model = LISTA(input_shape=(8, 8, 1), folds=2, upsampling=2)
        x = rng.random((1, 8, 8, 1)).astype("float32")
        assert model(x).shape == (1, 16, 16, 1)

    def test_folds_add_learned_parameters(self):
        """Every extra fold adds a proximal threshold and two convolutions."""
        few = LISTA(input_shape=(8, 8, 1), folds=2)
        many = LISTA(input_shape=(8, 8, 1), folds=4)
        assert many.network.count_params() > few.network.count_params()

        # One Prox layer per fold (the final one included)
        prox_layers = [layer for layer in many.network.layers if isinstance(layer, Prox)]
        assert len(prox_layers) == 4

    def test_final_activation_is_applied(self, rng):
        """The final activation is applied to the output."""
        model = LISTA(input_shape=(8, 8, 1), folds=2, activation="relu")
        x = rng.standard_normal((1, 8, 8, 1)).astype("float32")
        assert np.all(np.asarray(model(x)) >= 0)

    def test_config_roundtrip(self, rng):
        """The model can be rebuilt from its config."""
        model = LISTA(input_shape=(8, 8, 2), folds=3, filters=4, kernel_size=3)
        restored = LISTA.from_config(model.get_config())

        assert restored.get_config() == model.get_config()
        x = rng.random((1, 8, 8, 2)).astype("float32")
        assert restored(x).shape == model(x).shape

    def test_invalid_input_shape(self):
        """A non-image input shape is rejected."""
        with pytest.raises(AssertionError, match="input_shape"):
            LISTA(input_shape=(8, 8), folds=2)


@backend_equality_check(decimal=5)
def test_prox_soft_thresholds_inputs():
    """The proximal layer soft-thresholds its input by softplus(alpha)."""
    import keras

    layer = Prox()
    layer.build((None, 5))
    layer.alpha.assign(np.zeros((1, 1), dtype="float32"))
    threshold = float(np.log(2.0))  # softplus(0)

    x = np.array([[-2.0, -0.1, 0.0, 0.1, 2.0]], dtype="float32")
    out = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x)))

    expected = np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    # Values below the threshold are zeroed out
    assert np.all(out[0, 1:4] == 0)
    return out
