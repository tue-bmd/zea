"""Tests for the InversionNet speed-of-sound reconstruction model."""

import keras
import numpy as np
import pytest
from keras import ops

from zea.models.inversionnet import OPENPROS_INPUT_SHAPE, InversionNet

# The pretrained architecture is far too large to run in a test, so the tests below use a
# miniature configuration with the same structure: a side-strided encoder level, a regular
# one, a bottleneck that collapses to 1 x 1, and a cropped decoder.
TINY_INPUT_SHAPE = (40, 9, 3)
TINY_KWARGS = {
    "waveform_shape": TINY_INPUT_SHAPE,
    "enc_ch": (2, 2, 2, 3),
    "enc_side": (1, 0),
    "bottle_conv": (5, 5),
    "bottle_deconv": (5, 3),
    "dec_ch": (3, 2, 2),
    "crop": (1, 2, 3, 4),
}
TINY_OUTPUT_SHAPE = (17, 5, 1)
# Trainable parameter count of the original OpenPros checkpoint.
OPENPROS_PARAMS = 20_447_515


@pytest.fixture
def tiny_model():
    """Miniature InversionNet with randomly initialized weights."""
    return InversionNet(**TINY_KWARGS)


def test_call_returns_a_single_channel_map(tiny_model, rng):
    """The network maps waveforms to a one-channel image in ``[-1, 1]``."""
    x = rng.random((2, *TINY_INPUT_SHAPE)).astype("float32")
    out = tiny_model(x)
    assert out.shape == (2, *TINY_OUTPUT_SHAPE)
    assert float(ops.min(out)) >= -1.0
    assert float(ops.max(out)) <= 1.0


@pytest.mark.parametrize(
    ("shape", "match"),
    [
        ((*TINY_INPUT_SHAPE,), "4 dimensions"),
        ((1, 40, 9, 5), r"shape \(batch, 40, 9, 3\)"),
    ],
)
def test_call_rejects_bad_input(tiny_model, shape, match):
    """Only 4D input with the configured number of channels is accepted."""
    with pytest.raises(ValueError, match=match):
        tiny_model(np.zeros(shape, dtype="float32"))


def test_mismatched_encoder_arguments_raise():
    """``enc_side`` covers the levels between the input and bottleneck convolutions."""
    with pytest.raises(ValueError, match="two more entries"):
        InversionNet(enc_ch=(6, 6, 7), enc_side=(1, 0))


def test_config_round_trip(tiny_model, rng):
    """A model rebuilt from its config has the same architecture."""
    rebuilt = InversionNet.from_config(tiny_model.get_config())
    x = rng.random((1, *TINY_INPUT_SHAPE)).astype("float32")
    assert rebuilt(x).shape == tiny_model(x).shape


def test_default_architecture_matches_the_openpros_checkpoint():
    """The default arguments reproduce the released OpenPros network.

    Built symbolically, so the 20M-parameter network is never actually run.
    """
    model = InversionNet()
    output = model(keras.Input(batch_shape=(None, *OPENPROS_INPUT_SHAPE)))
    assert tuple(output.shape[1:]) == (401, 161, 1)
    assert sum(np.prod(w.shape) for w in model.trainable_weights) == OPENPROS_PARAMS


def test_weights_round_trip_through_a_preset(tiny_model, tmp_path, rng):
    """Saving to a preset directory and loading it back preserves the weights."""
    x = rng.random((1, *TINY_INPUT_SHAPE)).astype("float32")
    expected = ops.convert_to_numpy(tiny_model(x))

    tiny_model.save_to_preset(str(tmp_path))
    reloaded = InversionNet.from_preset(str(tmp_path))

    assert np.allclose(ops.convert_to_numpy(reloaded(x)), expected, atol=1e-6)


def test_fit_updates_batch_norm_like_pytorch(tiny_model, rng):
    """Fine-tuning updates the running statistics, with the original's weighting.

    Goes through ``fit()`` rather than a direct ``training=True`` call on purpose:
    the moving statistics only update if ``call()`` forwards ``training`` down to the
    normalization layers, and the direct call papers over a model that does not.

    Keras' ``momentum`` is the weight of the *existing* statistic where PyTorch's is
    the weight of the *incoming* batch, so PyTorch's default of ``0.1`` is Keras'
    ``0.9``. Getting that wrong is invisible at inference time and shows up only as
    running statistics that adapt 10x too slowly.
    """
    x = rng.random((4, *TINY_INPUT_SHAPE)).astype("float32")
    y = np.zeros((4, *TINY_OUTPUT_SHAPE), dtype="float32")
    block = tiny_model.encoder[0]
    tiny_model(x)  # build

    # Read the statistic the normalization layer will see, before the optimizer step
    # of the single training batch moves the convolution that produces it.
    batch_mean = ops.convert_to_numpy(block.conv(block.pad(x))).mean(axis=(0, 1, 2))

    tiny_model.compile(optimizer="sgd", loss="mse")
    tiny_model.fit(x, y, epochs=1, batch_size=4, verbose=0)

    # The moving mean starts at zero, so one update leaves it at 0.1 * the batch mean.
    assert np.allclose(ops.convert_to_numpy(block.norm.moving_mean), 0.1 * batch_mean, atol=1e-5)


def test_architecture_that_does_not_collapse_to_a_point_is_rejected():
    """A ``waveform_shape`` the encoder cannot reduce to 1 x 1 fails at construction.

    Left unchecked, the bottleneck convolution passes the leftover extent on and the
    decoder upsamples it into a wrongly sized map, with nothing raised anywhere.
    """
    # Twice the time samples the rest of TINY_KWARGS is sized for.
    kwargs = {**TINY_KWARGS, "waveform_shape": (80, 9, 3)}

    with pytest.raises(ValueError, match=r"collapse to 1 x 1.*bottle_conv=\(10, 5\)"):
        InversionNet(**kwargs)

    # The size the error suggests is the one that works.
    model = InversionNet(**{**kwargs, "bottle_conv": (10, 5)})
    output = model(keras.Input(batch_shape=(None, 80, 9, 3)))
    assert tuple(output.shape[1:]) == TINY_OUTPUT_SHAPE
