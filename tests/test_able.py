"""Tests for the ABLE (Adaptive Beamforming by Deep LEarning) model."""

import numpy as np
import pytest

from . import DEFAULT_TEST_SEED


@pytest.fixture
def able_model():
    """Return a small ABLE model (2 latent layers, dim 8)."""
    from zea.models.able import ABLE

    return ABLE(latent_dim=8, n_latent_layers=2, kernel_size=1)


def test_able_rf_output_shape(able_model):
    """ABLE preserves shape for RF (single-channel) input."""
    n_tx, n_pix, n_el = 3, 16, 8
    x = (
        np.random.default_rng(DEFAULT_TEST_SEED)
        .standard_normal((n_tx, n_pix, n_el))
        .astype(np.float32)
    )
    y = able_model(x)
    assert y.shape == x.shape


def test_able_iq_output_shape(able_model):
    """ABLE preserves shape for IQ (two-channel) input."""
    n_tx, n_pix, n_el, n_ch = 3, 16, 8, 2
    x = (
        np.random.default_rng(DEFAULT_TEST_SEED)
        .standard_normal((n_tx, n_pix, n_el, n_ch))
        .astype(np.float32)
    )
    y = able_model(x)
    assert y.shape == x.shape


def test_able_output_finite(able_model):
    """ABLE output contains no NaN or Inf for random IQ input."""
    import keras

    n_tx, n_pix, n_el, n_ch = 2, 12, 8, 2
    x = (
        np.random.default_rng(DEFAULT_TEST_SEED)
        .standard_normal((n_tx, n_pix, n_el, n_ch))
        .astype(np.float32)
    )
    y = able_model(x)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(y))), "ABLE output contains NaN or Inf"


def test_able_custom_latent_dim():
    """Different latent_dim values produce models with different parameter counts."""
    from zea.models.able import ABLE

    n_tx, n_pix, n_el = 2, 8, 4
    x = np.random.randn(n_tx, n_pix, n_el).astype(np.float32)

    m8 = ABLE(latent_dim=8, n_latent_layers=2)
    m16 = ABLE(latent_dim=16, n_latent_layers=2)
    m8(x)
    m16(x)

    assert m8.count_params() < m16.count_params(), "Larger latent_dim should yield more parameters"


def test_able_latent_layers_override():
    """Explicit latent_layers list overrides latent_dim and n_latent_layers."""
    from zea.models.able import ABLE

    n_tx, n_pix, n_el = 2, 8, 4
    x = np.random.randn(n_tx, n_pix, n_el).astype(np.float32)

    m = ABLE(n_latent_layers=2, latent_layers=[12, 12])
    m(x)
    # layer_dims: [n_el, 12, 12, n_el] -> 4 Conv2D layers
    assert len(m._able_layers) == 4


def test_able_invalid_kernel_size():
    """Non-1x1 kernel sizes raise ValueError at build/call time."""
    from zea.models.able import ABLE

    model = ABLE(kernel_size=3)
    x = np.random.randn(2, 8, 4).astype(np.float32)
    with pytest.raises(ValueError, match="kernel_size"):
        model(x)


def test_able_stack_unstack_rf(able_model):
    """stack_channels -> unstack_channels is a round-trip for RF data."""
    import keras

    n_pix, n_el = 16, 8
    x = keras.ops.convert_to_tensor(
        np.random.default_rng(DEFAULT_TEST_SEED).standard_normal((n_pix, n_el)).astype(np.float32)
    )
    stacked, meta = able_model.stack_channels(x, able_model.axis)
    recovered = able_model.unstack_channels(stacked, meta)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(x), keras.ops.convert_to_numpy(recovered), atol=1e-6
    )


def test_able_stack_unstack_iq(able_model):
    """stack_channels -> unstack_channels is a round-trip for IQ data."""
    import keras

    n_pix, n_el, n_ch = 16, 8, 2
    x = keras.ops.convert_to_tensor(
        np.random.default_rng(DEFAULT_TEST_SEED)
        .standard_normal((n_pix, n_el, n_ch))
        .astype(np.float32)
    )
    stacked, meta = able_model.stack_channels(x, able_model.axis)
    recovered = able_model.unstack_channels(stacked, meta)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(x), keras.ops.convert_to_numpy(recovered), atol=1e-6
    )


def test_able_beamform_output_shape():
    """ABLEBeamform reduces TOF-corrected data to a beamformed image."""
    import keras

    from zea.models.able import ABLE, ABLEBeamform

    n_tx, n_pix, n_el, n_ch = 3, 16, 8, 2
    data = keras.ops.convert_to_tensor(
        np.random.default_rng(DEFAULT_TEST_SEED)
        .standard_normal((1, n_tx, n_pix, n_el, n_ch))
        .astype(np.float32)
    )
    operation = ABLEBeamform(model=ABLE(latent_dim=8, n_latent_layers=2), with_batch_dim=True)
    out = operation(data=data)["data"]
    assert out.shape == (1, n_pix, n_ch)


def test_able_beamform_builds_default_model():
    """Omitting the model creates one, built from the shape of the first input."""
    import keras

    from zea.models.able import ABLEBeamform

    operation = ABLEBeamform(with_batch_dim=False)
    assert not operation.model.built

    n_tx, n_pix, n_el, n_ch = 2, 8, 4, 1
    data = keras.ops.convert_to_tensor(
        np.random.default_rng(DEFAULT_TEST_SEED)
        .standard_normal((n_tx, n_pix, n_el, n_ch))
        .astype(np.float32)
    )
    out = operation(data=data)["data"]
    assert operation.model.built
    assert out.shape == (n_pix, n_ch)


def test_able_beamform_is_registered_as_beamformer():
    """The operation plugs into `zea.ops.Beamform` under the name "able"."""
    from zea.models.able import ABLE, ABLEBeamform
    from zea.ops import Beamform, beamformer_registry

    assert beamformer_registry["able"] is ABLEBeamform

    model = ABLE(latent_dim=8)
    beamform = Beamform(beamformer="able", model=model, num_patches=2)
    operations = beamform.operations[0].operations
    assert isinstance(operations[-1], ABLEBeamform)
    assert operations[-1].model is model


def test_able_beamform_not_serialized_into_config():
    """The trained model is left out of the pipeline config, but stays on the operation."""
    from zea.models.able import ABLE
    from zea.ops import Beamform

    model = ABLE(latent_dim=8)
    beamform = Beamform(beamformer="able", model=model, num_patches=2)
    config = beamform.get_dict()
    assert config["params"]["beamformer"] == "able"
    assert "model" not in config["params"]
    assert beamform.operations[0].operations[-1].model is model
