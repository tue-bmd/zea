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


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"kernel_size": (3, 3)}, "Only kernel_size="),
        ({"kernel_size": 1.5}, "kernel_size must be int, tuple"),
        ({"kernel_size": [1, 1, 1]}, "length must match"),
        ({"latent_layers": 12}, "must be a list/tuple of integers"),
        ({"latent_layers": [8]}, "exactly"),
        ({"latent_layers": [8, 0]}, "positive integers"),
    ],
)
def test_able_rejects_invalid_configuration(kwargs, match):
    """Invalid kernel and layer specifications are rejected when the model builds."""
    from zea.models.able import ABLE

    model = ABLE(n_latent_layers=2, **kwargs)
    x = np.zeros((2, 8, 4), dtype=np.float32)
    with pytest.raises(ValueError, match=match):
        model(x)


def test_able_rejects_unsupported_rank():
    """Only the documented rank-3 and rank-4 inputs are accepted."""
    from zea.models.able import ABLE

    model = ABLE(latent_dim=4)
    with pytest.raises(ValueError, match="rank-3 or rank-4"):
        model(np.zeros((2, 2, 2, 2, 2), dtype=np.float32))


def test_able_config_round_trip(able_model):
    """The constructor arguments survive get_config / from_config."""
    from zea.models.able import ABLE

    able_model(np.zeros((2, 8, 4), dtype=np.float32))
    clone = ABLE.from_config(able_model.get_config())

    assert clone.latent_dim == able_model.latent_dim
    assert clone.n_latent_layers == able_model.n_latent_layers
    assert clone.kernel_size == able_model.kernel_size
    assert clone.latent_layers == able_model.latent_layers


def test_able_stack_unstack_single_channel(able_model):
    """RF data in zea's layout carries a trailing channel axis of one, and survives."""
    import keras

    n_pix, n_el, n_ch = 16, 8, 1
    x = keras.ops.convert_to_tensor(
        np.random.default_rng(DEFAULT_TEST_SEED)
        .standard_normal((n_pix, n_el, n_ch))
        .astype(np.float32)
    )
    stacked, meta = able_model.stack_channels(x, able_model.axis)
    assert not meta["stacked"]
    recovered = able_model.unstack_channels(stacked, meta)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(x), keras.ops.convert_to_numpy(recovered), atol=1e-6
    )


@pytest.mark.parametrize("kernel_size", [(1, 1), [1, 1, 1, 1]])
def test_able_accepts_documented_kernel_forms(kernel_size):
    """A 1x1 tuple, and a per-layer list of them, are both accepted."""
    from zea.models.able import ABLE

    model = ABLE(latent_dim=8, n_latent_layers=2, kernel_size=kernel_size)
    x = np.zeros((2, 8, 4), dtype=np.float32)
    assert model(x).shape == x.shape
    assert model.kernel_sizes == [(1, 1)] * 4


def test_able_stack_channels_passes_through_unsupported_rank(able_model):
    """Ranks the helper does not reshape are handed back untouched, as documented."""
    import keras

    x = keras.ops.convert_to_tensor(np.zeros((2, 4, 8, 2), dtype=np.float32))
    stacked, meta = able_model.stack_channels(x, able_model.axis)
    assert stacked is x
    assert able_model.unstack_channels(stacked, meta) is x
