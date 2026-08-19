"""Tests for the hierarchical VAE.

The published preset is a 256x256 nine-stage model, which is far too large to
build in a unit test. The architecture is fully described by the ``args.pkl``
that ships with the preset, so these tests build the same architecture at 4x4
with three stages instead: every code path is the same, only the tensors are
small. :class:`TinyParameters` is the only thing standing between this file and
the real preset.
"""

import pickle
import types

import keras
import numpy as np
import pytest
from keras import ops

import zea.models.hvae as hvae_module
from zea.func import vmap
from zea.models.hvae import SUPPORTED_VERSIONS, HierarchicalVAE
from zea.models.hvae.model import VAE, Block, PoolLayer
from zea.models.hvae.utils import (
    DiscMixLogistic,
    GaussianAnalyticalKL,
    GradientNorms,
    Parameters,
    SoftPlus,
    cone_loss_mask,
)

from .. import DEFAULT_TEST_SEED

IMAGE_SIZE = 4
NUM_STAGES = 3
CHANNELS = 3  # the "cifar10" dataset context is RGB
NUM_MIXTURES = 2
BATCH_SIZE = 2


def tiny_args(**overrides):
    """The ``args`` namespace an HVAE preset pickles, filled in with small values."""
    args = dict(
        save=False,
        save_dir=".",
        dataset="cifar10",
        batch_size_div=1,
        batch_size=2,
        jit=False,
        gpu=0,
        b_act="gelu",
        p_act="gelu",
        init_zeros=True,
        use_spatial_attention=False,
        use_depthwise_attention=False,
        query_width=0,
        num_queries=1,
        block_gn=True,
        depthwise=False,
        num_output_mixtures=NUM_MIXTURES,
        gradient_smoothing=0.69314718056,
        gradient_clipnorm=0.0,
        gradient_skipnorm=0.0,
        flow_type="none",
        spectral_norm=False,
        epochs=2,
        early_stopping=1,
        beta=1.0,
        beta_warmup_epochs=0,
        cyclic_beta=False,
        number_cycles=1,
        learning_rate=1e-3,
        learning_rate_end=1e-5,
        lr_warmup_epochs=0,
        weight_decay=0.0,
        use_ema=False,
        optimizer="adamax",
        scheduler="none",
        increase_kernelsize=False,
        z_width=[2],
        num_flows=[0],
        flow_in_ch=[0],
        num_ortho_vecs=[0],
        convsylv_channels=[0],
        convsylv_flows_per_stage=[0],
        convsylv_splitfirst=[False],
        convsylv_stage_limit=[0],
        stage_in_width=[8],
        enc_middle_width=[8],
        enc_num_blocks=[1],
        s_a_width=[0],
        dec_middle_width=[8],
        dec_num_blocks=[1],
        output_blocks=1,
        z_out_width=8,
        z_out_middle_width=8,
    )
    args.update(overrides)
    return types.SimpleNamespace(**args)


class TinyParameters(Parameters):
    """The real parameters, shrunk to a 4x4 image with three stages."""

    def add_dataset_context(self):
        super().add_dataset_context()
        self.enc_input_size = [4, 2, 1]
        self.dec_input_size = [1, 2, 4]
        self.num_stages = len(self.enc_input_size)
        # The reconstruction loss weights pixels by their cone area, at a mask size
        # that is baked in for the real 256x256 model.
        self.loss_fn.cone_loss_mask = cone_loss_mask(
            batch_size=1, r_max=IMAGE_SIZE, angle=0.7854, shape=(IMAGE_SIZE, IMAGE_SIZE)
        )


@pytest.fixture(scope="module")
def tiny_preset(tmp_path_factory):
    """A local HVAE preset: a config plus the pickled architecture arguments."""
    preset_dir = tmp_path_factory.mktemp("hvae-preset")
    (preset_dir / "config.json").write_text(
        '{"module": "zea.models.hvae", "class_name": "HierarchicalVAE", '
        '"registered_name": "HierarchicalVAE", "config": {}, "build_config": null}'
    )
    with open(preset_dir / "args.pkl", "wb") as file:
        pickle.dump(tiny_args(), file)
    return str(preset_dir)


@pytest.fixture(scope="module")
def model(tiny_preset):
    """A HierarchicalVAE built from the tiny preset (module-scoped: building is slow)."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(hvae_module, "Parameters", TinyParameters)
        yield HierarchicalVAE.from_preset(tiny_preset, load_weights=False)


@pytest.fixture
def images(rng):
    """A batch of images in the [-1, 1] range the model works in."""
    return rng.uniform(-1, 1, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)).astype("float32")


@pytest.fixture
def image(images):
    """A single image, which is what posterior sampling conditions on."""
    return images[0]


# ---------------------------------------------------------------- zea.models.hvae.utils


class TestSoftPlus:
    """Softplus with a smoothing factor and a floor."""

    def test_matches_the_closed_form(self):
        layer = SoftPlus(gradient_smoothing=1.0)
        x = np.array([-2.0, 0.0, 3.0], dtype="float32")

        out = ops.convert_to_numpy(layer(x))

        np.testing.assert_allclose(out, np.log1p(np.exp(x)), rtol=1e-5)

    def test_smoothing_sharpens_the_transition(self):
        """A larger beta makes softplus hug ``max(x, 0)`` more closely."""
        x = np.array([0.5], dtype="float32")

        smooth = float(ops.convert_to_numpy(SoftPlus(1.0)(x))[0])
        sharp = float(ops.convert_to_numpy(SoftPlus(10.0)(x))[0])

        assert 0.5 < sharp < smooth

    def test_output_is_floored(self):
        layer = SoftPlus(gradient_smoothing=1.0, min=0.5)

        assert float(ops.convert_to_numpy(layer(np.array([-50.0], dtype="float32")))[0]) == 0.5

    def test_smoothing_must_be_positive(self):
        with pytest.raises(AssertionError, match="gradient_smoothing"):
            SoftPlus(gradient_smoothing=0)


class TestGaussianAnalyticalKL:
    """Closed-form KL between two diagonal Gaussians."""

    def test_identical_distributions_have_zero_divergence(self):
        mu = np.array([[0.5, -1.0]], dtype="float32")
        std = np.array([[2.0, 0.5]], dtype="float32")

        kl = ops.convert_to_numpy(GaussianAnalyticalKL().call(mu, std, mu, std))

        np.testing.assert_allclose(kl, 0.0, atol=1e-6)

    def test_matches_the_closed_form(self):
        q_mu, q_std = np.float32(1.0), np.float32(2.0)
        p_mu, p_std = np.float32(0.0), np.float32(1.0)

        kl = float(ops.convert_to_numpy(GaussianAnalyticalKL().call(q_mu, q_std, p_mu, p_std)))

        expected = np.log(p_std / q_std) + (q_std**2 + (q_mu - p_mu) ** 2) / (2 * p_std**2) - 0.5
        np.testing.assert_allclose(kl, expected, rtol=1e-5)


class TestConeLossMask:
    """The mask that reweights a polar-domain loss as if it were cartesian."""

    def test_shape_and_mean(self):
        mask = ops.convert_to_numpy(cone_loss_mask(1, r_max=32, angle=0.7854, shape=(16, 8)))

        assert mask.shape == (1, 16, 8)
        np.testing.assert_allclose(mask.mean(), 1.0, rtol=1e-5)

    def test_weights_grow_with_depth(self):
        """Rows further from the apex cover more area, so they weigh more."""
        mask = ops.convert_to_numpy(cone_loss_mask(1, r_max=32, angle=0.7854, shape=(16, 8)))

        assert np.all(np.diff(mask[0, :, 0]) > 0)

    def test_every_column_of_a_row_weighs_the_same(self):
        mask = ops.convert_to_numpy(cone_loss_mask(1, r_max=32, angle=0.7854, shape=(16, 8)))

        np.testing.assert_allclose(mask[0], np.broadcast_to(mask[0, :, :1], (16, 8)), rtol=1e-6)


class TestDiscMixLogistic:
    """The discretized mixture-of-logistics reconstruction loss."""

    @pytest.fixture
    def loss_fn(self):
        loss = DiscMixLogistic(num_bits=8, num_mixtures=NUM_MIXTURES, num_channels=CHANNELS)
        loss.cone_loss_mask = cone_loss_mask(
            1, r_max=IMAGE_SIZE, angle=0.7854, shape=(IMAGE_SIZE, IMAGE_SIZE)
        )
        return loss

    def test_returns_one_negative_log_likelihood_per_image(self, loss_fn, rng):
        targets = rng.uniform(-1, 1, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
        logits = rng.standard_normal(
            (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_MIXTURES * (3 * CHANNELS + 1))
        )

        loss = ops.convert_to_numpy(
            loss_fn.call(targets.astype("float32"), logits.astype("float32"))
        )

        assert loss.shape == (BATCH_SIZE,)
        assert np.all(loss > 0)

    def test_a_better_fit_costs_less(self, loss_fn, rng):
        """Logits whose means sit on the target give a lower loss than ones that do not."""
        targets = np.full((1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), 0.5, dtype="float32")
        shape = (1, IMAGE_SIZE, IMAGE_SIZE, NUM_MIXTURES * (3 * CHANNELS + 1))
        on_target = np.full(shape, 0.5, dtype="float32")
        off_target = np.full(shape, -0.9, dtype="float32")

        good = float(ops.convert_to_numpy(loss_fn.call(targets, on_target))[0])
        bad = float(ops.convert_to_numpy(loss_fn.call(targets, off_target))[0])

        assert good < bad

    def test_a_mask_restricts_the_loss_to_the_measured_pixels(self, loss_fn, rng):
        """Masked-out channels do not contribute; the rest is rescaled to compensate."""
        targets = rng.uniform(-1, 1, (1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)).astype("float32")
        logits = rng.standard_normal(
            (1, IMAGE_SIZE, IMAGE_SIZE, NUM_MIXTURES * (3 * CHANNELS + 1))
        ).astype("float32")
        mask = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype="float32")
        mask[:, :, :2] = 1.0

        masked = ops.convert_to_numpy(loss_fn.call(targets, logits, mask=mask))
        full = ops.convert_to_numpy(loss_fn.call(targets, logits))

        assert masked.shape == (1,)
        assert masked[0] != full[0]

    def test_grayscale_targets_skip_the_autoregressive_channel_mixing(self, rng):
        """With one channel there are no cross-channel coefficients to apply."""
        loss = DiscMixLogistic(num_bits=8, num_mixtures=NUM_MIXTURES, num_channels=1)
        loss.cone_loss_mask = cone_loss_mask(
            1, r_max=IMAGE_SIZE, angle=0.7854, shape=(IMAGE_SIZE, IMAGE_SIZE)
        )
        targets = rng.uniform(-1, 1, (1, IMAGE_SIZE, IMAGE_SIZE, 1)).astype("float32")
        logits = rng.standard_normal((1, IMAGE_SIZE, IMAGE_SIZE, NUM_MIXTURES * 4)).astype(
            "float32"
        )

        assert ops.convert_to_numpy(loss.call(targets, logits)).shape == (1,)

    def test_only_rgb_or_grayscale_targets(self, loss_fn, rng):
        """Two-channel targets are rejected; everything else about the call is valid."""
        logits = rng.standard_normal(
            (1, IMAGE_SIZE, IMAGE_SIZE, NUM_MIXTURES * (3 * CHANNELS + 1))
        ).astype("float32")
        good = rng.uniform(-1, 1, (1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)).astype("float32")
        assert loss_fn.call(good, logits) is not None  # same logits, valid target

        two_channel = rng.uniform(-1, 1, (1, IMAGE_SIZE, IMAGE_SIZE, 2)).astype("float32")
        with pytest.raises(AssertionError):
            loss_fn.call(two_channel, logits)


def test_gradient_norms_metric_tracks_the_last_value():
    """The metric reports the most recent gradient norm, not a running mean."""
    metric = GradientNorms()

    metric.update_state(1.0)
    metric.update_state(4.0)

    assert float(ops.convert_to_numpy(metric.result())) == 4.0


class TestParameters:
    """The parameter object an HVAE preset is described by."""

    @pytest.mark.parametrize(
        ("dataset", "input_size", "data_width"),
        [
            ("cifar10", 32, 3),
            ("echonet", 128, 1),
            ("imagenet32", 32, 3),
            ("celeba64", 64, 3),
            ("echonetlvh", 256, 3),
        ],
    )
    def test_dataset_context_sets_the_resolution_pyramid(self, dataset, input_size, data_width):
        """Every supported dataset halves the resolution down to a single pixel."""
        params = Parameters(tiny_args(dataset=dataset))

        assert params.enc_input_size[0] == input_size
        assert params.enc_input_size[-1] == 1
        assert params.data_width == data_width
        assert params.dec_input_size == list(reversed(params.enc_input_size))
        assert params.channels_out == (data_width * 3 + 1) * NUM_MIXTURES

    def test_unknown_dataset_is_rejected(self):
        with pytest.raises(ValueError, match="No valid dataset"):
            Parameters(tiny_args(dataset="not-a-dataset"))

    def test_per_stage_lists_are_broadcast(self):
        """A single value is repeated for every stage; a full list is used as given."""
        params = TinyParameters(tiny_args())

        assert params.zdim == [2] * NUM_STAGES
        assert params.enc_num_blocks == [1] * NUM_STAGES
        assert params.model_depth == sum(params.dec_num_blocks)
        assert params.z_out == (IMAGE_SIZE, IMAGE_SIZE, params.dec_in_width[-1])

    def test_decoder_lists_are_reversed(self):
        """The decoder runs bottom-up, so its per-stage lists mirror the encoder's."""
        args = tiny_args(
            stage_in_width=[8, 16, 32],
            dec_middle_width=[8, 16, 32],
            enc_middle_width=[8, 16, 32],
            enc_num_blocks=[1, 1, 1],
            dec_num_blocks=[1, 2, 3],
        )
        params = TinyParameters(args)

        assert params.enc_in_width == [8, 16, 32]
        assert params.dec_in_width == [32, 16, 8]
        assert params.dec_num_blocks == [3, 2, 1]

    def test_gradient_thresholds_are_normalized(self):
        """Zero means "off": no clipping, and an effectively infinite skip threshold."""
        params = TinyParameters(tiny_args(gradient_clipnorm=0.0, gradient_skipnorm=0.0))

        assert params.gradient_clipnorm is None
        assert params.gradient_skipnorm == 1e9

    def test_optional_arguments_have_defaults(self):
        """Older presets were pickled without these, so they fall back."""
        params = TinyParameters(tiny_args())

        assert params.retrain_encoder is False
        assert params.num_lines == 256

    def test_optional_arguments_are_taken_from_the_preset_when_present(self):
        params = TinyParameters(tiny_args(retrain_encoder=True, num_lines=24))

        assert params.retrain_encoder is True
        assert params.num_lines == 24

    @pytest.mark.parametrize("increase_kernelsize", [False, True])
    def test_kernel_sizes_per_resolution(self, increase_kernelsize):
        """Kernels grow with resolution only when asked for; 1x1 at the coarsest levels."""
        params = TinyParameters(tiny_args(increase_kernelsize=increase_kernelsize))

        assert params.kernelsizes["2"] == 1
        assert params.kernelsizes["256"] == (7 if increase_kernelsize else 3)

    @pytest.mark.parametrize("scheduler", ["none", "exp", "cosd", "cosdr"])
    @pytest.mark.parametrize("optimizer", ["adamax", "adamw", "sgd"])
    def test_get_optimizer(self, scheduler, optimizer):
        params = TinyParameters(tiny_args(scheduler=scheduler, optimizer=optimizer))

        assert isinstance(params.get_optimizer(), keras.optimizers.Optimizer)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"scheduler": "nope"}, "lr scheduler"),
            ({"optimizer": "nope"}, "optimizer"),
        ],
    )
    def test_unknown_optimizer_or_scheduler(self, overrides, match):
        params = TinyParameters(tiny_args(**overrides))

        with pytest.raises(ValueError, match=match):
            params.get_optimizer()

    def test_cyclic_beta_must_fit_in_the_training_run(self):
        with pytest.raises(AssertionError):
            TinyParameters(tiny_args(cyclic_beta=True, beta_warmup_epochs=5, number_cycles=10))

    def test_stage_lists_must_have_one_entry_per_stage(self):
        """A list that is neither one value nor one per stage is caught up front."""
        with pytest.raises(AssertionError):
            TinyParameters(tiny_args(z_width=[2, 4]))


# ---------------------------------------------------------------- zea.models.hvae.model


class TestBuildingBlocks:
    """The two leaf layers of the architecture, which everything else is made of."""

    def test_block_is_residual_when_the_width_is_kept(self):
        block = Block(
            input_size=4,
            in_width=8,
            middle_width=8,
            out_width=8,
            kernelsize=3,
            activation=keras.layers.Activation("gelu"),
            bn=True,
            residual=True,
            zero_last=True,
            model_depth=1,
            depthwise=False,
            use_attention=False,
            attention_width=0,
        )
        block.build()
        x = np.ones((1, 4, 4, 8), dtype="float32")

        # zero_last makes the block start as the identity, so the skip is all there is.
        np.testing.assert_allclose(ops.convert_to_numpy(block.call(x)), x, atol=1e-6)

    def test_block_can_change_the_width(self):
        block = Block(
            input_size=4,
            in_width=8,
            middle_width=8,
            out_width=6,
            kernelsize=3,
            activation=keras.layers.Activation("gelu"),
            bn=False,
            residual=False,
            zero_last=False,
            model_depth=1,
            depthwise=True,
            use_attention=False,
            attention_width=0,
        )
        block.build()

        out = block.call(np.ones((1, 4, 4, 8), dtype="float32"))

        assert out.shape == (1, 4, 4, 6)

    def test_pool_layer_downsamples_for_the_encoder(self):
        pool = PoolLayer(
            input_size=4,
            in_width=8,
            out_width=16,
            pool_activation=keras.layers.Activation("gelu"),
            unpool=False,
        )
        pool.build()

        assert pool.call(np.ones((1, 4, 4, 8), dtype="float32")).shape == (1, 2, 2, 16)

    def test_pool_layer_upsamples_for_the_decoder(self):
        pool = PoolLayer(
            input_size=4,
            in_width=8,
            out_width=8,
            pool_activation=keras.layers.Activation("gelu"),
            unpool=True,
            data_size=8,
        )
        pool.build()

        assert pool.call(np.ones((1, 4, 4, 8), dtype="float32")).shape == (1, 8, 8, 8)

    def test_pool_layer_is_a_no_op_at_the_smallest_resolution(self):
        """There is nothing left to pool at 1x1, and nothing to unpool at full size."""
        encoder_pool = PoolLayer(1, 8, 8, keras.layers.Activation("gelu"), unpool=False)
        decoder_pool = PoolLayer(8, 8, 8, keras.layers.Activation("gelu"), unpool=True, data_size=8)

        assert isinstance(encoder_pool.pool, keras.layers.Identity)
        assert isinstance(decoder_pool.pool, keras.layers.Identity)


class TestVAE:
    """The encoder/decoder pair underneath :class:`HierarchicalVAE`."""

    @pytest.fixture
    def vae(self, model):
        return model.network

    def test_encoder_returns_one_activation_per_stage(self, vae, images):
        activations = vae.encoder(images)

        assert len(activations) == NUM_STAGES
        assert [act.shape[1] for act in activations] == [4, 2, 1]

    def test_call_returns_logits_latents_and_divergences(self, vae, images):
        logits, z_stages, kl_stages = vae.call(images)

        assert logits.shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, vae.params.channels_out)
        assert len(z_stages) == len(kl_stages) == NUM_STAGES

    def test_sample_from_mol_produces_an_image_in_range(self, vae, images):
        logits, _, _ = vae.call(images)

        samples = vae.sample_from_mol(logits)

        assert samples.shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        assert float(ops.min(samples)) >= -1.0
        assert float(ops.max(samples)) <= 1.0

    def test_get_elbo_splits_into_reconstruction_and_kl(self, vae, images):
        logits, _, kl = vae.call(images)

        elbo, recon, kl_total = vae.get_elbo(images, logits, kl)

        assert float(recon) > 0
        assert float(kl_total) >= 0
        np.testing.assert_allclose(float(elbo), float(recon) + float(kl_total), rtol=1e-5)

    def test_decoder_samples_from_the_prior_without_an_input(self, vae):
        logits = vae.decoder.call_uncond(3)

        assert logits.shape == (3, IMAGE_SIZE, IMAGE_SIZE, vae.params.channels_out)

    def test_temperature_zero_removes_the_prior_noise(self, vae):
        """At ``t=0`` the prior sampling is deterministic."""
        first = ops.convert_to_numpy(vae.decoder.call_uncond(1, t=0))
        second = ops.convert_to_numpy(vae.decoder.call_uncond(1, t=0))

        np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-5)

    def test_print_model_summarizes_every_stage(self, vae, capsys):
        vae.print_model()

        printed = capsys.readouterr().out
        assert "------ Encoder ------" in printed
        assert printed.count("stage:") == 2 * NUM_STAGES
        assert f"model_depth:    {vae.params.model_depth}" in printed

    def test_sample_from_mol_of_a_grayscale_model(self, rng):
        """A single-channel model has no cross-channel coefficients to apply."""
        params = TinyParameters(tiny_args(dataset="echonet"))
        vae = VAE(params)  # not built: sampling from logits needs no weights
        logits = rng.standard_normal((1, IMAGE_SIZE, IMAGE_SIZE, params.channels_out)).astype(
            "float32"
        )

        samples = vae.sample_from_mol(logits)

        assert samples.shape == (1, IMAGE_SIZE, IMAGE_SIZE, 1)
        assert float(ops.min(samples)) >= -1.0 and float(ops.max(samples)) <= 1.0

    def test_learned_initial_bias(self, images):
        """With ``init_zeros=False`` the decoder starts from a learned bias instead."""
        params = TinyParameters(tiny_args(init_zeros=False))
        vae = VAE(params)
        vae.build()

        assert vae.decoder.init_bias.shape == (1, 1, 1, params.dec_in_width[0])
        assert vae.decoder.call_uncond(2).shape[0] == 2


# ------------------------------------------------------------------- zea.models.hvae


class TestHierarchicalVAE:
    """The generative-model interface on top of the VAE."""

    def test_version_must_be_supported(self):
        with pytest.raises(AssertionError, match="Unsupported version"):
            HierarchicalVAE(version="not-a-version")

    def test_supported_versions_are_accepted(self):
        for version in SUPPORTED_VERSIONS:
            assert HierarchicalVAE(version=version).version == version

    def test_from_preset_builds_the_architecture_from_args(self, model):
        """Unlike the other models, the HVAE architecture comes out of the preset."""
        assert isinstance(model.network, VAE)
        assert model.depth == NUM_STAGES
        assert model.z_out == (IMAGE_SIZE, IMAGE_SIZE, 8)

    def test_sample(self, model):
        samples = model.sample(n_samples=3)

        assert samples.shape == (3, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        assert float(ops.min(samples)) >= -1.0 and float(ops.max(samples)) <= 1.0

    def test_call_reconstructs_the_input(self, model, images):
        recon, z_samples, kl = model(images)

        assert recon.shape == images.shape
        assert len(z_samples) == len(kl) == NUM_STAGES

    @pytest.mark.parametrize("n_samples", [1, 3])
    def test_posterior_sample_is_led_by_the_sample_axis(self, model, image, n_samples):
        """A single measurement, and one encoder pass shared by every sample."""
        samples = model.posterior_sample(image, n_samples=n_samples)

        assert samples.shape == (n_samples, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

    def test_posterior_sample_accepts_one_measurement_per_sample(self, model, images):
        """A leading axis of ``n_samples`` conditions each sample on its own image,
        which costs an encoder pass each instead of repeating a single one."""
        samples = model.posterior_sample(images, n_samples=BATCH_SIZE)

        assert samples.shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

    def test_posterior_sample_maps_over_a_batch_with_vmap(self, model, images):
        """The batch axis the model no longer takes itself is recovered with vmap."""
        samples = vmap(lambda image: model.posterior_sample(image, n_samples=2))(images)

        assert samples.shape == (BATCH_SIZE, 2, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

    def test_posterior_sample_is_reproducible_given_a_seed(self, model, image):
        """The seed is split down to every block, so it fixes the whole sample."""
        samples = [
            model.posterior_sample(image, n_samples=2, seed=DEFAULT_TEST_SEED) for _ in range(2)
        ]

        assert np.array_equal(*[ops.convert_to_numpy(s) for s in samples])

    def test_log_density_is_the_negative_elbo(self, model, images):
        log_density = model.log_density(images)

        assert float(log_density) < 0  # the ELBO is a positive loss here

    @pytest.mark.parametrize("num_layers", [0.5, 1.0, 1, NUM_STAGES])
    def test_partial_inference(self, model, image, num_layers):
        """Posterior sampling for the top layers, prior sampling below."""
        samples = model.partial_inference(image, num_layers=num_layers, n_samples=2)

        assert samples.shape == (2, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

    def test_partial_inference_accepts_one_measurement_per_sample(self, model, images):
        """With one measurement per sample the top layers are not repeated either."""
        samples = model.partial_inference(images, num_layers=0.5, n_samples=BATCH_SIZE)

        assert samples.shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

    @pytest.mark.parametrize(
        ("num_layers", "error", "match"),
        [
            (0.0, AssertionError, "float must be in"),
            (1.5, AssertionError, "float must be in"),
            (0, AssertionError, "int must be in"),
            (NUM_STAGES + 1, AssertionError, "int must be in"),
            ("half", ValueError, "either a float or an int"),
        ],
    )
    def test_partial_inference_rejects_an_out_of_range_depth(
        self, model, image, num_layers, error, match
    ):
        with pytest.raises(error, match=match):
            model.partial_inference(image, num_layers=num_layers)
