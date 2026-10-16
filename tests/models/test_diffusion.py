"""Unit tests for the pieces a :class:`~zea.models.diffusion.DiffusionModel` is made of.

End-to-end behaviour (fitting, sampling and posterior sampling) is covered in
``test_generative.py``; this module sticks to the individually testable parts:
how an operator and a guidance strategy are wired up, the noise schedules, the
reverse-diffusion step, and the rank penalties used for dehazing.
"""

import numpy as np
import pytest
from keras import ops

from zea.internal.operators import InpaintingOperator
from zea.models.diffusion import DPS, DiffusionModel, NuclearDiffusion

TINY_DENSE = dict(
    input_shape=(2,),
    network_name="dense_time_conditional",
    network_kwargs={"widths": [8], "output_dim": 2},
)


@pytest.fixture
def model():
    """A tiny 2D diffusion model, without guidance."""
    return DiffusionModel(**TINY_DENSE, operator=None, guidance=None)


class TestNetworkSelection:
    """The backbone is picked by name."""

    def test_dense_network(self, model):
        assert model.network.name == "dense_time_conditional_net"

    def test_dit_network(self):
        diffusion = DiffusionModel(
            input_shape=(16, 16, 1),
            network_name="dit_time_conditional",
            network_kwargs=dict(patch_size=4, hidden_size=16, depth=1, num_heads=2),
            operator=None,
            guidance=None,
        )

        assert diffusion.network.name == "diffusion_transformer"

    def test_dense_network_needs_a_1d_input_shape(self):
        with pytest.raises(AssertionError, match="only supports 1D"):
            DiffusionModel(
                input_shape=(8, 8, 1),
                network_name="dense_time_conditional",
                operator=None,
                guidance=None,
            )

    def test_unknown_network_name(self):
        with pytest.raises(ValueError, match="Invalid network name"):
            DiffusionModel(input_shape=(2,), network_name="not-a-network")


class TestOperatorAndGuidanceWiring:
    """Operator and guidance can be given as a name, a config dict, or an object."""

    def test_by_name(self):
        diffusion = DiffusionModel(**TINY_DENSE, operator="inpainting", guidance="dps")

        assert isinstance(diffusion.operator, InpaintingOperator)
        assert isinstance(diffusion.guidance_fn, DPS)

    def test_by_object(self):
        operator = InpaintingOperator()
        diffusion = DiffusionModel(**TINY_DENSE, operator=operator, guidance=None)

        assert diffusion.operator is operator

    def test_by_config_dict(self):
        diffusion = DiffusionModel(
            **TINY_DENSE,
            operator={"name": "inpainting"},
            guidance={"name": "dps", "params": {"disable_jit": True}},
        )

        assert isinstance(diffusion.operator, InpaintingOperator)
        assert isinstance(diffusion.guidance_fn, DPS)
        assert diffusion.guidance_fn.disable_jit is True

    def test_guidance_object_is_used_as_is(self):
        operator = InpaintingOperator()
        diffusion = DiffusionModel(**TINY_DENSE, operator=operator, guidance=None)
        guidance = DPS(diffusion_model=diffusion, operator=operator)

        rewired = DiffusionModel(**TINY_DENSE, operator=operator, guidance=guidance)

        assert rewired.guidance_fn is guidance

    def test_guidance_without_an_operator_is_rejected(self):
        """There is nothing to guide towards without a forward operator."""
        with pytest.raises(AssertionError, match="Operator must be provided"):
            DiffusionModel(**TINY_DENSE, operator=None, guidance="dps")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"operator": 42}, "Invalid operator"),
            ({"operator": "inpainting", "guidance": 42}, "Invalid guidance"),
        ],
    )
    def test_rejects_anything_else(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            DiffusionModel(**TINY_DENSE, **{"guidance": None, **kwargs})

    def test_config_roundtrip(self, model):
        config = model.get_config()
        restored = DiffusionModel.from_config(config)

        assert restored.get_config() == config
        assert config["network_name"] == "dense_time_conditional"


@pytest.mark.parametrize("schedule", ["diffusion_schedule", "linear_diffusion_schedule"])
class TestSchedules:
    """The noise schedules map a diffusion time onto signal and noise rates.

    Both are interchangeable, so they are held to the same contract.
    """

    def test_rates_are_variance_preserving(self, model, schedule):
        times = ops.convert_to_tensor(np.linspace(0.0, 1.0, 5, dtype="float32"))

        noise_rates, signal_rates = getattr(model, schedule)(times)

        np.testing.assert_allclose(
            ops.convert_to_numpy(noise_rates**2 + signal_rates**2), 1.0, rtol=1e-5
        )

    def test_starts_at_max_signal_and_ends_at_min(self, model, schedule):
        times = ops.convert_to_tensor(np.array([0.0, 1.0], dtype="float32"))

        _, signal_rates = getattr(model, schedule)(times)

        signal_rates = ops.convert_to_numpy(signal_rates)
        np.testing.assert_allclose(signal_rates[0], model.max_signal_rate, atol=1e-6)
        np.testing.assert_allclose(signal_rates[1], model.min_signal_rate, atol=1e-6)

    def test_signal_decays_monotonically(self, model, schedule):
        times = ops.convert_to_tensor(np.linspace(0.0, 1.0, 8, dtype="float32"))

        _, signal_rates = getattr(model, schedule)(times)

        assert np.all(np.diff(ops.convert_to_numpy(signal_rates)) < 0)

    def test_works_on_the_shapes_sampling_uses(self, model, schedule, rng):
        """Sampling broadcasts a per-image time over the image dimensions."""
        times = ops.convert_to_tensor(rng.random((3, 1, 1, 1)).astype("float32"))

        noise_rates, signal_rates = getattr(model, schedule)(times)

        assert noise_rates.shape == signal_rates.shape == (3, 1, 1, 1)


class TestReverseDiffusionStep:
    """One step of the reverse process, deterministic (DDIM) or stochastic (DDPM)."""

    @pytest.fixture
    def step_inputs(self, rng):
        shape = (2, 2)
        return dict(
            pred_images=rng.standard_normal(shape).astype("float32"),
            pred_noises=rng.standard_normal(shape).astype("float32"),
            signal_rates=np.float32(0.9),
            next_signal_rates=np.float32(0.95),
            next_noise_rates=np.float32(np.sqrt(1 - 0.95**2)),
            shape=shape,
        )

    def test_deterministic_step_recombines_the_prediction(self, model, step_inputs):
        out = model.reverse_diffusion_step(**step_inputs, stochastic_sampling=False)

        expected = (
            step_inputs["next_signal_rates"] * step_inputs["pred_images"]
            + step_inputs["next_noise_rates"] * step_inputs["pred_noises"]
        )
        np.testing.assert_allclose(ops.convert_to_numpy(out), expected, rtol=1e-5)

    def test_stochastic_step_adds_noise(self, model, step_inputs):
        """The DDPM step differs from the DDIM one by an injected noise term."""
        deterministic = model.reverse_diffusion_step(**step_inputs, stochastic_sampling=False)
        stochastic = model.reverse_diffusion_step(**step_inputs, stochastic_sampling=True)

        assert stochastic.shape == deterministic.shape
        assert not np.allclose(
            ops.convert_to_numpy(stochastic), ops.convert_to_numpy(deterministic)
        )


class TestNuclearPenalties:
    """The rank penalties that keep the estimated haze low-rank."""

    @pytest.fixture
    def frames(self, rng):
        return rng.standard_normal((2, 3, 4, 4, 1)).astype("float32")

    def test_nuclear_norm_penalty_is_lower_for_a_rank_one_stack(self):
        """Identical frames span a single direction, so the penalty is smaller."""
        frames = np.tile(np.ones((1, 1, 4, 4, 1), dtype="float32"), (1, 3, 1, 1, 1))

        rank_one = float(NuclearDiffusion.nuclear_norm_penalty(frames))
        full_rank = float(
            NuclearDiffusion.nuclear_norm_penalty(
                np.random.default_rng(0).standard_normal((1, 3, 4, 4, 1)).astype("float32")
            )
        )

        assert rank_one < full_rank

    def test_weighted_penalty_leans_on_the_tail_of_the_spectrum(self, frames):
        """Weighting the smaller singular values more gives a larger penalty."""
        plain = float(NuclearDiffusion.nuclear_norm_penalty(frames))
        weighted = float(NuclearDiffusion.weighted_nuclear_norm_penalty(frames))

        assert weighted > plain

    def test_weight_factor_scales_the_penalty(self, frames):
        light = float(NuclearDiffusion.weighted_nuclear_norm_penalty(frames, weight_factor=1.0))
        heavy = float(NuclearDiffusion.weighted_nuclear_norm_penalty(frames, weight_factor=4.0))

        assert heavy > light


def test_test_step_reports_both_losses(model, rng):
    """``model.evaluate`` runs the custom test step and tracks noise and image loss."""
    import keras

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
    data = rng.standard_normal((8, 2)).astype("float32")

    results = model.evaluate(data, batch_size=4, verbose=0, return_dict=True)

    assert {"n_loss", "i_loss"} <= set(results)
    assert all(np.isfinite(value) for value in results.values())
