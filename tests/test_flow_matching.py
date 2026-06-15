"""Tests for the FlowMatchingModel."""

import keras
import numpy as np
import pytest

from zea.models.diffusion import DPS
from zea.models.flow_matching import FlowMatchingModel

from . import DEFAULT_TEST_SEED


def _make_minimal_flow_model(input_shape, guidance=None, operator=None):
    """Return a minimal FlowMatchingModel (no guidance) for unit-testing."""
    return FlowMatchingModel(
        input_shape=input_shape,
        network_name="dense_time_conditional",
        network_kwargs={"widths": [8], "output_dim": input_shape[0]},
        guidance=guidance,
        operator=operator,
    )


def test_flow_matching_diffusion_schedule():
    """Linear schedule: noise_rates == t, signal_rates == 1 - t."""
    model = _make_minimal_flow_model((2,))
    t = keras.ops.convert_to_tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype="float32")
    noise_rates, signal_rates = model.diffusion_schedule(t)
    np.testing.assert_allclose(keras.ops.convert_to_numpy(noise_rates), [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(signal_rates), [1.0, 0.75, 0.5, 0.25, 0.0]
    )


def test_flow_matching_schedule_sums_to_one():
    """noise_rates + signal_rates should equal 1 for all t."""
    model = _make_minimal_flow_model((2,))
    t = keras.random.uniform((4,), minval=0.0, maxval=1.0)
    noise_rates, signal_rates = model.diffusion_schedule(t)
    total = keras.ops.convert_to_numpy(noise_rates + signal_rates)
    np.testing.assert_allclose(total, np.ones_like(total), atol=1e-6)


def test_flow_matching_denoise_output_shapes():
    """denoise() returns (pred_noises, pred_images) with the same shape as input."""
    batch_size, n_features = 2, 2
    model = _make_minimal_flow_model((n_features,))

    noisy = keras.random.uniform((batch_size, n_features))
    noise_rates = keras.ops.ones((batch_size, 1)) * 0.5
    signal_rates = keras.ops.ones((batch_size, 1)) * 0.5

    pred_noises, pred_images = model.denoise(noisy, noise_rates, signal_rates, training=False)
    assert pred_noises.shape == noisy.shape
    assert pred_images.shape == noisy.shape


def test_flow_matching_denoise_at_zero_t():
    """At t=0 the noisy image equals the clean image; x̂₀ should be noisy."""
    batch_size, n_features = 2, 2
    model = _make_minimal_flow_model((n_features,))

    clean = keras.random.normal((batch_size, n_features))
    noise_rates = keras.ops.zeros((batch_size, 1))  # t = 0
    signal_rates = keras.ops.ones((batch_size, 1))  # 1 − t = 1

    # x_t = clean at t=0 regardless of velocity; x̂₀ = x_t − 0·v = x_t
    pred_noises, pred_images = model.denoise(clean, noise_rates, signal_rates, training=False)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(pred_images),
        keras.ops.convert_to_numpy(clean),
        atol=1e-5,
    )


def test_flow_matching_reverse_step_deterministic():
    """Deterministic Euler step: x_{t-dt} = (1-t+dt)*x̂₀ + (t-dt)*ε̂."""
    batch_size, n_features = 2, 2
    model = _make_minimal_flow_model((n_features,))

    pred_images = keras.random.normal((batch_size, n_features))
    pred_noises = keras.random.normal((batch_size, n_features))
    signal_rates = keras.ops.ones((batch_size, 1)) * 0.5  # α_t
    next_signal_rates = keras.ops.ones((batch_size, 1)) * 0.6  # α_{t−Δt}
    next_noise_rates = keras.ops.ones((batch_size, 1)) * 0.4  # σ_{t−Δt}

    result = model.reverse_diffusion_step(
        shape=(batch_size, n_features),
        pred_images=pred_images,
        pred_noises=pred_noises,
        signal_rates=signal_rates,
        next_signal_rates=next_signal_rates,
        next_noise_rates=next_noise_rates,
        stochastic_sampling=False,
    )

    expected = keras.ops.convert_to_numpy(next_signal_rates) * keras.ops.convert_to_numpy(
        pred_images
    ) + keras.ops.convert_to_numpy(next_noise_rates) * keras.ops.convert_to_numpy(pred_noises)
    np.testing.assert_allclose(keras.ops.convert_to_numpy(result), expected, atol=1e-6)


def test_flow_matching_reverse_step_stochastic_shape():
    """Stochastic reverse step returns the correct shape."""
    batch_size, n_features = 2, 2
    model = _make_minimal_flow_model((n_features,))
    keras.utils.set_random_seed(DEFAULT_TEST_SEED)

    pred_images = keras.random.normal((batch_size, n_features))
    pred_noises = keras.random.normal((batch_size, n_features))
    signal_rates = keras.ops.ones((batch_size, 1)) * 0.5
    next_signal_rates = keras.ops.ones((batch_size, 1)) * 0.6
    next_noise_rates = keras.ops.ones((batch_size, 1)) * 0.4

    result = model.reverse_diffusion_step(
        shape=(batch_size, n_features),
        pred_images=pred_images,
        pred_noises=pred_noises,
        signal_rates=signal_rates,
        next_signal_rates=next_signal_rates,
        next_noise_rates=next_noise_rates,
        stochastic_sampling=True,
    )
    assert result.shape == (batch_size, n_features)


def test_flow_matching_get_config_no_signal_rate():
    """get_config() must not contain cosine-schedule keys."""
    model = _make_minimal_flow_model((4,))
    config = model.get_config()
    assert "min_signal_rate" not in config
    assert "max_signal_rate" not in config


def test_flow_matching_get_config_round_trip():
    """FlowMatchingModel can be reconstructed from its own config."""
    model = _make_minimal_flow_model((4,))
    config = model.get_config()
    restored = FlowMatchingModel.from_config(config)
    assert restored.input_shape == model.input_shape


def test_flow_matching_times_in_range():
    """_sample_diffusion_times returns values in [min_t, max_t]."""
    model = _make_minimal_flow_model((2,))
    times = keras.ops.convert_to_numpy(model._sample_diffusion_times(1000, 1))
    assert times.min() >= model.min_t
    assert times.max() <= model.max_t


def test_flow_matching_metrics_names():
    """metrics property should expose v_loss and i_loss trackers."""
    model = _make_minimal_flow_model((2,))
    # Trackers are created lazily; populate them directly without training.
    model.velocity_loss_tracker.update_state(keras.ops.convert_to_tensor(0.0))
    model.image_loss_tracker.update_state(keras.ops.convert_to_tensor(0.0))

    names = [m.name for m in model.metrics]
    assert any("v_loss" in n for n in names)
    assert any("i_loss" in n for n in names)


def test_flow_matching_sample_shape():
    """model.sample() returns the expected shape."""
    keras.utils.set_random_seed(DEFAULT_TEST_SEED)
    seed_gen = keras.random.SeedGenerator(DEFAULT_TEST_SEED)

    n_features = 2
    n_samples = 2
    model = _make_minimal_flow_model((n_features,))

    samples = model.sample(n_samples=n_samples, n_steps=2, seed=seed_gen)
    assert samples.shape == (n_samples, n_features)


def test_flow_matching_sample_finite():
    """Sampled values should be finite."""
    keras.utils.set_random_seed(DEFAULT_TEST_SEED)
    seed_gen = keras.random.SeedGenerator(DEFAULT_TEST_SEED)

    n_features = 2
    model = _make_minimal_flow_model((n_features,))
    samples = model.sample(n_samples=2, n_steps=2, seed=seed_gen)
    assert np.isfinite(keras.ops.convert_to_numpy(samples)).all()


def test_flow_matching_posterior_sample_shape():
    """posterior_sample returns (n_measurements, n_samples, n_features)."""
    keras.utils.set_random_seed(DEFAULT_TEST_SEED)
    seed_gen = keras.random.SeedGenerator(DEFAULT_TEST_SEED)

    n_features = 2
    n_measurements = 2
    n_samples = 2

    # Use default guidance ("dps") so posterior_sample works
    model = FlowMatchingModel(
        input_shape=(n_features,),
        network_name="dense_time_conditional",
        network_kwargs={"widths": [8], "output_dim": n_features},
    )

    measurements = keras.random.uniform((n_measurements, n_features), minval=-1, maxval=1)
    mask = keras.ops.ones((n_measurements, n_features))

    out = model.posterior_sample(
        measurements=measurements,
        n_samples=n_samples,
        mask=mask,
        n_steps=2,
        omega=1.0,
        seed=seed_gen,
        verbose=False,
    )
    assert out.shape == (n_measurements, n_samples, n_features)


def test_flow_matching_test_step_returns_losses():
    """test_step returns a dict with v_loss and i_loss keys."""
    batch_size, n_features = 4, 2
    model = _make_minimal_flow_model((n_features,))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.MeanSquaredError(),
    )

    data = keras.random.normal((batch_size, n_features))
    metrics = model.test_step(data)

    assert "v_loss" in metrics
    assert "i_loss" in metrics
    assert np.isfinite(float(metrics["v_loss"]))
    assert np.isfinite(float(metrics["i_loss"]))


def test_flow_matching_train_step_returns_losses():
    """train_step returns a dict with v_loss and i_loss keys (TF only)."""
    pytest.importorskip("tensorflow")

    batch_size, n_features = 4, 2
    model = _make_minimal_flow_model((n_features,))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.MeanSquaredError(),
    )

    data = keras.random.normal((batch_size, n_features))
    metrics = model.train_step(data)

    assert "v_loss" in metrics
    assert "i_loss" in metrics
    assert np.isfinite(float(metrics["v_loss"]))
    assert np.isfinite(float(metrics["i_loss"]))


def _make_dit_flow_model(image_shape=(16, 16, 1), guidance=None, operator=None):
    """Return a small DiT-backed FlowMatchingModel for unit-testing."""
    return FlowMatchingModel(
        input_shape=image_shape,
        network_name="dit_time_conditional",
        network_kwargs={
            "patch_size": 8,
            "hidden_size": 32,
            "depth": 2,
            "num_heads": 4,
            "embedding_dims": 16,
        },
        guidance=guidance,
        operator=operator,
    )


def test_flow_matching_dit_denoise_output_shapes():
    """DiT backend: denoise() preserves the image shape."""
    image_shape = (16, 16, 1)
    model = _make_dit_flow_model(image_shape)

    noisy = keras.random.uniform((2, *image_shape))
    noise_rates = keras.ops.ones((2, 1, 1, 1)) * 0.5
    signal_rates = keras.ops.ones((2, 1, 1, 1)) * 0.5

    pred_noises, pred_images = model.denoise(noisy, noise_rates, signal_rates, training=False)
    assert pred_noises.shape == noisy.shape
    assert pred_images.shape == noisy.shape


def test_flow_matching_dit_sample_shape_and_finite():
    """DiT backend: model.sample() returns the right shape and finite values."""
    keras.utils.set_random_seed(DEFAULT_TEST_SEED)
    seed_gen = keras.random.SeedGenerator(DEFAULT_TEST_SEED)

    image_shape = (16, 16, 1)
    model = _make_dit_flow_model(image_shape)

    samples = model.sample(n_samples=2, n_steps=2, seed=seed_gen, verbose=False)
    assert samples.shape == (2, *image_shape)
    assert np.isfinite(keras.ops.convert_to_numpy(samples)).all()


def test_flow_matching_dit_config_round_trip():
    """DiT-backed FlowMatchingModel can be reconstructed from its own config."""
    image_shape = (16, 16, 1)
    model = _make_dit_flow_model(image_shape)
    config = model.get_config()
    assert config["network_name"] == "dit_time_conditional"

    restored = FlowMatchingModel.from_config(config)
    assert tuple(restored.input_shape) == image_shape
    assert restored.network.count_params() == model.network.count_params()


def test_flow_matching_default_solver_is_heun():
    """Flow matching defaults to the second-order Heun solver."""
    model = _make_minimal_flow_model((2,))
    assert model.solver == "heun"


def test_flow_matching_invalid_solver_raises():
    """An unknown solver name is rejected at construction time."""
    with pytest.raises(ValueError):
        FlowMatchingModel(
            input_shape=(2,),
            network_name="dense_time_conditional",
            network_kwargs={"widths": [8], "output_dim": 2},
            guidance=None,
            operator=None,
            solver="rk4",
        )


def test_flow_matching_solver_config_round_trip():
    """The solver choice survives a get_config/from_config round trip."""
    model = FlowMatchingModel(
        input_shape=(2,),
        network_name="dense_time_conditional",
        network_kwargs={"widths": [8], "output_dim": 2},
        guidance=None,
        operator=None,
        solver="euler",
    )
    restored = FlowMatchingModel.from_config(model.get_config())
    assert restored.solver == "euler"


def test_flow_matching_heun_sample_shape_and_finite():
    """Both solvers produce samples of the right shape with finite values."""
    keras.utils.set_random_seed(DEFAULT_TEST_SEED)
    n_features, n_samples = 2, 2
    for solver in ("euler", "heun"):
        model = FlowMatchingModel(
            input_shape=(n_features,),
            network_name="dense_time_conditional",
            network_kwargs={"widths": [8], "output_dim": n_features},
            guidance=None,
            operator=None,
            solver=solver,
        )
        seed_gen = keras.random.SeedGenerator(DEFAULT_TEST_SEED)
        samples = model.sample(n_samples=n_samples, n_steps=3, seed=seed_gen, verbose=False)
        assert samples.shape == (n_samples, n_features)
        assert np.isfinite(keras.ops.convert_to_numpy(samples)).all()


def test_flow_matching_heun_exact_on_linear_velocity():
    """Heun integrates an x-independent, linear-in-t velocity field exactly.

    For ``dx/dt = v(t) = t`` integrated from ``t=1`` to ``t=0`` the exact
    solution is ``x(0) = x(1) - 0.5``.  The trapezoidal Heun corrector is exact
    for a velocity that is linear in ``t``, whereas Euler incurs an O(Δt) error.
    """

    class _MockFM(FlowMatchingModel):
        def denoise(self, noisy_images, noise_rates, signal_rates, training, network=None):
            velocity = noise_rates  # v = t, independent of x
            pred_images = noisy_images - noise_rates * velocity  # x̂₀ = x_t - t·v
            pred_noises = pred_images + velocity  # ε̂ = x̂₀ + v
            return pred_noises, pred_images

    x1 = 3.0
    exact = x1 - 0.5

    def _final_x0(solver):
        model = _MockFM(
            input_shape=(1,),
            network_name="dense_time_conditional",
            network_kwargs={"widths": [2], "output_dim": 1},
            guidance=None,
            operator=None,
            solver=solver,
        )
        init = keras.ops.ones((1, 1)) * x1
        model.reverse_diffusion(
            initial_noise=init, diffusion_steps=4, verbose=False, track_progress_type="x_t"
        )
        return float(np.asarray(model.track_progress[-1]).ravel()[0])

    heun_val = _final_x0("heun")
    euler_val = _final_x0("euler")

    np.testing.assert_allclose(heun_val, exact, atol=1e-5)
    # Euler is only first-order, so it should be strictly less accurate here.
    assert abs(euler_val - exact) > abs(heun_val - exact)


def test_flow_matching_dps_guidance_call():
    """DPS guidance works with FlowMatchingModel and returns correct shapes."""
    from zea.internal.operators import InpaintingOperator

    n_features, batch_size = 2, 2

    model = _make_minimal_flow_model((n_features,))
    guidance = DPS(diffusion_model=model, operator=InpaintingOperator(), disable_jit=True)

    noisy = keras.random.uniform((batch_size, n_features))
    measurements = keras.random.uniform((batch_size, n_features))
    mask = keras.ops.ones((batch_size, n_features))
    noise_rates = keras.ops.ones((batch_size, 1)) * 0.5
    signal_rates = keras.ops.ones((batch_size, 1)) * 0.5

    gradients, (error, (pred_noises, pred_images)) = guidance(
        noisy,
        measurements=measurements,
        noise_rates=noise_rates,
        signal_rates=signal_rates,
        omega=1.0,
        mask=mask,
    )

    assert gradients.shape == noisy.shape
    assert pred_noises.shape == noisy.shape
    assert pred_images.shape == noisy.shape
    assert np.isfinite(float(error))
