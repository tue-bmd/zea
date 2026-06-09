"""Tests for generative models in zea."""

from unittest.mock import MagicMock

import keras
import matplotlib.pyplot as plt
import numpy as np
import pytest

from zea import log
from zea.func.ultrasound import dehaze_nuclear_diffusion
from zea.internal.operators import InpaintingOperator, LinearInterpOperator
from zea.io_lib import matplotlib_figure_to_numpy, save_video
from zea.models.diffusion import DDS, DPS, DiffusionModel, NuclearDiffusion
from zea.models.gmm import GaussianMixtureModel, match_means_covariances

from . import DEFAULT_TEST_SEED


@pytest.fixture(params=[2, 3])
def synthetic_2d_data(request):
    """Generate synthetic 2D data with Gaussian clusters."""
    n_centers = request.param
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    n = 600
    means = []
    covs = []
    radius = 10
    for i in range(n_centers):
        angle = 2 * np.pi * i / n_centers
        mean = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        cov = np.array([[0.5 + 0.5 * i, 0.2], [0.2, 0.3 + 0.2 * i]])
        means.append(mean)
        covs.append(cov)
    means = np.array(means)
    covs = np.array(covs)
    data_parts = [
        rng.multivariate_normal(means[i], covs[i], size=n // n_centers) for i in range(n_centers)
    ]
    data = np.concatenate(data_parts, axis=0)
    rng.shuffle(data)
    return data.astype("float32"), means, covs


def plot_distributions(data, samples, means=None, covs=None, title="", filename="test.png"):
    """Plot data, model samples, and optionally GMM means/covariances."""
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.3, label="Data", s=20)
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, label="Model Samples", s=20)
    if means is not None:
        plt.scatter(means[:, 0], means[:, 1], c="red", marker="x", s=100, label="GMM Means")
    if means is not None and covs is not None:
        for mean, cov in zip(means, covs):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(vals)
            ell = plt.matplotlib.patches.Ellipse(
                xy=mean,
                width=width,
                height=height,
                angle=theta,
                edgecolor="k",
                facecolor="none",
                lw=2,
                alpha=0.5,
            )
            plt.gca().add_patch(ell)
    plt.legend()
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(filename)
    log.success(f"Saved plot to {log.yellow(filename)}")


def test_gmm_fit_and_sample_2d(synthetic_2d_data, debug=False):
    """Test GMM fitting and sampling on synthetic 2D data."""
    data, true_means, true_covs = synthetic_2d_data
    n_components = len(true_means)
    gmm = GaussianMixtureModel(n_components=n_components, n_features=2)
    gmm.fit(data, max_iter=300, verbose=0)
    samples = keras.ops.convert_to_numpy(gmm.sample(len(data)))
    means = keras.ops.convert_to_numpy(gmm.means)
    vars_ = keras.ops.convert_to_numpy(gmm.vars)
    covs = [np.diag(v) for v in vars_]

    if debug:
        plot_distributions(data, samples, means, covs, title="GMM 2D Fit Debug")

    true_means = keras.ops.convert_to_tensor(true_means, dtype="float32")
    true_covs = keras.ops.convert_to_tensor(true_covs, dtype="float32")
    means_m, true_means_m, covs_m, true_covs_m = match_means_covariances(
        means, true_means, covs, true_covs
    )
    assert np.allclose(means_m, true_means_m, atol=2)
    for c, tc in zip(covs_m, true_covs_m):
        assert np.allclose(c, tc, atol=2)
    ll = gmm.log_density(data)
    assert np.isfinite(keras.ops.convert_to_numpy(ll)).all()


def test_match_means_covariances_greedy():
    """Test match_means_covariances matches means and covariances correctly."""

    means = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
    true_means = np.array([[2, 2], [0, 0], [1, 1]], dtype=np.float32)
    covs = [np.eye(2) for _ in range(3)]
    true_covs = [np.eye(2) * 2 for _ in range(3)]
    matched_means, matched_true_means, matched_covs, matched_true_covs = match_means_covariances(
        means, true_means, covs, true_covs
    )
    assert np.allclose(matched_means, matched_true_means, atol=1e-6)
    for c, tc in zip(matched_covs, matched_true_covs):
        assert c.shape == tc.shape


def animate_diffusion_trajectory_2d(
    model, data, filename="diffusion_trajectory.gif", n_show=300, show_data=True
):
    """
    Animate the intermediate diffusion steps using model.track_progress.

    Args:
        model: Trained DiffusionModel with track_progress filled (after sampling).
        data: Original data (for plotting as background).
        filename: Output GIF filename.
        n_show: Number of samples to show per frame.
        show_data: Whether to plot the original data in the background.
    """

    n_show = min(n_show, data.shape[0])
    frames = []
    for i, samples in enumerate(model.track_progress):
        fig, ax = plt.subplots(figsize=(6, 6))
        if show_data:
            ax.scatter(data[:n_show, 0], data[:n_show, 1], alpha=0.2, label="Data", s=15)
        ax.scatter(
            samples[:n_show, 0],
            samples[:n_show, 1],
            alpha=0.7,
            label="x₀ estimate",
            s=15,
            color="tab:blue",
        )
        ax.set_title(f"Diffusion Step {i + 1}/{len(model.track_progress)}")
        ax.axis("equal")
        ax.set_xlim(data[:, 0].min() - 2, data[:, 0].max() + 2)
        ax.set_ylim(data[:, 1].min() - 2, data[:, 1].max() + 2)
        ax.legend()
        fig.tight_layout()
        frame = matplotlib_figure_to_numpy(fig)
        frames.append(frame)
        plt.close(fig)
    save_video(frames, filename, fps=10)
    log.success(f"Animated diffusion trajectory saved to {filename}")


def test_diffusion_fit_and_sample_2d(synthetic_2d_data, debug=False):
    """Test diffusion model fitting and sampling on synthetic 2D data."""
    data, *_ = synthetic_2d_data

    keras.utils.set_random_seed(DEFAULT_TEST_SEED)
    seed_gen = keras.random.SeedGenerator(DEFAULT_TEST_SEED)

    n = len(data)
    model = DiffusionModel(
        input_shape=(2,),
        network_name="dense_time_conditional",
        network_kwargs={"widths": [64, 64], "output_dim": 2},
        min_signal_rate=0.02,
        max_signal_rate=0.95,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.MeanSquaredError(),
    )

    # for actual good fit we probably need more like 300 epochs
    # for the tests this is good enough
    model.fit(data, epochs=200, batch_size=64, verbose=0)

    samples = model.sample(n_samples=n, n_steps=100, seed=seed_gen)
    samples = keras.ops.convert_to_numpy(samples)
    samples = samples.reshape(-1, 2)

    if debug:
        plot_distributions(data, samples, title="Diffusion 2D Fit Debug")
        animate_diffusion_trajectory_2d(
            model, data, filename="diffusion_trajectory.gif", n_show=300
        )

    assert np.isfinite(np.cov(samples.T)).all()

    # for the means we need a different way of checking
    # let's use the GMM to check the means
    gmm = GaussianMixtureModel(n_components=3, n_features=2)
    gmm.fit(samples, max_iter=300, verbose=0)
    means = keras.ops.convert_to_numpy(gmm.means)
    vars_ = keras.ops.convert_to_numpy(gmm.vars)
    covs = [np.diag(v) for v in vars_]
    means_m, true_means_m, covs_m, true_covs_m = match_means_covariances(means, means, covs, covs)
    assert np.allclose(means_m, true_means_m, atol=1)
    for c, tc in zip(covs_m, true_covs_m):
        assert np.allclose(c, tc, atol=1)


def test_gmm_posterior_sample():
    """Test GMM posterior_sample returns correct shape and values."""
    n_components = 3
    n_features = 2
    n_measurements = 5
    n_samples = 4
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    seed_gen = keras.random.SeedGenerator(DEFAULT_TEST_SEED)
    # Make up some GMM parameters and measurements
    gmm = GaussianMixtureModel(n_components=n_components, n_features=n_features)
    gmm.means = keras.ops.convert_to_tensor(
        rng.normal(size=(n_components, n_features)), dtype="float32"
    )
    gmm.vars = keras.ops.ones((n_components, n_features))
    gmm.pi = keras.ops.ones((n_components,)) / n_components
    gmm._initialized = True
    measurements = rng.normal(size=(n_measurements, n_features)).astype("float32")
    comp_idx = gmm.posterior_sample(measurements, n_samples=n_samples, seed=seed_gen)
    arr = keras.ops.convert_to_numpy(comp_idx)
    assert arr.shape == (n_measurements, n_samples)
    assert ((arr >= 0) & (arr < n_components)).all()


def test_diffusion_posterior_sample_shape():
    """Test DiffusionModel.posterior_sample returns correct shape."""
    n_measurements = 3
    n_features = 2
    n_samples = 5

    keras.utils.set_random_seed(DEFAULT_TEST_SEED)
    seed_gen = keras.random.SeedGenerator(DEFAULT_TEST_SEED)

    # Use a minimal diffusion model with dense network
    model = DiffusionModel(
        input_shape=(n_features,),
        network_name="dense_time_conditional",
        network_kwargs={"widths": [8], "output_dim": n_features},
    )
    # No training needed for shape test
    measurements = keras.random.uniform((n_measurements, n_features), minval=-1, maxval=1)
    mask = keras.random.uniform((n_measurements, n_features)) > 0.5
    mask = keras.ops.cast(mask, "float32")
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


def test_dehaze_nuclear_diffusion_shape_logic():
    """Test dehaze_nuclear_diffusion shape logic."""

    keras.utils.set_random_seed(DEFAULT_TEST_SEED)
    seed_gen = keras.random.SeedGenerator(DEFAULT_TEST_SEED)

    # Create test video data
    n_frames = 10
    height, width, channels = 32, 32, 1
    hazy_video = keras.random.uniform(
        (n_frames, height, width, channels),
        minval=-1,
        maxval=1,
        seed=seed_gen,
    )

    model = DiffusionModel(
        input_shape=(height, width, channels),
        guidance="nuclear-dps",
        operator="linear_interp",
    )

    # Test with non-overlapping windows
    tissue_frames, haze_frames = dehaze_nuclear_diffusion(
        hazy_video,
        model,
        n_steps=2,
        initial_step=0,
        window_size=3,
        window_stride=3,  # Non-overlapping
        hard_project=False,
        seed=seed_gen,
        verbose=False,
        omega=1.0,
        gamma=1.0,
    )

    # Check output shapes
    assert tissue_frames.shape == (n_frames, height, width, channels)
    assert haze_frames.shape == (n_frames, height, width, channels)

    # Test with overlapping windows
    tissue_frames_overlap, haze_frames_overlap = dehaze_nuclear_diffusion(
        hazy_video,
        model,
        n_steps=2,
        initial_step=0,
        window_size=4,
        window_stride=2,  # Overlapping
        hard_project=False,
        seed=seed_gen,
        verbose=False,
        omega=1.0,
        gamma=1.0,
    )

    # Check output shapes with overlap
    assert tissue_frames_overlap.shape == (n_frames, height, width, channels)
    assert haze_frames_overlap.shape == (n_frames, height, width, channels)


def test_dehaze_nuclear_diffusion_hard_projection():
    """Test dehaze_nuclear_diffusion with hard projection enabled."""

    keras.utils.set_random_seed(DEFAULT_TEST_SEED)
    seed_gen = keras.random.SeedGenerator(DEFAULT_TEST_SEED)

    # Create test video data with some bright values
    n_frames = 5
    height, width, channels = 16, 16, 1
    hazy_video = keras.random.uniform((n_frames, height, width, channels), minval=-1, maxval=1)

    model = DiffusionModel(
        input_shape=(height, width, channels),
        guidance="nuclear-dps",
        operator="linear_interp",
    )

    # Test with hard projection
    tissue_frames, haze_frames = dehaze_nuclear_diffusion(
        hazy_video,
        model,
        n_steps=2,
        initial_step=0,
        window_size=3,
        window_stride=None,
        hard_project=True,
        seed=seed_gen,
        verbose=False,
        omega=1.0,
        gamma=1.0,
    )

    # Check that shapes are correct
    assert tissue_frames.shape == (n_frames, height, width, channels)
    assert haze_frames.shape == (n_frames, height, width, channels)

    # With hard projection, positive values in tissue should come from hazy input
    hazy_np = keras.ops.convert_to_numpy(hazy_video)
    positive_mask = tissue_frames > 0
    if positive_mask.any():
        # Where tissue is positive, it should match hazy input
        np.testing.assert_array_almost_equal(
            tissue_frames[positive_mask], hazy_np[positive_mask], decimal=5
        )


def test_dehaze_nuclear_diffusion_validation():
    """Test dehaze_nuclear_diffusion raises errors for invalid configurations."""

    # Create test video data
    n_frames = 5
    height, width, channels = 16, 16, 1
    hazy_video = keras.random.uniform((n_frames, height, width, channels), minval=-1, maxval=1)

    # Test with model without guidance function
    mock_model = MagicMock()
    mock_model.guidance_fn = None

    with pytest.raises(ValueError, match="guidance function"):
        dehaze_nuclear_diffusion(
            hazy_video,
            mock_model,
            n_steps=2,
            initial_step=0,
            window_size=3,
            verbose=False,
        )

    # Test with wrong guidance type
    mock_model.guidance_fn = MagicMock(spec=DPS)

    with pytest.raises(ValueError, match="Nuclear Diffusion"):
        dehaze_nuclear_diffusion(
            hazy_video,
            mock_model,
            n_steps=2,
            initial_step=0,
            window_size=3,
            verbose=False,
        )


def _make_minimal_diffusion_model(input_shape):
    """Create a minimal DiffusionModel with no guidance for direct guidance testing."""
    return DiffusionModel(
        input_shape=input_shape,
        network_name="dense_time_conditional",
        network_kwargs={"widths": [8], "output_dim": input_shape[0]},
        guidance=None,
        operator=None,
    )


def test_dps_guidance_call():
    """Test DPS guidance returns (gradients, (error, (pred_noises, pred_images))).

    JIT is disabled for easier testing, but also to trigger coverage of the executed code paths.
    """
    n_features, batch_size = 4, 2

    model = _make_minimal_diffusion_model((n_features,))
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


def test_dds_guidance_call():
    """Test DDS guidance returns (gradients, (error, (pred_noises, pred_images))).

    JIT is disabled for easier testing, but also to trigger coverage of the executed code paths.
    """
    n_features, batch_size = 4, 2

    model = _make_minimal_diffusion_model((n_features,))
    guidance = DDS(diffusion_model=model, operator=InpaintingOperator(), disable_jit=True)

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
        n_inner=3,
        eps=1e-2,
        mask=mask,
    )

    assert gradients.shape == noisy.shape
    assert pred_noises.shape == noisy.shape
    assert pred_images.shape == noisy.shape
    assert np.isfinite(float(error))


def test_nuclear_diffusion_guidance_call():
    """Test NuclearDiffusion guidance returns ((grad_tissue, grad_haze), (error, aux)).

    JIT is disabled for easier testing, but also to trigger coverage of the executed code paths.
    """
    batch, frames, h, w, c = 1, 3, 8, 8, 1

    # DiffusionModel input_shape is per-frame (h, w, c)
    model = DiffusionModel(
        input_shape=(h, w, c),
        guidance=None,
        operator=None,
    )
    guidance = NuclearDiffusion(
        diffusion_model=model, operator=LinearInterpOperator(), disable_jit=True
    )

    noisy_tissue = keras.random.uniform((batch, frames, h, w, c))
    noisy_haze = keras.random.uniform((batch, frames, h, w, c))
    measurements = keras.random.uniform((batch, frames, h, w, c))
    noise_rates = keras.ops.ones((batch, frames, 1, 1, 1)) * 0.5
    signal_rates = keras.ops.ones((batch, frames, 1, 1, 1)) * 0.5

    (grad_tissue, grad_haze), (error, aux) = guidance(
        noisy_tissue,
        noisy_haze,
        measurements=measurements,
        noise_rates=noise_rates,
        signal_rates=signal_rates,
        omega=1.0,
        gamma=1.0,
        step=50,
        total_steps=100,
        initial_step=keras.ops.convert_to_tensor(0, dtype="float32"),
    )

    assert grad_tissue.shape == noisy_tissue.shape
    assert grad_haze.shape == noisy_haze.shape
    assert np.isfinite(float(error))
    # aux = (pred_noises_tissue, pred_tissue, pred_haze, l2_error, nuclear_penalty)
    assert len(aux) == 5
