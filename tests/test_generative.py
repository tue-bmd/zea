"""Tests for generative models in zea."""

import keras
import matplotlib.pyplot as plt
import numpy as np
import pytest

from zea import log
from zea.io_lib import matplotlib_figure_to_numpy, save_video
from zea.models.diffusion import DiffusionModel
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
