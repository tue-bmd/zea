import keras
import matplotlib.pyplot as plt
import numpy as np
import pytest

from usbmd import log
from usbmd.models.diffusion import DiffusionModel
from usbmd.models.gmm import GaussianMixtureModel


@pytest.fixture(params=[2, 3])
def synthetic_2d_data(request):
    n_centers = request.param
    rng = np.random.default_rng(42)
    n = 600
    means = []
    covs = []
    for i in range(n_centers):
        angle = 2 * np.pi * i / n_centers
        mean = np.array([5 * np.cos(angle), 5 * np.sin(angle)])
        cov = np.array([[0.5 + 0.5 * i, 0.2], [0.2, 0.3 + 0.2 * i]])
        means.append(mean)
        covs.append(cov)
    means = np.array(means)
    covs = np.array(covs)
    data_parts = [
        rng.multivariate_normal(means[i], covs[i], size=n // n_centers)
        for i in range(n_centers)
    ]
    data = np.concatenate(data_parts, axis=0)
    rng.shuffle(data)
    return data.astype("float32"), means, covs


def plot_distributions(
    data, samples, means=None, covs=None, title="", filename="test.png"
):
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.3, label="Data", s=20)
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, label="Model Samples", s=20)
    if means is not None:
        plt.scatter(
            means[:, 0], means[:, 1], c="red", marker="x", s=100, label="GMM Means"
        )
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
    # Check means close to true means (up to permutation)
    means_sorted = means[np.argsort(means[:, 0])]
    true_means_sorted = true_means[np.argsort(true_means[:, 0])]
    assert np.allclose(means_sorted, true_means_sorted, atol=2.0)
    # Check if covariances are close to true covariances
    for i in range(n_components):
        assert np.allclose(covs[i], true_covs[i], atol=2.0)
    # Check log likelihood is finite
    ll = gmm.log_likelihood(data)
    assert np.isfinite(keras.ops.convert_to_numpy(ll)).all()
