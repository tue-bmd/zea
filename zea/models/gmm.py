"""Gaussian Mixture Model (GMM) implementation"""

import keras
import numpy as np
from keras import ops

from zea.func.tensor import linear_sum_assignment
from zea.models.generative import GenerativeModel


class GaussianMixtureModel(GenerativeModel):
    """
    Gaussian Mixture Model fitted with EM algorithm.

    Args:
        n_components: Number of mixture components.
        n_features: Number of features (dimensions).
        max_iter: Maximum number of EM steps.
        tol: Convergence tolerance.
        seed: Random seed for reproducibility.

    Example:
    ```python
    gmm = GaussianMixtureModel(n_components=2, n_features=2)
    gmm.fit(data, max_iter=100)
    samples = gmm.sample(100)
    ```
    """

    def __init__(self, n_components=2, n_features=1, tol=1e-4, seed=None):
        self.n_components = n_components
        self.n_features = n_features
        self.tol = tol
        self.seed = seed
        self._initialized = False

        self.means = None  # (n_components, n_features)
        self.vars = None  # (n_components, n_features)
        self.pi = None  # (n_components,)

    def _initialize(self, X):
        # X: (n_samples, n_features)
        n_samples = ops.shape(X)[0]
        n_features = ops.shape(X)[1]
        chosen = []
        # Pick the first mean randomly
        idx = ops.cast(
            keras.random.uniform(
                shape=(),
                minval=0,
                maxval=n_samples,
                seed=self.seed,
            ),
            "int32",
        )
        chosen.append(idx)
        for _ in range(1, self.n_components):
            # Gather chosen means so far
            chosen_means = ops.stack(
                [ops.take(X, i, axis=0) for i in chosen], axis=0
            )  # (len(chosen), n_features)
            # Compute distances from all points to each chosen mean
            # (n_samples, len(chosen), n_features)
            diffs = ops.expand_dims(X, 1) - ops.expand_dims(chosen_means, 0)
            dists = ops.sqrt(ops.sum(diffs**2, axis=-1))  # (n_samples, len(chosen))
            min_dists = ops.min(dists, axis=1)  # (n_samples,)
            idx = ops.argmax(min_dists, axis=0)
            chosen.append(idx)
        means = ops.stack([ops.take(X, i, axis=0) for i in chosen], axis=0)
        self.means = means
        # Initialize variances to variance of data
        var = ops.var(X, axis=0)
        self.vars = ops.ones((self.n_components, n_features)) * var
        # Initialize mixture weights uniformly
        self.pi = ops.ones((self.n_components,)) / self.n_components
        self._initialized = True

    def _e_step(self, X):
        # X: (n_samples, n_features)
        X_exp = ops.expand_dims(X, axis=1)  # (n_samples, 1, n_features)
        means = ops.expand_dims(self.means, axis=0)  # (1, n_components, n_features)
        vars_ = ops.expand_dims(self.vars, axis=0)  # (1, n_components, n_features)
        pi = self.pi  # (n_components,)

        # Compute log Gaussian pdf for each component
        log_prob = -0.5 * ops.sum(
            ops.log(2 * np.pi * vars_) + ((X_exp - means) ** 2) / vars_, axis=-1
        )  # (n_samples, n_components)
        # Add log mixture weights
        log_prob = log_prob + ops.log(pi)
        # Normalize to get responsibilities
        log_prob_norm = log_prob - ops.logsumexp(log_prob, axis=1, keepdims=True)
        gamma = ops.exp(log_prob_norm)  # (n_samples, n_components)
        return gamma  # responsibilities

    def _m_step(self, X, gamma):
        # X: (n_samples, n_features)
        # gamma: (n_samples, n_components)
        Nk = ops.sum(gamma, axis=0)  # (n_components,)
        # Update means
        means = ops.sum(
            ops.expand_dims(gamma, -1) * ops.expand_dims(X, 1), axis=0
        ) / ops.expand_dims(Nk, -1)
        # Update variances
        X_exp = ops.expand_dims(X, axis=1)  # (n_samples, 1, n_features)
        means_exp = ops.expand_dims(means, axis=0)  # (1, n_components, n_features)
        vars_ = ops.sum(gamma[..., None] * (X_exp - means_exp) ** 2, axis=0) / ops.expand_dims(
            Nk, -1
        )
        # Update mixture weights
        pi = Nk / ops.sum(Nk)
        return means, vars_, pi

    def fit(self, data, max_iter=100, verbose=0, **kwargs):
        X = ops.convert_to_tensor(data, dtype="float32")
        if not self._initialized:
            self._initialize(X)

        prev_ll = None
        progbar = keras.utils.Progbar(max_iter, verbose=verbose)
        for i in range(max_iter):
            # E-step
            gamma = self._e_step(X)
            # M-step
            means, vars_, pi = self._m_step(X, gamma)
            # Compute log-likelihood
            self.means, self.vars, self.pi = means, vars_, pi
            ll = ops.sum(ops.log(ops.sum(self._component_pdf(X) * self.pi, axis=1)))
            if verbose:
                progbar.update(i + 1, values=[("log-likelihood", float(ll))])
            if prev_ll is not None and abs(float(ll) - float(prev_ll)) < self.tol:
                if verbose:
                    print(f"\nConverged at iter {i}")
                break
            prev_ll = ll

    def _component_pdf(self, X):
        # X: (n_samples, n_features)
        X_exp = ops.expand_dims(X, axis=1)  # (n_samples, 1, n_features)
        means = ops.expand_dims(self.means, axis=0)  # (1, n_components, n_features)
        vars_ = ops.expand_dims(self.vars, axis=0)  # (1, n_components, n_features)
        # Gaussian PDF (no mixture weights)
        norm = ops.prod(ops.sqrt(2 * np.pi * vars_), axis=-1)
        exp_term = ops.exp(-0.5 * ops.sum(((X_exp - means) ** 2) / vars_, axis=-1))
        return exp_term / norm  # (n_samples, n_components)

    def sample(self, n_samples=1, seed=None, **kwargs):
        # Sample component indices
        comp_idx = keras.random.categorical(ops.log(self.pi[None, :]), n_samples, seed=seed)
        comp_idx = ops.squeeze(comp_idx, axis=0)
        means = ops.take(self.means, comp_idx, axis=0)
        vars_ = ops.take(self.vars, comp_idx, axis=0)
        eps = keras.random.normal(ops.shape(means), seed=seed)
        samples = means + eps * ops.sqrt(vars_)
        return samples

    def posterior_sample(self, measurements, n_samples=1, seed=None, **kwargs):
        """
        Sample component indices from the posterior p(z|x) for each measurement.

        Args:
            measurements: Input data, shape (batch, n_features).
            n_samples: Number of posterior samples per measurement.
            seed: Random seed.

        Returns:
            Component indices, shape (batch, n_samples).
        """
        X = ops.convert_to_tensor(measurements, dtype="float32")
        gamma = self._e_step(X)  # (batch, n_components)
        # Sample n_samples times for each measurement
        comp_idx = keras.random.categorical(
            ops.log(gamma), n_samples, seed=seed
        )  # (batch, n_samples)
        # Return as (batch, n_samples)
        return comp_idx

    def log_density(self, data, **kwargs):
        X = ops.convert_to_tensor(data, dtype="float32")
        pdf = ops.sum(self._component_pdf(X) * self.pi, axis=1)
        return ops.log(pdf)


def match_means_covariances(means, true_means, covs, true_covs):
    """Match estimated means/covs to true ones.

    Uses greedy minimal distance assignment.

    Args:
        means: Estimated means (n_components, n_features).
        true_means: True means (n_components, n_features).
        covs: Estimated covariances (n_components, n_features, n_features).
        true_covs: True covariances (n_components, n_features, n_features).

    Returns:
        means_matched: Matched estimated means.
        true_means_matched: Matched true means.
        covs_matched: Matched estimated covariances.
        true_covs_matched: Matched true covariances.
    """
    diff = ops.expand_dims(means, 1) - ops.expand_dims(true_means, 0)
    cost = ops.sqrt(ops.sum(diff**2, axis=-1))
    row_ind, col_ind = linear_sum_assignment(cost)
    means_matched = ops.take(means, row_ind, axis=0)
    true_means_matched = ops.take(true_means, col_ind, axis=0)
    covs_matched = [covs[i] for i in row_ind]
    true_covs_matched = [true_covs[j] for j in col_ind]
    return means_matched, true_means_matched, covs_matched, true_covs_matched
