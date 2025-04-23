import keras
import numpy as np
from keras import ops

from usbmd.models.generative import GenerativeModel


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

    def _initialize(self, X):
        # X: (n_samples, n_features)
        n_samples = ops.shape(X)[0]
        # Initialize means by randomly picking data points
        idx = keras.random.shuffle(ops.arange(n_samples), seed=self.seed)[
            : self.n_components
        ]
        self.means = ops.take(X, idx, axis=0)  # (n_components, n_features)
        # Initialize variances to variance of data
        var = ops.var(X, axis=0)
        self.vars = ops.ones((self.n_components, self.n_features)) * var
        # Initialize mixture weights uniformly
        self.pi = ops.ones((self.n_components,)) / self.n_components
        self._initialized = True

    def _e_step(self, X):
        # X: (n_samples, n_features)
        n_samples = ops.shape(X)[0]
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
        vars_ = ops.sum(
            gamma[..., None] * (X_exp - means_exp) ** 2, axis=0
        ) / ops.expand_dims(Nk, -1)
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
        comp_idx = keras.random.categorical(
            ops.log(self.pi[None, :]), n_samples, seed=seed
        )
        comp_idx = ops.squeeze(comp_idx, axis=0)
        means = ops.take(self.means, comp_idx, axis=0)
        vars_ = ops.take(self.vars, comp_idx, axis=0)
        eps = keras.random.normal(ops.shape(means), seed=seed)
        samples = means + eps * ops.sqrt(vars_)
        return samples

    def posterior_sample(self, data, seed=None, **kwargs):
        X = ops.convert_to_tensor(data, dtype="float32")
        gamma = self._e_step(X)
        comp_idx = keras.random.categorical(ops.log(gamma), 1, seed=seed)
        comp_idx = ops.squeeze(comp_idx, axis=-1)
        return comp_idx

    def log_likelihood(self, data, **kwargs):
        X = ops.convert_to_tensor(data, dtype="float32")
        pdf = ops.sum(self._component_pdf(X) * self.pi, axis=1)
        return ops.log(pdf)
