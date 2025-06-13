"""Generative models for zea."""

import abc

from zea.models.base import BaseModel


class GenerativeModel(abc.ABC):
    """Abstract base class for generative models."""

    def fit(self, data, **kwargs):
        """Fit the model to the data.

        Args:
            data: The data to fit the model to.
            **kwargs: Additional arguments to pass to the fitting procedure.
        """
        raise NotImplementedError("fit() must be implemented in subclasses.")

    def sample(self, n_samples=1, **kwargs):
        r"""Draw samples $x \sim p(x)$ from the model.

        Args:
            n_samples: Number of samples to generate.
            **kwargs: Additional arguments to pass to the sampling procedure.

        Returns:
            Samples $x$ from the model distribution $p(x)$.
        """
        raise NotImplementedError("sample() must be implemented in subclasses.")

    def posterior_sample(self, measurements, n_samples=1, **kwargs):
        r"""Draw samples $z \sim p(z \mid x)$ from the posterior given measurements.

        Args:
            measurements: The measurements $x$ to condition the posterior on.
            n_samples: Number of posterior samples to generate. This will add
                an additional dimension to the output. For instance,
                if `measurements` has shape `(batch_size, ...)`, the output will
                have shape `(batch_size, n_samples, ...)`.
            **kwargs: Additional arguments to pass to the sampling procedure.

        Returns:
            Samples $z$ from the posterior $p(z \mid x)$.
        """
        raise NotImplementedError("posterior_sample() must be implemented in subclasses.")

    def log_density(self, data, **kwargs):
        r"""Compute the log-density $\log p(x)$ of the data under the model.

        Args:
            data: The data $x$ to compute the log-density for.
            **kwargs: Additional arguments.

        Returns:
            Log-density $\log p(x)$ of the data.
        """
        raise NotImplementedError("log_density() must be implemented in subclasses.")


class DeepGenerativeModel(BaseModel, GenerativeModel):
    """Base class for deep generative models.

    Inherits from both GenerativeModel and BaseModel to combine
    generative capabilities with Keras model functionality.
    """

    def __init__(self, name="deep_generative_model", **kwargs):
        """Initialize a deep generative model.

        Args:
            name: Name of the model.
            **kwargs: Additional arguments to pass to BaseModel.
        """
        BaseModel.__init__(self, name=name, **kwargs)
