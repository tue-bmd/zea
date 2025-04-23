"""Generative models for usbmd.

- **Author(s)**: Tristan Stevens
- **Date**: 24/01/2025
"""

import abc

from usbmd.models.base import BaseModel


class GenerativeModel(abc.ABC):
    """Abstract base class for generative models."""

    @abc.abstractmethod
    def fit(self, data, **kwargs):
        """Fit the model to the data.

        Args:
            data: The data to fit the model to.
            **kwargs: Additional arguments to pass to the fitting procedure.
        """
        pass

    @abc.abstractmethod
    def sample(self, n_samples=1, **kwargs):
        """Sample from the model.

        Args:
            n_samples: Number of samples to generate.
            **kwargs: Additional arguments to pass to the sampling procedure.

        Returns:
            Samples from the model.
        """
        pass

    @abc.abstractmethod
    def posterior_sample(self, data, **kwargs):
        """Sample from the posterior distribution given data.

        Args:
            data: The data to condition the posterior on.
            **kwargs: Additional arguments to pass to the sampling procedure.

        Returns:
            Samples from the posterior distribution.
        """
        pass

    @abc.abstractmethod
    def log_likelihood(self, data, **kwargs):
        """Compute the log-likelihood of the data under the model.

        Args:
            data: The data to compute the log-likelihood for.
            **kwargs: Additional arguments.

        Returns:
            Log-likelihood of the data.
        """
        pass


class DeepGenerativeModel(GenerativeModel, BaseModel):
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
        self.built = False

    def fit(self, data, **kwargs):
        """Fit the model to the data.

        This implementation delegates to Keras's fit method.

        Args:
            data: The data to fit the model to.
            **kwargs: Additional arguments to pass to keras.Model.fit.

        Returns:
            Training history.
        """
        return self.network.fit(data, **kwargs)

    def build(self, input_shape=None):
        """Build the model architecture.

        Args:
            input_shape: Shape of the input tensor.
        """
        self.built = True

    @abc.abstractmethod
    def sample(self, n_samples=1, **kwargs):
        """Sample from the model."""
        pass

    @abc.abstractmethod
    def posterior_sample(self, data, **kwargs):
        """Sample from the posterior distribution given data."""
        pass

    @abc.abstractmethod
    def log_likelihood(self, data, **kwargs):
        """Compute the log-likelihood of the data under the model."""
        pass
