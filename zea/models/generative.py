"""Generative models for zea."""

import abc

from keras import ops

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
            measurements: The measurements $x$ to condition the posterior on, of
                shape `(*input_shape)`. May also be given as `(n_samples,
                *input_shape)` to condition each sample on a different
                measurement. Use :func:`zea.func.vmap` to sample from a batch of
                measurements.
            n_samples: Number of posterior samples to generate. This is the
                leading dimension of the output.
            **kwargs: Additional arguments to pass to the sampling procedure.

        Returns:
            Samples $z$ from the posterior $p(z \mid x)$, with `n_samples` as the
            leading dimension. The remaining dimensions are model-specific.
        """
        raise NotImplementedError("posterior_sample() must be implemented in subclasses.")

    @staticmethod
    def _as_measurement_batch(measurements, n_samples, event_ndim):
        """Give `measurements` a leading axis, and say whether it was already there.

        Posterior sampling conditions on a single measurement: either one shared by
        every sample, or exactly one measurement per requested sample. A leading axis
        of any other size is almost always a batch of measurements, which should be
        mapped over with :func:`zea.func.vmap` rather than silently misread here.

        Args:
            measurements: A single measurement of shape `(*event_shape)`, or of
                shape `(1 | n_samples, *event_shape)`.
            n_samples: Number of posterior samples requested.
            event_ndim: Rank of a single measurement, i.e. `len(event_shape)`.

        Returns:
            tuple: `(measurements, per_sample)`, where `measurements` has a leading
            axis and `per_sample` is True when the caller supplied one measurement
            per sample.

        Raises:
            ValueError: If `measurements` has neither the rank of a single
                measurement nor that of a batch of them, or if a leading axis of
                known size conditions neither every sample (size 1) nor one sample
                each (size `n_samples`).
        """
        measurements = ops.convert_to_tensor(measurements)
        # Rank is static on every Keras backend, so it is safe to branch on.
        ndim = ops.ndim(measurements)
        if ndim == event_ndim:
            return measurements[None], False
        if ndim != event_ndim + 1:
            raise ValueError(
                f"Expected measurements of rank {event_ndim} (a single measurement) or "
                f"{event_ndim + 1} (a leading axis of n_samples={n_samples}, or of 1 to "
                f"share one measurement with every sample), but got shape "
                f"{tuple(measurements.shape)}."
            )

        # Read the leading size off the static shape rather than `ops.shape`, which
        # returns a dynamic tensor for unknown dimensions under `tf.function` that
        # cannot be compared here. An unknown size is simply left unvalidated.
        batch_size = measurements.shape[0]
        if batch_size is None:
            return measurements, True
        # A singleton leading axis is shared by every sample, just like no axis.
        if batch_size == 1:
            return measurements, False
        if batch_size != n_samples:
            raise ValueError(
                f"Expected a leading axis of n_samples={n_samples} (one measurement per "
                f"sample) or of 1 (shared by every sample), but got {batch_size}. "
                "Use zea.func.vmap to sample from a batch of measurements."
            )
        return measurements, True

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
