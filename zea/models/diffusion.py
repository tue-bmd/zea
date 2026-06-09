"""
Diffusion models for ultrasound image generation and posterior sampling.

To try this model, simply load one of the available presets:

.. doctest::

    >>> from zea.models.diffusion import DiffusionModel

    >>> model = DiffusionModel.from_preset("diffusion-echonet-dynamic")  # doctest: +SKIP

.. seealso::
    A tutorial notebook where this model is used:
    :doc:`../notebooks/models/diffusion_model_example`.

"""

from __future__ import annotations

import abc
from typing import Literal

import keras
from keras import ops

from zea.backend import _import_tf, jit
from zea.backend.autograd import AutoGrad
from zea.func.tensor import L2, fori_loop, split_seed
from zea.internal.core import Object
from zea.internal.operators import Operator
from zea.internal.registry import diffusion_guidance_registry, model_registry, operator_registry
from zea.internal.utils import fn_requires_argument
from zea.models.dense import get_time_conditional_dense_network
from zea.models.dit import get_time_conditional_dit_network
from zea.models.generative import DeepGenerativeModel
from zea.models.preset_utils import register_presets
from zea.models.presets import diffusion_model_presets
from zea.models.unet import get_time_conditional_unetwork
from zea.models.utils import LossTrackerWrapper

tf = _import_tf()


@model_registry(name="diffusion")
class DiffusionModel(DeepGenerativeModel):
    """Implementation of a diffusion generative model.
    Heavily inspired from https://keras.io/examples/generative/ddim/
    """

    def __init__(
        self,
        input_shape,
        input_range=(0, 1),
        min_signal_rate=0.02,
        max_signal_rate=0.95,
        network_name="unet_time_conditional",
        network_kwargs=None,
        name="diffusion_model",
        guidance="dps",
        operator="inpainting",
        ema_val=0.999,
        min_t=0.0,
        max_t=1.0,
        **kwargs,
    ):
        """Initialize a diffusion model.

        Args:
            input_shape: Shape of the input data. Typically of the form
                `(height, width, channels)` for images.
            input_range: Range of the input data.
            min_signal_rate: Minimum signal rate for the diffusion schedule.
            max_signal_rate: Maximum signal rate for the diffusion schedule.
            network_name: Name of the network architecture to use. Options are
                "unet_time_conditional", "dense_time_conditional", or
                "dit_time_conditional" (Diffusion Transformer).
            network_kwargs: Additional keyword arguments for the network.
            name: Name of the model.
            guidance: Guidance method to use. Can be a string, or dict with
                "name" and "params" keys. Additionally, can be a `DiffusionGuidance` object.
            operator: Operator to use. Can be a string, or dict with
                "name" and "params" keys. Additionally, can be a `Operator` object.
            ema_val: Exponential moving average value for the network weights.
            min_t: Minimum diffusion time for sampling during training.
            max_t: Maximum diffusion time for sampling during training.
            **kwargs: Additional arguments.
        """
        super().__init__(name=name, **kwargs)

        self.input_shape = input_shape
        self.input_range = input_range
        self.min_signal_rate = min_signal_rate
        self.max_signal_rate = max_signal_rate
        self.network_name = network_name
        self.network_kwargs = network_kwargs or {}
        self.ema_val = ema_val

        # reverse diffusion (i.e. sampling) goes from t = max_t to t = min_t
        self.min_t = min_t
        self.max_t = max_t

        if network_name == "unet_time_conditional":
            self.network = get_time_conditional_unetwork(
                image_shape=self.input_shape,
                **self.network_kwargs,
            )
        elif network_name == "dense_time_conditional":
            assert len(input_shape) == 1, "Dense network only supports 1D input"
            self.network = get_time_conditional_dense_network(
                input_dim=self.input_shape[0],
                **self.network_kwargs,
            )
        elif network_name == "dit_time_conditional":
            self.network = get_time_conditional_dit_network(
                image_shape=self.input_shape,
                **self.network_kwargs,
            )
        else:
            raise ValueError("Invalid network name provided.")

        # Also initialize the exponential moving average network
        self.ema_network = keras.models.clone_model(self.network)
        self.ema_network.trainable = False

        self.image_loss_tracker = LossTrackerWrapper("i_loss")
        self.noise_loss_tracker = LossTrackerWrapper("n_loss")

        # for storing intermediate results (i.e. diffusion trajectory)
        self.track_progress_interval = 1
        self.track_progress = []

        # for guidance / conditional sampling
        self.guidance_fn = None
        self.operator = None
        self._init_operator_and_guidance(operator, guidance)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape,
                "input_range": self.input_range,
                "min_signal_rate": self.min_signal_rate,
                "max_signal_rate": self.max_signal_rate,
                "min_t": self.min_t,
                "max_t": self.max_t,
                "network_name": self.network_name,
                "network_kwargs": self.network_kwargs,
                "ema_val": self.ema_val,
            }
        )
        return config

    def _init_operator_and_guidance(self, operator, guidance):
        if operator is not None:
            if isinstance(operator, str):
                operator_class = operator_registry[operator]
                self.operator = operator_class()
            elif isinstance(operator, Operator):
                self.operator = operator
            elif isinstance(operator, dict):
                operator_class = operator_registry[operator["name"]]
                if "params" not in operator:
                    operator["params"] = {}
                if (
                    fn_requires_argument(operator_class.__init__, "image_range")
                    and "image_range" not in operator["params"]
                ):
                    operator["params"]["image_range"] = self.input_range
                self.operator = operator_class(**operator["params"])
            else:
                raise ValueError(
                    f"Invalid operator provided, must be a string, dict or "
                    f"Operator object, got {operator}"
                )

        if guidance is not None:
            assert operator is not None, "Operator must be provided for guidance"
            if isinstance(guidance, str):
                guidance_class = diffusion_guidance_registry[guidance]
                self.guidance_fn = guidance_class(
                    diffusion_model=self,
                    operator=self.operator,
                )
            elif isinstance(guidance, DiffusionGuidance):
                self.guidance_fn = guidance
            elif isinstance(guidance, dict):
                guidance_class = diffusion_guidance_registry[guidance["name"]]
                self.guidance_fn = guidance_class(
                    diffusion_model=self, operator=self.operator, **guidance["params"]
                )
            else:
                raise ValueError(
                    f"Invalid guidance provided, must be a string, dict or "
                    f"DiffusionGuidance object, got {guidance}"
                )

    def call(self, inputs, training: bool = False, network=None, **kwargs):
        """Call the score network.

        Args:
            inputs: A list ``[noisy_images, noise_rates_squared]`` as
                expected by the underlying time-conditional network.
            training: Whether to run in training mode.  When ``False`` and
                ``network`` is ``None``, the EMA network is used.
            network: Explicit network to call.  If ``None``, the EMA network
                is used during inference and the online network during
                training.
            **kwargs: Extra keyword arguments forwarded to the network.

        Returns:
            Predicted noise tensor of the same shape as the input images.
        """
        if network is None:
            network = self.network if training else self.ema_network

        return network(inputs, training=training, **kwargs)

    def sample(self, n_samples=1, n_steps=20, seed=None, **kwargs):
        """Sample from the model.

        Args:
            n_samples: Number of samples to generate.
            n_steps: Number of diffusion steps.
            seed: Random seed generator.
            **kwargs: Additional arguments.

        Returns:
            Generated samples of shape `(n_samples, *input_shape)`.
        """
        seed, seed1 = split_seed(seed, 2)

        # Generate random noise
        noise = keras.random.normal(
            shape=(n_samples, *self.input_shape),
            seed=seed1,
        )
        # Reverse diffusion process
        return self.reverse_diffusion(
            initial_noise=noise, diffusion_steps=n_steps, seed=seed, **kwargs
        )

    def posterior_sample(
        self,
        measurements,
        n_samples: int = 1,
        n_steps: int = 20,
        initial_step: int = 0,
        initial_samples=None,
        seed=None,
        **kwargs,
    ):
        """Sample from the posterior distribution given measurements.

        Args:
            measurements: Input measurements. Typically of shape
                ``(batch_size, *input_shape)``.
            n_samples: Number of posterior samples to generate.
                Will generate ``n_samples`` samples for each measurement
                in the ``measurements`` batch.
            n_steps: Number of diffusion steps.
            initial_step: Step at which to begin the reverse diffusion loop.
                ``0`` runs all ``diffusion_steps`` steps from maximum noise.
                Higher values skip early (high-noise) steps and require
                ``initial_samples`` to be provided. Number of effective steps
                will be ``diffusion_steps - initial_step``.
            initial_samples: Optional initial samples to warm-start the
                diffusion process. The diffusion process now starts from a
                *noised* version of these samples. This can be used to speed
                up the diffusion process.
                When ``initial_step == 0``, samples are noised at the maximum
                noise level (``max_t``). When ``initial_step > 0``, samples
                are noised at the noise level corresponding to ``initial_step``.
                These ``initial_samples`` can be initial guesses such as solutions
                of previous frames (for sequences), see for instance
                `SeqDiff <https://arxiv.org/abs/2409.05399>`_.
                Must be of shape ``(batch_size, n_samples, *input_shape)``.
            seed: Random seed generator.
            **kwargs: Additional arguments passed to
                :meth:`reverse_conditional_diffusion`.

        Returns:
            Posterior samples ``p(x|y)`` of shape
            ``(batch_size, n_samples, *input_shape)``.

        """
        batch_size = ops.shape(measurements)[0]
        shape = (batch_size, n_samples, *self.input_shape)

        def _tile_with_sample_dim(tensor):
            """Tile the tensor with an additional sample dimension."""
            shape = ops.shape(tensor)
            tensor = ops.repeat(tensor[:, None], n_samples, axis=1)  # (batch, n_samples, ...)
            return ops.reshape(tensor, (-1, *shape[1:]))

        measurements = _tile_with_sample_dim(measurements)
        if initial_samples is not None:
            initial_samples = ops.reshape(initial_samples, (-1, *self.input_shape))
        if "mask" in kwargs:
            kwargs["mask"] = _tile_with_sample_dim(kwargs["mask"])

        seed1, seed2 = split_seed(seed, 2)

        initial_noise = keras.random.normal(
            shape=(batch_size * n_samples, *self.input_shape),
            seed=seed1,
        )

        out = self.reverse_conditional_diffusion(
            measurements=measurements,
            initial_noise=initial_noise,
            diffusion_steps=n_steps,
            initial_samples=initial_samples,
            initial_step=initial_step,
            seed=seed2,
            **kwargs,
        )  # ( batch_size * n_samples, *self.input_shape)

        return ops.reshape(out, shape)  # (batch_size, n_samples, *input_shape)

    def log_likelihood(self, data, **kwargs):
        """Approximate log-likelihood of the data under the model.

        Args:
            data: Data to compute log-likelihood for.
            **kwargs: Additional arguments.

        Returns:
            Approximate log-likelihood.
        """
        # This is a placeholder for likelihood estimation
        raise NotImplementedError("Likelihood computation for diffusion models not implemented yet")

    @property
    def metrics(self):
        """Metrics for training."""
        return [*self.noise_loss_tracker, *self.image_loss_tracker]

    def train_step(self, data):
        """Custom train step so we can call model.fit() on the diffusion model.
        Note:
            - Only implemented for the TensorFlow backend.
        """
        if tf is None:
            raise NotImplementedError(
                "DiffusionModel.train_step is only implemented for the TensorFlow backend."
            )

        # Get batch size and image shape
        batch_size, *input_shape = ops.shape(data)
        n_dims = len(input_shape)

        # Generate random noise
        noises = keras.random.normal(shape=ops.shape(data))

        # Sample uniform random diffusion times in [min_t, max_t]
        diffusion_times = keras.random.uniform(
            shape=ops.stack([batch_size, *([1] * n_dims)]),
            minval=self.min_t,
            maxval=self.max_t,
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        # Mix data and noises
        noisy_data = signal_rates * data + noise_rates * noises

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(
                noisy_data, noise_rates, signal_rates, training=True
            )
            noise_loss = self.loss(noises, pred_noises)
            image_loss = self.loss(data, pred_images)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights.
        # ema_network is used for inference / sampling
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema_val * ema_weight + (1 - self.ema_val) * weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """
        Custom test step so we can call model.fit() on the diffusion model.
        """
        batch_size, *input_shape = ops.shape(data)
        n_dims = len(input_shape)

        noises = keras.random.normal(shape=ops.shape(data))

        # sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=ops.stack([batch_size, *([1] * n_dims)]),
            minval=self.min_t,
            maxval=self.max_t,
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * data + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(data, pred_images)

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        return {m.name: m.result() for m in self.metrics}

    def diffusion_schedule(self, diffusion_times):
        """Cosine diffusion schedule.

        Implements the cosine schedule from `Nichol & Dhariwal (2021)
        <https://arxiv.org/abs/2102.09672>`_.

        The noisy image at time ``t`` is defined as:

        Args:
            diffusion_times: Tensor of diffusion times in ``[min_t, max_t]``.

        Returns:
            A ``(noise_rates, signal_rates)`` tuple of tensors with the
            same shape as ``diffusion_times``.
        """  # noqa: E501
        # diffusion times -> angles
        start_angle = ops.cast(ops.arccos(self.max_signal_rate), "float32")
        end_angle = ops.cast(ops.arccos(self.min_signal_rate), "float32")

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = ops.cos(diffusion_angles)
        noise_rates = ops.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1
        return noise_rates, signal_rates

    def linear_diffusion_schedule(self, diffusion_times):
        """Create a linear diffusion schedule"""

        def _compute_alpha_t(t):
            """Compute alpha_t for linear diffusion schedule"""
            return ops.prod(1 - diffusion_times[:t], axis=diffusion_times.shape[1:])

        alphas = ops.vectorized_map(_compute_alpha_t, ops.arange(len(diffusion_times)))
        signal_rates = ops.sqrt(alphas)
        noise_rates = ops.sqrt(1 - alphas)
        return signal_rates, noise_rates

    def denoise(
        self,
        noisy_images,
        noise_rates,
        signal_rates,
        training: bool,
        network=None,
    ):
        """Predict the noise component and derive the clean-image estimate.

        Uses the score network to predict the noise ``ε`` in ``x_t``, then
        computes the Tweedie estimate of ``x_0``.

        Args:
            noisy_images: Noisy images ``x_t`` of shape
                ``(n_images, *input_shape)``.
            noise_rates: Noise rates at the current diffusion time, broadcastable
                to ``noisy_images``.
            signal_rates: Signal rates at the current diffusion time,
                broadcastable to ``noisy_images``.
            training: Whether to call the network in training mode.
            network: Explicit network to use.  If ``None``, chosen based on
                ``training`` (see :meth:`call`).

        Returns:
            A ``(pred_noises, pred_images)`` tuple where ``pred_noises`` is
            the predicted noise ``ε`` and ``pred_images`` is the Tweedie
            estimate of ``x_0``.
        """

        pred_noises = self([noisy_images, noise_rates**2], training=training, network=network)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion_step(
        self,
        shape,
        pred_images,
        pred_noises,
        signal_rates,
        next_signal_rates,
        next_noise_rates,
        seed=None,
        stochastic_sampling=False,
    ):
        """A single reverse diffusion step.

        Args:
            shape: Shape of the input tensor.
            pred_images: Predicted images.
            pred_noises: Predicted noises.
            signal_rates: Current signal rates.
            next_signal_rates: Next signal rates.
            next_noise_rates: Next noise rates.
            seed: Random seed generator.
            stochastic_sampling: Whether to use stochastic sampling (DDPM).

        Returns:
            next_noisy_images: Noisy images after the reverse diffusion step.
        """
        if not stochastic_sampling:
            next_noisy_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
            return next_noisy_images

        alpha_prev = signal_rates**2
        alpha = next_signal_rates**2

        sigma_t = ops.sqrt((1 - alpha) / (1 - alpha_prev)) * ops.sqrt(1 - alpha_prev / alpha)
        epsilon = keras.random.normal(shape=shape, seed=seed)

        next_noise_rates = ops.sqrt(1 - alpha - sigma_t**2)
        next_noisy_images = (
            next_signal_rates * pred_images + next_noise_rates * pred_noises + sigma_t * epsilon
        )
        return next_noisy_images

    def solver_step(
        self,
        noisy_images,
        noise_rates,
        signal_rates,
        next_noise_rates,
        next_signal_rates,
        shape,
        network=None,
        training: bool = False,
        seed=None,
        stochastic_sampling: bool = False,
    ):
        """Take a single solver step from ``x_t`` to ``x_{t-Δt}``.

        Denoises the current noisy images and applies one reverse-diffusion
        (DDIM / DDPM) update.  This is the integration step used by
        :meth:`reverse_diffusion`; subclasses may override it to implement
        higher-order ODE solvers (see
        :meth:`~zea.models.flow_matching.FlowMatchingModel.solver_step`).

        Args:
            noisy_images: Current noisy images ``x_t``.
            noise_rates: Noise rates at the current time.
            signal_rates: Signal rates at the current time.
            next_noise_rates: Noise rates at the next (lower) time.
            next_signal_rates: Signal rates at the next (lower) time.
            shape: Shape of the image tensor.
            network: Explicit network to use (``None`` selects based on
                ``training``).
            training: Whether to call the network in training mode.
            seed: Random seed generator (for stochastic sampling).
            stochastic_sampling: Whether to use stochastic (DDPM) sampling.

        Returns:
            A ``(next_noisy_images, pred_images)`` tuple where
            ``next_noisy_images`` is ``x_{t-Δt}`` and ``pred_images`` is the
            clean-image estimate ``x̂₀`` at the current step.
        """
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=training, network=network
        )
        next_noisy_images = self.reverse_diffusion_step(
            shape=shape,
            pred_images=pred_images,
            pred_noises=pred_noises,
            signal_rates=signal_rates,
            next_signal_rates=next_signal_rates,
            next_noise_rates=next_noise_rates,
            seed=seed,
            stochastic_sampling=stochastic_sampling,
        )
        return next_noisy_images, pred_images

    def reverse_diffusion(
        self,
        initial_noise,
        diffusion_steps: int,
        initial_samples=None,
        initial_step: int = 0,
        stochastic_sampling: bool = False,
        seed: keras.random.SeedGenerator | None = None,
        verbose: bool = True,
        track_progress_type: Literal[None, "x_0", "x_t"] = "x_0",
        disable_jit: bool = False,
        training: bool = False,
        network_type: Literal[None, "main", "ema"] = None,
    ):
        """Reverse diffusion process to generate images from noise.

        Args:
            initial_noise: Initial noise tensor of shape
                ``(n_images, *input_shape)``.
            diffusion_steps: Total number of diffusion steps.
            initial_samples: Optional initial samples to warm-start the
                diffusion process. The diffusion process now starts from a
                *noised* version of these samples.
                When ``initial_step == 0``, samples are noised at the maximum
                noise level (``max_t``). When ``initial_step > 0``, samples
                are noised at the noise level corresponding to ``initial_step``.
            initial_step: Step at which to begin the reverse diffusion loop.
                ``0`` runs all ``diffusion_steps`` steps from maximum noise.
                Higher values skip early (high-noise) steps and require
                ``initial_samples`` to be provided. Number of effective steps
                will be ``diffusion_steps - initial_step``.
            stochastic_sampling: Whether to use stochastic DDPM sampling
                instead of deterministic DDIM sampling.
            seed: Random seed generator.
            verbose: Whether to show a Keras progress bar.
            track_progress_type: Intermediate output to store at each step.
                ``"x_0"`` stores the Tweedie-denoised estimate; ``"x_t"``
                stores the noisy intermediate image; ``None`` disables
                tracking.
            disable_jit: Whether to disable JIT compilation.
            training: Whether to call the network in training mode.
            network_type: Which network weights to use. ``"main"`` uses the
                online network, ``"ema"`` uses the exponential-moving-average
                network. If ``None``, the choice is determined by the
                ``training`` argument.

        Returns:
            Generated images of shape ``(n_images, *input_shape)``.
        """
        num_images, *input_shape = ops.shape(initial_noise)
        step_size, progbar = self.prepare_diffusion(diffusion_steps, initial_step, verbose)

        n_dims = len(input_shape)

        base_diffusion_times = ops.ones((num_images, *[1] * n_dims)) * self.max_t

        next_noisy_images = self.prepare_schedule(
            base_diffusion_times,
            initial_noise,
            initial_samples,
            initial_step,
            step_size,
        )

        def step_fn(step, loop_state):
            noisy_images, pred_images, seed = loop_state

            # separate the current noisy image to its components
            diffusion_times = base_diffusion_times - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)

            # select the network used for sampling
            if network_type == "ema":
                network = self.ema_network
            elif network_type == "main":
                network = self.network
            else:
                network = None

            seed, seed1 = split_seed(seed, 2)

            next_noisy_images, pred_images = self.solver_step(
                noisy_images=noisy_images,
                noise_rates=noise_rates,
                signal_rates=signal_rates,
                next_noise_rates=next_noise_rates,
                next_signal_rates=next_signal_rates,
                shape=(num_images, *input_shape),
                network=network,
                training=training,
                seed=seed1,
                stochastic_sampling=stochastic_sampling,
            )

            # this new noisy image will be used in the next step
            if progbar is not None:
                progbar.update(step + 1)

            self.store_progress(step, track_progress_type, next_noisy_images, pred_images)

            loop_state = (next_noisy_images, pred_images, seed)

            return loop_state

        _, pred_images, _ = fori_loop(
            initial_step,
            diffusion_steps,
            step_fn,
            (
                next_noisy_images,
                ops.zeros_like(initial_noise),
                seed,
            ),
            # can't jit this with progbar or tracking intermediate values
            disable_jit=verbose or track_progress_type or disable_jit,
        )

        return pred_images

    def reverse_conditional_diffusion(
        self,
        measurements,
        initial_noise,
        diffusion_steps: int,
        initial_samples=None,
        initial_step: int = 0,
        stochastic_sampling: bool = False,
        seed=None,
        verbose: bool = False,
        track_progress_type: Literal[None, "x_0", "x_t"] = "x_0",
        disable_jit=False,
        **kwargs,
    ):
        """Reverse diffusion process conditioned on some measurement.

        Effectively performs diffusion posterior sampling ``p(x_0 | y)``
        by interleaving reverse diffusion steps with gradient-based guidance
        (e.g. DPS or DDS).

        Args:
            measurements: Conditioning observations of shape
                ``(n_images, *measurement_shape)``.
            initial_noise: Initial noise tensor of shape
                ``(n_images, *input_shape)``.
            diffusion_steps: Total number of diffusion steps.
            initial_samples: Optional initial samples to warm-start the
                diffusion process. The diffusion process now starts from a
                *noised* version of these samples.
                When ``initial_step == 0``, samples are noised at the maximum
                noise level (``max_t``). When ``initial_step > 0``, samples
                are noised at the noise level corresponding to ``initial_step``.
            initial_step: Step at which to begin the reverse diffusion loop.
                ``0`` runs all ``diffusion_steps`` steps from maximum noise.
                Higher values skip early (high-noise) steps and require
                ``initial_samples`` to be provided. Number of effective steps
                will be ``diffusion_steps - initial_step``.
            stochastic_sampling: Whether to use stochastic DDPM sampling
                instead of deterministic DDIM sampling.
            seed: Random seed generator.
            verbose: Whether to show a Keras progress bar with the guidance
                error at each step.
            track_progress_type: Intermediate output to store at each step.
                ``"x_0"`` stores the Tweedie-denoised estimate; ``"x_t"``
                stores the noisy intermediate image; ``None`` disables
                tracking.
            disable_jit: Whether to disable JIT compilation.
            **kwargs: Additional keyword arguments forwarded to the guidance
                function and operator (e.g. ``omega``, ``mask``).

        Returns:
            Generated images of shape ``(n_images, *input_shape)``.

        """
        num_images, *input_shape = ops.shape(initial_noise)

        step_size, progbar = self.prepare_diffusion(
            diffusion_steps,
            initial_step,
            verbose,
        )

        n_dims = len(input_shape)
        base_diffusion_times = ops.ones((num_images, *[1] * n_dims)) * self.max_t

        next_noisy_images = self.prepare_schedule(
            base_diffusion_times,
            initial_noise,
            initial_samples,
            initial_step,
            step_size,
        )

        def step_fn(step, loop_state):
            noisy_images, pred_images, seed = loop_state

            diffusion_times = base_diffusion_times - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)

            gradients, (error, (pred_noises, pred_images)) = self.guidance_fn(
                noisy_images,
                measurements=measurements,
                noise_rates=noise_rates,
                signal_rates=signal_rates,
                **kwargs,
            )

            seed, seed1 = split_seed(seed, 2)
            next_noisy_images = self.reverse_diffusion_step(
                shape=(num_images, *input_shape),
                pred_images=pred_images,
                pred_noises=pred_noises,
                signal_rates=signal_rates,
                next_signal_rates=next_signal_rates,
                next_noise_rates=next_noise_rates,
                seed=seed1,
                stochastic_sampling=stochastic_sampling,
            )

            next_noisy_images = next_noisy_images - gradients
            pred_images = pred_images - gradients

            # this new noisy image will be used in the next step
            if verbose:
                progbar.update(step + 1, [("error", error)])

            self.store_progress(step, track_progress_type, next_noisy_images, pred_images)

            loop_state = (next_noisy_images, pred_images, seed)

            return loop_state

        _, pred_images, _ = fori_loop(
            initial_step,
            diffusion_steps,
            step_fn,
            (
                next_noisy_images,
                ops.zeros_like(initial_noise),
                seed,
            ),
            # can't jit this with progbar or tracking intermediate values
            disable_jit=verbose or track_progress_type or disable_jit,
        )

        return pred_images

    def prepare_diffusion(
        self,
        diffusion_steps: int,
        initial_step: int,
        verbose: bool,
        disable_jit: bool = False,
    ):
        """Prepare the diffusion process.

        Validates ``initial_step``, computes the step size, and optionally
        creates a Keras progress bar.

        Args:
            diffusion_steps: Total number of diffusion steps.
            initial_step: Step index at which reverse diffusion begins.
                Must satisfy ``0 <= initial_step < diffusion_steps``.
            verbose: Whether to create a Keras :class:`~keras.utils.Progbar`.
            disable_jit: When ``True``, skip the ``initial_step`` range
                assertions (required when values are runtime tensors).

        Returns:
            A ``(step_size, progbar)`` tuple where ``step_size`` is the
            uniform time increment per step and ``progbar`` is a
            :class:`~keras.utils.Progbar` instance or ``None``.
        """
        # Asserts
        if not disable_jit:
            assert initial_step >= 0, f"initial_step must be non-negative, got {initial_step}"
            assert initial_step < diffusion_steps, (
                f"initial_step must be less than diffusion_steps, got {initial_step}"
            )

        step_size = self.max_t / diffusion_steps

        if verbose:
            progbar = keras.utils.Progbar(diffusion_steps, verbose=verbose)
        else:
            progbar = None

        self.start_track_progress(diffusion_steps)

        return step_size, progbar

    def prepare_schedule(
        self,
        base_diffusion_times,
        initial_noise,
        initial_samples,
        initial_step: int,
        step_size: float,
    ):
        """Prepare the starting noisy images for the reverse diffusion loop.

        Constructs the initial ``x_t`` tensor that is fed into the first
        diffusion step.  Three cases are handled:

        - ``initial_samples`` provided and ``initial_step > 0``:
          samples are mixed with noise at the noise level that corresponds
          to ``initial_step``, skipping the highest-noise diffusion steps.
        - ``initial_samples`` provided and ``initial_step == 0``:
          samples are mixed with noise at the maximum noise level
          (``max_t``), running the full diffusion process from a noised
          version of the samples.x
        - ``initial_samples is None`` and ``initial_step == 0``:
          the starting point is pure noise (``initial_noise``).

        Args:
            base_diffusion_times: Tensor of shape
                ``(n_images, *[1]*n_dims)`` filled with ``max_t``.
            initial_noise: Pure noise tensor of shape
                ``(n_images, *input_shape)``.
            initial_samples: Optional samples of shape ``(n_images, *input_shape)``.
                The diffusion process always starts from a *noised*
                version of these samples.
            initial_step: Step index at which reverse diffusion begins.
            step_size: Uniform time increment per diffusion step.

        Returns:
            Noisy images tensor of shape ``(n_images, *input_shape)`` to
            use as the starting point ``x_t`` for the diffusion loop.
        """
        # We can optionally start with a set of samples that are partially noised
        if initial_samples is not None and initial_step > 0:
            starting_diffusion_times = base_diffusion_times - ((initial_step - 1) * step_size)
            noise_rates, signal_rates = self.diffusion_schedule(starting_diffusion_times)
            next_noisy_images = signal_rates * initial_samples + noise_rates * initial_noise
        elif initial_samples is not None:
            noise_rates, signal_rates = self.diffusion_schedule(base_diffusion_times)
            next_noisy_images = signal_rates * initial_samples + noise_rates * initial_noise
        elif initial_samples is None and initial_step == 0:
            # important line:
            # at the first sampling step, the "noisy image" is pure noise
            # but its signal rate is assumed to be nonzero (min_signal_rate)
            next_noisy_images = initial_noise
        else:
            raise ValueError(
                "Why are you trying to do this? Initial samples should be provided "
                "if initial_step is greater than 0 (i.e. you want to start with "
                "a partially noised image)"
            )
        return next_noisy_images

    def start_track_progress(self, diffusion_steps: int, initial_step: int = 0):
        """Initialize progress tracking for the diffusion process.

        Resets :attr:`track_progress` and sets
        :attr:`track_progress_interval` so that at most 50 frames are
        stored during the diffusion trajectory (to keep memory usage
        bounded for large step counts).

        Args:
            diffusion_steps: Total number of diffusion steps.
            initial_step: Step index at which reverse diffusion begins.
        """
        self.track_progress = []
        remaining = max(1, diffusion_steps - int(initial_step))
        if remaining > 50:
            self.track_progress_interval = remaining // 50
        else:
            self.track_progress_interval = 1

    def store_progress(
        self,
        step: int,
        track_progress_type: Literal[None, "x_0", "x_t"],
        next_noisy_images,
        pred_images,
    ):
        """Store an intermediate diffusion frame in :attr:`track_progress`.

        Frames are stored every :attr:`track_progress_interval` steps.
        Does nothing when ``track_progress_type`` is ``None``.

        Args:
            step: Current diffusion step index.
            track_progress_type: Which tensor to store.  ``"x_0"`` stores
                the Tweedie-denoised estimate (predicted clean image);
                ``"x_t"`` stores the noisy intermediate image.
            next_noisy_images: Noisy images ``x_t`` after the current step.
            pred_images: Predicted clean images ``x_0`` at the current step.
        """
        if not track_progress_type:
            return
        if step % self.track_progress_interval == 0:
            if track_progress_type == "x_0":
                self.track_progress.append(ops.convert_to_numpy(pred_images))
            elif track_progress_type == "x_t":
                self.track_progress.append(ops.convert_to_numpy(next_noisy_images))
            else:
                raise ValueError("Invalid track_progress_type")


register_presets(diffusion_model_presets, DiffusionModel)


class DiffusionGuidance(abc.ABC, Object):
    """Base class for diffusion guidance methods."""

    def __init__(
        self,
        diffusion_model: DiffusionModel,
        operator: Operator,
        disable_jit: bool = False,
    ):
        """Initialize the diffusion guidance.

        Args:
            diffusion_model: The diffusion model to use for guidance.
            operator: The forward operator :math:`A` that maps clean images
                to the measurement space.
            disable_jit: Whether to disable JIT compilation of the guidance
                function.
        """
        super().__init__()

        self.diffusion_model = diffusion_model
        self.operator = operator
        self.disable_jit = disable_jit
        self.setup()

    @abc.abstractmethod
    def setup(self):
        """Setup the guidance function. Should be implemented by subclasses."""
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """Call the guidance function."""
        raise NotImplementedError


@diffusion_guidance_registry(name="dps")
class DPS(DiffusionGuidance):
    """Diffusion Posterior Sampling guidance."""

    def setup(self):
        """Setup the autograd function for DPS."""
        self.autograd = AutoGrad()
        self.autograd.set_function(self.compute_error)
        self.gradient_fn = self.autograd.get_gradient_and_value_jit_fn(
            has_aux=True,
            disable_jit=self.disable_jit,
        )

    def compute_error(
        self,
        noisy_images,
        measurements,
        noise_rates,
        signal_rates,
        omega: float,
        **kwargs,
    ):
        r"""Compute the DPS measurement error for gradient computation.

        Following the DPS implementation, the
        loss is a standard L2 norm.

        Args:
            noisy_images: Noisy images ``x_t`` of shape
                ``(n_images, *input_shape)``.
            measurements: Target observations ``y``.
            noise_rates: Noise rates at the current diffusion time.
            signal_rates: Signal rates at the current diffusion time.
            omega: Scalar step-size weight for the measurement gradient.
            **kwargs: Additional keyword arguments forwarded to the
                operator's ``forward`` method (e.g. ``mask``).

        Returns:
            A ``(measurement_error, (pred_noises, pred_images))`` tuple
            where ``measurement_error`` is the scalar loss and
            ``pred_noises`` / ``pred_images`` are the denoiser outputs.
        """
        pred_noises, pred_images = self.diffusion_model.denoise(
            noisy_images,
            noise_rates,
            signal_rates,
            training=False,
        )

        # See the original DPS implementation
        # https://github.com/DPS2022/diffusion-posterior-sampling/blob/effbde7325b22ce8dc3e2c06c160c021e743a12d/guided_diffusion/condition_methods.py#L31  # noqa: E501
        # As well as interesting discussion on the DPS loss:
        # https://github.com/DPS2022/diffusion-posterior-sampling/issues/20
        measurement_error = omega * L2(measurements - self.operator.forward(pred_images, **kwargs))

        return measurement_error, (pred_noises, pred_images)

    def __call__(self, noisy_images, **kwargs):
        """Compute DPS gradients and denoiser outputs.

        Calls the JIT-compiled gradient function obtained from
        :meth:`setup`.

        Args:
            noisy_images: Noisy images ``x_t`` of shape
                ``(n_images, *input_shape)``.
            **kwargs: Keyword arguments forwarded to :meth:`compute_error`
                (``measurements``, ``noise_rates``, ``signal_rates``,
                ``omega``, and any operator kwargs such as ``mask``).

        Returns:
            A ``(gradients, (measurement_error, (pred_noises, pred_images)))``
            tuple.  ``gradients`` is the gradient of the measurement error
            w.r.t. ``noisy_images`` and can be subtracted directly from the
            reverse-diffusion update.
        """
        return self.gradient_fn(noisy_images, **kwargs)


@diffusion_guidance_registry(name="dds")
class DDS(DiffusionGuidance):
    """
    Decomposed Diffusion Sampling guidance.

    Reference paper: https://arxiv.org/pdf/2303.05754
    """

    def setup(self):
        """Setup DDS guidance function."""
        if not self.disable_jit:
            self.call = jit(self.call)

    def Acg(self, x, **op_kwargs):
        # we transform the operator from A(x) to A.T(A(x)) to get the normal equations,
        # so that it is suitable for conjugate gradient. (symmetric, positive definite)
        # Normal equations: A^T y = A^T A x
        return self.operator.transpose(self.operator.forward(x, **op_kwargs), **op_kwargs)

    def conjugate_gradient_inner_loop(self, i, loop_state, eps=1e-5):
        """
        A single iteration of the conjugate gradient method.
        This involves minimizing the error of x along the current search
        vector p, and then choosing the next search vector.

        Reference code from: https://github.com/svi-diffusion/
        """
        p, rs_old, r, x, eps, op_kwargs = loop_state

        # compute alpha
        Ap = self.Acg(p, **op_kwargs)  # transform search vector p by A
        a = rs_old / ops.sum(p * Ap)  # minimize f along the line p

        x_new = x + a * p  # set new x at the minimum of f along line p
        r_new = r - a * Ap  # shortcut to compute next residual

        # compute Gram-Schmidt coefficient beta to choose next search vector
        # so that p_new is A-orthogonal to p_current.
        rs_new = ops.sum(r_new * r_new)
        p_new = r_new + (rs_new / rs_old) * p

        # this is like a jittable 'break' -- if the residual
        # is less than eps, then we just return the old
        # loop state rather than the updated one.
        next_loop_state = ops.cond(
            ops.abs(ops.sqrt(rs_old)) < eps,
            lambda: (p, rs_old, r, x, eps, op_kwargs),
            lambda: (p_new, rs_new, r_new, x_new, eps, op_kwargs),
        )

        return next_loop_state

    def call(
        self,
        noisy_images,
        measurements,
        noise_rates,
        signal_rates,
        n_inner: int,
        eps: float,
        verbose: bool,
        **op_kwargs,
    ):
        r"""Run one DDS guidance step via conjugate gradient.

        Denoises ``x_t`` to obtain an initial ``x_0`` estimate, then
        refines it by solving the normal equations
        :math:`A^\top A\, x = A^\top y` with ``n_inner`` conjugate
        gradient iterations.

        Args:
            noisy_images: Noisy images ``x_t`` of shape
                ``(n_images, *input_shape)``.
            measurements: Target observations ``y``.
            noise_rates: Noise rates at the current diffusion time.
            signal_rates: Signal rates at the current diffusion time.
            n_inner: Number of conjugate gradient iterations.
            eps: Convergence tolerance; CG stops early when the residual
                norm falls below this threshold.
            verbose: When ``True``, compute and return the measurement
                error ``‖y - A(x̂_0)‖``. When ``False``, the error is
                returned as ``0.0`` to avoid extra computation.
            **op_kwargs: Additional keyword arguments forwarded to the
                operator (e.g. ``mask``).

        Returns:
            A ``(gradients, (measurement_error, (pred_noises, pred_images)))``
            tuple.  ``gradients`` is a zero tensor because DDS performs
            guidance entirely inside the CG loop; the caller subtracts it
            as a no-op.
        """
        pred_noises, pred_images = self.diffusion_model.denoise(
            noisy_images,
            noise_rates,
            signal_rates,
            training=False,
        )
        measurements_cg = self.operator.transpose(measurements, **op_kwargs)
        r = measurements_cg - self.Acg(pred_images, **op_kwargs)  # residual
        p = ops.copy(r)  # initial search vector = residual
        rs_old = ops.sum(r * r)  # residual dot product
        _, _, _, pred_images_updated_cg, _, _ = fori_loop(
            0,
            n_inner,
            self.conjugate_gradient_inner_loop,
            (p, rs_old, r, pred_images, eps, op_kwargs),
        )

        # Not strictly necessary, just for debugging
        error = ops.cond(
            verbose,
            lambda: L2(measurements - self.operator.forward(pred_images_updated_cg, **op_kwargs)),
            lambda: 0.0,
        )

        pred_images = pred_images_updated_cg
        # we have already performed the guidance steps in self.conjugate_gradient_method, so
        # we can set these gradients to zero.
        gradients = ops.zeros_like(pred_images)
        return gradients, (error, (pred_noises, pred_images))

    def __call__(
        self,
        noisy_images,
        measurements,
        noise_rates,
        signal_rates,
        n_inner: int = 5,
        eps: float = 1e-5,
        verbose: bool = False,
        **op_kwargs,
    ):
        """Run one DDS guidance step (public entry point).

        Delegates to :meth:`call`, which may be JIT-compiled depending on
        :attr:`disable_jit`.

        Args:
            noisy_images: Noisy images ``x_t`` of shape
                ``(n_images, *input_shape)``.
            measurements: Target observations ``y``.
            noise_rates: Noise rates at the current diffusion time.
            signal_rates: Signal rates at the current diffusion time.
            n_inner: Number of conjugate gradient iterations. Default: ``5``.
            eps: Convergence tolerance for the CG solver. Default: ``1e-5``.
            verbose: When ``True``, compute and return the measurement
                error for logging. Default: ``False``.
            **op_kwargs: Additional keyword arguments forwarded to the
                operator (e.g. ``mask``).

        Returns:
            A ``(gradients, (measurement_error, (pred_noises, pred_images)))``
            tuple (see :meth:`call`).
        """
        return self.call(
            noisy_images,
            measurements,
            noise_rates,
            signal_rates,
            n_inner,
            eps,
            verbose,
            **op_kwargs,
        )


@diffusion_guidance_registry(name="nuclear-dps")
class NuclearDiffusion(DPS):
    r"""Nuclear Diffusion posterior sampling guidance.

    A hybrid framework that combines diffusion posterior sampling (DPS) with low-rank
    temporal modeling for video restoration. This method replaces the sparsity assumption
    in Robust Principal Component Analysis (RPCA) with a learned diffusion prior while
    maintaining a nuclear norm penalty on the background component to encourage low-rank
    temporal structure.


    .. seealso::

        - :func:`~zea.func.dehaze_nuclear_diffusion`: The dehazing application of this method
        - :doc:`../notebooks/models/nuclear_dehazing_example`: Example notebook demonstrating
          the method on cardiac ultrasound dehazing
        - :class:`DPS`: Base diffusion posterior sampling guidance

    **Mathematical Formulation:**

    Given observations :math:`\mathbf{Y} \in \mathbb{R}^{n \times p}` (video frames),
    Nuclear Diffusion jointly samples the signal :math:`\mathbf{X}` and low-rank background
    :math:`\mathbf{L}` from the posterior:

    .. math::

        \mathbf{X}, \mathbf{L} \sim p_\theta(\mathbf{X}, \mathbf{L} \mid \mathbf{Y})

    The posterior is factorized as:

    .. math::

        p(\mathbf{Y}, \mathbf{L}, \mathbf{X}) = p(\mathbf{Y} \mid \mathbf{L}, \mathbf{X}) \, p(\mathbf{L}) \, p_\theta(\mathbf{X})

    where:

    - :math:`p(\mathbf{Y} \mid \mathbf{L}, \mathbf{X}) = \mathcal{N}(\mathbf{Y}; \mathbf{L}+\mathbf{X}, \mu^{-1} \mathbf{I})`
      is the likelihood (measurement model)
    - :math:`p(\mathbf{L}) \propto \exp(-\gamma \|\mathbf{L}\|_*)` enforces low-rank structure
      via the nuclear norm :math:`\|\mathbf{L}\|_* = \sum_i \sigma_i(\mathbf{L})`
    - :math:`p_\theta(\mathbf{X})` is a learned diffusion prior capturing complex signal structure

    The diffusion prior operates on individual frames :math:`\mathbf{x}^t \in \mathbb{R}^n`,
    while temporal dependencies are enforced through the nuclear norm on :math:`\mathbf{L}`.

    This guidance method alternates between reverse diffusion and measurement-guided updates,
    computing gradients from both the measurement error and the nuclear norm penalty:

    Args:
        diffusion_model: The diffusion model for the signal component.
        operator: Forward operator defining the measurement model.
        disable_jit: Whether to disable JIT compilation.

    .. admonition:: Reference

        T. Stevens, O. Nolan, J. L. Robert, and R. J. G. van Sloun,
        "Nuclear Diffusion Models for Low-Rank Background Suppression in Videos,"
        *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2026.
        https://arxiv.org/abs/2509.20886

    """  # noqa: E501

    @staticmethod
    def nuclear_norm_penalty(background_images):
        r"""Compute nuclear norm penalty for low-rank enforcement.

        The nuclear norm (sum of singular values) encourages low-rank structure in
        the background component across time. For a matrix :math:`\mathbf{L}`, it is
        defined as:

        .. math::

            \|\mathbf{L}\|_* = \sum_{i=1}^{r} \sigma_i(\mathbf{L})

        where :math:`\sigma_i` are the singular values and :math:`r` is the rank.

        Args:
            background_images: Background images of shape
                ``(batch, frames, height, width, channels)``.
                Each sequence is reshaped to a matrix of shape ``(frames, height x width x channels)``
                before computing the nuclear norm.

        Returns:
            Nuclear norm penalty summed across the batch and normalized by number of frames.

        Note:
            The input is reshaped from ``(batch, frames, H, W, C)`` to ``(batch, frames, HxWxC)``
            before computing the singular values.
        """  # noqa: E501
        n_batch, n_frames, height, width, channels = ops.shape(background_images)
        background_images_flattened = ops.reshape(
            background_images, (n_batch, n_frames, height * width * channels)
        )
        background_nuclear_penalty = ops.norm(background_images_flattened, axis=(1, 2), ord="nuc")

        # normalize nuclear penalty
        background_nuclear_penalty /= n_frames

        # sum across batches
        return ops.sum(background_nuclear_penalty)

    @staticmethod
    def weighted_nuclear_norm_penalty(background_images, weight_factor: float = 2.0):
        r"""Compute weighted nuclear norm penalty with enhanced rank control.

        This implements a WNNM-style (Weighted Nuclear Norm Minimization) penalty that
        penalizes smaller singular values more heavily than larger ones, suppressing the
        spectrum tail to enforce low-rank structure. The weighted penalty is:

        .. math::

            \|\mathbf{L}\|_{w,*} = \sum_{i=1}^{r} w_i \cdot \sigma_i(\mathbf{L})

        where :math:`w_i = 1 + \alpha \cdot \frac{i}{r}` increases linearly with the
        index :math:`i`, and :math:`\alpha` is the ``weight_factor``. Since ``ops.svd``
        returns singular values in descending order (:math:`\sigma_1 \geq \sigma_2 \geq \cdots`),
        higher indices correspond to smaller singular values, which receive larger weights.

        Args:
            background_images: Background images of shape ``(batch, frames, height, width, channels)``.
            weight_factor: Scaling factor :math:`\alpha` controlling how much more to penalize
                smaller singular values (the spectrum tail). Default is 2.0.

        Returns:
            Weighted nuclear norm penalty summed across the batch and normalized by number of frames.

        Note:
            This is a drop-in replacement for :meth:`nuclear_norm_penalty` that provides
            better rank control by more aggressively penalizing the tail of the singular value
            spectrum (smaller singular values) rather than the leading ones.
        """  # noqa: E501
        n_batch, n_frames, height, width, channels = ops.shape(background_images)
        background_images_flattened = ops.reshape(
            background_images, (n_batch, n_frames, height * width * channels)
        )

        def weighted_svd_penalty(matrix):
            """Compute weighted SVD penalty for a matrix"""
            _, s_vals, _ = ops.svd(matrix, full_matrices=False)
            n_sv = ops.shape(s_vals)[0]
            weights = 1.0 + weight_factor * ops.arange(n_sv, dtype="float32") / ops.cast(
                n_sv, "float32"
            )
            return ops.sum(weights * s_vals)

        # Apply weighted penalty to each batch element
        weighted_penalties = ops.vectorized_map(weighted_svd_penalty, background_images_flattened)

        # normalize by number of frames
        weighted_penalties /= n_frames

        # sum across batches (same as original)
        return ops.sum(weighted_penalties)

    def compute_error(
        self,
        combined_images,
        measurements,
        noise_rates,
        signal_rates,
        omega: float = 1.0,
        gamma: float = 1.0,
        rank_weight_factor: float | None = None,
        step: int | None = None,
        total_steps: int | None = None,
        initial_step: int = 100,
        max_alpha: float = 0.5,
        **kwargs,
    ):
        r"""Compute measurement error for joint diffusion posterior sampling.

        Args:
            combined_images: Concatenated noisy images, containing both foreground and background
                components, shape ``(batch, frames, H, W, 2C)``. In the context of cardiac
                ultrasound dehazing, the first C channels correspond to the tissue signal
                (foreground), and the next C channels correspond to the haze (background) component.
            measurements: Target measurements :math:`\mathbf{Y}`, shape ``(batch, frames, H, W, C)``.
            noise_rates: Current noise rates from the diffusion schedule, shape ``(batch, frames, 1, 1, 1)``.
            signal_rates: Current signal rates from the diffusion schedule, shape ``(batch, frames, 1, 1, 1)``.
            omega: Weight :math:`\omega` for the measurement error term (L2 reconstruction loss).
            gamma: Weight :math:`\gamma` for the nuclear norm penalty term.
            rank_weight_factor: Optional weight factor for :meth:`weighted_nuclear_norm_penalty`.
                If ``None``, uses standard :meth:`nuclear_norm_penalty`.
            step: Current diffusion step for progressive blending. Used to compute :math:`\alpha(t)`.
            total_steps: Total number of diffusion steps.
            initial_step: Step at which to start progressive blending.
            max_alpha: Maximum value for :math:`\alpha` at the final step. The alpha parameter mixes
                foreground and background predictions, but only after the initial_step to allow the
                diffusion model to first focus on generating the foreground signal before blending
                in the background component.
            **kwargs: Additional arguments (unused).

        Returns:
            A tuple containing:

            - **measurement_error** (float): Combined loss :math:`\mathcal{L}`.
            - **aux** (tuple): Auxiliary outputs:
              ``(pred_noises_foreground, pred_images_foreground, noisy_background_images, l2_error, nuclear_penalty)``

        .. note::
            The progressive blending factor :math:`\alpha(t)` linearly increases from 0
            at ``initial_step`` and plateaus at ``max_alpha`` once normalized progress
            reaches ``max_alpha``, allowing the background component to gradually influence
            the reconstruction and then saturate for the remainder of sampling.

        """  # noqa: E501
        channels = ops.shape(combined_images)[-1] // 2
        noisy_foreground_images = combined_images[..., :channels]
        noisy_background_images = combined_images[..., channels:]

        # Transpose for ops.map
        noisy_tissue_seq = ops.swapaxes(noisy_foreground_images, 0, 1)  # [S, B, H, W, C]
        # Signal and noise rates are the same throughout the sequence, so can just
        # grab the first batch and reuse that
        noise_rates_s = noise_rates[:, 0, ...]
        signal_rates_s = signal_rates[:, 0, ...]

        def denoise_step(x_s):
            pred_noises, pred_images = self.diffusion_model.denoise(
                x_s, noise_rates_s, signal_rates_s, training=False
            )
            return {"pred_noises": pred_noises, "pred_images": pred_images}

        denoised = ops.map(denoise_step, noisy_tissue_seq)
        pred_noises_foreground = ops.swapaxes(denoised["pred_noises"], 0, 1)  # [B, S, H, W, C]
        pred_images_foreground = ops.swapaxes(denoised["pred_images"], 0, 1)  # [B, S, H, W, C]

        alpha = ops.clip(
            (step - initial_step) / (total_steps - initial_step), 0.0, max_alpha
        )  # linear after initial_step
        pred_measurements = (1 - alpha) * pred_images_foreground + (alpha) * noisy_background_images

        l2_error = L2(measurements - pred_measurements)

        # Choose penalty function for nuclear norm
        if rank_weight_factor is not None:
            background_nuclear_penalty = self.weighted_nuclear_norm_penalty(
                noisy_background_images, rank_weight_factor
            )
        else:
            background_nuclear_penalty = self.nuclear_norm_penalty(noisy_background_images)

        # NOTE: we sum across batches for the nuclear norm here.
        # the gradient of sums = sum of gradients
        nuclear_penalty = ops.sum(background_nuclear_penalty)

        # Combine all penalty terms
        measurement_error = omega * l2_error + gamma * nuclear_penalty

        return measurement_error, (
            pred_noises_foreground,
            pred_images_foreground,
            noisy_background_images,
            l2_error,
            nuclear_penalty,
        )

    def __call__(
        self,
        noisy_images1,
        noisy_images2,
        measurements,
        noise_rates,
        signal_rates,
        omega: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ):
        r"""Compute guidance gradients for posterior sampling.

        This method concatenates the noisy foreground and background images, computes the
        combined loss via :meth:`compute_error`, and returns separate gradients
        for each component.

        Args:
            noisy_images1: Noisy foreground images :math:`\mathbf{x}_t` from the diffusion model,
                shape ``(batch, frames, H, W, C)``.
            noisy_images2: Noisy background images :math:`\mathbf{L}_t`,
                shape ``(batch, frames, H, W, C)``.
            measurements: Target measurements :math:`\mathbf{Y}`, shape ``(batch, frames, H, W, C)``.
            noise_rates: Current noise rates from diffusion schedule.
            signal_rates: Current signal rates from diffusion schedule.
            omega: Weight for the measurement error term. Default is 1.0.
            gamma: Weight for the nuclear norm penalty term. Default is 1.0.
            **kwargs: Additional arguments passed to :meth:`compute_error` (e.g., ``gamma``,
                ``rank_weight_factor``, ``step``, ``total_steps``).

        Returns:
            A tuple containing:

            - **gradients** (tuple): ``(grad_foreground, grad_background)`` - gradients for foreground and background.
            - **loss_info** (tuple): ``(loss, aux)`` where:

              - **loss** (float): Combined loss value.
              - **aux** (tuple): Auxiliary outputs from :meth:`compute_error`.
        """  # noqa: E501

        combined_input = ops.concatenate([noisy_images1, noisy_images2], axis=-1)
        gradients, (loss, aux) = self.gradient_fn(
            combined_input,
            measurements=measurements,
            noise_rates=noise_rates,
            signal_rates=signal_rates,
            omega=omega,
            gamma=gamma,
            **kwargs,
        )
        channels = ops.shape(gradients)[-1] // 2
        grad1 = gradients[..., :channels]
        grad2 = gradients[..., channels:]
        return (grad1, grad2), (loss, aux)
