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
                "unet_time_conditional" or "dense_time_conditional".
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

    def call(self, inputs, training=False, network=None, **kwargs):
        """Calls the score network.

        If network is not provided, will use the exponential moving
        average network if training is False, otherwise the regular network.
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
        n_samples=1,
        n_steps=20,
        initial_step=0,
        initial_samples=None,
        seed=None,
        **kwargs,
    ):
        """Sample from the posterior distribution given measurements.

        Args:
            measurements: Input measurements. Typically of shape
                `(batch_size, *input_shape)`.
            n_samples: Number of posterior samples to generate.
                Will generate `n_samples` samples for each measurement
                in the `measurements` batch.
            n_steps: Number of diffusion steps.
            initial_step: Initial step to start from. Can warm start the
                diffusion process with a partially noised image, thereby
                skipping part of the diffusion process. Initial step
                closer to n_steps, will result in a shorter diffusion process
                (i.e. less noise added to the initial image). A value of 0
                means that the diffusion process starts from pure noise.
            initial_samples: Optional initial samples to start from.
                If provided, these samples will be used as the starting point
                for the diffusion process. Only used if `initial_step` is
                greater than 0. Must be of shape `(batch_size, n_samples, *input_shape)`.
            seed: Random seed generator.
            **kwargs: Additional arguments.

        Returns:
            Posterior samples p(x|y), of shape:
                `(batch_size, n_samples, *input_shape)`.

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
            shape=[batch_size, *[1] * n_dims],
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
            shape=[batch_size, *[1] * n_dims],
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
        """Cosine diffusion schedule https://arxiv.org/abs/2102.09672

        Args:
            diffusion_times: tensor with diffusion times in [0, 1]

        Returns:
            noise_rates: tensor with noise rates
            signal_rates: tensor with signal rates

            according to:
            - x_t = signal_rate * x_0 + noise_rate * noise
            - x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise

            or with stochastic sampling:
            - x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t - sigma_t^2) * noise + sigma_t * epsilon

            where:
            - sigma_t = sqrt((1 - alpha_t) / (1 - alpha_{t+1})) * sqrt(1 - alpha_{t+1} / alpha_t)

        Note:
            t+1 = previous time step
            t = current time step

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
        training,
        network=None,
    ):
        """Predict noise component and calculate the image component using it."""

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
            initial_noise: Initial noise tensor.
            diffusion_steps: Number of diffusion steps.
            initial_samples: Optional initial samples to start from.
            initial_step: Initial step to start from.
            stochastic_sampling: Whether to use stochastic sampling (DDPM).
            seed: Random seed generator.
            verbose: Whether to show a progress bar.
            track_progress_type: Type of progress tracking ("x_0" or "x_t").
            disable_jit: Whether to disable JIT compilation.
            training: Whether to use the training mode of the network.
            network_type: Which network to use ("main" or "ema"). If None, uses the
                network based on the `training` argument.

        Returns:
            Generated images.
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

            # denoise
            if network_type == "ema":
                network = self.ema_network
            elif network_type == "main":
                network = self.network
            else:
                network = None

            pred_noises, pred_images = self.denoise(
                noisy_images,
                noise_rates,
                signal_rates,
                training=training,
                network=network,
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

        Effectively performs diffusion posterior sampling p(x_0 | y).

        Args:
            measurements: Conditioning data.
            initial_noise: Initial noise tensor.
            diffusion_steps: Number of diffusion steps.
            initial_samples: Optional initial samples to start from.
            initial_step: Initial step to start from.
            stochastic_sampling: Whether to use stochastic sampling (DDPM).
            seed: Random seed generator.
            verbose: Whether to show a progress bar.
            track_progress_type: Type of progress tracking ("x_0" or "x_t").
            **kwargs: Additional arguments. These are passed to the guidance
                function and the operator. Examples are omega, mask, etc.

        Returns:
            Generated images.

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

    def prepare_diffusion(self, diffusion_steps, initial_step, verbose, disable_jit=False):
        """Prepare the diffusion process.

        This method sets up the parameters for the diffusion process, including
        validation of the initial step and calculation of the step size.
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
        initial_step,
        step_size,
    ):
        """Prepare the diffusion schedule.

        This method sets up the initial noisy images based on the provided
        initial noise and samples. It handles the case where the initial step
        is greater than 0, allowing for the use of partially noised images for
        initialization of the diffusion process.

        Args:
            base_diffusion_times: Base diffusion times.
            initial_noise: Initial noise tensor.
            initial_samples: Optional initial samples to start from.
            initial_step: Initial step to start from.
            step_size: Step size for the diffusion process.

        Returns:
            next_noisy_images: Noisy images after the initial step.
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

    def start_track_progress(self, diffusion_steps, initial_step=0):
        """Initialize the progress tracking for the diffusion process.
        For diffusion animation we keep track of the diffusion progress.
        For large number of steps, we do not store all the images due to memory constraints.
        """
        self.track_progress = []
        remaining = max(1, diffusion_steps - int(initial_step))
        if remaining > 50:
            self.track_progress_interval = remaining // 50
        else:
            self.track_progress_interval = 1

    def store_progress(
        self,
        step,
        track_progress_type,
        next_noisy_images,
        pred_images,
    ):
        """Store the progress of the diffusion process.

        Args:
            step: Current diffusion step.
            track_progress_type: Type of progress tracking ("x_0" or "x_t").
            next_noisy_images: Noisy images after the current step.
            pred_images: Predicted images.

        Notes:
            - x_0 is considered the predicted image (aka Tweedie estimate)
            - x_t is the noisy intermediate image
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
            disable_jit: Whether to disable JIT compilation.
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
        omega,
        **kwargs,
    ):
        """
        Compute measurement error for diffusion posterior sampling.

        Args:
            noisy_images: Noisy images.
            measurements: Target measurement.
            noise_rates: Current noise rates.
            signal_rates: Current signal rates.
            omega: Weight for the measurement error.
            **kwargs: Additional arguments for the operator.

        Returns:
            Tuple of (measurement_error, (pred_noises, pred_images))
        """
        pred_noises, pred_images = self.diffusion_model.denoise(
            noisy_images,
            noise_rates,
            signal_rates,
            training=False,
        )

        measurement_error = omega * L2(measurements - self.operator.forward(pred_images, **kwargs))

        return measurement_error, (pred_noises, pred_images)

    def __call__(self, noisy_images, **kwargs):
        """
        Call the gradient function.

        Args:
            noisy_images: Noisy images.
            measurement: Target measurement.
            operator: Forward operator.
            noise_rates: Current noise rates.
            signal_rates: Current signal rates.
            omega: Weight for the measurement error.
            **kwargs: Additional arguments for the operator.

        Returns:
            Tuple of (gradients, (measurement_error, (pred_noises, pred_images)))
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
        n_inner,
        eps,
        verbose,
        **op_kwargs,
    ):
        """
        Call the DDS guidance function

        Args:
            noisy_images: Noisy images.
            measurement: Target measurement.
            noise_rates: Current noise rates.
            signal_rates: Current signal rates.
            n_inner: Number of conjugate gradient steps.
            verbose: Whether to calculate error.

        Returns:
            Tuple of (gradients, (measurement_error, (pred_noises, pred_images)))
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
        n_inner=5,
        eps=1e-5,
        verbose=False,
        **op_kwargs,
    ):
        """
        Call the DDS guidance function

        Args:
            noisy_images: Noisy images.
            measurement: Target measurement.
            noise_rates: Current noise rates.
            signal_rates: Current signal rates.
            n_inner: Number of conjugate gradient steps.
            verbose: Whether to calculate error.
            **kwargs: Additional arguments for the operator.

        Returns:
            Tuple of (gradients, (measurement_error, (pred_noises, pred_images)))
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
