"""
Hierarchical Variational Auto-Encoder for image generation, posterior sampling and inference tasks.
To try this model, simply load one of the available presets:

.. doctest::

    >>> from zea.models.hvae import HierarchicalVAE

    >>> model = HierarchicalVAE.from_preset("hvae")  # doctest: +SKIP

.. important::
    This is a ``zea`` implementation of the model.
    For the original code, see `here <https://github.com/swpenninga/hvae>`_.

.. seealso::
    A tutorial notebook where this model is used:
    :doc:`../notebooks/models/hvae_model_example`.

"""

import pickle

from keras import ops

from zea.internal.registry import model_registry
from zea.models.generative import DeepGenerativeModel
from zea.models.hvae.model import VAE
from zea.models.hvae.utils import Parameters
from zea.models.preset_utils import get_preset_loader, register_presets
from zea.models.presets import hvae_presets

SUPPORTED_VERSIONS = [
    "lvh",
    "lvh_ur24",
    "lvh_ur16",
    "lvh_ur8",
    "lvh_ur4",
    "lvh_ge24",
    "lvh_ge16",
    "lvh_ge8",
    "lvh_ge4",
]


@model_registry(name="hvae")
class HierarchicalVAE(DeepGenerativeModel):
    """
    Hierarchical Variational Autoencoder (HVAE) model.
    The network as defined here is a snippet of the complete model at:
    https://github.com/swpenninga/hvae

    The lvh versions are trained on EchoNetLVH at 256x256 resolution with 3 channels.
    (video-frames as channel dimension)
    The ur(.) versions denote retraining with a UniformRandom agent with (.)/256 lines.

    Unlike the other models, this network is built when the weights are loaded.
    """

    def __init__(self, name="hvae", version="lvh", **kwargs):
        """
        Args:
            name (str): Name of the model.
            version (str): Version of the HVAE model to use.
                Supported versions are: "lvh", "lvh_ur24", "lvh_ur16", "lvh_ur8", "lvh_ur4", "lvh_ge24", "lvh_ge16", "lvh_ge8", "lvh_ge4".
        """

        super().__init__(name, **kwargs)
        assert version in SUPPORTED_VERSIONS, (
            f"Unsupported version '{version}' for HVAE model."
            f"Current supported versions are: {', '.join(SUPPORTED_VERSIONS)}."
        )
        self.version = version
        self.network = None

    def custom_load_weights(self, preset):
        """
        Load the pretrained weights of the HVAE model from a preset.
        First builds the model architecture from args.pkl,
        then loads the weights into the model.
        """
        loader = get_preset_loader(preset)
        args_file = loader.get_file("args.pkl")
        weights_file = loader.get_file(f"hvae_{self.version}.weights.h5")

        # Build the model architecture from args.pkl
        with open(args_file, "rb") as f:
            args = pickle.load(f)
        params = Parameters(args)

        vae = VAE(params)
        vae.build()

        # Load and copy the weights
        vae.load_weights(weights_file)
        vae.trainable = False
        self.network = vae

        # Set model parameters that are used in partial_inference
        self.depth = params.model_depth
        self.stage_depth = params.dec_num_blocks
        self.z_out = params.z_out

    def sample(self, n_samples=1, **kwargs):
        """
        Samples from the prior distribution.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            samples (tensor): Generated samples of shape (n_samples, 256, 256, 3) in [-1, 1].
        """
        logits = self.network.decoder.call_uncond(n_samples, **kwargs)
        # Returns a 100 channel mixture of logistic functions (logits).
        samples = self.network.sample_from_mol(logits)
        return samples

    def posterior_sample(self, measurements, n_samples=1, **kwargs):
        """
        Performs posterior sampling on a batch of measurements.
        Only does a single encoder pass since it is deterministic,
        but does n_samples decoder passes to create posterior samples.

        Args:
            measurements (tensor): Input measurements of shape [B, 256, 256, 3].
            n_samples (int, optional): Number of posterior samples to generate. Defaults to 1.

        Returns:
            output (tensor): Posterior samples of shape [B, n_samples, 256, 256, 3].
        """

        # Measurements is [B, 256, 256, 3] in [-1, 1]
        b = ops.shape(measurements)[0]
        # Only need a single deterministic encoder pass
        activations = self.network.encoder(measurements)
        # Repeat the tensors in the list of activations n_samples amount of times
        # This repeats elementwise, so: [1, 2, 3] -> [1, 1, 2, 2, 3, 3]
        activations = [ops.repeat(a, repeats=n_samples, axis=0) for a in activations]

        # Logits are of shape [B * n_samples, 256, 256, 100]
        logits, _, _ = self.network.decoder.call(activations)
        # Samples are of shape [B * n_samples, 256, 256, 3] in [-1, 1]
        samples = self.network.sample_from_mol(logits)

        # Split the samples into [B, n_samples, 256, 256, 3]
        output = ops.stack(ops.split(samples, b, axis=0), axis=0)
        return output

    def call(self, measurements):
        """
        Returns a reconstruction of the input, together with the latent samples and KL divergences.

        Args:
            measurements (tensor): Input measurements of shape [B, 256, 256, 3].

        Returns:
            recon (tensor): Reconstructed output of shape [B, 256, 256, 3],
            List of latent samples from the decoder, and list of KL divergences
            at each latent layer.

        """
        # Returns reconstruction, latent samples, kl divergences
        recon, z_samples, kl = self.network.call(measurements)
        recon = self.network.sample_from_mol(recon)
        return recon, z_samples, kl

    def partial_inference(self, measurements, num_layers=0.5, n_samples=1, **kwargs):
        """
        Performs TopDown inference with the HVAE up until a certain layer,
        after which it continues in the decoder with multiple prior streams.

        Args:
            measurements (tensor): Input measurements of shape [B, 256, 256, 3].
            num_layers (float or int): If float, fraction of total layers to use from the top.
                If int, number of layers to use from the top.
            n_samples (int): Number of posterior samples to generate.

        Returns:
            output (tensor): Posterior samples of shape [B, n_samples, 256, 256, 3].
        """
        # Make sure num_layers is a float between 0 and 1 or an integer between 1 and depth
        if isinstance(num_layers, float):
            assert 0.0 < num_layers <= 1.0, "num_layers as float must be in (0.0, 1.0]"
            num_layers = int(num_layers * self.depth)
        elif isinstance(num_layers, int):
            assert 1 <= num_layers <= self.depth, f"num_layers as int must be in [1, {self.depth}]"
        else:
            raise ValueError("num_layers must be either a float or an int.")

        b = ops.shape(measurements)[0]
        # Only need a single deterministic encoder pass
        activations = self.network.encoder(measurements)

        # Single pass through the top num_layers of the decoder
        # Adding the same latent to z_stage n_samples times
        x = ops.zeros_like(activations[-1])
        z = ops.tile(ops.zeros([1, *self.z_out]), (b * n_samples, 1, 1, 1))
        current_layer = 0
        for dec_stage, act in zip(self.network.decoder.stages.layers, reversed(activations)):
            for dec_block in dec_stage.blocks.layers:
                if current_layer < num_layers:
                    # Use posterior sampling for the first num_layers
                    x, z_block, _ = dec_block.call(x, act)
                    z += ops.repeat(z_block, repeats=n_samples, axis=0)
                else:
                    # Use prior sampling for the remaining layers
                    if current_layer == num_layers:
                        # At the threshold, we duplicate the rest of the chain
                        x = ops.repeat(x, repeats=n_samples, axis=0)
                    x, z_block = dec_block.call_uncond(x)
                    z += z_block
                current_layer += 1
            x = dec_stage.pool(x)

        z /= ops.sqrt(self.depth)
        px_z = self.network.decoder.activation(self.network.decoder.z_to_features(z))
        for out_block in self.network.decoder.output_blocks.layers:
            px_z = out_block(px_z)
        px_z = self.network.decoder.last_conv(px_z)
        px_z = self.network.sample_from_mol(px_z)

        return ops.stack(ops.split(px_z, b, axis=0), axis=0)

    def log_density(self, measurements, **kwargs):
        """
        Calculates the log density (ELBO) of the data under the model.

        Args:
            measurements (tensor): Input measurements of shape [B, 256, 256, 3].

        Returns:
            -elbo (tensor): negative ELBO of the input measurements, averaged over the batch.

        """
        recon, _, kl = self.network.call(measurements)
        # elbo is averaged over batch dimension
        elbo, _, _ = self.network.get_elbo(measurements, recon, kl, **kwargs)
        return -elbo


register_presets(hvae_presets, HierarchicalVAE)

__all__ = ["HierarchicalVAE"]
