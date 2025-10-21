"""Adaptive Beamforming by Deep LEarning (ABLE).

Original implementation of paper:
    - "Adaptive Ultrasound Beamforming Using Deep Learning"
    - https://doi.org/10.1109/TMI.2020.3008537
    - Author: Ben Luijten
"""

import keras
from keras import ops
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dropout,
    Input,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
)

from zea.internal.registry import model_registry
from zea.models.base import BaseModel


@model_registry(name="able")
class ABLE(BaseModel):
    """Adaptive Beamforming by Deep LEarning (ABLE) model.

    This class implements a small configurable convolutional encoder/decoder
    architecture used for adaptive ultrasound beamforming. The constructor
    accepts either an explicit list of layer channel sizes via ``layers`` or
    constructs a symmetric list using ``num_layers`` and ``latent_dim``. The
    ``kernel_size`` argument supports multiple forms (int, tuple, or list) and
    is parsed into a list of 2-tuples (height, width) matching the number
    of layers. The default configuration matches the original paper.

    Parameters
    ----------
    elements : int
        Number of fast-time channels / transducer elements (used as first and
        last layer channel size when ``layers`` is not provided).
    latent_dim : int
        Channel size for middle (latent) layers when ``layers`` is not provided.
    kernel_size : int, tuple, or list
        Kernel size specification. Accepted forms:
          - int -> every conv uses (k, k)
          - tuple (h, w) -> every conv uses (h, w)
          - list of ints/tuples -> layer-wise kernels (length must match layers)
    num_layers : int
        Number of layers to construct when ``layers`` is not provided. Must be
        >= 2. The constructed list will be: [elements, latent_dim, ..., elements].
    layers : list or None
        Optional explicit list of channel sizes per layer. When provided this
        overrides ``num_layers`` and ``latent_dim``.
    name : str
        Model name forwarded to :class:`BaseModel`.


    Example
    -------
    .. code-block:: python

        # create a 4-layer ABLE model with default (1,1) kernels.
        model = ABLE(elements=128, latent_dim=32, num_layers=4, kernel_size=(1,1))
    """

    def __init__(
        self,
        elements=128,  # Number of fast-time channels/transducer elements
        latent_dim=32,  # Latent dimension size (used for middle layers)
        kernel_size=1,  # See normalization rules below
        num_layers=4,  # number of layers (default 4 -> [elements, latent_dim..., elements])
        layers=None,  # explicit list of layer channel sizes, overrides num_layers
        name="able",
        **kwargs,
    ):
        """Initializes the ABLE model.

        Kernel size handling (accepted forms):
        - single int (e.g. 1) -> every conv uses (1,1)
        - single tuple (h, w) (e.g. (1,3)) -> every conv uses (h,w)
        - list of ints (e.g. [1,3,3,1]) -> converted to [(1,1),(3,3),...]
        - list of tuples (e.g. [(1,1),(1,3),(1,3),(1,1)]) -> used as-is

        Layer / dimensionality handling:
        - If `layers` is provided and is a list, that list is used as the channel
          sizes for each layer.
        - Else, `num_layers` is used to create a list of length `num_layers`:
            [elements, latent_dim, latent_dim, ..., elements]
          (first and last are `elements`, middle layers are `latent_dim`).
        """
        super().__init__(name=name, **kwargs)

        # network will be constructed by an external builder; leave as None
        # or construct with a helper if/when available
        self.network = None

        # initialize and validate layer dims and kernel sizes using helpers
        self.layer_dims = self._init_layers(layers, num_layers, elements, latent_dim)
        self.kernel_sizes = self._init_kernels(kernel_size, len(self.layer_dims))

        # store for potential external inspection
        self.elements = elements
        self.latent_dim = latent_dim
        self.num_layers = len(self.layer_dims)

    def _init_kernels(self, kernel_size, n_layers):
        """Normalize kernel_size into a list of (h, w) tuples of length n_layers.

        Accepts:
          - int -> repeated (k,k)
          - tuple -> repeated (h,w)
          - list of int/tuple -> converted per-entry

        Raises ValueError on invalid input or length mismatch.
        """

        def _normalize_entry(k):
            if isinstance(k, int):
                return (k, k)
            if isinstance(k, tuple) and len(k) == 2 and all(isinstance(x, int) for x in k):
                return k
            raise ValueError("kernel entries must be int or tuple(int,int)")

        if isinstance(kernel_size, (int, tuple)):
            return [_normalize_entry(kernel_size)] * n_layers
        if isinstance(kernel_size, (list, tuple)):
            kernels = [_normalize_entry(k) for k in kernel_size]
            if len(kernels) != n_layers:
                raise ValueError(
                    "When kernel_size is a list, its length must match "
                    f"number of layers ({n_layers})."
                )
            return kernels
        raise ValueError("kernel_size must be int, tuple, or list/tuple of those.")

    def _init_layers(self, layers, num_layers, elements, latent_dim):
        """Normalize layer specification into a list of positive ints.

        If ``layers`` is provided it is validated and returned as a list. Otherwise
        ``num_layers``, ``elements`` and ``latent_dim`` are used to build a
        symmetric list where first and last are ``elements`` and middle entries
        are ``latent_dim``.
        """
        if layers is not None:
            if not isinstance(layers, (list, tuple)):
                raise ValueError("`layers` must be a list/tuple of integers when provided.")
            layer_dims = list(layers)
            if len(layer_dims) < 2:
                raise ValueError(
                    "`layers` must contain at least two elements (input and output dims)."
                )
            if not all(isinstance(d, int) and d > 0 for d in layer_dims):
                raise ValueError("all `layers` entries must be positive integers.")
            return layer_dims

        if not isinstance(num_layers, int) or num_layers < 2:
            raise ValueError("`num_layers` must be an int >= 2 when `layers` is not provided.")
        # first and last are `elements`, middle ones are `latent_dim`
        if num_layers == 2:
            return [elements, elements]
        middle = [latent_dim] * (num_layers - 2)
        return [elements] + middle + [elements]

    def _get_network(self, layer_dims, kernel_sizes):
        """Builds the internal network based on layer dimensions and kernel sizes."""

        # build a simple sequential stack of Conv2D layers.
        # input channels are layer_dims[0]; create an Input layer accordingly.
        inp = Input(shape=(None, None, None, layer_dims[0]))
        x = inp

        for idx, (layer_dim, kernel_size) in enumerate(zip(layer_dims, kernel_sizes)):
            # last layer: no activation; intermediate layers: use antirectifier
            activation = None if idx == (len(layer_dims) - 1) else self.antirectifier
            x = Conv2D(layer_dim, kernel_size, activation=activation, padding="same")(x)

        return keras.Model(inputs=inp, outputs=x)

    def antirectifier(self, x):
        """Applies the anti-rectifier activation function."""
        mean = ops.reduce_mean(x, axis=1, keepdims=True)
        x_centered = x - mean
        pos = ops.nn.relu(x_centered)
        neg = ops.nn.relu(-x_centered)
        output = ops.concatenate([pos, neg], axis=1)
        norm = ops.sqrt(ops.reduce_mean(ops.square(output), axis=1, keepdims=True) + 1e-6)
        return output / norm


if __name__ == "__main__":
    # simple test
    model = ABLE(elements=128, latent_dim=32, num_layers=4, kernel_size=(1, 3))
    print("Layer dims:", model.layer_dims)
    print("Kernel sizes:", model.kernel_sizes)
