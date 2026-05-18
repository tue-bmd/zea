"""Adaptive Beamforming by Deep LEarning (ABLE).

Original implementation of paper:
    - "Adaptive Ultrasound Beamforming Using Deep Learning"
    - https://doi.org/10.1109/TMI.2020.3008537
    - Author: Ben Luijten
"""

import keras
from keras import ops
from keras.layers import Conv2D

from zea.internal.registry import model_registry
from zea.models.base import BaseModel


@model_registry(name="able")
class ABLE(BaseModel):
    """Adaptive Beamforming by Deep LEarning (ABLE) model.

    This class implements a configurable convolutional encoder/decoder
    architecture for adaptive ultrasound beamforming. The constructor
    allows for flexible configuration of layer dimensions and kernel sizes.
    The default configuration corresponds to the setup used in the original paper
    "Adaptive Ultrasound Beamforming Using Deep Learning"
    (DOI: https://doi.org/10.1109/TMI.2020.3008537).

    Args:
    latent_dim : int, optional
        Channel size for the middle (latent) layers when `latent_layers` is not provided.
        Default is 32.
    kernel_size : int, tuple, or list, optional
        Kernel size specification. Accepted forms:
          - int -> every convolution uses (k, k)
          - tuple (h, w) -> every convolution uses (h, w)
          - list of ints/tuples -> layer-wise kernels (length must match layers)

        !! WARNING: Currently only 1x1 kernels are supported in order to be compatible with
        the PatchedGrid implementation that assumes independent processing of each pixel.
        #TODO: Remove this restriction

        Default is 1.
    n_latent_layers : int, optional
        Number of inner layers to construct when `latent_layers` is not provided.
        Must be >= 2. Default is 2.
    latent_layers : list or None, optional
        Explicit list of channel sizes for inner layers. Overrides `n_latent_layers`
        and `latent_dim` when provided. Default is None.
    axis : int, optional
        Axis that contains the transducer elements. Default is 3.
    name : str, optional
        Model name forwarded to `BaseModel`. Default is "able".
    **kwargs : dict
        Additional keyword arguments forwarded to `BaseModel`.
    """

    def __init__(
        self,
        latent_dim=32,
        kernel_size=1,
        n_latent_layers=2,
        latent_layers=None,
        axis=None,
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
        - If `latent_layers` is provided and is a list, that list is used as the
          channel sizes for the inner layers (excluding the first and last layers,
          which are dynamically set based on the input data).
        - Else, `n_latent_layers` is used to create a list of length
          `n_latent_layers + 2`:
            [input_dim, latent_dim, ..., latent_dim, input_dim]
          (first and last are dynamically set, middle layers are `latent_dim`).
        """
        super().__init__(name=name, **kwargs)

        # Initialize parameters
        self.kernel_size = kernel_size
        self.n_latent_layers = n_latent_layers
        self.latent_layers = latent_layers
        self.axis = axis
        self.latent_dim = latent_dim

        # Initialized in build()
        self.layer_dims = None
        self.kernel_sizes = None
        self._able_layers = []

    def _init_kernels(self, kernel_size, n_layers):
        """Normalize kernel_size into a list of (h, w) tuples.

        Args:
            kernel_size (int, tuple, or list): Kernel size specification. Accepted forms:
                - int -> repeated (k, k)
                - tuple (h, w) -> repeated (h, w)
                - list of int/tuple -> converted per entry.
            n_layers (int): Number of layers for which kernel sizes are required.

        Returns:
            list of tuple: List of (h, w) kernel tuples of length `n_layers`.

        Raises:
            ValueError: If `kernel_size` is invalid or its length does not match `n_layers`.
        """

        def _allowed_kernel_size(k):
            # TODO: Remove this restriction once PatchedGrid supports larger kernels
            # only allow 1x1 kernels for now
            if isinstance(k, int):
                if k != 1:
                    raise ValueError(
                        "Only kernel_size=1 is currently supported due to "
                        "PatchedGrid limitations."
                    )
                return True
            if (
                isinstance(k, tuple)
                and len(k) == 2
                and all(isinstance(x, int) for x in k)
            ):
                if k != (1, 1):
                    raise ValueError(
                        "Only kernel_size=(1,1) is currently supported due to "
                        "PatchedGrid limitations."
                    )
                return True
            raise ValueError("kernel_size entries must be int or tuple(int,int)")

        def _normalize_entry(k):
            _allowed_kernel_size(k) # TODO: Remove this check once larger kernels are supported
            if isinstance(k, int):
                return (k, k)
            if (
                isinstance(k, tuple)
                and len(k) == 2
                and all(isinstance(x, int) for x in k)
            ):
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

    def _init_layers(self, latent_layers, n_latent_layers, input_dim, latent_dim):
        """Normalize layer dimensions into a list of positive integers.

        Args:
            latent_layers (list or None): Explicit list of channel sizes for inner layers.
                If provided, it overrides `n_latent_layers` and `latent_dim`.
            n_latent_layers (int): Number of inner layers to construct when `latent_layers`
                is not provided.
            input_dim (int): Channel size for the input layer.
            latent_dim (int): Channel size for the middle (latent) layers.

        Returns:
            list of int: List of channel sizes for all layers, including input and output layers.

        Raises:
            ValueError: If `latent_layers` is invalid or its length does not match `n_latent_layers`.
        """

        if latent_layers is not None:
            if not isinstance(latent_layers, (list, tuple)):
                raise ValueError(
                    "`latent_layers` must be a list/tuple of integers when provided."
                )
            layer_dims = list(latent_layers)
            if len(layer_dims) != n_latent_layers:
                raise ValueError(
                    "`latent_layers` must contain exactly `n_latent_layers` entries for the inner layers."
                )
            if not all(isinstance(d, int) and d > 0 for d in layer_dims):
                raise ValueError(
                    "All `latent_layers` entries must be positive integers."
                )
            return [input_dim] + layer_dims + [input_dim]

        # Default behavior: create symmetric layers
        middle = [latent_dim] * n_latent_layers
        return [input_dim] + middle + [input_dim]

    def stack_channels(self, x, axis):
        """Reshape input into 4D for use with Conv2D.

        PatchedGrid passes per-pixel data with the spatial dims collapsed:
        - Rank 2: (pixels, elements)        -> (pixels, 1, 1, elements)
        - Rank 3: (pixels, elements, n_ch)  -> (pixels, 1, 1, elements*n_ch)
          (elements and n_ch are merged into a single channel axis, with n_ch
          varying fastest so that unstack_channels can restore the original layout).
        """
        rank = len(x.shape)
        shape = ops.shape(x)
        meta = {"rank": rank, "stacked": False}

        if rank == 2:
            # (pixels, elements) -> (pixels, 1, 1, elements)
            return ops.reshape(x, (shape[0], 1, 1, shape[1])), meta

        if rank == 3:
            # (pixels, elements, n_ch)
            elem = shape[1]
            ch = shape[2]
            meta.update({"elem": elem, "ch": ch})

            if ch == 1:
                # (pixels, elements, 1) -> (pixels, 1, 1, elements)
                return ops.reshape(x, (shape[0], 1, 1, elem)), meta

            # (pixels, elements, n_ch) -> transpose to (pixels, n_ch, elements)
            # -> reshape to (pixels, 1, 1, n_ch*elements)
            x_swapped = ops.transpose(x, axes=[0, 2, 1])
            meta["stacked"] = True
            return ops.reshape(x_swapped, (shape[0], 1, 1, elem * ch)), meta

        # unsupported rank: leave unchanged
        return x, meta

    def unstack_channels(self, x, meta):
        """Inverse of stack_channels. Restores the original shape from Conv2D output.

        Expects x with shape (pixels, 1, 1, C) and restores:
        - rank 2 input: (pixels, 1, 1, elements)       -> (pixels, elements)
        - rank 3, ch==1: (pixels, 1, 1, elements)      -> (pixels, elements, 1)
        - rank 3, ch>1:  (pixels, 1, 1, n_ch*elements) -> (pixels, elements, n_ch)
        """
        rank = meta.get("rank", None)
        shape = ops.shape(x)  # (pixels, 1, 1, C)
        pixels = shape[0]
        C = shape[-1]

        if rank == 2:
            # (pixels, 1, 1, elements) -> (pixels, elements)
            return ops.reshape(x, (pixels, C))

        if rank == 3:
            elem = meta["elem"]
            ch = meta["ch"]
            if meta.get("stacked", False):
                # (pixels, 1, 1, n_ch*elements) -> (pixels, n_ch, elements) -> (pixels, elements, n_ch)
                x_r = ops.reshape(x, (pixels, ch, elem))
                return ops.transpose(x_r, axes=[0, 2, 1])
            else:
                # ch == 1: (pixels, 1, 1, elements) -> (pixels, elements, 1)
                return ops.reshape(x, (pixels, elem, 1))

        # unsupported rank: return as-is
        return x

    def antirectifier(self, x):
        """Apply the anti-rectifier activation function.

        This function centers the input, splits it into positive and negative
        components, and normalizes the result.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Transformed tensor with anti-rectifier activation applied.
        """
        mean = ops.mean(x, axis=-1, keepdims=True)
        x_centered = x - mean
        pos = ops.nn.relu(x_centered)
        neg = ops.nn.relu(-x_centered)
        output = ops.concatenate([pos, neg], axis=-1)
        norm = ops.sqrt(
            ops.mean(ops.square(output), axis=-1, keepdims=True)
            + keras.backend.epsilon()  # noqa: E501
        )
        return output / norm

    def build(self, input_shape):
        """Build the ABLE model based on the input shape.

        Args:
            input_shape (tuple): Shape of the input tensor.
        """
        # Check that input_shape is one of the supported formats:
        #   - (batch, pixels, elements)        -> rank == 3
        #   - (batch, pixels, elements, n_ch)  -> rank == 4
        if len(input_shape) not in (3, 4):
            raise ValueError(
                "Input shape must be 3D or 4D tensor. Supported shapes:\n"
                "- (batch, pixels, elements)\n"
                "- (batch, pixels, elements, n_ch)"
            )

        # Compute stacked input channels: last axis after stack_channels is channel axis.
        if len(input_shape) == 3:
            stacked_input_dim = input_shape[-1]
        else:  # len == 4
            stacked_input_dim = input_shape[-2] * input_shape[-1]

        # Dynamically initialize layer dimensions and kernel sizes
        self.layer_dims = self._init_layers(
            self.latent_layers, self.n_latent_layers, stacked_input_dim, self.latent_dim
        )
        self.kernel_sizes = self._init_kernels(self.kernel_size, len(self.layer_dims))

        # Build conv layers and track them in _able_layers
        for idx, (dim, kernel) in enumerate(zip(self.layer_dims, self.kernel_sizes)):
            activation = (
                None if idx == (len(self.layer_dims) - 1) else self.antirectifier
            )
            layer = Conv2D(dim, kernel, activation=activation, padding="same")
            self._able_layers.append(layer)
            # Register as a named attribute so Keras tracks the weights.
            setattr(self, f"_able_conv_{idx}", layer)

        # Eagerly build every Conv2D layer with a concrete dummy input so that
        # their kernels are fully initialised before any JAX tracing occurs.
        # This prevents lazy-build (and random weight initialisation) from being
        # triggered inside jax.value_and_grad / stateless_call, which would
        # cause DynamicJaxprTracers to escape into variable._value.
        dummy = ops.zeros((1, 1, 1, stacked_input_dim))
        for layer in self._able_layers:
            dummy = layer(dummy)

        # Mark the model as built
        super().build(input_shape)

    def call(self, inputs):
        """Apply ABLE to the input data."""

        # This assumes the first dimension is batch (with_batch_dim=True)
        weighed_data = []

        for data in inputs:
            weighed_data.append(self.apply_model(data))

        weighed_data = ops.stack(weighed_data, axis=0)

        return weighed_data

    def apply_model(self, inputs):
        """Apply the ABLE network to a single batch element.

        Args:
            inputs (Tensor): Input tensor (TOF corrected data)

        Returns:
            Tensor: Output tensor after applying the ABLE network (weighted tof-corrected data).
        """

        # Stack final channel dim into element axis when needed
        x_reshaped, meta = self.stack_channels(inputs, self.axis)

        # Compute adaptive weights by passing the (stacked) input
        # through the sequence of convolutional layers that make up the ABLE network.
        weights = x_reshaped
        for conv in self._able_layers:
            weights = conv(weights)

        # Multiply input with computed weights (apply adaptive weighting).
        out = ops.multiply(x_reshaped, weights)

        # Unstack channel dim back to original layout when necessary
        out = self.unstack_channels(out, meta)

        return out


if __name__ == "__main__":
    import os

    os.environ["KERAS_BACKEND"] = "jax"

    # simple test
    model = ABLE(latent_dim=32, n_latent_layers=2, kernel_size=(1, 3))
    print("Layer dims:", model.layer_dims)
    print("Kernel sizes:", model.kernel_sizes)

    # test with dummy input
    import keras
    import numpy as np

    batch_size = 1
    transmits = 5
    height = 64
    width = 64
    elements = 128
    x = np.random.randn(batch_size, transmits, height, width, elements).astype(
        np.float32
    )

    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

    # test with extra channel dim
    model = ABLE(latent_dim=32, n_latent_layers=2, kernel_size=(1, 3))
    n_ch = 2
    x2 = np.random.randn(batch_size, transmits, height, width, elements, n_ch).astype(
        np.float32
    )
    y2 = model(x2)

    print("Input with channel dim shape:", x2.shape)
    print("Output with channel dim shape:", y2.shape)

    # time able with 128 elements and 512x512 grid, batch size 11. Use jit
    from time import perf_counter

    model = ABLE(latent_dim=32, n_latent_layers=1, kernel_size=(3, 3))
    model.compile(jit_compile=True)
    batch_size = 1
    x_large = np.random.randn(batch_size, transmits, 256, 256, 128, 2).astype(
        np.float32
    )
    # move to device
    x_large = keras.ops.convert_to_tensor(x_large)

    # warmup run (traces/compiles the function)
    print("Warmup run (compiling/tracing)...")
    t0 = perf_counter()
    _ = model(x_large).block_until_ready()
    t1 = perf_counter()
    print(f"Warmup run time: {t1 - t0:.6f} s")

    # timed runs
    N = 10

    start = perf_counter()
    for i in range(N):
        out = model(x_large)

    out.block_until_ready()
    end = perf_counter()

    avg_time = (end - start) / N
    print(
        f"Average time per iteration over {N} runs: {avg_time:.6f} s (total {end - start:.6f} s)"
    )
