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

    Implements a configurable pixel-wise convolutional encoder/decoder that
    computes per-element adaptive weights from time-of-flight-corrected
    channel data.  The weighted data is then summed by a subsequent
    :class:`~zea.ops.DelayAndSum` operation to form the final image.

    .. admonition:: Reference

        Luijten, B. *et al.*, "Adaptive Ultrasound Beamforming Using Deep
        Learning," *IEEE Trans. Med. Imaging* **39** (12), 2020.
        https://doi.org/10.1109/TMI.2020.3008537

    Expected input shape when used inside a
    :class:`~zea.ops.PatchedGrid` pipeline via :class:`~zea.ops.Lambda`:

    - ``(n_tx, n_pix, n_el)``       — RF data
    - ``(n_tx, n_pix, n_el, n_ch)`` — IQ data

    The model maps over the transmit axis (``n_tx``) internally; each
    ``(n_pix, n_el[, n_ch])`` slice is processed independently.

    .. note::
        Only 1×1 convolutions (``kernel_size=1``) are currently supported
        because :class:`~zea.ops.PatchedGrid` processes pixels independently.

    Args:
        latent_dim (int): Channel size for the hidden layers when
            ``latent_layers`` is not supplied. Default is ``32``.
        kernel_size (int, tuple, or list): Kernel size specification.

            - ``int``  — every convolution uses ``(k, k)``.
            - ``tuple (h, w)`` — every convolution uses ``(h, w)``.
            - ``list`` of ints/tuples — per-layer kernel sizes
              (length must equal the total number of layers).

            Default is ``1``.
        n_latent_layers (int): Number of hidden layers when ``latent_layers``
            is not supplied. Must be ≥ 1. Default is ``2``.
        latent_layers (list or None): Explicit list of channel sizes for the
            hidden layers.  Overrides ``n_latent_layers`` and ``latent_dim``
            when provided. Default is ``None``.
        axis (int or None): Reserved for future use. Default is ``None``.
        name (str): Model name forwarded to :class:`~zea.models.base.BaseModel`.
            Default is ``"able"``.
        **kwargs: Additional keyword arguments forwarded to
            :class:`~zea.models.base.BaseModel`.
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "latent_dim": self.latent_dim,
                "kernel_size": self.kernel_size,
                "n_latent_layers": self.n_latent_layers,
                "latent_layers": self.latent_layers,
                "axis": self.axis,
            }
        )
        return config

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
                        "Only kernel_size=1 is currently supported due to PatchedGrid limitations."
                    )
                return True
            if isinstance(k, tuple) and len(k) == 2 and all(isinstance(x, int) for x in k):
                if k != (1, 1):
                    raise ValueError(
                        "Only kernel_size=(1,1) is currently supported due to "
                        "PatchedGrid limitations."
                    )
                return True
            raise ValueError("kernel_size entries must be int or tuple(int,int)")

        def _normalize_entry(k):
            _allowed_kernel_size(k)  # TODO: Remove this check once larger kernels are supported
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
            ValueError: If ``latent_layers`` is invalid or its length does not match
                ``n_latent_layers``.
        """

        if latent_layers is not None:
            if not isinstance(latent_layers, (list, tuple)):
                raise ValueError("`latent_layers` must be a list/tuple of integers when provided.")
            layer_dims = list(latent_layers)
            if len(layer_dims) != n_latent_layers:
                raise ValueError(
                    "`latent_layers` must contain exactly `n_latent_layers`"
                    " entries for the inner layers."
                )
            if not all(isinstance(d, int) and d > 0 for d in layer_dims):
                raise ValueError("All `latent_layers` entries must be positive integers.")
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
                # (pixels, 1, 1, n_ch*elements) -> (pixels, n_ch, elements)
                # -> (pixels, elements, n_ch)
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
            ops.mean(ops.square(output), axis=-1, keepdims=True) + keras.backend.epsilon()  # noqa: E501
        )
        return output / norm

    def build(self, input_shape):
        """Build the ABLE model based on the input shape.

        Args:
            input_shape (tuple): Shape of the input tensor.  Supported
                formats (``n_tx`` acts as the batch axis that ``call``
                maps over):

                - ``(n_tx, n_pix, n_el)``       — RF data, rank 3
                - ``(n_tx, n_pix, n_el, n_ch)`` — IQ data, rank 4
        """
        # Supported ranks:
        #   rank 3: (n_tx, n_pix, n_el)
        #   rank 4: (n_tx, n_pix, n_el, n_ch)
        if len(input_shape) not in (3, 4):
            raise ValueError(
                "Input shape must be rank-3 or rank-4. Supported shapes:\n"
                "- (n_tx, n_pix, n_el)\n"
                "- (n_tx, n_pix, n_el, n_ch)"
            )

        # The per-transmit slice passed to apply_model has shape
        # (n_pix, n_el) or (n_pix, n_el, n_ch); stack_channels merges
        # the element and channel axes so Conv2D sees (n_pix, 1, 1, n_el*n_ch).
        if len(input_shape) == 3:
            stacked_input_dim = input_shape[-1]  # n_el
        else:  # len == 4
            stacked_input_dim = input_shape[-2] * input_shape[-1]  # n_el * n_ch

        # Dynamically initialize layer dimensions and kernel sizes
        self.layer_dims = self._init_layers(
            self.latent_layers, self.n_latent_layers, stacked_input_dim, self.latent_dim
        )
        self.kernel_sizes = self._init_kernels(self.kernel_size, len(self.layer_dims))

        # Build conv layers and track them in _able_layers
        for idx, (dim, kernel) in enumerate(zip(self.layer_dims, self.kernel_sizes)):
            activation = None if idx == (len(self.layer_dims) - 1) else self.antirectifier
            layer = Conv2D(dim, kernel, activation=activation, padding="same")
            self._able_layers.append(layer)
            # Register as a named attribute so Keras tracks the weights.
            setattr(self, f"_able_conv_{idx}", layer)

        # Build each Conv2D layer by calling layer.build() directly (no forward
        # pass).  Calling ops.zeros() + layer(dummy) inside a lax.map/scan
        # trace (which Pipeline triggers via ops.map with with_batch_dim=True)
        # would run add_weight() inside the JAX trace context, producing
        # DynamicJaxprTracers that escape the scan scope →
        # UnexpectedTracerError.  Calling layer.build(input_shape) only creates
        # Keras Variable objects; any JAX random ops for weight initialisation
        # that run here produce *concrete* DeviceArrays because we are in eager
        # mode (no active lax.map trace) when build() is called correctly.
        #
        # IMPORTANT: ensure able_model.build() (or an initial forward pass) is
        # called in eager mode *before* the model is first used inside a
        # Pipeline with with_batch_dim=True, so this code never runs inside a
        # lax.map trace.
        current_in_channels = stacked_input_dim
        for idx, layer in enumerate(self._able_layers):
            layer.build((1, 1, 1, current_in_channels))
            is_last = idx == len(self._able_layers) - 1
            # antirectifier activation doubles the channel count;
            # the final layer has no activation, so channels stay at layer.filters.
            current_in_channels = layer.filters if is_last else layer.filters * 2

        # Mark the model as built
        super().build(input_shape)

    def call(self, inputs):
        """Apply ABLE to the input data.

        Maps ``apply_model`` over the first axis (``n_tx``) using
        :func:`keras.ops.map` so the forward pass is fully traceable
        by JAX and compatible with gradient computation.

        Args:
            inputs (Tensor): Shape ``(n_tx, n_pix, n_el[, n_ch])``.

        Returns:
            Tensor: Adaptively weighted data with the same shape as
            ``inputs``.
        """
        return ops.map(self.apply_model, inputs)

    def apply_model(self, inputs):
        """Apply the ABLE network to a single transmit slice.

        Args:
            inputs (Tensor): TOF-corrected data for one transmit event,
                shape ``(n_pix, n_el)`` or ``(n_pix, n_el, n_ch)``.

        Returns:
            Tensor: Adaptively weighted data with the same shape as
            ``inputs``.
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

    import keras
    import numpy as np

    # Expected input shape: (n_tx, n_pix, n_el) or (n_tx, n_pix, n_el, n_ch)
    # The model maps over n_tx internally.
    n_tx = 5
    n_pix = 64 * 64  # flattened spatial grid
    n_el = 128

    # --- RF data (no channel dim) ---
    model = ABLE(latent_dim=32, n_latent_layers=2, kernel_size=1)
    x_rf = np.random.randn(n_tx, n_pix, n_el).astype(np.float32)
    y_rf = model(x_rf)
    print("RF  input shape: ", x_rf.shape)
    print("RF  output shape:", y_rf.shape)
    assert y_rf.shape == x_rf.shape, "RF output shape mismatch"

    # --- IQ data (n_ch = 2) ---
    n_ch = 2
    model_iq = ABLE(latent_dim=32, n_latent_layers=2, kernel_size=1)
    x_iq = np.random.randn(n_tx, n_pix, n_el, n_ch).astype(np.float32)
    y_iq = model_iq(x_iq)
    print("IQ  input shape: ", x_iq.shape)
    print("IQ  output shape:", y_iq.shape)
    assert y_iq.shape == x_iq.shape, "IQ output shape mismatch"

    print("Layer dims: ", model_iq.layer_dims)
    print("Kernel sizes:", model_iq.kernel_sizes)

    # --- Timing with JIT (IQ, larger grid) ---
    from time import perf_counter

    model_jit = ABLE(latent_dim=32, n_latent_layers=1, kernel_size=1)
    model_jit.compile(jit_compile=True)
    x_large = np.random.randn(n_tx, 256 * 256, 128, 2).astype(np.float32)
    x_large = keras.ops.convert_to_tensor(x_large)

    print("Warmup run (compiling/tracing)...")
    t0 = perf_counter()
    _ = model_jit(x_large).block_until_ready()
    t1 = perf_counter()
    print(f"Warmup: {t1 - t0:.3f} s")

    N = 5
    start = perf_counter()
    for _ in range(N):
        out = model_jit(x_large)
    out.block_until_ready()
    end = perf_counter()
    print(f"Avg over {N} runs: {(end - start) / N:.3f} s")
