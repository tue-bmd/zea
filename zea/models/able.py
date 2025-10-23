"""Adaptive Beamforming by Deep LEarning (ABLE).

Original implementation of paper:
    - "Adaptive Ultrasound Beamforming Using Deep Learning"
    - https://doi.org/10.1109/TMI.2020.3008537
    - Author: Ben Luijten
"""

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
        axis=3,
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

        # Placeholder for dynamically set layer dimensions and network
        self.layer_dims = None
        self.kernel_sizes = None
        self.network = self._get_network(self.layer_dims, self.kernel_sizes, axis=self.axis)

        # Track layers belonging to ABLE
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
                raise ValueError("`latent_layers` must be a list/tuple of integers when provided.")
            layer_dims = list(latent_layers)
            if len(layer_dims) != n_latent_layers:
                raise ValueError(
                    "`latent_layers` must contain exactly `n_latent_layers` entries for the inner layers."
                )
            if not all(isinstance(d, int) and d > 0 for d in layer_dims):
                raise ValueError("All `latent_layers` entries must be positive integers.")
            return [input_dim] + layer_dims + [input_dim]

        # Default behavior: create symmetric layers
        middle = [latent_dim] * n_latent_layers
        return [input_dim] + middle + [input_dim]

    def _get_network(self, layer_dims, kernel_sizes, axis=3):
        """Create a callable network function based on layer dimensions and kernel sizes.

        Args:
            layer_dims (list of int): Channel sizes for each layer.
            kernel_sizes (list of tuple): List of (h, w) kernel tuples for each layer
                (same length as `layer_dims`).
            axis (int): Axis index in the input tensor that corresponds to the transducer elements.

        Returns:
            function: A callable function that applies the network to an input tensor.
        """

        def forward_pass(x):
            """Apply the network to input tensor `x`.

            Supports inputs with shapes like:
              - (batch, H, W, elements) -- standard
              - (batch, H, W, elements, n_ch) -- extra channel dim (n_ch==1 or 2)
              - or any permutation where ``axis`` points to the elements axis.
            """

            # Infer input shape and dynamically initialize layers on first pass
            if not self._able_layers:
                # Stack final channel dim into element axis when needed
                x_reshaped, meta = self.stack_channels(x, axis)
                stacked_input_dim = ops.shape(x_reshaped)[axis]

                # Dynamically initialize layer dimensions and kernel sizes
                self.layer_dims = self._init_layers(
                    self.latent_layers, self.n_latent_layers, stacked_input_dim, self.latent_dim
                )
                self.kernel_sizes = self._init_kernels(self.kernel_size, len(self.layer_dims))

                # Build conv layers and track them in _able_layers
                for idx, (dim, kernel) in enumerate(zip(self.layer_dims, self.kernel_sizes)):
                    activation = None if idx == (len(self.layer_dims) - 1) else self.antirectifier
                    self._able_layers.append(
                        Conv2D(dim, kernel, activation=activation, padding="same")
                    )

            # Use the reshaped input for the forward pass
            x_reshaped, meta = self.stack_channels(x, axis)

            out = x_reshaped
            for conv in self._able_layers:
                out = conv(out)

            # Multiply input with output (apply adaptive weighting)
            # Ensure shapes match: broadcast if needed
            out = ops.multiply(x_reshaped, out)

            # Unstack channel dim back to original layout when necessary
            out = self.unstack_channels(out, meta)

            return out

        return forward_pass

    def stack_channels(self, x, axis):
        """Stack the final channel dimension into the element axis.

        Args:
            x (Tensor): Input tensor.
            axis (int): Axis index corresponding to the transducer elements.

        Returns:
            tuple: Transformed tensor and metadata for reversing the operation.
        """
        rank = len(x.shape)
        elem_axis = axis if axis >= 0 else rank + axis
        meta = {"stacked": False}

        if rank >= 5:
            last_idx = rank - 1
            perm = list(range(rank))
            if last_idx != elem_axis + 1:
                perm.pop(last_idx)
                perm.insert(elem_axis + 1, last_idx)
                x_perm = ops.permute_dimensions(x, perm)
                meta["perm"] = perm
            else:
                x_perm = x
                meta["perm"] = None

            shape = ops.shape(x_perm)
            left = shape[:elem_axis]
            elem = shape[elem_axis]
            ch = shape[elem_axis + 1]
            right = shape[elem_axis + 2 :]
            new_elem = elem * ch
            new_shape = left + (new_elem,) + right
            x_reshaped = ops.reshape(x_perm, new_shape)
            meta.update({"stacked": True, "elem_axis": elem_axis, "elem": elem, "ch": ch})
            return x_reshaped, meta

        return x, meta

    def unstack_channels(self, x, meta):
        """Reverse the stacking operation performed by `stack_channels`.

        Args:
            x (Tensor): Input tensor.
            meta (dict): Metadata produced during the stacking operation.

        Returns:
            Tensor: Tensor with the original layout restored.
        """
        if not meta.get("stacked", False):
            return x

        elem_axis = meta["elem_axis"]
        # shape used when stacking
        shape = ops.shape(x)
        left = shape[:elem_axis]
        right = shape[elem_axis + 1 :]
        # recover original elem and ch
        elem = meta["elem"]
        ch = meta["ch"]
        out_shape = left + (elem, ch) + right
        x_unreshaped = ops.reshape(x, out_shape)

        perm = meta.get("perm")
        if perm is not None:
            # invert permutation
            inv = [0] * len(perm)
            for i, p in enumerate(perm):
                inv[p] = i
            x_final = ops.permute_dimensions(x_unreshaped, inv)
            return x_final

        return x_unreshaped

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
            ops.mean(ops.square(output), axis=-1, keepdims=True) + keras.backend.epsilon()
        )
        return output / norm

    def build(self, input_shape):
        """Build the ABLE model based on the input shape.

        Args:
            input_shape (tuple): Shape of the input tensor.
        """
        # Infer the stacked input dimension by simulating the stacking process
        elem_axis = self.axis if self.axis >= 0 else len(input_shape) + self.axis
        trailing_axis = -1  # Always use the final axis as the trailing axis

        # Multiply the trailing dimension with the element dimension if it exists
        if len(input_shape) > elem_axis + 1:
            stacked_input_dim = input_shape[elem_axis] * input_shape[trailing_axis]
        else:
            stacked_input_dim = input_shape[elem_axis]

        # Dynamically initialize layer dimensions and kernel sizes
        self.layer_dims = self._init_layers(
            self.latent_layers, self.n_latent_layers, stacked_input_dim, self.latent_dim
        )
        self.kernel_sizes = self._init_kernels(self.kernel_size, len(self.layer_dims))

        # Build conv layers and track them in _able_layers
        for idx, (dim, kernel) in enumerate(zip(self.layer_dims, self.kernel_sizes)):
            activation = None if idx == (len(self.layer_dims) - 1) else self.antirectifier
            self._able_layers.append(Conv2D(dim, kernel, activation=activation, padding="same"))

        # Mark the model as built
        super().build(input_shape)

    def call(self, inputs):
        """Apply the ABLE network to the input data.

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

    batch_size = 5
    height = 64
    width = 64
    elements = 128
    x = np.random.randn(batch_size, height, width, elements).astype(np.float32)

    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

    # test with extra channel dim
    model = ABLE(latent_dim=32, n_latent_layers=2, kernel_size=(1, 3))
    n_ch = 2
    x2 = np.random.randn(batch_size, height, width, elements, n_ch).astype(np.float32)
    y2 = model(x2)

    print("Input with channel dim shape:", x2.shape)
    print("Output with channel dim shape:", y2.shape)

    # time able with 128 elements and 512x512 grid, batch size 11. Use jit
    from time import perf_counter

    model = ABLE(latent_dim=32, n_latent_layers=1, kernel_size=(3, 3))
    model.compile(jit_compile=True)
    batch_size = 1
    x_large = np.random.randn(batch_size, 256, 256, 128, 2).astype(np.float32)
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
    print(f"Average time per iteration over {N} runs: {avg_time:.6f} s (total {end - start:.6f} s)")
