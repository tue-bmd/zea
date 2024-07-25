"""The beamformers module contains time of flight correction and beamforming
layers for tensorflow.

- **Author(s)**     : Ben Luijten
- **Date**          : Thu Feb 2 2021

## Usage
The tensorflow beamformer has several modes of operation:
1. Using static parameters, i.e. the parameters given by the probe and scan objects during
initialization.
2. Using dynamic parameters, which allows for changing beamforming settings at runtime.
This can be done by:
    a. Passing the probe and scan objects to the call function of the beamformer. This is
    b. Passing the parameters as tensors in the kwargs of the call function.
Option 2.a provides the best integration with the rest of the USBMD framework, but requires
the Beamformer model to be run in eager mode, as the probe and scan objects no tf.Tensors. During
the call function, the probe and scan objects are converted to tf.Tensors, by taking a snapshot of
the objects.
It is still possible to benefit from graph mode, and jit compilation, by passing the jit=True
flag to the Beamformer model during initialization (via the config object). This will compile the
internal beamforming functions, but not the call function of the Beamformer model.
Option 2.b requires manual passing of the parameters as tensors, but allows the full model to be
compiled. Since we don't have to take a snapshot of the probe and scan objects, this is the fastest
option. In fact, it is just as fast as using static parameters.
3. A combination of static and dynamic parameters. If no probe and scan objects, or parameter
Tensors passed as kwargs, are passed to the call function, the Beamformer model will use the
internal probe and scan objects. If probe and scan objects are passed, the internal probe and scan
objects will be updated. If this is not desired, the stateful flag can be set to False during
initialization of the Beamformer model.

Note that for option 1, it is still possible to compile the whole model, including the call, since
we don't have non-tensor inputs. However, this does disable the option to update parameters as in 3.

## Examples
```python
model = get_model(probe, scan, config, stateful=True) # Stateful is on by default
bf = model(data) # beamform using stored settings
bf = model(data, probe2, scan2) # beamform using new settings
bf = model(data) beamform using settings from probe2/scan2
sound_speed = tf.constant(1500, dtype=tf.float32)
bf = model(data, sound_speed=sound_speed) # beamform using new sound speed setting
```

## Training
The beamformer model can also be used (as part of) a training model. The same rules apply as for
inference, however jit compilation should be handled with care. If you want to use jit compilation
for training, it is advised to do this via model.compile(jit_compile=True), as this
will compile the entire model including the gradient computation. This is needed if you have custom
loss functions or metrics (e.g. via model.add_loss() or model.add_metric()), since they need to be
compiled in the same scope as the rest of the model.
Note that this way of compiling only affects the model.fit, model.evaluate and model.predict
functions, but not the direct __call__ function, i.e. model(data).
"""

# pylint: disable=arguments-differ, arguments-renamed, unused-argument

import warnings
from typing import Tuple

import numpy as np
import tensorflow as tf

from usbmd.backend.tensorflow.utils.utils import tf_snapshot
from usbmd.registry import tf_beamformer_registry
from usbmd.utils import log


# TODO This function is possibly not needed anymore
def get_beamformer(probe, scan, config):
    """Function that returns a beamformer model based on the provided configuration."""
    model = Beamformer(probe, scan, config)
    return model


class Beamformer(tf.keras.Model):
    """Beamformer model"""

    def __init__(self, probe, scan, config, stateful=True):
        super().__init__()
        self.probe = tf_snapshot(probe)
        self.scan = tf_snapshot(scan)
        self.config = config

        self.patches = config.model.beamformer.get("patches", 1)

        self.stateful = stateful

        # Beamformer layers
        self.tof_layer = TOFLayer(probe, scan, config)
        self.beamsumming_layer = tf_beamformer_registry[
            config.model.beamformer.get("type", "das")  # default to DAS
        ](probe, scan, config)

        if self.patches > 1:
            self.beamform = self._beamform_patch_wise
        else:
            self.beamform = self._beamform

        # Check if we should jit_compile the model
        try:
            self.jit = config.model.beamformer.get("jit", False)
        except AttributeError:
            self.jit = False
        if self.jit:
            log.info("JIT compiling TF Beamformer")
            self.beamform = tf.function(self.beamform, jit_compile=True)

        if config.model.beamformer.get("auto_pressure_weighting"):
            log.warning(
                "Auto pressure weighting is not yet implemented in the TF Beamformer"
            )

    def call(self, inputs, probe=None, scan=None, **kwargs):
        """Performs beamforming on input data, based on the provided probe and scan.
        Probe and Scan objects can only be given if this function was not graph compiled, i.e.
        build with tf.function().
        You can compile the call function if you do not use the probe and scan objects as
        arguments, but instead pass the values you want to update as tensors in the kwargs.
        """

        # Update probe and scan if new ones are provided.
        # We create a snapshot in the function call such that it is backwards compatible.
        _probe = tf_snapshot(probe) if probe else self.probe
        _scan = tf_snapshot(scan) if scan else self.scan

        if self.stateful:
            self.probe = _probe
            self.scan = _scan

        # In the latest tf versions (2.12 and up as far as I could test), we can no
        # longer pass dicts of Tensors to a tf.function. Instead we add unpack and add to kwargs
        kwargs = {**_scan, **_probe, **kwargs}

        return self.beamform(inputs, **kwargs)["beamformed"]

    def _beamform(self, inputs, **kwargs):
        """Performs beamforming on input data."""
        aligned_data = self.tof_layer(inputs, **kwargs)
        beamformed_data = self.beamsumming_layer(aligned_data, **kwargs)
        return beamformed_data

    def _beamform_patch_wise(self, inputs, **kwargs):
        grid = kwargs["grid"]
        patches, padding_length = split_uneven(grid, self.patches, axis=0)

        beamformed_patches = tf.TensorArray(
            tf.float32, size=self.patches, dynamic_size=False, clear_after_read=True
        )
        for i in range(self.patches):
            patch = patches[i]
            kwargs["grid"] = patch

            beamformed_patch = self._beamform(inputs, **kwargs)["beamformed"]
            beamformed_patches = beamformed_patches.write(i, beamformed_patch)

        beamformed_data = tf.squeeze(
            tf.concat(tf.split(beamformed_patches.stack(), self.patches), axis=2),
            axis=0,
        )
        if padding_length:
            beamformed_data = beamformed_data[:, :-padding_length, :, :]

        return {"beamformed": beamformed_data}


class BeamSumming(tf.keras.layers.Layer):
    """Base class for beamsumming layers"""

    def __init__(self, probe=None, scan=None, config=None, **kwargs):
        self.probe = probe
        self.scan = scan
        self.config = config

        # Show warning if no probe, scan or config is provided
        if not probe:
            log.warning("No probe provided to beamformer layer")
        if not scan:
            log.warning(" No scan provided to beamformer layer")
        if not config:
            log.warning(" No config provided to beamformer layer")

        super().__init__(**kwargs)


@tf_beamformer_registry(name="das", framework="tensorflow")
class DAS_layer(BeamSumming):
    """Layer that implements DAS beamforming"""

    def __init__(self, probe, scan, config, rx_apo=None, tx_apo=None, **kwargs):
        super().__init__(probe, scan, config, **kwargs)
        self.rx_apo = rx_apo if rx_apo else 1
        self.tx_apo = tx_apo if tx_apo else 1

    # pylint: disable=arguments-differ
    def call(self, inputs, **kwargs):
        """Performs DAS beamforming on tof-corrected input.

        Args:
            inputs (tf.Tensor): The TOF corrected input of shape
            (batch_size, n_tx, N_x, N_z, 1 if RF/2 if IQ)

        Returns:
            dict: Output dict with keys ('beamformed') with shape
            (batch_size, N_x, N_z, 1 if RF/2 if IQ)
        """
        # sum channels, i.e. DAS
        summed = tf.reduce_sum(self.rx_apo * inputs, axis=-2)
        # sum transmits, i.e. Compounding
        compounded = tf.reduce_sum(self.tx_apo * summed, axis=1)
        return {"beamformed": compounded}


class TOFLayer(tf.keras.layers.Layer):
    """Time-flight-correction layer"""

    def __init__(
        self,
        probe,
        scan,
        config,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Lets not store probe, scan and config here, as this will be stored in the Beamformer model
        # Here we only need to store derived parameters, such as delays and mask
        self.scan = scan  # only used in  format_input , which needs a rewrite

    def call(self, inputs, **kwargs):
        """Performs time-of-flight correction on input data.

        Args:
            inputs (tf.Tensor): The input of shape
            (batch_size, n_tx, n_ax, n_el, 1 if RF/2 if IQ)
        Returns:
            tf.Tensor: The TOF corrected input of shape
            (batch_size, n_tx, n_ax, n_el, 1 if RF/2 if IQ)
        """
        # Pre-process input
        inputs = self.format_input(inputs)  # TODO check overhead in compute time
        # inputs = tf.transpose(inputs, perm=[0, 1, 3, 2, 4])

        # Compute delays and mask
        # It is currently faster to recompute the delays and mask for each batch, than to store
        # them and update them when the probe or scan changes. Most likely because the overhead of
        # branching (tf.cond) is larger than the overhead of recomputing the delays and mask.
        # The improvement in compute time without having to recompute the delays and mask is
        # approximately 1ms for a single PW.

        grid_shape = tf.shape(kwargs["grid"])
        Nz = grid_shape[0]
        Nx = grid_shape[1]

        raw_data_shape = tf.shape(inputs)
        batch_size = raw_data_shape[0]
        n_tx = raw_data_shape[1]
        n_el = inputs.shape[2]
        n_ax = raw_data_shape[3]
        n_ch = inputs.shape[
            4
        ]  # we use the static shape such that we only trace one branch
        # batch_size, n_tx, n_el, n_ax, n_ch = tf.shape(inputs)

        flatgrid = tf.reshape(
            kwargs["grid"], shape=(-1, 3)
        )  # Flatten grid to simplify calculations
        txdel, rxdel = compute_delays(
            flatgrid,
            kwargs["probe_geometry"],
            kwargs["initial_times"],
            kwargs["t0_delays"],
            kwargs["fs"],
            kwargs["sound_speed"],
            n_tx,
            n_el,
        )
        mask = compute_mask(flatgrid, kwargs["probe_geometry"], kwargs["f_number"])

        # Perform time-of-flight correction on each batch
        aligned_data = tf.TensorArray(
            tf.float32, size=batch_size, dynamic_size=False, clear_after_read=True
        )
        for b in tf.range(batch_size):
            aligned_data = aligned_data.write(
                b,
                tof_correction(
                    inputs[b],
                    flatgrid,
                    txdel,
                    rxdel,
                    kwargs["sound_speed"],
                    kwargs["fs"],
                    kwargs["fc"],
                    kwargs["fdemod"],
                    n_tx,
                    n_el,
                    n_ax,
                    n_ch,
                    Nz,
                    Nx,
                    mask,
                ),
            )

        # Stack batches of TOF corrected data in a single tensor
        return aligned_data.stack()

    def format_input(self, inputs):
        """Verify input data shape and transpose if necessary"""
        # TODO: This function needs to be rewritten using kwargs
        usbmd_shape = (self.scan.n_tx, self.scan.n_ax, self.scan.n_el, self.scan.n_ch)
        orig_shape = (self.scan.n_tx, self.scan.n_el, self.scan.n_ax, self.scan.n_ch)
        inputs_shape = inputs.shape[1:]

        # Lets check the shape of the input data, we don't need to check the first dimension (batch)
        if inputs_shape == usbmd_shape:
            return tf.transpose(inputs, perm=[0, 1, 3, 2, 4])
        if inputs_shape == orig_shape:
            warnings.warn(
                "Warning: The dimensions of the input data seem to be in the wrong "
                "order. Permuting dimensions from (n_frames, n_tx, n_ax, n_el, n_ch) "
                "to (n_frames, n_tx, n_ax, n_el, n_ch).\n"
                "This functionality is for backwards compatibility and will be "
                "removed in a future update."
            )
            return inputs
        if inputs.shape[2] is None and inputs.shape[3] is None:
            return inputs
        raise ValueError(
            f"Input data should have shape {usbmd_shape} or {orig_shape} (deprecated), "
            f"got {inputs_shape}"
        )


def compute_output_shape(self):
    """Computes the output shape of the model"""
    output_shape = (
        self.batch_size,
        self.scan.Nx,
        self.scan.Nz,
        self.scan.n_tx,
        self.scan.n_el,
    )
    return output_shape


def tof_correction(
    data: tf.Tensor,
    grid,
    txdel,
    rxdel,
    sound_speed,
    fs,
    fc,
    fdemod,
    n_tx,
    n_el,
    n_ax,
    n_ch,
    Nz,
    Nx,
    mask,
) -> tf.Tensor:
    """
    Args:
        data (tf.Tensor): Input RF/IQ data. (n_tx, n_el, n_ax, 1 if RF/2 if IQ)

    Returns:
        output (tf.Tensor): time-of-flight corrected data
        with shape: (Nz, Nx, n_tx, n_el).

    """
    # Apply delays
    bf_tx = tf.TensorArray(
        tf.float32, size=n_tx, dynamic_size=False, clear_after_read=True
    )
    for tx in tf.range(n_tx):
        data_tx = data[tx]
        delays = txdel[tx] + rxdel
        delays = tf.expand_dims(delays, axis=-1)

        tof_tx = apply_delays(data_tx, delays, clip=[0, n_ax - 1]) * mask

        if n_ch == 2 and fdemod != 0:
            # Apply phase rotation
            tshift = delays[:, :, 0] / fs
            tdemod = tf.expand_dims(grid[:, 2], 0) * 2 / sound_speed
            theta = 2 * tf.constant(np.pi, dtype=tf.float32) * fc * (tshift - tdemod)
            tof_tx = complex_rotate(tof_tx, theta)

        tof_tx = tf.transpose(
            tof_tx, perm=[1, 0, 2]
        )  # Transpose to ensure correct reshape order
        tof_tx = tf.reshape(
            tof_tx,
            shape=(Nz, Nx, n_el, n_ch),
        )
        bf_tx = bf_tx.write(tx, tof_tx)

    return bf_tx.stack()


def compute_delays(
    grid: tf.Tensor,
    probe_geometry: tf.Tensor,
    initial_times: tf.Tensor,
    t0_delays: tf.Tensor,
    fs: tf.Tensor,
    sound_speed: tf.Tensor,
    n_tx: int,
    n_el: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Function that calculates delays for time-of-flight correction

    Args:
        grid (tf.Tensor): Pixel positions in cartesian coordinates (N_pixels, 3)
        angles (tf.Tensor): Plane wave angles (n_tx,)
        focus (tf.Tensor): Virtual focus point, not used for PW transmissions (n_tx, 3)
        probe_geometry (tf.Tensor): Array element positions cartesian coordinates (n_el, 3)
        initial_times (tf.Tensor): Time-offset for each transmit (n_tx,)
        t0_delays (tf.Tensor): Times at which the elements fire
        fs (tf.Tensor): Sampling frequency in Hz (1,)
        sound_speed (tf.Tensor): Speed-of-sound (1,)

    Returns:
        txdel (tf.Tensor): Transmit delays (n_tx, N_pixels)
        rxdel (tf.Tensor): Receive delays (n_el, N_pixels)
    """

    # Initialize delay variables
    tx_distances = tf.TensorArray(
        tf.float32, size=n_tx, dynamic_size=False, clear_after_read=True
    )
    rx_distances = tf.TensorArray(
        tf.float32, size=n_el, dynamic_size=False, clear_after_read=True
    )

    # Compute transmit and receive delays and apodizations
    # pylint: disable=cell-var-from-loop
    for tx in tf.range(n_tx):
        tx_distance = distance_tx_generic(
            grid, t0_delays[tx], probe_geometry, sound_speed=sound_speed
        )
        tx_distances = tx_distances.write(tx, tx_distance)
    tx_distances = tx_distances.stack()

    for el in tf.range(n_el):
        rx_distance = distance_rx(grid, probe_geometry[el])
        rx_distances = rx_distances.write(el, rx_distance)
    rx_distances = rx_distances.stack()

    tx_delays = (tx_distances / sound_speed - tf.expand_dims(initial_times, -1)) * fs
    rx_delays = (rx_distances / sound_speed) * fs

    return tx_delays, rx_delays


def apply_delays(  # TODO add more interpolations methods (nearest, cubic, etc.)
    iq: tf.Tensor, d: tf.Tensor, clip: Tuple[float, float] = None
) -> tf.Tensor:
    """Apply time delays using linear interpolation.

    Args:
        iq (tf.Tensor): Input tensor of shape [batch_size, pixels, RF/IQ channels].
            RF / IQ channels are assumed to be the last dimension and either
            1 (RF) or 2 (IQ) channels respectively.
        d (tf.Tensor): Time delays tensor of shape [batch_size, pixels].
        clip (tuple, optional): Tuple of two values representing the lower and upper
            bounds for clipping the delays.

    Returns:
        tf.Tensor: Output tensor of shape [batch_size, pixels, RF/IQ channels].
    """
    # Get lower and upper values around delays dd
    d0 = tf.cast(tf.floor(d), "int32")  # Cast to integer
    d1 = d0 + 1
    # Apply clipping of delays clipping to ensure correct behavior on cpu
    if clip:
        d0 = tf.clip_by_value(d0, clip[0], clip[1])
        d1 = tf.clip_by_value(d1, clip[0], clip[1])
    # Gather pixel values
    iq0 = tf.gather_nd(iq, d0, batch_dims=1)
    iq1 = tf.gather_nd(iq, d1, batch_dims=1)
    # Compute interpolated pixel value
    d0 = tf.cast(d0, "float")  # Cast to float
    d1 = tf.cast(d1, "float")  # Cast to float
    out = (d1 - d) * iq0 + (d - d0) * iq1
    return out  # [batch_size, pixels, RF/IQ channels]


def complex_rotate(iq: tf.Tensor, theta: tf.Tensor) -> tf.Tensor:
    """Phase rotation of I and Q component by complex angle theta

    Args:
        iq (tf.Tensor): IQ data (N_pixels, n_ch, 2)
        theta (tf.Tensor): Phase angle (N_pixels, n_ch)

    Returns:
        (tf.Tensor): Rotated IQ data (N_pixels, n_ch, 2)
    """
    i = iq[:, :, 0]
    q = iq[:, :, -1]

    ir = i * tf.cos(theta) - q * tf.sin(theta)
    qr = q * tf.cos(theta) + i * tf.sin(theta)

    return tf.stack([ir, qr], axis=-1)


def distance_rx(grid: tf.Tensor, probe_geometry: tf.Tensor) -> tf.Tensor:
    """Calculate receive delays

    Args:
        grid (tf.Tensor): Pixel positions in cartesian coordinates (N_pixels, 3)
        probe_geometry (tf.Tensor): Array element positions cartesian coordinates (n_el, 3)

    Returns:
        (tf.Tensor): Distance from each pixel to each array element (N_pixels, n_el)
    """
    # Get norm of distance vector between elements and pixels via broadcasting
    dist = tf.norm(grid - tf.expand_dims(probe_geometry, 0), axis=-1)
    return dist


def distance_tx_pw(grid: tf.Tensor, angle: tf.Tensor) -> tf.Tensor:
    """Calculate transmit delays for planar transmissions

    Args:
        grid (tf.Tensor): Pixel positions in cartesian coordinates (N_pixels, 3)
        angle (tf.Tensor): Plane wave angles (1,)

    Returns:
        dist (tf.Tensor): Distance from PW source to each pixel (N_pixels)
    """
    # Use broadcasting to simplify computations
    x = grid[:, 0]
    z = grid[:, 2]
    # For each element, compute distance to pixels
    angle = tf.cast(angle, "float32")
    dist = x * tf.math.sin(angle) + z * tf.math.cos(angle)
    return dist


def distance_tx_foc(grid: tf.Tensor, focus_distances: tf.Tensor) -> tf.Tensor:
    """Calculate transmit delays for focused transmissions

    Args:
        grid (tf.Tensor): Pixel positions in cartesian coordinates (N_pixels, 3)
        focus_distances (tf.Tensor): Virtual focus point (3,)

    Returns:
        (tf.Tensor): Distance from virtual focus to each pixel (N_pixels)
    """
    # Use broadcasting to simplify computations
    x = grid[:, 0]
    z = grid[:, 2]
    # For each element, compute distance to pixels
    zs = focus_distances[2]
    xs = focus_distances[0]
    dist = tf.sqrt((xs - x) ** 2 + (zs - z) ** 2)
    return dist


def distance_tx_generic(
    grid: tf.Tensor,
    t0_delays: tf.Tensor,
    probe_geometry: tf.Tensor,
    sound_speed: tf.Tensor = 1540,
) -> tf.Tensor:
    """Computes distance to user-defined pixels for generic transmits based on
    the t0_delays.

    Args:
        grid (tf.Tensor): Flattened tensor of pixel positions in x,y,z of shape
        `(n_pix, 3)`
        t0_delays (tf.Tensor): The transmit delays in seconds of shape (n_el,), shifted
            such that the smallest delay is 0. Defaults to None.
        probe_geometry (tf.Tensor): The positions of the transducer elements of shape
            `(n_el, 3)`.
        sound_speed (float): The speed of sound in m/s. Defaults to 1540.

    Returns:
        tf.Tensor: The distances of the wavefront to the pixels of shape `(n_pix,)`
    """

    # Expanding dimensions to allow for broadcasting
    diff = tf.expand_dims(grid, 0) - tf.expand_dims(probe_geometry, 1)
    dist = t0_delays[..., tf.newaxis] * sound_speed + tf.norm(
        diff, axis=-1, keepdims=False
    )
    dist = tf.reduce_min(dist, axis=0)

    return dist


def compute_mask(
    grid: tf.Tensor, probe_geometry: tf.Tensor, f_number: tf.Tensor
) -> tf.Tensor:
    """Calculates an apodization mask based on the f-number (receive aperture)

    Args:
        grid (tf.Tensor): Pixel positions in cartesian coordinates (n_pixels, 3)
        probe_geometry (tf.Tensor): Array element positions cartesian coordinates (n_el, 3)
        f_number (tf.Tensor): F-number of the scan

    Returns:
        mask (tf.Tensor): Apodization mask (n_el, N_pixels, 1)
    """

    if f_number == 0:
        mask = tf.ones(
            (tf.shape(probe_geometry)[0], tf.shape(grid)[0], 1), dtype=tf.float32
        )
    else:
        n_pixels = tf.shape(grid)[0]
        n_el = tf.shape(probe_geometry)[0]

        aperture = grid[:, 2] / f_number
        aperture = tf.matmul(tf.expand_dims(aperture, axis=1), tf.ones((1, n_el)))
        distance = tf.abs(
            tf.matmul(tf.expand_dims(grid[:, 0], axis=1), tf.ones((1, n_el)))
            - tf.matmul(
                tf.ones((n_pixels, 1)), tf.expand_dims(probe_geometry[:, 0], axis=0)
            )
        )

        mask = tf.cast(distance <= aperture / 2, "float32")
        mask = tf.transpose(mask)
        mask = tf.expand_dims(mask, axis=-1)

    return mask


def split_uneven(tensor: tf.Tensor, N: int, axis: int = 0):
    """Function similar to tf.split, but also works if the tensor cannot be split in an equally
    sized parts. The last part will be smaller and zero padding is added to ensure the correct
    shape.

    args:
        tensor (Tensor): Tensor to split
        N (int): Number of parts to split the tensor in

    returns:
        list: List of tensors
        remainder (int): Number of elements in the last part
    """

    # check if the tensor can be split in N equal parts
    remainder = tensor.shape[axis] % N

    if remainder:
        padding_length = N - remainder
        padding_shape = list(tensor.shape)
        padding_shape[axis] = padding_length
        padding = tf.zeros(padding_shape)
        tensor = tf.concat([tensor, padding], axis=axis)
    else:
        padding_length = 0

    return tf.split(tensor, N, axis=axis), padding_length
