"""Pytorch version of the tensorflow beamformer.

- **Author(s)**     : Vincent van de Schaft
- **Date**          : Thu Feb 16 2023
Beamformer functionality implemented in pytorch.

## Glossary
- When the documentation states that a tensor is 'of shape `(n_tx, n_el)`' it means
    that `(a, b, c)` are the sizes of the dimensions of the tensor.
- When the documentation states that a tensor is of shape `(tx, el)` it means
    that the first dimension selects over the transmits and the second
    dimension selects over elements without stating explicitly how large these
    dimensions are.

### Abbreviations
| abbreviation  | description                                      |
| ------------- | ------------------------------------------------ |
| `tx`          | Transmit                                         |
| `el`          | Transducer element                               |
| `ax`          | Axial sample                                     |
| `ch`          | Data channel (only 1 for rf-data, 2 for IQ-data) |
| `pix`         | Pixel                                            |
| `xyz`         | Spatial dimension of size 3                      |
| `n_something` | Number of something                              |
| `n_z`         | Height of the beamforming grid                   |
| `n_x`         | Width of the beamforming grid                    |

### Variables
These variable names are used throughout the code and documentation.

| name            | description                                           | shape          |
| --------------- | ----------------------------------------------------- | -------------- |
| `grid`          | The grid of points to beamform to                     | `(n_z, n_x, 3)`|
| `flatgrid`      | The grid of points to beamform to, flattened          | `(n_pix, 3)`   |
| `probe_geometry`| The positions of the elements                         | `(n_el, 3)`    |
| `t0_delays`     | The delay to firing with respect to the first element | `(n_tx, n_el)` |
"""

# pylint: disable=no-member
import numpy as np
import torch

from usbmd.registry import torch_beamformer_registry
from usbmd.utils import log
from usbmd.utils.checks import _check_raw_data


def get_beamformer(probe, scan, config, aux_inputs=("grid"), aux_outputs=()):
    """Creates a beamformer from a probe, a scan, and a config file.

    Args:
        probe (Probe): The probe object.
        scan (Scan): The scan object.
        config (utils.config.Config): The config.
        aux_inputs (tuple, optional): Additional elements to input. Defaults
            to ('grid').
        aux_outputs (tuple, optional): Additional elements to output. Defaults
            to ().

    Returns:
        Beamformer: The initialized beamformer object.
    """
    if aux_inputs != ("grid"):
        log.warning("aux_inputs not implemented for torch beamformer.")
    if aux_outputs != ():
        log.warning("aux_outputs not implemented for torch beamformer.")
    beamformer = Beamformer(probe, scan, config)

    jit_compile = config.model.get("jit")
    if not jit_compile:
        return beamformer

    # Compile the beamformer
    try:
        beamformer = torch.jit.script(beamformer)
    except Exception as e:
        print(f"Could not compile beamformer: {e}")
        print("Falling back to non-compiled beamformer.")
    return beamformer


class Beamformer(torch.nn.Module):
    """Generic beamformer class."""

    def __init__(self, probe, scan, config) -> None:
        """
        Initializes a beamformer
        """
        super().__init__()

        #: Scan class with all relevant scan parameters.
        self.scan = scan

        #: The `usbmd.probes.Probe` object to beamform with.
        self.probe = probe
        #: The `usbmd.config.Config` object to beamform with.
        self.config = config

        #: The time-of-flight correction layer.
        self.tof_layer = TOF_layer(probe, scan, config.model.batch_size)
        #: The delay-and-sum layer, which performs the beamsumming.
        self.das_layer = DAS_layer()

        # Store modules in module dictionary to properly register them
        #: A `torch.nn.ModuleDict` containing all the modules of the beamformer.
        self.all_modules = torch.nn.ModuleDict(
            {"tof_layer": self.tof_layer, "das_layer": self.das_layer}
        )

    def forward(self, data, scan=None, probe=None):
        """Applies delay and sum beamforming to the input data by first applying
        time-of-flight correction and then summing over the transmits.

        Args:
            data (torch.tensor): The data tensor of shape
                `(n_frames, n_tx, n_ax, n_el, n_ch)`

        Returns:
            tensor: The beamformed and beamsummed data of shape `(n_frames, n_z, n_x)`
        """
        assert isinstance(data, torch.Tensor), (
            "The input data should be a torch tensor, " f"got {type(data)} instead."
        )

        if scan is not None:
            self.scan = scan
        if probe is not None:
            self.probe = probe

        _check_raw_data(data, with_batch_dim=True)

        if data.shape[2] == self.scan.n_el and data.shape[3] == self.scan.n_ax:
            log.warning(
                "Warning: The dimensions of the input data seem to be in the wrong "
                "order. Permuting dimensions from (n_frames, n_tx, n_ax, n_el, n_ch) "
                "to (n_frames, n_tx, n_ax, n_el, n_ch).\n"
                "This functionality is for backwards compatibility and will be "
                "removed in a future update."
            )
            data = torch.permute(data, dims=[0, 1, 3, 2, 4])

        assert data.shape[1] == self.scan.n_tx, (
            "The second dimension of the input data should be the number of "
            f"transmits {self.scan.n_tx}, got {data.shape[1]} instead."
        )
        assert data.shape[2] == self.scan.n_ax, (
            "The third dimension of the input data should be the number of "
            f"axial samples {self.scan.n_ax}, got {data.shape[2]} instead."
        )
        assert data.shape[3] == self.scan.n_el, (
            "The fourth dimension of the input data should be the number of "
            f"elements {self.scan.n_tx}, got {data.shape[3]} instead."
        )
        assert data.shape[4] == self.scan.n_ch, (
            "The fifth dimension of the input data should be the number of "
            f"channels {self.scan.n_ch}, got {data.shape[4]} instead."
        )

        # Perform time-of-flight correction
        data_tof_corrected = self.tof_layer(data)

        data_beamformed = self.das_layer(data_tof_corrected)

        return data_beamformed


@torch_beamformer_registry(name="das", framework="pytorch")
class DAS_layer(torch.nn.Module):
    """Delay-and-sum layer that sums the tof-corrected data over the channels
    and transmits to compute the final beamformed image.
    """

    def __init__(self, rx_apo=None, tx_apo=None):
        super().__init__()
        self.rx_apo = rx_apo if rx_apo else 1
        self.tx_apo = tx_apo if tx_apo else 1

    def forward(self, x):
        """Performs DAS beamforming on tof-corrected input.

        Args:
            x (torch.Tensor): The TOF corrected input of shape
                `(n_frames, n_tx, n_ax, n_el, n_ch)`

        Returns:
            torch.Tensor: The beamformed data of shape `(n_frames, n_z, n_x)`
        """

        # Sum over the channels, i.e. DAS
        x = torch.sum(self.rx_apo * x, dim=-2)

        # Sum over transmits, i.e. Compounding
        x = torch.sum(self.tx_apo * x, dim=1)

        return x


# pylint: disable=too-many-public-methods
class TOF_layer(torch.nn.Module):
    """
    Layer that performs time-of-flight correction by selecting the right data
    samples from the raw data for every pixel in the image grid.
    """

    def __init__(self, probe, scan, batch_size=1, **kwargs):
        """Initializes a beamformer object

        Args:
            probe (Probe): The probe object.
            scan (Scan): The scan object.
            batch_size (int, optional): The number of images per batch.
                Defaults to 1.
        """
        super().__init__(**kwargs)
        #: The `usbmd.probes.Probe` object to beamform with.
        self.probe = probe
        #: The `usbmd.scans.Scan` object to beamform with.
        self.scan = scan
        #: The number of frames to process at once.
        self.batch_size = batch_size

        # TODO: store a device attr instead of relying on the device of the first created tensor
        self.register_buffer("_dummy", torch.tensor(0))

        # Initialize probe parameters
        # This is done via the register_buffer method to ensure that all
        # tensors are properly registered and are affected by methods such as
        # .to(device).
        self.pol_angles = self.polar_angles
        # Note: The registered buffers are set to themselves to properly register them
        # with Pdoc3.

        #: Alias of `polar_angles`. Added for backwards compatibility.
        self.angles = self.polar_angles

    @property
    def device(self):
        """The device the beamformer is on."""
        return self._dummy.device

    @property
    def f_number(self):
        """The f-number used during reconstruction."""
        self.register_buffer("_f_number", torch.tensor(self.scan.f_number))
        return self._f_number

    @property
    def n_tx(self):
        """The number of transmits."""
        self.register_buffer("_n_tx", torch.tensor(self.scan.n_tx))
        return self._n_tx

    @property
    def n_ax(self):
        """The number of axial time samples per element per transmit."""
        self.register_buffer("_n_ax", torch.tensor(self.scan.n_ax))
        return self._n_ax

    @property
    def n_el(self):
        """The number of elements in the probe."""
        self.register_buffer("_n_el", torch.tensor(self.probe.n_el))
        return self._n_el

    @property
    def n_ch(self):
        """The number of RF/IQ channels."""
        self.register_buffer("_n_ch", torch.tensor(self.scan.n_ch))
        return self._n_ch

    @property
    def output_shape(self):
        """The output shape of the beamformed grid."""
        return self.scan.grid.shape[:-1]

    @property
    def n_x(self):
        """The number of pixels in the x-direction."""
        return self.scan.grid.shape[1]

    @property
    def n_z(self):
        """The number of pixels in the z-direction."""
        return self.scan.grid.shape[0]

    @property
    def grid(self):
        """The grid of pixel locations to beamform to. (Does not have to be a grid.)"""
        self.register_buffer("_grid", torch.from_numpy(self.scan.grid).float())
        return self._grid.to(self.device)

    @property
    def polar_angles(self):
        """The polar angles of the plane waves in radians. Of shape `(n_tx,)`."""
        self.register_buffer("_polar_angles", torch.from_numpy(self.scan.polar_angles))
        return self._polar_angles.to(self.device)

    @property
    def initial_times(self):
        """The initial times of the probe in seconds. Of shape `(n_tx,)`. This is the time
        between the first element firing and the first sample being recorded by all
        elements."""
        self.register_buffer(
            "_initial_times", torch.from_numpy(self.scan.initial_times)
        )
        return self._initial_times.to(self.device)

    @property
    def t0_delays(self):
        """The transmit delays in seconds. Of shape `(n_tx, n_el)`. This is the time
        between the first element firing and the first sample being recorded by each
        element."""
        self.register_buffer("_t0_delays", torch.from_numpy(self.scan.t0_delays))
        return self._t0_delays.to(self.device)

    @property
    def sound_speed(self):
        """The speed of sound in m/s."""
        self.register_buffer(
            "_sound_speed", torch.Tensor(np.array(self.scan.sound_speed))
        )
        return self._sound_speed.to(self.device)

    @property
    def probe_geometry(self):
        """The element positions of the probe of shape `(n_el, 3)`."""
        self.register_buffer(
            "_probe_geometry", torch.from_numpy(self.probe.probe_geometry)
        )
        return self._probe_geometry.to(self.device)

    @property
    def sampling_frequency(self):
        """The sampling frequency of the probe in Hz."""
        self.register_buffer("_sampling_frequency", torch.Tensor([self.scan.fs]))
        return self._sampling_frequency.to(self.device)

    @property
    def center_frequency(self):
        """The center frequency of the probe in Hz."""
        self.register_buffer("_center_frequency", torch.Tensor([self.scan.fc]))
        return self._center_frequency.to(self.device)

    @property
    def fdemod(self):
        """The frequency of the demodulation in Hz."""
        self.register_buffer("_fdemod", torch.Tensor([self.scan.fdemod]))
        return self._fdemod.to(self.device)

    @property
    def apply_phase_rotation(self):
        """Whether to apply phase rotation to the data."""
        self.register_buffer(
            "_apply_phase_rotation", torch.Tensor([bool(self.scan.fdemod)])
        )
        return self._apply_phase_rotation.to(self.device)

    @property
    def focus_distances(self):
        """The focus distances of the probe in m. This is the distance from the origin to
        the virtual focus point of each transmit. Of shape `(n_tx,)`."""
        self.register_buffer(
            "_focus_distances", torch.from_numpy(self.scan.focus_distances)
        )
        return self._focus_distances.to(self.device)

    @property
    def azimuth_angles(self):
        """The azimuth angles of the probe in radians. Of shape `(n_tx,)`."""
        self.register_buffer(
            "_azimuth_angles", torch.from_numpy(self.scan.polar_angles)
        )
        return self._azimuth_angles.to(self.device)

    def forward(self, x):
        """Performs time-of-flight correction on the input.

        Args:
            x (torch.Tensor): The raw data of shape (num_transmits, num_elements,
                num_samples, num_rf_iq_channels)

        Returns:
            Tensor: The corrected input
        """
        assert len(x.shape) == 5, (
            "The input data should have 5 dimensions, "
            f"namely num_transmits, num_elements, num_samples, "
            f"num_rf_iq_channels, got {len(x.shape)} dimensions: ."
            f"{x.shape}"
        )

        tof_data = []

        for b in range(self.batch_size):
            tof_corrected_batch = self.tof_correction(
                x[b],
                self.grid,
                self.t0_delays,
                self.sound_speed,
                self.probe_geometry,
                self.initial_times,
                self.sampling_frequency,
                self.f_number,
                self.angles,
                self.focus_distances,
            )

            # Add batch dimension
            # The shape is now (batch, z, x, num_transmits, num_elements)
            tof_corrected_batch = tof_corrected_batch[None]

            tof_data.append(tof_corrected_batch)

        # Stack batches of TOF corrected data in a single tensor
        output = torch.cat(tof_data, dim=0)

        return output

    def compute_output_shape(self):
        """Computes the output shape of the beamformed data.

        Returns:
            tuple: Tuple of size (n_frames, n_z, n_x, n_tx, n_el)
        """
        output_shape = (self.batch_size, self.n_z, self.n_x, self.n_tx, self.n_el)
        return output_shape

    def tof_correction(
        self,
        data,
        grid,
        t0_delays,
        sound_speed,
        probe_geometry,
        initial_times,
        sampling_frequency,
        fnum,
        angles,
        vfocus,
    ):
        """
        Args:
            data (torch.Tensor): Input RF/IQ data of shape `(n_tx, n_ax, n_el, n_ch)`.
            grid (torch.Tensor): Pixel locations x, y, z of shape `(n_z, n_x, 3)`
            t0_delays (torch.Tensor): Times at which the elements fire shifted such
                that the first element fires at t=0 of shape `(n_tx, n_el)`
            c (float): Speed-of-sound.
            probe_geometry (torch.Tensor): Element positions x, y, z of shape
            (num_samples, 3)
            initial_times (torch.Tensor): Time-ofsampling_frequencyet per transmission of shape
                `(n_tx,)`.
            sampling_frequency (float): Sampling frequency.
            n_tx (int): Number of transmissions (e.g. plane waves).
            fnum (int, optional): Focus number. Defaults to 1.
            angles (torch.Tensor): The angles of the plane waves in radians of shape
                `(n_tx,)`
            vfocus (torch.Tensor): The focus distance of shape `(n_tx,)`

        Returns:
            output (torch.Tensor): time-of-flight corrected data
            with shape: `(n_z, n_x, n_tx, n_el)`.

        """

        assert len(data.shape) == 4, (
            "The input data should have 4 dimensions, "
            f"namely num_transmits, num_elements, num_samples, "
            f"num_rf_iq_channels, got {len(data.shape)} dimensions: ."
            f"{data.shape}"
        )
        assert data.shape[0] == self.n_tx, (
            "The first dimension of the input data should be the number of "
            f"transmits {self.n_tx}, got {data.shape[0]} instead."
        )
        assert data.shape[1] == self.n_ax, (
            "The third dimension of the input data should be the number of "
            f"axial samples {self.n_ax}, got {data.shape[1]} instead."
        )
        assert data.shape[2] == self.n_el, (
            "The second dimension of the input data should be the number of "
            f"elements {self.n_el}, got {data.shape[2]} instead."
        )

        # Flatten grid to simplify calculations
        gridshape = grid.shape
        flatgrid = torch.reshape(grid, shape=(-1, 3))

        # Calculate delays
        # --------------------------------------------------------------------
        # txdel: The delay from t=0 to the wavefront reaching the pixel
        # txdel has shape (n_tx, n_pix)
        #
        # rxdel: The delay from the wavefront reaching the pixel to the scattered wave
        # reaching the transducer element.
        # rxdel has shape (n_el, n_pix)
        # --------------------------------------------------------------------
        txdel, rxdel = self.calculate_delays(
            flatgrid,
            t0_delays,
            probe_geometry,
            initial_times,
            sampling_frequency,
            sound_speed,
            vfocus,
            angles,
        )

        mask = apod_mask(flatgrid, probe_geometry, fnum)

        # Apply delays
        bf_tx = []
        for tx in range(self.n_tx):
            # Get the raw data for this transmit
            # data_tx is of shape (num_elements, num_samples, 1 or 2)
            data_tx = data[tx]
            # Take receive delays and add the transmit delays for this transmit
            # The txdel tensor has one fewer dimensions because the transmit
            # delays are the same for all dimensions
            # delays is of shape (n_pix, n_el)
            delays = rxdel + txdel[:, tx, None]

            # Compute the time-of-flight corrected samples for each element
            # from each pixel of shape (n_pix, n_el, n_ch)
            tof_tx = apply_delays(data_tx, delays, clip_min=0, clip_max=self.n_ax - 1)

            # Apply the mask
            tof_tx = tof_tx * mask

            # Phase correction
            if self.apply_phase_rotation:
                tshift = delays[:, :] / self.sampling_frequency
                tdemod = flatgrid[:, None, 2] * 2 / self.sound_speed
                theta = 2 * np.pi * self.fdemod * (tshift - tdemod)
                tof_tx = _complex_rotate(tof_tx, theta)

            # Reshape to reintroduce the x- and z-dimensions
            tof_tx = torch.reshape(
                tof_tx,
                shape=(1, gridshape[0], gridshape[1], self.n_el, tof_tx.shape[-1]),
            )

            bf_tx.append(tof_tx)

        output = torch.cat(bf_tx, dim=0)

        return output

    def calculate_delays(
        self,
        grid,
        t0_delays,
        probe_geometry,
        initial_times,
        sampling_frequency,
        sound_speed,
        focus_distances,
        polar_angles,
    ):
        """
        Calculates the delays in samples to every pixel in the grid.

        The delay consists of two components: The transmit delay and the
        receive delay.

        The transmit delay is the delay between transmission and the
        wavefront reaching the pixel.

        The receive delay is the delay between the
        wavefront reaching a pixel and the reflections returning to a specific
        element.

        Args:
            grid (torch.Tensor): The pixel coordinates to beamform to of shape `(n_pix,
                3)`.
            t0_delays (torch.Tensor): The transmit delays in seconds of shape
                `(n_tx, n_el)`, shifted such that the smallest delay is 0. Defaults to
                None.
            probe_geometry (torch.Tensor): The positions of the transducer elements of shape
                `(n_el, 3)`.
            initial_times (torch.Tensor): The probe transmit time offsets of shape
                `(n_tx,)`.
            sampling_frequency (float): The sampling frequency of the probe in Hz.
            sound_speed (float): The assumed speed of sound in m/s.
            focus_distances (torch.Tensor): The focus distances of shape `(n_tx,)`.
                If the focus distance is set to infinity, the beamformer will
                assume plane wave transmission.
            polar_angles (torch.Tensor): The polar angles of the plane waves in radians
                of shape `(n_tx,)`.

        Returns:
            torch.Tensor, torch.Tensor: transmit_delays, receive_delays

            The tensor transmit delays to every pixel has shape
            `(n_pix, n_tx)`

            the tensor of receive delays from every pixel back to the
            transducer element has shape of shape `(n_pix, n_tx)`
        """

        # Initialize delay variables
        tx_distances = []
        rx_distances = []

        # Compute transmit distances
        for tx in range(self.n_tx):
            if torch.isinf(focus_distances[tx]):
                distance_to_pixels = distance_Tx_planewave(grid, polar_angles[tx])
            else:
                distance_to_pixels = distance_Tx_generic(
                    grid, t0_delays[tx], probe_geometry, sound_speed
                )

            tx_distances.append(distance_to_pixels[..., None])

        # Compute receive distances
        for el in range(self.n_el):
            distances = distance_Rx(grid, probe_geometry[el])
            # Add transducer element dimension
            distances = distances[..., None]
            rx_distances.append(distances)

        # Concatenate all values into one long tensor
        # The shape is now (n_pix, n_tx)
        tx_distances = torch.cat(tx_distances, dim=1)
        # The shape is now (n_pix, n_el)
        rx_distances = torch.cat(rx_distances, dim=1)

        # Compute the delays [in samples] from the distances
        # The units here are ([m]/[m/s]-[s])*[1/s] resulting in a unitless quantity
        # TODO: Add pulse width to transmit delays
        tx_delays = (
            tx_distances / sound_speed - initial_times[None]
        ) * sampling_frequency
        rx_delays = (rx_distances / sound_speed) * sampling_frequency

        assert tuple(tx_delays.shape) == (self.n_x * self.n_z, self.n_tx), (
            "The output shape of tx_delays is incorrect!"
            f"Expected {(self.n_x * self.n_z, self.n_tx)}, got {tx_delays.shape}"
        )
        assert tuple(rx_delays.shape) == (self.n_x * self.n_z, self.n_el), (
            "The output shape of rx_delays is incorrect!"
            f"Expected {(self.n_x * self.n_z, self.n_el)}, got {rx_delays.shape}"
        )

        return tx_delays, rx_delays


def apply_delays(data, delays, clip_min: int = -1, clip_max: int = -1):
    """
    Applies time delays for a single transmit using linear interpolation.

    Most delays in d will not be by an integer number of samples, which means
    we have no measurement for that time instant. This function solves this by
    finding the sample before and after and interpolating the data to the
    desired delays in d using linear interpolation.

    Args:
        data (torch.Tensor): The RF or IQ data of shape `(n_ax, n_el, n_ch)`. This is
            the data we are drawing samples from to for each element for each pixel.
        delays (torch.Tensor): The delays in samples of shape `(n_pix, n_el)`. Contains
            one delay value for every pixel in the image for every transducer element.
        clip_min (int, optional): The minimum delay value to use. If set to -1 no
            clipping is applied. Defaults to -1.
        clip_max (int, optional): The maximum delay value to use. If set to -1 no
            clipping is applied. Defaults to -1.

    Returns:
        torch.Tensor: The samples received by each transducer element corresponding to the
            reflections of each pixel in the image of shape `(n_el, n_pix, n_ch)`.
    """

    # Add a dummy channel dimension to the delays tensor to ensure it has the
    # same number of dimensions as the data. The new shape is (1, n_el, n_pix)
    delays = delays[..., None]

    # Get the integer values above and below the exact delay values
    # Floor to get the integers below
    # (num_elements, num_pixels, 1)
    d0 = torch.floor(delays)

    # Cast to integer to be able to use as indices
    d0 = d0.type(torch.int64)
    # Add 1 to find the integers above the exact delay values
    d1 = d0 + 1

    # Apply clipping of delays clipping to ensure correct behavior on cpu
    if clip_min != -1 and clip_max != -1:
        d0 = torch.clip(d0, clip_min, clip_max)
        d1 = torch.clip(d1, clip_min, clip_max)

    if data.shape[-1] == 2:
        d0 = d0.repeat((1, 1, 2))
        d1 = d1.repeat((1, 1, 2))

    # Gather pixel values
    # Here we extract for each transducer element the sample containing the
    # reflection from each pixel. These are of shape `(n_el, n_pix, n_ch)`.
    data0 = torch.gather(data, 0, d0)
    data1 = torch.gather(data, 0, d1)

    # Compute interpolated pixel value
    d0 = d0.float()  # Cast to float
    d1 = d1.float()  # Cast to float
    reflection_samples = (d1 - delays) * data0 + (delays - d0) * data1

    return reflection_samples


def _complex_rotate(iq, theta):
    """
    Performs a simple phase rotation of I and Q component by complex angle
    theta.

    Args:
        iq (torch.Tensor): The iq data of shape `(..., 2)`.
        theta (float): The angle to rotate by.

    Returns:
        Tensor: The rotated tensor of shape `(..., 2)`.
    """
    assert iq.shape[-1] == 2, (
        "The last dimension of the input tensor should be 2, "
        f"got {iq.shape[-1]} dimensions."
    )
    # Select i and q channels
    i = iq[..., 0]
    q = iq[..., 1]

    # Compute rotated components
    ir = i * torch.cos(theta) - q * torch.sin(theta)
    qr = q * torch.cos(theta) + i * torch.sin(theta)

    # Reintroduce channel dimension
    ir = ir[..., None]
    qr = qr[..., None]

    return torch.cat([ir, qr], dim=-1)


def distance_Rx(grid, probe_geometry):
    """
    Computes distance to user-defined pixels from elements
    Expects all inputs to be numpy arrays specified in SI units.

    Args:
        grid (torch.Tensor): Pixel positions in x,y,z of shape `(n_pix, 3)`.
        probe_geometry (torch.Tensor): Element positions in x,y,z of shape `(n_el, 3)`.

    Returns:
        dist (torch.Tensor): Distance from each pixel to each element of shape
            `(n_pix, n_el)`.
    """
    # Get norm of distance vector between elements and pixels via broadcasting
    dist = torch.norm(grid - probe_geometry[None, ...].float(), dim=-1)
    return dist


def distance_Tx_planewave(grid, angle):
    """
    Computes distance to user-defined pixels for plane wave transmits.

    Args:
        grid (torch.Tensor): Flattened tensor of pixel positions in x,y,z of shape
           `(n_pix, 3)`.
        angle (torch.Tensor, float): Plane wave angle (radians).

    Returns:
        Tensor: Distance from each pixel to each element in meters of shape
            `(n_pix,)`.
    """
    # Use broadcasting to simplify computations
    x = grid[..., 0]
    z = grid[..., 2]
    # For each element, compute distance to pixels
    angle = angle.float()
    dist = x * torch.sin(angle) + z * torch.cos(angle)

    return dist


def distance_Tx_generic(grid, t0_delays, probe_geometry, sound_speed=1540):
    """
    Computes distance to user-defined pixels for generic transmits based on
    the t0_delays.

    Args:
        grid (torch.Tensor): Flattened tensor of pixel positions in x,y,z of shape
            `(n_pix, 3)`
        t0_delays (torch.Tensor): The transmit delays in seconds of shape `(n_el,)`,
            shifted such that the smallest delay is 0. Defaults to None.
        probe_geometry (torch.Tensor): The positions of the transducer elements of shape
            `(n_el, 3)`.
        sound_speed (float): The speed of sound in m/s. Defaults to 1540.

    Returns:
        Tensor: Distance from each pixel to each element in meters
        of shape `(n_pix,)`
    """
    # Get the individual x, y, and z components of the pixel coordinates
    x = grid[:, 0]
    y = grid[:, 1]
    z = grid[:, 2]

    # Reshape x, y, and z to shape (n_pix, 1)
    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    # Get the individual x, y, and z coordinates of the elements and add a
    # dummy dimension at the beginning to shape (1, n_el).
    ele_x = probe_geometry[None, :, 0]
    ele_y = probe_geometry[None, :, 1]
    ele_z = probe_geometry[None, :, 2]

    # Compute the differences dx, dy, and dz of shape (n_pix, n_el)
    dx = x - ele_x
    dy = y - ele_y
    dz = z - ele_z

    # Compute the distance between the elements and the pixels of shape
    # (n_pix, n_el)
    dist = t0_delays[None] * sound_speed + torch.sqrt(dx**2 + dy**2 + dz**2)

    # Compute the effective distance of the pixels to the wavefront by
    # computing the smallest distance over all the elements. This is the wave
    # front that reaches the pixel first and thus is the overal wavefront
    # distance.
    dist = torch.min(dist, dim=1)[0]

    return dist


def apod_mask(grid, probe_geometry, f_number):
    """
    Computes a binary mask to disregard pixels outside of the vision cone of a
    transducer element. Transducer elements can only accurately measure
    signals within some range of incidence angles. Waves coming in from the
    side do not register correctly leading to a worse image.

    Args:
        grid (torch.Tensor): The flattened image grid `(n_pix, 3)`.
        probe_geometry (torch.Tensor): The transducer element positions of shape
            `(n_el, 3)`.
        f_number (int): The receive f-number. Set to zero to not use masking and
            return 1. (The f-number is the  ratio between distance from the transducer
            and the size of the aperture below which transducer elements contribute to
            the signal for a pixel.).

    Returns:
        Tensor: Mask of shape `(n_pix, n_el, 1)`
    """
    # If the f-number is set to 0, return 1
    if f_number == 0:
        mask = torch.ones((1))
        return mask

    n_pix = grid.shape[0]
    n_el = probe_geometry.shape[0]

    # Get the depth of every pixel
    z_pixel = grid[:, 2]
    # Get the lateral location of each pixel
    x_pixel = grid[:, 0]
    # Get the lateral location of each element
    x_element = probe_geometry[:, 0]

    # Compute the aperture size for every pixel
    # The f-number is by definition f=z/aperture
    aperture = z_pixel / f_number

    device = aperture.device

    # Use matrix multiplication to expand aperture tensor, x_pixel tensor, and
    # x_element tensor to shape (n_pix, n_el)
    aperture = aperture[..., None] @ torch.ones((1, n_el), device=device)

    expanded_x_pixel = x_pixel[..., None] @ torch.ones((1, n_el), device=device)

    expanded_x_element = torch.ones((n_pix, 1), device=device) @ x_element[None]

    # Compute the lateral distance between elements and pixels
    # Of shape (n_pix, n_el)
    distance = torch.abs(expanded_x_pixel - expanded_x_element)

    # Compute binary mask for which the lateral pixel distance is less than
    # half
    # the aperture i.e. where the pixel lies within the vision cone of the
    # element
    mask = distance <= aperture / 2
    mask = mask.float()

    # Add dummy dimension for RF/IQ channel channel
    mask = mask[..., None]

    return mask
