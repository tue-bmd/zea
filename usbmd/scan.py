"""Class structures containing parameters defining an ultrasound scan and the
beamforming grid.

- **Author(s)**     : Vincent van de Schaft
- **Date**          : Wed Feb 15 2024
"""

import matplotlib.pyplot as plt
import numpy as np

from usbmd.utils import log
from usbmd.utils.pixelgrid import check_for_aliasing, get_grid
from usbmd.utils.utils import deprecated

SCAN_PARAM_TYPES = {
    "n_ax": int,
    "n_el": int,
    "n_tx": int,
    "n_ch": int,
    "n_frames": int,
    "PRF": float,  # deprecated
    "sampling_frequency": float,
    "center_frequency": float,
    "sound_speed": float,
    "bandwidth_percent": float,
    "t0_delays": np.array,
    "initial_times": np.array,
    "tx_apodizations": np.array,
    "polar_angles": np.array,
    "azimuth_angles": np.array,
    "angles": np.array,  # deprecated
    "focus_distances": np.array,
    "time_to_next_transmit": np.array,
    "probe_geometry": np.array,
    "origin": np.array,
}


def cast_scan_parameters(scan_parameters: dict) -> dict:
    """Casts scan parameters (from hdf5 file) to the correct type.

    Args:
        scan_parameters (dict): The scan parameters.

    Raises:
        ValueError: If an unknown scan parameter is encountered.
    Returns:
        dict: The scan parameters with the correct types.
    """
    # Cast all parameters to the correct type
    for key, value in scan_parameters.items():
        if key in SCAN_PARAM_TYPES:
            scan_parameters[key] = SCAN_PARAM_TYPES[key](value)
        else:
            raise ValueError(f"Unknown scan parameter: {key}.")

    return scan_parameters


class Scan:
    """Scan base class."""

    def __init__(
        self,
        n_tx: int,
        n_ax: int,
        n_el: int,
        center_frequency: float,
        sampling_frequency: float,
        demodulation_frequency: float = None,
        xlims=None,
        ylims=None,
        zlims=None,
        bandwidth_percent: int = 200,
        sound_speed: float = 1540,
        n_ch: int = None,
        Nx: int = None,
        Nz: int = None,
        pixels_per_wvln: int = 3,
        downsample: int = 1,
        polar_angles: np.ndarray = None,
        azimuth_angles: np.ndarray = None,
        t0_delays: np.ndarray = None,
        tx_apodizations: np.ndarray = None,
        focus_distances: np.ndarray = None,
        initial_times: np.ndarray = None,
        selected_transmits: list = None,
        probe_geometry: np.ndarray = None,
        time_to_next_transmit: np.ndarray = None,
    ):
        """Initializes a Scan object representing the number and type of
        transmits, and the target pixels to beamform to.

        The Scan object generates a pixel grid based on Nx, Nz, xlims, and zlims when it
        is initialized. When any of these parameters are changed the grid is recomputed
        automatically.

        Args:
            n_tx (int): The number of transmits to produce a single frame.
            n_ax (int, optional): The number of samples per in a receive
                recording per channel. Defaults to None.
            n_el (int, optional): The number of elements in the array.
            center_frequency (float): The modulation carrier frequency.
            sampling_frequency (float): The sampling rate to sample rf- or
                iq-signals with.
            demodulation_frequency (float): The demodulation frequency.
                Usually set to 0.0 if rf data and to transmit frequency if iq data.
                For iq data it can vary depending on the approach used to defined
                the ultrasound echo center frequency.
            xlims (tuple, optional): The x-limits in the beamforming grid.
                Defaults to (probe_geometry[0, 0], probe_geometry[-1, 0]).
            ylims (tuple, optional): The y-limits in the beamforming grid.
                Defaults to (0, 0).
            zlims (tuple, optional): The z-limits in the beamforming grid.
                Defaults to (0, n_ax * sound_speed / fs / 2).
            bandwidth_percent: Receive bandwidth of RF signal in % of center
                frequency. Not necessarily the same as probe bandwidth. Defaults to 200.
            sound_speed (float, optional): The speed of sound in m/s. Defaults to 1540.
            n_ch (int): The number of channels. This will determine the modulation type.
                Can be either RF (when `n_ch = 1`) or IQ (when `n_ch=2`).
            Nx (int, optional): The number of pixels in the lateral direction
                in the beamforming grid. Defaults to None.
            Nz (int, optional): The number of pixels in the axial direction in
                the beamforming grid. Defaults to None.
            pixels_per_wvln (int, optional): The number of pixels per wavelength
                to use in the beamforming grid. Only used when Nx and Nz are not
                defined. Defaults to 3.
            downsample (int, optional): Decimation factor applied after downconverting
                data to baseband (RF to IQ). Defaults to 1.
            polar_angles (np.ndarray, optional): The polar angles of the
                transmits in radians of shape (n_tx,). These are the angles usually used
                in 2D imaging. Defaults to None.
            azimuth_angles (np.ndarray, optional): The azimuth angles of the
                transmits in radians of shape (n_tx,). These are the angles usually used
                in 3D imaging. Defaults to None.
            t0_delays (np.ndarray, optional): The transmit delays in seconds of
                shape (n_tx, n_el), shifted such that the smallest delay is 0. Defaults
                to None.
            tx_apodizations (np.ndarray, float, optional): The transmit
                apodizations of shape (n_tx, n_el) or a single float to use for all
                apodizations. Defaults to None.
            focus_distances (np.ndarray, optional): The focus distances of the
                transmits in meters of shape (n_tx,). Defaults to None.
            initial_times (np.ndarray, optional): The initial times of the
                transmits in seconds of shape (n_tx,). Defaults to None.
            selected_transmits (int, list, optional): Used to select a subset of the
                transmits to use for beamforming. If set to an integer, then that number
                of transmits is selected as homogeneously as possible. If set to a list
                of integers, then the transmits with those indices are selected. If set
                to None, then all transmits are used. Defaults to None.
            probe_geometry (np.ndarray, optional): (n_el, 3) array with element positions
                in meters. Necessary for automatic xlim calculation if not set. Defaults to None.
            time_to_next_transmit (np.ndarray, float, optional): The time between subsequent
                transmit events of shape (n_tx*n_frames,). Defaults to None.

        Raises:
            NotImplementedError: Initializing from probe not yet implemented.
        """

        # Attributes concerning channel data : The number of transmissions in a frame
        self._n_tx = int(n_tx)
        #: The number of samples per channel per acquisition
        self._n_ax = n_ax
        #: The number of elements in the array
        self._n_el = int(n_el)
        #: The modulation carrier frequency [Hz]
        self.fc = float(center_frequency)
        #: The sampling rate [Hz]
        self.fs = float(sampling_frequency)
        #: The percent bandwidth []
        self.bandwidth_percent = float(bandwidth_percent)
        #: The speed of sound [m/s]
        self.sound_speed = float(sound_speed)
        #: The number of rf/iq channels (1 for rf, 2 for iq)
        self._n_ch = n_ch
        #: The demodulation frequency [Hz]
        self._fdemod = demodulation_frequency
        #: The wavelength of the modulation carrier [m]
        self.wvln = self.sound_speed / self.fc
        #: The number of pixels per wavelength in the beamforming grid
        self.pixels_per_wavelength = pixels_per_wvln
        #: The decimation factor applied after downconverting data to baseband (RF to IQ)
        self.downsample = downsample
        #: The probe geometry of shape (n_el, 3)
        self.probe_geometry = probe_geometry
        #: The time between subsequent transmit events of shape (n_tx*n_frames,)
        self.time_to_next_transmit = time_to_next_transmit

        # Beamforming grid related attributes
        # ---------------------------------------------------------------------
        #: The x-limits of the beamforming grid [m]
        self._xlims = xlims
        #: The y-limits of the beamforming grid [m]
        self._ylims = ylims
        #: The z-limits of the beamforming grid [m]
        self._zlims = zlims

        #: The number of pixels in the lateral direction in the beamforming grid
        self._Nx = int(Nx) if Nx is not None else None
        #: The number of pixels in the axial direction in the beamforming grid
        self._Nz = int(Nz) if Nz is not None else None

        # Compute the zlims from the other values if not supplied
        if zlims:
            self.zlims = zlims
        else:
            # Compute the depth of the scan from the number of axial samples
            self.zlims = [0, self.sound_speed * self.n_ax / self.fs / 2]
        if ylims:
            self.ylims = ylims
        else:
            self.ylims = [0, 0]
        if xlims:
            self.xlims = xlims
        else:
            # Set the scan limits to the limits of the probe and
            if self.probe_geometry is None:
                raise ValueError(
                    "Please provide probe_geometry or xlims, currently neither is set."
                )
            self.xlims = self.probe_geometry[0, 0], self.probe_geometry[-1, 0]

        self.z_axis = np.linspace(*self.zlims, self.n_ax)

        #: The beamforming grid of shape (Nx, Nz, 3)
        self._grid = self.grid

        if initial_times is None:
            log.warning("No initial times provided. Assuming all zeros.")
            initial_times = np.zeros(self._n_tx)

        if t0_delays is None:
            log.warning(
                "No t0_delays provided. Assuming all zeros and 128 element probe."
            )
            t0_delays = np.zeros((self._n_tx, self._n_el))
        else:
            assert t0_delays.shape == (self._n_tx, self._n_el), (
                f"t0_delays must have shape (n_tx, n_el). "
                f"Got shape {t0_delays.shape}. Please set t0_delays either to None in which "
                f"case all zeros are assumed, or set the n_tx and n_el params to match the "
                "t0_delays shape."
            )

        if tx_apodizations is None:
            log.warning(
                "No tx_apodizations provided. Assuming all ones and "
                "128 element probe."
            )
            tx_apodizations = np.ones((self._n_tx, self._n_el))

        if polar_angles is None:
            log.warning("No polar_angles provided. Assuming all zeros.")
            polar_angles = np.zeros(self._n_tx)

        if azimuth_angles is None:
            log.warning("No azimuth_angles provided. Assuming all zeros.")
            azimuth_angles = np.zeros(self._n_tx)

        if focus_distances is None:
            log.warning("No focus_distances provided. Assuming all zeros.")
            focus_distances = np.zeros(self._n_tx)

        self._t0_delays = t0_delays
        self._tx_apodizations = tx_apodizations
        self._polar_angles = polar_angles
        self._angles = polar_angles  # deprecated
        self._azimuth_angles = azimuth_angles
        self._focus_distances = focus_distances
        self._initial_times = initial_times

        self.selected_transmits = selected_transmits

    def _select_transmits(self, selected_transmits):
        """Interprets the selected transmits argument and returns an array of transmit
        indices.

        Args:
            selected_transmits (int, list, None): The selected transmits input. If set
            to an integer, then that number of transmits is selected as homogeneously as
            possible. If set to a list of integers, then the transmits with those
            indices are selected. If set to None, then all transmits are used. Defaults
            to None.

        Returns:
            list: The selected transmits as a list of indices
        """
        if selected_transmits is None:
            return list(range(self._n_tx))

        # Convert numpy array to list or single integer
        if isinstance(selected_transmits, np.ndarray):
            if len(np.shape(selected_transmits)) == 0:
                selected_transmits = int(selected_transmits)
            elif len(np.shape(selected_transmits)) == 1:
                selected_transmits = selected_transmits.tolist()
            else:
                raise ValueError(
                    f"Invalid shape for selected_transmits: {np.shape(selected_transmits)}."
                )

        # 'all', 'center'
        if isinstance(selected_transmits, str):
            if selected_transmits == "all":
                return list(range(self._n_tx))
            elif selected_transmits == "center":
                return [self._n_tx // 2]
            else:
                raise ValueError(
                    f"Invalid value for selected_transmits: {selected_transmits}."
                )

        if isinstance(selected_transmits, int):
            # Do an error check if the number of selected transmits is not too large
            assert selected_transmits <= self._n_tx, (
                f"Number of selected transmits ({selected_transmits}) "
                f"exceeds number of transmits in scan ({self._n_tx})."
            )

            # If the number of selected transmits is 1, then pick the middle transmit
            if selected_transmits == 1:
                tx_indices = [self._n_tx // 2]
            else:
                # Compute selected_transmits evenly spaced indices for reduced angles
                tx_indices = np.linspace(0, self._n_tx - 1, selected_transmits)

                # Round the computed angles to integers and turn into list
                tx_indices = list(np.rint(tx_indices).astype("int"))

            return list(tx_indices)

        if isinstance(selected_transmits, list):
            assert all(
                isinstance(n, int) for n in selected_transmits
            ), "selected_transmits must be a list of integers."
            # Check if the selected transmits are not too large
            assert all(n < self._n_tx for n in selected_transmits), (
                f"Selected transmits {selected_transmits} exceed the number of "
                f"transmits in the scan ({self._n_tx})."
            )

            return selected_transmits

    @property
    def selected_transmits(self):
        """Used to select a subset of the transmits to use for beamforming. If set to an
        integer, then that number of transmits is selected as homogeneously as possible.
        If set to a list of integers, then the transmits with those indices are
        selected. If set to None, then all transmits are used. Defaults to None."""
        return self._selected_transmits

    @selected_transmits.setter
    def selected_transmits(self, value):
        self._selected_transmits = self._select_transmits(value)
        check_for_aliasing(self)

    @property
    def n_tx(self):
        """The number of transmits to produce a single frame."""
        return len(self.selected_transmits)

    @property
    def n_ax(self):
        """The number of samples in a receive recording per channel."""
        return int(np.ceil(self._n_ax / self.downsample))

    @property
    def n_el(self):
        """The number of elements in the array."""
        return self._n_el

    @property
    def n_ch(self):
        """The number of channels."""
        return self._n_ch

    @n_ch.setter
    def n_ch(self, value):
        self._n_ch = value
        self._fdemod = None  # Reset fdemod
        log.warning(
            f"Resetting fdemod to {self.fdemod} because n_ch was changed to {value}."
        )

    @property
    def fdemod(self):
        """The demodulation frequency."""
        if self._fdemod is not None:
            return self._fdemod

        if self.n_ch is None:
            raise ValueError(
                "Please set scan.n_ch or scan.fdemod. Currently neither is set.\n"
                "\tif n_ch is set to 1 (RF), then fdemod is set to 0.0.\n"
                "\tif n_ch is set to 2 (IQ), then fdemod is set to fc.\n"
                "\tfdemod can be set to any other value manually."
            )

        return self.fc if self.n_ch == 2 else 0.0

    @fdemod.setter
    def fdemod(self, value):
        self._fdemod = value

    @property
    def t0_delays(self):
        """The transmit delays in seconds of shape (n_tx, n_el), shifted such that the
        smallest delay is 0. For instance for a straight planewave transmit all delays
        are zero."""
        return self._t0_delays[self.selected_transmits]

    @property
    def tx_apodizations(self):
        """The transmit apodizations of shape (n_tx, n_el) or a single float to use for
        all apodizations. These values indicate both windowing (apodization) over the
        aperture and the subaperture that is used during transmit."""
        return self._tx_apodizations[self.selected_transmits]

    @property
    def polar_angles(self):
        """The polar angles of the transmits in radians of shape (n_tx,). These are the
        angles usually used in 2D imaging."""
        return self._polar_angles[self.selected_transmits]

    @deprecated("Scan.polar_angles")
    @property
    def angles(self):
        """Identical to `Scan.polar_angles`. This attribute is added for backward
        compatibility."""
        return self.polar_angles

    @property
    def azimuth_angles(self):
        """The azimuth angles of the transmits in radians of shape (n_tx,). These are
        the angles usually used in 3D imaging."""
        return self._azimuth_angles[self.selected_transmits]

    @property
    def focus_distances(self):
        """The focus distances of the transmits in meters of shape (n_tx,). These are
        the distances of the virtual focus points from the origin. For a planewave
        these should be set to Inf."""
        return self._focus_distances[self.selected_transmits]

    @property
    def initial_times(self):
        """The initial times of the transmits in seconds of shape (n_tx,). These are the
        time intervals between the first element firing and the first sample in the
        receive recording."""
        return self._initial_times[self.selected_transmits]

    @property
    def Nx(self):
        """The number of pixels in the lateral direction in the beamforming
        grid."""
        return self._Nx

    @Nx.setter
    def Nx(self, value):
        self._Nx = value
        self._grid = None

    @property
    def Nz(self):
        """The number of pixels in the axial direction in the beamforming grid."""
        return self._Nz

    @Nz.setter
    def Nz(self, value):
        self._Nz = value
        self._grid = None

    @property
    def xlims(self):
        """The x-limits of the beamforming grid [m]."""
        return self._xlims

    @xlims.setter
    def xlims(self, value):
        self._xlims = value
        self._grid = None

    @property
    def ylims(self):
        """The y-limits of the beamforming grid [m]."""
        return self._ylims

    @ylims.setter
    def ylims(self, value):
        self._ylims = value
        self._grid = None

    @property
    def zlims(self):
        """The z-limits of the beamforming grid [m]."""
        return self._zlims

    @zlims.setter
    def zlims(self, value):
        self._zlims = value
        self._grid = None

    @property
    def grid(self):
        """The beamforming grid of shape (Nx, Nz, 3)."""
        if self._grid is None:
            self._grid = get_grid(self)
            self._Nz, self._Nx, _ = self._grid.shape
        return self._grid


class FocussedScan(Scan):
    """
    Class representing a focussed beam scan where every transmit has a beam origin,
    angle, and focus defined.
    """


class PlaneWaveScan(Scan):
    """
    Class representing a plane wave scan. Supplied with an array of angles instead of
    with focus distances or t0_delays.
    """

    def __init__(
        self,
        probe_geometry: np.ndarray,
        angles=None,
        n_tx=75,
        n_el=128,
        n_ch=1,
        xlims=(-0.01, 0.01),
        ylims=(0, 0),
        zlims=(0, 0.04),
        center_frequency=7e6,
        sampling_frequency=28e6,
        demodulation_frequency=0.0,
        sound_speed=1540,
        n_ax=3328,
        Nx=None,
        Nz=None,
        pixels_per_wvln=3,
        polar_angles=None,
        azimuth_angles=None,
        tx_apodizations=None,
        downsample=1,
        initial_times=None,
        selected_transmits=None,
        time_to_next_transmit: np.ndarray = None,
    ):
        """
        Initializes a PlaneWaveScan object.

        Args:
            probe_geometry (np.ndarray): The positions of the elements in the array of
                shape (n_el, 3).
            angles (list, optional): The angles of the planewaves. Defaults to
                None.
            n_tx (int): The number of transmits to produce a single frame. xlims (tuple,
                optional): The x-limits in the beamforming grid.
                Defaults to (-0.01, 0.01).
            n_el (int, optional): The number of elements in the array. Defaults to 128.
            n_ch (int): The number of channels. Defaults to 1.
            center_frequency (float): The modulation carrier frequency.
            sampling_frequency (float): The sampling rate to sample rf- or
                iq-signals with.
            demodulation_frequency (float): The demodulation frequency. Defaults to 0.
            ylims (tuple, optional): The y-limits in the beamforming grid.
                Defaults to (0, 0).
            zlims (tuple, optional): The z-limits in the beamforming grid.
                Defaults to (0,0.04).
            sound_speed (float, optional): The speed of sound in m/s. Defaults to 1540.
            n_ax (int, optional): The number of samples per in a receive
                recording per channel. Defaults to None.
            Nx (int, optional): The number of pixels in the lateral direction
                in the beamforming grid. Defaults to None.
            Nz (int, optional): The number of pixels in the axial direction in
                the beamforming grid. Defaults to None.
            polar_angles (np.ndarray, optional): The polar angles of the
                transmits in radians of shape (n_tx,). These are the angles usually used
                in 2D imaging. Defaults to None.
            azimuth_angles (np.ndarray, optional): The azimuth angles of the
                transmits in radians of shape (n_tx,). These are the angles usually used
                in 3D imaging. Defaults to None.
            t0_delays (np.ndarray, optional): The transmit delays in seconds of
                shape (n_tx, n_el), shifted such that the smallest delay is 0. Defaults
                to None.
            tx_apodizations (np.ndarray, float, optional): The transmit
                apodizations of shape (n_tx, n_el) or a single float to use for all
                apodizations. Defaults to None.
            pixels_per_wvln (int, optional): The number of pixels per wavelength
                to use in the beamforming grid. Only used when Nx and Nz are not
                defined. Defaults to 3.
            focus_distances (np.ndarray, optional): The focus distances of the
                transmits in meters of shape (n_tx,). Defaults to None.
            downsample (int, optional): Decimation factor applied after downconverting
                data to baseband (RF to IQ). Defaults to 1.
            initial_times (np.ndarray, optional): The initial times of the
                transmits in seconds of shape (n_tx,). Defaults to None.
            selected_transmits (int, list, optional): Used to select a subset of the
                transmits to use for beamforming. If set to an integer, then that number
                of transmits is selected as homogeneously as possible. If set to a list
                of integers, then the transmits with those indices are selected. If set
                to None, then all transmits are used. Defaults to None.

        Raises:
            ValueError: If selected_transmits has an invalid value.
        """

        assert (
            angles is not None or polar_angles is not None
        ), "Please provide angles at which plane wave dataset was recorded"
        if angles is not None:
            self._angles = angles
            polar_angles = angles
        else:
            angles = polar_angles
            polar_angles = polar_angles

        if azimuth_angles is None:
            # We assume azimuth angles are zero for plane wave scans if not provided
            azimuth_angles = np.zeros(len(polar_angles))

        if not n_tx:
            n_tx = len(polar_angles)
        else:
            assert n_tx == len(polar_angles), (
                "Number of transmits does not match the number of polar angles. "
                "Please provide the correct number of transmits, or let the Scan object set it."
            )

        t0_delays = compute_t0_delays_planewave(
            probe_geometry, polar_angles, azimuth_angles, sound_speed
        )

        # Pass all arguments to the Scan base class
        super().__init__(
            n_tx=n_tx,
            n_el=n_el,
            n_ch=n_ch,
            xlims=xlims,
            ylims=ylims,
            zlims=zlims,
            center_frequency=center_frequency,
            sampling_frequency=sampling_frequency,
            demodulation_frequency=demodulation_frequency,
            sound_speed=sound_speed,
            n_ax=n_ax,
            Nx=Nx,
            Nz=Nz,
            pixels_per_wvln=pixels_per_wvln,
            polar_angles=polar_angles,
            azimuth_angles=azimuth_angles,
            t0_delays=t0_delays,
            tx_apodizations=tx_apodizations,
            downsample=downsample,
            initial_times=initial_times,
            selected_transmits=selected_transmits,
            focus_distances=np.inf * np.ones(n_tx),
            probe_geometry=probe_geometry,
            time_to_next_transmit=time_to_next_transmit,
        )


class DivergingWaveScan(Scan):
    """Class representing a scan with diverging wave transmits."""

    def __init__(
        self,
        n_tx=75,
        n_el=128,
        n_ch=1,
        xlims=(-0.01, 0.01),
        ylims=(0, 0),
        zlims=(0, 0.04),
        center_frequency=7e6,
        sampling_frequency=28e6,
        demodulation_frequency=0.0,
        sound_speed=1540,
        n_ax=256,
        Nx=128,
        Nz=128,
        downsample=1,
        focus=None,
    ):
        super().__init__(
            n_tx=n_tx,
            n_el=n_el,
            n_ch=n_ch,
            xlims=xlims,
            ylims=ylims,
            zlims=zlims,
            center_frequency=center_frequency,
            sampling_frequency=sampling_frequency,
            demodulation_frequency=demodulation_frequency,
            sound_speed=sound_speed,
            n_ax=n_ax,
            Nx=Nx,
            Nz=Nz,
            downsample=downsample,
            initial_times=None,
        )

        self.focus = focus
        raise NotImplementedError("CircularWaveScan has not been implemented.")


def compute_t0_delays_planewave(
    probe_geometry, polar_angles, azimuth_angles=0, sound_speed=1540
):
    """Computes the transmit delays for a planewave, shifted such that the
    first element fires at t=0.

    Args:
        probe_geometry (np.ndarray): The positions of the elements in the array of
            shape (n_el, 3).
        polar_angles (np.ndarray): The polar angles of the planewave in radians of shape (n_tx,).
        azimuth_angles (np.ndarray, optional): The azimuth angles of the planewave
            in radians of shape (n_tx,). Defaults to 0.
        sound_speed (float, optional): The speed of sound. Defaults to 1540.

    Returns:
        np.ndarray: The transmit delays for each element of shape (n_tx, n_el).
    """
    assert (
        probe_geometry is not None
    ), "Probe geometry must be provided to compute t0_delays."

    # Convert single angles to arrays for broadcasting
    polar_angles = np.atleast_1d(polar_angles)
    azimuth_angles = np.atleast_1d(azimuth_angles)

    # Compute v for all angles
    v = np.stack(
        [
            np.sin(polar_angles) * np.cos(azimuth_angles),
            np.sin(polar_angles) * np.sin(azimuth_angles),
            np.cos(polar_angles),
        ],
        axis=-1,
    )

    # Compute the projection of the element positions onto the wave vectors
    projection = np.sum(probe_geometry[:, None, :] * v, axis=-1).T

    # Convert from distance to time to compute the transmit delays.
    t0_delays_not_zero_aligned = projection / sound_speed

    # The smallest (possibly negative) time corresponds to the moment when
    # the first element fires.
    t_first_fire = np.min(t0_delays_not_zero_aligned, axis=1)

    # The transmit delays are the projection minus the offset. This ensures
    # that the first element fires at t=0.
    t0_delays = t0_delays_not_zero_aligned - t_first_fire[:, None]
    return t0_delays


def compute_t0_delays_focused(
    origin,
    focus_distance,
    probe_geometry,
    polar_angles,
    azimuth_angles=0,
    sound_speed=1540,
):
    """Computes the transmit delays for a focused transmit, shifted such that
    the first element fires at t=0.

    Args:
        origin (np.ndarray): The origin of the focused transmit of shape (3,).
        focus_distance (float): The distance to the focus.
        probe_geometry (np.ndarray): The positions of the elements in the array of
            shape (element, 3).
        polar_angles (np.ndarray): The polar angles of the planewave in radians.
        azimuth_angles (np.ndarray, optional): The azimuth angles of the planewave
            in radians. Defaults to 0.
        sound_speed (float, optional): The speed of sound. Defaults to 1540.

    Returns:
        np.ndarray: The transmit delays for each element of shape (n_tx, element).
    """
    # Convert single angles to arrays for broadcasting
    polar_angles = np.atleast_1d(polar_angles)
    azimuth_angles = np.atleast_1d(azimuth_angles)

    # Compute v for all angles
    v = np.stack(
        [
            np.sin(polar_angles) * np.cos(azimuth_angles),
            np.sin(polar_angles) * np.sin(azimuth_angles),
            np.cos(polar_angles),
        ],
        axis=-1,
    )

    # Add a new dimension for broadcasting
    v = np.expand_dims(v, axis=1)

    # Compute the location of the virtual source by adding the focus distance
    # to the origin along the wave vectors.
    virtual_sources = origin + focus_distance * v

    # Compute the distances between the virtual sources and each element
    dist = np.linalg.norm(virtual_sources - probe_geometry, axis=-1)

    # Adjust distances based on the direction of focus
    dist *= -np.sign(focus_distance)

    # Convert from distance to time to compute the
    # transmit delays/travel times.
    travel_times = dist / sound_speed

    # The smallest (possibly negative) time corresponds to the moment when
    # the first element fires.
    t_first_fire = np.min(travel_times, axis=1)

    # Shift the transmit delays such that the first element fires at t=0.
    t0_delays = travel_times - t_first_fire[:, None]

    return t0_delays.T


def plot_t0_delays(t0_delays):
    """Plot the t0_delays for each transducer element
    Elements are on the x-axis, and the t0_delays are on the y-axis.
    We plot multiple lines for each angle/transmit in the scan object."""
    n_tx = t0_delays.shape[0]
    _, ax = plt.subplots()
    for tx in range(n_tx):
        ax.plot(t0_delays[tx], label=f"Transmit {tx}")
    ax.set_xlabel("Element number")
    ax.set_ylabel("t0 delay [s]")
    plt.show()
