"""Class structures containing parameters defining an ultrasound scan and the
beamforming grid.

- **Author(s)**     : Vincent van de Schaft
- **Date**          : Wed Feb 15 2024
"""
import warnings

import numpy as np

from usbmd.utils.pixelgrid import check_for_aliasing, get_grid

_MOD_TYPES = [None, "rf", "iq"]


class Scan:
    """Scan base class."""

    def __init__(
        self,
        n_tx=75,
        n_el=128,
        xlims=(-0.01, 0.01),
        ylims=(0, 0),
        zlims=(0, 0.04),
        fc=7e6,
        fs=28e6,
        c=1540,
        modtype="rf",
        n_ax=3328,
        Nx=None,
        Nz=None,
        pixels_per_wvln=3,
        polar_angles=None,
        azimuth_angles=None,
        t0_delays=None,
        tx_apodizations=None,
        focus_distances=None,
        downsample=1,
        initial_times=None,
        selected_transmits=None,
    ):
        """Initializes a Scan object representing the number and type of
        transmits, and the target pixels to beamform to.

        The Scan object generates a pixel grid based on Nx, Nz, xlims, and zlims when it
        is initialized. When any of these parameters are changed the grid is recomputed
        automatically.

        Args:
            n_tx (int): The number of transmits to produce a single frame. xlims (tuple,
            optional): The x-limits in the beamforming grid.
                Defaults to (-0.01, 0.01).
            n_el (int, optional): The number of elements in the array. Defaults to 128.
            ylims (tuple, optional): The y-limits in the beamforming grid.
                Defaults to (0, 0).
            zlims (tuple, optional): The z-limits in the beamforming grid.
                Defaults to (0,0.04).
            fc (float, optional): The modulation carrier frequency.
                Defaults to 7e6.
            fs (float, optional): The sampling rate to sample rf- or
                iq-signals with. Defaults to 28e6.
            c (float, optional): The speed of sound in m/s. Defaults to 1540.
                modtype(string, optional): The modulation type. ('rf' or 'iq'). Defaults
                to 'rf'
            modtype (str, optional): The modulation type. ('rf' or 'iq'). n_ax (int,
            optional): The number of samples per in a receive
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
            NotImplementedError: Initializing from probe not yet implemented.
        """
        assert modtype in _MOD_TYPES, "modtype must be either 'rf' or 'iq'."

        # Attributes concerning channel data : The number of transmissions in a frame
        self.n_tx = int(n_tx)
        #: The number of elements in the array
        self.n_el = int(n_el)
        #: The modulation carrier frequency [Hz]
        self.fc = float(fc)
        #: The sampling rate [Hz]
        self.fs = float(fs)
        #: The speed of sound [m/s]
        self.c = float(c)
        #: The modulation type of the raw data ('rf' or 'iq')
        self.modtype = modtype
        #: The number of samples per channel per acquisition
        self.n_ax = n_ax // downsample
        #: The demodulation frequency [Hz]
        self.fdemod = self.fc if modtype == "iq" else 0.0
        #: The number of rf/iq channels (1 for rf, 2 for iq)
        self.n_ch = 2 if modtype == "iq" else 1
        #: The wavelength of the modulation carrier [m]
        self.wvln = self.c / self.fc
        #: The number of pixels per wavelength in the beamforming grid
        self.pixels_per_wavelength = pixels_per_wvln

        # Beamforming grid related attributes
        # ---------------------------------------------------------------------
        #: The x-limits of the beamforming grid [m]
        self._xlims = xlims
        #: The y-limits of the beamforming grid [m]
        self._ylims = ylims
        #: The z-limits of the beamforming grid [m]
        self._zlims = zlims

        #: The number of pixels in the lateral direction in the beamforming : grid
        self._Nx = int(Nx) if Nx else self.n_el
        #: The number of pixels in the axial direction in the beamforming grid
        self._Nz = int(Nz) if Nz else self.n_el

        #: The beamforming grid of shape (Nx, Nz, 3)
        self._grid = None

        # Compute the zlims from the other values if not supplied
        if zlims:
            self.zlims = zlims
        else:
            self.zlims = [0, self.c * self.n_ax / self.fs / 2]
            print(self.zlims)

        self.z_axis = np.linspace(*self.zlims, self.n_ax)

        if initial_times is None:
            warnings.warn("No initial times provided. Assuming all zeros.")
            initial_times = np.zeros(self.n_tx)

        #: The initial times of the transmits in seconds of shape (n_tx,). These are the
        # time intervals between the first element firing and the first sample in the
        # receive recording..
        self.initial_times = initial_times

        if t0_delays is None:
            warnings.warn(
                "No t0_delays provided. Assuming all zeros and 128 element probe."
            )
            t0_delays = np.zeros((self.n_tx, self.n_el))
        else:
            assert t0_delays.shape == (self.n_tx, self.n_el), (
                f"t0_delays must have shape (n_tx, n_el). "
                f"Got shape {t0_delays.shape}. Please set t0_delays either to None in which "
                f"case all zeros are assumed, or set the n_tx and n_el params to match the "
                "t0_delays shape."
            )
        #: The transmit delays in seconds of shape (n_tx, n_el), shifted such : that the
        # smallest delay is 0. For instance for a straight planewave : transmit all
        # delays are zero.
        self.t0_delays = t0_delays

        if tx_apodizations is None:
            warnings.warn(
                "No tx_apodizations provided. Assuming all ones and "
                "128 element probe."
            )
            tx_apodizations = np.ones((n_tx, self.n_el))
        #: The transmit apodizations of shape (n_tx, n_el) or a single float to : use
        # for all apodizations. These values indicate both windowing : (apodization) over
        # the aperture and the subaperture that is used : during transmit.
        self.tx_apodizations = tx_apodizations

        if polar_angles is None:
            warnings.warn("No polar_angles provided. Assuming all zeros.")
            polar_angles = np.zeros(self.n_tx)
        #: The polar angles of the transmits in radians of shape (n_tx,). These : are
        # the angles usually used in 2D imaging.
        self.polar_angles = polar_angles
        #: Identical to `Scan.polar_angles`. This attribute is added for : backward
        # compatibility.
        self.angles = self.polar_angles

        if azimuth_angles is None:
            warnings.warn("No azimuth_angles provided. Assuming all zeros.")
            azimuth_angles = np.zeros(self.n_tx)
        #: The azimuth angles of the transmits in radians of shape (n_tx,). : These are
        # the angles usually only used in 3D imaging.
        self.azimuth_angles = azimuth_angles

        if focus_distances is None:
            warnings.warn("No focus_distances provided. Assuming all zeros.")
            focus_distances = np.zeros(self.n_tx)
        #: The focus distances of the transmits in meters of shape (n_tx,). : These are
        # the distances of the virtual focus points from the origin. : For a planewave
        # these should be set to Inf.
        self.focus_distances = focus_distances

        #: Used to select a subset of the transmits to use for beamforming. If set to an
        # integer, then that number of transmits is selected as homogeneously as
        # possible. If set to a list of integers, then the transmits with those indices
        # are selected. If set to None, then all transmits are used. Defaults to None.
        self.selected_transmits = self.select_transmits(selected_transmits)

        check_for_aliasing(self)

    def select_transmits(self, selected_transmits):
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
            return list(range(self.n_tx))

        if isinstance(selected_transmits, int):
            # Do an error check if the number of selected transmits is not too large
            assert selected_transmits <= self.n_tx, (
                f"Number of selected transmits ({selected_transmits}) "
                f"exceeds number of transmits in scan ({self.n_tx})."
            )

            # If the number of selected transmits is 1, then pick the middle transmit
            if selected_transmits == 1:
                tx_indices = [self.n_tx // 2]
            else:
                # Compute selected_transmits evenly spaced indices for reduced angles
                tx_indices = np.linspace(0, self.n_tx - 1, selected_transmits)

                # Round the computed angles to integers and turn into list
                tx_indices = list(np.rint(tx_indices).astype("int"))

            # Update the number of transmits
            self.n_tx = selected_transmits

            return list(tx_indices)

        if isinstance(selected_transmits, list):
            assert all(
                isinstance(n, int) for n in selected_transmits
            ), "selected_transmits must be a list of integers."
            # Check if the selected transmits are not too large
            assert all(n < self.n_tx for n in selected_transmits), (
                f"Selected transmits {selected_transmits} exceed the number of "
                f"transmits in the scan ({self.n_tx})."
            )

            # Update the number of transmits
            self.n_tx = len(selected_transmits)

            return selected_transmits

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
        """The number of pixels in the axial direction in the beamforming
        grid."""
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
        angles=None,
        n_tx=75,
        n_el=128,
        xlims=(-0.01, 0.01),
        ylims=(0, 0),
        zlims=(0, 0.04),
        fc=7e6,
        fs=28e6,
        c=1540,
        modtype="rf",
        n_ax=3328,
        Nx=128,
        Nz=128,
        pixels_per_wvln=3,
        polar_angles=None,
        azimuth_angles=None,
        tx_apodizations=None,
        downsample=1,
        initial_times=None,
        selected_transmits=None,
    ):
        """
        Initializes a PlaneWaveScan object.

        Args:
            angles (list, optional): The angles of the planewaves. Defaults to
                None.
            n_tx (int): The number of transmits to produce a single frame. xlims (tuple,
                optional): The x-limits in the beamforming grid.
                Defaults to (-0.01, 0.01).
            n_el (int, optional): The number of elements in the array. Defaults to 128.
            ylims (tuple, optional): The y-limits in the beamforming grid.
                Defaults to (0, 0).
            zlims (tuple, optional): The z-limits in the beamforming grid.
                Defaults to (0,0.04).
            fc (float, optional): The modulation carrier frequency.
                Defaults to 7e6.
            fs (float, optional): The sampling rate to sample rf- or
                iq-signals with. Defaults to 28e6.
            c (float, optional): The speed of sound in m/s. Defaults to 1540.
                modtype(string, optional): The modulation type. ('rf' or 'iq'). Defaults
                to 'rf'
            modtype (str, optional): The modulation type. ('rf' or 'iq'). n_ax (int,
            optional): The number of samples per in a receive
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

        # Pass all arguments to the Scan base class
        super().__init__(
            n_tx=n_tx,
            n_el=n_el,
            xlims=xlims,
            ylims=ylims,
            zlims=zlims,
            fc=fc,
            fs=fs,
            c=c,
            modtype=modtype,
            n_ax=n_ax,
            Nx=Nx,
            Nz=Nz,
            pixels_per_wvln=pixels_per_wvln,
            polar_angles=angles,
            azimuth_angles=azimuth_angles,
            tx_apodizations=tx_apodizations,
            downsample=downsample,
            initial_times=initial_times,
            selected_transmits=selected_transmits,
            focus_distances=np.inf * np.ones(n_tx),
        )

        assert (
            angles is not None or polar_angles is not None
        ), "Please provide angles at which plane wave dataset was recorded"
        if angles is not None:
            self.angles = angles
            self.polar_angles = angles
        else:
            self.angles = polar_angles
            self.polar_angles = polar_angles


class DivergingWaveScan(Scan):
    """Class representing a scan with diverging wave transmits."""

    def __init__(
        self,
        n_tx=75,
        n_el=128,
        xlims=(-0.01, 0.01),
        ylims=(0, 0),
        zlims=(0, 0.04),
        fc=7e6,
        fs=28e6,
        c=1540,
        modtype="rf",
        n_ax=256,
        Nx=128,
        Nz=128,
        downsample=1,
        focus=None,
    ):
        super().__init__(
            n_tx=n_tx,
            n_el=n_el,
            xlims=xlims,
            ylims=ylims,
            zlims=zlims,
            fc=fc,
            fs=fs,
            c=c,
            modtype=modtype,
            n_ax=n_ax,
            Nx=Nx,
            Nz=Nz,
            downsample=downsample,
            initial_times=None,
        )

        self.focus = focus
        raise NotImplementedError("CircularWaveScan has not been implemented.")


def compute_t0_delays_planewave(ele_pos, polar_angle, azimuth_angle=0, c=1540):
    """Computes the transmit delays for a planewave, shifted such that the
    first element fires at t=0.

    Args:
        ele_pos (np.ndarray): The positions of the elements in the array of
            shape (element, 3).
        polar_angle (float): The polar angle of the planewave in radians.
        azimuth_angle (float, optional): The azimuth angle of the planewave
            in radians. Defaults to 0.
        c (float, optional): The speed of sound. Defaults to 1540.

    Returns:
        np.ndarray: The transmit delays for each element of shape (element,).
    """
    # The wave vector of the planewave of shape (1, 3)
    v = np.stack(
        [
            np.sin(polar_angle) * np.cos(azimuth_angle),
            np.sin(polar_angle) * np.sin(azimuth_angle),
            np.cos(polar_angle),
        ]
    )[None]

    # Compute the projection of the element positions onto the wave vector
    projection = np.sum(ele_pos * v, axis=1)

    # Convert from distance to time to compute the transmit delays.
    t0_delays_not_zero_algined = projection / c

    # The smallest (possibly negative) time corresponds to the moment when
    # the first element fires.
    t_first_fire = np.min(projection) / c

    # The transmit delays are the projection minus the offset. This ensures
    # that the first element fires at t=0.
    t0_delays = t0_delays_not_zero_algined - t_first_fire

    return t0_delays


def compute_t0_delays_focused(
    origin, focus_distance, ele_pos, polar_angle, azimuth_angle=0, c=1540
):
    """Computes the transmit delays for a focused transmit, shifted such that
    the first element fires at t=0.

    Args:
        origin (np.ndarray): The origin of the focused transmit of shape (3,).
        focus_distance (float): The distance to the focus.
        ele_pos (np.ndarray): The positions of the elements in the array of
            shape (element, 3).
        polar_angle (float): The polar angle of the planewave in radians.
        azimuth_angle (float, optional): The azimuth angle of the planewave
            in radians. Defaults to 0.
        c (float, optional): The speed of sound. Defaults to 1540.

    Returns:
        np.ndarray: The transmit delays for each element of shape (element,).
    """
    # Compute the wave vector of shape (1, 3)
    v = np.stack(
        [
            np.sin(polar_angle) * np.cos(azimuth_angle),
            np.sin(polar_angle) * np.sin(azimuth_angle),
            np.cos(polar_angle),
        ]
    )

    # Compute the location of the virtual source by adding the focus distance
    # to the origin along the wave vector.
    virtual_source = origin + focus_distance * v

    # Add a dummy dimension for the element dimension
    virtual_source = virtual_source[None]

    # Compute the distance between the virtual source and each element
    dist = np.linalg.norm(virtual_source - ele_pos, axis=1)

    dist *= -np.sign(focus_distance)

    # Convert from distance to time to compute the
    # transmit delays/travel times.
    travel_times = dist / c

    # The smallest (possibly negative) time corresponds to the moment when
    # the first element fires.
    t_first_fire = np.min(travel_times, axis=0)

    # Shift the transmit delays such that the first element fires at t=0.
    t0_delays = travel_times - t_first_fire

    return t0_delays
