"""Class structures containing parameters defining an ultrasound scan and the
beamforming grid.

- **Author(s)**     : Vincent van de Schaft
- **Date**          : Wed Feb 15 2024
"""
import warnings
import numpy as np
from usbmd.utils.pixelgrid import get_grid

_MOD_TYPES = [None, 'rf', 'iq']

class Scan:
    """Scan base class."""

    def __init__(self, N_tx, fc=7e6, fs=28e6, c=1540, modtype='rf', N_ax=3328,
                 initial_times=None, t0_delays=None, tx_apodizations=None,
                 Nx=128, Nz=128, xlims=(-0.01, 0.01), ylims=(0, 0),
                 zlims=(0, 0.04), pixels_per_wvln=3, downsample=1):
        """Initializes a Scan object representing the number and type of
        transmits, and the target pixels to beamform to.

        The Scan object generates a pixel grid based on Nx, Nz, xlims, and
        zlims when it is initialized. When any of these parameters are changed
        the grid is recomputed automatically.

        Args:
            N_tx (int): The number of transmits to produce a single frame.
            xlim (tuple, optional): The x-limits in the beamforming grid.
                Defaults to (-0.01, 0.01).
            ylim (tuple, optional): The y-limits in the beamforming grid.
                Defaults to (0, 0).
            zlim (tuple, optional): The z-limits in the beamforming grid.
                Defaults to (0,0.04).
            fc (float, optional): The modulation carrier frequency.
                Defaults to 7e6.
            fs (float, optional): The sampling rate to sample rf- or
                iq-signals with. Defaults to 28e6.
            c (float, optional): The speed of sound in m/s. Defaults to 1540.
                modtype(string, optional): The modulation type. ('rf' or 'iq').
                Defaults to 'rf'
            N_ax (int, optional): The number of samples per in a receive
                recording per channel. Defaults to None.
            Nx (int, optional): The number of pixels in the lateral direction
                in the beamforming grid. Defaults to None.
            Nz (int, optional): The number of pixels in the axial direction in
                the beamforming grid. Defaults to None.
            pixels_per_wvln (int, optional): The number of pixels per wavelength
                to use in the beamforming grid. Only used when Nx and Nz are not
                defined. Defaults to 3.
            downsample (int, optional): Decimation factor applied after downconverting
                data to baseband (RF to IQ). Defaults to 1.

        Raises:
            NotImplementedError: Initializing from probe not yet implemented.
        """
        assert modtype in _MOD_TYPES, "modtype must be either 'rf' or 'iq'."

        # Attributes concerning channel data
        #: The number of transmissions in a frame
        self.N_tx = N_tx
        #: The modulation carrier frequency [Hz]
        self.fc = fc
        #: The sampling rate [Hz]
        self.fs = fs
        #: The speed of sound [m/s]
        self.c = c
        #: The modulation type of the raw data ('rf' or 'iq')
        self.modtype = modtype
        #: The number of samples per channel per acquisition
        self.N_ax = N_ax // downsample
        #: The demodulation frequency [Hz]
        self.fdemod = self.fc if modtype == 'iq' else 0.
        #: The number of rf/iq channels (1 for rf, 2 for iq)
        self.n_ch = 2 if modtype == 'iq' else 1
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

        #: The number of pixels in the lateral direction in the beamforming
        #: grid
        self._Nx = Nx
        #: The number of pixels in the axial direction in the beamforming grid
        self._Nz = Nz

        #: The beamforming grid of shape (Nx, Nz, 3)
        self._grid = None

        # Compute the zlims from the other values if not supplied
        if zlims:
            self.zlims = zlims
        else:
            self.zlims = [0, self.c * self.N_ax / self.fs / 2]
            print(self.zlims)

        self.z_axis = np.linspace(*self.zlims, N_ax)

        if initial_times is None:
            warnings.warn('No initial times provided. Assuming all zeros.')
            initial_times = np.zeros(N_tx)

        self.initial_times = initial_times

        if t0_delays is None:
            warnings.warn('No t0_delays provided. Assuming all zeros and '
                          '128 element probe.')
            t0_delays = np.zeros((N_tx, 128))
        self.t0_delays = t0_delays

        if tx_apodizations is None:
            warnings.warn('No tx_apodizations provided. Assuming all ones and '
                          '128 element probe.')
            tx_apodizations = np.ones((N_tx, 128))
        self.tx_apodizations = tx_apodizations

    # def add_transmit(self, index, t0_delays, tx_apodizations, c, initial_time=0.0):
    #     """Adds a transmit to the scan.

    #     Args:
    #         index (int): The index of the transmit.
    #         t0_delays (np.ndarray): The transmit delays in seconds of shape
    #             (n_el,), shifted such that the smallest delay is 0.
    #         tx_apodizations (np.ndarray, float): The transmit apodizations of
    #             shape (n_el,) or a single float to use for all
    #             apodizations.
    #         c (float): The speed of sound in m/s.
    #         initial_time (float, optional): The initial time of the transmit
    #             in seconds. Defaults to 0.0.

    #     """
    #     assert isinstance(tx_apodizations, (np.ndarray, float)), \
    #         "tx_apodizations must be a numpy array or a float."
    #     assert isinstance(t0_delays, np.ndarray), \
    #         "tx_delays must be a numpy array."

    #     if isinstance(tx_apodizations, float):
    #         # Set the transmit apodizations to all ones if not supplied
    #         tx_apodizations = np.ones_like(t0_delays)*tx_apodizations

    #     self.initial_times[index] = initial_time
    #     self.t0_delays[index] = t0_delays
    #     self.tx_apodizations[index] = tx_apodizations

    # def add_planewave_transmit(self, ele_pos, polar_angle, azimuth_angle=0,
    #                            apodization=1., c=1540, initial_time=0.0):
    #     """Adds a planewave transmit to the scan.

    #     Args:
    #         ele_pos (np.ndarray): The element positions in meters of shape
    #             (n_el, 3).
    #         polar_angle (float): The polar angle of the planewave wave vector
    #             in degrees. (This is the one that is used in 2D imaging.)
    #         azimuth_angle (float): The azimuth angle of the planewave wave
    #             vector in degrees. (This is the one that is used in 3D
    #             imaging.)
    #         focus_distance (float): The focus of the planewave in meters.
    #         apodization (float, optional): The apodization to use for the
    #             transmit. Defaults to 1.
    #         c (float): The speed of sound in m/s.
    #         initial_time (float, optional): The initial time of the transmit
    #             in seconds. Defaults to 0.0.
    #     """

    #     # Create a new transmit object and add it to the list of transmits
    #     transmit = PlanewaveTransmit(ele_pos, polar_angle, azimuth_angle,
    #                                  apodization, c, initial_time)
    #     self.transmits.append(transmit)

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
    Class representing a focussed beam scan where every transmit has a beam
    origin, angle, and focus defined.

        Args:
            probe (Probe, str, optional): Probe object to read values from or
                probe name to initialize. Defaults to None.
            n_tx (int, optional): The number of transmits to produce a single
                frame. Defaults to 75.
            xlim (tuple, optional): The x-limits in the beamforming grid.
                Defaults to (-0.01, 0.01).
            ylim (tuple, optional): The y-limits in the beamforming grid.
                Defaults to (0, 0).
            zlim (tuple, optional): The z-limits in the beamforming grid.
                Defaults to (0,0.04).
            fc (float, optional): The modulation carrier frequency.
                Defaults to 7e6.
            fs (float, optional): The sampling rate to sample rf- or
                iq-signals with. Defaults to 28e6.
            c (float, optional): The speed of sound in m/s. Defaults to 1540.
                modtype(string, optional): The modulation type. ('rf' or 'iq').
                Defaults to 'rf'
            N_ax (int, optional): The number of samples per in a receive
                recording per channel. Defaults to None.
            origins (ndarray, optional): The origins of the beams of shape
                (tx, 3). Defaults to None.
            focus_distances (ndarray, optional): The focus distances of the
                beams in m of shape (tx,). Defaults to None.
            Nx (int, optional): The number of pixels in the lateral direction
                in the beamforming grid. Defaults to None.
            Nz (int, optional): The number of pixels in the axial direction in
                the beamforming grid. Defaults to None.
            pixels_per_wvln (int, optional): The number of pixels per wavelength
                to use in the beamforming grid. Only used when Nx and Nz are not
                defined. Defaults to 3.
            downsample (int, optional): Decimation factor applied after downconverting
                data to baseband (RF to IQ). Defaults to 1.
    """

    def __init__(self, angles, origins, focus_distances, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Change to vfocus?
        self.origins = origins
        self.focus_distances = focus_distances
        self.angles = angles


class PlaneWaveScan(Scan):
    """
    Class representing a plane wave scan where every transmit has an angle.
    """

    def __init__(self, xlims=(-0.01, 0.01), ylims=(0, 0), zlims=(0, 0.04),
                 fc=7e6, fs=28e6, c=1540, modtype='rf',N_ax=256, Nx=128,
                 Nz=128, downsample=1, angles=None,
                 n_angles=None, initial_times=None):
        """
        Initializes a PlaneWaveScan object.

        Args:
            n_tx (int, optional): The number of planewave transmits. Defaults
                to 75.
            xlims (tuple, optional): The min and max x value in beamforming
                grid. Defaults to (-0.01, 0.01).
            ylims (tuple, optional): _description_. Defaults to (0, 0).
            zlims (tuple, optional): _description_. Defaults to (0, 0.04).
            fc (float, int, optional): The carrier frequency. Defaults to 7e6.
            fs (float, int, optional): The sampling rate. Defaults to 28e6.
            c (int, optional): The assumed speed of sound. Defaults to 1540.
            modtype (str, optional): The modulation type (rf or iq). Defaults
                to 'rf'.
            N_ax (int, optional): The number of axial samples per element per
                transmit. Defaults to 256.
            Nx (int, optional): The number of pixels in the x direction in the
                beamforming grid. Defaults to 128.
            Nz (int, optional): The number of pixels in the z direction in the
                beamforming grid. Defaults to 128.
            angles (list, optional): The angles of the planewaves. Defaults to
                None.
            n_angles (int, list, optional): The number of angles to use for
                beamforming. The angles will be sampled evenly from the angles
                list. Alternatively n_angles can contain a list of indices to
                use. Defaults to None.

        Raises:
            ValueError: If n_angles has an invalid value.
        """

        super().__init__(xlims=xlims, ylims=ylims, zlims=zlims, fc=fc, fs=fs,
                         c=c, modtype=modtype, N_ax=N_ax, Nx=Nx, Nz=Nz,
                         downsample=downsample)

        assert angles is not None, \
            'Please provide angles at which plane wave dataset was recorded'
        self.angles = angles
        if n_angles:
            if isinstance(n_angles, list):
                try:
                    self.n_angles = n_angles
                    self.angles = self.angles[n_angles]
                except Exception as exc:
                    raise ValueError(
                        'Angle indexing does not match the number of '
                        'recorded angles') from exc
            elif n_angles > len(self.angles):
                raise ValueError(
                    f'Number of angles {n_angles} specified supersedes '
                    f'number of recorded angles {len(self.angles)}.'
                )
            else:
                # Compute n_angles evenly spaced indices for reduced angles
                angle_indices = np.linspace(0, len(self.angles)-1, n_angles)
                # Round the computed angles to integers and turn into list
                angle_indices = list(np.rint(angle_indices).astype('int'))
                # Store the values and indices in the object
                self.n_angles = angle_indices
                self.angles = self.angles[angle_indices]
        else:
            self.n_angles = list(range(len(self.angles)))

        self.N_tx = len(self.angles)


class DivergingWaveScan(Scan):
    """Class representing a scan with diverging wave transmits."""

    def __init__(self, focus, *args, **kwargs):

        # super().__init__(xlims=xlims, ylims=ylims, zlims=zlims, fc=fc, fs=fs,
        #                  c=c, modtype=modtype, N_ax=N_ax, Nx=Nx, Nz=Nz,
        #                  downsample=downsample)

        self.focus = focus
        raise NotImplementedError('DivergingWaveScan has not been implemented.')


class Transmit:
    """Transmit class for storing the parameters of a single transmit."""

    def __init__(self, t0_delays, tx_apodizations=1.0, initial_time=0) -> None:
        """Initializes a Transmit object.

        Args:
            t0_delays (np.ndarray, optional): The transmit delays for each
                element of shape (n_el,) (This is the time delay between the
                first element firing and this element firing). Defaults to
                None in which case tzero is set to all zeros.
            tx_apodizations (np.ndarray, float, optional): The transmit
                apodizations for each element of shape (n_el,) or a float
                value in which case the apodization will be the same for all
                elements. Defaults to 1.0 in which case the apodization is set
                to all ones.
            initial_time (float, optional): The time instant at which the
                A/D conversion starts and records the first sample, where t=0
                is defined as the moment the first element starts firing.
                Defaults to 0.0.
        """

        #: The transmit apodizations of shape (n_el,)
        self.tx_apodizations = tx_apodizations

        # Set to all ones if not supplied
        if isinstance(tx_apodizations, float):
            self.tx_apodizations = np.ones_like(t0_delays)

        #: The transmit delays for each element (This is the time delay between
        #: the first element firing and this element firing)
        self.t0_delays = t0_delays

        #: The time instant at which the A/D conversion starts and records the
        # first sample, where t=0 is defined as the moment the first element
        # starts firing. Defaults to 0.0.
        self.initial_time = initial_time

    def _compute_tx_delays(self, pixels, ele_pos, c=1540):
        """Computes the transmit delays for every pixel in `pixels`.
        That is the time delay between the first element firing and the
        wavefront reaching each pixel.

        Args:
            pixels (np.ndarray): The pixel positions in the array of shape
                (n_pixels, 3).
            ele_pos (np.ndarray): The positions of the elements in the array of
                shape (element, 3).
            c (float, optional): The speed of sound in m/s. Defaults to 1540.

        Returns:
            np.ndarray: The transmit delays for each pixel in s.
        """
        # Add an element dimension to pixels (el, pix, xyz)
        pixels = pixels[None]
        # Add a pixel dimension to ele_pos (el, pix, xyz)
        ele_pos = ele_pos[:, None]
        # Compute the distances between each pixel and each element
        # This brings dists to shape (el, pix)
        dists = np.linalg.norm(pixels-ele_pos, axis=2)
        # Compute the transmit delays
        tx_delays = dists/c
        # Add the transmit delays prior to firing for each element
        tx_delays += self.t0_delays[:, None]
        # TODO: Add delay for elements that are not in the subaperture.
        # (see torch version)

        # Compute the minimum transmit delay as this is the moment when the
        # wavefront reaches each pixel.
        tx_delay_min = np.min(tx_delays, axis=0)

        return tx_delay_min

class PlanewaveTransmit(Transmit):
    """Planewave transmit class. Initializes the Transmit base class with the
    correct transmit delays for a planewave."""

    def __init__(self, ele_pos, polar_angle, azimuth_angle=0,
                 tx_apodizations=1, c=1540, initial_time=0.0) -> None:
        """Initializes a TransmitPlanewave object. This is effectively a
        normal Transmit object with the transmit delays set to the correct
        values for a planewave.

        Args:
            ele_pos (np.ndarray): The positions of the elements in the array of
                shape (element, 3).
            polar_angle (float): The polar angle of the planewave in radians.
            azimuth_angle (float, optional): The azimuth angle of the planewave
                in radians. Defaults to 0.
            tx_apodizations (np.ndarray, optional): The transmit apodizations
                for each element. Defaults to None in which case the
                apodization is set to all ones.
            c (float, optional): The speed of sound in m/s. Defaults to 1540.
            initial_time (float, optional): The time instant at which the
                A/D conversion starts and records the first sample, where t=0
                is defined as the moment the first element starts firing.
                Defaults to 0.0.
        """
        # Compute the transmit delays
        t0_delays = compute_t0_delays_planewave(ele_pos, polar_angle,
                                                azimuth_angle, c)

        #: The angle of the planewave in radians
        #: This can be used to compute the transmit delays for the placewave
        #: transmit instead of using the tx_delays attribute.
        self.polar_angle = float(polar_angle)
        self.azimuth_angle = float(azimuth_angle)

        #: The wave vector of the planewave of shape `(3,)`
        self.v = np.array([np.sin(polar_angle)*np.cos(azimuth_angle),
                           np.sin(polar_angle)*np.sin(azimuth_angle),
                           np.cos(polar_angle)])

        # Initialize the Transmit base class
        super().__init__(t0_delays, tx_apodizations, initial_time)

    def _compute_tx_delays(self, pixels, ele_pos, c=1540):
        """Computes the transmit delays for every pixel in `pixels`.
        That is the time delay between the first element firing and the
        wavefront reaching each pixel.

        Args:
            pixels (np.ndarray): The pixel positions in the array of shape
                (n_pixels, 3).
            ele_pos (np.ndarray): The positions of the elements in the array of
                shape (element, 3).
            c (float, optional): The speed of sound in m/s. Defaults to 1540.

        Returns:
            np.ndarray: The transmit delays for each pixel in s.
        """
        # Add an element dimension to pixels (el, pix, xyz)
        pixels = pixels[None]
        # Add a pixel dimension to ele_pos (el, pix, xyz)
        ele_pos = ele_pos[:, None]
        # Compute the projection of the element positions onto the wave vector
        # and convert to time
        projection = np.sum(ele_pos*self.v[:, None, None], axis=1)/c

        # The smallest (possibly negative) time corresponds to the moment when
        # the first element fires.
        t_first_fire = np.min(projection)

        # Project the pixel positions onto the wave vector and convert to time
        pixel_projection = np.sum(pixels*self.v[:, None, None], axis=2)/c

        # Compute the transmit delays
        tx_delays = t_first_fire + pixel_projection

        return tx_delays

# WIP
class FocusedTransmit(Transmit):
    """Focused transmit class. Initializes the Transmit base class with the
    correct transmit delays for a focused transmit."""

    def __init__(self, origin, focus_distance, ele_pos, polar_angle,
                 azimuth_angle, c=1540, tx_apodizations=None) -> None:
        """Initializes a TransmitFocused object. This is effectively a normal
        Transmit object with the transmit delays set to the correct values for
        a focused transmit.

        Args:
            origin (np.ndarray): The origin of the focused transmit of shape
                (3,).
            focus_distance (float): The distance to the focus.
            polar_angle (float): The polar angle of the focus.
            azimuthal_angle (float): The azimuthal angle of the focus.
            ele_pos (np.ndarray): The positions of the elements in the array
                of shape (element, 3).
            c (float, optional): The speed of sound in m/s. Defaults to 1540.
            tx_apodizations (np.ndarray, optional): The transmit apodizations
                for each element. Defaults to None in which case the
                apodization is set to all ones.
        """
        # Compute the transmit delays
        t0_delays = compute_t0_delays_focused(origin, focus_distance, polar_angle,
                                              ele_pos, c)

        #: The trasmit origin in meters of shape (3,)
        #: This can be used to compute the transmit delays for a focused
        #: transmit instead of using the tx_delays attribute.
        self.origin = origin

        #: The distance to the virtual source/focal point in meters
        #: This can be used to compute the transmit delays for a focused
        #: transmit instead of using the tx_delays attribute.
        self.focus_distance = focus_distance

        #: The angle of the transmit in radians
        #: This can be used to compute the transmit delays for a focused
        #: transmit instead of using the tx_delays attribute.
        self.polar_angle = polar_angle

        self.azimuth_angle = azimuth_angle

        # Initialize the Transmit base class
        super().__init__(t0_delays, tx_apodizations)

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
    v = np.stack([np.sin(polar_angle)*np.cos(azimuth_angle),
                  np.sin(polar_angle)*np.sin(azimuth_angle),
                  np.cos(polar_angle)])[None]

    # Compute the projection of the element positions onto the wave vector
    projection = np.sum(ele_pos*v, axis=1)

    # Convert from distance to time to compute the transmit delays.
    t0_delays_not_zero_algined = projection/c

    # The smallest (possibly negative) time corresponds to the moment when
    # the first element fires.
    t_first_fire = np.min(projection)/c

    # The transmit delays are the projection minus the offset. This ensures
    # that the first element fires at t=0.
    t0_delays = t0_delays_not_zero_algined-t_first_fire

    return t0_delays


def compute_t0_delays_focused(origin, focus_distance, ele_pos, polar_angle,
                              azimuth_angle=0, c=1540):
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
    v = np.stack([np.sin(polar_angle)*np.cos(azimuth_angle),
                  np.sin(polar_angle)*np.sin(azimuth_angle),
                  np.cos(polar_angle)])

    # Compute the location of the virtual source by adding the focus distance
    # to the origin along the wave vector.
    virtual_source = origin + focus_distance*v

    # Add a dummy dimension for the element dimension
    virtual_source = virtual_source[None]

    # Compute the distance between the virtual source and each element
    dist = np.linalg.norm(virtual_source-ele_pos, axis=1)

    dist *= -np.sign(focus_distance)

    # Convert from distance to time to compute the
    # transmit delays/travel times.
    travel_times = dist/c

    # The smallest (possibly negative) time corresponds to the moment when
    # the first element fires.
    t_first_fire = np.min(travel_times, axis=0)

    # Shift the transmit delays such that the first element fires at t=0.
    t0_delays = travel_times-t_first_fire

    return t0_delays
