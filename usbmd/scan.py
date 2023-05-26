"""Class structures containing parameters defining an ultrasound scan and the
beamforming grid.

- **Author(s)**     : Vincent van de Schaft
- **Date**          : Wed Feb 15 2024
"""
import numpy as np

from usbmd.utils.pixelgrid import get_grid

_MOD_TYPES = [None, 'rf', 'iq']

class Scan:
    """Scan base class."""
    def __init__(self, N_tx=75, xlims=(-0.01, 0.01), ylims=(0, 0),
                 zlims=(0, 0.04), fc=7e6, fs=28e6, c=1540, modtype='rf',
                 N_ax=3328, Nx=128, Nz=128, pixels_per_wvln=3,
                 downsample=1, initial_times=None):
        """
        Initializes a Scan object representing the number and type of transmits,
        and the target pixels to beamform to.

        The Scan object generates a pixel grid when it is initialized.

        Args:
            probe (Probe, str, optional): Probe object to read values from or
                probe name to initialize. Defaults to None.
            N_tx (int, optional): The number of transmits to produce a single
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
            Nx (int, optional): The number of pixels in the lateral direction
                in the beamforming grid. Defaults to None.
            Nz (int, optional): The number of pixels in the axial direction in
                the beamforming grid. Defaults to None.
            pixels_per_wvln (int, optional): The number of pixels per wavelength
                to use in the beamforming grid. Only used when Nx and Nz are not
                defined. Defaults to 3.
            downsample (int, optional): Decimation factor applied after downconverting
                data to baseband (RF to IQ). Defaults to 1.
            initial_times (ndarray, optional): The initial times of the
                transmits of shape (n_tx). Given that the first element fires
                at time 0, this is the time at which the first sample is
                recorded. Defaults to None in which case the initial times are
                set to zeros.


        Raises:
            NotImplementedError: Initializing from probe not yet implemented.
        """
        assert modtype in _MOD_TYPES, "modtype must be either 'rf' or 'iq'."

        # Attributes concerning channel data
        #: The number of transmits in a single frame
        self.N_tx = N_tx
        #: The transmit center frequency. This is the frequency of the
        #: modulation
        self.fc = fc
        #: The sampling frequency of the receive data
        self.fs = fs
        #: The average speed of sound
        self.c = c
        #: The modulation type. This can be 'rf' or 'iq'
        self.modtype = modtype
        #: The number of axial samples in a single receive recording
        self.N_ax = N_ax // downsample
        #: ?
        self.fdemod = self.fc if modtype == 'iq' else 0.
        #: The number of channels in a single receive recording. For IQ-data
        #: there are 2 channels (the I and the Q channel). For rf-data there
        #: is only 1 channel.
        self.N_ch = 2 if modtype == 'iq' else 1
        #: The wavelength of the carrier signal
        self.wvln = self.c / self.fc
        #: The number of pixels per wavelength
        self.pixels_per_wavelength = pixels_per_wvln
        if initial_times is None:
            #: The initial times of the transmits. This is the time between the
            #: first element firing and the first sample being recorded.
            self.initial_times = np.zeros(N_tx)
        else:
            self.initial_times = initial_times

        # Beamforming grid related attributes
        #: The lateral limits of the beamforming grid
        self.xlims = xlims
        #: The y-limits of the beamforming grid (only used in 3D imaging)
        self.ylims = ylims
        if zlims:
            #: The axial limits of the beamforming grid
            self.zlims = zlims
        else:
            self.zlims = [0, self.c * self.N_ax / self.fs / 2]
            print(self.zlims)

        #: The number of pixels in the lateral direction in the beamforming
        #: grid
        self.Nx = Nx
        #: The number of pixels in the axial direction in the beamforming grid
        self.Nz = Nz

        #: The z_axis locations of the pixels in the beamforming grid
        self.z_axis = np.linspace(*self.zlims, N_ax)

        #: The beamforming grid of shape (Nx, Nz, 3)
        self.grid = get_grid(self)

class FocussedScan(Scan):
    """
    Class representing a focussed beam scan where every transmit has a beam
    origin, angle, and focus defined.
    """

    def __init__(self, N_tx=75, xlims=(-0.01, 0.01), ylims=(0, 0),
                 zlims=(0, 0.04), fc=7e6, fs=28e6, c=1540, modtype='rf',
                 N_ax=256, origins=None,focus_distances=None, Nx=128, Nz=128,
                 angles=None, downsample=1, initial_times=None):
        super().__init__(
            N_tx=N_tx, xlims=xlims, ylims=ylims, zlims=zlims, fc=fc, fs=fs, c=c,
            modtype=modtype, N_ax=N_ax, Nx=Nx, Nz=Nz, downsample=downsample)

        self.origins = origins
        self.focus_distances = focus_distances
        self.angles = angles

class PlaneWaveScan(Scan):
    """
    Class representing a plane wave scan where every transmit has an angle.
    """

    def __init__(self, N_tx=75, xlims=(-0.01, 0.01), ylims=(0, 0),
                 zlims=(0, 0.04), fc=7e6, fs=28e6, c=1540, modtype='rf',
                 N_ax=256, Nx=128, Nz=128, downsample=1, pixels_per_wvln=3,
                 angles=None, n_angles=None):
        """
        Initializes a PlaneWaveScan object.

        Args:
            N_tx (int, optional): The number of planewave transmits. Defaults
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
            initial_times (ndarray, optional): The initial times of the
                transmits of shape (n_tx). Given that the first element fires
                at time 0, this is the time at which the first sample is
                recorded. Defaults to None in which case the initial times are
                set to zeros.

        Raises:
            ValueError: If n_angles has an invalid value.
        """

        super().__init__(
            N_tx=N_tx, xlims=xlims, ylims=ylims, zlims=zlims, fc=fc, fs=fs, c=c,
            modtype=modtype, N_ax=N_ax, Nx=Nx, Nz=Nz,
            downsample=downsample, pixels_per_wvln=pixels_per_wvln)

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
                        'Angle indexing does not match the number of '\
                        'recorded angles') from exc
            elif n_angles > len(self.angles):
                raise ValueError(
                    f'Number of angles {n_angles} specified supersedes '\
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


class CircularWaveScan(Scan):
    """Class representing a scan with diverging wave transmits."""
    def __init__(self, N_tx=75, xlims=(-0.01, 0.01), ylims=(0, 0),
                 zlims=(0, 0.04), fc=7e6, fs=28e6, c=1540, modtype='rf',
                 N_ax=256, Nx=128, Nz=128, downsample=1, focus=None):

        super().__init__(
            N_tx=N_tx, xlims=xlims, ylims=ylims, zlims=zlims, fc=fc, fs=fs, c=c,
            modtype=modtype, N_ax=N_ax, Nx=Nx, Nz=Nz, downsample=downsample,
            initial_times=None)

        self.focus = focus
        raise NotImplementedError('CircularWaveScan has not been implemented.')


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
