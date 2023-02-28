"""
==============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : scan.py

    Author(s)     : Vincent van de Schaft
    Date          : Wed Feb 15 2024

    Class structures containing parameters defining an ultrasound scan and the
    beamforming grid.

==============================================================================
"""
import numpy as np

from usbmd.probes import get_probe
from usbmd.utils.pixelgrid import make_pixel_grid, make_pixel_grid_v2


def initialize_scan_from_probe(probe):
    """
    Initializes a Scan object based on the default scan parameters of the given
    probe. Any parameters for which no default values are defined in the probe
    class are initialized with the default arguments in the Scan constructor.

    Args:
        probe (Probe or str): A probe object or probe name.

    Returns:
        Scan: A Scan object that is compatible with the probe.
    """
    if isinstance(probe, str):
        probe = get_probe(probe)

    default_parameters = probe.get_default_scan_parameters()

    scan = Scan(**default_parameters)
    return scan

def initialize_scan_from_config(config):
    """
    Defines a scan based on parameters in a config.

    Args:
        config (Config): The config object to read parameters from.

    Raises:
        NotImplementedError: This method is not implemented and always raises
        this error.
    """
    raise NotImplementedError

class Scan:
    """Scan base class."""
    def __init__(self, N_tx=75, xlims=(-0.01, 0.01), ylims=(0, 0),
                 zlims=(0, 0.04), fc=7e6, fs=28e6, c=1540, modtype='rf',
                 N_ax=3328, Nx=128, Nz=128, tzero_correct=True):
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
            tzero_correct (bool, optional): Set to False to disable tzero
                correction. This is useful for datasets that have this
                correction in the raw data already. Defaults to True.

        Raises:
            NotImplementedError: Initializing from probe not yet implemented.
        """
        assert modtype in ['rf', 'iq'], "modtype must be either 'rf' or 'iq'."

        # Attributes concerning channel data
        self.N_tx = N_tx
        self.fc = fc
        self.fs = fs
        self.c = c
        self.modtype = modtype
        self.N_ax = N_ax
        self.fdemod = self.fc if modtype == 'iq' else 0.
        self.N_ch = 2 if modtype == 'iq' else 1
        self.wvln = self.c / self.fc
        self.tzero_correct = tzero_correct

        # Beamforming grid related attributes
        self.xlims = xlims
        self.ylims = ylims
        if zlims:
            self.zlims = zlims
        else:
            self.zlims = [0, self.c * self.N_ax / self.fs / 2]
            print(self.zlims)
        self.grid = None
        self.Nx = Nx
        self.Nz = Nz

        self.z_axis = np.linspace(*self.zlims, N_ax)

        # !!! TODO, implement this such that no aliasing occurs
        if self.fdemod == 0:
            self.grid = make_pixel_grid_v2(
                self.xlims, self.zlims, Nx, Nz)
        else:
            pixels_per_wavelength = 3
            dx = self.wvln / pixels_per_wavelength
            dz = dx
            self.grid = make_pixel_grid(self.xlims, self.zlims, dx, dz)

    def get_time_zero(self, element_positions, c=1540, offset=0):
        """Returns an ndarray with the delay between the first element firing
        and that element firing.

        Args:
            element_positions (ndarray): The element positions as specified in
                the Probe object.
            c (int, float): The assumed speed of sound.
            offset (float, optional): Additional offset to add. Defaults to 0.

        Returns:
            ndarray: The delays
        """
        # pylint: disable=unused-argument
        if self.tzero_correct:
            return np.ones(self.N_tx)*offset
        else:
            return np.zeros(self.N_tx)

class FocussedScan(Scan):
    """
    Class representing a focussed beam scan where every transmit has a beam
    origin, angle, and focus defined.
    """

    def __init__(self, N_tx=75, xlims=(-0.01, 0.01), ylims=(0, 0),
                 zlims=(0, 0.04), fc=7e6, fs=28e6, c=1540, modtype='rf',
                 N_ax=256, origins=None,focus_distances=None, Nx=128, Nz=128,
                 tzero_correct=True, angles=None):
        super().__init__(N_tx, xlims, ylims, zlims, fc, fs, c, modtype, N_ax,
                         Nx, Nz, tzero_correct)

        self.origins = origins
        self.focus_distances = focus_distances
        self.angles = angles


class PlaneWaveScan(Scan):
    """
    Class representing a plane wave scan where every transmit has an angle.
    """

    def __init__(self, N_tx=75, xlims=(-0.01, 0.01), ylims=(0, 0),
                 zlims=(0, 0.04), fc=7e6, fs=28e6, c=1540, modtype='rf',
                 N_ax=256, Nx=128, Nz=128, tzero_correct=True, angles=None,
                 n_angles=None):
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
            tzero_correct (bool, optional): Set to False to disable tzero
                correction. This is useful for datasets that have this
                correction in the raw data already. Defaults to True.
            angles (list, optional): The angles of the planewaves. Defaults to
                None.
            n_angles (int, list, optional): The number of angles to use for
                beamforming. The angles will be sampled evenly from the angles
                list. Alternatively n_angles can contain a list of indices to
                use. Defaults to None.

        Raises:
            ValueError: If n_angles has an invalid value.
        """

        super().__init__(N_tx, xlims, ylims, zlims, fc, fs, c, modtype, N_ax,
                         Nx, Nz, tzero_correct)

        assert angles is not None, \
            'Please provide the angles at which plane wave dataset was recorded'
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

    def get_time_zero(self, element_positions, c=1540, offset=0):
        """Returns an ndarray with the delay between the first element firing
        and that element firing.

        Args:
            element_positions (ndarray): The element positions as specified in
                the Probe object.
            c (int, float): The assumed speed of sound.
            offset (float, optional): Additional offset to add. Defaults to 0.

        Returns:
            ndarray: The delays
        """
        # pylint: disable=unused-argument
        if self.tzero_correct:
            return np.abs(np.sin(self.angles)) * element_positions.max() / c
        else:
            return np.zeros((self.N_tx,))


class CircularWaveScan(Scan):
    """Class representing a scan with diverging wave transmits."""
    def __init__(self, N_tx=75, xlims=(-0.01, 0.01), ylims=(0, 0),
                 zlims=(0, 0.04), fc=7e6, fs=28e6, c=1540, modtype='rf',
                 N_ax=256, Nx=128, Nz=128, tzero_correct=True, focus=None):

        super().__init__(N_tx, xlims, ylims, zlims, fc, fs, c, modtype, N_ax, Nx, Nz)
        self.focus = focus
        raise NotImplementedError('CircularWaveScan has not been implemented.')
