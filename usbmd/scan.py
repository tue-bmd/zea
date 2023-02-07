import numpy as np

from usbmd.utils.pixelgrid import make_pixel_grid, make_pixel_grid_v2
from usbmd.probes import get_probe


def initialize_scan_from_probe(probe):
    """Initializes a Scan object based on the default scan parameters of the given probe. Any parameters for which no default values are defined in the probe class are initialized with the default arguments in the Scan constructor.

    Args:
        probe (Probe or str): A probe object or probe name.

    Returns:
        Scan: A Scan object that is compatible with the probe.
    """
    if type(probe) == str:
        probe = get_probe(probe)

    default_parameters = probe.get_default_scan_parameters()

    scan = Scan(**default_parameters)
    return scan


class Scan:
    def __init__(self, N_tx=75, xlims=(-0.01, 0.01), ylims=(0, 0),
                 zlims=(0, 0.04), f_c=7e6, f_s=28e6, c=1540, modtype='rf',
                 N_ax=256, N_lat=128):
        """
        Initializes a Scan object representing the number and type of transmits,
        and the target pixels to beamform to.

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
            f_c (float, optional): The modulation carrier frequency.
            Defaults to 7e6.
            f_s (float, optional): The sampling rate to sample rf- or
            iq-signals with. Defaults to 28e6.
            c (float, optional): The speed of sound in m/s. Defaults to 1540.
            modtype(string, optional): The modulation type. ('rf' or 'iq').
            Defaults to 'rf'
            N_ax (int, optional): The number of pixels in the axial direction.
            Defaults to None.
            N_lat (int, optional): The number of pixels in the lateral
            direction. Defaults to None.

        Raises:
            NotImplementedError: Initializing from probe not yet implemented.
        """
        assert modtype in ['rf', 'iq'], "modtype must be either 'rf' or 'iq'."

        self.N_tx = N_tx
        self.xlims = xlims
        self.ylims = ylims
        self.zlims = zlims
        self.f_c = f_c
        self.f_s = f_s
        self.c = c
        self.modtype = modtype
        self.N_ax = N_ax
        self.N_lat = N_lat
        self.time_zero = np.zeros((N_tx,))
        self.fdemod = self.f_c if modtype == 'iq' else 0.
        self.wvln = self.c / self.f_c
        self.grid = None

        # !!! TODO, implement this such that no aliasing occurs
        if self.fdemod == 0:
            self.grid = make_pixel_grid_v2(
                self.xlims, self.zlims, self.N_lat, self.N_ax)
        else:
            pixels_per_wavelength = 3
            dx = self.wvln / pixels_per_wavelength
            dz = dx
            self.grid = make_pixel_grid(self.xlims, self.zlims, dx, dz)


class FocussedScan(Scan):
    """
    Class representing a focussed beam scan where every transmit has a beam
    origin, angle, and focus defined.
    """

    def __init__(self, probe=None, N_tx=75, xlim=(-0.01, 0.01), ylim=(0, 0),
                 zlim=(0, 0.04), f_c=7e6, f_s=28e6, N_ax=None, N_lat=None, origins=None,
                 focus_distances=None, angles=None):
        super().__init__(probe, N_tx, xlim, ylim, zlim, f_c, f_s, N_ax, N_lat)

        self.origins = origins
        self.focus_distances = focus_distances
        self.angles = angles


class PlaneWaveScan(Scan):
    """
    Class representing a plane wave scan where every transmit has an angle.
    """

    def __init__(self, probe=None, N_tx=75, xlim=(-0.01, 0.01), ylim=(0, 0),
                 zlim=(0, 0.04), f_c=7e6, f_s=28e6, N_ax=None, N_lat=None, angles=None):

        super().__init__(probe, N_tx, xlim, ylim, zlim, f_c, f_s, N_ax, N_lat)

        self.angles = angles


class CircularWaveScan(Scan):
    def __init__(self, probe=None, N_tx=75, xlim=(-0.01, 0.01), ylim=(0, 0),
                 zlim=(0, 0.04), f_c=7e6, f_s=28e6, N_ax=None, N_lat=None, focus=None):

        super().__init__(probe, N_tx, xlim, ylim, zlim, f_c, f_s, N_ax, N_lat)
        raise NotImplementedError('CircularWaveScan has not been implemented.')


if __name__ == '__main__':
    scan = initialize_scan_from_probe('verasonics_l11_4v')

    print('fpmje')
