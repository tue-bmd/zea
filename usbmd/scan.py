"""Structure containing parameters defining an ultrasound scan."""

# pylint: disable=too-many-public-methods

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from keras import ops

from usbmd import log
from usbmd.beamform.delays import compute_t0_delays_planewave
from usbmd.beamform.pfield import compute_pfield
from usbmd.beamform.pixelgrid import check_for_aliasing, get_grid
from usbmd.display import (
    compute_scan_convert_2d_coordinates,
    compute_scan_convert_3d_coordinates,
)
from usbmd.internal.core import STATIC, Object
from usbmd.internal.parameters import Parameters, cache_with_dependencies


class Scan(Parameters):
    """Represents an ultrasound scan configuration with computed properties.

    Args:
        Nx (int): Number of samples in the x-direction (lateral).
        Nz (int): Number of samples in the z-direction (axial).
        sound_speed (float, optional): Speed of sound in the medium in m/s. Defaults to 1540.0.
        sampling_frequency (float): Sampling frequency in Hz.
        center_frequency (float): Center frequency of the transducer in Hz.
        n_el (int): Number of elements in the transducer array.
        n_tx (int): Number of transmit events in the dataset.
        n_ax (int): Number of axial samples in the received signal.
        n_ch (int, optional): Number of channels (1 for RF, 2 for IQ data). Defaults to 1.
        xlims (tuple of float): Lateral (x) limits of the imaging region in meters (min, max).
        ylims (tuple of float, optional): Elevation (y) limits of the imaging region in meters (min, max).
        zlims (tuple of float): Axial (z) limits of the imaging region in meters (min, max).
        probe_geometry (np.ndarray): Element positions as array of shape (n_el, 3).
        polar_angles (np.ndarray): Polar angles for each transmit event in radians.
        azimuth_angles (np.ndarray): Azimuth angles for each transmit event in radians.
        t0_delays (np.ndarray): Transmit delays in seconds for each event.
        tx_apodizations (np.ndarray): Transmit apodizations for each event.
        focus_distances (np.ndarray): Focus distances in meters for each event.
        initial_times (np.ndarray): Initial times in seconds for each event.
        bandwidth_percent (float, optional): Bandwidth as percentage of center frequency. Defaults to 200.0.
        demodulation_frequency (float, optional): Demodulation frequency in Hz.
        time_to_next_transmit (np.ndarray): Time between transmit events.
        pixels_per_wavelength (int, optional): Number of pixels per wavelength. Defaults to 4.
        downsample (int, optional): Downsampling factor for the data. Defaults to 1.
        element_width (float, optional): Width of each transducer element in meters. Defaults to 0.2e-3.
        resolution (float, optional): Desired spatial resolution in meters.
        theta_range (tuple, optional): Range of theta angles for 3D imaging.
        phi_range (tuple, optional): Range of phi angles for 3D imaging.
        rho_range (tuple, optional): Range of rho (radial) distances for 3D imaging.
        fill_value (float, optional): Value to use for out-of-bounds pixels. Defaults to 0.0.
        attenuation_coef (float, optional): Attenuation coefficient in dB/(MHz*cm). Defaults to 0.0.
        selected_transmits (None, str, int, list, or np.ndarray, optional): Specifies which transmit events to select.
            - None or "all": Use all transmits.
            - "center": Use only the center transmit.
            - int: Select this many evenly spaced transmits.
            - list/array: Use these specific transmit indices.

    Properties:
        grid (np.ndarray): Meshgrid of x and z coordinates, shape (Nx, Nz, 3).
        grid_spacing (tuple): Grid spacing in x and z directions (dx, dz).
        wavelength (float): Wavelength based on sound speed and center frequency.
        z_axis (np.ndarray): The z-axis of the beamforming grid [m].
        flatgrid (np.ndarray): Flattened beamforming grid of shape (Nz*Nx, 3).
        selected_transmits (list): List of selected transmit indices.
        n_tx_total (int): Total number of transmits in the full dataset.
        n_tx (int): Number of currently selected transmits.
        demodulation_frequency (float): Demodulation frequency in Hz.
        polar_angles (np.ndarray): Polar angles of selected transmits in radians.
        azimuth_angles (np.ndarray): Azimuth angles of selected transmits in radians.
        t0_delays (np.ndarray): Transmit delays of selected transmits in seconds.
        tx_apodizations (np.ndarray): Transmit apodizations of selected transmits.
        focus_distances (np.ndarray): Focus distances of selected transmits in meters.
        initial_times (np.ndarray): Initial times of selected transmits in seconds.
        time_to_next_transmit (np.ndarray): Time between selected transmit events.
    Methods:
        set_transmits(selection): Select which transmit events to use.

    """

    VALID_PARAMS = {
        # beamforming related parameters
        "Nx": {"type": int, "default": None},
        "Nz": {"type": int, "default": None},
        "xlims": {"type": tuple, "default": None},
        "ylims": {"type": tuple, "default": None},
        "zlims": {"type": tuple, "default": None},
        "pixels_per_wavelength": {"type": int, "default": 4},
        "downsample": {"type": int, "default": 1},
        "resolution": {"type": float, "default": None},
        # acquisition parameters
        "sound_speed": {"type": float, "default": 1540.0},
        "sampling_frequency": {"type": float, "default": None},
        "center_frequency": {"type": float, "default": None},
        "n_el": {"type": int, "default": None},
        "n_tx": {"type": int, "default": None},
        "n_ax": {"type": int, "default": None},
        "n_ch": {"type": int, "default": 1},
        "bandwidth_percent": {"type": float, "default": 200.0},
        "demodulation_frequency": {"type": float, "default": None},
        "element_width": {"type": float, "default": 0.2e-3},
        "attenuation_coef": {"type": float, "default": 0.0},
        # array parameters
        "probe_geometry": {"type": np.ndarray, "default": None},
        "polar_angles": {"type": np.ndarray, "default": None},
        "azimuth_angles": {"type": np.ndarray, "default": None},
        "t0_delays": {"type": np.ndarray, "default": None},
        "tx_apodizations": {"type": np.ndarray, "default": None},
        "focus_distances": {"type": np.ndarray, "default": None},
        "initial_times": {"type": np.ndarray, "default": None},
        "time_to_next_transmit": {"type": np.ndarray, "default": None},
        # scan conversion parameters
        "theta_range": {"type": tuple, "default": None},
        "phi_range": {"type": tuple, "default": None},
        "rho_range": {"type": tuple, "default": None},
        "fill_value": {"type": float, "default": 0.0},
    }

    def __init__(self, **kwargs):
        # Store the current selection state before initialization
        selected_transmits_input = kwargs.pop("selected_transmits", None)

        # Initialize parent class
        super().__init__(**kwargs)

        # Initialize selection to None
        self._selected_transmits = None

        # Apply selection from input if provided
        if selected_transmits_input is not None:
            self.set_transmits(selected_transmits_input)

    # Core properties with dependency tracking
    @cache_with_dependencies("Nx", "Nz")
    def grid(self):
        """Get meshgrid of x and z coordinates."""
        Nx, Nz = self.Nx, self.Nz

        # Calculate grid based on specified dimensions
        x = np.linspace(self.xlims[0], self.xlims[1], Nx)
        z = np.linspace(self.zlims[0], self.zlims[1], Nz)
        xgrid, zgrid = np.meshgrid(x, z, indexing="ij")

        # Create 3D grid with y=0 for compatibility with 3D code
        ygrid = np.zeros_like(xgrid)
        grid = np.stack([xgrid, ygrid, zgrid], axis=-1)

        return grid

    @cache_with_dependencies("sound_speed", "center_frequency")
    def wavelength(self):
        """Calculate the wavelength based on sound speed and center frequency."""
        return self.sound_speed / self.center_frequency

    @cache_with_dependencies("sound_speed", "sampling_frequency", "n_ax")
    def z_axis(self):
        """The z-axis of the beamforming grid [m]."""
        if self.zlims is None:
            zlims = [0, self.sound_speed * self.n_ax / self.sampling_frequency / 2]
        else:
            zlims = self.zlims
        return np.linspace(zlims[0], zlims[1], self.n_ax)

    @cache_with_dependencies("grid")
    def flatgrid(self):
        """The beamforming grid of shape (Nz*Nx, 3)."""
        return self.grid.reshape(-1, 3)

    @cache_with_dependencies()
    def selected_transmits(self):
        """Get the currently selected transmit indices.

        Returns:
            list: The list of selected transmit indices. If none were explicitly
            selected and n_tx is available, all transmits are used.
        """
        # Return all transmits if none explicitly selected
        if self._selected_transmits is None:
            if "n_tx" in self._params:
                return list(range(self._params["n_tx"]))
            return []
        return self._selected_transmits

    @property
    def n_tx_total(self):
        """The total number of transmits in the full dataset."""
        if "n_tx" not in self._params:
            raise ValueError("n_tx must be set first")
        return self._params["n_tx"]

    @property
    def n_tx(self):
        """The number of currently selected transmits."""
        return len(self.selected_transmits)

    def _invalidate_selected_transmits(self):
        """Explicitly invalidate the selected_transmits cache."""
        if "selected_transmits" in self._cache:
            self._cache.pop("selected_transmits")
            self._computed.discard("selected_transmits")
            # Also explicitly invalidate all dependents
            self._invalidate_dependents("selected_transmits")

    def set_transmits(self, selection):
        """Select which transmit events to use.

        This method provides flexible ways to select transmit events:

        Args:
            selection: Specifies which transmits to select:
                - None: Use all transmits
                - "all": Use all transmits
                - "center": Use only the center transmit
                - int: Select this many evenly spaced transmits
                - list/array: Use these specific transmit indices

        Returns:
            The current instance for method chaining.

        Raises:
            ValueError: If the selection is invalid or incompatible with the scan.
        """
        n_tx_total = self._params.get("n_tx")
        if n_tx_total is None:
            raise ValueError("n_tx must be set before calling set_transmits")

        # Handle None and "all" - use all transmits
        if selection is None or selection == "all":
            self._selected_transmits = None
            self._invalidate_selected_transmits()
            return self

        # Handle "center" - use center transmit
        if selection == "center":
            self._selected_transmits = [n_tx_total // 2]
            self._invalidate_selected_transmits()
            return self

        # Handle integer - select evenly spaced transmits
        if isinstance(selection, (int, np.integer)):
            selection = int(selection)  # Convert numpy integer to Python int
            if selection <= 0:
                raise ValueError(
                    f"Number of transmits must be positive, got {selection}"
                )

            if selection > n_tx_total:
                raise ValueError(
                    f"Requested {selection} transmits exceeds available transmits ({n_tx_total})"
                )

            if selection == 1:
                self._selected_transmits = [n_tx_total // 2]
            else:
                # Compute evenly spaced indices
                tx_indices = np.linspace(0, n_tx_total - 1, selection)
                self._selected_transmits = list(np.rint(tx_indices).astype(int))

            self._invalidate_selected_transmits()
            return self

        # Handle array-like - convert to list of indices
        if isinstance(selection, np.ndarray):
            if len(selection.shape) == 0:
                # Handle scalar numpy array
                return self.set_transmits(int(selection))
            elif len(selection.shape) == 1:
                selection = selection.tolist()
            else:
                raise ValueError(f"Invalid array shape: {selection.shape}")

        # Handle list of indices
        if isinstance(selection, list):
            # Validate indices
            if not all(isinstance(i, (int, np.integer)) for i in selection):
                raise ValueError("All transmit indices must be integers")

            if any(i < 0 or i >= n_tx_total for i in selection):
                raise ValueError(
                    f"Transmit indices must be between 0 and {n_tx_total-1}"
                )

            self._selected_transmits = [
                int(i) for i in selection
            ]  # Convert numpy integers to Python ints
            self._invalidate_selected_transmits()
            return self

        raise ValueError(f"Unsupported selection type: {type(selection)}")

    @property
    def demodulation_frequency(self):
        """The demodulation frequency."""
        if self._params.get("demodulation_frequency") is not None:
            return self._params["demodulation_frequency"]

        if self.n_ch is None:
            raise ValueError("Please set scan.n_ch or scan.demodulation_frequency.")

        # Default behavior based on n_ch
        return self.center_frequency if self.n_ch == 2 else 0.0

    @cache_with_dependencies("selected_transmits")
    def polar_angles(self):
        """The polar angles of transmits in radians."""
        value = self._params.get("polar_angles")
        if value is None:
            return None

        # Always filter based on selected_transmits if value matches n_tx shape
        selected = self.selected_transmits
        if len(value) == self._params.get("n_tx", 0):
            return value[selected]

        return value

    @cache_with_dependencies("selected_transmits")
    def azimuth_angles(self):
        """The azimuth angles of transmits in radians."""
        value = self._params.get("azimuth_angles")
        if value is None:
            return None

        selected = self.selected_transmits
        n_tx = self._params.get("n_tx", 0)
        if len(value) != n_tx:
            return value  # Return unfiltered if shape doesn't match n_tx

        return value[selected]

    @cache_with_dependencies("selected_transmits")
    def t0_delays(self):
        """The transmit delays in seconds."""
        value = self._params.get("t0_delays")
        if value is None:
            return None

        selected = self.selected_transmits
        n_tx = self._params.get("n_tx", 0)
        if len(value) != n_tx:
            return value  # Return unfiltered if shape doesn't match n_tx

        return value[selected]

    @cache_with_dependencies("selected_transmits")
    def tx_apodizations(self):
        """The transmit apodizations."""
        value = self._params.get("tx_apodizations")
        if value is None:
            return None

        selected = self.selected_transmits
        n_tx = self._params.get("n_tx", 0)
        if len(value) != n_tx:
            return value  # Return unfiltered if shape doesn't match n_tx

        return value[selected]

    @cache_with_dependencies("selected_transmits")
    def focus_distances(self):
        """The focus distances in meters."""
        value = self._params.get("focus_distances")
        if value is None:
            return None

        selected = self.selected_transmits
        n_tx = self._params.get("n_tx", 0)
        if len(value) != n_tx:
            return value  # Return unfiltered if shape doesn't match n_tx

        return value[selected]

    @cache_with_dependencies("selected_transmits")
    def initial_times(self):
        """The initial times in seconds."""
        value = self._params.get("initial_times")
        if value is None:
            return None

        selected = self.selected_transmits
        n_tx = self._params.get("n_tx", 0)
        if len(value) != n_tx:
            return value  # Return unfiltered if shape doesn't match n_tx

        return value[selected]

    @cache_with_dependencies("selected_transmits")
    def time_to_next_transmit(self):
        """Time between transmit events."""
        value = self._params.get("time_to_next_transmit")
        if value is None:
            return None

        selected = self.selected_transmits
        return value[:, selected]


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
    "element_width": float,
    "lens_correction": float,
    "tgc_gain_curve": np.array,
    "tx_waveform_indices": np.array,
    "waveforms_two_way": dict,
    "waveforms_one_way": dict,
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
            log.error(
                f"Unknown scan parameter: {key}, cannot cast to correct type "
                f"but will proceed anyways. Please add {key} to the `SCAN_PARAM_TYPES`."
            )

    return scan_parameters


class OldScan(Object):
    """Scan base class."""

    def __init__(
        self,
        n_tx: Union[int, None] = None,
        n_ax: Union[int, None] = None,
        n_el: Union[int, None] = None,
        n_ch: Union[int, None] = None,
        center_frequency: Union[float, None] = None,
        sampling_frequency: Union[float, None] = None,
        demodulation_frequency: Union[float, None] = None,
        xlims: Union[tuple[float, float], None] = None,
        ylims: Union[tuple[float, float], None] = None,
        zlims: Union[tuple[float, float], None] = None,
        bandwidth_percent: int = 200,
        sound_speed: float = 1540,
        Nx: Union[int, None] = None,
        Nz: Union[int, None] = None,
        pixels_per_wavelength: int = 4,
        downsample: int = 1,
        pfield: Union[np.ndarray, None] = None,
        pfield_kwargs: Union[dict, None] = None,
        apply_lens_correction: bool = False,
        lens_thickness: Union[float, None] = None,
        lens_sound_speed: Union[float, None] = None,
        f_number: float = 1.0,
        element_width: float = 0.2e-3,
        attenuation_coef: float = 0.0,
        theta_range: Union[tuple[float, float], None] = None,
        phi_range: Union[tuple[float, float], None] = None,
        rho_range: Union[tuple[float, float], None] = None,
        fill_value: float = 0.0,
        # arrays that can be set manually or lazily initialized
        polar_angles: Union[np.ndarray, None] = None,
        azimuth_angles: Union[np.ndarray, None] = None,
        t0_delays: Union[np.ndarray, None] = None,
        tx_apodizations: Union[np.ndarray, None] = None,
        focus_distances: Union[np.ndarray, None] = None,
        initial_times: Union[np.ndarray, None] = None,
        selected_transmits: Union[int, list[int], str, None] = None,
        probe_geometry: Union[np.ndarray, None] = None,
        time_to_next_transmit: Union[np.ndarray, None] = None,
        resolution: Union[float, None] = None,
        coordinates: Union[np.ndarray, None] = None,
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
            n_ch (int): The number of channels. This will determine the modulation type.
                Can be either RF (when `n_ch = 1`) or IQ (when `n_ch=2`).
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
                Defaults to (0, n_ax * sound_speed / sampling_frequency / 2).
            bandwidth_percent: Receive bandwidth of RF signal in % of center
                frequency. Not necessarily the same as probe bandwidth. Defaults to 200.
            sound_speed (float, optional): The speed of sound in m/s. Defaults to 1540.
            Nx (int, optional): The number of pixels in the lateral direction
                in the beamforming grid. Defaults to None.
            Nz (int, optional): The number of pixels in the axial direction in
                the beamforming grid. Defaults to None.
            pixels_per_wavelength (int, optional): The number of pixels per wavelength
                to use in the beamforming grid. Only used when Nx and Nz are not
                defined. Defaults to 3.
            downsample (int, optional): Decimation factor applied after downconverting
                data to baseband (RF to IQ). Defaults to 1.
            pfield (np.ndarray, float, optional): The estimated pressure field of shape
                (n_tx, Nz, Nx, 1) to perform automatic weighting. Defaults to None. In that
                case pfield is computed with pfield_kwargs options when called.
            pfield_kwargs (np.ndarray, float, optional): Arguments to calculate the estimated
                pressure field of shape (n_tx, Nz, Nx, 1) with to perform automatic weighting.
                Defaults to None. In that case default arguments are used. If pfield
                can be used by the beamformer with option
                config.model.beamformer.auto_pressure_weighting. If set to True, the
                pfield is used, if False, beamformer will compound all
                transmit data coherently without pfield weighting.
            apply_lens_correction (bool, optional): Whether to apply lens correction to the
                delay computation. Defaults to False.
            lens_thickness (float, optional): The thickness of the lens in meters.
                Defaults to None.
            lens_sound_speed (float, optional): The speed of sound in the lens in m/s.
                Defaults to None.
            theta_range (tuple, optional): The range of theta values in radians.
                Defaults to None.
            phi_range (tuple, optional): The range of phi values in radians.
                Defaults to None.
            rho_range (tuple, optional): The range of rho values in meters.
                Defaults to None.
            attenuation_coef (float, optional): The attenuation coefficient in
                dB/cm/MHz. Defaults to 0.0.
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
                in meters. Necessary for automatic xlim calculation if not set.
                Defaults to None.
            time_to_next_transmit (np.ndarray, float, optional): The time between subsequent
                transmit events of shape (n_frames, n_tx). Defaults to None.
            resolution (float, optional): The resolution for scan conversion.
                Defaults to None, in which case it will be automatically computed.
            coordinates (np.ndarray, optional): The coordinates for scan conversion.
                Defaults to None.

        """
        super().__init__()

        # explicitely set parameters to None for linter to recognize them
        # only necessary for parameters that don't have @property decorators
        self.apply_lens_correction = None
        self.f_number = None
        self.probe_geometry = None
        self.pixels_per_wavelength = None
        self.downsample = None
        self.phi_range = None
        self.rho_range = None
        self.fill_value = None
        self._n_tx = None
        self.resolution = None
        self._coordinates = None
        self._theta_range = None
        self._time_to_next_transmit = None

        # Dictionary to track which parameters have been set
        self._set_params = {}

        # Store values and mark as set if not None
        # Basic parameters
        self._set_param("n_tx", n_tx)
        self._set_param("n_ax", n_ax)
        self._set_param("n_el", n_el)
        self._set_param("n_ch", n_ch)
        self._set_param("center_frequency", center_frequency)
        self._set_param("sampling_frequency", sampling_frequency)
        self._set_param("bandwidth_percent", bandwidth_percent)
        self._set_param("sound_speed", sound_speed)
        self._set_param("demodulation_frequency", demodulation_frequency)
        self._set_param("xlims", xlims)
        self._set_param("ylims", ylims)
        self._set_param("zlims", zlims)
        self._set_param("Nx", Nx)
        self._set_param("Nz", Nz)
        self._set_param("resolution", resolution)
        self._set_param("theta_range", theta_range)

        # Store array values and mark as set if not None
        self._set_param("t0_delays", t0_delays)
        self._set_param("tx_apodizations", tx_apodizations)
        self._set_param("polar_angles", polar_angles)
        self._set_param("azimuth_angles", azimuth_angles)
        self._set_param("focus_distances", focus_distances)
        self._set_param("initial_times", initial_times)
        self._set_param("pfield", pfield)
        self._set_param("coordinates", coordinates)
        self._set_param("time_to_next_transmit", time_to_next_transmit)

        if pfield is None:
            self._set_params["flat_pfield"] = False

        self._set_param("grid", None)
        self._set_params["flatgrid"] = False

        if not (
            self._set_params["sound_speed"] and self._set_params["center_frequency"]
        ):
            self._set_params["wvln"] = False

        if not (self._set_params["zlims"] and self._set_params["n_ax"]):
            self._set_params["z_axis"] = False

        # Additional properties that don't need lazy initialization
        self._set_param(
            "pixels_per_wavelength", float(pixels_per_wavelength), dunder=False
        )
        self._set_param("downsample", downsample, dunder=False)
        self._set_param("probe_geometry", probe_geometry, dunder=False)
        self._set_param("f_number", f_number, dunder=False)
        self._set_param("pfield_kwargs", pfield_kwargs, dunder=False)
        self._set_param("apply_lens_correction", apply_lens_correction, dunder=False)
        self._set_param("lens_thickness", lens_thickness, dunder=False)
        self._set_param("lens_sound_speed", lens_sound_speed, dunder=False)
        self._set_param("element_width", element_width, dunder=False)
        self._set_param("attenuation_coef", attenuation_coef, dunder=False)
        self._set_param("fill_value", fill_value, dunder=False)
        self._set_param("phi_range", phi_range, dunder=False)
        self._set_param("rho_range", rho_range, dunder=False)

        self.selected_transmits = selected_transmits
        self._static_attrs = STATIC

        # Put attributes here that are (very) slow to compute
        # They will only be computed if the pipeline actually needs them
        self._on_request = ["pfield", "flat_pfield"]

    def _set_param(self, name, value, dunder=True):
        """Set a parameter value and mark it as set in _set_params if not None.

        Args:
            name (str): The parameter name
            value: The parameter value
            dunder (bool, optional): Whether to set the parameter as a dunder attribute.
        """
        if dunder:
            setattr(self, f"_{name}", value)
        else:
            setattr(self, name, value)

        if value is None:
            self._set_params[name] = False
        else:
            self._set_params[name] = True

    # Add property getters and setters for each array that needs lazy initialization
    @property
    def t0_delays(self):
        """The transmit delays in seconds of shape (n_tx, n_el), shifted such that the
        smallest delay is 0. For instance for a straight planewave transmit all delays
        are zero."""
        if self._set_params["t0_delays"] is False:
            if self._n_tx is None or self._n_el is None:
                raise ValueError(
                    "Cannot initialize t0_delays: n_tx or n_el is not set. "
                    "Please set scan.n_tx and scan.n_el first."
                )
            log.warning(
                "No t0_delays provided. Assuming all zeros and "
                f"{self._n_el} element probe."
            )
            self._t0_delays = np.zeros((self._n_tx, self._n_el))
            self._set_params["t0_delays"] = True
        return self._t0_delays[self.selected_transmits]

    @t0_delays.setter
    def t0_delays(self, value):
        if value is not None:
            if self._n_tx is None or self._n_el is None:
                # Store for later validation once n_tx and n_el are set
                self._t0_delays = value
            else:
                assert value.shape == (self._n_tx, self._n_el), (
                    f"t0_delays must have shape (n_tx, n_el): {self._n_tx, self._n_el}. "
                    f"Got shape {value.shape}. Please set t0_delays either to None in which "
                    f"case all zeros are assumed, or set the n_tx and n_el params to match the "
                    "t0_delays shape."
                )
                self._t0_delays = value
            self._set_params["t0_delays"] = True

    @property
    def tx_apodizations(self):
        """The transmit apodizations of shape (n_tx, n_el) or a single float to use for
        all apodizations. These values indicate both windowing (apodization) over the
        aperture and the subaperture that is used during transmit."""
        if self._set_params["tx_apodizations"] is False:
            if self._n_tx is None or self._n_el is None:
                raise ValueError(
                    "Cannot initialize tx_apodizations: n_tx or n_el is not set. "
                    "Please set scan.n_tx and scan.n_el first."
                )
            log.warning(
                "No tx_apodizations provided. Assuming all ones and "
                f"{self._n_el} element probe."
            )
            self._tx_apodizations = np.ones((self._n_tx, self._n_el))
            self._set_params["tx_apodizations"] = True
        return self._tx_apodizations[self.selected_transmits]

    @tx_apodizations.setter
    def tx_apodizations(self, value):
        if value is not None:
            if self._n_tx is None or self._n_el is None:
                # Store for later validation once n_tx and n_el are set
                self._tx_apodizations = value
            else:
                assert value.shape == (self._n_tx, self._n_el), (
                    f"tx_apodizations must have shape (n_tx, n_el) = "
                    f"{self._n_tx, self._n_el}. "
                    f"Got shape {value.shape}."
                )
                self._tx_apodizations = value
            self._set_params["tx_apodizations"] = True

    @property
    def polar_angles(self):
        """The polar angles of the transmits in radians of shape (n_tx,). These are the
        angles usually used in 2D imaging."""
        if self._set_params["polar_angles"] is False:
            if self._n_tx is None:
                raise ValueError(
                    "Cannot initialize polar_angles: n_tx is not set. "
                    "Please set scan.n_tx first."
                )
            log.warning("No polar_angles provided. Assuming all zeros.")
            self._polar_angles = np.zeros(self._n_tx)
            self._set_params["polar_angles"] = True
        return self._polar_angles[self.selected_transmits]

    @polar_angles.setter
    def polar_angles(self, value):
        if value is not None:
            if self._n_tx is None:
                # Store for later validation once n_tx is set
                self._polar_angles = value
            else:
                assert len(value) == self._n_tx, (
                    f"polar_angles must have length n_tx = {self._n_tx}. "
                    f"Got length {len(value)}."
                )
                self._polar_angles = value
            self._set_params["polar_angles"] = True

    @property
    def azimuth_angles(self):
        """The azimuth angles of the transmits in radians of shape (n_tx,). These are
        the angles usually used in 3D imaging."""
        if self._set_params["azimuth_angles"] is False:
            if self._n_tx is None:
                raise ValueError(
                    "Cannot initialize azimuth_angles: n_tx is not set. "
                    "Please set scan.n_tx first."
                )
            log.warning("No azimuth_angles provided. Assuming all zeros.")
            self._azimuth_angles = np.zeros(self._n_tx)
            self._set_params["azimuth_angles"] = True
        return self._azimuth_angles[self.selected_transmits]

    @azimuth_angles.setter
    def azimuth_angles(self, value):
        if value is not None:
            if self._n_tx is None:
                # Store for later validation once n_tx is set
                self._azimuth_angles = value
            else:
                assert len(value) == self._n_tx, (
                    f"azimuth_angles must have length n_tx = {self._n_tx}. "
                    f"Got length {len(value)}."
                )
                self._azimuth_angles = value
            self._set_params["azimuth_angles"] = True

    @property
    def focus_distances(self):
        """The focus distances of the transmits in meters of shape (n_tx,). These are
        the distances of the virtual focus points from the origin. For a planewave
        these should be set to Inf."""
        if self._set_params["focus_distances"] is False:
            if self._n_tx is None:
                raise ValueError(
                    "Cannot initialize focus_distances: n_tx is not set. "
                    "Please set scan.n_tx first."
                )
            log.warning("No focus_distances provided. Assuming all zeros.")
            self._focus_distances = np.zeros(self._n_tx)
            self._set_params["focus_distances"] = True
        return self._focus_distances[self.selected_transmits]

    @focus_distances.setter
    def focus_distances(self, value):
        if value is not None:
            if self._n_tx is None:
                # Store for later validation once n_tx is set
                self._focus_distances = value
            else:
                assert len(value) == self._n_tx, (
                    f"focus_distances must have length n_tx = {self._n_tx}. "
                    f"Got length {len(value)}."
                )
                self._focus_distances = value
            self._set_params["focus_distances"] = True

    @property
    def initial_times(self):
        """The initial times of the transmits in seconds of shape (n_tx,). These are the
        time intervals between the first element firing and the first sample in the
        receive recording."""
        if self._set_params["initial_times"] is False:
            if self._n_tx is None:
                raise ValueError(
                    "Cannot initialize initial_times: n_tx is not set. "
                    "Please set scan.n_tx first."
                )
            log.warning("No initial times provided. Assuming all zeros.")
            self._initial_times = np.zeros(self._n_tx)
            self._set_params["initial_times"] = True
        return self._initial_times[self.selected_transmits]

    @initial_times.setter
    def initial_times(self, value):
        if value is not None:
            if self._n_tx is None:
                # Store for later validation once n_tx is set
                self._initial_times = value
            else:
                assert len(value) == self._n_tx, (
                    f"initial_times must have length n_tx = {self._n_tx}. "
                    f"Got length {len(value)}."
                )
                self._initial_times = value
            self._set_params["initial_times"] = True

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

        # Catch all other cases
        raise ValueError(f"Invalid value for selected_transmits: {selected_transmits}.")

    @property
    def selected_transmits(self):
        """Used to select a subset of the transmits to use for beamforming. If set to an
        integer, then that number of transmits is selected as homogeneously as possible.
        If set to a list of integers, then the transmits with those indices are
        selected. If set to None, then all transmits are used. Defaults to None."""
        return self._selected_transmits

    def reset_pfield(self):
        """Reset the pfield to None such that it will be recomputed."""
        self._pfield = None

    @selected_transmits.setter
    def selected_transmits(self, value):
        self._selected_transmits = self._select_transmits(value)
        try:
            check_for_aliasing(self)
        except ValueError as e:
            log.warning(f"Error checking for aliasing: {e}")

        self.reset_pfield()  # also trigger update of the pressure fields

    @property
    def n_tx(self):
        """The number of transmits to produce a single frame."""
        return len(self.selected_transmits)

    @property
    def n_ax(self):
        """The number of samples in a receive recording per channel."""
        if self._n_ax is None:
            raise ValueError("Please set scan.n_ax.")
        return int(self._n_ax)

    @n_ax.setter
    def n_ax(self, value):
        value = int(value)
        assert value > 0, "n_ax must be positive"
        self._n_ax = value
        self._set_params["n_ax"] = True

    @property
    def n_el(self):
        """The number of elements in the array."""
        if self._n_el is None:
            raise ValueError("Please set scan.n_el.")
        return int(self._n_el)

    @n_el.setter
    def n_el(self, value):
        value = int(value)
        assert value > 0, "n_el must be positive"
        self._n_el = value
        self._set_params["n_el"] = True

    @property
    def n_ch(self):
        """The number of channels."""
        if self._n_ch is None:
            raise ValueError("Please set scan.n_ch.")
        return int(self._n_ch)

    @n_ch.setter
    def n_ch(self, value):
        self._n_ch = value
        self._demodulation_frequency = None  # Reset demodulation_frequency
        self._set_params["n_ch"] = True

    @property
    def center_frequency(self):
        """The modulation carrier frequency."""
        if self._center_frequency is None:
            raise ValueError("Please set scan.center_frequency.")
        return float(self._center_frequency)

    @center_frequency.setter
    def center_frequency(self, value):
        self._center_frequency = value
        self._grid = None
        self._set_params["center_frequency"] = True

    @property
    def sampling_frequency(self):
        """The sampling rate."""
        if self._sampling_frequency is None:
            raise ValueError("Please set scan.sampling_frequency.")
        return float(self._sampling_frequency)

    @sampling_frequency.setter
    def sampling_frequency(self, value):
        self._sampling_frequency = value
        self._set_params["sampling_frequency"] = True

    @property
    def bandwidth_percent(self):
        """The percent bandwidth."""
        if self._bandwidth_percent is None:
            raise ValueError("Please set scan.bandwidth_percent.")
        return float(self._bandwidth_percent)

    @bandwidth_percent.setter
    def bandwidth_percent(self, value):
        self._bandwidth_percent = value
        self._set_params["bandwidth_percent"] = True

    @property
    def sound_speed(self):
        """The speed of sound."""
        if self._sound_speed is None:
            raise ValueError("Please set scan.sound_speed.")
        return float(self._sound_speed)

    @sound_speed.setter
    def sound_speed(self, value):
        self._sound_speed = value
        self._grid = None
        self._set_params["sound_speed"] = True

    @property
    def demodulation_frequency(self):
        """The demodulation frequency."""
        if self._demodulation_frequency is not None:
            return self._demodulation_frequency

        if self.n_ch is None:
            raise ValueError(
                "Please set scan.n_ch or scan.demodulation_frequency. Currently neither is set.\n"
                "\tif n_ch is set to 1 (RF), demodulation_frequency is set to 0.0.\n"
                "\tif n_ch is set to 2 (IQ), demodulation_frequency is set to center_frequency.\n"
                "\tdemodulation_frequency can be set to any other value manually."
            )

        return self.center_frequency if self.n_ch == 2 else 0.0

    @demodulation_frequency.setter
    def demodulation_frequency(self, value):
        self._demodulation_frequency = value

    @property
    def Nx(self):
        """The number of pixels in the lateral direction in the
        beamforming grid.

        If None, the number of pixels is calculated based on the
        `pixels_per_wavelength` parameter. See `usbmd.beamform.pixelgrid.get_grid`.
        """
        _ = self.grid  # Ensure grid is initialized
        return int(self._Nx) if self._Nx is not None else None

    @Nx.setter
    def Nx(self, value):
        self._Nx = int(value)
        self._grid = None

    @property
    def Nz(self):
        """The number of pixels in the axial direction in the
        beamforming grid.

        If None, the number of pixels is calculated based on the
        `pixels_per_wavelength` parameter. See `usbmd.beamform.pixelgrid.get_grid`.
        """
        _ = self.grid  # Ensure grid is initialized
        return int(self._Nz) if self._Nz is not None else None

    @Nz.setter
    def Nz(self, value):
        self._Nz = int(value)
        self._grid = None

    @property
    def wvln(self):
        """The wavelength of the modulation carrier [m]."""
        return self.sound_speed / self.center_frequency

    @property
    def xlims(self):
        """The x-limits of the beamforming grid [m]."""
        if self._xlims is None:
            if self.probe_geometry is None:
                raise ValueError(
                    "Please provide probe_geometry or xlims, currently neither is set."
                )
            self.xlims = [self.probe_geometry[0, 0], self.probe_geometry[-1, 0]]
        return self._xlims

    @xlims.setter
    def xlims(self, value):
        self._xlims = value
        self._grid = None

    @property
    def ylims(self):
        """The y-limits of the beamforming grid [m]."""
        if self._ylims is None:
            self.ylims = [0, 0]
        return self._ylims

    @ylims.setter
    def ylims(self, value):
        self._ylims = value
        self._grid = None

    @property
    def zlims(self):
        """The z-limits of the beamforming grid [m]."""
        if self._zlims is None:
            self.zlims = [0, self.sound_speed * self.n_ax / self.sampling_frequency / 2]
        return self._zlims

    @zlims.setter
    def zlims(self, value):
        self._zlims = value
        self._grid = None

    @property
    def z_axis(self):
        """The z-axis of the beamforming grid [m]."""
        return np.linspace(*self.zlims, self.n_ax)

    @property
    def grid(self):
        """The beamforming grid of shape (Nz, Nx, 3)."""
        if self._grid is None:
            self.grid = get_grid(
                self.xlims,
                self.zlims,
                self._Nx,
                self._Nz,
                self.sound_speed,
                self.center_frequency,
                self.pixels_per_wavelength,
            )

        return self._grid

    @grid.setter
    def grid(self, value):
        """Manually set the grid."""
        self._grid = value
        self._Nz, self._Nx, _ = self._grid.shape
        self.reset_pfield()  # also trigger update of the pressure fields

    @property
    def flatgrid(self):
        """The beamforming grid of shape (Nz*Nx, 3)."""
        if self.grid is None:
            return None
        return self.grid.reshape(-1, 3)

    def _default_pfield(self):
        """Default pfield weighting."""
        return np.ones((self.n_tx, self.Nz, self.Nx))

    @property
    def pfield(self):
        """The pfield grid of shape (n_tx, Nz, Nx)."""
        # TODO: pfield should be recomputed when some scan parameters change...
        # e.g. you change Nx, the grid gets changed, but only once you try to access self.grid,
        # which does not happen if you access self.pfield.
        # We should either make it very clear what each attribute depends on, or we should just
        # always recompute everything...
        if self._pfield is not None:
            return self._pfield

        if self.probe_geometry is None:
            log.warning(
                "scan.probe_geometry not set. Cannot compute pfield."
                "Defaulting to uniform weights."
            )
            if None in [self.Nz, self.Nx, self.n_tx]:
                log.warning("Nx, Nz, or n_tx not set. Cannot compute pfield.")
                return self._pfield

            self._pfield = self._default_pfield()
        else:
            if self.pfield_kwargs is None:
                pfield_kwargs = {}
            else:
                pfield_kwargs = self.pfield_kwargs

            self._pfield = ops.convert_to_numpy(
                compute_pfield(
                    self.sound_speed,
                    self.center_frequency,
                    self.bandwidth_percent,
                    self.n_el,
                    self.probe_geometry,
                    self.tx_apodizations,
                    self.grid,
                    self.t0_delays,
                    **pfield_kwargs,
                )
            )

        return self._pfield

    @property
    def flat_pfield(self):
        """The pfield grid of shape (Nz*Nx, n_tx)."""
        if self.pfield is None:
            return None
        return self.pfield.reshape(self.n_tx, -1).swapaxes(0, 1)

    @pfield.setter
    def pfield(self, value):
        """Manually set the pfield."""
        self.pfield_kwargs = None
        self._pfield = value
        assert self._pfield.shape == (self.n_tx, self.Nz, self.Nx), (
            f"pfield must have shape (n_tx, Nz, Nx) = {self.n_tx, self.Nz, self.Nx}. "
            f"Got shape {self._pfield.shape}."
        )

    @property
    def element_width(self):
        """The width of a single transducer elemen in meters."""
        return self._element_width

    @element_width.setter
    def element_width(self, value):
        """Set the element width in meters."""
        value = float(value)
        assert value > 0.0, "Element width must be positive"
        self._element_width = value

    @property
    def attenuation_coef(self):
        """The attenuation coefficient in dB/cm/MHz."""
        return self._attenuation_coef

    @attenuation_coef.setter
    def attenuation_coef(self, value):
        """Set the attenuation coefficient in dB/cm/MHz."""
        value = float(value)
        assert value >= 0.0, "Attenuation coefficient must be non-negative"
        self._attenuation_coef = value

    @property
    def theta_range(self):
        """The theta range for scan conversion."""
        if self._theta_range is None and self.polar_angles is not None:
            self._theta_range = (self.polar_angles.min(), self.polar_angles.max())
        return self._theta_range

    @property
    def coordinates(self):
        """The coordinates for scan conversion."""
        if self._coordinates is not None:
            return self._coordinates

        # If rho_range or theta_range is not set, return None
        if self.rho_range is None or self.theta_range is None:
            return None

        # If phi_range is set, use 3D scan conversion
        if self.phi_range is not None:
            self._coordinates, _ = compute_scan_convert_3d_coordinates(
                (self.Nz, self.Nx),
                self.rho_range,
                self.theta_range,
                self.phi_range,
                self.resolution,
            )

        # If phi_range is not set, use 2D scan conversion
        else:
            self._coordinates, _ = compute_scan_convert_2d_coordinates(
                (self.Nz, self.Nx),
                self.rho_range,
                self.theta_range,
                self.resolution,
            )
        return self._coordinates

    @property
    def time_to_next_transmit(self):
        """The time between subsequent transmit events of shape (n_frames, n_tx)."""
        if self._time_to_next_transmit is not None:
            return self._time_to_next_transmit[:, self.selected_transmits]
        else:
            return None

    @property
    def frames_per_second(self):
        """The number of frames per second [Hz]. Assumes a constant frame rate.

        Frames per second computed based on time between transmits within a frame.
        Ignores time between frames (e.g. due to processing).

        Uses the time it took to do all transmits (per frame). So if you only use some portion
        of the transmits, the fps will still be calculated based on all.
        """
        if self._time_to_next_transmit is None:
            raise ValueError(
                "Please set scan.time_to_next_transmit. Currently not set."
            )

        # Check if fps is constant
        uniq = np.unique(self._time_to_next_transmit, axis=0)  # frame axis
        if uniq.shape[0] != 1:
            log.warning("Time to next transmit is not constant")

        # Compute fps
        time = np.mean(np.sum(self._time_to_next_transmit, axis=1))
        fps = 1 / time
        return fps

    def get_scan_parameters(self):
        """Returns a dictionary with all the parameters of the scan.
        Note that these are the parameters under the currently set selected_transmits.
        """
        return {
            "n_tx": self.n_tx,
            "n_ax": self.n_ax,
            "n_el": self.n_el,
            "center_frequency": self.center_frequency,
            "sampling_frequency": self.sampling_frequency,
            "demodulation_frequency": self.demodulation_frequency,
            "xlims": self.xlims,
            "ylims": self.ylims,
            "zlims": self.zlims,
            "bandwidth_percent": self.bandwidth_percent,
            "sound_speed": self.sound_speed,
            "n_ch": self.n_ch,
            "Nx": self.Nx,
            "Nz": self.Nz,
            "pixels_per_wavelength": self.pixels_per_wavelength,
            "polar_angles": self.polar_angles,
            "azimuth_angles": self.azimuth_angles,
            "t0_delays": self.t0_delays,
            "tx_apodizations": self.tx_apodizations,
            "focus_distances": self.focus_distances,
            "initial_times": self.initial_times,
            "selected_transmits": self.selected_transmits,
            "probe_geometry": self.probe_geometry,
            "time_to_next_transmit": self.time_to_next_transmit,
            "pfield": self.pfield,
            "theta_range": self.theta_range,
            "phi_range": self.phi_range,
            "rho_range": self.rho_range,
            "fill_value": self.fill_value,
        }


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
        pixels_per_wavelength=4,
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
            pixels_per_wavelength (int, optional): The number of pixels per wavelength
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
            time_to_next_transmit (np.ndarray, optional): The time between
                subsequent transmit events of shape (n_frames, n_tx). Defaults to None.

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
            pixels_per_wavelength=pixels_per_wavelength,
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
