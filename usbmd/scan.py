"""Class structures containing parameters defining an ultrasound scan and the
beamforming grid.

- **Author(s)**     : Vincent van de Schaft, Tristan Stevens
- **Date**          : Wed Feb 15 2024
"""

# pylint: disable=no-member

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from keras import ops

from usbmd.core import STATIC, Object
from usbmd.utils import log
from usbmd.utils.pfield import compute_pfield
from usbmd.utils.pixelgrid import check_for_aliasing, get_grid

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


class Scan(Object):
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
        pixels_per_wvln: int = 4,
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
                Defaults to (0, n_ax * sound_speed / fs / 2).
            bandwidth_percent: Receive bandwidth of RF signal in % of center
                frequency. Not necessarily the same as probe bandwidth. Defaults to 200.
            sound_speed (float, optional): The speed of sound in m/s. Defaults to 1540.
            Nx (int, optional): The number of pixels in the lateral direction
                in the beamforming grid. Defaults to None.
            Nz (int, optional): The number of pixels in the axial direction in
                the beamforming grid. Defaults to None.
            pixels_per_wvln (int, optional): The number of pixels per wavelength
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
            selected_transmits (int, list[int], str, optional): Used to select a subset
                of the transmits. Defaults to 0.2e-3.
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
                transmit events of shape (n_tx*n_frames,). Defaults to None.

        """
        super().__init__()

        # Dictionary to track which parameters have been set
        self._set_params = {}

        # Store values and mark as set if not None
        # Basic parameters
        self._set_param("n_tx", n_tx)
        self._set_param("n_ax", n_ax)
        self._set_param("n_el", n_el)
        self._set_param("n_ch", n_ch)
        self._set_param("fc", center_frequency)
        self._set_param("fs", sampling_frequency)
        self._set_param("bandwidth_percent", bandwidth_percent)
        self._set_param("sound_speed", sound_speed)
        self._set_param("fdemod", demodulation_frequency)
        self._set_param("xlims", xlims)
        self._set_param("ylims", ylims)
        self._set_param("zlims", zlims)
        self._set_param("Nx", Nx)
        self._set_param("Nz", Nz)

        # Store array values and mark as set if not None
        self._set_param("t0_delays", t0_delays)
        self._set_param("tx_apodizations", tx_apodizations)
        self._set_param("polar_angles", polar_angles)
        self._set_param("azimuth_angles", azimuth_angles)
        self._set_param("focus_distances", focus_distances)
        self._set_param("initial_times", initial_times)
        self._set_param("pfield", pfield)

        if pfield is None:
            self._set_params["flat_pfield"] = False

        self._set_param("grid", None)
        self._set_params["flatgrid"] = False

        if not (self._set_params["sound_speed"] and self._set_params["fc"]):
            self._set_params["wvln"] = False

        if not (self._set_params["zlims"] and self._set_params["n_ax"]):
            self._set_params["z_axis"] = False

        # Additional properties that don't need lazy initialization
        self._set_param("pixels_per_wavelength", float(pixels_per_wvln), dunder=False)
        self._set_param("downsample", downsample, dunder=False)
        self._set_param("probe_geometry", probe_geometry, dunder=False)
        self._set_param("time_to_next_transmit", time_to_next_transmit, dunder=False)
        self._set_param("f_number", f_number, dunder=False)
        self._set_param("pfield_kwargs", pfield_kwargs, dunder=False)
        self._set_param("apply_lens_correction", apply_lens_correction, dunder=False)
        self._set_param("lens_thickness", lens_thickness, dunder=False)
        self._set_param("lens_sound_speed", lens_sound_speed, dunder=False)
        self._set_param("element_width", element_width, dunder=False)
        self._set_param("attenuation_coef", attenuation_coef, dunder=False)
        self._set_param("fill_value", fill_value, dunder=False)
        self._set_param("theta_range", theta_range, dunder=False)
        self._set_param("phi_range", phi_range, dunder=False)
        self._set_param("rho_range", rho_range, dunder=False)

        self.selected_transmits = selected_transmits
        self._static_attrs = STATIC

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

    @selected_transmits.setter
    def selected_transmits(self, value):
        self._selected_transmits = self._select_transmits(value)
        try:
            check_for_aliasing(self)
        except ValueError as e:
            log.warning(f"Error checking for aliasing: {e}")

        self._pfield = None  # also trigger update of the pressure fields

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
        self._fdemod = None  # Reset fdemod
        self._set_params["n_ch"] = True

    @property
    def fc(self):
        """The modulation carrier frequency."""
        if self._fc is None:
            raise ValueError("Please set scan.center_frequency.")
        return float(self._fc)

    @fc.setter
    def fc(self, value):
        self._fc = value
        self._grid = None
        self._set_params["fc"] = True

    @property
    def fs(self):
        """The sampling rate."""
        if self._fs is None:
            raise ValueError("Please set scan.sampling_rate.")
        return float(self._fs)

    @fs.setter
    def fs(self, value):
        self._fs = value
        self._set_params["fs"] = True

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
    def Nx(self):
        """The number of pixels in the lateral direction in the
        beamforming grid.

        If None, the number of pixels is calculated based on the
        `pixels_per_wavelength` parameter. See `usbmd.utils.pixel_grid.get_grid`.
        """
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
        `pixels_per_wavelength` parameter. See `usbmd.utils.pixel_grid.get_grid`.
        """
        return int(self._Nz) if self._Nz is not None else None

    @Nz.setter
    def Nz(self, value):
        self._Nz = int(value)
        self._grid = None

    @property
    def wvln(self):
        """The wavelength of the modulation carrier [m]."""
        return self.sound_speed / self.fc

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
            self.zlims = [0, self.sound_speed * self.n_ax / self.fs / 2]
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
            self._grid = get_grid(self)
            self._Nz, self._Nx, _ = self._grid.shape
            self._pfield = None  # also trigger update of the pressure fields

        return self._grid

    @grid.setter
    def grid(self, value):
        """Manually set the grid."""
        self._grid = value

    @property
    def flatgrid(self):
        """The beamforming grid of shape (Nz*Nx, 3)."""
        if self.grid is None:
            return None
        return self.grid.reshape(-1, 3)

    @property
    def pfield(self):
        """The pfield grid of shape (n_tx, Nz, Nx)."""
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

            self._pfield = np.ones((self.n_tx, self.Nz, self.Nx))
        else:
            if self.pfield_kwargs is None:
                pfield_kwargs = {}
            else:
                pfield_kwargs = self.pfield_kwargs

            self._pfield = ops.convert_to_numpy(compute_pfield(self, **pfield_kwargs))

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

    def get_scan_parameters(self):
        """Returns a dictionary with all the parameters of the scan.
        Note that these are the parameters under the currently set selected_transmits.
        """
        return {
            "n_tx": self.n_tx,
            "n_ax": self.n_ax,
            "n_el": self.n_el,
            "center_frequency": self.fc,
            "sampling_frequency": self.fs,
            "demodulation_frequency": self.fdemod,
            "xlims": self.xlims,
            "ylims": self.ylims,
            "zlims": self.zlims,
            "bandwidth_percent": self.bandwidth_percent,
            "sound_speed": self.sound_speed,
            "n_ch": self.n_ch,
            "Nx": self.Nx,
            "Nz": self.Nz,
            "pixels_per_wvln": self.pixels_per_wavelength,
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
        pixels_per_wvln=4,
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
