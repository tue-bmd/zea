"""Structure containing parameters defining an ultrasound scan.

This module provides the :class:`Scan` class, a flexible structure
for managing all parameters related to an ultrasound scan acquisition.

Features
^^^^^^^^

- **Flexible initialization:** The :class:`Scan` class supports lazy initialization,
  allowing you to specify any combination of supported parameters. You can pass only
  the parameters you have, and the rest will be computed or set to defaults as needed.

- **Automatic computation:** Many scan properties (such as
  grid, number of pixels, wavelength, etc.) are computed automatically from the
  provided parameters. This enables you to work with minimal input and still obtain
  all necessary scan configuration details.

- **Dependency tracking and lazy evaluation:** Derived properties are computed only
  when accessed, and are automatically invalidated and recomputed if their dependencies
  change. This ensures efficient memory usage and avoids unnecessary computations.

- **Parameter validation:** All parameters are type-checked and validated against
  a predefined schema, reducing errors and improving robustness.

- **Selection of transmits:** The scan supports flexible selection of transmit events,
  using the :meth:`set_transmits` method. You can select all, a specific number,
  or specific transmit indices. The selection is stored and can be accessed via
  the :attr:`selected_transmits` property.

Comparison to ``zea.Config`` and ``zea.Probe``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :class:`zea.config.Config`: A general-purpose parameter dictionary for experiment and pipeline
  configuration. It is not specific to ultrasound acquisition and does not compute
  derived parameters.

- :class:`zea.probes.Probe`: Contains only probe-specific parameters (e.g., geometry, frequency).

- :class:`zea.scan.Scan`: Combines all parameters relevant to an ultrasound acquisition,
  including probe, acquisition, and scan region. It also provides automatic computation
  of derived properties and dependency management.

Example Usage
^^^^^^^^^^^^^

.. doctest::

    >>> from zea import Config, Probe, Scan

    >>> # Initialize Scan from a Probe's parameters
    >>> probe = Probe.from_name("verasonics_l11_4v")
    >>> scan = Scan(**probe.get_parameters(), grid_size_z=256, n_tx=11)

    >>> # Or initialize from a Config object
    >>> config = Config.from_hf("zeahub/configs", "config_picmus_rf.yaml", repo_type="dataset")
    >>> scan = Scan(n_tx=11, **config.scan)

    >>> # Or manually specify parameters
    >>> scan = Scan(
    ...     grid_size_x=128,
    ...     grid_size_z=256,
    ...     xlims=(-0.02, 0.02),
    ...     zlims=(0.0, 0.06),
    ...     center_frequency=6.25e6,
    ...     sound_speed=1540.0,
    ...     sampling_frequency=25e6,
    ...     n_el=128,
    ...     n_tx=11,
    ... )

    >>> # Access a derived property (computed lazily)
    >>> grid = scan.grid  # shape: (grid_size_z, grid_size_x, 3)

    >>> # Select a subset of transmit events
    >>> _ = scan.set_transmits(3)  # Use 3 evenly spaced transmits
    >>> _ = scan.set_transmits([0, 2, 4])  # Use specific transmit indices
    >>> _ = scan.set_transmits("all")  # Use all transmits

"""

import numpy as np
from keras import ops

from zea import log
from zea.beamform.pfield import compute_pfield
from zea.beamform.pixelgrid import (
    cartesian_pixel_grid,
    check_for_aliasing,
    polar_pixel_grid,
)
from zea.display import (
    compute_scan_convert_2d_coordinates,
    compute_scan_convert_3d_coordinates,
)
from zea.internal.core import DEFAULT_DYNAMIC_RANGE
from zea.internal.parameters import Parameters, cache_with_dependencies


class Scan(Parameters):
    """Represents an ultrasound scan configuration with computed properties.

    Args:
        grid_size_x (int): Grid width in pixels. For a cartesian grid, this is the lateral (x)
            pixels in the grid, set to prevent aliasing if not provided. For a polar grid, this can
            be thought of as the number for rays in the polar direction.
        grid_size_z (int): Grid height in pixels. This is the number of axial (z) pixels in the
            grid, set to prevent aliasing if not provided.
        sound_speed (float, optional): Speed of sound in the medium in m/s.
            Defaults to 1540.0.
        sampling_frequency (float): Sampling frequency in Hz.
        center_frequency (float): Center frequency of the transducer in Hz.
        n_el (int): Number of elements in the transducer array.
        n_tx (int): Number of transmit events in the dataset.
        n_ax (int): Number of axial samples in the received signal.
        n_ch (int, optional): Number of channels (1 for RF, 2 for IQ data).
        xlims (tuple of float): Lateral (x) limits of the imaging region in
            meters (min, max).
        ylims (tuple of float, optional): Elevation (y) limits of the imaging
            region in meters (min, max).
        zlims (tuple of float): Axial (z) limits of the imaging region
            in meters (min, max).
        probe_geometry (np.ndarray): Element positions as array of shape (n_el, 3).
        polar_angles (np.ndarray): Polar angles for each transmit event in radians of shape (n_tx,).
            These angles are often used in 2D imaging.
        azimuth_angles (np.ndarray): Azimuth angles for each transmit event in radians
            of shape (n_tx,). These angles are often used in 3D imaging.
        t0_delays (np.ndarray): Transmit delays in seconds of
            shape (n_tx, n_el), shifted such that the smallest delay is 0.
        tx_apodizations (np.ndarray): Transmit apodizations of shape (n_tx, n_el).
        focus_distances (np.ndarray): Focus distances in meters for each event of shape (n_tx,).
        initial_times (np.ndarray): Initial times in seconds for each event of shape (n_tx,).
        bandwidth_percent (float, optional): Bandwidth as percentage of center
            frequency. Defaults to 200.0.
        demodulation_frequency (float, optional): Demodulation frequency in Hz.
        time_to_next_transmit (np.ndarray): The time between subsequent
            transmit events of shape (n_frames, n_tx).
        tgc_gain_curve (np.ndarray): Time gain compensation (TGC) curve of shape (n_ax,).
        waveforms_one_way (np.ndarray): The one-way transmit waveforms of shape
            (n_waveforms, n_samples).
        waveforms_two_way (np.ndarray): The two-way transmit waveforms of shape
            (n_waveforms, n_samples).
        tx_waveform_indices (np.ndarray): Indices of the waveform used for each
            transmit event of shape (n_tx,).
        t_peak (np.ndarray, optional): The time of the peak of the pulse of every transmit waveform
            of shape (n_waveforms,).
        pixels_per_wavelength (int, optional): Number of pixels per wavelength.
            Defaults to 4.
        element_width (float, optional): Width of each transducer element in meters.
        resolution (float, optional): Resolution for scan conversion in mm / pixel.
            If None, it is calculated based on the input image.
        pfield_kwargs (dict, optional): Additional parameters for pressure field computation.
            See `zea.beamform.pfield.compute_pfield` for details.
        apply_lens_correction (bool, optional): Whether to apply lens correction to
            delays. Defaults to False.
        lens_thickness (float, optional): Thickness of the lens in meters.
        f_number (float, optional): F-number of the transducer. Defaults to 1.0.
        theta_range (tuple, optional): Range of theta angles for 3D imaging.
        phi_range (tuple, optional): Range of phi angles for 3D imaging.
        rho_range (tuple, optional): Range of rho (radial) distances for 3D imaging.
        fill_value (float, optional): Value to use for out-of-bounds pixels.
        attenuation_coef (float, optional): Attenuation coefficient in dB/(MHz*cm).
            Defaults to 0.0.
        selected_transmits (None, str, int, list, slice, or np.ndarray, optional):
            Specifies which transmit events to select.
            - None or "all": Use all transmits.
            - "center": Use only the center transmit.
            - int: Select this many evenly spaced transmits.
            - list/array: Use these specific transmit indices.
            - slice: Use transmits specified by the slice (e.g., slice(0, 10, 2)).
        grid_type (str, optional): Type of grid to use for beamforming.
            Can be "cartesian" or "polar". Defaults to "cartesian".
        dynamic_range (tuple, optional): Dynamic range for image display.
            Defined in dB as (min_dB, max_dB). Defaults to (-60, 0).

    """

    VALID_PARAMS = {
        # beamforming related parameters
        "grid_size_x": {"type": int},
        "grid_size_z": {"type": int},
        "xlims": {"type": (tuple, list)},
        "ylims": {"type": (tuple, list)},
        "zlims": {"type": (tuple, list)},
        "pixels_per_wavelength": {"type": int, "default": 4},
        "pfield_kwargs": {"type": dict, "default": {}},
        "apply_lens_correction": {"type": bool, "default": False},
        "lens_sound_speed": {"type": float},
        "lens_thickness": {"type": float},
        "grid_type": {"type": str, "default": "cartesian"},
        "polar_limits": {"type": (tuple, list)},
        "dynamic_range": {"type": (tuple, list), "default": DEFAULT_DYNAMIC_RANGE},
        "selected_transmits": {"type": (type(None), str, int, list, slice, np.ndarray)},
        # acquisition parameters
        "sound_speed": {"type": float, "default": 1540.0},
        "sampling_frequency": {"type": float},
        "center_frequency": {"type": float},
        "n_el": {"type": int},
        "n_tx": {"type": int},
        "n_ax": {"type": int},
        "n_ch": {"type": int},
        "bandwidth_percent": {"type": float, "default": 200.0},
        "demodulation_frequency": {"type": float},
        "element_width": {"type": float},
        "attenuation_coef": {"type": float, "default": 0.0},
        "f_number": {"type": float, "default": 1.0},
        # array parameters
        "probe_geometry": {"type": np.ndarray},
        "polar_angles": {"type": np.ndarray},
        "azimuth_angles": {"type": np.ndarray},
        "t0_delays": {"type": np.ndarray},
        "tx_apodizations": {"type": np.ndarray},
        "focus_distances": {"type": np.ndarray},
        "initial_times": {"type": np.ndarray},
        "time_to_next_transmit": {"type": np.ndarray},
        "tgc_gain_curve": {"type": np.ndarray},
        "waveforms_one_way": {"type": np.ndarray, "default": None},
        "waveforms_two_way": {"type": np.ndarray, "default": None},
        "tx_waveform_indices": {"type": np.ndarray},
        "t_peak": {"type": np.ndarray},
        # scan conversion parameters
        "theta_range": {"type": (tuple, list)},
        "phi_range": {"type": (tuple, list)},
        "rho_range": {"type": (tuple, list)},
        "fill_value": {"type": float},
        "resolution": {"type": float, "default": None},
    }

    def __init__(self, **kwargs):
        # Ensure that selected_transmits is present and set to None by default
        selected_transmits_input = kwargs.get("selected_transmits", None)
        kwargs["selected_transmits"] = None

        # Initialize parent class
        super().__init__(**kwargs)

        # Initialize selection to None
        self._selected_transmits = None

        # Apply selection from input if provided
        if selected_transmits_input is not None:
            self.set_transmits(selected_transmits_input)

    @cache_with_dependencies("xlims", "zlims", "grid_size_x", "grid_size_z", "grid_type")
    def grid(self):
        """The beamforming grid of shape (grid_size_z, grid_size_x, 3)."""
        if self.grid_type == "polar":
            return polar_pixel_grid(
                self.polar_limits, self.zlims, self.grid_size_z, self.grid_size_x
            )
        elif self.grid_type == "cartesian":
            return cartesian_pixel_grid(
                self.xlims,
                self.zlims,
                grid_size_z=self.grid_size_z,
                grid_size_x=self.grid_size_x,
            )
        else:
            raise ValueError(
                f"Unsupported grid type: {self.grid_type}. Supported types are "
                "'cartesian' and 'polar'."
            )

    @cache_with_dependencies("xlims", "wavelength", "pixels_per_wavelength")
    def grid_size_x(self):
        """Grid width in pixels. For a cartesian grid, this is the lateral (x) pixels in the grid,
        set to prevent aliasing if not provided. For a polar grid, this can be thought of as
        the number for rays in the polar direction.
        """
        grid_size_x = self._params.get("grid_size_x")
        if grid_size_x is not None:
            return grid_size_x

        width = self.xlims[1] - self.xlims[0]
        min_grid_size_x = int(np.ceil(width / (self.wavelength / self.pixels_per_wavelength)))
        return max(min_grid_size_x, 1)

    @cache_with_dependencies(
        "zlims",
        "wavelength",
        "pixels_per_wavelength",
    )
    def grid_size_z(self):
        """Grid height in pixels. This is the number of axial (z) pixels in the grid,
        set to prevent aliasing if not provided."""
        grid_size_z = self._params.get("grid_size_z")
        if grid_size_z is not None:
            return grid_size_z

        depth = self.zlims[1] - self.zlims[0]
        min_grid_size_z = int(np.ceil(depth / (self.wavelength / self.pixels_per_wavelength)))
        return max(min_grid_size_z, 1)

    @cache_with_dependencies("sound_speed", "center_frequency")
    def wavelength(self):
        """Calculate the wavelength based on sound speed and center frequency."""
        return self.sound_speed / self.center_frequency

    @cache_with_dependencies("zlims", "polar_limits", "probe_geometry")
    def xlims(self):
        """The x-limits of the beamforming grid [m]. If not explicitly set, it is computed based
        on the polar limits and probe geometry.
        """
        xlims = self._params.get("xlims")
        if xlims is None:
            radius = max(self.zlims)
            xlims_polar = (
                radius * np.cos(-np.pi / 2 + self.polar_limits[0]),
                radius * np.cos(-np.pi / 2 + self.polar_limits[1]),
            )
            xlims_plane = (self.probe_geometry[0, 0], self.probe_geometry[-1, 0])
            xlims = (
                min(xlims_polar[0], xlims_plane[0]),
                max(xlims_polar[1], xlims_plane[1]),
            )
        return xlims

    @cache_with_dependencies("sound_speed", "sampling_frequency", "n_ax")
    def zlims(self):
        """The z-limits of the beamforming grid [m]."""
        zlims = self._params.get("zlims")
        if zlims is None:
            return [0, self.sound_speed * self.n_ax / self.sampling_frequency / 2]
        return zlims

    @cache_with_dependencies("xlims", "zlims")
    def extent(self):
        """The extent of the beamforming grid in the format (xmin, xmax, zmax, zmin).
        Can be directly used with `plt.imshow(x, extent=scan.extent)` for visualization.
        """
        return np.array([self.xlims[0], self.xlims[1], self.zlims[1], self.zlims[0]])

    @cache_with_dependencies("grid")
    def flatgrid(self):
        """The beamforming grid of shape (grid_size_z*grid_size_x, 3)."""
        return self.grid.reshape(-1, 3)

    @property
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
        return self._params["n_tx"]

    @property
    def n_tx_selected(self):
        """The number of currently selected transmits."""
        return len(self.selected_transmits)

    @cache_with_dependencies("selected_transmits")
    def n_tx(self):
        """The number of currently selected transmits."""
        return len(self.selected_transmits)

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
                - slice: Use transmits specified by the slice (e.g., slice(0, 10, 2))

        Returns:
            The current instance for method chaining.

        Raises:
            ValueError: If the selection is invalid or incompatible with the scan.
        """
        n_tx_total = self._params.get("n_tx")
        if n_tx_total is None:
            raise ValueError("n_tx must be set before calling set_transmits")

        # Handle array-like - convert to list of indices
        if isinstance(selection, np.ndarray):
            if len(selection.shape) == 0:
                # Handle scalar numpy array
                return self.set_transmits(int(selection))
            elif len(selection.shape) == 1:
                selection = selection.tolist()
            else:
                raise ValueError(f"Invalid array shape: {selection.shape}")

        # Handle None and "all" - use all transmits
        if selection is None or selection == "all":
            self._selected_transmits = None
            self._invalidate("selected_transmits")
            return self

        # Handle "center" - use center transmit
        if selection == "center":
            self._selected_transmits = [n_tx_total // 2]
            self._invalidate("selected_transmits")
            return self

        # Handle integer - select evenly spaced transmits
        if isinstance(selection, (int, np.integer)):
            selection = int(selection)  # Convert numpy integer to Python int
            if selection <= 0:
                raise ValueError(f"Number of transmits must be positive, got {selection}")

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

            self._invalidate("selected_transmits")
            return self

        # Handle slice - convert to list of indices
        if isinstance(selection, slice):
            selection = list(range(n_tx_total))[selection]

        # Handle list of indices
        if isinstance(selection, list):
            # Validate indices
            if not all(isinstance(i, (int, np.integer)) for i in selection):
                raise ValueError("All transmit indices must be integers")

            if any(i < 0 or i >= n_tx_total for i in selection):
                raise ValueError(f"Transmit indices must be between 0 and {n_tx_total - 1}")

            self._selected_transmits = [
                int(i) for i in selection
            ]  # Convert numpy integers to Python ints
            self._invalidate("selected_transmits")
            return self

        # Aliasing check
        check_for_aliasing(self)

        raise ValueError(f"Unsupported selection type: {type(selection)}")

    @cache_with_dependencies("n_ch", "center_frequency")
    def demodulation_frequency(self):
        """The demodulation frequency."""
        if self._params.get("demodulation_frequency") is not None:
            return self._params["demodulation_frequency"]

        # Default behavior based on n_ch
        return self.center_frequency if self.n_ch == 2 else 0.0

    @cache_with_dependencies("selected_transmits")
    def polar_angles(self):
        """Polar angles for each transmit event in radians of shape (n_tx,).
        These angles are often used in 2D imaging."""
        value = self._params.get("polar_angles")
        if value is None:
            return None

        return value[self.selected_transmits]

    @cache_with_dependencies("polar_angles")
    def polar_limits(self):
        """The limits of the polar angles."""
        value = self._params.get("polar_limits")
        if value is None and self.polar_angles is not None:
            value = self.polar_angles.min(), self.polar_angles.max()
            diff = value[1] - value[0]
            # add 15% margin to the limits
            value = (value[0] - 0.15 * diff, value[1] + 0.15 * diff)
        return value

    @cache_with_dependencies("selected_transmits")
    def azimuth_angles(self):
        """Azimuth angles for each transmit event in radians
        of shape (n_tx,). These angles are often used in 3D imaging."""
        value = self._params.get("azimuth_angles")
        if value is None:
            log.warning("No azimuth angles provided, using zeros")
            return np.zeros(self.n_tx_selected)

        return value[self.selected_transmits]

    @cache_with_dependencies("selected_transmits", "n_el")
    def t0_delays(self):
        """Transmit delays in seconds of
        shape (n_tx, n_el), shifted such that the smallest delay is 0."""
        value = self._params.get("t0_delays")
        if value is None:
            log.warning("No transmit delays provided, using zeros")
            return np.zeros((self.n_tx_selected, self.n_el))

        return value[self.selected_transmits]

    @cache_with_dependencies("selected_transmits")
    def tx_apodizations(self):
        """Transmit apodizations of shape (n_tx, n_el)."""
        value = self._params.get("tx_apodizations")
        if value is None:
            log.warning("No transmit apodizations provided, using ones")
            return np.ones((self.n_tx_selected, self.n_el))

        return value[self.selected_transmits]

    @cache_with_dependencies("selected_transmits")
    def focus_distances(self):
        """Focus distances in meters for each event of shape (n_tx,)."""
        value = self._params.get("focus_distances")
        if value is None:
            log.warning("No focus distances provided, using zeros")
            return np.zeros(self.n_tx_selected)

        return value[self.selected_transmits]

    @cache_with_dependencies("selected_transmits")
    def initial_times(self):
        """Initial times in seconds for each event of shape (n_tx,)."""
        value = self._params.get("initial_times")
        if value is None:
            log.warning("No initial times provided, using zeros")
            return np.zeros(self.n_tx_selected)

        return value[self.selected_transmits]

    @property
    def n_waveforms(self):
        """The number of unique transmit waveforms."""

        if self.waveforms_one_way is not None:
            return self.waveforms_one_way.shape[0]

        if self.waveforms_two_way is not None:
            return self.waveforms_two_way.shape[0]

        return 1

    @property
    def t_peak(self):
        """The time of the peak of the pulse in seconds of shape (n_waveforms,)."""
        t_peak = self._params.get("t_peak")
        if t_peak is None:
            t_peak = np.array([1 / self.center_frequency] * self.n_waveforms)

        return t_peak

    @cache_with_dependencies("selected_transmits")
    def time_to_next_transmit(self):
        """The time between subsequent transmit events of shape (n_frames, n_tx)."""
        value = self._params.get("time_to_next_transmit")
        if value is None:
            return None

        selected = self.selected_transmits
        return value[:, selected]

    @cache_with_dependencies("n_ax")
    def tgc_gain_curve(self):
        """Time gain compensation (TGC) curve of shape (n_ax,)."""
        value = self._params.get("tgc_gain_curve")
        if value is None:
            return np.ones(self.n_ax)
        return value[: self.n_ax]

    @cache_with_dependencies("selected_transmits")
    def tx_waveform_indices(self):
        """Indices of the waveform used for each transmit event of shape (n_tx,)."""
        value = self._params.get("tx_waveform_indices")
        if value is None:
            return np.zeros(self.n_tx_selected, dtype=int)

        return value[self.selected_transmits]

    @cache_with_dependencies(
        "sound_speed",
        "center_frequency",
        "bandwidth_percent",
        "n_el",
        "probe_geometry",
        "tx_apodizations",
        "grid",
        "t0_delays",
        "pfield_kwargs",
    )
    def pfield(self) -> np.ndarray:
        """Compute or return the pressure field (pfield) for weighting."""
        pfield = compute_pfield(
            sound_speed=self.sound_speed,
            center_frequency=self.center_frequency,
            bandwidth_percent=self.bandwidth_percent,
            n_el=self.n_el,
            probe_geometry=self.probe_geometry,
            tx_apodizations=self.tx_apodizations,
            grid=self.grid,
            t0_delays=self.t0_delays,
            **self.pfield_kwargs,
        )
        return ops.convert_to_numpy(pfield)

    @cache_with_dependencies("pfield")
    def flat_pfield(self):
        """Flattened pfield for weighting of shape (n_pix, n_tx)."""
        return self.pfield.reshape(self.n_tx, -1).swapaxes(0, 1)

    @cache_with_dependencies("zlims")
    def rho_range(self):
        """A tuple specifying the range of rho values (min_rho, max_rho). Defined in mm.
        Used for scan conversion."""
        value = self._params.get("rho_range")
        if value is None:
            return self.zlims
        return value

    @cache_with_dependencies("polar_limits")
    def theta_range(self):
        """A tuple specifying the range of theta values (min_theta, max_theta).
        Defined in radians. Used for scan conversion."""
        value = self._params.get("theta_range")
        if value is None and self.polar_limits is not None:
            return self.polar_limits
        return value

    @cache_with_dependencies("rho_range", "theta_range", "resolution", "grid_size_z", "grid_size_x")
    def coordinates_2d(self):
        """The coordinates for scan conversion."""
        coords, _ = compute_scan_convert_2d_coordinates(
            (self.grid_size_z, self.grid_size_x),
            self.rho_range,
            self.theta_range,
            self.resolution,
        )
        return coords

    @cache_with_dependencies(
        "rho_range",
        "theta_range",
        "phi_range",
        "resolution",
        "grid_size_z",
        "grid_size_x",
    )
    def coordinates_3d(self):
        """The coordinates for scan conversion."""
        coords, _ = compute_scan_convert_3d_coordinates(
            (self.grid_size_z, self.grid_size_x),
            self.rho_range,
            self.theta_range,
            self.phi_range,
            self.resolution,
        )
        return coords

    @property
    def coordinates(self):
        """Get the coordinates for scan conversion, will be 3D if phi_range is set,
        otherwise 2D."""
        return self.coordinates_3d if getattr(self, "phi_range", None) else self.coordinates_2d

    @property
    def pulse_repetition_frequency(self):
        """The pulse repetition frequency (PRF) [Hz]. Assumes a constant PRF."""
        if self.time_to_next_transmit is None:
            log.warning("Time to next transmit is not set, cannot compute PRF")
            return None

        pulse_repetition_interval = np.mean(self.time_to_next_transmit)

        return 1 / pulse_repetition_interval

    @cache_with_dependencies("time_to_next_transmit")
    def frames_per_second(self):
        """The number of frames per second [Hz]. Assumes a constant frame rate.

        Frames per second computed based on time between transmits within a frame.
        Ignores time between frames (e.g. due to processing).

        Uses the time it took to do all transmits (per frame). So if you only use some portion
        of the transmits, the fps will still be calculated based on all.
        """
        if self.time_to_next_transmit is None:
            log.warning("Time to next transmit is not set, cannot compute fps")
            return None

        # Check if fps is constant
        uniq = np.unique(self.time_to_next_transmit, axis=0)  # frame axis
        if uniq.shape[0] != 1:
            log.warning("Time to next transmit is not constant")

        # Compute fps
        time = np.mean(np.sum(self.time_to_next_transmit, axis=1))
        fps = 1 / time
        return fps

    @cache_with_dependencies("probe_geometry")
    def element_width(self):
        """The width of each transducer element in meters."""
        value = self._params.get("element_width")
        if value is None:
            # assume uniform spacing
            return np.linalg.norm(self.probe_geometry[1] - self.probe_geometry[0])
        return value

    def __setattr__(self, key, value):
        if key == "selected_transmits":
            # If setting selected_transmits, call set_transmits to handle logic
            self.set_transmits(value)
            return super().__setattr__(key, self.selected_transmits)
        return super().__setattr__(key, value)
