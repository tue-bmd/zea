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

.. code-block:: python

    from zea import Config, Probe, Scan

    # Initialize Scan from a Probe's parameters
    probe = Probe.from_name("verasonics_l11_4v")
    scan = Scan(**probe.get_parameters(), Nz=256)

    # Or initialize from a Config object
    config = Config.from_hf("zeahub/configs", "config_picmus_rf.yaml", repo_type="dataset")
    scan = Scan(**config.scan, n_tx=11)

    # Or manually specify parameters
    scan = Scan(
        Nx=128,
        Nz=256,
        xlims=(-0.02, 0.02),
        zlims=(0.0, 0.06),
        center_frequency=6.25e6,
        sound_speed=1540.0,
        sampling_frequency=25e6,
        n_el=128,
        n_tx=11,
    )

    # Access a derived property (computed lazily)
    grid = scan.grid  # shape: (Nz, Nx, 3)

    # Select a subset of transmit events
    scan.set_transmits(3)  # Use 3 evenly spaced transmits
    scan.set_transmits([0, 2, 4])  # Use specific transmit indices
    scan.set_transmits("all")  # Use all transmits

"""

import numpy as np
from keras import ops

from zea import log
from zea.beamform.pfield import compute_pfield
from zea.beamform.pixelgrid import check_for_aliasing, get_grid
from zea.display import compute_scan_convert_2d_coordinates, compute_scan_convert_3d_coordinates
from zea.internal.parameters import Parameters, cache_with_dependencies


class Scan(Parameters):
    """Represents an ultrasound scan configuration with computed properties.

    Args:
        Nx (int): Number of samples in the x-direction (lateral).
        Nz (int): Number of samples in the z-direction (axial).
        sound_speed (float, optional): Speed of sound in the medium in m/s.
            Defaults to 1540.0.
        sampling_frequency (float): Sampling frequency in Hz.
        center_frequency (float): Center frequency of the transducer in Hz.
        n_el (int): Number of elements in the transducer array.
        n_tx (int): Number of transmit events in the dataset.
        n_ax (int): Number of axial samples in the received signal.
        n_ch (int, optional): Number of channels (1 for RF, 2 for IQ data).
            Defaults to 1.
        xlims (tuple of float): Lateral (x) limits of the imaging region in
            meters (min, max).
        ylims (tuple of float, optional): Elevation (y) limits of the imaging
            region in meters (min, max).
        zlims (tuple of float): Axial (z) limits of the imaging region
            in meters (min, max).
        probe_geometry (np.ndarray): Element positions as array of shape (n_el, 3).
        polar_angles (np.ndarray): Polar angles for each transmit event in radians.
        azimuth_angles (np.ndarray): Azimuth angles for each transmit event in radians.
        t0_delays (np.ndarray): Transmit delays in seconds for each event.
        tx_apodizations (np.ndarray): Transmit apodizations for each event.
        focus_distances (np.ndarray): Focus distances in meters for each event.
        initial_times (np.ndarray): Initial times in seconds for each event.
        bandwidth_percent (float, optional): Bandwidth as percentage of center
            frequency. Defaults to 200.0.
        demodulation_frequency (float, optional): Demodulation frequency in Hz.
        time_to_next_transmit (np.ndarray): Time between transmit events.
        pixels_per_wavelength (int, optional): Number of pixels per wavelength.
            Defaults to 4.
        downsample (int, optional): Downsampling factor for the data. Defaults to 1.
        element_width (float, optional): Width of each transducer element in meters.
            Defaults to 0.2e-3.
        resolution (float, optional): Desired spatial resolution in meters.
        pfield_kwargs (dict, optional): Additional parameters for pressure field computation.
            See `zea.beamform.pfield.compute_pfield` for details.
        apply_lens_correction (bool, optional): Whether to apply lens correction to
            delays. Defaults to False.
        lens_thickness (float, optional): Thickness of the lens in meters.
            Defaults to None.
        f_number (float, optional): F-number of the transducer. Defaults to 1.0.
        theta_range (tuple, optional): Range of theta angles for 3D imaging.
        phi_range (tuple, optional): Range of phi angles for 3D imaging.
        rho_range (tuple, optional): Range of rho (radial) distances for 3D imaging.
        fill_value (float, optional): Value to use for out-of-bounds pixels.
            Defaults to 0.0.
        attenuation_coef (float, optional): Attenuation coefficient in dB/(MHz*cm).
            Defaults to 0.0.
        selected_transmits (None, str, int, list, or np.ndarray, optional):
            Specifies which transmit events to select.
            - None or "all": Use all transmits.
            - "center": Use only the center transmit.
            - int: Select this many evenly spaced transmits.
            - list/array: Use these specific transmit indices.
    """

    VALID_PARAMS = {
        # beamforming related parameters
        "Nx": {"type": int, "default": None},
        "Nz": {"type": int, "default": None},
        "xlims": {"type": (tuple, list), "default": None},
        "ylims": {"type": (tuple, list), "default": None},
        "zlims": {"type": (tuple, list), "default": None},
        "pixels_per_wavelength": {"type": int, "default": 4},
        "downsample": {"type": int, "default": 1},
        "resolution": {"type": float, "default": None},
        "pfield_kwargs": {"type": dict, "default": {}},
        "apply_lens_correction": {"type": bool, "default": False},
        "lens_sound_speed": {"type": (float, int), "default": None},
        "lens_thickness": {"type": float, "default": None},
        # acquisition parameters
        "sound_speed": {"type": (float, int), "default": 1540.0},
        "sampling_frequency": {"type": float, "default": None},
        "center_frequency": {"type": float, "default": None},
        "n_el": {"type": int, "default": None},
        "n_tx": {"type": int, "default": None},
        "n_ax": {"type": int, "default": None},
        "n_ch": {"type": int, "default": None},
        "bandwidth_percent": {"type": float, "default": 200.0},
        "demodulation_frequency": {"type": float, "default": None},
        "element_width": {"type": float, "default": 0.2e-3},
        "attenuation_coef": {"type": float, "default": 0.0},
        "f_number": {"type": float, "default": 1.0},
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
        "theta_range": {"type": (tuple, list), "default": None},
        "phi_range": {"type": (tuple, list), "default": None},
        "rho_range": {"type": (tuple, list), "default": None},
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

    @cache_with_dependencies(
        "xlims",
        "zlims",
        "Nx",
        "Nz",
        "sound_speed",
        "center_frequency",
        "pixels_per_wavelength",
    )
    def grid(self):
        """The beamforming grid of shape (Nz, Nx, 3)."""
        return get_grid(
            self.xlims,
            self.zlims,
            self.Nx,
            self.Nz,
            self.sound_speed,
            self.center_frequency,
            self.pixels_per_wavelength,
        )

    @cache_with_dependencies(
        "xlims",
        "wavelength",
    )
    def Nx(self):
        """Number of lateral (x) pixels, set to prevent aliasing if not provided."""
        Nx = self._params.get("Nx")
        if Nx is not None:
            return Nx

        width = self.xlims[1] - self.xlims[0]
        min_Nx = int(np.ceil(width / (self.wavelength / 2)))
        return max(min_Nx, 1)

    @cache_with_dependencies(
        "zlims",
        "wavelength",
    )
    def Nz(self):
        """Number of axial (z) pixels, set to prevent aliasing if not provided."""
        Nz = self._params.get("Nz")
        if Nz is not None:
            return Nz

        depth = self.zlims[1] - self.zlims[0]
        min_Nz = int(np.ceil(depth / (self.wavelength / 2)))
        return max(min_Nz, 1)

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

    @cache_with_dependencies("n_tx")
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

    @cache_with_dependencies("n_tx")
    def n_tx_total(self):
        """The total number of transmits in the full dataset."""
        return self._params["n_tx"]

    @cache_with_dependencies("selected_transmits")
    def n_tx(self):
        """The number of currently selected transmits."""
        return len(self.selected_transmits)

    def _invalidate_selected_transmits(self):
        """Explicitly invalidate the selected_transmits cache and its dependents."""
        for key in [
            "selected_transmits",
            # also invalidate properties that depend on selected_transmits
            "polar_angles",
            "azimuth_angles",
            "t0_delays",
            "tx_apodizations",
            "focus_distances",
            "initial_times",
            "time_to_next_transmit",
            "pfield",
            "flat_pfield",
        ]:
            if key in self._cache:
                self._cache.pop(key)
                self._computed.discard(key)
                self._dependency_versions.pop(key, None)

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
                raise ValueError(f"Transmit indices must be between 0 and {n_tx_total - 1}")

            self._selected_transmits = [
                int(i) for i in selection
            ]  # Convert numpy integers to Python ints
            self._invalidate_selected_transmits()
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
        """The polar angles of transmits in radians."""
        value = self._params.get("polar_angles")
        if value is None:
            log.warning("No polar angles provided, using zeros")
            value = np.zeros(self.n_tx_total)

        return value[self.selected_transmits]

    @cache_with_dependencies("selected_transmits")
    def azimuth_angles(self):
        """The azimuth angles of transmits in radians."""
        value = self._params.get("azimuth_angles")
        if value is None:
            log.warning("No azimuth angles provided, using zeros")
            value = np.zeros(self.n_tx_total)

        return value[self.selected_transmits]

    @cache_with_dependencies("selected_transmits")
    def t0_delays(self):
        """The transmit delays in seconds."""
        value = self._params.get("t0_delays")
        if value is None:
            log.warning("No transmit delays provided, using zeros")
            return np.zeros((self.n_tx_total, self.n_el))

        return value[self.selected_transmits]

    @cache_with_dependencies("selected_transmits")
    def tx_apodizations(self):
        """The transmit apodizations."""
        value = self._params.get("tx_apodizations")
        if value is None:
            log.warning("No transmit apodizations provided, using ones")
            value = np.ones((self.n_tx_total, self.n_el))

        return value[self.selected_transmits]

    @cache_with_dependencies("selected_transmits")
    def focus_distances(self):
        """The focus distances in meters."""
        value = self._params.get("focus_distances")
        if value is None:
            log.warning("No focus distances provided, using zeros")
            value = np.zeros(self.n_tx_total)

        return value[self.selected_transmits]

    @cache_with_dependencies("selected_transmits")
    def initial_times(self):
        """The initial times in seconds."""
        value = self._params.get("initial_times")
        if value is None:
            log.warning("No initial times provided, using zeros")
            value = np.zeros(self.n_tx_total)

        return value[self.selected_transmits]

    @cache_with_dependencies("selected_transmits")
    def time_to_next_transmit(self):
        """Time between transmit events."""
        value = self._params.get("time_to_next_transmit")
        if value is None:
            return None

        selected = self.selected_transmits
        return value[:, selected]

    @cache_with_dependencies(
        "sound_speed",
        "center_frequency",
        "bandwidth_percent",
        "n_el",
        "probe_geometry",
        "tx_apodizations",
        "grid",
        "t0_delays",
    )
    def pfield(self):
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
        """Flattened pfield for weighting."""
        return self.pfield.reshape(self.n_tx, -1).swapaxes(0, 1)

    @cache_with_dependencies("rho_range", "theta_range", "resolution", "Nz", "Nx")
    def coordinates_2d(self):
        """The coordinates for scan conversion."""
        coords, _ = compute_scan_convert_2d_coordinates(
            (self.Nz, self.Nx),
            self.rho_range,
            self.theta_range,
            self.resolution,
        )
        return coords

    @cache_with_dependencies("rho_range", "theta_range", "phi_range", "resolution", "Nz", "Nx")
    def coordinates_3d(self):
        """The coordinates for scan conversion."""
        coords, _ = compute_scan_convert_3d_coordinates(
            (self.Nz, self.Nx),
            self.rho_range,
            self.theta_range,
            self.phi_range,
            self.resolution,
        )
        return coords

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
