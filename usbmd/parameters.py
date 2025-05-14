import functools
import hashlib
import time

import keras
import numpy as np


def cache_with_dependencies(*deps):
    def decorator(func):
        func._dependencies = deps

        @functools.wraps(func)
        def wrapper(self):
            failed = set()
            if not self._resolve_dependency_tree(func.__name__, failed):
                raise AttributeError(
                    f"Cannot access '{func.__name__}' due to missing base dependencies: {sorted(failed)}"
                )

            if func.__name__ in self._cache:
                # Check if dependencies changed
                current_hash = self._current_dependency_hash(deps)
                if current_hash == self._dependency_versions.get(func.__name__):
                    return self._cache[func.__name__]

            result = func(self)
            self._computed.add(func.__name__)
            self._cache[func.__name__] = result
            self._dependency_versions[func.__name__] = self._current_dependency_hash(
                deps
            )
            return result

        return property(wrapper)

    return decorator


class Parameter:
    """Base class for parameters with dependencies."""

    VALID_PARAMS = None

    def __init__(self, **kwargs):

        if self.VALID_PARAMS is None:
            raise NotImplementedError(
                "VALID_PARAMS must be defined in subclasses of Parameter."
            )

        for param, config in self.VALID_PARAMS.items():
            if param not in kwargs and config["default"] is not None:
                kwargs[param] = config["default"]

        # Validate parameter types
        for param, value in kwargs.items():
            if param not in self.VALID_PARAMS:
                raise ValueError(
                    f"Invalid parameter: {param}. "
                    f"Valid parameters are: {list(self.VALID_PARAMS.keys())}"
                )
            expected_type = self.VALID_PARAMS[param]["type"]
            if (
                expected_type is not None
                and value is not None
                and not isinstance(value, expected_type)
            ):
                raise TypeError(
                    f"Parameter '{param}' expected type {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

        self._params = {}
        self._computed = set()
        self._cache = {}
        self._dependency_versions = {}
        for k, v in kwargs.items():
            self._params[k] = v

    def __getattr__(self, item):
        # First check regular params
        if item in self._params:
            return self._params[item]

        # Then check if it's a known property on the class with dependencies
        cls_attr = getattr(type(self), item, None)
        if isinstance(cls_attr, property) and hasattr(cls_attr.fget, "_dependencies"):
            # Try to resolve dependencies
            failed = set()
            if self._resolve_dependency_tree(item, failed):
                # Use descriptor protocol directly
                try:
                    return cls_attr.__get__(self, self.__class__)
                except Exception as e:
                    raise AttributeError(f"Error computing '{item}': {str(e)}")
            else:
                raise AttributeError(
                    f"Cannot access '{item}' due to missing base dependencies: {sorted(failed)}"
                )

        # Otherwise raise normal attribute error
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{item}'")

    def __setattr__(self, key, value):
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self._params[key] = value
            self._invalidate_dependents(key)

    def _invalidate_dependents(self, changed_key):
        to_invalidate = []
        for name, deps_hash in self._dependency_versions.items():
            deps = getattr(self.__class__, name).fget._dependencies
            if changed_key in deps:
                to_invalidate.append(name)
        for key in to_invalidate:
            self._cache.pop(key, None)
            self._computed.discard(key)
            self._dependency_versions.pop(key, None)

    def _current_dependency_hash(self, deps):
        relevant = [str(self._params.get(dep, None)) for dep in deps]
        return hashlib.sha1("".join(relevant).encode()).hexdigest()

    def _resolve_dependency_tree(self, name, failed=None):
        if failed is None:
            failed = set()

        # Return immediately if already in params or cache
        if name in self._params:
            return True
        if name in self._cache:
            return True

        cls_attr = getattr(self.__class__, name, None)
        if isinstance(cls_attr, property):
            func = cls_attr.fget
            if hasattr(func, "_dependencies"):
                all_ok = True
                for dep in func._dependencies:
                    if not self._resolve_dependency_tree(dep, failed):
                        all_ok = False
                if all_ok:
                    # Don't actually access the property here
                    # Just mark that all dependencies are met
                    return True
                else:
                    return False
        else:
            failed.add(name)
            return False

    def to_tensor(self, compute_missing=False):
        tensor_dict = {}

        # First include all direct parameters
        for key, val in self._params.items():
            tensor_dict[key] = keras.ops.convert_to_tensor(val)

        if compute_missing:
            # Find all properties that have dependencies
            for name in dir(self.__class__):
                attr = getattr(self.__class__, name)
                if isinstance(attr, property) and hasattr(attr.fget, "_dependencies"):
                    # Try to resolve and compute the property
                    failed = set()
                    if self._resolve_dependency_tree(name, failed):
                        # Actually compute the property
                        try:
                            val = getattr(self, name)
                            if val is not None:
                                tensor_dict[name] = keras.ops.convert_to_tensor(val)
                                # Make sure it's in the cache for future use
                                if name not in the self._cache:
                                    self._cache[name] = val
                                    self._computed.add(name)
                        except Exception as e:
                            print(f"Warning: Could not compute '{name}': {str(e)}")
        else:
            # Just include what's already been computed
            for key in self._computed:
                val = getattr(self, key)
                tensor_dict[key] = keras.ops.convert_to_tensor(val)

        return tensor_dict


class ScanTestClass(Parameter):
    """Represents an ultrasound scan configuration with computed properties.

    This class manages parameters related to ultrasound scanning and provides
    computed properties that depend on these parameters.

    Args:
        Nx: Number of samples in the x-direction (lateral).
        Nz: Number of samples in the z-direction (axial).
        sound_speed: Speed of sound in the medium in m/s. Defaults to 1540.0.
        sampling_frequency: Sampling frequency in Hz.
        center_frequency: Optional. Center frequency of the transducer in Hz.
        elevational_focus: Optional. Focal length in the elevational direction in m.

    Attributes:
        grid: Meshgrid of x and z coordinates.
        grid_spacing: Spacing between samples in x and z directions.
        wavelength: Wavelength of sound at the sampling frequency.
    """

    VALID_PARAMS = {
        "Nx": {"type": int, "default": None},
        "Nz": {"type": int, "default": None},
        "sound_speed": {"type": float, "default": 1540.0},
        "sampling_frequency": {"type": float, "default": None},
        "center_frequency": {"type": float, "default": None},
        "elevational_focus": {"type": float, "default": None},
    }

    def _timestamp(self):
        return time.time()

    @cache_with_dependencies("Nx", "Nz")
    def grid(self):
        self._grid_called = self._timestamp()
        Nx, Nz = self.Nx, self.Nz
        return np.meshgrid(np.arange(Nx), np.arange(Nz), indexing="ij")

    @cache_with_dependencies("grid")
    def grid_spacing(self):
        self._grid_spacing_called = self._timestamp()
        x, z = self.grid
        dx = np.mean(np.diff(x[:, 0]))
        dz = np.mean(np.diff(z[0, :]))
        return dx, dz

    @cache_with_dependencies("sound_speed", "sampling_frequency")
    def wavelength(self):
        self._wavelength_called = self._timestamp()
        return self.sound_speed / self.sampling_frequency


class Scan(Parameter):
    """Represents an ultrasound scan configuration with computed properties.

    This class manages parameters related to ultrasound scanning and provides
    computed properties that depend on these parameters.

    Args:
        Nx: Number of samples in the x-direction (lateral).
        Nz: Number of samples in the z-direction (axial).
        sound_speed: Speed of sound in the medium in m/s. Defaults to 1540.0.
        sampling_frequency: Sampling frequency in Hz.
        center_frequency: Center frequency of the transducer in Hz.
        n_el: Number of elements in the transducer array.
        n_tx: Number of transmit events.
        n_ax: Number of axial samples in the received signal.
        n_ch: Number of channels (1 for RF, 2 for IQ data).
        xlims: Lateral limits of the imaging region in meters (min, max).
        zlims: Axial limits of the imaging region in meters (min, max).
        probe_geometry: Element positions as array of shape (n_el, 3).
        polar_angles: Polar angles for each transmit event in radians.
        bandwidth_percent: Bandwidth as percentage of center frequency.

    Attributes:
        grid: Meshgrid of x and z coordinates.
        grid_spacing: Spacing between samples in x and z directions.
        wavelength: Wavelength of sound at the sampling frequency.
    """

    VALID_PARAMS = {
        "Nx": {"type": int, "default": None},
        "Nz": {"type": int, "default": None},
        "sound_speed": {"type": float, "default": 1540.0},
        "sampling_frequency": {"type": float, "default": None},
        "center_frequency": {"type": float, "default": None},
        "n_el": {"type": int, "default": None},
        "n_tx": {"type": int, "default": None},
        "n_ax": {"type": int, "default": None},
        "n_ch": {"type": int, "default": 1},
        "xlims": {"type": tuple, "default": None},
        "zlims": {"type": tuple, "default": None},
        "probe_geometry": {"type": np.ndarray, "default": None},
        "polar_angles": {"type": np.ndarray, "default": None},
        "azimuth_angles": {"type": np.ndarray, "default": None},
        "t0_delays": {"type": np.ndarray, "default": None},
        "tx_apodizations": {"type": np.ndarray, "default": None},
        "focus_distances": {"type": np.ndarray, "default": None},
        "initial_times": {"type": np.ndarray, "default": None},
        "bandwidth_percent": {"type": float, "default": 200.0},
        "demodulation_frequency": {"type": float, "default": None},
        "time_to_next_transmit": {"type": np.ndarray, "default": None},
        "pixels_per_wavelength": {"type": int, "default": 4},
        "downsample": {"type": int, "default": 1},
        "element_width": {"type": float, "default": 0.2e-3},
    }

    def __init__(self, **kwargs):
        # Apply defaults for parameters not provided

        # Initialize selected_transmits before calling super().__init__
        self._selected_transmits = None

        # Initialize parent class
        super().__init__(**kwargs)

        # Set selected_transmits after parameters are initialized
        self.selected_transmits = kwargs.get("selected_transmits", None)

    def _timestamp(self):
        return time.time()

    # Core properties with dependency tracking
    @cache_with_dependencies("Nx", "Nz")
    def grid(self):
        """Get meshgrid of x and z coordinates."""
        self._grid_called = self._timestamp()
        Nx, Nz = self.Nx, self.Nz

        # Calculate grid based on specified dimensions
        x = np.linspace(self.xlims[0], self.xlims[1], Nx)
        z = np.linspace(self.zlims[0], self.zlims[1], Nz)
        xgrid, zgrid = np.meshgrid(x, z, indexing="ij")

        # Create 3D grid with y=0 for compatibility with 3D code
        ygrid = np.zeros_like(xgrid)
        grid = np.stack([xgrid, ygrid, zgrid], axis=-1)

        return grid

    @cache_with_dependencies("grid")
    def grid_spacing(self):
        """Get the grid spacing in x and z directions."""
        self._grid_spacing_called = self._timestamp()
        grid = self.grid
        x, _, z = grid[..., 0], grid[..., 1], grid[..., 2]
        dx = np.mean(np.diff(x[:, 0]))
        dz = np.mean(np.diff(z[0, :]))
        return dx, dz

    @cache_with_dependencies("sound_speed", "center_frequency")
    def wavelength(self):
        """Calculate the wavelength based on sound speed and center frequency."""
        self._wavelength_called = self._timestamp()
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

    @property
    def selected_transmits(self):
        """Get the currently selected transmit indices.

        Returns:
            list or None: The list of selected transmit indices, or None if not set.
            If None and n_tx is available, all transmits are used.
        """
        if self._selected_transmits is None and self._params.get("n_tx") is not None:
            # Default to all transmits if none selected
            return list(range(self._params["n_tx"]))
        return self._selected_transmits

    @selected_transmits.setter
    def selected_transmits(self, value):
        """Set the selected transmits directly.

        This only accepts a list of indices or None. For more advanced selection
        options, use the set_transmits() method.

        Args:
            value: A list of transmit indices or None to use all transmits.
        """
        if value is None:
            self._selected_transmits = None
            return

        # Only accept list-like objects for direct setting
        if not isinstance(value, (list, np.ndarray)):
            raise ValueError(
                f"selected_transmits only accepts a list of indices or None. "
                f"Got {type(value)}. Use set_transmits() for more options."
            )

        # Convert numpy array to list
        if isinstance(value, np.ndarray):
            value = value.tolist()

        # Validate indices
        n_tx = self._params.get("n_tx")
        if n_tx is not None:
            if any(idx >= n_tx for idx in value):
                raise ValueError(
                    f"Selected transmit indices {value} exceed available transmits {n_tx}"
                )

        self._selected_transmits = value

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
        n_tx = self._params.get("n_tx")

        # Handle None - use all transmits
        if selection is None:
            self._selected_transmits = None
            return self

        # Convert numpy array to list or single integer
        if isinstance(selection, np.ndarray):
            if len(np.shape(selection)) == 0:
                selection = int(selection)
            elif len(np.shape(selection)) == 1:
                self.selected_transmits = selection.tolist()
                return self
            else:
                raise ValueError(f"Invalid shape for selection: {np.shape(selection)}.")

        # Handle string options
        if isinstance(selection, str):
            if selection == "all":
                self._selected_transmits = None
                return self
            elif selection == "center":
                if n_tx is None:
                    raise ValueError(
                        "Cannot select 'center' transmit when n_tx is not set."
                    )
                self.selected_transmits = [n_tx // 2]
                return self
            else:
                raise ValueError(
                    f"Invalid string value for selection: {selection}. "
                    f"Valid options are 'all' or 'center'."
                )

        # Handle integer - select this many evenly spaced transmits
        if isinstance(selection, int):
            if n_tx is None:
                raise ValueError(
                    "Cannot select transmits by count when n_tx is not set."
                )

            if selection > n_tx:
                raise ValueError(
                    f"Requested {selection} transmits exceeds available transmits ({n_tx})."
                )

            if selection == 1:
                self.selected_transmits = [n_tx // 2]
            else:
                # Compute evenly spaced indices
                tx_indices = np.linspace(0, n_tx - 1, selection)
                self.selected_transmits = list(np.rint(tx_indices).astype(int))
            return self

        # For list-like objects, delegate to the property setter
        self.selected_transmits = selection
        return self

    @property
    def demodulation_frequency(self):
        """The demodulation frequency."""
        if self._params.get("demodulation_frequency") is not None:
            return self._params["demodulation_frequency"]

        if self.n_ch is None:
            raise ValueError("Please set scan.n_ch or scan.demodulation_frequency.")

        # Default behavior based on n_ch
        return self.center_frequency if self.n_ch == 2 else 0.0

    # Array parameters with dependency on selected_transmits
    @cache_with_dependencies("selected_transmits")
    def polar_angles_selected(self):
        """The polar angles of transmits in radians for selected transmits."""
        value = self._params.get("polar_angles")
        if value is None:
            return None

        n_tx = self._params.get("n_tx", 0)
        if len(value) != n_tx:
            return value  # Return unfiltered if shape doesn't match n_tx

        return value[self.selected_transmits]

    @property
    def polar_angles(self):
        """The polar angles of transmits in radians."""
        return self.polar_angles_selected

    @cache_with_dependencies("selected_transmits")
    def azimuth_angles_selected(self):
        """The azimuth angles of transmits in radians for selected transmits."""
        value = self._params.get("azimuth_angles")
        if value is None:
            return None

        n_tx = self._params.get("n_tx", 0)
        if len(value) != n_tx:
            return value  # Return unfiltered if shape doesn't match n_tx

        return value[self.selected_transmits]

    @property
    def azimuth_angles(self):
        """The azimuth angles of transmits in radians."""
        return self.azimuth_angles_selected

    @cache_with_dependencies("selected_transmits")
    def t0_delays_selected(self):
        """The transmit delays in seconds for selected transmits."""
        value = self._params.get("t0_delays")
        if value is None:
            return None

        n_tx = self._params.get("n_tx", 0)
        if len(value) != n_tx:
            return value  # Return unfiltered if shape doesn't match n_tx

        return value[self.selected_transmits]

    @property
    def t0_delays(self):
        """The transmit delays in seconds."""
        return self.t0_delays_selected

    @cache_with_dependencies("selected_transmits")
    def tx_apodizations_selected(self):
        """The transmit apodizations for selected transmits."""
        value = self._params.get("tx_apodizations")
        if value is None:
            return None

        n_tx = self._params.get("n_tx", 0)
        if len(value) != n_tx:
            return value  # Return unfiltered if shape doesn't match n_tx

        return value[self.selected_transmits]

    @property
    def tx_apodizations(self):
        """The transmit apodizations."""
        return self.tx_apodizations_selected

    @cache_with_dependencies("selected_transmits")
    def focus_distances_selected(self):
        """The focus distances in meters for selected transmits."""
        value = self._params.get("focus_distances")
        if value is None:
            return None

        n_tx = self._params.get("n_tx", 0)
        if len(value) != n_tx:
            return value  # Return unfiltered if shape doesn't match n_tx

        return value[self.selected_transmits]

    @property
    def focus_distances(self):
        """The focus distances in meters."""
        return self.focus_distances_selected

    @cache_with_dependencies("selected_transmits")
    def initial_times_selected(self):
        """The initial times in seconds for selected transmits."""
        value = self._params.get("initial_times")
        if value is None:
            return None

        n_tx = self._params.get("n_tx", 0)
        if len(value) != n_tx:
            return value  # Return unfiltered if shape doesn't match n_tx

        return value[self.selected_transmits]

    @property
    def initial_times(self):
        """The initial times in seconds."""
        return self.initial_times_selected

    @cache_with_dependencies("selected_transmits")
    def time_to_next_transmit_selected(self):
        """Time between transmit events for selected transmits."""
        value = self._params.get("time_to_next_transmit")
        if value is None:
            return None

        return value[:, self.selected_transmits]

    @property
    def time_to_next_transmit(self):
        """Time between transmit events."""
        return self.time_to_next_transmit_selected

    def get_scan_parameters(self):
        """Returns a dictionary with all the parameters of the scan."""
        return {
            param: getattr(self, param)
            for param in self.VALID_PARAMS.keys()
            if hasattr(self, param)
        }


def test_chain_dependency_grid_spacing():
    s = ScanTestClass(Nx=10, Nz=20)
    _ = s.grid_spacing
    assert hasattr(s, "_grid_called")
    assert hasattr(s, "_grid_spacing_called")


def test_recursive_error_message_is_clear():
    s = ScanTestClass()
    try:
        _ = s.grid_spacing
    except AttributeError as e:
        assert "Nx" in str(e) and "Nz" in str(e), f"Expected Nx/Nz in error, got: {e}"


def test_no_recompute_if_dependencies_unchanged():
    s = ScanTestClass(Nx=5, Nz=5)
    _ = s.grid
    first_call_time = s._grid_called
    time.sleep(0.01)
    _ = s.grid
    assert s._grid_called == first_call_time


def test_recompute_if_dependency_changed():
    s = ScanTestClass(Nx=5, Nz=5)
    _ = s.grid
    old_time = s._grid_called
    time.sleep(0.01)
    s.Nx = 6
    _ = s.grid
    assert s._grid_called > old_time


def test_to_tensor_computes_chain():
    s = ScanTestClass(Nx=5, Nz=5, sound_speed=1500.0, sampling_frequency=5e6)
    tensors = s.to_tensor(compute_missing=True)
    assert "grid" in tensors
    assert "grid_spacing" in tensors
    assert "wavelength" in tensors
    assert tensors["wavelength"] == keras.ops.convert_to_tensor(1500.0 / 5e6)


def test_invalid_param_raises_error():
    try:
        s = ScanTestClass(Nx=5, Nz=5, invalid_param=10)
    except ValueError as e:
        assert "Invalid parameter: invalid_param" in str(e), str(e)

    try:
        s = ScanTestClass(Nx=5, Nz=5, sound_speed=1500.0, sampling_frequency="fast")
    except TypeError as e:
        assert "Parameter 'sampling_frequency' expected type float" in str(e), str(e)


## Tests for the actual Scan class


def test_scan_grid_creation():
    s = Scan(
        Nx=10,
        Nz=20,
        xlims=(-0.01, 0.01),
        zlims=(0, 0.04),
        center_frequency=5e6,
        sampling_frequency=20e6,
    )
    grid = s.grid
    assert grid.shape == (10, 20, 3)
    assert np.isclose(grid[0, 0, 0], -0.01)  # x min
    assert np.isclose(grid[-1, 0, 0], 0.01)  # x max
    assert np.isclose(grid[0, 0, 2], 0)  # z min
    assert np.isclose(grid[0, -1, 2], 0.04)  # z max


def test_scan_array_params():
    # Create scan with array parameters
    n_tx, n_el = 5, 8
    polar_angles = np.linspace(-30, 30, n_tx) * np.pi / 180
    t0_delays = np.zeros((n_tx, n_el))
    tx_apodizations = np.ones((n_tx, n_el))

    s = Scan(
        n_tx=n_tx,
        n_el=n_el,
        polar_angles=polar_angles,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        center_frequency=5e6,
        sampling_frequency=20e6,
    )

    # Verify the array parameters are set correctly
    assert len(s.polar_angles) == n_tx
    assert s.t0_delays.shape == (n_tx, n_el)
    assert s.tx_apodizations.shape == (n_tx, n_el)


def test_scan_selected_transmits():
    # Create scan with multiple transmits
    n_tx, n_el = 10, 8
    polar_angles = np.linspace(-30, 30, n_tx) * np.pi / 180

    s = Scan(
        n_tx=n_tx,
        n_el=n_el,
        polar_angles=polar_angles,
        center_frequency=5e6,
        sampling_frequency=20e6,
    )

    # Test selecting a subset of transmits
    s.selected_transmits = 3  # Select 3 evenly spaced transmits
    assert len(s.selected_transmits) == 3
    assert len(s.polar_angles) == 3

    # Test selecting specific transmits
    s.selected_transmits = [0, 3, 9]
    assert s.selected_transmits == [0, 3, 9]
    assert len(s.polar_angles) == 3
    assert np.isclose(s.polar_angles[0], polar_angles[0])
    assert np.isclose(s.polar_angles[1], polar_angles[3])
    assert np.isclose(s.polar_angles[2], polar_angles[9])

    # Test 'center' transmit selection
    s.selected_transmits = "center"
    assert s.selected_transmits == [n_tx // 2]
    assert len(s.polar_angles) == 1


def test_scan_wavelength():
    s = Scan(center_frequency=5e6, sound_speed=1500.0)
    assert np.isclose(s.wavelength, 1500.0 / 5e6)

    # Test wavelength updates when dependencies change
    s.sound_speed = 1540.0
    assert np.isclose(s.wavelength, 1540.0 / 5e6)

    s.center_frequency = 7e6
    assert np.isclose(s.wavelength, 1540.0 / 7e6)


def test_scan_demodulation_frequency():
    # Test default RF mode (n_ch=1)
    s = Scan(n_ch=1, center_frequency=5e6)
    assert np.isclose(s.demodulation_frequency, 0.0)

    # Test default IQ mode (n_ch=2)
    s = Scan(n_ch=2, center_frequency=5e6)
    assert np.isclose(s.demodulation_frequency, 5e6)

    # Test explicit override
    s = Scan(n_ch=2, center_frequency=5e6, demodulation_frequency=4e6)
    assert np.isclose(s.demodulation_frequency, 4e6)


def test_scan_flatgrid():
    s = Scan(Nx=10, Nz=20, xlims=(-0.01, 0.01), zlims=(0, 0.04))
    flat = s.flatgrid
    assert flat.shape == (200, 3)  # 10*20 = 200 points

    # The flattened grid should match the original grid
    assert np.array_equal(flat.reshape(10, 20, 3), s.grid)


def test_scan_to_tensor():
    s = Scan(
        Nx=5,
        Nz=5,
        xlims=(-0.01, 0.01),
        zlims=(0, 0.04),
        center_frequency=5e6,
        sound_speed=1500.0,
    )
    tensors = s.to_tensor(compute_missing=True)
    assert "grid" in tensors
    assert "grid_spacing" in tensors
    assert "wavelength" in tensors
    assert "flatgrid" in tensors

    # Extract numpy values for comparison
    wavelength_tensor = float(keras.ops.convert_to_numpy(tensors["wavelength"]))
    wavelength_expected = 1500.0 / 5e6
    assert np.isclose(wavelength_tensor, wavelength_expected)


if __name__ == "__main__":
    test_chain_dependency_grid_spacing()
    test_recursive_error_message_is_clear()
    test_no_recompute_if_dependencies_unchanged()
    test_recompute_if_dependency_changed()
    test_to_tensor_computes_chain()
    test_invalid_param_raises_error()

    # New tests for Scan class
    test_scan_grid_creation()
    test_scan_array_params()
    test_scan_selected_transmits()
    test_scan_wavelength()
    test_scan_demodulation_frequency()
    test_scan_flatgrid()
    test_scan_to_tensor()

    print("All tests passed!")
