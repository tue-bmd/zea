"""
Parameter System for Ultrasound Imaging

This module provides a parameter management system designed for ultrasound imaging applications,
with transparent dependency tracking, lazy computation, and efficient caching.

Key Features:
- Property-based computation with automatic dependency tracking
- Lazy computation of expensive properties
- Caching of computed results with automatic invalidation
- Type validation for parameters
- Dependency tree resolution to ensure proper computation order
- Tensor conversion for machine learning integration

The Parameter base class implements the core functionality:
- Automatic type checking of inputs
- Property-based access to parameters with caching
- Dependency tracking to avoid unnecessary recomputation
- Ability to convert parameters to tensors for neural network use

The cache_with_dependencies decorator allows defining properties that:
1. Automatically compute when accessed
2. Cache their results to avoid redundant computation
3. Invalidate when their dependencies change
4. Track dependencies for proper resolution order

The Scan class is a concrete implementation specifically for ultrasound
imaging configuration, defining specific parameters and their relationships
for ultrasound scan geometry, beam characteristics, and timing information.

"""

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
            # Validate that parameter is in VALID_PARAMS
            if key not in self.VALID_PARAMS:
                raise ValueError(
                    f"Invalid parameter: {key}. "
                    f"Valid parameters are: {list(self.VALID_PARAMS.keys())}"
                )

            # Validate parameter type
            expected_type = self.VALID_PARAMS[key]["type"]
            if (
                expected_type is not None
                and value is not None
                and not isinstance(value, expected_type)
            ):
                raise TypeError(
                    f"Parameter '{key}' expected type {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

            # Set the parameter and invalidate dependencies
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
                                if name not in self._cache:
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

    def __repr__(self):
        param_lines = []
        for k, v in self._params.items():
            if v is None:
                continue

            # Handle arrays by showing their shape instead of content
            if isinstance(v, np.ndarray):
                param_lines.append(f"{k}=array(shape={v.shape})")
            else:
                param_lines.append(f"{k}={repr(v)}")

        param_str = ", ".join(param_lines)
        return f"{self.__class__.__name__}({param_str})"

    def __str__(self):
        param_lines = []
        for k, v in self._params.items():
            if v is None:
                continue

            # Handle arrays by showing their shape instead of content
            if isinstance(v, np.ndarray):
                param_lines.append(f"    {k}=array(shape={v.shape})")
            else:
                param_lines.append(f"    {k}={v}")

        param_str = ",\n".join(param_lines)
        return f"{self.__class__.__name__}(\n{param_str}\n)"


class DummyParameters(Parameter):
    """A simple test class with parameters and computed properties.

    This class is used for testing the Parameter framework with simple
    dependencies between properties.

    Args:
        param1: First parameter (equivalent to Nx in the original)
        param2: Second parameter (equivalent to Nz in the original)
        param3: Third parameter with default value (like sound_speed)
        param4: Fourth parameter (like sampling_frequency)
        param5: Optional fifth parameter
        param6: Optional sixth parameter

    Attributes:
        computed1: A computed property depending on param1 and param2
        computed2: A computed property depending on computed1
        computed3: A computed property depending on param3 and param4
    """

    VALID_PARAMS = {
        "param1": {"type": int, "default": None},
        "param2": {"type": int, "default": None},
        "param3": {"type": float, "default": 1540.0},
        "param4": {"type": float, "default": None},
        "param5": {"type": float, "default": None},
        "param6": {"type": float, "default": None},
    }

    def _timestamp(self):
        return time.time()

    @cache_with_dependencies("param1", "param2")
    def computed1(self):
        self._computed1_called = self._timestamp()
        p1, p2 = self.param1, self.param2
        return np.meshgrid(np.arange(p1), np.arange(p2), indexing="ij")

    @cache_with_dependencies("computed1")
    def computed2(self):
        self._computed2_called = self._timestamp()
        x, z = self.computed1
        dx = np.mean(np.diff(x[:, 0]))
        dz = np.mean(np.diff(z[0, :]))
        return dx, dz

    @cache_with_dependencies("param3", "param4")
    def computed3(self):
        self._computed3_called = self._timestamp()
        return self.param3 / self.param4


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
        # Store the current selection state before initialization
        selected_transmits_input = kwargs.pop("selected_transmits", None)

        # Initialize parent class
        super().__init__(**kwargs)

        # Initialize selection to None
        self._selected_transmits = None

        # Apply selection from input if provided
        if selected_transmits_input is not None:
            self.set_transmits(selected_transmits_input)

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


def test_chain_dependency_grid_spacing():
    s = DummyParameters(param1=10, param2=20)
    _ = s.computed2
    assert hasattr(s, "_computed1_called")
    assert hasattr(s, "_computed2_called")


def test_recursive_error_message_is_clear():
    s = DummyParameters()
    try:
        _ = s.computed2
    except AttributeError as e:
        assert "param1" in str(e) and "param2" in str(
            e
        ), f"Expected param1/param2 in error, got: {e}"


def test_no_recompute_if_dependencies_unchanged():
    s = DummyParameters(param1=5, param2=5)
    _ = s.computed1
    first_call_time = s._computed1_called
    time.sleep(0.01)
    _ = s.computed1
    assert s._computed1_called == first_call_time


def test_recompute_if_dependency_changed():
    s = DummyParameters(param1=5, param2=5)
    _ = s.computed1
    old_time = s._computed1_called
    time.sleep(0.01)
    s.param1 = 6
    _ = s.computed1
    assert s._computed1_called > old_time


def test_to_tensor_computes_chain():
    s = DummyParameters(param1=5, param2=5, param3=1500.0, param4=5e6)
    tensors = s.to_tensor(compute_missing=True)
    assert "computed1" in tensors
    assert "computed2" in tensors
    assert "computed3" in tensors
    assert tensors["computed3"] == keras.ops.convert_to_tensor(1500.0 / 5e6)


def test_invalid_param_raises_error():
    try:
        s = DummyParameters(param1=5, param2=5, invalid_param=10)
    except ValueError as e:
        assert "Invalid parameter: invalid_param" in str(e), str(e)

    try:
        s = DummyParameters(param1=5, param2=5, param3=1500.0, param4="fast")
    except TypeError as e:
        assert "Parameter 'param4' expected type float" in str(e), str(e)


def test_set_invalid_param_after_init():
    """Test that assigning invalid parameters after initialization raises errors."""
    s = DummyParameters(param1=5, param2=5)

    # Test setting a valid parameter works
    s.param3 = 1600.0
    assert s.param3 == 1600.0

    # Test setting invalid parameter raises error
    try:
        s.invalid_param = 10
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid parameter: invalid_param" in str(e), str(e)

    # Test setting wrong type raises error
    try:
        s.param3 = "fast"
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Parameter 'param3' expected type float" in str(e)


def test_tensor_parameter_count():
    """Test that to_tensor includes the correct number of parameters."""
    s = DummyParameters(param1=5, param2=5, param3=1500.0, param4=5e6)

    # Before accessing any computed properties
    tensors_before = s.to_tensor(compute_missing=False)
    assert len(tensors_before) == 4  # param1, param2, param3, param4

    # Access a computed property to trigger caching
    _ = s.computed2

    # After accessing computed properties but without compute_missing
    tensors_after = s.to_tensor(compute_missing=False)
    assert len(tensors_after) > len(tensors_before)
    assert "computed1" in tensors_after
    assert "computed2" in tensors_after

    # With compute_missing=True, should compute computed3 additionally
    tensors_all = s.to_tensor(compute_missing=True)
    assert "computed3" in tensors_all
    assert len(tensors_all) >= len(tensors_after)

    # Direct check for specific counts
    # Initial params + computed1 + computed2 + computed3
    expected_count = 4 + 3  # 4 direct params + 3 computed properties
    assert (
        len(tensors_all) == expected_count
    ), f"Expected {expected_count} parameters, got {len(tensors_all)}"


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

    # Default should be all transmits
    assert len(s.selected_transmits) == n_tx
    assert len(s.polar_angles) == n_tx

    # Test selecting a subset of transmits
    s.set_transmits(3)  # Select 3 evenly spaced transmits
    # Print debug info
    print(
        f"Selected transmits: {s.selected_transmits}, length: {len(s.selected_transmits)}"
    )
    print(f"Polar angles: {s.polar_angles}, length: {len(s.polar_angles)}")
    assert len(s.selected_transmits) == 3
    assert len(s.polar_angles) == 3

    # Test selecting specific transmits
    s.set_transmits([0, 3, 9])
    assert s.selected_transmits == [0, 3, 9]
    assert len(s.polar_angles) == 3
    assert np.isclose(s.polar_angles[0], polar_angles[0])
    assert np.isclose(s.polar_angles[1], polar_angles[3])
    assert np.isclose(s.polar_angles[2], polar_angles[9])

    # Test 'center' transmit selection
    s.set_transmits("center")
    assert s.selected_transmits == [n_tx // 2]
    assert len(s.polar_angles) == 1

    # Test 'all' transmits selection
    s.set_transmits("all")
    assert len(s.selected_transmits) == n_tx
    assert len(s.polar_angles) == n_tx

    # Test method chaining
    result = s.set_transmits(2)
    assert result is s
    assert len(s.selected_transmits) == 2


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
    test_set_invalid_param_after_init()
    test_tensor_parameter_count()

    test_scan_grid_creation()
    test_scan_array_params()
    test_scan_wavelength()
    test_scan_demodulation_frequency()
    test_scan_flatgrid()
    test_scan_to_tensor()
    test_scan_selected_transmits()

    print("All tests passed!")
