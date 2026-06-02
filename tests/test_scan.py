"""Tests for the Parameters class."""

import pickle

import numpy as np
import pytest

import zea
from zea import Parameters
from zea.data.spec import ProbeSpec, ScanSpec
from zea.internal.dummy_scan import get_scan

scan_args = {
    "n_tx": 10,
    "n_el": 10,
    "n_ch": 1,
    "xlims": (-0.019, 0.019),
    "ylims": (0, 0),
    "zlims": (0, 0.04),
    "center_frequency": 7e6,
    "sampling_frequency": 28e6,
    "demodulation_frequency": 0.0,
    "sound_speed": 1540.0,
    "n_ax": 3328,
    "grid_size_x": 64,
    "grid_size_z": 128,
    "pixels_per_wavelength": 4,
    "polar_angles": np.linspace(-np.pi / 2, np.pi / 2, 10),
    "azimuth_angles": np.linspace(-np.pi / 2, np.pi / 2, 10),
    "t0_delays": np.repeat(np.linspace(0, 1e-6, 10)[..., None], 10, axis=-1),
    "tx_apodizations": np.ones((10, 10)),
    "focus_distances": np.ones(10) * 0.04,
    "initial_times": np.zeros((10,)),
    "waveforms_one_way": np.zeros((2, 64)),
    "waveforms_two_way": np.zeros((2, 64)),
    "tgc_gain_curve": np.ones((3328,)),
    "probe_geometry": np.column_stack(
        (
            np.linspace(-0.019, 0.019, 10),
            np.zeros(10),
            np.zeros(10),
        )
    ),
}


def test_scan_repr():
    """Parameters repr is a single-line constructor-style string."""
    parameters = Parameters(**scan_args)
    r = repr(parameters)
    assert r.startswith("Parameters(")
    assert r.endswith(")")
    assert "\n" not in r
    assert "sampling_frequency=" in r
    assert "MHz" in r


def test_scan_str():
    """Parameters str is a multi-line constructor-style string."""
    parameters = Parameters(**scan_args)
    s = str(parameters)
    assert s.startswith("Parameters(\n")
    assert s.endswith("\n)")
    assert "\n" in s
    assert "sampling_frequency=" in s


def test_scan_compare():
    """Test comparison of Parameters objects."""
    parameters = Parameters(**scan_args)
    parameters2 = Parameters(**scan_args)
    parameters3 = Parameters(**scan_args)
    parameters3.sound_speed = 1000

    assert parameters == parameters2
    assert parameters != parameters3


def test_scan_copy():
    """Test copying of Parameters objects."""
    parameters = Parameters(**scan_args)
    parameters_copy = parameters.copy()

    assert parameters == parameters_copy
    parameters.n_tx = 20
    assert parameters != parameters_copy


@pytest.mark.parametrize(
    "selection",
    [
        None,
        [0, 1, 2],
    ],
)
def test_scan_copy_selected_transmits(selection):
    """Test that selected_transmits is copied correctly."""
    parameters = Parameters(**scan_args)
    parameters.set_transmits(selection)
    parameters_copy = parameters.copy()

    assert np.array_equal(parameters.selected_transmits, parameters_copy.selected_transmits)
    parameters.set_transmits(scan_args["n_tx"] // 5)
    assert not np.array_equal(parameters.selected_transmits, parameters_copy.selected_transmits)


@pytest.mark.parametrize(
    "selection",
    [
        None,
        "all",
        "center",
        "focused",
        "diverging",
        "plane",
        3,
        1,
        [0, 1, 2],
        np.array([0, 1, 2]),
        slice(0, 5, 2),
    ],
)
def test_set_transmits(selection):
    """Test setting transmits with various selection methods."""
    local_scan_args = scan_args.copy()

    if isinstance(selection, str):
        if selection == "diverging":
            local_scan_args["focus_distances"] = np.ones(scan_args["n_tx"]) * -0.02
        elif selection == "plane":
            local_scan_args["focus_distances"] = np.full(scan_args["n_tx"], np.inf)

    parameters = Parameters(**local_scan_args)
    parameters.set_transmits(selection)

    if selection is None:
        assert parameters.n_tx == scan_args["n_tx"]
    elif isinstance(selection, str):
        if selection == "all":
            assert parameters.n_tx == scan_args["n_tx"]
        elif selection == "center":
            assert parameters.n_tx == 1
            assert parameters.selected_transmits[0] == scan_args["n_tx"] // 2
        elif selection == "focused":
            assert np.all(parameters.focus_distances > 0)
        elif selection == "diverging":
            assert np.all(parameters.focus_distances < 0)
        elif selection == "plane":
            assert np.all(np.isinf(parameters.focus_distances))
    elif isinstance(selection, int):
        assert parameters.n_tx == selection
    elif isinstance(selection, (list, np.ndarray)):
        expected = selection if isinstance(selection, list) else selection.tolist()
        assert np.array_equal(parameters.selected_transmits, expected)
    elif isinstance(selection, slice):
        expected = list(range(*selection.indices(scan_args["n_tx"])))
        assert np.array_equal(parameters.selected_transmits, expected)


def test_scan_erroneous_set_transmits():
    """Test erroneous inputs to set_transmits."""
    parameters = Parameters(**scan_args)

    with pytest.raises(ValueError):
        parameters.set_transmits(-1)

    with pytest.raises(ValueError):
        parameters.set_transmits(scan_args["n_tx"] + 1)

    with pytest.raises(ValueError):
        parameters.set_transmits([0, scan_args["n_tx"]])

    with pytest.raises(ValueError):
        parameters.set_transmits([0, 1, 2.3])

    with pytest.raises(ValueError):
        parameters.set_transmits("invalid_string")


def test_initialization():
    """Test initialization of Parameters class."""
    parameters = Parameters(**scan_args)

    assert parameters.n_tx == scan_args["n_tx"]
    assert parameters.n_el == scan_args["n_el"]
    assert parameters.n_ch == scan_args["n_ch"]
    assert np.allclose(parameters.xlims, scan_args["xlims"])
    assert np.allclose(parameters.ylims, scan_args["ylims"])
    assert np.allclose(parameters.zlims, scan_args["zlims"])
    assert np.allclose(parameters.center_frequency, scan_args["center_frequency"])
    assert np.allclose(parameters.sampling_frequency, scan_args["sampling_frequency"])
    assert np.allclose(parameters.demodulation_frequency, scan_args["demodulation_frequency"])
    assert np.allclose(parameters.sound_speed, scan_args["sound_speed"])
    assert np.allclose(parameters.n_ax, scan_args["n_ax"])
    assert np.allclose(parameters.grid_size_x, scan_args["grid_size_x"])
    assert np.allclose(parameters.grid_size_z, scan_args["grid_size_z"])
    assert np.allclose(parameters.polar_angles, scan_args["polar_angles"])
    assert np.allclose(parameters.azimuth_angles, scan_args["azimuth_angles"])
    assert np.allclose(parameters.t0_delays, scan_args["t0_delays"])
    assert np.allclose(parameters.tx_apodizations, scan_args["tx_apodizations"])
    assert np.allclose(parameters.focus_distances, scan_args["focus_distances"])
    assert np.allclose(parameters.initial_times, scan_args["initial_times"])
    assert np.allclose(parameters.pixels_per_wavelength, scan_args["pixels_per_wavelength"])


@pytest.mark.parametrize(
    "attr, expected_shape",
    [
        ("polar_angles", (10,)),
        ("azimuth_angles", (10,)),
        ("t0_delays", (10, 10)),
        ("tx_apodizations", (10, 10)),
        ("focus_distances", (10,)),
        ("initial_times", (10,)),
    ],
)
def test_selected_transmits_affects_shape(attr, expected_shape):
    parameters = Parameters(**scan_args)
    # Check initial shape
    val = getattr(parameters, attr)
    val_tensor = parameters.to_tensor(include=[attr])[attr]
    assert val.shape == val_tensor.shape == expected_shape

    # Select 3 transmits
    parameters.set_transmits(3)
    val = getattr(parameters, attr)
    val_tensor = parameters.to_tensor(include=[attr])[attr]

    # For 2D arrays, first dimension is always n_tx
    assert val.shape[0] == val_tensor.shape[0] == 3

    # Select center transmit
    parameters.set_transmits("center")
    val = getattr(parameters, attr)
    val_tensor = parameters.to_tensor(include=[attr])[attr]
    assert val.shape[0] == val_tensor.shape[0] == 1

    # Select all again
    parameters.set_transmits("all")
    val = getattr(parameters, attr)
    val_tensor = parameters.to_tensor(include=[attr])[attr]
    assert val.shape[0] == val_tensor.shape[0] == expected_shape[0]

    # Select with some numpy array
    parameters.set_transmits(np.arange(3))
    val = getattr(parameters, attr)
    val_tensor = parameters.to_tensor(include=[attr])[attr]
    assert val.shape[0] == val_tensor.shape[0] == 3

    # Select with a list
    parameters.set_transmits([1, 2, 3])
    val = getattr(parameters, attr)
    val_tensor = parameters.to_tensor(include=[attr])[attr]
    assert val.shape[0] == val_tensor.shape[0] == 3

    # Select with a slice
    parameters.set_transmits(slice(0, 5, 2))
    val = getattr(parameters, attr)
    val_tensor = parameters.to_tensor(include=[attr])[attr]
    assert val.shape[0] == val_tensor.shape[0] == 3


def test_set_attributes():
    """Test setting attributes of Parameters class."""
    parameters = Parameters(**scan_args)

    parameters.selected_transmits = [0]

    with pytest.raises(ValueError):
        parameters.grid = np.zeros((10, 10))


def test_accessing_valid_but_unset_attributes():
    """Test accessing valid but unset attributes of Parameters class."""

    parameters = Parameters(n_tx=5)
    parameters.focus_distances


def test_missing_transmit_defaults_warn_once_on_access(monkeypatch):
    local_scan_args = scan_args.copy()
    local_scan_args.pop("azimuth_angles", None)
    local_scan_args.pop("t0_delays", None)
    local_scan_args.pop("tx_apodizations", None)
    local_scan_args.pop("focus_distances", None)
    local_scan_args.pop("transmit_origins", None)
    local_scan_args.pop("initial_times", None)
    local_scan_args.pop("tgc_gain_curve", None)

    warnings = []

    # Reset warning_once state to make this test deterministic.
    zea.log._warned_locations.clear()

    def _capture_warning(message, *args, **kwargs):
        warnings.append(message)
        return message

    monkeypatch.setattr("zea.scan.log.warning", _capture_warning)

    # Nothing should be warned at initialization, only on-demand when fallback
    # properties are actually accessed.
    scan = Parameters(**local_scan_args)
    assert len(warnings) == 0

    for i in range(5):
        scan.selected_transmits = slice(0, i + 1)
        _ = scan.azimuth_angles
        _ = scan.t0_delays
        _ = scan.tx_apodizations
        _ = scan.focus_distances
        _ = scan.transmit_origins
        _ = scan.initial_times
        _ = scan.tgc_gain_curve

    assert warnings.count("No ``azimuth_angles`` provided, using zeros") == 1
    assert warnings.count("No ``t0_delays`` provided, using zeros") == 1
    assert warnings.count("No ``tx_apodizations`` provided, using ones") == 1
    assert warnings.count("No ``focus_distances`` provided, using zeros") == 1
    assert warnings.count("No ``transmit_origins`` provided, using zeros") == 1
    assert warnings.count("No ``initial_times`` provided, using zeros") == 1
    assert warnings.count("No ``tgc_gain_curve`` provided, using ones") == 1


def test_missing_defaults_warn_once_per_scan_instance(monkeypatch):
    local_scan_args = scan_args.copy()
    local_scan_args.pop("azimuth_angles", None)

    warnings = []

    zea.log._warned_locations.clear()

    def _capture_warning(message, *args, **kwargs):
        warnings.append(message)
        return message

    monkeypatch.setattr("zea.scan.log.warning", _capture_warning)

    parameters1 = Parameters(**local_scan_args)
    parameters2 = Parameters(**local_scan_args)

    # First access in each instance should warn.
    _ = parameters1.azimuth_angles
    _ = parameters2.azimuth_angles

    # Repeated access in same instance should not warn again.
    _ = parameters1.azimuth_angles
    _ = parameters2.azimuth_angles

    assert warnings.count("No ``azimuth_angles`` provided, using zeros") == 2


def test_scan_pickle():
    """Test pickling and unpickling of Parameters class."""

    parameters = Parameters(**scan_args)
    parameters_pickled = pickle.dumps(parameters)
    parameters_unpickled = pickle.loads(parameters_pickled)

    assert parameters == parameters_unpickled, (
        "Unpickled Parameters object does not match the original"
    )
    assert parameters is not parameters_unpickled, (
        "Unpickled Parameters object is the same instance as the original"
    )


def test_valid_params_default():
    """Test that modifying pfield_kwargs in one Parameters instance does not affect another.

    The origin of this test is a bug where in VALID_PARAMS, the default value for pfield_kwargs
    was a mutable dictionary, leading to shared state across instances.
    """

    scan1 = get_scan()
    scan1.pfield_kwargs["norm"] = False

    parameters2 = get_scan()
    assert parameters2.pfield_kwargs == {}, (
        "scan2.pfield_kwargs seems to be affected by scan1 modification"
    )
    assert scan1 != parameters2, "scan1 and scan2 should be different after modification of scan1"


def test_inplace_modification():
    """Test that modifying pfield_kwargs in-place, will update the pfield."""

    def edit1(parameters):
        """edit direct dependency (dict) in-place"""
        parameters.pfield_kwargs["norm"] = False
        return parameters

    def edit2(parameters):
        """edit another indirect dependency (np.ndarray) in-place"""
        parameters.probe_geometry[:, 0] *= 1.02
        return parameters

    def edit3(parameters):
        """edit indirect dependency (list) in-place
        pfield -> grid -> zlims"""
        # convert to list to allow in-place edit
        # this will invalidate pfield
        parameters.zlims = list(parameters.zlims)
        # therefore we need to force a computation of pfield to cache it
        _ = parameters.pfield.copy()
        # and then edit in-place
        parameters.zlims[1] += 0.01
        return parameters

    for edit_fn in (edit1, edit2, edit3):
        parameters = get_scan(pfield_kwargs={"norm": True})
        original_pfield = parameters.pfield.copy()
        assert "pfield" in parameters._cache, "pfield should be cached after first access"

        # Modify something in-place
        parameters = edit_fn(parameters)

        # Check that the grid has been updated
        assert not np.array_equal(original_pfield, parameters.pfield), (
            f"scan.pfield seems to be unaffected by in-place modification in {edit_fn.__name__}"
        )


def test_inplace_modification_tensor_cache():
    """Test that modifying pfield_kwargs in-place, will update the pfield_tensor."""

    parameters = get_scan(pfield_kwargs={"norm": True})
    tensor_dict = parameters.to_tensor(include=["pfield"])
    parameters.pfield_kwargs["norm"] = False  # in-place modification
    tensor_dict2 = parameters.to_tensor(include=["pfield"])

    assert not np.array_equal(tensor_dict["pfield"], tensor_dict2["pfield"]), (
        "_tensor_cache['pfield'] seems to be unaffected by in-place modification"
    )


def test_update_behaviour_and_cache_invalidation():
    """Test Parameters.update: skipping unchanged values and force invalidation."""
    parameters = Parameters(**scan_args)

    # Access grid to populate cache
    _ = parameters.grid
    assert "grid" in parameters._cache
    cached_before = parameters._cache.get("grid")

    # Update with the same value (should be a no-op and keep cache)
    parameters.update(center_frequency=parameters.center_frequency)
    cached_after = parameters._cache.get("grid")
    assert cached_before is cached_after

    # Force update with same value should invalidate cache (grid removed until next access)
    parameters.update(force=True, center_frequency=parameters.center_frequency)
    assert "grid" not in parameters._cache

    # Update with a different value should also invalidate cache
    _ = parameters.grid  # repopulate cache
    parameters.update(center_frequency=parameters.center_frequency * 1.01)
    assert "grid" not in parameters._cache


def test_update_stores_unknown_keys_as_custom():
    """Ensure update stores unknown keys as custom (passthrough) parameters."""

    parameters = Parameters(**scan_args)

    # Unknown key is stored as a custom passthrough parameter (not rejected).
    parameters.update(nonexistent_param=123)
    assert parameters.nonexistent_param == 123
    assert parameters._custom_params["nonexistent_param"] == 123


def test_valid_params_cover_specs():
    """Every ScanSpec and ProbeSpec field must be a valid Parameters key.

    Enforces the single-source-of-truth contract: any file-backed parameter
    (scan or probe) can be held by the Parameters class. ``center_frequency``
    is intentionally excluded from ProbeSpec (renamed to
    ``probe_center_frequency``) to avoid colliding with the scan field.
    """
    valid = set(Parameters.VALID_PARAMS)
    probe_spec = set(ProbeSpec.SCHEMA)
    probe_spec.remove("name")
    probe_spec.remove("type")
    missing_scan = set(ScanSpec.SCHEMA) - valid
    missing_probe = probe_spec - valid
    assert not missing_scan, f"ScanSpec fields missing from Parameters.VALID_PARAMS: {missing_scan}"
    assert not missing_probe, (
        f"ProbeSpec fields missing from Parameters.VALID_PARAMS: {missing_probe}"
    )


def test_scan_and_probe_specs_are_disjoint():
    """ScanSpec and ProbeSpec field names must not collide.

    A collision would make merging probe + scan parameters into a single
    Parameters object ambiguous. This guards against re-introducing one.
    """
    overlap = set(ScanSpec.SCHEMA) & set(ProbeSpec.SCHEMA)
    assert overlap == set(), f"ScanSpec and ProbeSpec share field names: {overlap}"


def test_custom_parameters_passthrough_to_tensor():
    """Custom params are stored as-is, ignored by derivation, and surface in to_tensor."""
    parameters = Parameters(**scan_args)
    parameters.update(my_custom_parameter=42)
    assert parameters.my_custom_parameter == 42
    # Custom param is not a validated leaf param.
    assert "my_custom_parameter" not in parameters._params
    # It still flows through to_tensor when requested.
    tensors = parameters.to_tensor(include=["my_custom_parameter", "center_frequency"])
    assert "my_custom_parameter" in tensors
    # Derived properties still compute (custom params don't interfere).
    assert parameters.wavelength == parameters.sound_speed / parameters.center_frequency
