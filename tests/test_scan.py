"""Tests for the Parameters class."""

import pickle
from unittest.mock import patch

import numpy as np
import pytest

from zea import Parameters
from zea.data.spec import ProbeSpec, ScanSpec
from zea.internal.dummy_scan import get_parameters
from zea.probes import create_curved_probe_geometry

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


def test_grid_warns_on_aliasing():
    """An under-sized cartesian grid (pixel pitch > wavelength/2) warns about aliasing."""
    # scan_args sets grid_size_x=64, grid_size_z=128, which under-sample the imaging region.
    parameters = Parameters(**scan_args)
    with patch("zea.beamform.pixelgrid.log.warning") as mock_warn:
        _ = parameters.grid
    msgs = " ".join(str(c.args[0]) for c in mock_warn.call_args_list)
    assert "wavelength/2" in msgs


def test_grid_no_aliasing_warning_when_well_sampled():
    """A sufficiently dense cartesian grid does not warn."""
    args = scan_args.copy()
    args["grid_size_x"] = 512
    args["grid_size_z"] = 512
    parameters = Parameters(**args)
    with patch("zea.beamform.pixelgrid.log.warning") as mock_warn:
        _ = parameters.grid
    assert mock_warn.call_count == 0


def test_polar_grid_no_aliasing_warning():
    """The cartesian aliasing check is not applied to polar grids."""
    parameters = Parameters(**scan_args, grid_type="polar")
    with patch("zea.beamform.pixelgrid.log.warning") as mock_warn:
        _ = parameters.grid
    assert mock_warn.call_count == 0


def test_set_transmits_focused_excludes_plane_waves():
    """'focused' must select only finite-focus transmits, not plane waves (inf)."""
    local_scan_args = scan_args.copy()
    # Mix focused (finite > 0) and plane-wave (inf) transmits.
    focus = np.full(scan_args["n_tx"], np.inf)
    focus[: scan_args["n_tx"] // 2] = 0.04
    local_scan_args["focus_distances"] = focus

    parameters = Parameters(**local_scan_args)
    parameters.set_transmits("focused")

    assert list(parameters.selected_transmits) == list(range(scan_args["n_tx"] // 2))
    assert np.all(np.isfinite(parameters.focus_distances))


def _xlims_scan_args(focus_distances, polar_angles):
    """Build scan args for computing ``xlims`` (no explicit xlims/grid sizes)."""
    n_tx = len(focus_distances)
    aperture = np.linspace(-0.019, 0.019, 10)
    args = {
        "n_tx": n_tx,
        "n_el": 10,
        "n_ch": 1,
        "zlims": (0, 0.12),
        "center_frequency": 7e6,
        "sampling_frequency": 28e6,
        "sound_speed": 1540.0,
        "n_ax": 1024 * 16,
        "pixels_per_wavelength": 4,
        "polar_angles": np.asarray(polar_angles, dtype=np.float32),
        "focus_distances": np.asarray(focus_distances, dtype=np.float32),
        "probe_geometry": np.column_stack((aperture, np.zeros(10), np.zeros(10))),
    }
    return args


@pytest.mark.parametrize("focus", [np.inf, 0.0, -np.inf, 0.04])
def test_xlims_unsteered_hugs_aperture(focus):
    parameters = Parameters(**_xlims_scan_args([focus], [0.0]))

    aperture = (
        float(np.min(parameters.probe_geometry[:, 0])),
        float(np.max(parameters.probe_geometry[:, 0])),
    )
    assert np.allclose(parameters.xlims, aperture)


@pytest.mark.parametrize(
    "focus_distances, polar_angles",
    [
        ([-np.inf], [0.25]),
        ([0.05, 0.05], [0.75, 0]),
        ([-0.05], [0.0]),
    ],
)
def test_xlims_fans_out_by_fnumber(focus_distances, polar_angles):
    """Steered or diverging transmits use the f-number cone to select xlims."""
    parameters = Parameters(**_xlims_scan_args(focus_distances, polar_angles))

    aperture_min = float(np.min(parameters.probe_geometry[:, 0]))
    aperture_max = float(np.max(parameters.probe_geometry[:, 0]))
    reach = max(parameters.zlims) / (2 * parameters.f_number)
    assert np.allclose(parameters.xlims, (aperture_min - reach, aperture_max + reach))


def test_xlims_scales_with_fnumber():
    args = _xlims_scan_args([-0.02], [0.0])
    narrow = Parameters(**args, f_number=2.0)
    wide = Parameters(**args, f_number=1.0)

    assert narrow.xlims[0] > wide.xlims[0]
    assert narrow.xlims[1] < wide.xlims[1]


def test_xlims_zero_fnumber_uses_45_degree_cone():
    """``f_number=0`` disables the receive mask, so a 45 degree cone is used instead."""
    args = _xlims_scan_args([-0.02], [0.0])
    parameters = Parameters(**args, f_number=0.0)

    aperture_min = float(np.min(parameters.probe_geometry[:, 0]))
    aperture_max = float(np.max(parameters.probe_geometry[:, 0]))
    reach = max(parameters.zlims)
    assert np.allclose(parameters.xlims, (aperture_min - reach, aperture_max + reach))


def test_xlims_limited_by_record_length():
    args = _xlims_scan_args([-0.02], [0.0])
    args["n_ax"] = 1024
    parameters = Parameters(**args, f_number=0.0)  # 45 degree cone

    max_range = args["sound_speed"] * args["n_ax"] / args["sampling_frequency"] / 2
    reach = max_range * np.sin(np.pi / 4)
    aperture_min = float(np.min(parameters.probe_geometry[:, 0]))
    aperture_max = float(np.max(parameters.probe_geometry[:, 0]))

    assert reach < max(parameters.zlims)  # the depth-limited cone would reach further
    assert np.allclose(parameters.xlims, (aperture_min - reach, aperture_max + reach))


def test_xlims_curved_probe_widens_with_element_tilt():
    """A curved array's edge elements tilt outward, so xlims should also widen."""
    n_el, pitch, radius = 64, 1.6e-4, 0.03  # arc of +-10 degrees
    args = _xlims_scan_args([np.inf, np.inf], np.asarray([-0.3, 0.0], dtype=np.float32))
    args.update(
        n_el=n_el,
        n_ax=16384,
        zlims=(0, 0.15),
        probe_geometry=create_curved_probe_geometry(n_el=n_el, pitch=pitch, radius=radius),
    )
    parameters = Parameters(**args)

    tilt = (n_el - 1) * pitch / 2 / radius
    edge_x, edge_z = radius * np.sin(tilt), radius * np.cos(tilt) - radius
    half_angle = np.arctan(1 / (2 * parameters.f_number))
    expected = edge_x + (max(parameters.zlims) - edge_z) * np.tan(tilt + half_angle)

    assert np.allclose(parameters.xlims, (-expected, expected), rtol=1e-2)
    assert expected > edge_x + max(parameters.zlims) / (2 * parameters.f_number)


def test_xlims_unsteered_ignores_angle_noise():
    """Converted data stores a nominally unsteered scan as float noise, not exact zeros."""
    noise = np.full(4, 1e-9, dtype=np.float32)
    parameters = Parameters(**_xlims_scan_args([np.inf] * 4, noise))

    aperture = (
        float(np.min(parameters.probe_geometry[:, 0])),
        float(np.max(parameters.probe_geometry[:, 0])),
    )
    assert np.allclose(parameters.xlims, aperture)


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


def test_flat_aligned_apodization_derived():
    """Derived: None, unless scanline mode, where it is the one-hot transmit mask."""
    assert Parameters(**scan_args).flat_aligned_apodization is None

    scanline = Parameters(**scan_args, enable_scanline=True)
    n_tx, grid_size_z = scan_args["n_tx"], scan_args["grid_size_z"]
    assert scanline.flat_aligned_apodization.shape == (grid_size_z * n_tx, n_tx)


def test_flat_aligned_apodization_explicit_value_wins():
    """An explicit mask overrides the derived default, and follows the selection.

    It is stored over the full transmit axis, so a transmit selection slices it
    (and invalidates the cached value). The pixel axis matches the grid, which
    does not depend on the selection outside of scanline mode.
    """
    n_tx = scan_args["n_tx"]
    n_pix = scan_args["grid_size_z"] * scan_args["grid_size_x"]
    apodization = np.arange(n_pix * n_tx, dtype=np.float32).reshape(n_pix, n_tx)

    parameters = Parameters(**scan_args, flat_aligned_apodization=apodization)
    np.testing.assert_array_equal(parameters.flat_aligned_apodization, apodization)

    selection = [1, 3]
    parameters.set_transmits(selection)
    np.testing.assert_array_equal(parameters.flat_aligned_apodization, apodization[:, selection])
    # The mask stays aligned with the active grid: (n_pix, n_tx).
    assert parameters.flat_aligned_apodization.shape == (
        np.prod(parameters.grid.shape[:-1]),
        parameters.n_tx,
    )

    parameters.set_transmits("all")
    np.testing.assert_array_equal(parameters.flat_aligned_apodization, apodization)

    # Explicitly unsetting it falls back to the derived value.
    parameters.flat_aligned_apodization = None
    assert parameters.flat_aligned_apodization is None


def test_flat_aligned_apodization_rejected_in_scanline_mode():
    """A scanline grid derives its own mask; an explicit one cannot stay aligned with it."""
    n_tx, grid_size_z = scan_args["n_tx"], scan_args["grid_size_z"]
    apodization = np.ones((grid_size_z * n_tx, n_tx), dtype=np.float32)

    parameters = Parameters(
        **scan_args,
        enable_scanline=True,
        flat_aligned_apodization=apodization,
    )
    with pytest.raises(ValueError, match="enable_scanline"):
        _ = parameters.flat_aligned_apodization

    # Unsetting it (or disabling scanline mode) resolves the conflict either way.
    parameters.flat_aligned_apodization = None
    assert parameters.flat_aligned_apodization.shape == (grid_size_z * n_tx, n_tx)


def test_flat_aligned_apodization_transmit_axis_is_validated():
    """A mask over the selection instead of the full transmit axis is rejected."""
    n_pix = scan_args["grid_size_z"] * scan_args["grid_size_x"]

    parameters = Parameters(
        **scan_args,
        flat_aligned_apodization=np.ones((n_pix, 2), dtype=np.float32),
    )
    parameters.set_transmits([1, 3])
    with pytest.raises(ValueError, match="full transmit axis"):
        _ = parameters.flat_aligned_apodization


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


def test_t_peak_default_and_waveform_derived():
    """t_peak falls back to 1/f0, but is estimated from waveforms_two_way when provided."""
    center_frequency = 5e6
    n_tx = 3

    # Default: no waveform or explicit t_peak provided -> falls back to 1 / f0.
    parameters = Parameters(n_tx=n_tx, center_frequency=center_frequency)
    parameters.selected_transmits = "all"
    assert np.allclose(parameters.t_peak, 1 / center_frequency)

    # Build a synthetic pulse-echo waveform with a known envelope peak time.
    sampling_frequency = 250e6
    true_t_peak = 1.5e-6
    t = np.arange(512) / sampling_frequency
    pulse = np.exp(-((t - true_t_peak) ** 2) / (2 * (0.2e-6) ** 2)) * np.cos(
        2 * np.pi * center_frequency * (t - true_t_peak)
    )
    waveforms_two_way = np.tile(pulse, (n_tx, 1))

    parameters = Parameters(
        n_tx=n_tx,
        center_frequency=center_frequency,
        waveforms_two_way=waveforms_two_way,
    )
    parameters.selected_transmits = "all"
    assert np.allclose(parameters.t_peak, true_t_peak, atol=1e-8)

    # An explicitly provided t_peak still takes priority over the waveform estimate.
    explicit_t_peak = np.full(n_tx, 9e-7, dtype=np.float32)
    parameters = Parameters(
        n_tx=n_tx,
        center_frequency=center_frequency,
        waveforms_two_way=waveforms_two_way,
        t_peak=explicit_t_peak,
    )
    parameters.selected_transmits = "all"
    assert np.allclose(parameters.t_peak, explicit_t_peak)

    # waveforms_one_way alone is not used to derive t_peak.
    parameters = Parameters(
        n_tx=n_tx,
        center_frequency=center_frequency,
        waveforms_one_way=waveforms_two_way,
    )
    parameters.selected_transmits = "all"
    assert np.allclose(parameters.t_peak, 1 / center_frequency)


def test_missing_transmit_defaults_warn_once_on_access(monkeypatch, reset_warning_once):
    local_scan_args = scan_args.copy()
    local_scan_args.pop("azimuth_angles", None)
    local_scan_args.pop("t0_delays", None)
    local_scan_args.pop("tx_apodizations", None)
    local_scan_args.pop("focus_distances", None)
    local_scan_args.pop("transmit_origins", None)
    local_scan_args.pop("initial_times", None)
    local_scan_args.pop("tgc_gain_curve", None)

    warnings = []

    def _capture_warning(message, *args, **kwargs):
        warnings.append(message)
        return message

    monkeypatch.setattr("zea.parameters.log.warning", _capture_warning)

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


def test_missing_defaults_warn_once_per_scan_instance(monkeypatch, reset_warning_once):
    local_scan_args = scan_args.copy()
    local_scan_args.pop("azimuth_angles", None)

    warnings = []

    def _capture_warning(message, *args, **kwargs):
        warnings.append(message)
        return message

    monkeypatch.setattr("zea.parameters.log.warning", _capture_warning)

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

    parameters1 = get_parameters()
    parameters1.pfield_kwargs["norm"] = False

    parameters2 = get_parameters()
    assert parameters2.pfield_kwargs == {}, (
        "parameters2.pfield_kwargs seems to be affected by parameters1 modification"
    )
    assert parameters1 != parameters2, (
        "parameters1 and parameters2 should differ after modifying parameters1"
    )  # noqa: E501


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
        parameters = get_parameters(pfield_kwargs={"norm": True})
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

    parameters = get_parameters(pfield_kwargs={"norm": True})
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


# --- distance_to_apex ---


def test_distance_to_apex_fitted_from_curved_probe():
    """Left unset, the apex distance is the curved probe's radius of curvature."""
    from zea.probes import create_curved_probe_geometry

    radius = 49.57e-3
    parameters = Parameters(
        probe_geometry=create_curved_probe_geometry(n_el=128, pitch=0.508e-3, radius=radius)
    )
    assert parameters.distance_to_apex == pytest.approx(radius, rel=1e-5)


def test_distance_to_apex_is_zero_for_a_flat_probe():
    """A linear or phased array fans out from the transducer surface itself."""
    from zea.probes import create_probe_geometry

    parameters = Parameters(probe_geometry=create_probe_geometry(n_el=128, pitch=0.3e-3))
    assert parameters.distance_to_apex == 0.0


def test_distance_to_apex_is_zero_without_probe_geometry():
    """Nothing to fit, so fall back to an apex at the transducer surface."""
    assert Parameters().distance_to_apex == 0.0


def test_distance_to_apex_explicit_value_wins():
    """An explicit value overrides the fit, and both paths agree bit for bit."""
    from zea.probes import create_curved_probe_geometry

    probe_geometry = create_curved_probe_geometry(n_el=128, pitch=0.508e-3, radius=49.57e-3)
    assert Parameters(probe_geometry=probe_geometry, distance_to_apex=0.0).distance_to_apex == 0.0

    fitted = Parameters(probe_geometry=probe_geometry).distance_to_apex
    explicit = Parameters(probe_geometry=probe_geometry, distance_to_apex=fitted)
    assert explicit.distance_to_apex == fitted


def test_polar_grid_uses_the_fitted_apex():
    """The fitted apex reaches the grid: rays start one zlims[0] below the transducer."""
    from zea.probes import create_curved_probe_geometry

    radius = 49.57e-3
    parameters = Parameters(
        probe_geometry=create_curved_probe_geometry(n_el=128, pitch=0.508e-3, radius=radius),
        grid_type="polar",
        polar_limits=(-0.3, 0.3),
        zlims=(0.005, 0.07),
        grid_size_z=256,
        grid_size_x=128,
        center_frequency=3.5e6,
    )
    assert parameters.distance_to_apex == pytest.approx(radius, rel=1e-5)
    assert parameters.rho_range[0] == pytest.approx(0.005 + radius, rel=1e-5)
    # Centre ray starts at zlims[0] below the transducer, not at the apex.
    assert parameters.grid[0, 64, 2] == pytest.approx(0.005, abs=1e-6)
