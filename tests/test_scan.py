"""Tests for the Parameters class."""

import pickle

import numpy as np
import pytest

from zea import Parameters, ops
from zea.beamform.pixelgrid import aliasing_limits, check_for_aliasing, transmit_direction_range
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


def _aliasing_args(parameters, grid=True):
    """The attributes of ``parameters`` that the aliasing functions take."""
    args = {
        "sound_speed": parameters.sound_speed,
        "center_frequency": parameters.center_frequency,
        "f_number": parameters.f_number,
        "demodulation_frequency": parameters.demodulation_frequency,
    }
    if grid:
        args.update(
            xlims=parameters.xlims,
            zlims=parameters.zlims,
            grid_size_x=parameters.grid_size_x,
            grid_size_z=parameters.grid_size_z,
        )
    return args


# `scan_args` steers its focused transmits over +-90 deg, which opens the transmit
# aperture to the full half-space; the aliasing tests want an unsteered one.
focused_scan_args = {**scan_args, "polar_angles": np.zeros(10)}


def _transmit_args(parameters):
    return {
        "focus_distances": parameters.focus_distances,
        "probe_geometry": parameters.probe_geometry,
        "tx_apodizations": parameters.tx_apodizations,
        "polar_angles": parameters.polar_angles,
        "transmit_origins": parameters.transmit_origins,
    }


def test_check_for_aliasing_rf_grid():
    """An under-sized cartesian grid carrying RF warns on both axes."""
    # scan_args sets grid_size_x=64, grid_size_z=128, which under-sample the region.
    parameters = Parameters(**focused_scan_args)
    msgs = check_for_aliasing(
        **_aliasing_args(parameters), **_transmit_args(parameters), demodulated=False
    )
    assert any("Lateral" in m for m in msgs)
    assert any("Axial" in m and "RF, top of the band" in m for m in msgs)


def test_aliasing_limits_rf_axial_is_strict_nyquist():
    """RF needs lambda_min / 4 axially: the round trip doubles k_z, the real image mirrors it."""
    limits = aliasing_limits(1540.0, 5e6, 1.0, **_plane_waves(0), bandwidth=2e6)
    assert limits["dz_max"] == pytest.approx(1540.0 / (4 * 6e6))


def test_check_for_aliasing_pitch_counts_both_endpoints():
    """Pixels sit on both limits, so n pixels span n - 1 pitches."""
    dz_max = aliasing_limits(1540.0, 5e6, 1.0, **_plane_waves(0))["dz_max"]
    args = dict(
        xlims=(0.0, 1e-3),
        grid_size_x=1024,
        sound_speed=1540.0,
        center_frequency=5e6,
        f_number=1.0,
        **_plane_waves(0),
    )
    # 100 pitches of exactly dz_max need 101 pixels; 100 pixels are too coarse.
    assert check_for_aliasing(zlims=(0.0, 100 * dz_max), grid_size_z=101, **args) == []
    assert check_for_aliasing(zlims=(0.0, 100 * dz_max), grid_size_z=100, **args) != []


def test_check_for_aliasing_reversed_limits():
    """Reversed limits give the same pitch, and so the same warnings, as ordered ones."""
    parameters = Parameters(**focused_scan_args)
    args = {**_aliasing_args(parameters), **_transmit_args(parameters), "demodulated": False}
    flipped = {**args, "xlims": args["xlims"][::-1], "zlims": args["zlims"][::-1]}
    assert check_for_aliasing(**flipped) == check_for_aliasing(**args) != []


def test_check_for_aliasing_accepts_automatic_grid_size():
    """A grid sized from ``pixels_per_wavelength`` passes the check it is meant to satisfy."""
    args = {k: v for k, v in scan_args.items() if k not in ("grid_size_x", "grid_size_z")}
    parameters = Parameters(**args)
    assert (
        check_for_aliasing(
            **_aliasing_args(parameters), **_transmit_args(parameters), demodulated=False
        )
        == []
    )


def test_check_for_aliasing_no_warning_when_well_sampled():
    """A sufficiently dense cartesian grid does not warn."""
    args = focused_scan_args.copy()
    args["grid_size_x"] = 1024
    args["grid_size_z"] = 1024
    parameters = Parameters(**args)
    assert (
        check_for_aliasing(
            **_aliasing_args(parameters), **_transmit_args(parameters), demodulated=False
        )
        == []
    )


def test_check_for_aliasing_axial_relaxed_by_demodulation():
    """Demodulation removes the carrier, so the axial limit follows the bandwidth.

    The lateral limit has no carrier to remove and must be unaffected.
    """
    args = focused_scan_args.copy()
    args["grid_size_z"] = 128  # dz = 312 um: aliases at 7 MHz RF (110 um)...
    parameters = Parameters(**args)

    rf = check_for_aliasing(
        **_aliasing_args(parameters), **_transmit_args(parameters), demodulated=False
    )
    assert any("Axial" in m for m in rf)

    # ...but a 1 MHz band is only ~2155 cycles/m tall in k_z (464 um), even with the
    # oblique paths of this wide aperture, so IQ fits comfortably.
    iq = check_for_aliasing(
        **_aliasing_args(parameters), **_transmit_args(parameters), demodulated=True, bandwidth=1e6
    )
    assert not any("Axial" in m for m in iq)
    assert any("Lateral" in m for m in iq)


def test_aliasing_limits_transmit_angle_from_geometry():
    """The lateral limit uses the real transmit aperture, not a symmetric guess."""
    parameters = Parameters(**focused_scan_args)
    limits = aliasing_limits(
        **_aliasing_args(parameters, grid=False),
        **_transmit_args(parameters),
        demodulated=True,
        bandwidth=2e6,
    )
    # 38 mm aperture focused at 40 mm -> sin(theta_t) spans +-19 / hypot(40, 19).
    sin_theta_t = 19 / np.hypot(40, 19)
    x_range = limits["transmit_direction"][0]
    assert x_range == pytest.approx((-sin_theta_t, sin_theta_t), rel=1e-6)
    # f_number defaults to 1.0 -> sin(theta_r) = 1 / sqrt(5).
    assert limits["sin_theta_r"] == pytest.approx(1 / np.sqrt(5), rel=1e-6)


def test_transmit_direction_range_steered_focus():
    """A steered focus skews the range to the edge rays through the focal point."""
    transmits = _plane_waves(20, n_el=3, pitch=10e-3)
    focus, angle = 40e-3, transmits["polar_angles"][0]
    transmits["focus_distances"] = np.array([focus])
    sin_range = transmit_direction_range(**transmits)[:, 0]

    focus_x, focus_z = focus * np.sin(angle), focus * np.cos(angle)
    edge_rays = [(focus_x - x) / np.hypot(focus_x - x, focus_z) for x in (10e-3, -10e-3)]
    np.testing.assert_allclose(sin_range, [edge_rays])

    # Unsteered it reduces to the symmetric +-s; mirrored for a diverging wave.
    transmits["polar_angles"] = np.zeros(1)
    s = 10 / np.hypot(40, 10)
    np.testing.assert_allclose(transmit_direction_range(**transmits)[:, 0], [[-s, s]])
    transmits["focus_distances"] = -transmits["focus_distances"]
    np.testing.assert_allclose(transmit_direction_range(**transmits)[:, 0], [[-s, s]])


def test_aliasing_limits_symmetric_fallback_without_active_elements():
    """Without an active transmit element the textbook symmetric aperture is assumed."""
    transmits = {**_plane_waves(0), "focus_distances": np.array([40e-3])}
    transmits["tx_apodizations"] = np.zeros_like(transmits["tx_apodizations"])
    limits = aliasing_limits(1540.0, 7e6, 1.0, **transmits, demodulated=False)
    sin_theta_r = limits["sin_theta_r"]
    np.testing.assert_allclose(limits["transmit_direction"][:2], [[-sin_theta_r, sin_theta_r]] * 2)


def _plane_waves(degrees, n_el=128, pitch=0.15e-3):
    """Transmit geometry of plane waves at the given angles."""
    angles = np.deg2rad(np.atleast_1d(degrees))
    return {
        "focus_distances": np.full(len(angles), np.inf),
        "probe_geometry": np.column_stack(
            [(np.arange(n_el) - (n_el - 1) / 2) * pitch, np.zeros(n_el), np.zeros(n_el)]
        ),
        "tx_apodizations": np.ones((len(angles), n_el)),
        "polar_angles": angles,
        "transmit_origins": np.zeros((len(angles), 3)),
    }


def test_aliasing_limits_single_steered_plane_wave_not_widened():
    """A single steered plane wave shifts the lateral band but does not widen it.

    With sin(alpha) < sin(theta_r) the band spans exactly 2 * f_max * sin(theta_r) / c,
    whatever the angle, so the lateral limit matches the unsteered one.
    """
    args = dict(sound_speed=1540.0, center_frequency=5e6, f_number=1.0, bandwidth=4e6)
    unsteered = aliasing_limits(**args, **_plane_waves(0))
    steered = aliasing_limits(**args, **_plane_waves(15))
    assert steered["dx_max"] == pytest.approx(unsteered["dx_max"])
    assert unsteered["dx_max"] == pytest.approx(1540.0 / 7e6 / (2 / np.sqrt(5)))


def test_aliasing_limits_compounding_widens_lateral_band():
    """Compounding over +-alpha gives the symmetric lambda_min / (2 (sin a + sin r))."""
    args = dict(sound_speed=1540.0, center_frequency=5e6, f_number=1.0, bandwidth=4e6)
    limits = aliasing_limits(**args, **_plane_waves([-15, 0, 15]))
    expected = 1540.0 / 7e6 / (2 * (np.sin(np.deg2rad(15)) + 1 / np.sqrt(5)))
    assert limits["dx_max"] == pytest.approx(expected)


def _matrix_plane_waves(polar_degrees, azimuth_degrees, n_side=32, pitch=0.3e-3):
    """Transmit geometry of plane waves from a square matrix probe."""
    side = (np.arange(n_side) - (n_side - 1) / 2) * pitch
    x, y = np.meshgrid(side, side)
    transmits = _plane_waves(polar_degrees, n_el=n_side**2)
    transmits["probe_geometry"] = np.column_stack([x.ravel(), y.ravel(), np.zeros(n_side**2)])
    transmits["azimuth_angles"] = np.deg2rad(np.atleast_1d(azimuth_degrees))
    return transmits


def test_aliasing_limits_azimuth_compounding_widens_elevational_band():
    """Plane waves steered over +-alpha in y widen the y band, not the x band."""
    args = dict(sound_speed=1540.0, center_frequency=5e6, f_number=1.0, bandwidth=4e6)
    limits = aliasing_limits(**args, **_matrix_plane_waves([15, 0, 15], [-90, 0, 90]))
    expected = 1540.0 / 7e6 / (2 * (np.sin(np.deg2rad(15)) + 1 / np.sqrt(5)))
    assert limits["dy_max"] == pytest.approx(expected)
    assert limits["dx_max"] == pytest.approx(1540.0 / 7e6 / (2 / np.sqrt(5)))


def test_transmit_direction_range_matrix_focus():
    """An unsteered focus on a square aperture spans +-s in x and y; on axis in z."""
    transmits = _matrix_plane_waves(0, 0, n_side=3, pitch=10e-3)
    transmits["focus_distances"] = np.array([40e-3])
    (x_range, y_range, z_range) = transmit_direction_range(**transmits)[0]
    s = 10 / np.hypot(40, 10)
    np.testing.assert_allclose(x_range, [-s, s])
    np.testing.assert_allclose(y_range, [-s, s])
    # The corner elements give the most oblique ray.
    np.testing.assert_allclose(z_range, [40 / np.sqrt(40**2 + 2 * 10**2), 1.0])


def test_check_for_aliasing_3d_grid_checks_elevation():
    """A coarse y pitch warns only when the grid has a y extent."""
    args = dict(
        xlims=(-5e-3, 5e-3),
        zlims=(0.0, 1e-3),
        grid_size_x=1024,
        grid_size_z=1024,
        sound_speed=1540.0,
        center_frequency=5e6,
        f_number=1.0,
        **_matrix_plane_waves(0, 0),
    )
    msgs = check_for_aliasing(**args, ylims=(-5e-3, 5e-3), grid_size_y=11)
    assert len(msgs) == 1 and "Elevational" in msgs[0] and "grid_size_y" in msgs[0]
    assert check_for_aliasing(**args, ylims=(-5e-3, 5e-3), grid_size_y=1024) == []
    assert check_for_aliasing(**args) == []


def test_aliasing_limits_iq_axial_includes_oblique_paths():
    """Oblique paths make the IQ band taller in k_z than 2 B / c."""
    args = dict(sound_speed=1540.0, center_frequency=5e6, bandwidth=4e6, demodulated=True)
    limits = aliasing_limits(**args, f_number=0.5, **_plane_waves(0))
    # k_z from 3 MHz * (1 + cos(theta_r)) / c to 7 MHz * 2 / c, sin(theta_r) = 1 / sqrt(2).
    expected = 1540.0 / (2 * 7e6 - 3e6 * (1 + 1 / np.sqrt(2)))
    assert limits["dz_max"] == pytest.approx(expected)
    assert limits["dz_max"] < 1540.0 / (2 * 4e6)


@pytest.fixture
def recorded_aliasing_checks(monkeypatch):
    """Record the calls to ``Parameters.check_for_aliasing`` instead of running them."""
    calls = []

    def record(self, demodulated=None, bandwidth=None):
        calls.append({"parameters": self, "demodulated": demodulated})
        return []

    monkeypatch.setattr(Parameters, "check_for_aliasing", record)
    return calls


def test_prepare_parameters_checks_aliasing_from_tof_correction(recorded_aliasing_checks):
    """TOFCorrection runs the aliasing check, stating demodulation from the ops before it."""
    parameters = get_parameters()

    ops.Pipeline([ops.TOFCorrection()]).prepare_parameters(parameters)
    ops.Pipeline([ops.Demodulate(), ops.Pipeline([ops.TOFCorrection()])]).prepare_parameters(
        parameters
    )
    assert [call["demodulated"] for call in recorded_aliasing_checks] == [None, True]

    # No grid, nothing to check; and the check can be turned off.
    ops.Pipeline([ops.Demodulate()]).prepare_parameters(parameters)
    ops.Pipeline([ops.TOFCorrection()]).prepare_parameters(parameters, check=False)
    assert len(recorded_aliasing_checks) == 2


def test_prepare_parameters_check_sees_overrides(recorded_aliasing_checks):
    """Overrides reach the check, without touching the caller's parameters."""
    parameters = get_parameters()
    ops.Pipeline([ops.TOFCorrection()]).prepare_parameters(parameters, f_number=0.0)
    assert recorded_aliasing_checks[0]["parameters"].f_number == 0.0
    assert parameters.f_number != 0.0


def test_prepare_parameters_survives_failing_check(monkeypatch):
    """A check that raises is logged and skipped; the parameters are still prepared."""

    def fail(self, parameters, upstream):
        raise RuntimeError("broken check")

    monkeypatch.setattr(ops.TOFCorrection, "check_parameters", fail)
    inputs = ops.Pipeline([ops.TOFCorrection()]).prepare_parameters(get_parameters())
    assert "flatgrid" in inputs


def test_parameters_check_for_aliasing_rf_vs_iq():
    """The axial limit depends on whether the grid carries RF or IQ."""
    parameters = Parameters(**focused_scan_args)  # dz = 315 um: aliases for RF only
    rf = parameters.check_for_aliasing(demodulated=False, bandwidth=1e6)
    iq = parameters.check_for_aliasing(demodulated=True, bandwidth=1e6)
    assert any("Axial" in m for m in rf)
    assert not any("Axial" in m for m in iq)


def test_parameters_check_for_aliasing_uses_bandwidth_parameter():
    """The ``bandwidth`` parameter is used when no ``bandwidth`` argument is passed."""
    without = Parameters(**focused_scan_args).check_for_aliasing(demodulated=True)
    with_bandwidth = Parameters(**focused_scan_args, bandwidth=1e6).check_for_aliasing(
        demodulated=True
    )
    assert any("Axial" in m for m in without)
    assert not any("Axial" in m for m in with_bandwidth)


def test_parameters_check_for_aliasing_assumes_probe_bandwidth():
    """Without a ``bandwidth``, RF on a lambda / 4 grid warns: the band reaches past fc."""
    wavelength = focused_scan_args["sound_speed"] / focused_scan_args["center_frequency"]
    grid_size_z = int(np.ceil(0.04 / (wavelength / 4.1))) + 1  # just finer than lambda / 4
    args = {**focused_scan_args, "grid_size_z": grid_size_z}
    assert any("Axial" in m for m in Parameters(**args).check_for_aliasing(demodulated=False))

    narrow = Parameters(**args, probe_bandwidth_percent=1.0).check_for_aliasing(demodulated=False)
    assert not any("Axial" in m for m in narrow)


def test_parameters_check_for_aliasing_skips_polar_grid():
    """Non-cartesian grids are not checked."""
    assert Parameters(**focused_scan_args, grid_type="polar").check_for_aliasing() == []


def test_parameters_check_for_aliasing_n_ch_marks_iq():
    """Data handed over as IQ (``n_ch == 2``) is treated as demodulated."""
    parameters = Parameters(**{**focused_scan_args, "n_ch": 2})
    msgs = parameters.check_for_aliasing(bandwidth=1e6)
    assert not any("Axial" in m for m in msgs)

    # An explicit `demodulated=False` takes priority over `n_ch`.
    msgs = parameters.check_for_aliasing(demodulated=False, bandwidth=1e6)
    assert any("Axial" in m for m in msgs)


def test_aliasing_limits_directivity_caps_receive():
    """Element directivity caps the receive term at 0.603 c / w, at any frequency."""
    args = dict(sound_speed=1540.0, center_frequency=5e6, f_number=0.0, bandwidth=4e6)
    width = 0.3e-3
    uncapped = aliasing_limits(**args, **_plane_waves(0))
    capped = aliasing_limits(**args, **_plane_waves(0), element_width=width)
    assert uncapped["dx_max"] == pytest.approx(1540.0 / 7e6 / 2)
    assert capped["dx_max"] == pytest.approx(width / (2 * 0.603))
    # A tiny element caps nothing within the band.
    tiny = aliasing_limits(**args, **_plane_waves(0), element_width=1e-6)
    assert tiny["dx_max"] == pytest.approx(uncapped["dx_max"])


def test_aliasing_limits_directivity_spares_plane_wave_steering():
    """A steered plane wave keeps its direction; only the receive term is capped."""
    args = dict(sound_speed=1540.0, center_frequency=5e6, f_number=0.0, bandwidth=4e6)
    width = 0.3e-3
    limits = aliasing_limits(**args, **_plane_waves([-60, 60]), element_width=width)
    k_edge = (7e6 * np.sin(np.deg2rad(60)) + 0.603 * 1540.0 / width) / 1540.0
    assert limits["dx_max"] == pytest.approx(1 / (2 * k_edge))


def test_aliasing_limits_directivity_caps_diverging_transmit():
    """Diverging rays are capped like receive paths, which relaxes the lateral limit."""
    transmits = {**_plane_waves(0), "focus_distances": np.array([-1e-3])}
    args = dict(sound_speed=1540.0, center_frequency=5e6, f_number=0.0, bandwidth=4e6)
    width = 0.3e-3
    uncapped = aliasing_limits(**args, **transmits)
    capped = aliasing_limits(**args, **transmits, element_width=width)
    # Both terms at their cap: k_x spans +-2 * 0.603 / w.
    assert capped["dx_max"] == pytest.approx(width / (4 * 0.603))
    assert capped["dx_max"] > uncapped["dx_max"]


def test_parameters_check_for_aliasing_without_polar_angles():
    """Missing ``polar_angles`` are treated as unsteered transmits."""
    args = {k: v for k, v in focused_scan_args.items() if k != "polar_angles"}
    without = Parameters(**args).check_for_aliasing(demodulated=False)
    assert without == Parameters(**focused_scan_args).check_for_aliasing(demodulated=False)


def test_transmit_direction_range_skips_nan_focus():
    """A ``NaN`` focus distance marks no transmit type, so that transmit is left out."""
    transmits = {**_plane_waves([0, 0]), "focus_distances": np.array([np.nan, 40e-3])}
    ranges = transmit_direction_range(**transmits)
    assert ranges.shape == (1, 3, 2)
    assert np.all(np.isfinite(ranges))


def test_transmit_direction_range_keeps_horizontal_rays():
    """A ray level with the focus (z component 0) keeps its direction."""
    transmits = _plane_waves(0, n_el=2, pitch=20e-3)
    transmits["probe_geometry"][0, 2] = 10e-3  # first element at the focal depth
    transmits["focus_distances"] = np.array([10e-3])
    x_range, _, z_range = transmit_direction_range(**transmits)[0]
    np.testing.assert_allclose(x_range, [-1 / np.sqrt(2), 1.0])
    assert z_range[0] == pytest.approx(0.0)


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


def test_rx_aperture_indices_absent_by_default():
    """Without a receive sub-aperture there is no mapping, and n_rx is n_el."""
    parameters = Parameters(**scan_args)
    assert parameters.rx_aperture_indices is None
    assert parameters.n_rx == parameters.n_el == scan_args["n_el"]


def test_rx_aperture_indices_follows_transmit_selection():
    """The mapping is stored over the full transmit axis, so a selection slices it."""
    n_tx, n_el, n_rx = scan_args["n_tx"], scan_args["n_el"], 4
    # A sliding receive window: transmit i receives on elements i .. i + n_rx - 1.
    indices = (np.arange(n_rx)[None] + np.arange(n_tx)[:, None]) % n_el
    indices = indices.astype(np.int32)

    parameters = Parameters(**scan_args, rx_aperture_indices=indices)
    np.testing.assert_array_equal(parameters.rx_aperture_indices, indices)
    assert parameters.n_rx == n_rx
    assert parameters.n_el == n_el

    selection = [1, 3]
    parameters.set_transmits(selection)
    np.testing.assert_array_equal(parameters.rx_aperture_indices, indices[selection])
    # The mapping stays aligned with the other per-transmit parameters.
    assert parameters.rx_aperture_indices.shape == (parameters.n_tx, parameters.n_rx)
    assert parameters.t0_delays.shape[0] == parameters.rx_aperture_indices.shape[0]

    parameters.set_transmits("all")
    np.testing.assert_array_equal(parameters.rx_aperture_indices, indices)


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
        sampling_frequency=20e6,
        n_ax=4096,
    )
    assert parameters.distance_to_apex == pytest.approx(radius, rel=1e-5)
    assert parameters.rho_range[0] == pytest.approx(0.005 + radius, rel=1e-5)
    # Centre ray starts at zlims[0] below the transducer, not at the apex.
    assert parameters.grid[0, 64, 2] == pytest.approx(0.005, abs=1e-6)
