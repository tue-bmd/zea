"""Test pressure field computation."""

import numpy as np
import pytest

from zea.beamform.delays import compute_t0_delays_planewave
from zea.beamform.pfield import compute_pfield
from zea.ops import Pipeline
from zea.parameters import Parameters
from zea.probes import Verasonics_l11_4v


def test_pfield():
    """Performs field computation on a scan object to verify that no errors occur.

    Note:
    - Does not check correctness of the output.
    - Only test with a plane wave type of scan.

    """

    probe = Verasonics_l11_4v()
    n_el = probe.n_el
    n_tx = 8

    tx_apodizations = np.ones((n_tx, n_el)) * np.hanning(n_el)[None]
    probe_geometry = probe.probe_geometry

    angles = np.linspace(10, -10, n_tx) * np.pi / 180

    focus_distances = np.ones(n_tx) * np.inf
    t0_delays = compute_t0_delays_planewave(
        probe_geometry=probe_geometry,
        polar_angles=angles,
    )

    parameters = Parameters(
        probe_geometry=probe.probe_geometry,
        n_tx=n_tx,
        n_el=n_el,
        xlims=(-19e-3, 19e-3),
        zlims=(0, 63e-3),
        n_ax=2047,
        sampling_frequency=probe.probe_center_frequency * 4,
        center_frequency=probe.probe_center_frequency,
        polar_angles=angles,
        t0_delays=t0_delays,
        focus_distances=focus_distances,
        tx_apodizations=tx_apodizations,
    )

    # Set scan grid parameters
    # The grid is updated automatically when it is accessed after the scan parameters
    # have been changed.
    dx = parameters.wavelength / 4
    dz = parameters.wavelength / 4
    parameters.grid_size_x = int(np.ceil((parameters.xlims[1] - parameters.xlims[0]) / dx))
    parameters.grid_size_z = int(np.ceil((parameters.zlims[1] - parameters.zlims[0]) / dz))

    pfield = parameters.pfield

    assert pfield.shape == (n_tx, parameters.grid_size_z, parameters.grid_size_x), (
        f"Expected pfield shape {(n_tx, parameters.grid_size_z, parameters.grid_size_x)}, "
        f"but got {pfield.shape}"
    )


def test_pfield_not_triggered():
    """Test that pfield is not computed when not needed for a Pipeline."""
    probe = Verasonics_l11_4v()
    parameters = Parameters(
        probe_geometry=probe.probe_geometry,
        n_tx=1,
        n_el=probe.n_el,
        xlims=(-20e-3, 20e-3),
        zlims=(0, 40e-3),
        n_ax=1024,
        sampling_frequency=probe.probe_center_frequency * 4,
        center_frequency=probe.probe_center_frequency,
    )

    pipeline = Pipeline.from_default(enable_pfield=False)
    inputs = pipeline.prepare_parameters(parameters)
    assert "flat_pfield" not in inputs and "pfield" not in inputs, (
        "pfield was computed in default pipeline but should not have been."
    )


def test_pfield_single_transmit_auto_xlims_is_finite():
    """Regression: grid points coinciding with an element must not produce NaNs.

    With a single 0° plane wave and auto-derived xlims, the grid spans the aperture
    exactly and starts at z=0, so some grid points land on an element position.
    """
    probe = Verasonics_l11_4v()
    n_el = probe.n_el
    angles = np.zeros(1)
    t0_delays = compute_t0_delays_planewave(
        probe_geometry=probe.probe_geometry,
        polar_angles=angles,
    )

    parameters = Parameters(
        probe_geometry=probe.probe_geometry,
        n_tx=1,
        n_el=n_el,
        zlims=(0, 40e-3),
        n_ax=1024,
        sampling_frequency=probe.probe_center_frequency * 4,
        center_frequency=probe.probe_center_frequency,
        polar_angles=angles,
        t0_delays=t0_delays,
        focus_distances=np.array([np.inf]),
        tx_apodizations=np.ones((1, n_el)),
    )

    pfield = np.asarray(parameters.pfield)
    assert np.all(np.isfinite(pfield)), (
        f"pfield contains {np.sum(~np.isfinite(pfield))} non-finite values"
    )


_CENTER_FREQUENCY = 5e6
_N_EL = 16


def _small_array_geometry():
    geometry = np.zeros((_N_EL, 3))
    geometry[:, 0] = (np.arange(_N_EL) - (_N_EL - 1) / 2) * 3e-4
    return geometry


def _small_array_pfield(grid, tx_apodizations=None, **kwargs):
    """compute_pfield on a small unfocused array, with no thresholding of the field."""
    if tx_apodizations is None:
        tx_apodizations = np.ones((1, _N_EL))
    pfield = compute_pfield(
        sound_speed=1540.0,
        center_frequency=_CENTER_FREQUENCY,
        probe_bandwidth_percent=60.0,
        probe_geometry=_small_array_geometry(),
        tx_apodizations=tx_apodizations,
        grid=grid,
        t0_delays=np.zeros_like(tx_apodizations, dtype=float),
        element_width=2.7e-4,
        frequency_step=1,
        downsample=1,
        percentile=0,
        **kwargs,
    )
    return np.asarray(pfield)


def _on_axis_grid(depths):
    grid = np.zeros((len(depths), 1, 3))
    grid[:, 0, 2] = depths
    return grid


@pytest.mark.parametrize("downmix", [1, 2, 4])
def test_pfield_attenuation_follows_true_frequency(downmix):
    """The one-way loss matches Eq. 40 at the true frequency, whatever the downmix."""
    attenuation_coef = 0.5  # dB/(cm·MHz)
    depths = np.array([0.01, 0.03, 0.05, 0.07])

    def on_axis_pressure(coef):
        pfield = _small_array_pfield(
            _on_axis_grid(depths), downmix=downmix, norm=False, attenuation_coef=coef
        )
        return pfield[0, :, 0]

    loss_db = 20 * np.log10(on_axis_pressure(0.0) / on_axis_pressure(attenuation_coef))
    expected_db = attenuation_coef * (depths * 100) * (_CENTER_FREQUENCY / 1e6)
    # Attenuation shifts the retained band below the center frequency, so the loss is
    # slightly under the value at the center frequency.
    np.testing.assert_allclose(loss_db, expected_db, rtol=0.15)


@pytest.mark.parametrize("downmix", [2, 4])
def test_pfield_unnormalized_amplitude_is_consistent_across_downmix(downmix):
    """Downmixing must not rescale the unnormalized field, beyond its effect on the beam."""
    # Past the near field of this aperture (~2 cm), where downmixing changes the
    # interference pattern itself and the amplitudes are not comparable.
    grid = _on_axis_grid(np.array([0.03, 0.05, 0.07]))

    reference = _small_array_pfield(grid, downmix=1, norm=False)
    pfield = _small_array_pfield(grid, downmix=downmix, norm=False)

    np.testing.assert_allclose(pfield, reference, rtol=0.2)


def test_pfield_attenuation_changes_normalized_weights():
    """Unequal transmit paths lose different amounts, which shifts the normalized weights."""
    geometry = _small_array_geometry()
    # Two transmits, one from each outer element.
    tx_apodizations = np.zeros((2, _N_EL))
    tx_apodizations[0, 0] = 1
    tx_apodizations[1, -1] = 1
    # A single point straight below the first element, so its path is the shorter one.
    grid = np.zeros((1, 1, 3))
    grid[0, 0, 0] = geometry[0, 0]
    grid[0, 0, 2] = 0.02

    lossless = _small_array_pfield(grid, tx_apodizations, attenuation_coef=0.0)[:, 0, 0]
    attenuated = _small_array_pfield(grid, tx_apodizations, attenuation_coef=0.5)[:, 0, 0]

    assert attenuated[0] > lossless[0] + 1e-3
    np.testing.assert_allclose(np.sum(attenuated), 1.0, rtol=1e-5)


@pytest.mark.parametrize("attenuation_coef", [float("nan"), float("inf"), -0.5])
def test_pfield_rejects_invalid_attenuation_coef(attenuation_coef):
    """Non-finite or negative attenuation raises instead of returning a corrupt field."""
    with pytest.raises(ValueError, match="attenuation_coef"):
        _small_array_pfield(_on_axis_grid(np.array([0.03])), attenuation_coef=attenuation_coef)


def _capture_pfield_kwargs(monkeypatch):
    """Replace compute_pfield in zea.parameters by a stub recording its keyword arguments."""
    calls = []

    def fake_compute_pfield(**kwargs):
        calls.append(kwargs)
        return np.zeros((1, 2, 2))

    monkeypatch.setattr("zea.parameters.compute_pfield", fake_compute_pfield)
    return calls


def _single_transmit_parameters(**kwargs):
    probe = Verasonics_l11_4v()
    angles = np.zeros(1)
    return Parameters(
        probe_geometry=probe.probe_geometry,
        n_tx=1,
        n_el=probe.n_el,
        zlims=(0, 40e-3),
        n_ax=1024,
        sampling_frequency=probe.probe_center_frequency * 4,
        center_frequency=probe.probe_center_frequency,
        polar_angles=angles,
        t0_delays=compute_t0_delays_planewave(
            probe_geometry=probe.probe_geometry, polar_angles=angles
        ),
        focus_distances=np.array([np.inf]),
        tx_apodizations=np.ones((1, probe.n_el)),
        **kwargs,
    )


def test_parameters_forward_attenuation_coef_to_pfield(monkeypatch):
    """pfield uses attenuation_coef, and recomputes when it changes."""
    calls = _capture_pfield_kwargs(monkeypatch)
    parameters = _single_transmit_parameters(attenuation_coef=0.5)

    _ = parameters.pfield
    assert calls[-1]["attenuation_coef"] == pytest.approx(0.5)

    parameters.attenuation_coef = 0.7
    _ = parameters.pfield
    assert len(calls) == 2
    assert calls[-1]["attenuation_coef"] == pytest.approx(0.7)


def test_pfield_kwargs_override_attenuation_coef(monkeypatch):
    """An attenuation_coef in pfield_kwargs takes precedence over the parameter."""
    calls = _capture_pfield_kwargs(monkeypatch)
    parameters = _single_transmit_parameters(
        attenuation_coef=0.5, pfield_kwargs={"attenuation_coef": 0.0}
    )

    _ = parameters.pfield
    assert calls[-1]["attenuation_coef"] == 0.0
