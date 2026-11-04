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


@pytest.mark.parametrize("downmix", [1, 2, 4])
def test_pfield_attenuation_follows_true_frequency(downmix):
    """The one-way loss matches Eq. 40 at the true frequency, whatever the downmix."""
    center_frequency = 5e6
    attenuation_coef = 0.5  # dB/(cm·MHz)
    n_el = 16
    geometry = np.zeros((n_el, 3))
    geometry[:, 0] = (np.arange(n_el) - (n_el - 1) / 2) * 3e-4
    depths = np.array([0.01, 0.03, 0.05, 0.07])
    grid = np.zeros((len(depths), 1, 3))
    grid[:, 0, 2] = depths

    def on_axis_pressure(coef):
        pfield = compute_pfield(
            sound_speed=1540.0,
            center_frequency=center_frequency,
            probe_bandwidth_percent=60.0,
            probe_geometry=geometry,
            tx_apodizations=np.ones((1, n_el)),
            grid=grid,
            t0_delays=np.zeros((1, n_el)),
            element_width=2.7e-4,
            frequency_step=1,
            downsample=1,
            downmix=downmix,
            percentile=0,
            norm=False,
            attenuation_coef=coef,
        )
        return np.asarray(pfield)[0, :, 0]

    loss_db = 20 * np.log10(on_axis_pressure(0.0) / on_axis_pressure(attenuation_coef))
    expected_db = attenuation_coef * (depths * 100) * (center_frequency / 1e6)
    # Attenuation shifts the retained band below the center frequency, so the loss is
    # slightly under the value at the center frequency.
    np.testing.assert_allclose(loss_db, expected_db, rtol=0.15)


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
