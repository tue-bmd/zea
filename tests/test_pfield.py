"""Test pressure field computation."""

import numpy as np

from zea.beamform.delays import compute_t0_delays_planewave
from zea.ops import Pipeline
from zea.probes import Verasonics_l11_4v
from zea.scan import Parameters


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
