"""Test pressure field computation."""

import numpy as np

from zea.beamform.delays import compute_t0_delays_planewave
from zea.ops import Pipeline
from zea.probes import Verasonics_l11_4v
from zea.scan import Scan


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

    scan = Scan(
        probe_geometry=probe.probe_geometry,
        n_tx=n_tx,
        n_el=n_el,
        xlims=(-19e-3, 19e-3),
        zlims=(0, 63e-3),
        n_ax=2047,
        sampling_frequency=probe.sampling_frequency,
        center_frequency=probe.center_frequency,
        polar_angles=np.array([0]),
        t0_delays=t0_delays,
        focus_distances=focus_distances,
        tx_apodizations=tx_apodizations,
    )

    # Set scan grid parameters
    # The grid is updated automatically when it is accessed after the scan parameters
    # have been changed.
    dx = scan.wavelength / 4
    dz = scan.wavelength / 4
    scan.grid_size_x = int(np.ceil((scan.xlims[1] - scan.xlims[0]) / dx))
    scan.grid_size_z = int(np.ceil((scan.zlims[1] - scan.zlims[0]) / dz))

    pfield = scan.pfield

    assert pfield.shape == (n_tx, scan.grid_size_z, scan.grid_size_x), (
        f"Expected pfield shape {(n_tx, scan.grid_size_z, scan.grid_size_x)}, "
        f"but got {pfield.shape}"
    )


def test_pfield_not_triggered():
    """Test that pfield is not computed when not needed for a Pipeline."""
    probe = Verasonics_l11_4v()
    scan = Scan(
        probe_geometry=probe.probe_geometry,
        n_tx=1,
        n_el=probe.n_el,
        xlims=(-20e-3, 20e-3),
        zlims=(0, 40e-3),
        n_ax=1024,
        sampling_frequency=probe.sampling_frequency,
        center_frequency=probe.center_frequency,
    )

    pipeline = Pipeline.from_default(enable_pfield=False)
    parameters = pipeline.prepare_parameters(probe, scan)
    assert "flat_pfield" not in parameters and "pfield" not in parameters, (
        "pfield was computed in default pipeline but should not have been."
    )
