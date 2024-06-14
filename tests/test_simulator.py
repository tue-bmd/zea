"""
Test the ultrasound simulator.
"""

from usbmd.probes import Verasonics_l11_4v
from usbmd.scan import PlaneWaveScan
from usbmd.utils.simulator import UltrasoundSimulator


def test_simulator():
    """Test ultrasound the simulator."""
    probe = Verasonics_l11_4v()
    probe_parameters = probe.get_parameters()
    scan = PlaneWaveScan(
        probe_geometry=probe.probe_geometry,
        n_tx=1,
        xlims=(-19e-3, 19e-3),
        zlims=(0, 63e-3),
        n_ax=2047,
        sampling_frequency=probe_parameters["sampling_frequency"],
        center_frequency=probe_parameters["center_frequency"],
        angles=[0],
    )

    simulator = UltrasoundSimulator(probe, scan)
    simulator.generate(200)
    simulator.generate()


def test_simulator_without_scan_probe():
    """Test ultrasound the simulator without scan and probe class."""
    simulator = UltrasoundSimulator()
    simulator.generate(200)
    simulator.generate()
