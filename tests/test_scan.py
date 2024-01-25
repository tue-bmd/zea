"""Tests for the Scan class.
"""
import numpy as np

from usbmd.scan import PlaneWaveScan, Scan

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
    "sound_speed": 1540,
    "n_ax": 3328,
    "Nx": 64,
    "Nz": 128,
    "pixels_per_wvln": 4,
    "polar_angles": np.linspace(-np.pi / 2, np.pi / 2, 10),
    "azimuth_angles": np.linspace(-np.pi / 2, np.pi / 2, 10),
    "t0_delays": np.repeat(np.linspace(0, 1e-6, 10)[..., None], 10, axis=-1),
    "tx_apodizations": np.ones((10, 10)),
    "focus_distances": np.ones(10) * 0.04,
    "downsample": 1,
    "initial_times": np.zeros((10,)),
}

planewave_scan_args = {
    "n_tx": 10,
    "n_el": 128,
    "n_ch" : 1,
    "xlims": (-0.019, 0.019),
    "ylims": (0, 0),
    "zlims": (0, 0.04),
    "center_frequency": 7e6,
    "sampling_frequency": 28e6,
    "demodulation_frequency": 0.0,
    "sound_speed": 1540,
    "n_ax": 3328,
    "Nx": 64,
    "Nz": 128,
    "pixels_per_wvln": 4,
    "polar_angles": np.linspace(-np.pi / 2, np.pi / 2, 10),
    "azimuth_angles": np.linspace(-np.pi / 2, np.pi / 2, 10),
    "tx_apodizations": np.ones((10, 10)),
    "downsample": 1,
    "initial_times": np.zeros((10,)),
}


def test_initialization():
    """Test initialization of Scan class."""
    scan = Scan(**scan_args)

    assert scan.n_tx == scan_args["n_tx"]
    assert scan.n_el == scan_args["n_el"]
    assert scan.n_ch == scan_args["n_ch"]
    assert scan.xlims == scan_args["xlims"]
    assert scan.ylims == scan_args["ylims"]
    assert scan.zlims == scan_args["zlims"]
    assert scan.fc == scan_args["center_frequency"]
    assert scan.fs == scan_args["sampling_frequency"]
    assert scan.fdemod == scan_args["demodulation_frequency"]
    assert scan.sound_speed == scan_args["sound_speed"]
    assert scan.n_ax == scan_args["n_ax"]
    assert scan.Nx == scan_args["Nx"]
    assert scan.Nz == scan_args["Nz"]
    assert np.all(scan.polar_angles == scan_args["polar_angles"])
    assert np.all(scan.azimuth_angles == scan_args["azimuth_angles"])
    assert np.all(scan.t0_delays == scan_args["t0_delays"])
    assert np.all(scan.tx_apodizations == scan_args["tx_apodizations"])
    assert np.all(scan.focus_distances == scan_args["focus_distances"])
    assert np.all(scan.initial_times == scan_args["initial_times"])
    assert scan.pixels_per_wavelength == scan_args["pixels_per_wvln"]


def test_planewave_scan():
    """Test initialization of PlaneWaveScan class."""
    scan = PlaneWaveScan(**planewave_scan_args)

    assert scan.n_tx == planewave_scan_args["n_tx"]
    assert scan.n_el == planewave_scan_args["n_el"]
    assert scan.n_ch == planewave_scan_args["n_ch"]
    assert scan.xlims == planewave_scan_args["xlims"]
    assert scan.ylims == planewave_scan_args["ylims"]
    assert scan.zlims == planewave_scan_args["zlims"]
    assert scan.fc == planewave_scan_args["center_frequency"]
    assert scan.fs == planewave_scan_args["sampling_frequency"]
    assert scan.fdemod == scan_args["demodulation_frequency"]
    assert scan.sound_speed == planewave_scan_args["sound_speed"]
    assert scan.n_ax == planewave_scan_args["n_ax"]
    assert scan.Nx == planewave_scan_args["Nx"]
    assert scan.Nz == planewave_scan_args["Nz"]
    assert np.all(scan.polar_angles == planewave_scan_args["polar_angles"])
    assert np.all(scan.azimuth_angles == planewave_scan_args["azimuth_angles"])
    assert np.all(scan.tx_apodizations == planewave_scan_args["tx_apodizations"])
    assert scan.pixels_per_wavelength == planewave_scan_args["pixels_per_wvln"]
    assert np.all(scan.initial_times == planewave_scan_args["initial_times"])
