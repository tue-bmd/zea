import pytest
import numpy as np

from usbmd.scan import Scan, FocussedScan, DivergingWaveScan, PlaneWaveScan

scan_args = dict(
    N_tx=10,
    xlims=(-0.019, 0.019),
    ylims=(0, 0),
    zlims=(0, 0.04),
    fc=7e6,
    fs=28e6,
    c=1540,
    modtype="rf",
    N_ax=3328,
    Nx=64,
    Nz=128,
    pixels_per_wvln=4,
    polar_angles=np.linspace(-np.pi/2, np.pi/2, 10),
    azimuth_angles=np.linspace(-np.pi/2, np.pi/2, 10),
    t0_delays=np.linspace(0, 1e-6, 10),
    tx_apodizations=np.ones((10, 10)),
    focus_distances=np.ones(10)*0.04,
    downsample=1,
    initial_times=np.zeros((10,)))


def test_initialization():
    scan = Scan(**scan_args)

    assert scan.N_tx == scan_args['N_tx']
    assert scan.xlims == scan_args['xlims']
    assert scan.ylims == scan_args['ylims']
    assert scan.zlims == scan_args['zlims']
    assert scan.fc == scan_args['fc']
    assert scan.fs == scan_args['fs']
    assert scan.c == scan_args['c']
    assert scan.modtype == scan_args['modtype']
    assert scan.N_ax == scan_args['N_ax']
    assert scan.Nx == scan_args['Nx']
    assert scan.Nz == scan_args['Nz']
    assert np.all(scan.polar_angles == scan_args['polar_angles'])
    assert np.all(scan.azimuth_angles == scan_args['azimuth_angles'])
    assert np.all(scan.t0_delays == scan_args['t0_delays'])
    assert np.all(scan.tx_apodizations == scan_args['tx_apodizations'])
    assert np.all(scan.focus_distances == scan_args['focus_distances'])
    assert np.all(scan.initial_times == scan_args['initial_times'])
    assert scan.pixels_per_wavelength == scan_args['pixels_per_wvln']
