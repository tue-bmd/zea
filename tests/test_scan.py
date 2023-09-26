import pytest
import numpy as np

from usbmd.scan import Scan, FocussedScan, DivergingWaveScan, PlaneWaveScan

scan_args = dict(
    xlims=(-0.02, 0.02),
    ylims=(0.01, 0.02),
    zlims=(0, 0.05),
    fc=7.2e6,
    fs=32e6,
    c=1550,
    modtype='iq',
    N_ax=1000,
    Nx=64,
    Nz=64,
    pixels_per_wvln=2,
    tzero_correct=False,
    downsample=2)

focussed_scan_args = dict(
    **scan_args,
    origins=np.linspace(-0.02, 0.02, 60),
    focus_distances=np.ones(60)*0.05)


def test_initialization():
    scan = Scan(**scan_args)

    assert scan.xlims == (-0.02, 0.02)
    assert scan.ylims == (0.01, 0.02)
    assert scan.zlims == (0, 0.05)
    assert scan.fc == 7.2e6
    assert scan.fs == 32e6
    assert scan.c == 1550
    assert scan.modtype == 'iq'
    assert scan.N_ax == 1000//2
    assert scan.Nx == 64
    assert scan.Nz == 64
    assert scan.pixels_per_wavelength == 2
    assert scan.tzero_correct is False

def test_get_time_zero_correct_focussed():
    scan = FocussedScan(**focussed_scan_args)

def test_add_transmit():
    scan = Scan(**scan_args)

    n_el = 128
    n_tx = 8

    t0_delays = np.linspace(0, 1e-6, n_el)
    tx_apodizations = np.ones(n_el)
    c = 1540

    for _ in range(n_tx):
        scan.add_transmit(t0_delays, tx_apodizations, c)

    assert scan.n_tx == n_tx, 'Incorrect number of transmits added'

def test_add_planewave_transmit():
    scan = Scan(**scan_args)

    n_el = 128
    ele_pos = np.concatenate((np.linspace(-0.02, 0.02, n_el),
                              np.zeros(n_el),
                              np.zeros(n_el))).reshape(-1, 3)

    angles = np.array([-20, -10, 0, 10, 20])/360*2*np.pi

    tx_apodizations = np.ones(n_el)
    c = 1540

    for angle in angles:
        scan.add_planewave_transmit(ele_pos,
                                    polar_angle=angle,
                                    azimuth_angle=0,
                                    apodization=tx_apodizations,
                                    c=c,
                                    initial_time=0)

    assert scan.n_tx == angles.shape[0], 'Incorrect number of transmits added'
