"""Equivalence of :func:`zea.simulator.simulate_rf` with PyMUST's ``simus3``.

Both simulators get the same scene. Time origin: sample n is at ``t = n / fs`` and the transmit
pulse is centred at ``t = 0`` plus the element delay (zea: ``initial_times = t_peak = 0``).
Settings that have no counterpart are pinned to the SIMUS convention: ``scatter_exponent=0``,
1/r spreading both ways, no lens correction, frequency dependent directivity.

The transmit pulses differ by design (zea: Hann windowed cosine times a Gaussian transfer,
SIMUS: rectangular windowed sine times a generalized-normal transfer). Each record is convolved
with the other simulator's pulse, so both become field times both pulses and only the
propagation physics is compared.
"""

from dataclasses import dataclass, replace

import numpy as np
import pytest
from keras import ops
from scipy.signal import fftconvolve
from scipy.special import fresnel

from zea.beamform.phantoms import fish
from zea.simulator import (
    chirp_spectrum,
    get_pulse_spectrum_fn,
    simulate_rf,
    transducer_transfer,
)

pymust = pytest.importorskip("pymust")

# PyMUST 0.1.9 bug: fresnelint indexes with `x[not issmall]`, which crashes for |x| > 1.6 (chirps).
pymust.utils.fresnelint = lambda x: (lambda s, c: c + 1j * s)(*fresnel(np.asarray(x, np.float64)))

FC, FS, C = 3e6, 12e6, 1540.0
DB_THRESH = -100.0


@dataclass
class Scene:
    """One scene, in the terms both simulators take."""

    geometry: np.ndarray  # (n_el, 3)
    element_width: float
    element_height: float
    positions: np.ndarray  # (n_scat, 3)
    t0_delays: np.ndarray  # (n_tx, n_el)
    apodizations: np.ndarray  # (n_tx, n_el)
    n_period: float = 2.0
    bandwidth_percent: float = 75.0
    attenuation_coef: float = 0.0
    rigid_baffle: bool = True
    chirp_sweep: float | None = None
    n_sub_elements: tuple | str | None = (1, 1)


def linear_probe(n_el=64, pitch=0.5e-3):
    x = (np.arange(n_el) - (n_el - 1) / 2) * pitch
    return np.stack([x, np.zeros(n_el), np.zeros(n_el)], axis=1)


def matrix_probe(n=12, pitch=0.5e-3):
    x = (np.arange(n) - (n - 1) / 2) * pitch
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), np.zeros(n * n)], axis=1)


def plane_waves(geometry, tilts):
    """tilts: list of (tilt_x, tilt_y) in degrees. The first element fires at t = 0."""
    delays = []
    for tx, ty in tilts:
        d = (geometry[:, 0] * np.sin(np.radians(tx)) + geometry[:, 1] * np.sin(np.radians(ty))) / C
        delays.append(d - d.min())
    return np.array(delays)


def focused_wave(geometry, focus):
    d = np.linalg.norm(geometry - np.asarray(focus)[None], axis=1)
    return (d.max() - d)[None] / C


def fish_scatterers(step=2, rot_deg=0.0):
    """Every step-th fish scatterer, rotated about the z axis so the fish leaves the y = 0 plane."""
    pos = fish()[::step].copy()
    a = np.radians(rot_deg)
    x, y = pos[:, 0].copy(), pos[:, 1].copy()
    pos[:, 0] = x * np.cos(a) - y * np.sin(a)
    pos[:, 1] = x * np.sin(a) + y * np.cos(a)
    return pos


def linear_scene(positions, delays, **overrides):
    geometry = linear_probe()
    return Scene(geometry, 0.45e-3, 5e-3, positions, delays, np.ones_like(delays), **overrides)


def point_2d_pulse():
    """One on-axis point, one plane wave: the time origin and pulse conventions."""
    return linear_scene(np.array([[0.0, 0.0, 20e-3]]), plane_waves(linear_probe(), [(0, 0)]))


def fish_2d_allinplane():
    """Fish in the y = 0 plane, two tilted plane waves, whole elements (no splitting)."""
    return linear_scene(fish_scatterers(), plane_waves(linear_probe(), [(-10, 0), (10, 0)]))


def fish_2d_45degrot_outofplane():
    """Fish rotated 45 deg about z (y up to 12 mm), auto splitting in both simulators."""
    return linear_scene(
        fish_scatterers(rot_deg=45),
        plane_waves(linear_probe(), [(-10, 0), (10, 0)]),
        n_sub_elements="auto",
    )


def fish_2d_45degrot_nosplit():
    """As fish_2d_45degrot_outofplane with whole 5 mm elements: far-field elevation sinc only."""
    return replace(fish_2d_45degrot_outofplane(), n_sub_elements=(1, 1))


def fish_2d_soft_baffle():
    """Soft baffle (cos theta obliquity) with steep plane waves."""
    return linear_scene(
        fish_scatterers(), plane_waves(linear_probe(), [(-25, 0), (25, 0)]), rigid_baffle=False
    )


def fish_2d_attenuation():
    """0.7 dB/cm/MHz attenuation, both ways."""
    return linear_scene(
        fish_scatterers(), plane_waves(linear_probe(), [(0, 0)]), attenuation_coef=0.7
    )


def fish_2d_chirp():
    """Linear chirp, 1.5 MHz sweep over 6 periods. zea sweeps up, SIMUS down, so only the
    equalized metrics are comparable."""
    return linear_scene(
        fish_scatterers(),
        plane_waves(linear_probe(), [(0, 0)]),
        chirp_sweep=1.5e6,
        n_period=6.0,
    )


def fish_2d_bandwidth50():
    """50% bandwidth: a Gaussian (zea) against a generalized normal (SIMUS) transfer."""
    return linear_scene(
        fish_scatterers(), plane_waves(linear_probe(), [(0, 0)]), bandwidth_percent=50.0
    )


def fish_2d_focused_apod():
    """Steered focused transmit at (5, 0, 25) mm with a Hann apodization."""
    geometry = linear_probe()
    scene = linear_scene(fish_scatterers(), focused_wave(geometry, [5e-3, 0.0, 25e-3]))
    scene.apodizations = np.hanning(len(geometry))[None]
    return scene


def fish_3d_tilted():
    """12x12 matrix probe, fish rotated 45 deg, plane waves tilted in x and in y."""
    geometry = matrix_probe()
    return Scene(
        geometry,
        0.45e-3,
        0.45e-3,
        fish_scatterers(step=3, rot_deg=45),
        plane_waves(geometry, [(10, 0), (0, -10)]),
        np.ones((2, len(geometry))),
    )


def fish_3d_focused_split():
    """8x8 matrix probe with 0.9 mm elements split 2x2, focused at (2, -2, 25) mm."""
    geometry = matrix_probe(n=8, pitch=1e-3)
    return Scene(
        geometry,
        0.9e-3,
        0.9e-3,
        fish_scatterers(step=3, rot_deg=45),
        focused_wave(geometry, [2e-3, -2e-3, 25e-3]),
        np.ones((1, len(geometry))),
        n_sub_elements=(2, 2),
    )


def fish_3d_soft_attenuated():
    """3D, soft baffle and 0.5 dB/cm/MHz together."""
    return replace(fish_3d_tilted(), rigid_baffle=False, attenuation_coef=0.5)


SCENES = {
    fn.__name__: fn
    for fn in (
        point_2d_pulse,
        fish_2d_allinplane,
        fish_2d_45degrot_nosplit,
        fish_2d_45degrot_outofplane,
        fish_2d_soft_baffle,
        fish_2d_attenuation,
        fish_2d_chirp,
        fish_2d_bandwidth50,
        fish_2d_focused_apod,
        fish_3d_tilted,
        fish_3d_focused_split,
        fish_3d_soft_attenuated,
    )
}


def record_samples(s):
    """Samples that hold the round trip to the farthest scatterer plus a whole pulse."""
    dist = np.linalg.norm(s.positions[:, None] - s.geometry[None], axis=-1).max()
    t_end = 2 * dist / C + s.t0_delays.max() + s.n_period / FC
    return int(np.ceil(t_end * FS)) + 16


def run_zea(s):
    n_tx = len(s.t0_delays)
    rf = simulate_rf(
        scatterer_positions=s.positions.astype(np.float32),
        scatterer_magnitudes=np.ones(len(s.positions), np.float32),
        probe_geometry=s.geometry.astype(np.float32),
        apply_lens_correction=False,
        lens_thickness=1e-3,
        lens_sound_speed=1000.0,
        sound_speed=C,
        n_ax=record_samples(s),
        center_frequency=FC,
        sampling_frequency=FS,
        t0_delays=s.t0_delays.astype(np.float32),
        initial_times=np.zeros(n_tx, np.float32),
        element_width=s.element_width,
        attenuation_coef=s.attenuation_coef,
        tx_apodizations=s.apodizations.astype(np.float32),
        t_peak=np.zeros(n_tx, np.float32),
        element_height=s.element_height,
        scatter_exponent=0.0,
        rigid_baffle=s.rigid_baffle,
        bandwidth_percent=s.bandwidth_percent,
        probe_center_frequency=FC,
        chirp_sweep=s.chirp_sweep,
        n_period=s.n_period,
        n_sub_elements=s.n_sub_elements,
    )
    return np.asarray(ops.convert_to_numpy(rf))[..., 0]  # (n_tx, n_t, n_el)


def zea_pulse(s, n=4096):
    """The transmit pulse of ``simulate_rf``, sampled on the rfft grid of ``n``."""
    freqs = ops.arange(n // 2 + 1, dtype="float32") / n * FS
    if s.chirp_sweep:
        spectrum = chirp_spectrum(n, FC, FS, s.n_period, s.chirp_sweep)
    else:
        spectrum = get_pulse_spectrum_fn(FC, n_period=s.n_period, sampling_frequency=FS)(freqs)
    transfer = transducer_transfer(freqs, FC, s.bandwidth_percent, FC)
    spectrum = spectrum * ops.cast(transfer, "complex64")
    return centred(np.fft.irfft(np.asarray(ops.convert_to_numpy(spectrum)), n))


def simus_param(s):
    p = pymust.utils.Param()
    p.fc, p.fs, p.c = FC, FS, C
    p.width, p.height = s.element_width, s.element_height
    p.bandwidth = s.bandwidth_percent
    p.baffle = "rigid" if s.rigid_baffle else "soft"
    p.attenuation = s.attenuation_coef
    p.TXnow = s.n_period
    if s.chirp_sweep:
        p.TXfreqsweep = abs(s.chirp_sweep)
    p.elements = s.geometry[:, :2].T.astype(np.float64)
    return p


def simus_options(s):
    o = pymust.utils.Options()
    o.WaitBar = False
    o.FullFrequencyDirectivity = True
    o.dBThresh = DB_THRESH
    if s.n_sub_elements == "auto":
        o.ElementSplitting = None  # SIMUS applies the same ceil(size / lambda_min) rule
    else:
        o.ElementSplitting = list(s.n_sub_elements)
    return o


def run_simus(s):
    x, y, z = (s.positions[:, k].astype(np.float64)[None] for k in range(3))
    rc = np.ones_like(x)
    out = []
    for tx in range(len(s.t0_delays)):
        p = simus_param(s)  # simus3 mutates param and options, so rebuild them per transmit
        p.TXapodization = s.apodizations[tx].astype(np.float64)[None]
        delays = s.t0_delays[tx].astype(np.float64)[None]
        rf, _ = pymust.simus3(x, y, z, rc, delays, p, simus_options(s))
        out.append(np.asarray(rf))
    n = min(len(r) for r in out)
    return np.stack([r[:n] for r in out])


def simus_pulse(s, n=4096):
    p = simus_param(s)
    w = 2 * np.pi * np.arange(n // 2 + 1) / n * FS
    spectrum = p.getPulseSpectrumFunction(p.TXfreqsweep if s.chirp_sweep else None)(w)
    spectrum = spectrum * p.getProbeFunction()(w) ** 2
    return centred(np.fft.irfft(np.conj(spectrum), n))  # simus3 inverts conj(spectrum) too


def centred(x):
    """Pulse centred on the middle sample of an odd-length array."""
    x = np.fft.fftshift(x)
    c = len(x) // 2
    keep = np.nonzero(np.abs(x) > 1e-4 * np.abs(x).max())[0]
    half = max(c - keep[0], keep[-1] - c) + 1
    return x[c - half : c + half + 1]


def equalize(rf, pulse):
    return fftconvolve(rf, pulse[None, :, None], mode="same", axes=1)


def metrics(a, b):
    """a: zea, b: reference, scaled onto a in the least-squares sense."""
    b = b * (a * b).sum() / (b * b).sum()
    sa, sb = a.sum(axis=(0, 2)), b.sum(axis=(0, 2))
    energy_a, energy_b = np.sqrt((a**2).sum(axis=1)), np.sqrt((b**2).sum(axis=1))
    return {
        "rel": np.linalg.norm(a - b) / np.linalg.norm(a),
        "corr": (a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)),
        "lag": int(np.argmax(np.correlate(sa, sb, mode="full")) - (len(sa) - 1)),
        "profile": np.linalg.norm(energy_a - energy_b) / np.linalg.norm(energy_a),
    }


@pytest.mark.parametrize("name", SCENES, ids=SCENES)
def test_simulate_rf_matches_simus3(name):
    scene = SCENES[name]()
    rf_zea, rf_ref = run_zea(scene), run_simus(scene)
    # SIMUS returns fewer samples than n_ax for some scenes; compare the common part.
    n = min(rf_zea.shape[1], rf_ref.shape[1])
    equalized = equalize(rf_zea[:, :n], simus_pulse(scene))
    reference = equalize(rf_ref[:, :n], zea_pulse(scene))
    m = metrics(equalized, reference)
    assert m["lag"] == 0
    assert m["rel"] < 1e-3
    assert m["corr"] > 0.9999
    assert m["profile"] < 1e-3
