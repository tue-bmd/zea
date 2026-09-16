"""Scenes and checks shared by the simulator tests: one scan, one phantom, one analytic pulse."""

import numpy as np
from keras import ops

from zea.simulator import transmit_pulse

SOUND_SPEED = 1540.0
CENTER_FREQUENCY = 3e6
SAMPLING_FREQUENCY = 12e6
N_AX = 512
# Periods of the Hann tone the analytic checks use: long enough to be narrow-band.
N_PERIOD = 4.0


def to_np(x):
    return np.asarray(ops.convert_to_numpy(x))


def tensors(kwargs):
    """Array arguments as backend tensors; ``simulate_rf`` mixes them with numpy otherwise."""
    return {
        k: ops.convert_to_tensor(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
    }


def linear_probe(n_el=16, pitch=0.3e-3):
    x = (np.arange(n_el) - (n_el - 1) / 2) * pitch
    return np.stack([x, np.zeros(n_el), np.zeros(n_el)], -1).astype(np.float32)


def matrix_probe(n_side=4, pitch=0.3e-3):
    x = (np.arange(n_side) - (n_side - 1) / 2) * pitch
    gx, gy = np.meshgrid(x, x, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), np.zeros(n_side**2)], -1).astype(np.float32)


def phantom(n=24, seed=0):
    """Random scatterers in a cone in front of the probe, with random magnitudes."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.01, 0.028, n)
    pos = np.stack([z * rng.uniform(-0.5, 0.5, n), rng.uniform(-1e-3, 1e-3, n), z], -1)
    return pos.astype(np.float32), rng.uniform(0.5, 1.0, n).astype(np.float32)


def scan(geometry, n_tx=4):
    """Focused, randomly apodized transmits with attenuation and scattering gain: the
    arguments of ``simulate_rf`` other than the scatterers."""
    rng = np.random.default_rng(1)
    focus = np.array(
        [[0.0, 0.0, 0.02], [0.005, 0.0, 0.025], [-0.004, 0.002, 0.03], [0.002, -0.003, 0.018]]
    )[:n_tx]
    dist = np.linalg.norm(focus[:, None] - geometry[None], axis=-1)
    t0 = ((dist.max(1, keepdims=True) - dist) / SOUND_SPEED).astype(np.float32)
    apod = rng.uniform(0.5, 1.0, (n_tx, len(geometry))).astype(np.float32)
    return {
        "probe_geometry": geometry,
        "apply_lens_correction": False,
        "lens_thickness": 1e-3,
        "lens_sound_speed": 1000.0,
        "sound_speed": SOUND_SPEED,
        "n_ax": N_AX,
        "center_frequency": CENTER_FREQUENCY,
        "sampling_frequency": SAMPLING_FREQUENCY,
        "t0_delays": t0,
        "initial_times": np.zeros(n_tx, np.float32),
        "element_width": 0.27e-3,
        "attenuation_coef": 0.5,
        "tx_apodizations": apod,
        "t_peak": np.zeros(n_tx, np.float32),
        "scatter_exponent": 1.5,
    }


def case(geometry, n_tx=4, **overrides):
    """The :func:`phantom` in the :func:`scan` of ``geometry``: a full ``simulate_rf`` call."""
    positions, magnitudes = phantom()
    return {
        "scatterer_positions": positions,
        "scatterer_magnitudes": magnitudes,
        **scan(geometry, n_tx),
        **overrides,
    }


def rel_err(reference, result):
    reference, result = to_np(reference), to_np(result)
    return np.linalg.norm(reference - result) / np.linalg.norm(reference)


def assert_close(reference, result, rel_tol=1e-3):
    reference, result = to_np(reference), to_np(result)
    assert result.shape == reference.shape
    rel = rel_err(reference, result)
    assert rel < rel_tol, rel


def correlation(a, b):
    a, b = to_np(a).ravel(), to_np(b).ravel()
    return a @ b / np.sqrt((a @ a) * (b @ b))


def stack_padded(*waveforms):
    """Waveforms of different lengths as one (n_tx, n_samples) array, zero-padded at the end."""
    n = max(len(w) for w in waveforms)
    return np.stack([np.pad(w, (0, n - len(w))) for w in waveforms])


def hann_tone(t):
    """Unit-peak Hann-windowed tone of ``N_PERIOD`` periods, centred at t = 0: the analytic
    form of :func:`hann_waveform`."""
    width = N_PERIOD / CENTER_FREQUENCY
    window = np.where(np.abs(t) < width / 2, np.cos(np.pi * t / width) ** 2, 0.0)
    return window * np.cos(2 * np.pi * CENTER_FREQUENCY * t)


def hann_waveform(n_period=N_PERIOD, **kwargs):
    """A Hann-windowed tone (or chirp) as the simulator takes it: its two-way waveform at
    250 MHz. Without ``bandwidth_percent`` it is the bare window."""
    kwargs.setdefault("bandwidth_percent", None)
    return transmit_pulse(
        CENTER_FREQUENCY, pulse_model="hann", n_period=n_period, **kwargs
    ).waveform()
