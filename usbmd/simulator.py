from usbmd import Scan
from keras import ops
import numpy as np


def simulate_rf(scan: Scan, scat_positions, scat_magnitudes):

    dist = ops.linalg.norm(scan.probe_geometry[None] - scat_positions[:, None], axis=-1)
    freqs = ops.arange(scan.n_ax // 2 + 1) / scan.n_ax * scan.fs

    print(freqs)


def travel(distance, sound_speed, frequency):
    return ops.exp(-1j * distance / sound_speed * frequency)


def directivity(theta, frequency, sound_speed):
    arg = np.pi * sound_speed / frequency * ops.sin(theta)
    return ops.sin(arg) * ops.cos(theta)


if __name__ == "__main__":
    n_el = 80
    n_scat = 5
    scan = Scan(
        n_tx=1,
        n_ax=512,
        n_el=n_el,
        center_frequency=3.125e6,
        sampling_frequency=12.5e6,
        probe_geometry=ops.stack(
            [ops.linspace(-10e-3, 10e-3, n_el), ops.zeros(n_el), ops.zeros(n_el)],
            axis=1,
        ),
        t0_delays=ops.zeros((1, n_el)),
    )

    simulate_rf(
        scan=scan,
        scat_positions=ops.stack(
            [
                ops.linspace(-10e-3, 10e-3, n_scat),
                ops.zeros(n_scat),
                ops.ones(n_scat) * 10e-3,
            ],
            axis=1,
        ),
        scat_magnitudes=ops.ones(n_scat),
    )
