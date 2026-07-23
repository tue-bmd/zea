"""Benchmark and validate :class:`zea.ops.MachBeamform` against zea's DAS.

TEMPORARY validation/benchmarking script (not part of the test suite). It needs a
CUDA GPU plus the optional ``mach-beamform`` and ``cupy`` packages
(``pip install 'zea[mach]'`` and a CUDA-matched cupy, e.g. ``cupy-cuda12x``).

It loads a PICMUS plane-wave IQ frame from the zea Hugging Face hub and runs three
reconstructions on the same data:

1. **zea DAS** -- the standard :class:`zea.ops.Beamform` (delay-and-sum) pipeline.
2. **zea + mach** -- the same pipeline with :class:`zea.ops.MachBeamform` in place
   of the whole beamforming block.
3. **mach API** -- the raw ``mach.experimental.beamform`` call, for reference.

For each it reports per-iteration timing and throughput (points/s), computes the
relative error against the zea DAS image, and writes a side-by-side PNG plus
absolute-difference images.

Usage::

    python scripts/benchmark_mach_beamformer.py [--transmits 3] [--iterations 100]
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

import zea

# PICMUS plane-wave IQ dataset (experimental contrast-speckle phantom).
PICMUS_PATH = (
    "hf://zeahub/picmus/database/experiments/contrast_speckle/"
    "contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"
)
DYNAMIC_RANGE = (-60, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _block_until_ready(value):
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def _time_call(label, fn, iterations, warmup=2):
    result = None
    for _ in range(warmup):
        result = fn()
    _block_until_ready(result)

    start = time.perf_counter()
    for _ in range(iterations):
        result = fn()
    _block_until_ready(result)
    per_iter_s = (time.perf_counter() - start) / iterations
    print(f"{label}: {per_iter_s * 1e3:.3f} ms/iter ({iterations} iters)")
    return result, per_iter_s


def _points_per_second(input_data, output_image, per_iter_s):
    """Rough throughput estimate: n_el * n_pixels * n_frames / time."""
    n_el = input_data.shape[-2]
    n_frames = input_data.shape[0] if input_data.ndim >= 5 else 1
    n_pix = int(np.prod(output_image.shape[:2]))
    return n_el * n_pix * n_frames / per_iter_s


def _rel_error(reference, other):
    return np.linalg.norm(reference - other) / (np.linalg.norm(reference) + 1e-12)


def _title(label, in_shape, out_shape, per_iter_s, pts_per_s):
    return (
        f"{label}\ninput {in_shape} | output {out_shape}\n"
        f"{per_iter_s * 1e3:.3f} ms | {pts_per_s:.2e} pts/s"
    )


def _to_numpy(image):
    import keras

    return keras.ops.convert_to_numpy(image)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_picmus(n_transmits):
    """Load a single PICMUS IQ frame and a matching parameters object."""
    with zea.File(PICMUS_PATH) as file:
        data = file.data.raw_data[0]  # (n_tx, n_ax, n_el, n_ch=2)
        parameters = file.load_parameters()

    parameters.zlims = (0.0, 0.06)
    parameters.set_transmits(n_transmits)
    data = np.asarray(data)[parameters.selected_transmits]
    return data, parameters


def _extent_mm(parameters):
    xlims_mm = [v * 1e3 for v in parameters.xlims]
    zlims_mm = [v * 1e3 for v in parameters.zlims]
    return [xlims_mm[0], xlims_mm[1], zlims_mm[1], zlims_mm[0]]


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def run_zea_das(data, parameters, iterations):
    from zea.ops import Beamform, EnvelopeDetect, LogCompress, Normalize, Pipeline

    pipeline = Pipeline(
        [Beamform(), EnvelopeDetect(), Normalize(), LogCompress()],
        with_batch_dim=False,
        jit_options="pipeline",
    )
    inputs = pipeline.prepare_parameters(parameters)
    inputs["data"] = data

    result, per_iter_s = _time_call("zea DAS", lambda: pipeline(**inputs)["data"], iterations)
    image = _to_numpy(result)
    pts = _points_per_second(data, image, per_iter_s)
    return image, _title("zea DAS", data.shape, image.shape, per_iter_s, pts)


def run_zea_mach(data, parameters, iterations):
    from zea.ops import (
        EnvelopeDetect,
        LogCompress,
        MachBeamform,
        Normalize,
        Pipeline,
        ReshapeGrid,
    )

    pipeline = Pipeline(
        [MachBeamform(), ReshapeGrid(), EnvelopeDetect(), Normalize(), LogCompress()],
        with_batch_dim=False,
        jit_options="ops",
    )
    inputs = pipeline.prepare_parameters(parameters)
    inputs["data"] = data

    result, per_iter_s = _time_call("zea + mach", lambda: pipeline(**inputs)["data"], iterations)
    image = _to_numpy(result)
    pts = _points_per_second(data, image, per_iter_s)
    return image, _title("zea + mach", data.shape, image.shape, per_iter_s, pts)


def run_mach_api(data, parameters, iterations):
    """Raw mach.experimental.beamform on the same data, for reference."""
    import mach
    import mach.experimental

    try:
        import cupy as cp
    except ImportError:
        cp = None

    # (n_tx, n_ax, n_el, 2) IQ -> complex (n_tx, n_ax, n_el)
    data_complex = data[..., 0] + 1j * data[..., 1]
    channel_data = np.ascontiguousarray(
        np.transpose(data_complex, (0, 2, 1))[..., None]
    )  # (n_tx, n_el, n_ax, 1)
    scan_coords_m = np.ascontiguousarray(parameters.flatgrid)
    rx_coords_m = np.ascontiguousarray(parameters.probe_geometry)

    # Plane-wave arrival times per steering angle.
    origin = np.zeros(3, dtype="float32")
    angles = np.atleast_1d(np.asarray(parameters.polar_angles, dtype="float32"))
    directions = np.stack(
        [np.array([np.sin(a), 0.0, np.cos(a)], "float32") for a in angles], axis=0
    )
    tx_wave_arrivals_s = np.stack(
        [
            mach.wavefront.plane(origin_m=origin, points_m=scan_coords_m, direction=d)
            / float(parameters.sound_speed)
            for d in directions
        ],
        axis=0,
    )  # (n_tx, n_scan)

    if cp is not None:
        channel_data = cp.asarray(channel_data)
        scan_coords_m = cp.asarray(scan_coords_m)
        rx_coords_m = cp.asarray(rx_coords_m)
        tx_wave_arrivals_s = cp.asarray(tx_wave_arrivals_s)

    kwargs = dict(
        channel_data=channel_data,
        rx_coords_m=rx_coords_m,
        scan_coords_m=scan_coords_m,
        tx_wave_arrivals_s=tx_wave_arrivals_s,
        rx_start_s=0.0,
        sampling_freq_hz=float(parameters.sampling_frequency),
        f_number=float(parameters.f_number),
        sound_speed_m_s=float(parameters.sound_speed),
        modulation_freq_hz=float(parameters.demodulation_frequency),
        tukey_alpha=0.0,
    )

    result, per_iter_s = _time_call(
        "mach API", lambda: mach.experimental.beamform(**kwargs), iterations
    )
    if cp is not None:
        result = cp.asnumpy(result)
    result = np.nan_to_num(result).reshape(parameters.grid.shape[:-1])

    envelope = np.abs(result)
    envelope /= envelope.max() + 1e-12
    image = np.clip(20.0 * np.log10(envelope + 1e-12), *DYNAMIC_RANGE)
    pts = _points_per_second(data, image, per_iter_s)
    return image, _title("mach API", data.shape, image.shape, per_iter_s, pts)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def save_comparison(path, images, titles, extent, suptitle):
    from zea.display import to_8bit

    fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 5), dpi=150)
    for ax, image, title in zip(np.atleast_1d(axes), images, titles):
        ax.imshow(
            to_8bit(image, dynamic_range=DYNAMIC_RANGE, pillow=False),
            cmap="gray",
            extent=extent,
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z (mm)")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transmits", type=int, default=3, help="Number of transmits.")
    parser.add_argument("--iterations", type=int, default=100, help="Timing iterations.")
    args = parser.parse_args()

    zea.init_device()

    data, parameters = load_picmus(args.transmits)
    extent = _extent_mm(parameters)
    print(f"Input data shape: {data.shape}")

    zea_img, zea_title = run_zea_das(data, parameters, args.iterations)
    mach_img, mach_title = run_zea_mach(data, parameters, args.iterations)
    api_img, api_title = run_mach_api(data, parameters, args.iterations)

    err_mach = _rel_error(zea_img, mach_img)
    err_api = _rel_error(zea_img, api_img)
    print(f"Relative error (zea DAS vs zea+mach): {err_mach:.3e}")
    print(f"Relative error (zea DAS vs mach API): {err_api:.3e}")

    suptitle = (
        f"PICMUS comparison | zea vs mach rel err {err_mach:.2e} | "
        f"zea vs mach-API rel err {err_api:.2e}"
    )
    save_comparison(
        "mach_beamformer_comparison.png",
        [zea_img, mach_img, api_img],
        [zea_title, mach_title, api_title],
        extent,
        suptitle,
    )
    save_comparison(
        "mach_beamformer_absdiff.png",
        [np.abs(zea_img - mach_img), np.abs(zea_img - api_img)],
        ["|zea DAS - zea+mach|", "|zea DAS - mach API|"],
        extent,
        "Absolute differences",
    )


if __name__ == "__main__":
    main()
