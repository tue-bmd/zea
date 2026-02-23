"""Standalone MachBeamform debug script."""

import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np


def _block_until_ready(value):
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def _time_call(label, fn, iterations=100, warmup=2):
    result = None
    for _ in range(warmup):
        result = fn()
    _block_until_ready(result)

    start = time.perf_counter()
    for _ in range(iterations):
        result = fn()
    _block_until_ready(result)
    elapsed = time.perf_counter() - start
    per_iter_s = elapsed / iterations
    print(f"{label} total time ({iterations} iters): {elapsed:.3f} s")
    print(f"{label} per iter: {per_iter_s * 1e3:.3f} ms")
    return result, per_iter_s


def _nan_stats(label, array):
    arr = np.asarray(array)
    nan_count = np.isnan(arr).sum()
    total = arr.size
    finite_min = np.nanmin(arr) if nan_count < total else np.nan
    finite_max = np.nanmax(arr) if nan_count < total else np.nan
    print(f"{label} nan count: {nan_count}/{total}, min: {finite_min:.3e}, max: {finite_max:.3e}")


def _save_image(path, image, scan, title, dynamic_range=(-60, 0)):
    from zea.display import to_8bit

    image_8bit = to_8bit(image, dynamic_range=dynamic_range, pillow=False)
    plt.figure(figsize=(6, 5))
    plt.imshow(image_8bit, cmap="gray", extent=scan.extent)
    plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _get_gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return lines[0] if lines else "Unknown GPU"
    except Exception:
        return "Unknown GPU"


def _format_pipeline_title(label, input_shape, output_shape, per_iter_s, points_per_second):
    per_iter_ms = per_iter_s * 1e3
    return (
        f"{label}\n"
        f"input {input_shape} | output {output_shape}\n"
        f"{per_iter_ms:.3f} ms | {points_per_second:.2e} pts/s"
    )


def _calculate_points_per_second(input_data, output_data, timing_seconds):
    if input_data.ndim >= 5:
        n_frames = input_data.shape[0]
        n_el = input_data.shape[3]
    else:
        n_frames = 1
        n_el = input_data.shape[2]

    if output_data.ndim >= 2:
        n_voxels = output_data.shape[0] * output_data.shape[1]
    else:
        n_voxels = output_data.size

    total_points = n_el * n_voxels * n_frames
    return total_points / timing_seconds


def _save_comparison_image(
    path,
    zea_image,
    mach_image,
    scan,
    zea_title,
    mach_title,
    full_title,
    dynamic_range=(-60, 0),
):
    from zea.display import to_8bit

    zea_8bit = to_8bit(zea_image, dynamic_range=dynamic_range, pillow=False)
    mach_8bit = to_8bit(mach_image, dynamic_range=dynamic_range, pillow=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    extent = scan.extent

    axes[0].imshow(zea_8bit, cmap="gray", extent=extent)
    axes[0].set_title(zea_title)
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Z (m)")

    axes[1].imshow(mach_8bit, cmap="gray", extent=extent)
    axes[1].set_title(mach_title)
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Z (m)")

    fig.suptitle(full_title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _load_picmus_sample():
    from zea.data import load_file

    path = (
        "hf://zeahub/picmus/database/experiments/contrast_speckle/"
        "contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"
    )
    data, scan, probe = load_file(
        path=path,
        indices=[0],
        data_type="raw_data",
    )

    scan.n_ch = 2
    scan.xlims = probe.xlims
    scan.zlims = (0.0, 0.06)

    scan.set_transmits(3)
    data_frame = data[0][scan.selected_transmits]
    return data_frame, scan, probe


def run_pipeline_zea_example(data_frame, scan, probe):
    import keras

    from zea.ops import Beamform, EnvelopeDetect, LogCompress, Normalize, Pipeline

    pipeline = Pipeline(
        [
            Beamform(num_patches=1),
            EnvelopeDetect(),
            Normalize(),
            LogCompress(),
        ],
        with_batch_dim=False,
        jit_options="pipeline",
    )

    inputs = pipeline.prepare_parameters(probe=probe, scan=scan)
    inputs["data"] = data_frame
    inputs["demodulation_frequency"] = scan.demodulation_frequency

    result, per_iter_s = _time_call(
        "Zea pipeline",
        lambda: pipeline(**inputs)["data"],
        iterations=10,
    )
    result = keras.ops.convert_to_numpy(result)
    _nan_stats("Zea pipeline output", result)
    print("Zea pipeline input shape:", data_frame.shape)
    print("Zea pipeline output shape:", result.shape)
    print("Zea pipeline output dtype:", result.dtype)
    _save_image("zea_pipeline_output.png", result, scan, "Zea pipeline output")
    points_per_second = _calculate_points_per_second(data_frame, result, per_iter_s)
    title = _format_pipeline_title(
        "Zea pipeline",
        data_frame.shape,
        result.shape,
        per_iter_s,
        points_per_second,
    )
    return result, title


def run_pipeline_mach_example(data_frame, scan, probe):
    import keras

    import mach
    from zea.ops import EnvelopeDetect, LogCompress, MachBeamform, Normalize, Pipeline, ReshapeGrid

    origin = np.array([0.0, 0.0, 0.0], dtype="float32")
    if scan.polar_angles is None:
        directions = np.array([[0.0, 0.0, 1.0]], dtype="float32")
    else:
        angles = np.asarray(scan.polar_angles, dtype="float32")
        directions = np.stack(
            [np.array([np.sin(angle), 0.0, np.cos(angle)], dtype="float32") for angle in angles],
            axis=0,
        )

    tx_wave_arrivals_s = np.stack(
        [
            mach.wavefront.plane(origin_m=origin, points_m=scan.flatgrid, direction=direction)
            / scan.sound_speed
            for direction in directions
        ],
        axis=0,
    )

    pipeline = Pipeline(
        [
            MachBeamform(),
            ReshapeGrid(),
            EnvelopeDetect(),
            Normalize(),
            LogCompress(),
        ],
        with_batch_dim=False,
        jit_options="ops",
    )

    inputs = pipeline.prepare_parameters(probe=probe, scan=scan)
    inputs["data"] = data_frame
    inputs["tx_wave_arrivals_s"] = tx_wave_arrivals_s
    inputs["demodulation_frequency"] = scan.demodulation_frequency

    _nan_stats("Mach input data", data_frame)
    _nan_stats("Mach tx arrivals", tx_wave_arrivals_s)

    mach_beamform = MachBeamform(with_batch_dim=False)
    beamformed = mach_beamform(
        data=data_frame,
        flatgrid=scan.flatgrid,
        probe_geometry=probe.probe_geometry,
        tx_wave_arrivals_s=tx_wave_arrivals_s,
        sampling_frequency=scan.sampling_frequency,
        sound_speed=scan.sound_speed,
        f_number=scan.f_number,
        demodulation_frequency=scan.demodulation_frequency,
        initial_times=scan.initial_times,
    )["data"]
    beamformed_np = keras.ops.convert_to_numpy(beamformed)
    _nan_stats("Mach beamformed (flat)", beamformed_np)

    result, per_iter_s = _time_call(
        "Mach pipeline",
        lambda: pipeline(**inputs)["data"],
        iterations=10,
    )
    result = keras.ops.convert_to_numpy(result)
    _nan_stats("Mach pipeline output", result)
    print("Mach pipeline input shape:", data_frame.shape)
    print("Mach pipeline tx arrivals shape:", tx_wave_arrivals_s.shape)
    print("Mach pipeline output shape:", result.shape)
    print("Mach pipeline output dtype:", result.dtype)
    _save_image("mach_pipeline_output.png", result, scan, "Mach pipeline output")
    points_per_second = _calculate_points_per_second(data_frame, result, per_iter_s)
    title = _format_pipeline_title(
        "Mach pipeline",
        data_frame.shape,
        result.shape,
        per_iter_s,
        points_per_second,
    )
    return result, title


if __name__ == "__main__":
    data_frame, scan, probe = _load_picmus_sample()
    zea_out, zea_title = run_pipeline_zea_example(data_frame, scan, probe)
    mach_out, mach_title = run_pipeline_mach_example(data_frame, scan, probe)
    diff = np.abs(zea_out - mach_out)
    _save_image("pipeline_abs_diff.png", diff, scan, "Pipeline abs diff")
    full_title = f"PICMUS pipeline comparison | GPU: {_get_gpu_info()}"
    _save_comparison_image(
        "pipeline_outputs_side_by_side.png",
        zea_out,
        mach_out,
        scan,
        zea_title,
        mach_title,
        full_title,
    )
    rel_error = np.linalg.norm(zea_out - mach_out) / (np.linalg.norm(zea_out) + 1e-12)
    print(f"Pipeline relative error (mach vs zea): {rel_error:.3e}")
