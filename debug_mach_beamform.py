"""Standalone MachBeamform debug script."""

import gc
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np

# =========================
# Generic helpers
# =========================


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


def _query_vram():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        stats = []
        for line in lines:
            used, free, total = [value.strip() for value in line.split(",")]
            stats.append(
                {
                    "used": int(used),
                    "free": int(free),
                    "total": int(total),
                }
            )
        return stats
    except Exception:
        return None


def _vram_checkpoint(label):
    stats = _query_vram()
    if not stats:
        print(f"VRAM {label}: unavailable")
        return
    parts = []
    for idx, gpu in enumerate(stats):
        parts.append(f"GPU{idx} {gpu['used']}MB/{gpu['total']}MB (free {gpu['free']}MB)")
    print(f"VRAM {label}: " + " | ".join(parts))


def _flush_vram():
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass
    gc.collect()


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
    mach_api_image,
    scan,
    zea_title,
    mach_title,
    mach_api_title,
    full_title,
    stats_text,
    dynamic_range=(-60, 0),
):
    from zea.display import to_8bit

    zea_8bit = to_8bit(zea_image, dynamic_range=dynamic_range, pillow=False)
    mach_8bit = to_8bit(mach_image, dynamic_range=dynamic_range, pillow=False)
    mach_api_8bit = to_8bit(mach_api_image, dynamic_range=dynamic_range, pillow=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=200)
    extent = scan.extent

    axes[0].imshow(zea_8bit, cmap="gray", extent=extent)
    axes[0].set_title(zea_title)
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Z (m)")

    axes[1].imshow(mach_8bit, cmap="gray", extent=extent)
    axes[1].set_title(mach_title)
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Z (m)")

    axes[2].imshow(mach_api_8bit, cmap="gray", extent=extent)
    axes[2].set_title(mach_api_title)
    axes[2].set_xlabel("X (m)")
    axes[2].set_ylabel("Z (m)")

    fig.suptitle(full_title)
    fig.text(0.5, 0.01, stats_text, ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _build_tx_wave_arrivals(scan):
    import mach

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
    return tx_wave_arrivals_s.T


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


# =========================
# Zea pipeline
# =========================


def run_pipeline_zea_example(data_frame, scan, probe):
    import keras

    from zea.ops import Demodulate, Beamform, EnvelopeDetect, LogCompress, Normalize, Pipeline

    pipeline = Pipeline(
        [
            Beamform(num_patches=20),
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

    one_pass = pipeline(**inputs)["data"]

    result, per_iter_s = _time_call(
        "Zea pipeline",
        lambda: pipeline(**inputs)["data"],
        iterations=100,
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


# =========================
# Zea + Mach pipeline
# =========================


def run_pipeline_mach_example(data_frame, scan, probe):
    import keras

    from zea.ops import EnvelopeDetect, LogCompress, MachBeamform, Normalize, Pipeline, ReshapeGrid

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

    _nan_stats("Mach input data", data_frame)

    one_pass = pipeline(**inputs)["data"]
    _block_until_ready(one_pass)

    mach_beamform = MachBeamform(with_batch_dim=False)
    beamformed = mach_beamform(
        data=data_frame,
        flatgrid=scan.flatgrid,
        probe_geometry=probe.probe_geometry,
        sampling_frequency=scan.sampling_frequency,
        sound_speed=scan.sound_speed,
        f_number=scan.f_number,
        demodulation_frequency=scan.demodulation_frequency,
        initial_times=scan.initial_times,
        t0_delays=scan.t0_delays,
        tx_apodizations=scan.tx_apodizations,
        focus_distances=scan.focus_distances,
        polar_angles=scan.polar_angles,
        t_peak=scan.t_peak,
        tx_waveform_indices=scan.tx_waveform_indices,
        transmit_origins=scan.transmit_origins,
        apply_lens_correction=scan.apply_lens_correction,
        lens_thickness=getattr(scan, "lens_thickness", None),
        lens_sound_speed=getattr(scan, "lens_sound_speed", None),
    )["data"]
    beamformed_np = keras.ops.convert_to_numpy(beamformed)
    _nan_stats("Mach beamformed (flat)", beamformed_np)

    result, per_iter_s = _time_call(
        "mach+zea",
        lambda: pipeline(**inputs)["data"],
        iterations=100,
    )
    result = keras.ops.convert_to_numpy(result)
    _nan_stats("mach+zea output", result)
    print("mach+zea input shape:", data_frame.shape)
    print("mach+zea output shape:", result.shape)
    print("mach+zea output dtype:", result.dtype)
    _save_image("mach_pipeline_output.png", result, scan, "mach+zea output")
    points_per_second = _calculate_points_per_second(data_frame, result, per_iter_s)
    title = _format_pipeline_title(
        "mach+zea",
        data_frame.shape,
        result.shape,
        per_iter_s,
        points_per_second,
    )
    return result, title


# =========================
# Mach API only
# =========================


def run_mach_api_example(data_frame, scan, tx_wave_arrivals_s):
    import mach
    import mach.experimental

    try:
        import cupy as cp
    except ImportError:
        cp = None

    if data_frame.shape[-1] == 2:
        data_complex = data_frame[..., 0] + 1j * data_frame[..., 1]
    else:
        data_complex = np.squeeze(data_frame, axis=-1)

    channel_data = np.transpose(data_complex, (0, 2, 1))
    channel_data = np.ascontiguousarray(channel_data[..., None])
    scan_coords_m = np.ascontiguousarray(scan.flatgrid)
    rx_coords_m = np.ascontiguousarray(scan.probe_geometry)
    tx_wave_arrivals_s = np.asarray(tx_wave_arrivals_s)
    if tx_wave_arrivals_s.ndim == 2 and tx_wave_arrivals_s.shape[0] == scan_coords_m.shape[0]:
        tx_wave_arrivals_s = tx_wave_arrivals_s.T
    elif tx_wave_arrivals_s.ndim == 1 and data_frame.shape[0] > 1:
        tx_wave_arrivals_s = np.tile(tx_wave_arrivals_s[None, :], (data_frame.shape[0], 1))

    if cp is not None:
        channel_data = cp.asarray(channel_data)
        scan_coords_m = cp.asarray(scan_coords_m)
        rx_coords_m = cp.asarray(rx_coords_m)
        tx_wave_arrivals_s = cp.asarray(tx_wave_arrivals_s)

    beamform_kwargs = {
        "channel_data": channel_data,
        "rx_coords_m": rx_coords_m,
        "scan_coords_m": scan_coords_m,
        "tx_wave_arrivals_s": tx_wave_arrivals_s,
        "rx_start_s": 0.0,
        "sampling_freq_hz": float(scan.sampling_frequency),
        "f_number": float(scan.f_number),
        "sound_speed_m_s": float(scan.sound_speed),
        "modulation_freq_hz": float(scan.demodulation_frequency),
        "tukey_alpha": 0.0,
    }

    one_pass = mach.experimental.beamform(**beamform_kwargs)
    _block_until_ready(one_pass)

    result, per_iter_s = _time_call(
        "Mach API",
        lambda: mach.experimental.beamform(**beamform_kwargs),
        iterations=100,
    )

    if cp is not None:
        result = cp.asnumpy(result)

    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]

    if result.ndim == 1:
        result = result.reshape(scan.grid.shape[:-1])

    result = np.nan_to_num(result)
    envelope = np.abs(result)
    max_val = np.max(envelope) if envelope.size else 1.0
    envelope = envelope / (max_val + 1e-12)
    mach_api_image = 20.0 * np.log10(envelope + 1e-12)
    mach_api_image = np.clip(mach_api_image, -60.0, 0.0)
    mach_api_image = np.nan_to_num(mach_api_image, nan=-60.0, posinf=0.0, neginf=-60.0)

    points_per_second = _calculate_points_per_second(data_frame, mach_api_image, per_iter_s)
    title = _format_pipeline_title(
        "Mach API",
        data_frame.shape,
        mach_api_image.shape,
        per_iter_s,
        points_per_second,
    )
    return mach_api_image, title


def _report_and_save_diffs(zea_out, mach_out, mach_api_out, scan):
    diff = np.abs(zea_out - mach_out)
    _save_image("pipeline_abs_diff.png", diff, scan, "Pipeline abs diff")

    diff_mach_api = np.abs(zea_out - mach_api_out)
    _save_image(
        "pipeline_abs_diff_mach_api.png", diff_mach_api, scan, "Pipeline abs diff (Mach API)"
    )

    rel_error = np.linalg.norm(zea_out - mach_out) / (np.linalg.norm(zea_out) + 1e-12)
    rel_error_api = np.linalg.norm(zea_out - mach_api_out) / (np.linalg.norm(zea_out) + 1e-12)
    print(f"Pipeline relative error (mach vs zea): {rel_error:.3e}")
    print(f"Pipeline relative error (mach api vs zea): {rel_error_api:.3e}")
    return rel_error, rel_error_api


def main():
    data_frame, scan, probe = _load_picmus_sample()

    zea_out, zea_title = run_pipeline_zea_example(data_frame, scan, probe)
    mach_out, mach_title = run_pipeline_mach_example(data_frame, scan, probe)

    tx_wave_arrivals_s = _build_tx_wave_arrivals(scan)
    mach_api_out, mach_api_title = run_mach_api_example(data_frame, scan, tx_wave_arrivals_s)

    full_title = f"PICMUS pipeline comparison | GPU: {_get_gpu_info()}"
    stats_text = (
        "Zea vs Mach pipeline rel error: "
        f"{np.linalg.norm(zea_out - mach_out) / (np.linalg.norm(zea_out) + 1e-12):.3e} | "
        "Zea vs Mach API rel error: "
        f"{np.linalg.norm(zea_out - mach_api_out) / (np.linalg.norm(zea_out) + 1e-12):.3e}"
    )

    _save_comparison_image(
        "pipeline_outputs_side_by_side.png",
        zea_out,
        mach_out,
        mach_api_out,
        scan,
        zea_title,
        mach_title,
        mach_api_title,
        full_title,
        stats_text,
    )

    _report_and_save_diffs(zea_out, mach_out, mach_api_out, scan)


if __name__ == "__main__":
    main()
