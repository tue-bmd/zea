"""Standalone MachBeamform debug script."""

import time

import matplotlib.pyplot as plt
import numpy as np


def _block_until_ready(value):
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def _time_call(label, fn, iterations=10, warmup=2):
    result = None
    for _ in range(warmup):
        result = fn()
    _block_until_ready(result)

    start = time.perf_counter()
    for _ in range(iterations):
        result = fn()
    _block_until_ready(result)
    elapsed = time.perf_counter() - start
    print(f"{label} total time ({iterations} iters): {elapsed:.3f} s")
    print(f"{label} per iter: {elapsed / iterations * 1e3:.3f} ms")
    return result


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


def _zea_beamform_reference(
    data_frames,
    flatgrid,
    probe_geometry,
    sampling_frequency,
    sound_speed,
    f_number,
    demodulation_frequency,
):
    import keras

    from zea.beamform.beamformer import tof_correction
    from zea.ops import DelayAndSum

    n_tx, n_el = data_frames.shape[1], data_frames.shape[3]

    t0_delays = np.zeros((n_tx, n_el), dtype="float32")
    tx_apodizations = np.ones((n_tx, n_el), dtype="float32")
    polar_angles = np.zeros((n_tx,), dtype="float32")
    focus_distances = np.ones((n_tx,), dtype="float32")
    t_peak = np.zeros((1,), dtype="float32")
    tx_waveform_indices = np.zeros((n_tx,), dtype="int32")
    transmit_origins = np.zeros((n_tx, 3), dtype="float32")
    initial_times = np.zeros((n_tx,), dtype="float32")

    flatgrid_t = keras.ops.convert_to_tensor(flatgrid)
    probe_geometry_t = keras.ops.convert_to_tensor(probe_geometry)
    t0_delays_t = keras.ops.convert_to_tensor(t0_delays)
    tx_apodizations_t = keras.ops.convert_to_tensor(tx_apodizations)
    polar_angles_t = keras.ops.convert_to_tensor(polar_angles)
    focus_distances_t = keras.ops.convert_to_tensor(focus_distances)
    t_peak_t = keras.ops.convert_to_tensor(t_peak)
    tx_waveform_indices_t = keras.ops.convert_to_tensor(tx_waveform_indices)
    transmit_origins_t = keras.ops.convert_to_tensor(transmit_origins)
    initial_times_t = keras.ops.convert_to_tensor(initial_times)

    das = DelayAndSum(with_batch_dim=False)
    results = []
    for frame in range(data_frames.shape[0]):
        frame_data = keras.ops.convert_to_tensor(data_frames[frame])
        aligned = tof_correction(
            frame_data,
            flatgrid=flatgrid_t,
            t0_delays=t0_delays_t,
            tx_apodizations=tx_apodizations_t,
            sound_speed=sound_speed,
            probe_geometry=probe_geometry_t,
            initial_times=initial_times_t,
            sampling_frequency=sampling_frequency,
            demodulation_frequency=demodulation_frequency,
            f_number=f_number,
            polar_angles=polar_angles_t,
            focus_distances=focus_distances_t,
            t_peak=t_peak_t,
            tx_waveform_indices=tx_waveform_indices_t,
            transmit_origins=transmit_origins_t,
        )
        beamformed = das(data=aligned)["data"]
        results.append(keras.ops.convert_to_numpy(beamformed))

    return np.stack(results, axis=0)


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

    scan.set_transmits(1)
    data_frame = data[0][scan.selected_transmits]
    return data_frame, scan, probe


def run_rf_example():
    from zea.ops import MachBeamform

    rng = np.random.default_rng(0)

    n_frames = 2
    n_tx = 1
    n_ax = 8
    n_el = 4
    n_pix = 6

    data = rng.standard_normal((n_frames, n_tx, n_ax, n_el, 1)).astype("float32")
    flatgrid = rng.standard_normal((n_pix, 3)).astype("float32")
    probe_geometry = rng.standard_normal((n_el, 3)).astype("float32")
    tx_wave_arrivals_s = np.zeros((n_tx, n_pix), dtype="float32")

    op = MachBeamform()
    result = _time_call(
        "MachBeamform RF",
        lambda: op(
            data=data,
            flatgrid=flatgrid,
            probe_geometry=probe_geometry,
            tx_wave_arrivals_s=tx_wave_arrivals_s,
            sampling_frequency=40e6,
            sound_speed=1540.0,
            f_number=1.5,
            demodulation_frequency=0.0,
            initial_times=np.zeros((n_tx,), dtype="float32"),
        )["data"],
    )
    result = np.asarray(result)
    print("RF result shape:", result.shape)
    print("RF result dtype:", result.dtype)

    zea_result = _time_call(
        "Zea beamformer RF",
        lambda: _zea_beamform_reference(
            data,
            flatgrid,
            probe_geometry,
            sampling_frequency=40e6,
            sound_speed=1540.0,
            f_number=1.5,
            demodulation_frequency=0.0,
        ),
    )
    rel_error = np.linalg.norm(result - zea_result) / (np.linalg.norm(zea_result) + 1e-12)
    print(f"RF relative error vs zea: {rel_error:.3e}")


def run_iq_example():
    from zea.ops import MachBeamform

    rng = np.random.default_rng(1)

    n_frames = 3
    n_tx = 1
    n_ax = 6
    n_el = 3
    n_pix = 5

    data = rng.standard_normal((n_frames, n_tx, n_ax, n_el, 2)).astype("float32")
    flatgrid = rng.standard_normal((n_pix, 3)).astype("float32")
    probe_geometry = rng.standard_normal((n_el, 3)).astype("float32")
    tx_wave_arrivals_s = np.zeros((n_tx, n_pix), dtype="float32")

    op = MachBeamform()
    result = _time_call(
        "MachBeamform IQ",
        lambda: op(
            data=data,
            flatgrid=flatgrid,
            probe_geometry=probe_geometry,
            tx_wave_arrivals_s=tx_wave_arrivals_s,
            sampling_frequency=20e6,
            sound_speed=1540.0,
            f_number=1.2,
            demodulation_frequency=5e6,
            initial_times=np.zeros((n_tx,), dtype="float32"),
        )["data"],
    )
    result = np.asarray(result)
    print("IQ result shape:", result.shape)
    print("IQ result dtype:", result.dtype)

    zea_result = _time_call(
        "Zea beamformer IQ",
        lambda: _zea_beamform_reference(
            data,
            flatgrid,
            probe_geometry,
            sampling_frequency=20e6,
            sound_speed=1540.0,
            f_number=1.2,
            demodulation_frequency=5e6,
        ),
    )
    rel_error = np.linalg.norm(result - zea_result) / (np.linalg.norm(zea_result) + 1e-12)
    print(f"IQ relative error vs zea: {rel_error:.3e}")


def _bytes_to_mb(num_bytes):
    return num_bytes / (1024 * 1024)


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
        jit_options="ops",
    )

    inputs = pipeline.prepare_parameters(probe=probe, scan=scan)
    inputs["data"] = data_frame
    inputs["demodulation_frequency"] = 0.0

    result = _time_call(
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
    return result


def run_pipeline_mach_example(data_frame, scan, probe):
    import keras

    import mach
    from zea.ops import EnvelopeDetect, LogCompress, MachBeamform, Normalize, Pipeline, ReshapeGrid

    origin = np.array([0.0, 0.0, 0.0], dtype="float32")
    direction = np.array([0.0, 0.0, 1.0], dtype="float32")
    tx_wave_arrivals_s = (
        mach.wavefront.plane(origin_m=origin, points_m=scan.flatgrid, direction=direction)
        / scan.sound_speed
    )
    tx_wave_arrivals_s = tx_wave_arrivals_s[None, :]

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
    inputs["demodulation_frequency"] = 0.0

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
        demodulation_frequency=0.0,
        initial_times=scan.initial_times,
    )["data"]
    beamformed_np = keras.ops.convert_to_numpy(beamformed)
    _nan_stats("Mach beamformed (flat)", beamformed_np)

    result = _time_call(
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
    return result


if __name__ == "__main__":
    # run_rf_example()
    # run_iq_example()
    data_frame, scan, probe = _load_picmus_sample()
    zea_out = run_pipeline_zea_example(data_frame, scan, probe)
    mach_out = run_pipeline_mach_example(data_frame, scan, probe)
    diff = np.abs(zea_out - mach_out)
    _save_image("pipeline_abs_diff.png", diff, scan, "Pipeline abs diff")
    rel_error = np.linalg.norm(zea_out - mach_out) / (np.linalg.norm(zea_out) + 1e-12)
    print(f"Pipeline relative error (mach vs zea): {rel_error:.3e}")
