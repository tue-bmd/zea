"""Split a single acquisition into focused and diverging tracks.

Both tracks are saved in one multi-track HDF5 file, re-loaded, beamformed as
B-mode, and exported as a side-by-side animated GIF.
"""

import os

os.environ["MPLBACKEND"] = "Agg"  # for headless environments
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np

import zea
from zea.data.spec import FileSpec, TrackSpec
from zea.io_lib import matplotlib_figure_to_numpy, save_to_gif

zea.init_device()
zea.visualize.set_mpl_style()
zea.log.set_level("INFO")

DATASET_PATH = "hf://zeahub/zea-cardiac-2026"
CONFIG_PATH = "hf://zeahub/zea-cardiac-2026/config.yaml"
SAVE_PATH = Path("multiple-tracks.hdf5")
GIF_PATH = Path("multiple-tracks.gif")

N_GIF_FRAMES = 8

config = zea.Config.from_path(CONFIG_PATH)
config.scan.dynamic_range = (-60, 0)
bmode_pipeline = zea.Pipeline.from_config(config)


def beamform_bmode(raw_data: np.ndarray, scan: zea.Scan, probe: zea.Probe) -> np.ndarray:
    """Beamform raw data to B-mode frames."""
    params = bmode_pipeline.prepare_parameters(probe, scan, config.scan)
    output = bmode_pipeline(data=raw_data[:, scan.selected_transmits], **params)
    images = np.squeeze(keras.ops.convert_to_numpy(output["data"]))
    images = np.nan_to_num(images, nan=float(scan.dynamic_range[0]))
    return zea.display.to_8bit(images, dynamic_range=scan.dynamic_range, pillow=False)


def render_comparison_frames(
    focused_frames: np.ndarray,
    focused_scan: zea.Scan,
    diverging_frames: np.ndarray,
    diverging_scan: zea.Scan,
) -> list[np.ndarray]:
    """Render side-by-side B-mode comparison frames for GIF export."""
    frames = []
    for i in range(len(focused_frames)):
        fig, (ax_f, ax_d) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        ax_f.imshow(
            focused_frames[i],
            cmap="gray",
            aspect="auto",
            extent=focused_scan.extent_imshow * 1e3,
        )
        ax_f.set_title(f"Focused B-mode ({focused_scan.n_tx} tx)")
        ax_f.set_xlabel("x [mm]")
        ax_f.set_ylabel("z [mm]")
        ax_d.imshow(
            diverging_frames[i],
            cmap="gray",
            aspect="auto",
            extent=diverging_scan.extent_imshow * 1e3,
        )
        ax_d.set_title(f"Diverging B-mode ({diverging_scan.n_tx} tx)")
        ax_d.set_xlabel("x [mm]")
        ax_d.set_ylabel("z [mm]")
        frames.append(matplotlib_figure_to_numpy(fig, dpi=150))
        plt.close(fig)
    return frames


def main() -> None:
    # --- Load source acquisition ---
    with zea.Dataset(DATASET_PATH) as dataset:
        source_file = dataset[0]
        probe = source_file.probe()
        source_scan = source_file.scan(**config.scan)
        source_scan.set_transmits("all")
        raw_data = source_file.data.raw_data[:N_GIF_FRAMES]
        probe_name = source_file.attrs.get("probe_name", source_file.attrs.get("probe"))
        us_machine = source_file.attrs.get("us_machine")

    # --- Split into focused and diverging tracks ---
    # Each track gets only its transmit indices and matching scan metadata.
    tracks = []
    for selection in ("focused", "diverging"):
        source_scan.set_transmits(selection)
        tx_indices = np.asarray(source_scan.selected_transmits, dtype=np.int32)
        tracks.append(
            TrackSpec(
                data={"raw_data": raw_data[:, tx_indices]},
                scan={
                    "probe_geometry": np.asarray(source_scan.probe_geometry, dtype=np.float32),
                    "sampling_frequency": np.float32(source_scan.sampling_frequency),
                    "center_frequency": np.float32(source_scan.center_frequency),
                    "demodulation_frequency": np.float32(source_scan.demodulation_frequency),
                    "initial_times": np.asarray(source_scan.initial_times, dtype=np.float32),
                    "t0_delays": np.asarray(source_scan.t0_delays, dtype=np.float32),
                    "tx_apodizations": np.asarray(source_scan.tx_apodizations, dtype=np.float32),
                    "focus_distances": np.asarray(source_scan.focus_distances, dtype=np.float32),
                    "transmit_origins": np.asarray(source_scan.transmit_origins, dtype=np.float32),
                    "polar_angles": np.asarray(source_scan.polar_angles, dtype=np.float32),
                    "sound_speed": np.float32(source_scan.sound_speed),
                    # time_to_next_transmit is (n_frames, n_tx) — truncate to loaded frames
                    "time_to_next_transmit": np.asarray(
                        source_scan.time_to_next_transmit[:N_GIF_FRAMES], dtype=np.float32
                    ),
                },
            )
        )
    source_scan.set_transmits("all")  # restore

    FileSpec(tracks=tracks, probe_name=probe_name, us_machine=us_machine).save(SAVE_PATH)
    print(f"Saved multi-track file to '{SAVE_PATH}'")

    # --- Reload and beamform each track ---
    with zea.File(SAVE_PATH) as saved_file:
        focused_track, diverging_track = saved_file.tracks

        focused_scan_cfg = config.scan.copy()
        focused_scan_cfg.selected_transmits = "focused"
        focused_scan = focused_track.scan(**focused_scan_cfg)
        focused_frames = beamform_bmode(focused_track.data.raw_data[:], focused_scan, probe)

        diverging_scan_cfg = config.scan.copy()
        diverging_scan_cfg.selected_transmits = "diverging"
        diverging_scan = diverging_track.scan(**diverging_scan_cfg)
        diverging_frames = beamform_bmode(diverging_track.data.raw_data[:], diverging_scan, probe)

    # --- Export GIF ---
    gif_frames = render_comparison_frames(
        focused_frames, focused_scan, diverging_frames, diverging_scan
    )
    save_to_gif(gif_frames, GIF_PATH, fps=10)
    print(f"Saved GIF to '{GIF_PATH}'")


if __name__ == "__main__":
    main()
