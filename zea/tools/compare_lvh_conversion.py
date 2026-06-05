"""
Sanity-check the EchoNet-LVH conversion on this branch against the older
``openh-rf`` branch implementation by side-by-side rendering of intermediate
steps on a handful of AVI files.

For each input AVI this script saves into ``--output_dir``:

1. ``<stem>_scan_cone_visualization.png`` — the cone detection overlay
   produced by :func:`zea.tools.fit_scan_cone.visualize_scan_cone`.
2. ``<stem>_comparison.png`` — a 1x4 panel of:
   original first frame | cropped+centered frame | polar (this branch) |
   polar (openh-rf style).

The only meaningful difference between the two branches is the polar
conversion: the old branch passed ``angle=opening_angle/2`` (symmetric, default
tip/r_max), while this branch builds an asymmetric ``theta_range`` from the
fitted left/right slopes, uses the detected apex as the polar tip and the
circle radius as ``r_max``.
"""

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import matplotlib.pyplot as plt
import numpy as np
from keras import ops

from zea import log
from zea.data.convert.echonetlvh import LVHProcessor
from zea.display import cartesian_to_polar_matrix
from zea.tools.fit_scan_cone import (
    _load_first_frame,
    crop_and_center_cone,
    detect_cone_parameters,
    visualize_scan_cone,
)


def polar_this_branch(cropped_frame, cone_params):
    """Polar conversion as performed on this branch (asymmetric, apex-anchored)."""
    apex_x_in_crop = cone_params["apex_x"] - cone_params["crop_left"]
    cropped_width = cone_params["crop_right"] - cone_params["crop_left"]
    left_padding = max(0, int(cropped_width / 2 - apex_x_in_crop))
    tip_x = apex_x_in_crop + left_padding
    tip_y = cone_params["apex_y"] - cone_params["crop_top"]
    theta_min = -math.atan(cone_params["right_slope"])
    theta_max = -math.atan(cone_params["left_slope"])
    return cartesian_to_polar_matrix(
        cropped_frame,
        tip=(tip_x, tip_y),
        r_max=cone_params["circle_radius"],
        theta_range=(theta_min, theta_max),
    )


def polar_openh_rf(cropped_frame, cone_params):
    """Polar conversion as performed on the openh-rf branch (symmetric, defaults)."""
    angle = cone_params["opening_angle"] / 2
    theta_range = (-angle, angle)
    return cartesian_to_polar_matrix(cropped_frame, theta_range=theta_range)


def polar_direct(original_frame, cone_params):
    """Polar conversion straight from the uncropped frame using apex coordinates."""
    theta_min = -math.atan(cone_params["right_slope"])
    theta_max = -math.atan(cone_params["left_slope"])
    return cartesian_to_polar_matrix(
        original_frame,
        tip=(cone_params["apex_x"], cone_params["apex_y"]),
        r_max=cone_params["circle_radius"],
        theta_range=(theta_min, theta_max),
    )


def save_comparison(
    original, cropped, polar_new, polar_old, polar_no_crop, back_cartesian, out_path
):
    """Save a 1x5 side-by-side comparison figure on a white background."""
    titles = [
        "Original first frame",
        "Cropped + centered",
        "Polar (this branch)\nfrom cropped frame",
        "Polar (openh-rf)\nsymmetric, default tip/r_max",
        "Polar direct (no crop)\napex_x, apex_y, r_max, theta_range",
        "Polar back to cartesian",
        "Error",
    ]
    fig, axes = plt.subplots(1, len(titles), figsize=(25, 6), facecolor="white")
    error = np.abs(cropped - back_cartesian)
    images = [original, cropped, polar_new, polar_old, polar_no_crop, back_cartesian, error]
    for ax, img, title in zip(axes, images, titles):
        if title == "Error":
            vmin = 0
            vmax = 10
        else:
            vmin = None
            vmax = None
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, color="black")
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def process_one(avi_path: Path, output_dir: Path):
    log.info(f"Processing {log.yellow(avi_path)}")
    frame = _load_first_frame(avi_path)

    cone_params = detect_cone_parameters(frame)
    if cone_params is None:
        log.error(f"  cone detection failed for {avi_path}")
        return

    visualize_scan_cone(frame, cone_params, output_dir=output_dir)
    (output_dir / "scan_cone_visualization.png").rename(
        output_dir / f"{avi_path.stem}_scan_cone_visualization.png"
    )

    frame_f32 = ops.cast(frame, "float32")
    cropped = crop_and_center_cone(frame_f32, cone_params, backend=ops)

    polar_new = polar_this_branch(cropped, cone_params)
    polar_old = polar_openh_rf(cropped, cone_params)
    polar_no_crop = polar_direct(frame_f32, cone_params)

    back_cartesian = LVHProcessor.scan_convert(polar_no_crop, cone_params, cropped.shape)

    save_comparison(
        np.asarray(frame),
        np.asarray(cropped),
        np.asarray(polar_new),
        np.asarray(polar_old),
        np.asarray(polar_no_crop),
        np.asarray(back_cartesian),
        output_dir / f"{avi_path.stem}_comparison.png",
    )
    log.info(f"  saved {avi_path.stem}_comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="One or more AVI file paths to compare.",
    )
    parser.add_argument(
        "--output_dir",
        default="output/lvh_branch_comparison",
        help="Directory where PNGs are written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for avi in args.input_files:
        process_one(Path(avi), output_dir)

    log.info(f"All comparisons written to {log.yellow(output_dir)}")


if __name__ == "__main__":
    main()
