"""
Survey the EchoNet-LVH AVI files to pick a fixed ``polar_shape`` for the
conversion.

For each AVI this script records:

- The raw frame size (height, width) reported by OpenCV.
- The fitted scan-cone dimensions produced by
  :func:`zea.tools.fit_scan_cone.detect_cone_parameters`:
  ``circle_radius``, ``opening_angle``, ``new_height``, ``new_width``.

It then reports summary statistics for each of these quantities, flags files
that deviate from the median by more than ``--outlier_sigma`` standard
deviations, prints a suggested ``polar_shape`` (rows = radial samples,
cols = angular samples), and optionally saves a histogram PNG.

Run with::

    python -m zea.tools.inspect_lvh_dimensions \
        --source_dir /path/to/EchoNet-LVH \
        --max_files 500
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from zea import log
from zea.tools.fit_scan_cone import _load_first_frame, detect_cone_parameters

METRICS = [
    "frame_height",
    "frame_width",
    "circle_radius",
    "opening_angle_deg",
    "new_height",
    "new_width",
]


def gather_one(avi_path: Path):
    """Return a dict of measurements for a single AVI, or None on failure."""
    try:
        frame = _load_first_frame(avi_path)
        params = detect_cone_parameters(frame)
        if params is None:
            return {"avi_path": str(avi_path), "error": "cone_detection_failed"}
        return {
            "avi_path": str(avi_path),
            "frame_height": frame.shape[0],
            "frame_width": frame.shape[1],
            "circle_radius": params["circle_radius"],
            "opening_angle_deg": float(np.degrees(params["opening_angle"])),
            "new_height": params["new_height"],
            "new_width": params["new_width"],
        }
    except Exception as exc:
        return {"avi_path": str(avi_path), "error": str(exc)}


def find_avis(source_dir: Path):
    """Walk Batch* directories and return all .avi paths."""
    return sorted(p for p in source_dir.glob("Batch*/*.avi"))


def summarize(values: np.ndarray, name: str):
    """Return (mean, median, std, p5, p95, min, max) as a printable dict."""
    return {
        "metric": name,
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def print_summary(rows):
    header = (
        f"{'metric':<20} {'n':>6} {'mean':>10} {'median':>10} {'std':>10} {'p5':>10} "
        + f"{'p95':>10} {'min':>10} {'max':>10}"
    )
    log.info(header)
    log.info("-" * len(header))
    for r in rows:
        log.info(
            f"{r['metric']:<20} {r['n']:>6d} {r['mean']:>10.2f} {r['median']:>10.2f} "
            f"{r['std']:>10.2f} {r['p5']:>10.2f} {r['p95']:>10.2f} "
            f"{r['min']:>10.2f} {r['max']:>10.2f}"
        )


def flag_outliers(records, metric, outlier_sigma):
    values = np.array([r[metric] for r in records])
    median = np.median(values)
    std = np.std(values)
    if std == 0:
        return []
    deviations = np.abs(values - median) / std
    outlier_indices = np.where(deviations > outlier_sigma)[0]
    return [(records[i]["avi_path"], values[i], deviations[i]) for i in outlier_indices]


def save_histogram(records, output_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="white")
    for ax, metric in zip(axes.ravel(), METRICS):
        vals = np.array([r[metric] for r in records])
        ax.hist(vals, bins=40, color="steelblue", edgecolor="black")
        ax.axvline(
            np.median(vals), color="red", linestyle="--", label=f"median={np.median(vals):.1f}"
        )
        ax.set_title(metric, color="black")
        ax.set_facecolor("white")
        ax.legend(loc="upper right")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def recommend_polar_shape(records):
    """Suggest polar_shape (rows, cols).

    rows: radial samples — use p95 of circle_radius rounded up to nearest 16.
    cols: angular samples — use p95 of new_width rounded up to nearest 16.
    """
    radii = np.array([r["circle_radius"] for r in records])
    widths = np.array([r["new_width"] for r in records])
    rows = int(np.ceil(np.percentile(radii, 95) / 16) * 16)
    cols = int(np.ceil(np.percentile(widths, 95) / 16) * 16)
    return rows, cols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_dir",
        required=True,
        type=Path,
        help="EchoNet-LVH root containing Batch1..Batch4 directories.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Cap the number of files scanned (default: all).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Thread pool size for loading AVI first frames.",
    )
    parser.add_argument(
        "--outlier_sigma",
        type=float,
        default=3.0,
        help="Flag files whose metric deviates more than this many std from the median.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/lvh_dimensions"),
        help="Where to save the histogram PNG and a CSV of per-file measurements.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    avis = find_avis(args.source_dir)
    if args.max_files is not None:
        avis = avis[: args.max_files]
    log.info(f"Found {len(avis)} AVI files under {log.yellow(args.source_dir)}")

    records = []
    errors = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(gather_one, p): p for p in avis}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning AVIs"):
            result = fut.result()
            if "error" in result:
                errors.append(result)
            else:
                records.append(result)

    log.info(f"Scanned {len(records)} files successfully, {len(errors)} failed.")
    for e in errors[:10]:
        log.warning(f"  failed: {e['avi_path']} ({e['error']})")
    if len(errors) > 10:
        log.warning(f"  ... and {len(errors) - 10} more failures")

    if not records:
        log.error("No successful measurements; nothing to summarize.")
        return

    log.info("Summary statistics:")
    print_summary([summarize(np.array([r[m] for r in records]), m) for m in METRICS])

    log.info(f"Outliers (>|{args.outlier_sigma}| sigma from median):")
    any_outliers = False
    for metric in METRICS:
        outs = flag_outliers(records, metric, args.outlier_sigma)
        if outs:
            any_outliers = True
            log.warning(f"  {metric}: {len(outs)} files")
            for path, value, dev in outs[:5]:
                log.warning(f"    {path}: {value:.2f} ({dev:.2f}σ)")
            if len(outs) > 5:
                log.warning(f"    ... and {len(outs) - 5} more")
    if not any_outliers:
        log.info("  (none)")

    rows, cols = recommend_polar_shape(records)
    log.info(
        f"Recommended polar_shape = ({rows}, {cols}) "
        "[p95(circle_radius), p95(new_width), rounded up to 16]"
    )

    # Save CSV
    csv_path = args.output_dir / "lvh_dimensions.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("avi_path," + ",".join(METRICS) + "\n")
        for r in records:
            f.write(r["avi_path"] + "," + ",".join(str(r[m]) for m in METRICS) + "\n")
    log.info(f"Per-file measurements saved to {log.yellow(csv_path)}")

    hist_path = args.output_dir / "lvh_dimensions_histograms.png"
    save_histogram(records, hist_path)
    log.info(f"Histograms saved to {log.yellow(hist_path)}")


if __name__ == "__main__":
    main()
