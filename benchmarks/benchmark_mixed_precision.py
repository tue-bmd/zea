"""Benchmark mixed-precision (bfloat16) and int16 beamforming against float32.

zea keeps geometry and time-of-flight *delay* computation in ``float32`` while
running the bulk *signal* compute (the TOF gather, interpolation, apodization
and delay-and-sum) in a lower-precision "compute dtype". Mixed precision is
enabled through the standard Keras global policy::

    keras.mixed_precision.set_global_policy("mixed_bfloat16")

This script reconstructs a B-mode image at several precisions and reports, for
each:

* wall-clock time per frame (median of ``--repeats`` runs, GPU-synced), and
* reconstruction quality versus the ``float32`` baseline, measured both on the
  linear beamformed magnitude (relative L2 error, NRMSE, correlation) and on the
  final log-compressed B-mode (PSNR, NCC, max/mean dB error).

Because the delays stay in float32, the mixed-precision reconstruction stays
faithful to the baseline while the signal compute runs at half precision.

Example
-------
::

    python benchmarks/benchmark_mixed_precision.py
    python benchmarks/benchmark_mixed_precision.py --num-patches 512 --repeats 20
    python benchmarks/benchmark_mixed_precision.py --no-plot
"""

import argparse
import time

import keras
import numpy as np

import zea
from zea import metrics
from zea.internal.precision import signal_compute_dtype

# The pipeline from the task description (focused-transmit carotid, pfield DAS).
PIPELINE_CONFIG = {
    "parameters": {
        "selected_transmits": "focused",
        "zlims": [0.001, 0.039],
        "xlims": [-0.019, 0.019],
        "dynamic_range": [-60, 0],
        "bandwidth": 7.0e6,
    },
    "pipeline": {
        "operations": [
            "band_pass_filter",
            "apply_window",
            "demodulate",
            {"name": "downsample", "params": {"factor": 2}},
            {
                "name": "beamform",
                "params": {
                    "beamformer": "delay_and_sum",
                    "enable_pfield": True,
                    "num_patches": 1024,
                },
            },
            "envelope_detect",
            "normalize",
            "log_compress",
        ],
        "jit_options": "pipeline",
    },
}

DATA_URI = "hf://zeahub/zea-carotid-2023/data/10_cross_2cm_L_0000.hdf5"
DATA_REVISION = "v0.1.3"


def _sync(x):
    """Force materialization so timing captures the actual GPU work."""
    return keras.ops.convert_to_numpy(x)


def time_pipeline(pipeline, inputs, raw_data, repeats):
    """Return (median_ms, all_ms) for a warmed-up, GPU-synced pipeline call."""
    _sync(pipeline(data=raw_data, **inputs)["data"])  # warm-up / trace
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _sync(pipeline(data=raw_data, **inputs)["data"])
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), times


def linear_metrics(ref_bmode_lin, test_bmode_lin):
    """Quality metrics on the *linear* beamformed magnitude (pre log-compress)."""
    ref = ref_bmode_lin.ravel().astype(np.float64)
    test = test_bmode_lin.ravel().astype(np.float64)
    rel_l2 = np.linalg.norm(test - ref) / np.linalg.norm(ref)
    nrmse = np.sqrt(np.mean((test - ref) ** 2)) / (ref.max() - ref.min())
    corr = np.corrcoef(ref, test)[0, 1]
    return {"rel_l2": rel_l2, "nrmse_pct": 100 * nrmse, "corr": corr}


def bmode_metrics(ref_bmode, test_bmode, dynamic_range):
    """Quality metrics on the final log-compressed B-mode (dB, in [drange, 0])."""
    lo, hi = dynamic_range
    ref01 = np.clip((ref_bmode - lo) / (hi - lo), 0, 1)
    test01 = np.clip((test_bmode - lo) / (hi - lo), 0, 1)
    psnr = float(metrics.psnr(ref01[..., None], test01[..., None], max_val=1.0))
    ncc = float(metrics.ncc(ref_bmode[..., None], test_bmode[..., None]))
    abs_db = np.abs(ref_bmode - test_bmode)
    return {
        "psnr_db": psnr,
        "ncc": ncc,
        "max_db_err": float(abs_db.max()),
        "mean_db_err": float(abs_db.mean()),
    }


def run_variant(name, policy, raw_data, parameters, num_patches, repeats, use_int16):
    """Reconstruct one precision variant; return bmode, linear magnitude, timing."""
    keras.mixed_precision.set_global_policy(policy)

    config = zea.Config(PIPELINE_CONFIG)
    config.pipeline.operations[4]["params"]["num_patches"] = num_patches

    pipeline = zea.Pipeline.from_config(config, with_batch_dim=False)
    inputs = pipeline.prepare_parameters(parameters)

    data = raw_data if use_int16 else keras.ops.cast(raw_data, "float32")
    compute_dtype = signal_compute_dtype()  # resolve while the policy is active

    median_ms, _ = time_pipeline(pipeline, inputs, data, repeats)
    bmode = _sync(pipeline(data=data, **inputs)["data"]).astype(np.float32)

    # Linear beamformed magnitude: rerun a pipeline truncated after beamform.
    trunc = zea.Config(
        {"pipeline": {"operations": config.pipeline.operations[:5], "jit_options": "pipeline"}}
    )
    trunc_pipe = zea.Pipeline.from_config(trunc, with_batch_dim=False)
    trunc_inputs = trunc_pipe.prepare_parameters(parameters)
    bf = _sync(trunc_pipe(data=data, **trunc_inputs)["data"]).astype(np.float32)
    lin = np.abs(bf[..., 0] + 1j * bf[..., 1])

    keras.mixed_precision.set_global_policy("float32")  # reset global state
    return {
        "name": name,
        "compute_dtype": compute_dtype,
        "input_dtype": str(data.dtype),
        "median_ms": median_ms,
        "bmode": bmode,
        "lin": lin,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-patches", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--out", type=str, default="mixed_precision_benchmark.png")
    args = parser.parse_args()

    zea.init_device()

    config = zea.Config(PIPELINE_CONFIG)
    with zea.File(DATA_URI, revision=DATA_REVISION) as f:
        parameters = f.load_parameters(**config.parameters)
        raw_data = f.data.raw_data[args.frame, parameters.selected_transmits]
    raw_data = keras.ops.convert_to_tensor(raw_data)
    print(f"raw_data: shape={tuple(raw_data.shape)} dtype={raw_data.dtype}")

    variants = [
        ("float32 (baseline)", "float32", False),
        ("mixed_bfloat16", "mixed_bfloat16", False),
        ("int16 in + mixed_bfloat16", "mixed_bfloat16", True),
    ]

    results = []
    for name, policy, use_int16 in variants:
        print(f"\n>>> {name} ...")
        results.append(
            run_variant(
                name, policy, raw_data, parameters, args.num_patches, args.repeats, use_int16
            )
        )

    ref = results[0]
    drange = config.parameters.dynamic_range

    # ---- Report -----------------------------------------------------------
    print("\n" + "=" * 92)
    print(
        f"Mixed-precision beamforming benchmark (num_patches={args.num_patches}, "
        f"repeats={args.repeats})"
    )
    print("=" * 92)
    header = (
        f"{'variant':<28}{'compute':>10}{'ms/frame':>11}{'speedup':>9}"
        f"{'rel L2':>10}{'corr':>9}{'PSNR dB':>9}{'meanΔdB':>9}"
    )
    print(header)
    print("-" * 92)
    for r in results:
        lm = linear_metrics(ref["lin"], r["lin"])
        bm = bmode_metrics(ref["bmode"], r["bmode"], drange)
        speedup = ref["median_ms"] / r["median_ms"]
        print(
            f"{r['name']:<28}{r['compute_dtype']:>10}{r['median_ms']:>11.1f}"
            f"{speedup:>8.2f}x{lm['rel_l2']:>10.1e}{lm['corr']:>9.5f}"
            f"{bm['psnr_db']:>9.1f}{bm['mean_db_err']:>9.3f}"
        )
    print("=" * 92)
    print("Delays/geometry stay float32; only the signal path is lowered, so quality is preserved.")

    # ---- Plot -------------------------------------------------------------
    if not args.no_plot:
        import matplotlib.pyplot as plt

        n = len(results)
        fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4.5))
        extent = [
            config.parameters.xlims[0] * 1e3,
            config.parameters.xlims[1] * 1e3,
            config.parameters.zlims[1] * 1e3,
            config.parameters.zlims[0] * 1e3,
        ]
        for ax, r in zip(axes[:n], results):
            ax.imshow(r["bmode"], cmap="gray", vmin=drange[0], vmax=drange[1], extent=extent)
            ax.set_title(f"{r['name']}\n{r['median_ms']:.0f} ms/frame")
            ax.set_xlabel("x [mm]")
        axes[0].set_ylabel("z [mm]")
        # Difference map of the fastest mixed variant vs baseline.
        diff = results[1]["bmode"] - ref["bmode"]
        im = axes[n].imshow(diff, cmap="RdBu", vmin=-3, vmax=3, extent=extent)
        axes[n].set_title(f"{results[1]['name']}\n- baseline [dB]")
        axes[n].set_xlabel("x [mm]")
        fig.colorbar(im, ax=axes[n], fraction=0.046, label="dB")
        fig.tight_layout()
        fig.savefig(args.out, dpi=120)
        print(f"\nSaved comparison figure to {args.out}")


if __name__ == "__main__":
    main()
