"""Profile the beamforming pipeline.

Runs on the JAX backend by default; pass ``--tensorflow`` to run on TensorFlow instead.

Reports compile time, steady-state latency (mean/std/min over repeats), throughput,
and device memory. Optionally saves a reference output (``--save-output``) or checks
the current output against a saved reference (``--check-against``) so optimizations
can be validated for numerical equivalence.

Example usage:

    python scripts/profile_beamforming.py --tag baseline
    python scripts/profile_beamforming.py --tensorflow --tag tf_baseline
    python scripts/profile_beamforming.py --num-patches 20 --tag fewer_patches
    python scripts/profile_beamforming.py --per-op
    python scripts/profile_beamforming.py --trace /tmp/jax-trace  # view in xprof/tensorboard
"""

import argparse
import json
import os
import time

# Cardiac apical-4-chamber case s1 (focused transmits, DMAS + pfield, polar grid).
# See https://huggingface.co/datasets/zeahub/zea-cardiac-2026
DATA_PATH = "hf://zeahub/zea-cardiac-2026/20251222_s1_a4ch_line_dw_0000.hdf5"
CONFIG_PATH = "hf://zeahub/zea-cardiac-2026/config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tensorflow",
        action="store_true",
        help="Use the TensorFlow backend instead of the default JAX backend.",
    )
    parser.add_argument("--data", default=DATA_PATH, help="Path to a zea file with raw_data.")
    parser.add_argument("--jit-options", default="pipeline", help="[ops/pipeline]")
    parser.add_argument(
        "--config", default=CONFIG_PATH, help="Path to a zea config file (parameters + pipeline)."
    )
    parser.add_argument("--num-patches", type=int, default=100, help="Beamform num_patches.")
    parser.add_argument(
        "--grid-size",
        default=None,
        help="Override the beamforming grid as 'ZxX' (e.g. 512x768: grid_size_z=512, "
        "grid_size_x=768). Default uses the config's derived grid.",
    )
    parser.add_argument(
        "--beamformer",
        default=None,
        help="Override the beamformer type (e.g. delay_and_sum, delay_multiply_and_sum). "
        "Changes the output, so it can't be checked against a DMAS reference.",
    )
    parser.add_argument(
        "--pfield",
        choices=["on", "off"],
        default=None,
        help="Override pressure-field weighting (enable_pfield). Default uses the config value.",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup calls after compilation.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed steady-state calls.")
    parser.add_argument("--device", default="auto:1", help="Device passed to init_device.")
    parser.add_argument(
        "--per-op",
        action="store_true",
        help="Also report a per-operation breakdown (each op jitted separately, "
        "so totals differ from the fused pipeline).",
    )
    parser.add_argument("--trace", default=None, help="Directory for a JAX profiler trace.")
    parser.add_argument("--save-output", default=None, help="Save output image to this .npz.")
    parser.add_argument(
        "--check-against", default=None, help="Compare output against a saved .npz reference."
    )
    parser.add_argument("--tag", default=None, help="Label recorded in the JSON result line.")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Append a JSON line with the results to this file (for sweeps).",
    )
    return parser.parse_args()


def block(outputs, backend):
    """Force computation to finish so elapsed time reflects actual compute."""
    if backend == "jax":
        import jax

        jax.block_until_ready(outputs)
    else:
        from keras import tree

        # No public "block until ready" for tensorflow/torch; forcing a host
        # transfer via .numpy() waits for the underlying op to complete.
        tree.map_structure(lambda x: x.numpy() if hasattr(x, "numpy") else x, outputs)
    return outputs


def time_pipeline(pipeline, inputs, warmup, repeats, backend):
    """Return (compile_seconds, [steady-state seconds]) for the jitted pipeline."""
    t0 = time.perf_counter()
    block(pipeline(**inputs), backend)
    compile_s = time.perf_counter() - t0

    for _ in range(warmup):
        block(pipeline(**inputs), backend)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        block(pipeline(**inputs), backend)
        times.append(time.perf_counter() - t0)
    return compile_s, times


def report_memory(backend):
    if backend == "jax":
        import jax

        stats = jax.local_devices()[0].memory_stats() or {}
        return {
            "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
            "bytes_in_use": stats.get("bytes_in_use"),
        }

    import tensorflow as tf

    try:
        stats = tf.config.experimental.get_memory_info("GPU:0")
    except ValueError:
        return {"peak_bytes_in_use": None, "bytes_in_use": None}
    return {"peak_bytes_in_use": stats["peak"], "bytes_in_use": stats["current"]}


def per_op_breakdown(config, inputs, warmup, repeats, backend):
    """Time each top-level op separately (jit_options='ops'). Indicative only:
    per-op jitting prevents cross-op fusion, so the sum exceeds the fused time.

    Builds the same pipeline as the config but with ``jit_options='ops'`` so each
    op is jitted (and timed) on its own."""
    import numpy as np

    import zea

    pipeline = zea.Pipeline.from_config(config, with_batch_dim=False, jit_options="ops")

    # Warm up (compiles every op)
    for _ in range(max(warmup, 1)):
        block(pipeline(**inputs), backend)

    op_times: dict[str, list] = {}
    for _ in range(repeats):
        stage_inputs = dict(inputs)
        for op in pipeline.operations:
            name = op.__class__.__name__
            t0 = time.perf_counter()
            stage_inputs = block(op(**stage_inputs), backend)
            op_times.setdefault(name, []).append(time.perf_counter() - t0)

    return {name: float(np.mean(ts)) for name, ts in op_times.items()}


def main():
    args = parse_args()

    backend = "tensorflow" if args.tensorflow else "jax"
    os.environ["KERAS_BACKEND"] = backend
    import zea  # noqa: E402

    from zea import init_device

    init_device(device=args.device, verbose=False)

    import numpy as np
    from keras import ops as kops

    if backend == "jax":
        import jax

        device_kind = jax.local_devices()[0].device_kind
    else:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        device_kind = (
            tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU")
            if gpus
            else "CPU"
        )

    with zea.File(args.data) as f:
        data = f.data.raw_data[0]  # first frame, shape (n_tx, n_ax, n_el, n_ch)
        parameters = f.load_parameters()

    config = zea.Config.from_path(args.config)
    # Override beamform op params. The op entry is {"name": "beamform", "params": {...}};
    # overrides must land inside ``params`` to reach the constructed op.
    beamform_params = config["pipeline"]["operations"]["beamform"].setdefault("params", {})
    if args.num_patches is not None:
        beamform_params["num_patches"] = args.num_patches
    if args.beamformer is not None:
        beamform_params["beamformer"] = args.beamformer
    if args.pfield is not None:
        beamform_params["enable_pfield"] = args.pfield == "on"

    if args.grid_size is not None:
        gz, gx = (int(v) for v in args.grid_size.lower().split("x"))
        # grid_size_z / grid_size_x override the wavelength-derived grid when set.
        config["parameters"]["grid_size_z"] = gz
        config["parameters"]["grid_size_x"] = gx

    parameters.update(**config.parameters)

    # using the pipeline as specified in the config file
    pipeline = zea.Pipeline.from_config(
        config,
        with_batch_dim=False,
        jit_options=args.jit_options,
    )
    # prepare the inputs (converts the needed parameters to tensors)
    inputs = pipeline.prepare_parameters(parameters)
    inputs["data"] = data[parameters.selected_transmits]

    n_pix = int(inputs["flatgrid"].shape[0])
    print(f"backend          : {backend}")
    print(f"device           : {device_kind}")
    print(f"data shape       : {tuple(data.shape)}  (n_tx, n_ax, n_el, n_ch)")
    print(f"grid             : {parameters.grid_size_z} x {parameters.grid_size_x} = {n_pix} px")
    print(f"num_patches      : {args.num_patches}")

    compile_s, times = time_pipeline(pipeline, inputs, args.warmup, args.repeats, backend)

    mean_s, std_s, min_s = float(np.mean(times)), float(np.std(times)), float(np.min(times))
    memory = report_memory(backend)
    print(f"compile + 1st run: {compile_s:8.3f} s")
    print(
        f"steady-state     : {mean_s * 1e3:8.2f} ms  ± {std_s * 1e3:.2f} ms  "
        f"(min {min_s * 1e3:.2f} ms)"
    )
    print(f"throughput       : {1.0 / mean_s:8.2f} frames/s")
    if memory["peak_bytes_in_use"]:
        print(f"peak device mem  : {memory['peak_bytes_in_use'] / 2**30:8.2f} GiB")

    if args.trace:
        if backend == "jax":
            with jax.profiler.trace(args.trace):
                for _ in range(3):
                    block(pipeline(**inputs), backend)
        else:
            import tensorflow as tf

            tf.profiler.experimental.start(args.trace)
            for _ in range(3):
                block(pipeline(**inputs), backend)
            tf.profiler.experimental.stop()
        print(f"trace written to : {args.trace}")

    outputs = block(pipeline(**inputs), backend)
    image = np.asarray(kops.convert_to_numpy(outputs[pipeline.output_key]))

    if args.save_output:
        np.savez(args.save_output, image=image, num_patches=args.num_patches)
        print(f"output saved to  : {args.save_output}")

    if args.check_against:
        ref = np.load(args.check_against)["image"]
        abs_err = np.abs(image - ref)
        # Values are log-compressed dB in [-inf, 0]; compare on the finite range.
        finite = np.isfinite(ref) & np.isfinite(image)
        max_err = float(abs_err[finite].max())
        print(f"max |err| vs ref : {max_err:.6f} dB (over {int(finite.sum())} finite px)")
        assert max_err < 1e-2, f"Output deviates from reference by {max_err} dB"
        print("output matches reference ✓")

    if args.per_op:
        print("\nper-op breakdown (jit_options='ops'; indicative, no cross-op fusion):")
        for name, mean in per_op_breakdown(
            config, inputs, args.warmup, args.repeats, backend
        ).items():
            print(f"  {name:20s} {mean * 1e3:8.2f} ms")

    if args.json_out:
        record = {
            "tag": args.tag,
            "backend": backend,
            "beamformer": args.beamformer,
            "pfield": args.pfield,
            "num_patches": args.num_patches,
            "grid_size": [int(parameters.grid_size_z), int(parameters.grid_size_x)],
            "compile_s": compile_s,
            "mean_ms": mean_s * 1e3,
            "std_ms": std_s * 1e3,
            "min_ms": min_s * 1e3,
            "fps": 1.0 / mean_s,
            "peak_bytes": memory["peak_bytes_in_use"],
            "n_pix": n_pix,
            "data_shape": list(data.shape),
        }
        with open(args.json_out, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"result appended  : {args.json_out}")


if __name__ == "__main__":
    main()
