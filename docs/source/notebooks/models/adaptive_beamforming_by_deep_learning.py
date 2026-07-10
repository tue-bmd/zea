"""Adaptive Beamforming by Deep LEarning (ABLE) — training script.

Trains the ABLE network to mimic a reference beamformer chosen at the command
line.  Pass ``--target mvb`` to train against Minimum Variance Beamforming,
``--target cf`` for Coherence Factor, etc.  Use ``--help`` for the full list.
"""

import argparse
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import matplotlib.pyplot as plt
import numpy as np

import zea
from zea import init_device, log
from zea.models.able import ABLE
from zea.ops import (
    Beamform,
    DelayAndSum,
    EnvelopeDetect,
    Lambda,
    LogCompress,
    Normalize,
    PatchedGrid,
    Pipeline,
    ReshapeGrid,
    TOFCorrection,
)
from zea.utils import FunctionTimer
from zea.visualize import set_mpl_style

# ── CLI ───────────────────────────────────────────────────────────────────────

# Short aliases so users don't have to remember the full registered name.
_BEAMFORMER_ALIASES: dict[str, str] = {
    "das": "delay_and_sum",
    "dmas": "delay_multiply_and_sum",
    "cf": "coherence_factor",
    "gcf": "generalized_coherence_factor",
    "mvb": "minimum_variance",
    "mv": "minimum_variance",
}

parser = argparse.ArgumentParser(
    description="Train ABLE against a configurable reference beamformer.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--target",
    default="das",
    metavar="BEAMFORMER",
    help=(
        "Reference beamformer used as training target.  "
        "Short aliases: das, dmas, cf, gcf, mvb.  "
        "Or pass a full registered name: delay_and_sum, minimum_variance, …"
    ),
)
parser.add_argument(
    "--epochs",
    type=int,
    default=300,
    help="Maximum number of training epochs.",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="Initial Adam learning rate.",
)
parser.add_argument(
    "--patience",
    type=int,
    default=25,
    help="Early-stopping patience in epochs (no improvement → stop).",
)
parser.add_argument(
    "--lr-patience",
    type=int,
    default=12,
    dest="lr_patience",
    help="Epochs with no improvement before halving the learning rate.",
)
parser.add_argument(
    "--lr-min",
    type=float,
    default=1e-5,
    dest="lr_min",
    help="Lower bound on the learning rate.",
)
parser.add_argument(
    "--num-patches",
    type=int,
    default=10,
    dest="num_patches",
    help="Number of spatial patches for PatchedGrid.",
)
parser.add_argument(
    "--n-repeats",
    type=int,
    default=3,
    dest="n_repeats",
    help="Number of timed runs per beamformer (excluding the warmup call).",
)
parser.add_argument(
    "--histogram-match",
    action="store_true",
    dest="histogram_match",
    help=(
        "Histogram-match the target and ABLE images to the DAS image before display, "
        "so brightness differences due to side-lobe suppression do not skew visual comparison."
    ),
)
# parse_known_args so notebooks can pass extra flags (e.g. IPython kernel args)
args, _unknown = parser.parse_known_args()

target_beamformer = _BEAMFORMER_ALIASES.get(args.target, args.target)

# ── Setup ─────────────────────────────────────────────────────────────────────

device = init_device()
set_mpl_style()

# ── Data ──────────────────────────────────────────────────────────────────────

path = (
    "hf://zeahub/picmus/database/experiments/contrast_speckle/"
    "contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"
)

with zea.File(path) as f:
    data = f.data.raw_data[0][None, ...]  # add batch dim → (1, n_tx, n_ax, n_el, n_ch)
    parameters = f.load_parameters()

parameters.set_transmits(1)
data = data[:, parameters.selected_transmits]

zlims = (0, 0.06)
xlims = (-0.019, 0.019)

parameters.zlims = zlims
parameters.xlims = xlims

# ── ABLE model ────────────────────────────────────────────────────────────────

able_model = ABLE()

# Pre-build ABLE in pure eager mode before any pipeline call.
# PatchedGrid (with with_batch_dim=True) maps the batch axis with ops.map
# (= lax.scan).  If ABLE.build() fires inside that scan the Conv2D weight
# tensors are created as DynamicJaxprTracers that escape the scan scope and
# cause an UnexpectedTracerError on the next forward pass.  Calling
# able_model here with a concrete numpy array guarantees the weights are real
# JAX arrays before any traced context is ever entered.
_n_el = data.shape[-2]
_n_ch = data.shape[-1]
able_model(np.zeros((1, 1, _n_el, _n_ch), dtype=np.float32))


def apply_able(x):
    return able_model(x)


# ── Pipelines ─────────────────────────────────────────────────────────────────

NUM_PATCHES = args.num_patches

# ABLE pipeline: TOF-corrects and applies per-element adaptive weights before DAS.
pipeline = Pipeline(
    operations=[
        PatchedGrid(
            operations=[
                TOFCorrection(),
                Lambda(apply_able, name="ABLE", jit_compile=False),
                DelayAndSum(),
            ],
            num_patches=NUM_PATCHES,
            jit_options=None,
        ),
        ReshapeGrid(),
        EnvelopeDetect(),
        Normalize(),
        LogCompress(),
    ],
    with_batch_dim=True,
    jit_options=None,
)

# Reference pipeline: plain beamformer used as the learning target.
# Beamform already wraps PatchedGrid(TOFCorrection + beamformer) + ReshapeGrid.
target_pipeline = Pipeline(
    operations=[
        Beamform(beamformer=target_beamformer, num_patches=NUM_PATCHES, jit_options=None),
        EnvelopeDetect(),
        Normalize(),
        LogCompress(),
    ],
    with_batch_dim=True,
    jit_options=None,
)

# DAS reference — always computed for the comparison figure and auto dynamic range.
das_pipeline = Pipeline(
    operations=[
        Beamform(beamformer="delay_and_sum", num_patches=NUM_PATCHES, jit_options=None),
        EnvelopeDetect(),
        Normalize(),
        LogCompress(),
    ],
    with_batch_dim=True,
    jit_options=None,
)

parameters = pipeline.prepare_parameters(parameters)
parameters["demodulation_frequency"] = parameters["sampling_frequency"]

inputs_tensor = keras.ops.convert_to_tensor(data)
inputs = {pipeline.key: inputs_tensor}

# ── Display helpers ────────────────────────────────────────────────────────────

FIXED_DR: tuple[float, float] = (-60.0, 0.0)


def histogram_match(source: np.ndarray, reference: np.ndarray, bins: int = 1024) -> np.ndarray:
    """Remap source pixel values so its histogram matches reference (CDF matching).

    Both images are assumed to be log-compressed and in the range given by
    FIXED_DR.  The output has the same shape as source and can be displayed
    with vmin/vmax = FIXED_DR for a visually calibrated comparison.
    """
    src = source.ravel()
    ref = reference.ravel()
    src_hist, edges = np.histogram(src, bins=bins, range=FIXED_DR)
    ref_hist, _ = np.histogram(ref, bins=bins, range=FIXED_DR)
    src_cdf = np.cumsum(src_hist).astype(float)
    src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(ref_hist).astype(float)
    ref_cdf /= ref_cdf[-1]
    centers = (edges[:-1] + edges[1:]) * 0.5
    matched = np.interp(np.interp(src, centers, src_cdf), ref_cdf, centers)
    return matched.reshape(source.shape)


# ── DAS reference (always computed) ───────────────────────────────────────────

print("Computing 'delay_and_sum' reference …")
das_outputs = das_pipeline(**{das_pipeline.key: inputs_tensor}, **parameters)
das_bmode = das_outputs[das_pipeline.output_key]

_das_np = np.array(das_bmode[0])
_das_vmin = float(_das_np.min())

das_img = zea.display.to_8bit(_das_np, dynamic_range=(_das_vmin, FIXED_DR[1]))
_das_img_path = "able_das_reference.png"
das_img.save(_das_img_path)
log.info(f"Saved DAS reference  → {log.yellow(_das_img_path)}")

# ── Target beamformer reference ────────────────────────────────────────────────

print(f"Computing '{target_beamformer}' reference target …")
target_outputs = target_pipeline(**{target_pipeline.key: inputs_tensor}, **parameters)
target_bmode = target_outputs[target_pipeline.output_key]

_target_np = np.array(target_bmode[0])
if args.histogram_match:
    _target_np = histogram_match(_target_np, _das_np)
target_img = zea.display.to_8bit(_target_np, dynamic_range=(float(_target_np.min()), FIXED_DR[1]))
_target_img_path = f"able_target_{target_beamformer}.png"
target_img.save(_target_img_path)
log.info(f"Saved target image   → {log.yellow(_target_img_path)}")

# ── Pre-training ABLE forward pass ────────────────────────────────────────────

outputs = pipeline(**inputs, **parameters)
bmode = outputs[pipeline.output_key]
print(f"ABLE trainable variables: {len(able_model.trainable_variables)}")

# ── Keras Model wrapper ───────────────────────────────────────────────────────


class ABLEPipelineModel(keras.Model):
    """Thin keras.Model wrapping the full ABLE pipeline for gradient-based training."""

    def __init__(self, able, pipeline, pipeline_params):
        super().__init__()
        self.able = able  # exposes ABLE weights as trainable_variables
        self._pipeline = pipeline
        self._params = pipeline_params

    def call(self, x):
        out = self._pipeline(**{self._pipeline.key: x}, **self._params)
        return out[self._pipeline.output_key]


trainable_model = ABLEPipelineModel(able_model, pipeline, parameters)
trainable_model(inputs_tensor)  # register variables on trainable_model
print(f"Trainable variables: {len(trainable_model.trainable_variables)}")

trainable_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=args.lr),
    loss=keras.losses.MeanSquaredError(),
)

# ── Training loop ─────────────────────────────────────────────────────────────

best_loss = float("inf")
best_weights = [keras.ops.convert_to_numpy(v) for v in able_model.trainable_variables]
patience_counter = 0
lr_counter = 0

print(f"\nTraining ABLE → '{target_beamformer}' target")
print(
    f"  max_epochs={args.epochs}, patience={args.patience}, "
    f"lr={args.lr:.1e}, lr_patience={args.lr_patience}\n"
)

for epoch in range(1, args.epochs + 1):
    loss_val = float(trainable_model.train_on_batch(inputs_tensor, target_bmode))

    improved = loss_val < best_loss - 1e-7
    if improved:
        best_loss = loss_val
        best_weights = [keras.ops.convert_to_numpy(v) for v in able_model.trainable_variables]
        patience_counter = 0
        lr_counter = 0
        marker = " *"
    else:
        patience_counter += 1
        lr_counter += 1
        marker = ""

    # Reduce LR when plateau is detected.
    if lr_counter >= args.lr_patience:
        current_lr = float(trainable_model.optimizer.learning_rate)
        new_lr = max(current_lr * 0.5, args.lr_min)
        trainable_model.optimizer.learning_rate.assign(new_lr)
        lr_counter = 0
        print(f"  [LR] {current_lr:.2e} → {new_lr:.2e}")

    print(f"Epoch {epoch:03d}/{args.epochs}: loss={loss_val:.6f}  best={best_loss:.6f}{marker}")

    if patience_counter >= args.patience:
        print(f"\nEarly stopping: no improvement for {args.patience} consecutive epochs.")
        break

# Restore best weights found during training.
for var, val in zip(able_model.trainable_variables, best_weights):
    var.assign(val)
print(f"\nRestored best weights  (loss = {best_loss:.6f})")

# ── Speed benchmarking ────────────────────────────────────────────────────────

N_REPEATS = args.n_repeats
timer = FunctionTimer()

_DAS_KEY = "DAS (delay_and_sum)"
_TARGET_KEY = f"Target ({target_beamformer})"
_ABLE_KEY = "ABLE (post-training)"


def _run_das():
    return das_pipeline(**{das_pipeline.key: inputs_tensor}, **parameters)


def _run_target():
    return target_pipeline(**{target_pipeline.key: inputs_tensor}, **parameters)


def _run_able():
    return pipeline(**inputs, **parameters)


timed_das = timer(_run_das, name=_DAS_KEY)
timed_target = timer(_run_target, name=_TARGET_KEY)
timed_able = timer(_run_able, name=_ABLE_KEY)

print(f"\nBenchmarking beamformers  ({N_REPEATS} timed runs + 1 warmup each) …")
for fn in (timed_das, timed_target, timed_able):
    for _ in range(1 + N_REPEATS):  # first call is warmup
        fn()

# drop_first=True drops the warmup call from all statistics.
timer.print(drop_first=True)

das_stats = timer.get_stats(_DAS_KEY, drop_first=True)
target_stats = timer.get_stats(_TARGET_KEY, drop_first=True)
able_stats = timer.get_stats(_ABLE_KEY, drop_first=True)

# ── Post-training ABLE image ──────────────────────────────────────────────────

outputs_trained = pipeline(**inputs, **parameters)
bmode_trained = outputs_trained[pipeline.output_key]

# ── Comparison figure: DAS | Target | ABLE ────────────────────────────────────

# Image extent in mm for axis labels (left, right, bottom, top).
_extent_mm = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]


def _display_img(img: np.ndarray, reference: np.ndarray | None) -> tuple[np.ndarray, float]:
    """Return (display_image, vmin); optionally histogram-match image to reference."""
    if reference is not None:
        img = histogram_match(img, reference)
    return img, float(img.min())


_ref = _das_np if args.histogram_match else None
_target_display, _target_vmin = _display_img(np.array(target_bmode[0]), _ref)
_able_display, _able_vmin = _display_img(np.array(bmode_trained[0]), _ref)

_panels = [
    ("DAS", _das_np, _das_vmin, das_stats),
    (target_beamformer.replace("_", " ").title(), _target_display, _target_vmin, target_stats),
    ("ABLE", _able_display, _able_vmin, able_stats),
]

_match_note = " (histogram matched to DAS)" if args.histogram_match else ""
fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
fig.suptitle(
    f"ABLE trained on '{target_beamformer.replace('_', ' ')}' target",
    fontsize=11,
)

for ax, (title, img_np, vmin, stats) in zip(axes, _panels):
    ax.imshow(
        img_np,
        cmap="gray",
        vmin=vmin,
        vmax=FIXED_DR[1],
        extent=_extent_mm,
        aspect="auto",
        interpolation="bilinear",
        origin="upper",
    )
    timing_ms = f"{stats['mean'] * 1e3:.0f} ± {stats['std_dev'] * 1e3:.0f} ms"
    suffix = "\n(hist. matched to DAS)" if (args.histogram_match and title != "DAS") else ""
    ax.set_title(f"{title}\n{timing_ms}{suffix}", fontsize=9)
    ax.set_xlabel("Lateral [mm]")
    if ax is axes[0]:
        ax.set_ylabel("Depth [mm]")

_fig_path = f"able_comparison_{target_beamformer}.png"
fig.savefig(_fig_path, dpi=150, bbox_inches="tight")
log.info(f"Saved comparison figure → {log.yellow(_fig_path)}")
plt.close(fig)
