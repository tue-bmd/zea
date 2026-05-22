"""Dev script for iterating on ABLE (Adaptive Beamforming by Deep Learning).

Replicates the notebook end-to-end but runs as a plain Python script:
  - Saves images to disk instead of displaying (works headless).
  - TARGET_BEAMFORMER toggle to switch training target easily.

Usage
-----
    python dev_able.py                        # fast DMAS run
    python dev_able.py --target mv            # MV target (slower)
    python dev_able.py --target dmas --epochs 50
"""

import argparse
import os
import time

# ── Keras backend must be set before importing keras / zea ─────────────────
os.environ["KERAS_BACKEND"] = "jax"
import keras
import matplotlib

import zea

matplotlib.use("Agg")  # headless backend — no display required

import matplotlib.pyplot as plt

from zea import init_device, load_file
from zea.models.able import ABLE
from zea.ops import (
    DelayAndSum,
    DelayMultiplyAndSum,
    EnvelopeDetect,
    Lambda,
    LogCompress,
    MinimumVariance,
    Normalize,
    PatchedGrid,
    Pipeline,
    ReshapeGrid,
    TOFCorrection,
)
from zea.visualize import set_mpl_style

# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ABLE dev script")
parser.add_argument(
    "--target",
    default="dmas",
    choices=["das", "dmas", "mv"],
    help="Training target beamformer (default: dmas)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=20,
    help="Number of training epochs (default: 20).",
)
parser.add_argument(
    "--n-tx",
    type=int,
    default=3,
    help="Number of transmits to use (default: 3 for speed).",
)
parser.add_argument(
    "--out-dir",
    default="dev_able_output",
    help="Directory to save output images (default: dev_able_output/).",
)
args = parser.parse_args()

zea.init_device()

TARGET_BEAMFORMER = args.target
N_TX = args.n_tx
OUT_DIR = args.out_dir
NUM_PATCHES = 40
N_EPOCHS = args.epochs

os.makedirs(OUT_DIR, exist_ok=True)

print(f"TARGET={TARGET_BEAMFORMER}  N_TX={N_TX}  NUM_PATCHES={NUM_PATCHES}  N_EPOCHS={N_EPOCHS}")

# ── Imports ────────────────────────────────────────────────────────────────


device = init_device(verbose=False)
set_mpl_style()

# ── 1. Load data ───────────────────────────────────────────────────────────
print("\n[1/6] Loading PICMUS data ...")
path = (
    "hf://zeahub/picmus/database/experiments/contrast_speckle/"
    "contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"
)
data, scan, probe = load_file(path=path, indices=[0], data_type="raw_data")

scan.set_transmits(N_TX)
data = data[:, scan.selected_transmits]

dynamic_range = (-50, 0)
scan.n_ch = data.shape[-1]
scan.zlims = (0, 0.06)
scan.xlims = (-0.019, 0.019)

print(f"  data shape : {data.shape}  (frames, tx, n_ax, n_el, n_ch)")
print(f"  probe      : {probe.__class__.__name__}, {probe.n_el} elements")

# ── 2. Build pipelines ─────────────────────────────────────────────────────
print("\n[2/6] Building pipelines ...")
n_el = data.shape[-2]
n_ch = data.shape[-1]
n_tx = data.shape[1]

able_model = ABLE()
able_model.build((n_tx, 1, n_el, n_ch))
print(f"  ABLE trainable variables: {len(able_model.trainable_variables)}")


def apply_able(x):
    return able_model(x)


# ABLE pipeline — gradient-flow note:
# After construction the outer Pipeline propagates jit_options="ops" to the
# inner PatchedGrid (Map subclass), which JIT-compiles jittable_call and bakes
# the model weights in as constants.  Resetting to None after construction
# calls Map.unjit() so value_and_grad can trace through the weights.
able_pipeline = Pipeline(
    operations=[
        PatchedGrid(
            operations=[
                TOFCorrection(),
                Lambda(apply_able, name="ABLE Reconstruction", jit_compile=False),
                DelayAndSum(),
            ],
            num_patches=NUM_PATCHES,
        ),
        ReshapeGrid(),
        EnvelopeDetect(),
        Normalize(),
        LogCompress(),
    ],
    with_batch_dim=True,
)
able_pipeline.operations[0].jit_options = None  # gradient-flow fix (see above)

das_pipeline = Pipeline(
    operations=[
        PatchedGrid(
            operations=[TOFCorrection(), DelayAndSum()],
            num_patches=NUM_PATCHES,
        ),
        ReshapeGrid(),
        EnvelopeDetect(),
        Normalize(),
        LogCompress(),
    ],
    with_batch_dim=True,
)

dmas_pipeline = Pipeline(
    operations=[
        PatchedGrid(
            operations=[TOFCorrection(), DelayMultiplyAndSum()],
            num_patches=NUM_PATCHES,
        ),
        ReshapeGrid(),
        EnvelopeDetect(),
        Normalize(),
        LogCompress(),
    ],
    with_batch_dim=True,
)

# MV: subarray_size=n_el//4 gives ~8× speedup (O(M³) eigendecomp, M=32 vs 64).
mv_pipeline = Pipeline(
    operations=[
        PatchedGrid(
            operations=[
                TOFCorrection(),
                MinimumVariance(subarray_size=n_el // 4, diagonal_loading=1e-2),
            ],
            num_patches=NUM_PATCHES,
        ),
        ReshapeGrid(),
        EnvelopeDetect(),
        Normalize(),
        LogCompress(),
    ],
    with_batch_dim=True,
)

parameters = able_pipeline.prepare_parameters(probe, scan, dynamic_range=dynamic_range)
inputs_tensor = keras.ops.convert_to_tensor(data)

# ── 3. Initial b-mode ──────────────────────────────────────────────────────
print("\n[3/6] Computing initial ABLE b-mode (before training) ...")
t0 = time.perf_counter()
out_before = able_pipeline(**{able_pipeline.key: inputs_tensor}, **parameters)
bmode_before = out_before[able_pipeline.output_key]
print(f"  done in {time.perf_counter() - t0:.1f}s")

img_path = os.path.join(OUT_DIR, "bmode_before.png")
zea.display.to_8bit(bmode_before[0], dynamic_range=dynamic_range).save(img_path)
print(f"  saved → {img_path}")

# ── 4. Compute training target ─────────────────────────────────────────────
print(f"\n[4/6] Computing {TARGET_BEAMFORMER.upper()} training target ...")
target_pipeline = {"das": das_pipeline, "dmas": dmas_pipeline, "mv": mv_pipeline}[TARGET_BEAMFORMER]
t0 = time.perf_counter()
target_out = target_pipeline(**{target_pipeline.key: inputs_tensor}, **parameters)
target_bmode = target_out[target_pipeline.output_key]
print(f"  done in {time.perf_counter() - t0:.1f}s  shape={target_bmode.shape}")

# ── 5. Train ───────────────────────────────────────────────────────────────
print(f"\n[5/6] Training ABLE ({N_EPOCHS} epochs) ...")


class ABLEPipelineModel(keras.Model):
    def __init__(self, model, pipeline, params):
        super().__init__()
        self.able = model  # register weights with Keras
        self._pipeline = pipeline
        self._params = params

    def call(self, x):
        out = self._pipeline(**{self._pipeline.key: x}, **self._params)
        return out[self._pipeline.output_key]


trainable_model = ABLEPipelineModel(able_model, able_pipeline, parameters)
trainable_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.MeanSquaredError(),
)

losses = []
t_train = time.perf_counter()
for epoch in range(N_EPOCHS):
    loss = trainable_model.train_on_batch(inputs_tensor, target_bmode)
    losses.append(float(loss))
    if (epoch + 1) % max(1, N_EPOCHS // 10) == 0 or epoch == 0:
        elapsed = time.perf_counter() - t_train
        print(f"  epoch {epoch + 1:03d}/{N_EPOCHS}  loss={float(loss):.6f}  ({elapsed:.0f}s)")

print(f"  training done in {time.perf_counter() - t_train:.1f}s")
print(f"  loss: {losses[0]:.6f} → {losses[-1]:.6f}")

# Loss curve
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE loss")
ax.set_title(f"ABLE training loss ({TARGET_BEAMFORMER.upper()} target)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
loss_path = os.path.join(OUT_DIR, "loss_curve.png")
plt.savefig(loss_path, dpi=150)
plt.close(fig)
print(f"  saved → {loss_path}")

# ── 6. Save results ────────────────────────────────────────────────────────
print("\n[6/6] Saving comparison images ...")

out_after = able_pipeline(**{able_pipeline.key: inputs_tensor}, **parameters)
bmode_after = out_after[able_pipeline.output_key]

das_out = das_pipeline(**{das_pipeline.key: inputs_tensor}, **parameters)
bmode_das = das_out[das_pipeline.output_key]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
panels = [
    (bmode_before, "ABLE (before)"),
    (bmode_das, "DAS"),
    (target_bmode, f"{TARGET_BEAMFORMER.upper()} (target)"),
    (bmode_after, "ABLE (after)"),
]
for ax, (bmode, title) in zip(axes, panels):
    ax.imshow(
        keras.ops.convert_to_numpy(bmode[0]),
        cmap="gray",
        vmin=dynamic_range[0],
        vmax=dynamic_range[1],
        aspect="auto",
    )
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
comp_path = os.path.join(OUT_DIR, "comparison.png")
plt.savefig(comp_path, dpi=150)
plt.close(fig)
print(f"  saved → {comp_path}")

after_path = os.path.join(OUT_DIR, "bmode_after.png")
zea.display.to_8bit(bmode_after[0], dynamic_range=dynamic_range).save(after_path)
print(f"  saved → {after_path}")

weights_path = os.path.join(OUT_DIR, "able_model.weights.h5")
able_model.save_weights(weights_path)
print(f"  saved → {weights_path}")

print("\nDone. Outputs written to:", OUT_DIR)
