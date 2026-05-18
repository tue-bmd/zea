"""test file to be converted to notebook"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
from IPython.display import display

import zea
from zea import init_device, load_file
from zea.models.able import ABLE
from zea.ops import (
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
from zea.visualize import set_mpl_style

device = init_device(verbose=False)
set_mpl_style()


# Load a single frame of raw RF data from PICMUS
path = "hf://zeahub/picmus/database/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"
data, scan, probe = load_file(
    path=path,
    indices=[0],
    data_type="raw_data",
)
scan.set_transmits(1)
data = data[:, scan.selected_transmits]
zlims = (0, 0.06)
xlims = (-0.019, 0.019)
dynamic_range = (-50, 0)

scan.n_ch = data.shape[-1]  # iq data
scan.zlims = zlims
scan.xlims = xlims

# Instantiate ABLE so we can access its weights for training
able_model = ABLE()


def apply_able(x):
    """Call able_model via __call__ so Keras triggers build() on first run."""
    return able_model(x)


# Build the beamforming pipeline with ABLE
pipeline = Pipeline(
    operations=[
        PatchedGrid(
            operations=[
                TOFCorrection(),
                Lambda(apply_able, name="ABLE Reconstruction", jit_compile=False),
                DelayAndSum(),
            ],
            num_patches=10,
            jit_options=None,
        ),
        ReshapeGrid(),
        EnvelopeDetect(),
        Normalize(),
        LogCompress(),
    ],
    with_batch_dim=True,
    jit_options=None#"pipeline",
)

target_pipeline = Pipeline(
    operations=[
        PatchedGrid(
            operations=[
                TOFCorrection(),
                DelayAndSum(),
            ],
            num_patches=10,
            jit_options=None,
        ),
        ReshapeGrid(),
        EnvelopeDetect(),
        Normalize(),
        LogCompress(),
    ],
    with_batch_dim=True,
    jit_options=None#"pipeline",
)


parameters = pipeline.prepare_parameters(probe, scan, dynamic_range=dynamic_range)
parameters["demodulation_frequency"] = parameters["sampling_frequency"]

inputs_tensor = keras.ops.convert_to_tensor(data)
inputs = {pipeline.key: inputs_tensor}

# First forward pass — builds ABLE and its trainable variables
outputs = pipeline(**inputs, **parameters)
bmode = outputs[pipeline.output_key]
print(f"ABLE trainable variables: {len(able_model.trainable_variables)}")

bmode_img = zea.display.to_8bit(bmode[0], dynamic_range=dynamic_range)
display(bmode_img)
bmode_img.save("able_bmode_before_training.png")

# --- Compute DAS targets (no ABLE, plain delay-and-sum) ---
target_outputs = target_pipeline(
    **{target_pipeline.key: inputs_tensor}, **parameters
)
target_bmode = target_outputs[target_pipeline.output_key]

# --- Manual training loop (no model.fit) ---
# Keras 3 + JAX defers gradient composition to the backend internally.
# `train_on_batch` runs exactly one gradient step per call, giving explicit
# per-step control without the epoch/dataset bookkeeping of model.fit.
import jax
import jax.numpy as jnp


class ABLEPipelineModel(keras.Model):
    """Thin keras.Model wrapping the full ABLE pipeline."""

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
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.MeanSquaredError(),
)

for epoch in range(20):
    loss = trainable_model.train_on_batch(inputs_tensor, target_bmode)
    print(f"Epoch {epoch + 1:02d}: loss={float(loss):.6f}")

# --- Visualise after training ---
outputs_trained = pipeline(**inputs, **parameters)
bmode_trained = outputs_trained[pipeline.output_key]
bmode_img_trained = zea.display.to_8bit(bmode_trained[0], dynamic_range=dynamic_range)
display(bmode_img_trained)
bmode_img_trained.save("able_bmode_after_training.png")
