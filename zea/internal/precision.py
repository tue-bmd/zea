"""Mixed-precision helpers for zea pipelines.

zea beamforming keeps geometry and time-of-flight *delay* computation in
``float32`` — those quantities need the full dynamic range (sample indices run
into the thousands, and interpolation weights derived from them must be
accurate) — while allowing the bulk *signal* compute (the TOF gather,
interpolation, apodization and delay-and-sum) to run in a lower-precision
"compute dtype" for speed.

The compute dtype is resolved from the active Keras mixed-precision policy, so::

    import keras

    keras.mixed_precision.set_global_policy("mixed_bfloat16")

is all that is needed to make the beamformer run in mixed precision. Reset with
``keras.mixed_precision.set_global_policy("float32")``.

The helpers here are intentionally tiny and side-effect free so they can be
called inside JIT-traced pipeline operations: the dtype is resolved once at
trace time.
"""

import keras
from keras import ops

#: Low-precision floating point dtypes that trigger mixed-precision beamforming.
LOW_PRECISION_DTYPES = ("bfloat16", "float16")


def signal_compute_dtype() -> str:
    """Return the dtype used for the bulk signal compute in the beamformer.

    Resolved from the active Keras mixed-precision policy:

    * ``"bfloat16"`` / ``"float16"`` when a mixed-precision policy is active
      (e.g. after ``keras.mixed_precision.set_global_policy("mixed_bfloat16")``),
    * otherwise :func:`keras.config.floatx` (``"float32"`` by default).

    Geometry and delay computations should *not* use this dtype; they stay in
    ``float32`` regardless of the policy.

    Returns:
        str: The compute dtype name, one of ``"bfloat16"``, ``"float16"`` or
        ``"float32"``.
    """
    try:
        compute_dtype = keras.mixed_precision.dtype_policy().compute_dtype
    except Exception:  # pragma: no cover - extremely defensive
        return keras.config.floatx()

    if compute_dtype in LOW_PRECISION_DTYPES:
        return compute_dtype
    return keras.config.floatx()


def is_mixed_precision() -> bool:
    """Whether a low-precision (``bfloat16``/``float16``) compute policy is active."""
    return signal_compute_dtype() in LOW_PRECISION_DTYPES


def cast_signal(x):
    """Cast a bulk-signal tensor to :func:`signal_compute_dtype`.

    Use this on the RF/IQ signal (and on any weight that multiplies it, so the
    product stays in the low-precision dtype instead of being up-cast by type
    promotion). Do **not** use it on delays, grids or other geometry tensors.
    """
    return ops.cast(x, signal_compute_dtype())
