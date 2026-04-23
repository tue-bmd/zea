"""Suppress harmless CUDA factory registration messages from ML framework imports.

When multiple GPU-enabled ML backends (JAX, TensorFlow) are co-installed, each
framework's CUDA plugin (jax[cuda12], tensorflow[and-cuda]) tries to register the
same XLA CUDA factories (cuFFT, cuDNN, cuBLAS) at shared-library load time. The
second registration attempt fails with a harmless error that is emitted to stderr
by ABSL's pre-initialisation logging path — before TF_CPP_MIN_LOG_LEVEL is read —
so that environment variable cannot suppress these messages.

This module is loaded by Python itself before any user code, which means the
stderr filter is already in place when the first ``import tensorflow`` /
``import jax`` triggers the CUDA shared-library load.
"""

import sys


class _CUDAMessageFilter:
    """Transparent stderr wrapper that drops known-harmless CUDA messages."""

    _FILTER_PATTERNS = (
        "Unable to register cuFFT factory",
        "Unable to register cuDNN factory",
        "Unable to register cuBLAS factory",
        "computation placer already registered",
        # ABSL pre-init preamble that accompanies the messages above
        "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR",
    )

    def __init__(self, stream):
        self._stream = stream

    def write(self, msg):
        # A single write() call may contain multiple lines; filter per-line so
        # that legitimate content in the same call is never silently dropped.
        filtered = "".join(
            line
            for line in msg.splitlines(keepends=True)
            if not any(pattern in line for pattern in self._FILTER_PATTERNS)
        )
        if filtered:
            return self._stream.write(filtered)
        # Return 0 to indicate no characters were forwarded to the stream.
        return 0

    def flush(self):
        self._stream.flush()

    @property
    def encoding(self):
        return self._stream.encoding

    @property
    def errors(self):
        return self._stream.errors

    def __getattr__(self, name):
        return getattr(self._stream, name)


sys.stderr = _CUDAMessageFilter(sys.stderr)
