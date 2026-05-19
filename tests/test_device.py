"""Tests for device selection, GPU memory, and device placement across backends.

Tests marked with ``@pytest.mark.gpu`` require a real GPU.  The CI environment
sets ``CUDA_VISIBLE_DEVICES=""`` (CPU-only), so those tests are skipped there
and should be run locally by:

    pytest -m gpu tests/test_device.py

"""

import os
from itertools import product
from unittest.mock import patch

import keras
import numpy as np
import pytest
from keras.ops import convert_to_numpy

import zea
from zea.backend import func_on_device
from zea.internal.device import (
    _cuda_visible_devices_disables_gpus,
    get_gpu_memory,
    init_device,
)
from zea.ops import Pipeline
from zea.ops.keras_ops import Abs

from . import DEFAULT_TEST_SEED, backend_equality_check

_DEVICES = ["cpu", "gpu:0", "cuda:0", "auto:-1", "auto:1"]
_BACKENDS = ["tensorflow", "torch", "jax", "auto", "numpy"]


def _tensor_device_name(tensor) -> str:
    """Return a lowercase device string for a tensor (e.g. ``'cpu'``, ``'cuda:0'``)."""
    backend = keras.backend.backend()
    if backend == "torch":
        import torch

        if isinstance(tensor, torch.Tensor):
            return str(tensor.device)
    if backend == "jax":
        import jax

        return str(jax.device_put(tensor).devices().pop()).lower()
    if backend == "tensorflow":
        return tensor.device.lower()
    return "unknown"


class TestCudaVisibleDevicesDisablesGpus:
    """Unit tests for the ``_cuda_visible_devices_disables_gpus`` helper."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, False),  # unset → all GPUs visible
            ("", True),  # empty string → disabled
            ("-1", True),  # single negative
            ("-1,-2", True),  # all negative
            ("0", False),  # valid GPU
            ("0,-1", False),  # mixed: at least one valid
            (" -1 ", True),  # whitespace around negative
            ("GPU-abc123,GPU-def456", False),  # UUID tokens → not integer, return False
        ],
    )
    def test_various_values(self, monkeypatch, value, expected):
        if value is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
        assert _cuda_visible_devices_disables_gpus() is expected


_SMI_TWO_GPUS = b"1000\n2000\n"


def _mock_smi(monkeypatch, raw_output):
    """Patch ``check_nvidia_smi`` and ``subprocess.check_output`` for unit tests."""
    monkeypatch.setattr("zea.internal.device.check_nvidia_smi", lambda: True)
    return patch("subprocess.check_output", return_value=raw_output)


class TestGetGpuMemory:
    """Tests for ``get_gpu_memory``: env-var gating and nvidia-smi output parsing."""

    @pytest.mark.parametrize("value", ["-1", ""])
    def test_returns_empty_when_gpus_disabled(self, monkeypatch, value):
        """Returns ``[]`` when ``CUDA_VISIBLE_DEVICES`` disables all GPUs."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
        assert get_gpu_memory(verbose=False) == []

    def test_parses_smi_output(self, monkeypatch):
        """Correctly parses multi-line nvidia-smi output."""
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        with _mock_smi(monkeypatch, _SMI_TWO_GPUS):
            assert get_gpu_memory(verbose=False) == [1000, 2000]

    def test_out_of_range_ids_filtered(self, monkeypatch):
        """GPU IDs beyond the detected count are silently removed."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,5")
        with _mock_smi(monkeypatch, _SMI_TWO_GPUS):
            assert get_gpu_memory(verbose=False) == [1000]

    def test_negative_ids_filtered_from_valid(self, monkeypatch):
        """Negative IDs mixed with valid ones are filtered; valid IDs are kept."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,-1")
        with _mock_smi(monkeypatch, _SMI_TWO_GPUS):
            assert get_gpu_memory(verbose=False) == [1000]


class TestInitDevice:
    """Tests for ``init_device``."""

    @pytest.mark.gpu
    @pytest.mark.parametrize("device, backend", list(product(_DEVICES, _BACKENDS)))
    def test_all_device_backend_combinations(self, device, backend):  # pragma: no cover
        """Smoke-test every (device, backend) combination.

        In CI all GPU strings fall back to CPU; run locally with a GPU for
        full coverage.
        """
        init_device(device=device, backend=backend, verbose=False)

    @pytest.mark.gpu
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_default_device_per_backend(self, backend):  # pragma: no cover
        """Smoke-test default device selection (no ``device=`` argument) per backend."""
        init_device(backend=backend, verbose=False)

    @pytest.mark.gpu
    @pytest.mark.parametrize("backend", ["tensorflow", "torch", "jax"])
    def test_multi_gpu_returns_list(self, monkeypatch, backend):  # pragma: no cover
        """``init_device('auto:2')`` returns a list of two device strings."""
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        if len(get_gpu_memory(verbose=False)) < 2:
            pytest.skip("Requires at least 2 GPUs")
        devices = init_device(device="auto:2", backend=backend, verbose=False)
        assert isinstance(devices, list), f"Expected list, got {type(devices)}"
        assert len(devices) == 2
        key = "cuda" if backend == "torch" else "gpu"
        assert devices == [f"{key}:0", f"{key}:1"]

    @pytest.mark.gpu
    @pytest.mark.parametrize("backend", ["tensorflow", "torch", "jax"])
    def test_multi_gpu_selects_correct_physical_gpus(
        self, monkeypatch, backend
    ):  # pragma: no cover
        """``init_device('auto:2')`` must select the 2 physical GPUs with the
        most free memory, not necessarily physical GPU 0 and 1.

        After the call, ``CUDA_VISIBLE_DEVICES`` must contain exactly those
        physical IDs.  This prevents the renumbering trap where ``gpu:0``
        inside the process silently refers to physical GPU 0 instead of the
        one that was actually chosen.
        """
        # Clear CUDA_VISIBLE_DEVICES so get_gpu_memory reports all physical
        # GPUs (monkeypatch restores it automatically after the test).
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        all_memories = get_gpu_memory(verbose=False)
        if len(all_memories) < 2:
            pytest.skip("Requires at least 2 physical GPUs")

        # Physical IDs of the top-2 GPUs by free memory
        sorted_ids = sorted(range(len(all_memories)), key=lambda i: all_memories[i], reverse=True)
        expected_physical = sorted(sorted_ids[:2])

        init_device(device="auto:2", backend=backend, verbose=False)

        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        actual_physical = sorted(int(x) for x in cuda_visible.split(",") if x.strip())

        assert actual_physical == expected_physical, (
            f"Expected CUDA_VISIBLE_DEVICES to map to physical GPUs "
            f"{expected_physical}, but got {cuda_visible!r}"
        )

        # Extra guard: physical GPU 0 must NOT be selected unless it is
        # genuinely one of the top-2 by free memory.
        if 0 not in expected_physical:
            assert 0 not in actual_physical, (
                "Physical GPU 0 was selected but is not in the top-2 by free memory. "
                "The renumbering after hide_gpus is likely broken."
            )

    @pytest.mark.parametrize("backend", ["tensorflow", "torch", "jax"])
    def test_falls_back_to_cpu_when_gpus_disabled(self, monkeypatch, backend):
        """Returns ``'cpu'`` when ``CUDA_VISIBLE_DEVICES`` disables all GPUs."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
        assert init_device(device="auto:1", backend=backend, verbose=False) == "cpu"


class TestOnDevice:
    """Tests for the ``device`` context manager and ``func_on_device``."""

    def test_normalize(self):
        """``_normalize`` translates device strings to backend conventions."""
        if keras.backend.backend() == "torch":
            assert zea.device._normalize("gpu:0") == "cuda:0"
        else:
            assert zea.device._normalize("cuda:0") == "gpu:0"
        assert zea.device._normalize("cpu") == "cpu"

    def test_none_is_noop(self):
        """``zea.device(None)`` is a no-op context manager."""
        with zea.device(None):
            assert keras.ops.ones((3,)).shape == (3,)

    def test_zea_namespace_export(self):
        """``device`` is re-exported at ``zea.device``."""
        import zea

        with zea.device("cpu"):
            assert keras.ops.ones((2, 2)).shape == (2, 2)

    @backend_equality_check()
    def test_cpu_all_backends(self):
        """``zea.device('cpu')`` produces consistent results across all backends."""
        import keras

        import zea

        with zea.device("cpu"):
            return keras.ops.abs(
                keras.ops.convert_to_tensor(np.array([-2.0, 3.0], dtype=np.float32))
            )

    @backend_equality_check()
    def test_func_on_device_all_backends(self):
        """``func_on_device`` produces consistent results across all backends."""
        import numpy as np

        rng = np.random.default_rng(DEFAULT_TEST_SEED)
        x = rng.standard_normal((3, 3)).astype(np.float32)
        y = rng.standard_normal((3, 3)).astype(np.float32)
        return func_on_device(lambda a, b: a + b, "cpu", x, y)

    @pytest.mark.gpu
    def test_gpu_tensor_placement(self):  # pragma: no cover
        """Tensors created inside ``device('gpu:0')`` reside on the GPU."""
        with zea.device("gpu:0"):
            x = keras.ops.ones((4,))
        assert "cpu" not in _tensor_device_name(x)

    @pytest.mark.gpu
    def test_gpu_correct_result(self):  # pragma: no cover
        """``zea.device('gpu:0')`` produces numerically correct results."""
        with zea.device("gpu:0"):
            result = keras.ops.abs(
                keras.ops.convert_to_tensor(np.array([-1.0, 2.0, -3.0], dtype=np.float32))
            )
        np.testing.assert_allclose(
            convert_to_numpy(result), np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )


class TestPipelineDevice:
    """Tests for the ``device`` parameter on ``Pipeline``."""

    @staticmethod
    def _pipe(**kwargs):
        return Pipeline([Abs()], jit_options=None, **kwargs)

    def test_default_device_is_none(self):
        """``Pipeline.device`` defaults to ``None``."""
        assert self._pipe().device is None

    def test_device_stored_at_construction(self):
        """``Pipeline.device`` stores the value passed at construction time."""
        assert self._pipe(device="cpu").device == "cpu"

    @backend_equality_check()
    def test_cpu_device_all_backends(self):
        """``Pipeline(device='cpu')`` gives consistent results across all backends."""
        import keras
        import numpy as np

        from zea.ops import Pipeline
        from zea.ops.keras_ops import Abs

        pipe = Pipeline([Abs()], jit_options=None, device="cpu")
        return pipe(
            data=keras.ops.convert_to_tensor(np.array([-1.0, 2.0, -3.0], dtype=np.float32))
        )["data"]

    def test_cpu_device_correct_values(self):
        """``Pipeline(device='cpu')`` returns correct values when device is set at construction."""
        pipe = self._pipe(device="cpu")
        data = keras.ops.convert_to_tensor(np.array([-1.0, 2.0, -3.0], dtype=np.float32))
        out = pipe(data=data)["data"]
        np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_per_call_device_override(self):
        """``device=`` at call time overrides the pipeline-level device."""
        pipe = self._pipe(device="cpu")
        data = keras.ops.convert_to_tensor(np.array([-4.0], dtype=np.float32))
        out = pipe(device="cpu", data=data)["data"]
        np.testing.assert_allclose(out, np.array([4.0], dtype=np.float32))

    def test_no_device_runs_normally(self):
        """``Pipeline`` with ``device=None`` runs without device placement."""
        pipe = self._pipe()
        data = keras.ops.convert_to_tensor(np.array([-5.0, 6.0], dtype=np.float32))
        out = pipe(data=data)["data"]
        np.testing.assert_allclose(out, np.array([5.0, 6.0], dtype=np.float32))

    def test_get_dict_serialisation(self):
        """``get_dict`` includes ``device`` when set and omits it when ``None``."""
        assert self._pipe(device="cpu").get_dict(compact=True)["params"]["device"] == "cpu"
        assert self._pipe(device="cpu").get_dict(compact=False)["params"]["device"] == "cpu"
        assert "device" not in self._pipe().get_dict(compact=True).get("params", {})

    @pytest.mark.gpu
    def test_gpu_output_placement(self):  # pragma: no cover
        """Output tensor from ``Pipeline(device='gpu:0')`` resides on the GPU."""
        out = self._pipe(device="gpu:0")(
            data=keras.ops.convert_to_tensor(np.array([-1.0, 2.0], dtype=np.float32))
        )["data"]
        assert "cpu" not in _tensor_device_name(out)

    @pytest.mark.gpu
    def test_gpu_correct_result(self):  # pragma: no cover
        """``Pipeline(device='gpu:0')`` produces numerically correct results."""
        pipe = Pipeline([Abs()], jit_options=None, device="gpu:0")
        out = pipe(data=keras.ops.convert_to_tensor(np.array([-1.0, 2.0, -3.0], dtype=np.float32)))[
            "data"
        ]
        np.testing.assert_allclose(
            convert_to_numpy(out), np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )

    @pytest.mark.gpu
    def test_context_manager_on_gpu(self):  # pragma: no cover
        """``zea.device`` context manager + ``Pipeline`` works on GPU."""
        import zea

        pipe = self._pipe()
        data = keras.ops.convert_to_tensor(np.array([-1.0, 2.0], dtype=np.float32))
        with zea.device("gpu:0"):
            out = pipe(data=data)["data"]
        np.testing.assert_allclose(
            np.abs(convert_to_numpy(out)), np.array([1.0, 2.0], dtype=np.float32)
        )
