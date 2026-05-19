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
def test_init_device_falls_back_to_cpu_when_gpus_disabled(monkeypatch, backend):
    """init_device should return 'cpu' when CUDA_VISIBLE_DEVICES disables GPUs."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    device = init_device(device="auto:1", backend=backend, verbose=False)
    assert device == "cpu"
