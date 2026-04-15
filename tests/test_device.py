"""GPU usage testing"""

from itertools import product
from unittest.mock import patch

import numpy as np
import pytest

from zea.backend import func_on_device
from zea.internal.device import (
    _cuda_visible_devices_disables_gpus,
    get_gpu_memory,
    init_device,
)

from . import DEFAULT_TEST_SEED, backend_equality_check

# NOTE: tests/__init__.py sets CUDA_VISIBLE_DEVICES="" so the full test suite runs
# in CPU-only mode in CI (no GPU required).  Tests marked with @pytest.mark.gpu test
# GPU-selection logic that falls back to CPU automatically in CI; they still contribute
# to coverage but are most meaningful when run locally with a real GPU.

devices = ["cpu", "gpu:0", "cuda:0", "auto:-1", "auto:1"]
backends = ["tensorflow", "torch", "jax", "auto", "numpy"]


@pytest.mark.gpu
@pytest.mark.parametrize("device, backend", list(product(devices, backends)))
def test_init_device(device, backend):
    """Test device initialization with combinations of device and backend.

    In CI (CUDA_VISIBLE_DEVICES="") all GPU device strings fall back to CPU;
    run locally with a GPU to exercise the full selection path.
    """
    init_device(device=device, backend=backend, verbose=False)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", backends)
def test_default_init_device(backend):
    """Test gpu usage setting script with defaults.

    In CI (CUDA_VISIBLE_DEVICES="") this exercises the CPU fallback path;
    run locally with a GPU to test real GPU selection.
    """
    init_device(backend=backend, verbose=False)


@backend_equality_check()
def test_func_on_device():
    """Test func_on_device with all backends."""

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    x = rng.standard_normal((3, 3))
    y = rng.standard_normal((3, 3))

    def f(x, y):
        return x + y

    return func_on_device(f, "cpu", x, y)


class TestCudaVisibleDevicesDisablesGpus:
    """Tests for _cuda_visible_devices_disables_gpus helper."""

    def test_unset(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert _cuda_visible_devices_disables_gpus() is False

    def test_empty_string(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert _cuda_visible_devices_disables_gpus() is True

    def test_minus_one(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
        assert _cuda_visible_devices_disables_gpus() is True

    def test_negative_ids(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1,-2")
        assert _cuda_visible_devices_disables_gpus() is True

    def test_valid_id(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        assert _cuda_visible_devices_disables_gpus() is False

    def test_mixed_ids(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,-1")
        assert _cuda_visible_devices_disables_gpus() is False

    def test_whitespace(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " -1 ")
        assert _cuda_visible_devices_disables_gpus() is True

    def test_uuid_style_value(self, monkeypatch):
        """Non-numeric UUID-style values should not raise and should return False."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc123,GPU-def456")
        assert _cuda_visible_devices_disables_gpus() is False


class TestGetGpuMemoryRespectsEnv:
    """get_gpu_memory must return [] when CUDA_VISIBLE_DEVICES disables GPUs."""

    def test_minus_one(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
        assert get_gpu_memory(verbose=False) == []

    def test_empty_string(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert get_gpu_memory(verbose=False) == []


_SMI_TWO_GPUS = b"1000\n2000\n"


def _mock_smi(monkeypatch, raw_output):
    """Patch check_nvidia_smi and sp.check_output for unit-testing get_gpu_memory."""
    monkeypatch.setattr("zea.internal.device.check_nvidia_smi", lambda: True)
    return patch("subprocess.check_output", return_value=raw_output)


class TestGetGpuMemoryFiltering:
    """Tests for GPU ID filtering and nvidia-smi output parsing in get_gpu_memory."""

    def test_output_to_list_parses_smi_output(self, monkeypatch):
        """get_gpu_memory correctly parses multi-line nvidia-smi output."""
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        with _mock_smi(monkeypatch, _SMI_TWO_GPUS):
            result = get_gpu_memory(verbose=False)
        assert result == [1000, 2000]

    def test_out_of_range_ids_filtered(self, monkeypatch):
        """GPU IDs beyond the number of detected GPUs are silently removed."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,5")
        with _mock_smi(monkeypatch, _SMI_TWO_GPUS):
            result = get_gpu_memory(verbose=False)
        assert result == [1000]  # GPU 5 is out of range, only GPU 0 kept

    def test_negative_in_mixed_ids_filtered(self, monkeypatch):
        """Negative IDs mixed with valid IDs are filtered without discarding valid ones."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,-1")
        with _mock_smi(monkeypatch, _SMI_TWO_GPUS):
            result = get_gpu_memory(verbose=False)
        assert result == [1000]  # -1 filtered out, GPU 0 kept


@pytest.mark.parametrize("backend", ["tensorflow", "torch", "jax"])
def test_init_device_falls_back_to_cpu_when_gpus_disabled(monkeypatch, backend):
    """init_device should return 'cpu' when CUDA_VISIBLE_DEVICES disables GPUs."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    device = init_device(device="auto:1", backend=backend, verbose=False)
    assert device == "cpu"
