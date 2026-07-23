"""Tests for the :class:`zea.ops.MachBeamform` operation.

The numerical work happens inside the ``mach`` CUDA kernel, which needs a GPU
and the optional ``mach-beamform`` / ``cupy`` packages. Those tests are marked
``@pytest.mark.gpu`` and auto-skip on CPU-only CI (see ``tests/conftest.py``).
The remaining tests are lightweight and run everywhere: they check registration,
export, lazy-import behaviour and the pure-Python validation branches (the
Zea->mach translation, including :func:`zea.beamform.beamformer.calculate_delays`,
runs on CPU up to the kernel launch).
"""

import importlib.util

import numpy as np
import pytest

from zea import ops
from zea.internal.registry import ops_registry

_MACH_AVAILABLE = importlib.util.find_spec("mach") is not None and (
    importlib.util.find_spec("cupy") is not None
)


def _synthetic_inputs(n_tx=2, n_el=4, n_ax=16, n_pix=8, n_ch=1):
    """Build a minimal, self-consistent set of scan parameters + raw data."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_tx, n_ax, n_el, n_ch)).astype("float32")
    params = dict(
        flatgrid=rng.random((n_pix, 3)).astype("float32"),
        probe_geometry=rng.random((n_el, 3)).astype("float32"),
        sampling_frequency=40e6,
        sound_speed=1540.0,
        f_number=1.0,
        t0_delays=np.zeros((n_tx, n_el), "float32"),
        tx_apodizations=np.ones((n_tx, n_el), "float32"),
        focus_distances=np.zeros((n_tx,), "float32"),
        polar_angles=np.zeros((n_tx,), "float32"),
        t_peak=np.zeros((n_tx,), "float32"),
        transmit_origins=np.zeros((n_tx, 3), "float32"),
        initial_times=np.zeros((n_tx,), "float32"),
    )
    return data, params


def test_mach_beamform_registered_and_exported():
    """Op is exported from ``zea.ops`` and registered under ``mach_beamform``."""
    assert hasattr(ops, "MachBeamform")
    assert "mach_beamform" in ops_registry
    assert ops_registry["mach_beamform"] is ops.MachBeamform


def test_mach_beamform_constructs_without_mach():
    """Op constructs on any machine (imports are deferred to call time)."""
    op = ops.MachBeamform(with_batch_dim=False)
    assert op.input_data_type.value == "raw_data"
    assert op.output_data_type.value == "beamformed_data"
    assert op.jittable is False
    assert op.static_params == ["interp_type", "tukey_alpha"]


def test_mach_beamform_invalid_interp_type():
    """An unknown interpolation type is rejected early, before any mach import."""
    with pytest.raises(ValueError, match="interp_type"):
        ops.MachBeamform(interp_type="bogus")


def test_mach_beamform_missing_parameters_raises():
    """Missing scan parameters raise a clear error before the kernel launch."""
    data, params = _synthetic_inputs()
    params.pop("t0_delays")
    op = ops.MachBeamform(with_batch_dim=False)
    with pytest.raises(ValueError, match="Missing Zea scan parameters"):
        op(data=data, **params)


def test_mach_beamform_bad_channel_dim_raises():
    """Only RF (n_ch=1) and IQ (n_ch=2) inputs are accepted."""
    data, params = _synthetic_inputs(n_ch=3)
    op = ops.MachBeamform(with_batch_dim=False)
    with pytest.raises(ValueError, match="RF .* or IQ"):
        op(data=data, **params)


def test_mach_beamform_iq_requires_demodulation_frequency():
    """IQ input without a demodulation frequency is rejected."""
    data, params = _synthetic_inputs(n_ch=2)
    op = ops.MachBeamform(with_batch_dim=False)
    with pytest.raises(ValueError, match="demodulation_frequency"):
        op(data=data, **params)


@pytest.mark.skipif(
    _MACH_AVAILABLE,
    reason="mach is installed; the informative ImportError path cannot be exercised",
)
def test_mach_beamform_importerror_without_mach():
    """Without mach installed, calling raises an actionable ImportError.

    This also exercises the CPU-side Zea->mach translation (stage 1), including
    ``calculate_delays``, up to the point of the kernel launch.
    """
    data, params = _synthetic_inputs()
    op = ops.MachBeamform(with_batch_dim=False)
    with pytest.raises(ImportError, match=r"pip install"):
        op(data=data, **params)


@pytest.mark.gpu
@pytest.mark.parametrize("n_ch", [1, 2])
def test_mach_beamform_runs_on_gpu(n_ch):
    """End-to-end kernel run on a real GPU produces sane, finite output."""
    if not _MACH_AVAILABLE:
        pytest.skip("mach and/or cupy not installed")
    import keras

    n_pix = 8
    data, params = _synthetic_inputs(n_pix=n_pix, n_ch=n_ch)
    if n_ch == 2:
        params["demodulation_frequency"] = 5e6

    op = ops.MachBeamform(with_batch_dim=False)
    out = op(data=data, **params)["data"]
    out = keras.ops.convert_to_numpy(out)

    assert out.shape == (n_pix, n_ch)
    assert np.isfinite(out).all()
