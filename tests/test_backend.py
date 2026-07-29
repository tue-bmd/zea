"""Tests for ``zea.backend``."""

import pytest

from . import run_in_backend
from .backend_utils import missing_required_backends


@pytest.mark.tensorflow
class TestImportTf:
    """Tests for ``_import_tf``: the lazy TensorFlow import helper in ``zea.backend``.

    Each test runs in an isolated backend worker so that ``keras.backend.backend()``
    reflects the target backend rather than the default test-session backend.
    """

    @staticmethod
    @run_in_backend("tensorflow")
    def test_returns_module_in_matching_backend():
        """Returns the tensorflow module when the active backend is tensorflow."""
        from zea.backend import _import_tf

        assert _import_tf(force=False) is not None

    @staticmethod
    @run_in_backend("jax")
    def test_returns_none_for_wrong_backend():
        """Returns None without attempting to import when the backend does not match."""
        from zea.backend import _import_tf

        assert _import_tf(force=False) is None

    @staticmethod
    @run_in_backend("jax")
    def test_force_bypasses_backend_check():
        """Returns the tensorflow module regardless of the active backend when force=True."""
        from zea.backend import _import_tf

        assert _import_tf(force=True) is not None

    @staticmethod
    @run_in_backend("tensorflow")
    def test_returns_none_on_import_error():
        """Returns None gracefully when tensorflow raises ImportError (e.g. not installed)."""
        import sys
        import unittest.mock

        from zea.backend import _import_tf

        with unittest.mock.patch.dict(sys.modules, {"tensorflow": None}):
            assert _import_tf(force=True) is None


@pytest.mark.jax
class TestImportJax:
    """Tests for ``_import_jax``: the lazy JAX import helper in ``zea.backend``."""

    @staticmethod
    @run_in_backend("jax")
    def test_returns_module_in_matching_backend():
        """Returns the jax module when the active backend is jax."""
        from zea.backend import _import_jax

        assert _import_jax(force=False) is not None

    @staticmethod
    @run_in_backend("tensorflow")
    def test_returns_none_for_wrong_backend():
        """Returns None without attempting to import when the backend does not match."""
        from zea.backend import _import_jax

        assert _import_jax(force=False) is None

    @staticmethod
    @run_in_backend("tensorflow")
    def test_force_bypasses_backend_check():
        """Returns the jax module regardless of the active backend when force=True."""
        from zea.backend import _import_jax

        assert _import_jax(force=True) is not None

    @staticmethod
    @run_in_backend("tensorflow")
    def test_returns_none_on_import_error():
        """Returns None gracefully when jax raises ImportError (e.g. not installed)."""
        import sys
        import unittest.mock

        from zea.backend import _import_jax

        with unittest.mock.patch.dict(sys.modules, {"jax": None}):
            assert _import_jax(force=True) is None


@pytest.mark.torch
class TestImportTorch:
    """Tests for ``_import_torch``: the lazy PyTorch import helper in ``zea.backend``."""

    @staticmethod
    @run_in_backend("torch")
    def test_returns_module_in_matching_backend():
        """Returns the torch module when the active backend is torch."""
        from zea.backend import _import_torch

        assert _import_torch(force=False) is not None

    @staticmethod
    @run_in_backend("jax")
    def test_returns_none_for_wrong_backend():
        """Returns None without attempting to import when the backend does not match."""
        from zea.backend import _import_torch

        assert _import_torch(force=False) is None

    @staticmethod
    @run_in_backend("jax")
    def test_force_bypasses_backend_check():
        """Returns the torch module when available.

        Regardless of the active backend when force=True.
        """
        from zea.backend import _import_torch

        result = _import_torch(force=True)
        if missing_required_backends(["torch"]):
            assert result is None, "torch not installed, should return None"
        else:
            assert result is not None, "torch is installed, should return the module"

    @staticmethod
    @run_in_backend("jax")
    def test_returns_none_on_import_error():
        """Returns None gracefully when torch raises ImportError (e.g. not installed)."""
        import sys
        import unittest.mock

        from zea.backend import _import_torch

        with unittest.mock.patch.dict(sys.modules, {"torch": None}):
            assert _import_torch(force=True) is None


class TestAdam:
    """Tests for the backend-agnostic Adam optimizer in ``zea.backend.optimizer``."""

    @staticmethod
    def _quadratic_grad(x, minimum=3.0):
        """Gradient of ``(x - minimum) ** 2``."""
        import keras

        return 2 * (keras.ops.convert_to_tensor(x) - minimum)

    def test_init_returns_zeroed_state(self):
        """``init`` returns the parameter alongside zeroed moments and a zero step count."""
        import keras
        import numpy as np

        from zea.backend.optimizer import adam

        init, _, get_params = adam(step_size=0.1)
        x0 = keras.ops.convert_to_tensor(np.ones((3,), dtype="float32"))
        x, m, v, i = init(x0)

        assert i == 0
        np.testing.assert_allclose(keras.ops.convert_to_numpy(get_params((x, m, v, i))), np.ones(3))
        np.testing.assert_allclose(keras.ops.convert_to_numpy(m), np.zeros(3))
        np.testing.assert_allclose(keras.ops.convert_to_numpy(v), np.zeros(3))

    def test_first_step_is_bias_corrected(self):
        """After bias correction, the first update is a full step in the gradient direction."""
        import keras
        import numpy as np

        from zea.backend.optimizer import adam

        step_size = 0.1
        init, update, get_params = adam(step_size=step_size)

        state = init(keras.ops.convert_to_tensor(np.array([0.0, 0.0], dtype="float32")))
        gradient = keras.ops.convert_to_tensor(np.array([1.0, -2.0], dtype="float32"))
        state = update(gradient, state)

        # mhat / sqrt(vhat) reduces to sign(g) on the very first step
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(get_params(state)),
            np.array([-step_size, step_size]),
            rtol=1e-4,
            atol=1e-6,
        )
        assert state[-1] == 1

    def test_converges_to_the_minimum(self):
        """Repeated updates drive the parameter to the minimum of a quadratic."""
        import keras
        import numpy as np

        from zea.backend.optimizer import adam

        init, update, get_params = adam(step_size=0.1)
        state = init(keras.ops.convert_to_tensor(np.array([0.0], dtype="float32")))

        for _ in range(500):
            state = update(self._quadratic_grad(get_params(state)), state)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(get_params(state)), np.array([3.0]), rtol=1e-3, atol=1e-3
        )

    def test_step_size_controls_progress(self):
        """A larger step size makes more progress in the same number of updates."""
        import keras
        import numpy as np

        from zea.backend.optimizer import adam

        def run(step_size, steps=10):
            init, update, get_params = adam(step_size=step_size)
            state = init(keras.ops.convert_to_tensor(np.array([0.0], dtype="float32")))
            for _ in range(steps):
                state = update(self._quadratic_grad(get_params(state)), state)
            return float(keras.ops.convert_to_numpy(get_params(state))[0])

        assert run(0.5) > run(0.05)


@pytest.mark.jax
class TestStrToJaxDevice:
    """Tests for ``str_to_jax_device``: the device-string parser for the JAX backend."""

    @staticmethod
    @run_in_backend("jax")
    def test_parses_device_string_with_and_without_index():
        """Both ``'cpu'`` and ``'cpu:0'`` resolve to the first CPU device."""
        import jax

        from zea.backend import str_to_jax_device

        assert str_to_jax_device("cpu") == jax.devices("cpu")[0]
        assert str_to_jax_device("cpu:0") == jax.devices("cpu")[0]

    @staticmethod
    @run_in_backend("jax")
    def test_is_case_insensitive():
        """Device strings are normalised to lowercase."""
        import jax

        from zea.backend import str_to_jax_device

        assert str_to_jax_device("CPU:0") == jax.devices("cpu")[0]

    @staticmethod
    @run_in_backend("jax")
    def test_rejects_non_string_device():
        """A non-string device raises a ValueError."""
        import pytest

        from zea.backend import str_to_jax_device

        with pytest.raises(ValueError, match="must be a string"):
            str_to_jax_device(0)

    @staticmethod
    @run_in_backend("jax")
    def test_rejects_unavailable_device_type():
        """A device type that jax cannot initialize propagates its RuntimeError."""
        import pytest

        from zea.backend import str_to_jax_device

        with pytest.raises(RuntimeError):
            str_to_jax_device("tpu:0")

    @staticmethod
    @run_in_backend("jax")
    def test_rejects_out_of_range_device_number():
        """Requesting a device index that does not exist raises a ValueError."""
        import pytest

        from zea.backend import str_to_jax_device

        with pytest.raises(ValueError, match="not available"):
            str_to_jax_device("cpu:99")
