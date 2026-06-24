"""Tests for ``zea.backend``."""

from . import run_in_backend


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
        """Returns the torch module regardless of the active backend when force=True."""
        from zea.backend import _import_torch

        assert _import_torch(force=True) is not None

    @staticmethod
    @run_in_backend("jax")
    def test_returns_none_on_import_error():
        """Returns None gracefully when torch raises ImportError (e.g. not installed)."""
        import sys
        import unittest.mock

        from zea.backend import _import_torch

        with unittest.mock.patch.dict(sys.modules, {"torch": None}):
            assert _import_torch(force=True) is None
