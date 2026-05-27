"""Tests for zea.internal._generate_keras_ops helper functions."""

import pytest

import zea.internal._generate_keras_ops as mod
from zea.internal._generate_keras_ops import (
    _check_version_and_generate,
    _get_generated_keras_version,
    _parse_version,
)


class TestParseVersion:
    def test_standard_version(self):
        assert _parse_version("3.14.0") == (3, 14, 0)

    def test_pre_release_rc(self):
        assert _parse_version("3.14.0rc1") == (3, 14, 0)

    def test_pre_release_dev(self):
        assert _parse_version("3.14.0.dev0") == (3, 14, 0)

    def test_two_part_version(self):
        assert _parse_version("3.14") == (3, 14)

    def test_older_version(self):
        assert _parse_version("3.10.0") == (3, 10, 0)

    def test_comparison_older_is_less(self):
        old = _parse_version("3.10.0")
        new = _parse_version("3.14.0")
        assert old < new

    def test_comparison_same_is_not_less(self):
        v1 = _parse_version("3.14.0")
        v2 = _parse_version("3.14.0")
        assert not (v1 < v2)

    def test_comparison_newer_is_not_less(self):
        newer = _parse_version("3.15.0")
        generated = _parse_version("3.14.0")
        assert not (newer < generated)


class TestGetGeneratedKerasVersion:
    def test_returns_none_for_missing_file(self, tmp_path):
        result = _get_generated_keras_version(tmp_path / "nonexistent.py")
        assert result is None

    def test_extracts_version_from_header(self, tmp_path):
        f = tmp_path / "keras_ops.py"
        f.write_text('"""...\nGenerated with Keras 3.14.0\n"""\n')
        assert _get_generated_keras_version(f) == (3, 14, 0)

    def test_returns_none_when_no_version_in_file(self, tmp_path):
        f = tmp_path / "keras_ops.py"
        f.write_text("# no version info here\n")
        assert _get_generated_keras_version(f) is None


class TestCheckVersionAndGenerate:
    def test_exits_when_installed_version_is_older(self, tmp_path, monkeypatch, capsys):
        """Older installed Keras should trigger the warning and exit."""
        f = tmp_path / "keras_ops.py"
        f.write_text('"""...\nGenerated with Keras 99.0.0\n"""\n')

        monkeypatch.setattr(mod.keras, "__version__", "1.0.0")

        with pytest.raises(SystemExit) as exc_info:
            _check_version_and_generate(f)

        assert exc_info.value.code == 1
        output = capsys.readouterr().out
        assert "WARNING" in output
        assert "pip install --upgrade keras" in output
