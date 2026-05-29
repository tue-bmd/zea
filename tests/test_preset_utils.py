"""Tests for zea.internal.preset_utils.

Covers the _hf_parse_path helper introduced/documented in this PR,
which underpins the Config.from_path("hf://...") API shown in configs/README.md.
"""

import pytest

from zea.internal.preset_utils import HF_PREFIX, _hf_parse_path


class TestHfParsePath:
    """Tests for the _hf_parse_path function."""

    def test_file_at_repo_root(self):
        """hf://org/repo/file.yaml -> ('org/repo', 'file.yaml')."""
        repo_id, subpath = _hf_parse_path("hf://org/repo/file.yaml")
        assert repo_id == "org/repo"
        assert subpath == "file.yaml"

    def test_nested_file(self):
        """hf://org/repo/subdir/file.yaml -> ('org/repo', 'subdir/file.yaml')."""
        repo_id, subpath = _hf_parse_path("hf://org/repo/subdir/file.yaml")
        assert repo_id == "org/repo"
        assert subpath == "subdir/file.yaml"

    def test_repo_only_no_subpath(self):
        """hf://org/repo with no subpath -> ('org/repo', None)."""
        repo_id, subpath = _hf_parse_path("hf://org/repo")
        assert repo_id == "org/repo"
        assert subpath is None

    def test_deeply_nested_file(self):
        """hf://org/repo/a/b/c/file.txt -> subpath includes all components after repo."""
        repo_id, subpath = _hf_parse_path("hf://org/repo/a/b/c/file.txt")
        assert repo_id == "org/repo"
        assert subpath == "a/b/c/file.txt"

    def test_picmus_config_from_readme(self):
        """Parses the exact URI documented in configs/README.md."""
        repo_id, subpath = _hf_parse_path("hf://zeahub/configs/config_picmus_rf.yaml")
        assert repo_id == "zeahub/configs"
        assert subpath == "config_picmus_rf.yaml"

    def test_camus_config_from_readme(self):
        """Parses the camus config URI referenced in configs/README.md."""
        repo_id, subpath = _hf_parse_path("hf://zeahub/configs/config_camus.yaml")
        assert repo_id == "zeahub/configs"
        assert subpath == "config_camus.yaml"

    def test_invalid_scheme_raises_value_error(self):
        """A path not starting with hf:// should raise ValueError."""
        with pytest.raises(ValueError, match="hf://"):
            _hf_parse_path("https://huggingface.co/org/repo/file.yaml")

    def test_plain_path_raises_value_error(self):
        """A plain local-style path should raise ValueError."""
        with pytest.raises(ValueError, match="hf://"):
            _hf_parse_path("/local/path/to/file.yaml")

    def test_empty_string_raises_value_error(self):
        """An empty string should raise ValueError."""
        with pytest.raises(ValueError, match="hf://"):
            _hf_parse_path("")

    def test_hf_prefix_constant(self):
        """HF_PREFIX should equal 'hf://'."""
        assert HF_PREFIX == "hf://"

    def test_repo_id_always_two_parts(self):
        """The repo_id part should always be 'org/repo' (two slash-separated parts)."""
        repo_id, _ = _hf_parse_path("hf://myorg/myrepo/somefile.yaml")
        parts = repo_id.split("/")
        assert len(parts) == 2
        assert parts[0] == "myorg"
        assert parts[1] == "myrepo"

    @pytest.mark.parametrize(
        "hf_path, expected_repo, expected_subpath",
        [
            ("hf://zeahub/configs/config_picmus_rf.yaml", "zeahub/configs", "config_picmus_rf.yaml"),
            ("hf://zeahub/configs/config_camus.yaml", "zeahub/configs", "config_camus.yaml"),
            ("hf://zeahub/configs/config_picmus_iq.yaml", "zeahub/configs", "config_picmus_iq.yaml"),
            ("hf://zeahub/datasets/scan.h5", "zeahub/datasets", "scan.h5"),
            ("hf://myorg/myrepo", "myorg/myrepo", None),
        ],
    )
    def test_parse_path_parametrized(self, hf_path, expected_repo, expected_subpath):
        """Parametrized test of various hf:// URI forms."""
        repo_id, subpath = _hf_parse_path(hf_path)
        assert repo_id == expected_repo
        assert subpath == expected_subpath