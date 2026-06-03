"""Tests for infrastructure and configuration changes introduced in this PR.

Covers:
- .gitignore: new *.gz and *.nii data-file patterns
- .pre-commit-config.yaml: new run-spec-doc hook
- configs/README.md: updated API (from_path) and version strings
- .github/workflows/docker.yaml: build-production-images timeout
- .github/workflows/tests.yaml: tests / heavy-tests / notebook-tests timeouts
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# .gitignore
# ---------------------------------------------------------------------------


def _gitignore_patterns() -> list[str]:
    """Return the list of non-empty, non-comment lines in .gitignore."""
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


class TestGitignore:
    """Tests for the .gitignore changes in this PR (*.gz and *.nii added)."""

    def test_gz_pattern_present(self):
        """*.gz must be listed so that gzip files are not tracked."""
        assert "*.gz" in _gitignore_patterns()

    def test_nii_pattern_present(self):
        """*.nii must be listed so that NIfTI image files are not tracked."""
        assert "*.nii" in _gitignore_patterns()

    def test_gz_and_nii_adjacent(self):
        """*.gz and *.nii should appear consecutively (same Data-files block)."""
        patterns = _gitignore_patterns()
        gz_idx = patterns.index("*.gz")
        nii_idx = patterns.index("*.nii")
        assert abs(gz_idx - nii_idx) == 1, (
            "*.gz and *.nii should be adjacent entries in .gitignore"
        )

    def test_pre_existing_data_patterns_preserved(self):
        """Pre-existing data file patterns must still be present."""
        patterns = _gitignore_patterns()
        for pattern in ("*.hdf5", "*.h5", "*.npy", "*.npz", "*.mat"):
            assert pattern in patterns, f"Pre-existing pattern {pattern!r} was removed"

    def test_gz_in_data_section(self):
        """*.gz should appear in the 'Data files' section of .gitignore."""
        raw = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        data_section_start = raw.index("# Data files")
        gz_pos = raw.index("*.gz")
        assert gz_pos > data_section_start, (
            "*.gz should appear after the '# Data files' comment"
        )

    def test_nii_in_data_section(self):
        """*.nii should appear in the 'Data files' section of .gitignore."""
        raw = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        data_section_start = raw.index("# Data files")
        nii_pos = raw.index("*.nii")
        assert nii_pos > data_section_start, (
            "*.nii should appear after the '# Data files' comment"
        )


# ---------------------------------------------------------------------------
# .pre-commit-config.yaml
# ---------------------------------------------------------------------------


def _load_precommit() -> dict:
    with open(REPO_ROOT / ".pre-commit-config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _find_local_hook(hook_id: str) -> dict | None:
    """Return the local hook dict with the given id, or None."""
    cfg = _load_precommit()
    for repo_entry in cfg.get("repos", []):
        if repo_entry.get("repo") == "local":
            for hook in repo_entry.get("hooks", []):
                if hook.get("id") == hook_id:
                    return hook
    return None


class TestPreCommitConfig:
    """Tests for the run-spec-doc hook added to .pre-commit-config.yaml."""

    def test_run_spec_doc_hook_exists(self):
        """The run-spec-doc hook must be present in the local hooks."""
        hook = _find_local_hook("run-spec-doc")
        assert hook is not None, "run-spec-doc hook not found in .pre-commit-config.yaml"

    def test_run_spec_doc_entry(self):
        """The hook entry must invoke spec_doc.py via Python."""
        hook = _find_local_hook("run-spec-doc")
        assert hook is not None
        assert hook.get("entry") == "python docs/source/spec_doc.py"

    def test_run_spec_doc_language(self):
        """The hook language must be 'system'."""
        hook = _find_local_hook("run-spec-doc")
        assert hook is not None
        assert hook.get("language") == "system"

    def test_run_spec_doc_files_pattern(self):
        """The hook must trigger on both spec_doc.py and zea/data/spec.py."""
        hook = _find_local_hook("run-spec-doc")
        assert hook is not None
        files_pattern = hook.get("files", "")
        assert "spec_doc" in files_pattern, (
            "files pattern should match spec_doc.py"
        )
        assert "zea/data/spec" in files_pattern, (
            "files pattern should also match zea/data/spec.py"
        )

    def test_run_spec_doc_pass_filenames_false(self):
        """pass_filenames must be False so the script is not given file args."""
        hook = _find_local_hook("run-spec-doc")
        assert hook is not None
        assert hook.get("pass_filenames") is False

    def test_run_spec_doc_name(self):
        """The hook must have a human-readable name."""
        hook = _find_local_hook("run-spec-doc")
        assert hook is not None
        assert hook.get("name") == "Run spec_doc.py"

    def test_existing_hooks_preserved(self):
        """Pre-existing local hooks must still be present."""
        for hook_id in ("run-parameters-doc", "generate-keras-ops", "notebook-clean-and-check"):
            assert _find_local_hook(hook_id) is not None, (
                f"Pre-existing hook {hook_id!r} was removed"
            )

    def test_precommit_yaml_is_valid(self):
        """The .pre-commit-config.yaml file must be parseable YAML."""
        cfg = _load_precommit()
        assert "repos" in cfg, "Expected top-level 'repos' key in pre-commit config"

    def test_run_spec_doc_files_regex_matches_spec_doc_py(self):
        """The files regex should match docs/source/spec_doc.py."""
        import re

        hook = _find_local_hook("run-spec-doc")
        assert hook is not None
        pattern = hook.get("files", "")
        assert re.search(pattern, "docs/source/spec_doc.py"), (
            f"Pattern {pattern!r} does not match 'docs/source/spec_doc.py'"
        )

    def test_run_spec_doc_files_regex_matches_zea_data_spec_py(self):
        """The files regex should match zea/data/spec.py."""
        import re

        hook = _find_local_hook("run-spec-doc")
        assert hook is not None
        pattern = hook.get("files", "")
        assert re.search(pattern, "zea/data/spec.py"), (
            f"Pattern {pattern!r} does not match 'zea/data/spec.py'"
        )


# ---------------------------------------------------------------------------
# configs/README.md
# ---------------------------------------------------------------------------


def _configs_readme_text() -> str:
    return (REPO_ROOT / "configs" / "README.md").read_text(encoding="utf-8")


class TestConfigsReadme:
    """Tests for the configs/README.md API and version-string changes."""

    def test_from_path_api_present(self):
        """README must use the updated Config.from_path() API."""
        assert "Config.from_path(" in _configs_readme_text()

    def test_from_hf_api_absent(self):
        """Deprecated Config.from_hf() API must not appear in README."""
        assert "Config.from_hf(" not in _configs_readme_text(), (
            "Deprecated from_hf() call still present in configs/README.md"
        )

    def test_hf_url_scheme_used(self):
        """README examples must use the hf:// URL scheme."""
        assert 'hf://' in _configs_readme_text()

    def test_version_v0_0_10_present(self):
        """README must reference release v0.0.10."""
        assert "v0.0.10" in _configs_readme_text()

    def test_version_v0_0_11_present(self):
        """README must reference release v0.0.11."""
        assert "v0.0.11" in _configs_readme_text()

    def test_old_version_v0_0_1_absent(self):
        """Old version string v0.0.1 (non-suffix match) must not appear."""
        import re

        # v0.0.1 as a standalone version (not as prefix of v0.0.10/v0.0.11)
        matches = re.findall(r"v0\.0\.1(?!\d)", _configs_readme_text())
        assert not matches, (
            f"Old version string 'v0.0.1' still present in configs/README.md: {matches}"
        )

    def test_old_version_v0_0_2_absent(self):
        """Old version string v0.0.2 must not appear in the README."""
        assert "v0.0.2" not in _configs_readme_text()

    def test_revision_kwarg_used(self):
        """The README example for loading a specific release must use revision=."""
        assert "revision=" in _configs_readme_text()

    def test_load_example_uses_correct_config_name(self):
        """Both README examples should reference config_picmus_rf.yaml."""
        text = _configs_readme_text()
        assert text.count("config_picmus_rf.yaml") >= 2, (
            "Expected at least two references to config_picmus_rf.yaml"
        )


# ---------------------------------------------------------------------------
# .github/workflows/docker.yaml
# ---------------------------------------------------------------------------


def _load_docker_workflow() -> dict:
    with open(REPO_ROOT / ".github" / "workflows" / "docker.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class TestDockerWorkflow:
    """Tests for the docker.yaml timeout change (120 → 200)."""

    def test_build_production_images_timeout(self):
        """build-production-images job timeout must be 200 minutes."""
        cfg = _load_docker_workflow()
        job = cfg["jobs"]["build-production-images"]
        assert job["timeout-minutes"] == 200

    def test_build_production_images_timeout_not_old_value(self):
        """build-production-images timeout must not still be 120."""
        cfg = _load_docker_workflow()
        job = cfg["jobs"]["build-production-images"]
        assert job["timeout-minutes"] != 120

    def test_docker_yaml_is_valid(self):
        """docker.yaml must be parseable YAML."""
        cfg = _load_docker_workflow()
        assert "jobs" in cfg


# ---------------------------------------------------------------------------
# .github/workflows/tests.yaml
# ---------------------------------------------------------------------------


def _load_tests_workflow() -> dict:
    with open(REPO_ROOT / ".github" / "workflows" / "tests.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class TestTestsWorkflow:
    """Tests for the tests.yaml timeout changes."""

    def test_tests_job_timeout(self):
        """tests job timeout must be 120 minutes (was 60)."""
        cfg = _load_tests_workflow()
        assert cfg["jobs"]["tests"]["timeout-minutes"] == 120

    def test_tests_job_timeout_not_old_value(self):
        """tests job timeout must not still be 60 minutes."""
        cfg = _load_tests_workflow()
        assert cfg["jobs"]["tests"]["timeout-minutes"] != 60

    def test_heavy_tests_timeout(self):
        """heavy-tests job timeout must be 120 minutes (was 100)."""
        cfg = _load_tests_workflow()
        assert cfg["jobs"]["heavy-tests"]["timeout-minutes"] == 120

    def test_heavy_tests_timeout_not_old_value(self):
        """heavy-tests job timeout must not still be 100 minutes."""
        cfg = _load_tests_workflow()
        assert cfg["jobs"]["heavy-tests"]["timeout-minutes"] != 100

    def test_notebook_tests_timeout(self):
        """notebook-tests job timeout must be 120 minutes (was 100)."""
        cfg = _load_tests_workflow()
        assert cfg["jobs"]["notebook-tests"]["timeout-minutes"] == 120

    def test_notebook_tests_timeout_not_old_value(self):
        """notebook-tests job timeout must not still be 100 minutes."""
        cfg = _load_tests_workflow()
        assert cfg["jobs"]["notebook-tests"]["timeout-minutes"] != 100

    def test_all_updated_timeouts_equal(self):
        """All three updated timeouts (tests, heavy-tests, notebook-tests) must be equal."""
        cfg = _load_tests_workflow()
        jobs = cfg["jobs"]
        timeouts = {
            "tests": jobs["tests"]["timeout-minutes"],
            "heavy-tests": jobs["heavy-tests"]["timeout-minutes"],
            "notebook-tests": jobs["notebook-tests"]["timeout-minutes"],
        }
        unique = set(timeouts.values())
        assert len(unique) == 1, (
            f"Expected all three timeouts to be equal, got: {timeouts}"
        )

    def test_tests_yaml_is_valid(self):
        """tests.yaml must be parseable YAML."""
        cfg = _load_tests_workflow()
        assert "jobs" in cfg

    def test_image_job_timeout_unchanged(self):
        """image job timeout must still be 100 minutes (not changed by this PR)."""
        cfg = _load_tests_workflow()
        assert cfg["jobs"]["image"]["timeout-minutes"] == 100
