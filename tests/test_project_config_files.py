"""Tests for project-level configuration files changed in this PR.

Covers:
  - .gitignore: added *.gz and *.nii patterns
  - .pre-commit-config.yaml: added run-spec-doc hook
  - .github/workflows/tests.yaml: added conditional skip for 'openh-rf' base branch
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# .gitignore tests
# ---------------------------------------------------------------------------


class TestGitignorePatterns:
    """Verify the new *.gz and *.nii patterns are present in .gitignore."""

    @pytest.fixture(autouse=True)
    def gitignore_lines(self):
        gitignore_path = REPO_ROOT / ".gitignore"
        assert gitignore_path.exists(), ".gitignore not found at repo root"
        self._lines = gitignore_path.read_text(encoding="utf-8").splitlines()

    def test_gz_pattern_present(self):
        """*.gz should be listed in .gitignore."""
        assert "*.gz" in self._lines, "*.gz not found in .gitignore"

    def test_nii_pattern_present(self):
        """*.nii should be listed in .gitignore."""
        assert "*.nii" in self._lines, "*.nii not found in .gitignore"

    def test_gz_and_nii_near_other_data_patterns(self):
        """*.gz and *.nii should be in the data-files section, near *.h5 etc."""
        # Locate the index of *.h5 (already present) and confirm *.gz and *.nii
        # are within a reasonable distance.
        h5_index = next((i for i, l in enumerate(self._lines) if l.strip() == "*.h5"), None)
        gz_index = next((i for i, l in enumerate(self._lines) if l.strip() == "*.gz"), None)
        nii_index = next((i for i, l in enumerate(self._lines) if l.strip() == "*.nii"), None)

        assert h5_index is not None, "*.h5 baseline pattern not found in .gitignore"
        assert gz_index is not None
        assert nii_index is not None
        # Both new patterns should be within 10 lines of *.h5
        assert abs(gz_index - h5_index) <= 10, "*.gz is unexpectedly far from *.h5 in .gitignore"
        assert abs(nii_index - h5_index) <= 10, "*.nii is unexpectedly far from *.h5 in .gitignore"

    def test_existing_data_patterns_still_present(self):
        """Regression: previously existing data patterns must not have been removed."""
        for pattern in ("*.hdf5", "*.h5", "*.npy", "*.npz", "*.mat"):
            assert pattern in self._lines, f"{pattern} was unexpectedly removed from .gitignore"


# ---------------------------------------------------------------------------
# .pre-commit-config.yaml tests
# ---------------------------------------------------------------------------


class TestPrecommitRunSpecDocHook:
    """Verify the new run-spec-doc hook in .pre-commit-config.yaml."""

    @pytest.fixture(autouse=True)
    def precommit_config(self):
        config_path = REPO_ROOT / ".pre-commit-config.yaml"
        assert config_path.exists(), ".pre-commit-config.yaml not found at repo root"
        with config_path.open(encoding="utf-8") as fh:
            self._config = yaml.safe_load(fh)
        # Collect all hooks from all repos
        self._all_hooks = []
        for repo in self._config.get("repos", []):
            self._all_hooks.extend(repo.get("hooks", []))

    def _find_hook(self, hook_id):
        return next((h for h in self._all_hooks if h.get("id") == hook_id), None)

    def test_run_spec_doc_hook_exists(self):
        """The run-spec-doc hook must be present."""
        hook = self._find_hook("run-spec-doc")
        assert hook is not None, "run-spec-doc hook not found in .pre-commit-config.yaml"

    def test_run_spec_doc_entry_command(self):
        """The run-spec-doc hook must invoke docs/source/spec_doc.py."""
        hook = self._find_hook("run-spec-doc")
        assert hook is not None
        assert hook.get("entry") == "python docs/source/spec_doc.py"

    def test_run_spec_doc_triggers_on_spec_py(self):
        """The hook's files pattern must match zea/data/spec.py."""
        import re

        hook = self._find_hook("run-spec-doc")
        assert hook is not None
        files_pattern = hook.get("files", "")
        assert re.search(r"zea/data/spec\.py", files_pattern), (
            f"run-spec-doc hook 'files' pattern does not cover zea/data/spec.py: {files_pattern!r}"
        )

    def test_run_spec_doc_triggers_on_spec_doc_py(self):
        """The hook's files pattern must match docs/source/spec_doc.py."""
        import re

        hook = self._find_hook("run-spec-doc")
        assert hook is not None
        files_pattern = hook.get("files", "")
        assert re.search(r"docs/source/spec_doc\.py", files_pattern), (
            f"run-spec-doc hook 'files' pattern does not cover docs/source/spec_doc.py: {files_pattern!r}"
        )

    def test_run_spec_doc_pass_filenames_false(self):
        """The hook must set pass_filenames: false."""
        hook = self._find_hook("run-spec-doc")
        assert hook is not None
        assert hook.get("pass_filenames") is False, (
            "run-spec-doc hook should have pass_filenames: false"
        )

    def test_run_spec_doc_language_system(self):
        """The hook must use language: system."""
        hook = self._find_hook("run-spec-doc")
        assert hook is not None
        assert hook.get("language") == "system"

    def test_existing_run_parameters_doc_hook_still_present(self):
        """Regression: the pre-existing run-parameters-doc hook must not have been removed."""
        hook = self._find_hook("run-parameters-doc")
        assert hook is not None, "run-parameters-doc hook was unexpectedly removed"

    def test_existing_generate_keras_ops_hook_still_present(self):
        """Regression: the generate-keras-ops hook must not have been removed."""
        hook = self._find_hook("generate-keras-ops")
        assert hook is not None, "generate-keras-ops hook was unexpectedly removed"


# ---------------------------------------------------------------------------
# .github/workflows/tests.yaml tests
# ---------------------------------------------------------------------------


class TestCIWorkflowHeavyTestsCondition:
    """Verify the new conditional skip added to the heavy-tests job."""

    @pytest.fixture(autouse=True)
    def ci_config(self):
        ci_path = REPO_ROOT / ".github" / "workflows" / "tests.yaml"
        assert ci_path.exists(), ".github/workflows/tests.yaml not found"
        with ci_path.open(encoding="utf-8") as fh:
            self._config = yaml.safe_load(fh)

    def test_heavy_tests_job_exists(self):
        """The heavy-tests job must exist in the workflow."""
        jobs = self._config.get("jobs", {})
        assert "heavy-tests" in jobs, "heavy-tests job not found in tests.yaml"

    def test_heavy_tests_has_if_condition(self):
        """The heavy-tests job must have an 'if' condition."""
        heavy_job = self._config["jobs"]["heavy-tests"]
        assert "if" in heavy_job, "heavy-tests job missing 'if' condition"

    def test_heavy_tests_if_excludes_openh_rf(self):
        """The 'if' condition must skip the job when base_ref is 'openh-rf'."""
        heavy_job = self._config["jobs"]["heavy-tests"]
        if_condition = str(heavy_job.get("if", ""))
        assert "openh-rf" in if_condition, (
            f"heavy-tests 'if' condition does not reference 'openh-rf': {if_condition!r}"
        )

    def test_heavy_tests_if_uses_base_ref(self):
        """The 'if' condition must reference github.base_ref."""
        heavy_job = self._config["jobs"]["heavy-tests"]
        if_condition = str(heavy_job.get("if", ""))
        assert "base_ref" in if_condition, (
            f"heavy-tests 'if' condition does not reference base_ref: {if_condition!r}"
        )

    def test_regular_tests_job_has_no_if_condition(self):
        """The regular 'tests' job must NOT have an 'if' condition (it always runs)."""
        tests_job = self._config["jobs"].get("tests", {})
        assert "if" not in tests_job, (
            "Regular 'tests' job should run unconditionally, but has an 'if' condition"
        )
