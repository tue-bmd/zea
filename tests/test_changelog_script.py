"""Tests for .github/scripts/changelog.py, which the release workflows use.

Those workflows only run at release time, so these guard the parts a release
depends on: cutting "Current" into a version, rendering release notes, and the
check that a pull request adds an entry.
"""

import importlib.util
import sys
import zlib
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "changelog.py"
_spec = importlib.util.spec_from_file_location("changelog_script", _MODULE_PATH)
changelog = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = changelog
_spec.loader.exec_module(changelog)

PREAMBLE = """Changelog
=========

Intro text.

..
   - How-to comment with a bullet, which is not an entry.
"""

ENTRY = "- :class:`zea.ops.Normalize` clips its output (:gh:`681` by :user:`someone`)"

# A minimal inventory: Normalize is documented, anything else is not.
INVENTORY = {
    "zea.ops.Normalize": [("py:class", "_autosummary/zea.ops.html#zea.ops.Normalize", "-")]
}


def make_changelog(current_fixed=""):
    return f"""{PREAMBLE}

Current
-------

New Features
^^^^^^^^^^^^

Changed
^^^^^^^

Fixed
^^^^^
{current_fixed}


v0.1.8 (2026-10-01)
-------------------

Fixed
^^^^^
- An older fix (:gh:`637` by :user:`someone`)
"""


@pytest.fixture
def changelog_file(tmp_path, monkeypatch):
    """Point the script at a changelog in a temporary directory, and keep it offline."""
    path = tmp_path / "changelog.rst"
    monkeypatch.setattr(changelog, "CHANGELOG", path)
    monkeypatch.setattr(changelog, "load_inventory", lambda: INVENTORY)
    return path


def test_release_moves_current_under_new_version(changelog_file):
    changelog_file.write_text(make_changelog(ENTRY))
    changelog.release("0.1.9", "2026-11-05")

    _, sections = changelog.split_sections(changelog_file.read_text())
    titles = [title for title, _ in sections]
    assert titles == ["Current", "v0.1.9 (2026-11-05)", "v0.1.8 (2026-10-01)"]

    current, released = sections[0][1], sections[1][1]
    assert not changelog.entries(current)
    assert "Changed" in current  # a fresh Current has all subsections
    assert ENTRY in released
    assert "New Features" not in released  # empty subsections are dropped


def test_release_leaves_rest_of_file_unchanged(changelog_file):
    before = make_changelog(ENTRY)
    changelog_file.write_text(before)
    changelog.release("0.1.9", "2026-11-05")
    after = changelog_file.read_text()

    assert after.startswith(PREAMBLE)
    assert after.endswith(before[before.index("v0.1.8") :])


@pytest.mark.parametrize(
    "current_fixed",
    [
        pytest.param("", id="empty"),
        pytest.param("..\n   - A commented-out draft entry", id="only-comments"),
        pytest.param("  .. draft\n     - An indented commented-out entry", id="indented-comment"),
        pytest.param(".. code-block:: rst\n\n   - An example entry", id="code-block"),
    ],
)
def test_release_refuses_without_entries(changelog_file, current_fixed):
    changelog_file.write_text(make_changelog(current_fixed))
    with pytest.raises(SystemExit, match="no entries"):
        changelog.release("0.1.9", "2026-11-05")


def test_release_refuses_existing_version(changelog_file):
    changelog_file.write_text(make_changelog(ENTRY))
    with pytest.raises(SystemExit, match="already has a section"):
        changelog.release("0.1.8", "2026-11-05")


def test_pre_release_leaves_changelog_unchanged(changelog_file):
    text = make_changelog(ENTRY)
    changelog_file.write_text(text)
    changelog.release("0.2.0rc1", "2026-11-05")
    assert changelog_file.read_text() == text


def test_rst_to_markdown():
    body = (
        "Fixed\n^^^^^\n"
        "- :class:`~zea.ops.Normalize` and :func:`zea.func.missing` use ``x`` and "
        "`a link <https://example.org>`_ (:gh:`681` by :user:`someone`)"
    )
    markdown = changelog.rst_to_markdown(body, INVENTORY, "https://zea.readthedocs.io/en/v0.1.9/")
    assert markdown == (
        "### Fixed\n"
        "- [`Normalize`](https://zea.readthedocs.io/en/v0.1.9/_autosummary/zea.ops.html"
        "#zea.ops.Normalize) and `zea.func.missing` use `x` and "
        "[a link](https://example.org) (#681 by @someone)\n"
    )


def test_notes_of_released_and_pre_release_versions(changelog_file, capsys):
    changelog_file.write_text(make_changelog(ENTRY))

    changelog.notes("0.1.8", None)
    assert capsys.readouterr().out == "### Fixed\n- An older fix (#637 by @someone)\n"

    # A pre-release has no section yet, so its notes come from Current.
    changelog.notes("0.2.0rc1", "latest")
    assert "/en/latest/" in capsys.readouterr().out

    with pytest.raises(SystemExit, match="no section"):
        changelog.notes("0.1.9", None)


def test_check_requires_a_new_entry_under_current(changelog_file, tmp_path):
    base = tmp_path / "base.rst"
    base.write_text(make_changelog())

    # No change, or only a change outside Current, does not pass.
    changelog_file.write_text(make_changelog().replace("An older fix", "An edited older fix"))
    with pytest.raises(SystemExit, match="no new entry"):
        changelog.check(base)

    changelog_file.write_text(make_changelog(ENTRY))
    changelog.check(base)

    # A base branch without a changelog counts as an empty one.
    changelog.check(tmp_path / "missing.rst")

    # Rewording an entry already under Current is not a new entry; adding one is.
    base.write_text(make_changelog(ENTRY))
    changelog_file.write_text(make_changelog(ENTRY.replace("clips", "now clips")))
    with pytest.raises(SystemExit, match="no new entry"):
        changelog.check(base)
    changelog_file.write_text(make_changelog(f"{ENTRY}\n- Another fix (:gh:`700` by :user:`b`)"))
    changelog.check(base)


@pytest.mark.parametrize(
    "attribution",
    [
        "(by :user:`someone`)",
        "(#689 by @someone)",
        "(:gh:`689`)",
        "(:gh:`689` by someone)",
        "",
    ],
)
def test_check_requires_pr_and_author(changelog_file, tmp_path, attribution):
    base = tmp_path / "base.rst"
    base.write_text(make_changelog())
    changelog_file.write_text(make_changelog(f"- A fix {attribution}".rstrip()))
    with pytest.raises(SystemExit, match="end each new entry with"):
        changelog.check(base)


@pytest.mark.parametrize(
    "attribution",
    [
        "(:gh:`689` by :user:`someone`)",
        "(:gh:`689` by :user:`a` and :user:`b`)",
        "(:gh:`1` and :gh:`2` by :user:`a`, :gh:`3` by :user:`b`)",
    ],
)
def test_check_accepts_pr_and_author(changelog_file, tmp_path, attribution):
    base = tmp_path / "base.rst"
    base.write_text(make_changelog())
    changelog_file.write_text(make_changelog(f"- A fix {attribution}"))
    changelog.check(base)


def test_check_accepts_a_wrapped_entry(changelog_file, tmp_path):
    """RST lets a bullet wrap onto indented lines; the attribution may be on the last one."""
    base = tmp_path / "base.rst"
    base.write_text(make_changelog())
    changelog_file.write_text(
        make_changelog("- A long fix that wraps\n  onto the next line (:gh:`1` by :user:`a`)")
    )
    changelog.check(base)


def test_entries_join_wrapped_lines():
    body = "- One that wraps\n  onto a second line\n- Another\n\nNot part of it"
    assert changelog.entries(body) == ["- One that wraps onto a second line", "- Another"]


def test_lint_requires_every_entry_to_be_attributed(changelog_file):
    """Unlike `check`, `lint` also catches an edit that breaks an existing entry."""
    changelog_file.write_text(make_changelog(ENTRY))
    changelog.lint()

    changelog_file.write_text(make_changelog(ENTRY).replace("(:gh:`637`", "(:gh:637"))
    with pytest.raises(SystemExit, match="An older fix"):
        changelog.lint()


def test_every_entry_in_the_changelog_is_attributed():
    """The real changelog follows the format the check enforces on new entries."""
    changelog.lint()


def test_load_inventory(tmp_path):
    """Reads a Sphinx objects.inv; urllib also opens file:// URLs, so no network is needed."""
    lines = [
        "zea.ops.Normalize py:class 1 _autosummary/zea.ops.html#$ -",
        "environment std:doc -1 environment.html Environment variables",
    ]
    header = b"# Sphinx inventory version 2\n# Project: zea\n# Version: \n# zlib below\n"
    inventory_path = tmp_path / "objects.inv"
    inventory_path.write_bytes(header + zlib.compress("\n".join(lines).encode()))

    inventory = changelog.load_inventory(inventory_path.as_uri())
    assert inventory["zea.ops.Normalize"] == [
        ("py:class", "_autosummary/zea.ops.html#zea.ops.Normalize", "zea.ops.Normalize")
    ]
    assert inventory["environment"] == [("std:doc", "environment.html", "Environment variables")]

    # An unreadable inventory gives notes without links instead of failing a release.
    inventory_path.write_bytes(b"not an inventory")
    assert changelog.load_inventory(inventory_path.as_uri()) == {}
    assert changelog.load_inventory((tmp_path / "missing.inv").as_uri()) == {}


def test_command_line(changelog_file, tmp_path, monkeypatch, capsys):
    def run(*args):
        monkeypatch.setattr(sys, "argv", ["changelog.py", *args])
        changelog.main()

    changelog_file.write_text(make_changelog(ENTRY))
    base = tmp_path / "base.rst"
    base.write_text(make_changelog())

    run("check", str(base))
    run("lint")
    run("release", "v0.1.9", "--date", "2026-11-05")
    assert "v0.1.9 (2026-11-05)" in changelog_file.read_text()

    capsys.readouterr()
    run("notes", "0.1.9", "--docs-version", "latest")
    assert "/en/latest/" in capsys.readouterr().out

    with pytest.raises(SystemExit, match="invalid version"):
        run("release", "0.1.9foo")
