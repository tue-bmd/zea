"""Cut a release from the changelog and render its release notes.

Used by the release workflows; see the comment at the top of the changelog for
the layout it relies on.

    # Move the "Current" section under a new "v0.1.9 (YYYY-MM-DD)" heading and
    # start a fresh, empty "Current" section. Pre-releases (0.2.0rc1) leave the
    # file untouched: their changes stay under "Current" until the final release.
    python .github/scripts/changelog.py release 0.1.9

    # Print the changes of a version as Markdown, for a PR or release body. A
    # version that has no section yet (a pre-release) prints "Current". API
    # references link to the docs of `--docs-version` (default: v0.1.9).
    python .github/scripts/changelog.py notes 0.1.9 [--docs-version latest]

    # Check that "Current" has an entry that the changelog of the base branch
    # (here saved to base.rst) does not have, ending with its PR and author.
    # Used by the Changelog workflow.
    python .github/scripts/changelog.py check base.rst

    # Check that every entry in the changelog, old or new, ends with its PR and
    # author. Used by the Changelog workflow, also on PRs that need no new entry.
    python .github/scripts/changelog.py lint
"""

import argparse
import datetime
import pathlib
import re
import sys
import urllib.request
import zlib

CHANGELOG = pathlib.Path(__file__).resolve().parents[2] / "docs" / "source" / "changelog.rst"
SUBSECTIONS = ("New Features", "Changed", "Fixed")
CURRENT = "Current"

DOCS_URL = "https://zea.readthedocs.io/en/{version}/"
# `latest` builds from `main`, so it knows every object a release can mention.
INVENTORY_URL = DOCS_URL.format(version="latest") + "objects.inv"
# The inventory types each Sphinx role can resolve to; None accepts any Python object.
ROLE_TYPES = {
    "class": ("py:class",),
    "func": ("py:function", "py:method"),
    "meth": ("py:method",),
    "mod": ("py:module",),
    "attr": ("py:attribute", "py:property"),
    "exc": ("py:exception",),
    "data": ("py:data",),
    "const": ("py:data",),
    "obj": None,
    "ref": ("std:label",),
    "doc": ("std:doc",),
}

# A title line followed by an underline of at least the same length.
HEADING = re.compile(r"^(?P<title>\S.*)\n(?P<underline>([-^])\3+)$", re.MULTILINE)
VERSION = re.compile(r"\d+\.\d+\.\d+((a|b|rc)\d+)?")
PRE_RELEASE = re.compile(r"\d+\.\d+\.\d+(a|b|rc)\d+")
# An RST comment: ".." alone or followed by text, but not a directive (".. note::"),
# link target (".. _label:") or footnote (".. [1]").
# The end of an entry: "(:gh:`681` by :user:`handle`)", with more PRs or authors joined
# by "and" or ", ", e.g. "(:gh:`1` and :gh:`2` by :user:`a`, :gh:`3` by :user:`b`)".
_GROUP = r":gh:`\d+`(?:(?:, | and ):gh:`\d+`)* by :user:`[^`]+`(?:(?:, | and ):user:`[^`]+`)*"
ATTRIBUTION = re.compile(rf"\({_GROUP}(?:, {_GROUP})*\)$")
COMMENT = re.compile(r"\.\.(?:\s+(?!_|\[)(?!\S+::).*)?")
CODE_BLOCK = re.compile(r"\.\.\s+(?:code-block|code|sourcecode)::.*")


def heading(title, char):
    return f"{title}\n{char * len(title)}\n"


def split_sections(text):
    """Split the changelog into its preamble and a list of (title, body) sections."""
    matches = [m for m in HEADING.finditer(text) if m.group("underline")[0] == "-"]
    if not matches:
        sys.exit(f"no sections found in {CHANGELOG}")
    sections = []
    for match, next_match in zip(matches, matches[1:] + [None]):
        end = next_match.start() if next_match else len(text)
        sections.append((match.group("title"), text[match.end() : end].strip("\n")))
    return text[: matches[0].start()], sections


def join_sections(preamble, sections):
    parts = [preamble.rstrip("\n") + "\n\n\n"]
    for title, body in sections:
        parts.append(heading(title, "-") + ("\n" + body + "\n" if body else "") + "\n\n")
    return "".join(parts).rstrip("\n") + "\n"


def find_section(sections, title):
    for index, (section_title, body) in enumerate(sections):
        if section_title == title:
            return index, body
    return None, None


def find_version(sections, version):
    """The body of the section of `version`, whose title is "vX.Y.Z (date)"."""
    for section_title, body in sections:
        if section_title.split(" ")[0] == f"v{version}":
            return body
    return None


def drop_empty_subsections(body):
    """Remove subsections without any content, e.g. a "Fixed" with no entries."""
    matches = list(HEADING.finditer(body))
    if not matches:
        return body
    kept = [body[: matches[0].start()].strip("\n")]
    for match, next_match in zip(matches, matches[1:] + [None]):
        end = next_match.start() if next_match else len(body)
        content = body[match.end() : end].strip("\n")
        if content:
            kept.append(match.group(0) + "\n" + content)
    return "\n\n".join(part for part in kept if part)


def strip_blocks(body, start):
    """Remove each block whose first line matches `start`, e.g. a comment or code block."""
    kept, block_indent = [], None
    for line in body.splitlines():
        indent = len(line) - len(line.lstrip())
        # A block runs on over blank lines and lines indented deeper than its "..".
        if block_indent is not None and (not line.strip() or indent > block_indent):
            continue
        block_indent = indent if start.fullmatch(line.strip()) else None
        if block_indent is None:
            kept.append(line)
    return "\n".join(kept)


def strip_comments(body):
    """Remove RST comments, e.g. commented-out draft entries, which never render."""
    return strip_blocks(body, COMMENT)


def entries(body):
    """The bullets of a section, each on one line, ignoring comments and code examples."""
    body = strip_blocks(strip_comments(body), CODE_BLOCK)
    found, in_bullet = [], False
    for line in body.splitlines():
        if re.match(r"\s*- \S", line):
            found.append(line.strip())
            in_bullet = True
        elif in_bullet and line[:1].isspace() and line.strip():
            # An indented line right below a bullet continues it.
            found[-1] += " " + line.strip()
        else:
            in_bullet = False
    return found


def release(version, date):
    text = CHANGELOG.read_text()
    preamble, sections = split_sections(text)

    if PRE_RELEASE.fullmatch(version):
        print(f"{version} is a pre-release; leaving {CHANGELOG.name} unchanged")
        return
    if find_version(sections, version) is not None:
        sys.exit(f"{CHANGELOG.name} already has a section for v{version}")
    index, body = find_section(sections, CURRENT)
    if index is None:
        sys.exit(f"{CHANGELOG.name} has no {CURRENT!r} section")
    body = strip_comments(body)
    if not entries(body):
        sys.exit(f"the {CURRENT!r} section of {CHANGELOG.name} has no entries to release")

    empty_current = "\n".join(heading(title, "^") for title in SUBSECTIONS).rstrip("\n")
    sections[index : index + 1] = [
        (CURRENT, empty_current),
        (f"v{version} ({date})", drop_empty_subsections(body)),
    ]
    CHANGELOG.write_text(join_sections(preamble, sections))
    print(f"moved {CURRENT!r} under v{version} ({date}) in {CHANGELOG.name}")


def load_inventory(url=INVENTORY_URL):
    """Map each object in a Sphinx `objects.inv` to its (type, uri, display name)s."""
    # Read the Docs answers 403 to urllib's default User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": "zea-changelog"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
        # Four header lines, then zlib-compressed "name type priority uri display-name" lines.
        lines = zlib.decompress(data.split(b"\n", 4)[4]).decode().splitlines()
    except (OSError, IndexError, zlib.error, UnicodeDecodeError) as error:
        print(
            f"warning: could not read {url} ({error}); API references will not link",
            file=sys.stderr,
        )
        return {}
    inventory = {}
    for line in lines:
        match = re.fullmatch(
            r"(?P<name>.+?)\s+(?P<type>\S+:\S+)\s+-?\d+\s+(?P<uri>\S*)\s+(?P<display>.*)", line
        )
        if match is None:
            continue
        name = match.group("name")
        uri = match.group("uri")
        uri = uri[:-1] + name if uri.endswith("$") else uri
        display = name if match.group("display") == "-" else match.group("display")
        inventory.setdefault(name, []).append((match.group("type"), uri, display))
    return inventory


def resolve(inventory, role, target):
    """The (uri, display name) `role` resolves `target` to, or None."""
    allowed = ROLE_TYPES[role]
    for object_type, uri, display in inventory.get(target, []):
        if (object_type in allowed) if allowed else object_type.startswith("py:"):
            return uri, display
    return None


def convert_role(match, inventory, docs_url):
    """Render a cross-reference as a Markdown link to the docs, like Sphinx shows it."""
    role, content = match.group("role"), match.group("content")
    explicit = re.fullmatch(r"(?P<title>.+?)\s*<(?P<target>[^>]+)>", content)
    if explicit:
        title, target = explicit.group("title"), explicit.group("target")
    elif content.startswith("~"):
        target = content[1:]
        title = target.rsplit(".", 1)[-1]
    else:
        title = target = content

    found = resolve(inventory, role, target)
    if found is None:
        if inventory:
            print(
                f"warning: {match.group(0)} is not in the docs; it will not link", file=sys.stderr
            )
    elif not explicit and role in ("ref", "doc"):
        title = found[1]
    text = title if role in ("ref", "doc") else f"`{title}`"
    return f"[{text}]({docs_url}{found[0]})" if found else text


def rst_to_markdown(body, inventory, docs_url):
    """Convert the subset of RST used in the changelog to GitHub Markdown."""
    # Headings first, while their underlines still match.
    body = HEADING.sub(lambda m: f"### {m.group('title')}", drop_empty_subsections(body))
    # GitHub links "#123" and "@handle" by itself.
    body = re.sub(r":gh:`(\d+)`", r"#\1", body)
    body = re.sub(r":user:`([^`]+)`", r"@\1", body)
    # Cross-references link to the API reference or doc page they point at.
    body = re.sub(
        rf":(?:py:)?(?P<role>{'|'.join(ROLE_TYPES)}):`(?P<content>[^`]+)`",
        lambda m: convert_role(m, inventory, docs_url),
        body,
    )
    body = re.sub(r"(?<!\])`([^`<]+?)\s*<([^>]+)>`__?", r"[\1](\2)", body)
    body = re.sub(r"``([^`]+)``", r"`\1`", body)
    return body.strip("\n") + "\n"


def notes(version, docs_version):
    _, sections = split_sections(CHANGELOG.read_text())
    body = find_version(sections, version)
    if body is None:
        if not PRE_RELEASE.fullmatch(version):
            sys.exit(f"{CHANGELOG.name} has no section for v{version}")
        _, body = find_section(sections, CURRENT)
    body = strip_comments(body or "")
    if not entries(body):
        sys.exit(f"no changelog entries for v{version}")
    docs_url = DOCS_URL.format(version=docs_version or f"v{version}")
    sys.stdout.write(rst_to_markdown(body, load_inventory(), docs_url))


def check(base_path):
    """Exit with an error unless "Current" gained an entry compared to `base_path`."""
    _, sections = split_sections(CHANGELOG.read_text())
    _, current = find_section(sections, CURRENT)
    base = pathlib.Path(base_path).read_text() if pathlib.Path(base_path).is_file() else ""
    _, base_current = (
        find_section(split_sections(base)[1], CURRENT) if base.strip() else (None, None)
    )

    current_entries = entries(current or "")
    base_entries = entries(base_current or "")
    added = set(current_entries) - set(base_entries)
    # Rewording an existing entry is not a new one, so the count has to go up too.
    if not added or len(current_entries) <= len(base_entries):
        sys.exit(f"no new entry under {CURRENT!r} in {CHANGELOG.name}")
    unattributed = sorted(entry for entry in added if not ATTRIBUTION.search(entry))
    if unattributed:
        sys.exit(
            "end each new entry with (:gh:`<PR number>` by :user:`<GitHub handle>`):\n"
            + "\n".join(f"  {entry}" for entry in unattributed)
        )
    print(f"new entries under {CURRENT!r}:")
    for entry in sorted(added):
        print(f"  {entry}")


def lint():
    """Exit with an error unless every entry in the changelog ends with its PR and author."""
    _, sections = split_sections(CHANGELOG.read_text())
    unattributed = [
        entry for _, body in sections for entry in entries(body) if not ATTRIBUTION.search(entry)
    ]
    if unattributed:
        sys.exit(
            "end each entry with (:gh:`<PR number>` by :user:`<GitHub handle>`):\n"
            + "\n".join(f"  {entry}" for entry in unattributed)
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    release_parser = commands.add_parser(
        "release", help="move 'Current' under a new version heading"
    )
    release_parser.add_argument("version", help="version without the leading v, e.g. 0.1.9")
    release_parser.add_argument(
        "--date", default=datetime.date.today().isoformat(), help="release date (default: today)"
    )
    notes_parser = commands.add_parser("notes", help="print the changes of a version as Markdown")
    notes_parser.add_argument("version", help="version without the leading v, e.g. 0.1.9")
    notes_parser.add_argument(
        "--docs-version",
        help="Read the Docs version API references link to (default: the released version, vX.Y.Z)",
    )
    check_parser = commands.add_parser(
        "check", help="check that 'Current' has a new entry compared to a base changelog"
    )
    check_parser.add_argument("base", help="the changelog of the base branch")
    commands.add_parser("lint", help="check that every entry ends with its PR and author")
    args = parser.parse_args()

    if args.command == "check":
        check(args.base)
        return
    if args.command == "lint":
        lint()
        return
    version = args.version.strip().removeprefix("v")
    if not VERSION.fullmatch(version):
        sys.exit(
            f"invalid version {version!r}: expected X.Y.Z, optionally followed by aN, bN or rcN"
        )
    if args.command == "release":
        release(version, args.date)
    else:
        notes(version, args.docs_version)


if __name__ == "__main__":
    main()
