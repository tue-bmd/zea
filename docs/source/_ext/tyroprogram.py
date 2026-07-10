"""Sphinx directive that documents a `tyro <https://brentyi.github.io/tyro/>`_ CLI.

This is the tyro counterpart of ``sphinxcontrib-autoprogram`` (which documents
``argparse`` parsers). Rather than build a throwaway ``argparse.ArgumentParser``
(tyro only exposes one through the deprecated ``tyro.extras.get_parser``), this
walks tyro's own :class:`tyro._parsers.ParserSpecification` — the exact
structured representation tyro uses to build both its CLI and its ``--help``
output. The docs are therefore generated from the *same* dataclasses that define
the CLI; there is no duplicated argument definition to keep in sync.

Usage in reStructuredText::

    Command line interface
    ======================

    .. tyroprogram:: zea.__main__:CLI
       :prog: zea

The directive argument is ``module:expr`` where ``expr`` evaluates to the type or
callable you would hand to :func:`tyro.cli` — typically a dataclass or a
``Union`` of ``Annotated[..., tyro.conf.subcommand(...)]`` subcommands. Place the
directive directly under a page/section title: it emits its own nested headings
for the program and each (sub)subcommand.

Options
    ``:prog:``      Program name shown instead of ``sys.argv[0]`` (e.g. ``zea``).
    ``:maxdepth:``  Stop recursing into subcommands beyond this depth (0 = no limit).

The tyro-specific introspection is confined to :func:`_build_spec`,
:func:`_evaluate`, :func:`_iter_options` and :func:`_subcommand_names`. tyro's
parser internals are private, so if a future release changes them, only those
helpers need updating.
"""

from __future__ import annotations

import builtins
import inspect
import re
from functools import reduce
from typing import Any, Iterator

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst.directives import unchanged
from docutils.statemachine import ViewList
from sphinx.domains import std
from sphinx.util.nodes import nested_parse_with_titles

# Underline characters used for the emitted headings, one per subcommand depth.
# The directive is expected to sit directly under a page title (typically "="),
# so the program heading (depth 0) starts one level below that.
_HEADING_CHARS = ["-", "~", "^", '"', "'"]

# tyro renders flags/usage as rich text; ``str()`` of it embeds ANSI colour
# escapes when tyro detects a colour-capable environment (e.g. a TTY, as with
# ``sphinx-autobuild``). Strip them so the docs always contain plain text.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: Any) -> str:
    """Render a tyro rich-text/help value to plain, ANSI-free text."""
    return _ANSI_RE.sub("", str(text))


# ── tyro introspection (private tyro API lives here and nowhere else) ──────────


def _build_spec(f: Any):
    """Build the tyro ``ParserSpecification`` for ``f`` (a ``tyro.cli`` target)."""
    from tyro import _parsers, _strings
    from tyro._singleton import MISSING_NONPROP

    # tyro renders flags with a configurable word delimiter; "-" matches the
    # default CLI (e.g. ``--save-dir``). The context manager is required for the
    # metavar/flag strings below to come out right.
    with _strings.delimiter_context("-"):
        return _parsers.ParserSpecification.from_callable_or_type(
            f,
            markers=set(),
            description=None,
            parent_classes=set(),
            default_instance=MISSING_NONPROP,
            intern_prefix="",
            extern_prefix="",
            subcommand_prefix="",
            support_single_arg_types=False,
            prog_suffix="",
        )


def _evaluate(pspec):
    """Materialise a (possibly lazy) child parser specification."""
    from tyro import _parsers

    if isinstance(pspec, _parsers.LazyParserSpecification):
        return pspec.evaluate()
    return pspec


def _iter_options(spec) -> Iterator[tuple[str, str]]:
    """Yield ``(invocation, help)`` for every visible argument of ``spec``.

    ``invocation`` is the flag/positional display exactly as tyro renders it in
    ``--help`` (aliases, metavars, ``--x/--no-x`` boolean pairs, …). ``help``
    already carries tyro's ``(default: …)`` / ``(required)`` suffix.
    """
    for arg in spec.args:
        if arg.is_suppressed():
            continue
        help_text = _plain(arg.lowered.help() or "")
        if arg.is_positional():
            # tyro's positional metavar (e.g. "[[PATH [PATH ...]]]") is not a valid
            # option-directive token; use the field name instead.
            invocation = arg.field.extern_name.replace("_", "-")
        else:
            _short, long = arg.get_invocation_text()
            invocation = _plain(long)
        yield invocation, help_text


def _subcommand_names(spec) -> list[str]:
    """Return the ordered subcommand names directly under ``spec`` (may be empty)."""
    names: list[str] = []
    for subparsers in spec.subparsers_from_intern_prefix.values():
        names.extend(subparsers.parser_from_name.keys())
    return names


def _iter_subcommands(spec) -> Iterator[tuple[str, Any]]:
    """Yield ``(name, evaluated_child_spec)`` for each subcommand of ``spec``."""
    for subparsers in spec.subparsers_from_intern_prefix.values():
        for name, child in subparsers.parser_from_name.items():
            yield name, _evaluate(child)


# ── module import (``module:expr`` → object), mirroring autoprogram ───────────


def _import_object(import_name: str) -> Any:
    module_name, expr = import_name.split(":", 1)
    mod = __import__(module_name)
    mod = reduce(getattr, module_name.split(".")[1:], mod)
    return eval(expr, dict(vars(builtins)), vars(mod))


# ── scanning + rendering ──────────────────────────────────────────────────────


def _scan(spec, command: list[str], maxdepth: int, depth: int) -> Iterator[tuple]:
    """Depth-first walk yielding ``(command_path, depth, spec)`` for each program."""
    yield command, depth, spec
    if maxdepth and depth + 1 >= maxdepth:
        return
    for name, child in _iter_subcommands(spec):
        yield from _scan(child, command + [name], maxdepth, depth + 1)


def _usage(title: str, spec) -> str:
    """Build an argparse-style ``usage:`` line from tyro's short invocations."""
    options, positionals = [], []
    for arg in spec.args:
        if arg.is_suppressed():
            continue
        short = _plain(arg.get_invocation_text()[0])
        (positionals if arg.is_positional() else options).append(short)
    parts = [title, "[-h]", *options, *positionals]
    sub_names = _subcommand_names(spec)
    if sub_names:
        parts.append("{" + ",".join(sub_names) + "}")
    return "usage: " + " ".join(parts)


def _render(title: str, depth: int, spec) -> Iterator[str]:
    """Yield reStructuredText lines documenting a single (sub)command."""
    underline = _HEADING_CHARS[min(depth, len(_HEADING_CHARS) - 1)]

    yield ""
    yield ".. program:: " + title
    yield ""
    yield title
    yield underline * len(title)
    yield ""

    for line in inspect.cleandoc(spec.description or "").splitlines():
        yield line
    yield ""

    yield ".. code-block:: text"
    yield ""
    for usage_line in _usage(title, spec).splitlines():
        yield "   " + usage_line
    yield ""

    # ``-h/--help`` is added by tyro's backend, not the spec; document it for parity.
    options = [("-h, --help", "show this help message and exit"), *_iter_options(spec)]
    for invocation, help_text in options:
        yield ".. option:: " + invocation
        yield ""
        for help_line in (help_text or "").splitlines() or [""]:
            yield "   " + help_line
        yield ""


class TyroProgramDirective(Directive):
    has_content = False
    required_arguments = 1
    option_spec = {
        "prog": unchanged,
        "maxdepth": unchanged,
    }

    def make_rst(self) -> Iterator[str]:
        (import_name,) = self.arguments
        target = _import_object(import_name)
        prog = self.options.get("prog", "").strip() or import_name.split(":", 1)[0]
        maxdepth = int(self.options.get("maxdepth", 0))

        spec = _build_spec(target)
        for command, depth, cmd_spec in _scan(spec, [], maxdepth, 0):
            title = " ".join([prog, *command])
            yield from _render(title, depth, cmd_spec)

    def run(self):
        node = nodes.section()
        node.document = self.state.document
        result = ViewList()
        for line in self.make_rst():
            result.append(line, "<tyroprogram>")
        nested_parse_with_titles(self.state, result, node)
        return node.children


def _patch_option_role_to_allow_argument_form() -> None:
    """Let ``.. option::`` accept bare positional names (e.g. ``input-paths``).

    Same monkeypatch as ``sphinxcontrib-autoprogram``: without it Sphinx rejects
    option signatures that do not start with ``-``/``--``/``/``.
    """
    std.option_desc_re = re.compile(r"((?:/|-|--)?[-_a-zA-Z0-9]+)(\s*.*)")


def setup(app) -> dict[str, Any]:
    app.add_directive("tyroprogram", TyroProgramDirective)
    _patch_option_role_to_allow_argument_form()
    return {"parallel_read_safe": True, "parallel_write_safe": True}
