"""Sphinx directive that renders a paper reference from a BibTeX key.

``zea`` builds on a lot of published work, and every operation, model and
dataset that comes from a paper should say so in the same way.  Before this
directive that attribution was written by hand, so it drifted: some references
sat in ``.. admonition:: Reference``, others in ``.. important::`` or
``.. note::``, each with its own author/title/journal ordering and its own way
of linking the DOI.  The ``citation`` directive replaces all of those with a
single block whose content comes from a ``.bib`` file, so a reference is
written once and rendered identically everywhere it is cited.

Usage in reStructuredText (or in a docstring, which autodoc feeds through the
same parser)::

    .. citation:: luijten2020adaptive

    .. citation:: bottenus2018recovery, ali2020extending

       See the `REFoCUS <https://github.com/nbottenus/REFoCUS>`_ repository for
       a reference implementation.

Keys are resolved against every file listed in :confval:`bibtex_bibfiles` --
:mod:`sphinxcontrib.bibtex`'s own setting -- so ``.. citation::`` and
``:cite:p:`` share one pool of references.  ``zea`` keeps two files there:
``paper/paper.bib`` (the JOSS paper and the papers-using-zea list) and
``docs/source/references.bib`` (everything cited from the API docs).  An
unknown key is a build warning, which is what keeps typos and silently dropped
entries from reaching Read the Docs.

Options
    ``:title:``  Admonition title. Defaults to ``Reference`` for a single key
                 and ``References`` for several.

The rendered block is a ``citation-admonition``; ``docs/_static/custom.css``
styles it.  Entry formatting is deliberately close to the IEEE style the
docstrings already used: authors as initials + surname, quoted title,
italicised journal or proceedings, then volume/issue/pages/year and a link.
"""

from __future__ import annotations

import codecs
import re
from pathlib import Path
from typing import Iterable

import latexcodec  # noqa: F401  (registers the "ulatex" codec used below)
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst.directives import unchanged
from pybtex.database import Entry, Person, parse_file
from sphinx.util import logging

logger = logging.getLogger(__name__)

#: Cache of merged BibTeX databases, keyed by the Sphinx source directory. The
#: ``.bib`` files are parsed once per build rather than once per directive --
#: ``zea`` cites a few dozen papers across the API docs.
_BIB_CACHE: dict[str, dict[str, Entry]] = {}

#: Fields tried in order when building the entry's hyperlink.
_URL_FIELDS = ("doi", "url", "eprint")


def _clean(text: str) -> str:
    """Turn a raw BibTeX field into display text.

    Decodes LaTeX escapes (``{\\~{a}}`` -> ``ã``), drops the braces that BibTeX
    styles use to protect capitalisation (``{D}eep`` -> ``Deep``), and turns
    ``--`` page ranges into en dashes.
    """
    try:
        text = codecs.decode(text, "ulatex")
    except (UnicodeDecodeError, ValueError):
        # A field latexcodec cannot decode is still better shown as-is than
        # dropped; the braces are stripped below either way.
        pass
    text = text.replace("{", "").replace("}", "")
    text = text.replace("---", "—").replace("--", "–")
    return " ".join(text.split())


def _initials(name: str) -> str:
    """Initialise one given name, keeping compounds intact (``Jean-Luc`` -> ``J.-L.``).

    A name BibTeX already spells as compact initials keeps every letter
    (``J.A.`` -> ``J. A.``): splitting on spaces and hyphens alone would treat
    it as a single word and drop everything after the first initial.
    """
    parts = []
    for chunk in re.split(r"[-\s]+", name):
        if not chunk:
            continue
        letters = [piece[0] for piece in chunk.split(".") if piece]
        if letters:
            parts.append(" ".join(f"{letter}." for letter in letters))
    return "-".join(parts)


def _format_person(person: Person) -> str:
    """Format one author as ``F. M. van der Surname``."""
    initials = [
        _initials(_clean(name))
        for name in list(person.first_names) + list(person.middle_names)
        if _clean(name)
    ]
    surname = " ".join(
        _clean(part) for part in list(person.prelast_names) + list(person.last_names)
    )
    return " ".join(initials + [surname]).strip()


def _format_authors(persons: Iterable[Person], max_names: int = 8) -> str:
    """Join authors, abbreviating long lists with ``et al.``.

    Author lists in ultrasound papers run long (CAMUS has fourteen), and a
    fourteen-name list crowds out the title it is meant to introduce. Lists
    longer than ``max_names`` are cut to the first six followed by ``et al.``.
    """
    persons = list(persons)
    # BibTeX spells an abridged author list as "... and others"; pybtex hands
    # that back as a Person named "others".
    truncated = bool(persons) and _format_person(persons[-1]).lower() == "others"
    if truncated:
        persons = persons[:-1]

    names = [_format_person(person) for person in persons]
    if not names:
        return ""
    if truncated:
        return ", ".join(names) + ", et al."
    if len(names) > max_names:
        return ", ".join(names[:6]) + ", et al."
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + ", and " + names[-1]


def _link_target(fields: dict) -> tuple[str, str] | None:
    """Return the ``(url, label)`` to link an entry with, if it has one.

    DOIs registered by arXiv (``10.48550/arXiv.XXXX.YYYYY``) are rewritten to
    the arXiv abstract page: it is the page a reader actually wants, and it is
    the form the docstrings used before this directive existed.
    """
    for field in _URL_FIELDS:
        value = fields.get(field, "").strip()
        if not value:
            continue
        if field == "doi":
            if value.lower().startswith("10.48550/arxiv."):
                arxiv_id = value.split(".", 2)[-1]
                return f"https://arxiv.org/abs/{arxiv_id}", f"arXiv:{arxiv_id}"
            return f"https://doi.org/{value}", f"doi.org/{value}"
        if field == "eprint":
            archive = fields.get("archiveprefix", "arXiv").strip() or "arXiv"
            if archive.lower() == "arxiv":
                return f"https://arxiv.org/abs/{value}", f"arXiv:{value}"
            continue
        return value, value
    return None


def _venue(entry: Entry) -> tuple[str, bool]:
    """Return the entry's venue and whether it needs an ``in`` prefix."""
    fields = entry.fields
    if entry.type in ("inproceedings", "conference", "incollection"):
        return fields.get("booktitle", ""), True
    for field in ("journal", "publisher", "school", "institution", "howpublished"):
        if fields.get(field):
            return fields[field], False
    return "", False


def _entry_nodes(key: str, entry: Entry) -> nodes.paragraph:
    """Render one BibTeX entry as a paragraph of docutils nodes.

    Nodes are built directly rather than by generating reStructuredText and
    re-parsing it: titles routinely contain ``*``, backslashes and backticks,
    all of which would need escaping on the way through the parser.
    """
    fields = {name.lower(): value for name, value in entry.fields.items()}
    para = nodes.paragraph()
    para["classes"].append("citation-entry")

    authors = _format_authors(entry.persons.get("author") or entry.persons.get("editor") or [])
    if authors:
        para += nodes.Text(f"{authors}, ")

    title = _clean(fields.get("title", key))
    # Books and theses are the work itself, so their title is set in italics;
    # for papers the title is quoted and the venue is italicised instead.
    if entry.type in ("book", "phdthesis", "mastersthesis"):
        para += nodes.emphasis(text=title)
        para += nodes.Text(". ")
    else:
        para += nodes.Text("“")
        para += nodes.Text(title)
        para += nodes.Text(",” ")

    venue, needs_in = _venue(entry)
    if venue:
        if needs_in:
            para += nodes.Text("in ")
        para += nodes.emphasis(text=_clean(venue))

    details = []
    if fields.get("volume"):
        details.append(f"vol. {_clean(fields['volume'])}")
    if fields.get("number"):
        details.append(f"no. {_clean(fields['number'])}")
    if fields.get("pages"):
        pages = _clean(fields["pages"])
        label = "pp." if "–" in pages or "," in pages else "p."
        details.append(f"{label} {pages}")
    if fields.get("year"):
        details.append(_clean(fields["year"]))

    if details:
        para += nodes.Text(f"{', ' if venue else ''}{', '.join(details)}")
    para += nodes.Text(".")

    note = fields.get("note", "").strip()
    # dblp-sourced entries carry a "Conference Name: ..." note that just
    # repeats the journal; anything else the note says is worth showing.
    if note and not note.lower().startswith("conference name:"):
        para += nodes.Text(f" {_clean(note)}.")

    link = _link_target(fields)
    if link:
        url, label = link
        para += nodes.Text(" ")
        reference = nodes.reference("", "", internal=False, refuri=url)
        reference += nodes.Text(label)
        para += reference

    return para


def _clear_bibliography_cache(app) -> None:
    """Drop the cached bibliography at the start of every build.

    The cache exists to parse each ``.bib`` file once per build rather than
    once per directive, so it must not outlive the build that filled it.
    ``sphinx-autobuild`` (``make docs-serve``) rebuilds in the same process
    with the same source directory, and without this a ``.bib`` edit would
    keep rendering from the copy read at start-up.
    """
    _BIB_CACHE.pop(str(app.srcdir), None)


def _load_bibliography(app) -> dict[str, Entry]:
    """Parse every file in :confval:`bibtex_bibfiles` into one key -> entry map.

    Duplicate keys across files are reported: sphinxcontrib-bibtex requires
    keys to be unique across the whole bibliography, so a duplicate would make
    ``:cite:`` ambiguous even though this directive could resolve it.
    """
    cache_key = str(app.srcdir)
    if cache_key in _BIB_CACHE:
        return _BIB_CACHE[cache_key]

    entries: dict[str, Entry] = {}
    for bibfile in app.config.bibtex_bibfiles:
        path = Path(app.srcdir) / bibfile
        if not path.is_file():
            logger.warning("citation: bibtex file not found: %s", path, type="citation")
            continue
        try:
            database = parse_file(str(path))
        except Exception as exc:  # pybtex raises a variety of parse errors
            logger.warning("citation: could not parse %s: %s", path, exc, type="citation")
            continue
        for key, entry in database.entries.items():
            if key in entries:
                logger.warning(
                    "citation: duplicate bibtex key %r (also in %s)", key, path, type="citation"
                )
            entries[key] = entry

    _BIB_CACHE[cache_key] = entries
    return entries


class CitationDirective(Directive):
    """``.. citation:: key[, key...]`` -- a reference block built from BibTeX."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    has_content = True
    option_spec = {"title": unchanged}

    def run(self):
        keys = [key.strip() for key in self.arguments[0].replace(",", " ").split() if key.strip()]
        bibliography = _load_bibliography(self.state.document.settings.env.app)

        admonition = nodes.admonition()
        admonition["classes"] = ["admonition", "citation-admonition"]

        title = self.options.get("title") or ("Reference" if len(keys) == 1 else "References")
        admonition += nodes.title(title, title)

        for key in keys:
            entry = bibliography.get(key)
            if entry is None:
                logger.warning(
                    "citation: unknown bibtex key %r; add it to one of the files in "
                    "bibtex_bibfiles",
                    key,
                    location=(self.state.document["source"], self.lineno),
                    type="citation",
                )
                problem = nodes.paragraph()
                problem["classes"].append("citation-entry")
                problem += nodes.problematic("", f"Unknown citation key: {key}")
                admonition += problem
                continue
            admonition += _entry_nodes(key, entry)

        if self.content:
            self.state.nested_parse(self.content, self.content_offset, admonition)

        return [admonition]


def setup(app):
    """Register the ``citation`` directive."""
    # sphinxcontrib.bibtex owns ``bibtex_bibfiles``; this directive only reads
    # it, so make sure it exists even if that extension loads after this one.
    app.setup_extension("sphinxcontrib.bibtex")
    app.add_directive("citation", CitationDirective)
    app.connect("builder-inited", _clear_bibliography_cache)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
