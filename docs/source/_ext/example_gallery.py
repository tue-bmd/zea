"""Sphinx directives that render the examples page as a thumbnail gallery.

Every example gets a card with a thumbnail, its title and a few tags. A row of
tag buttons above the gallery (``docs/_static/example-gallery.js``) filters
the cards, and ``docs/_static/custom.css`` styles them.

Usage in reStructuredText::

    .. example-gallery:: notebooks/models

    .. external-example:: Ultrasound regularized beamforming with zea
       :link: https://deepinv.github.io/deepinv/auto_examples/...
       :thumbnail: deepinv_zea_ultrasound.webp
       :source: DeepInverse
       :tags: beamforming, inverse problems

``example-gallery`` makes one card per notebook in the folder, in the same
order as the ``:glob:`` toctree of that folder. Each notebook needs:

- a thumbnail at ``docs/_static/thumbnails/<notebook name>.{webp,png,jpg}``:
  the image only, with no axes, ticks, titles or colorbars, made with
  ``docs/source/make_thumbnail.py``.
- tags in the notebook metadata, under ``"zea": {"tags": [...]}``. Reuse the
  tags that are already on the examples page where possible.

``external-example`` adds a card for an example that lives outside this
repository, such as a blog post or another library's docs. It needs a
``:link:``, a ``:thumbnail:`` (a file in ``docs/_static/thumbnails/``) and
``:tags:``.

A missing thumbnail or missing tags are reported as errors at the end of the
build, which then exits with a non-zero status, so that every card on the page
looks the same.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective
from sphinx.util.osutil import relative_uri

logger = logging.getLogger(__name__)

THUMBNAIL_DIR = "thumbnails"
THUMBNAIL_EXTENSIONS = (".webp", ".png", ".jpg", ".jpeg")


def _static_dirs(app) -> list[Path]:
    return [(Path(app.confdir) / p).resolve() for p in app.config.html_static_path]


def _find_static(app, relpath: str) -> Path | None:
    for d in _static_dirs(app):
        if (d / relpath).is_file():
            return d / relpath
    return None


def _inline_markup(text: str) -> str:
    """Escape ``text`` for HTML, turning ``code`` and `code` into <code>."""
    text = html.escape(text)
    return re.sub(r"``(.+?)``|`(.+?)`", lambda m: f"<code>{m.group(1) or m.group(2)}</code>", text)


def _notebook_title(nb: dict) -> str | None:
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        for line in "".join(cell["source"]).splitlines():
            if line.startswith("#"):
                return line.lstrip("#").strip()
    return None


def _card_html(href, thumb_src, title, tags, *, source=None, external=False) -> str:
    tag_attr = html.escape("|".join(tags))
    target = ' target="_blank" rel="noopener"' if external else ""
    source_html = (
        f'<span class="example-card-source">{html.escape(source)} ↗</span>' if source else ""
    )
    tags_html = "".join(f"<span>{html.escape(t)}</span>" for t in tags)
    return (
        f'<a class="example-card" href="{html.escape(href)}" data-tags="{tag_attr}"{target}>'
        f'<span class="example-card-thumb"><img src="{html.escape(thumb_src)}" alt="" '
        'loading="lazy"></span>'
        f'<span class="example-card-title">{_inline_markup(title)}</span>'
        f"{source_html}"
        f'<span class="example-card-tags">{tags_html}</span>'
        "</a>"
    )


class _GalleryDirective(SphinxDirective):
    def _uri_from_here(self, target_uri: str) -> str:
        builder = self.env.app.builder
        return relative_uri(builder.get_target_uri(self.env.docname), target_uri)

    def _thumbnail_src(self, filename: str) -> str | None:
        relpath = f"{THUMBNAIL_DIR}/{filename}"
        path = _find_static(self.env.app, relpath)
        # A missing file is a dependency too: Sphinx then rereads this page on
        # every build until the thumbnail exists.
        self.env.note_dependency(str(path or _static_dirs(self.env.app)[0] / relpath))
        return None if path is None else self._uri_from_here(f"_static/{relpath}")

    def _problem(self, message: str) -> None:
        """Record a problem; ``_report_problems`` fails the build at the end."""
        problems = getattr(self.env, "example_gallery_problems", {})
        problems.setdefault(self.env.docname, []).append(message)
        setattr(self.env, "example_gallery_problems", problems)

    def _raw(self, text: str) -> list[nodes.Node]:
        return [nodes.raw("", text, format="html")]


class ExampleGallery(_GalleryDirective):
    """One card per notebook in a folder."""

    required_arguments = 1

    def run(self):
        folder = self.arguments[0].strip().rstrip("/")
        src = Path(self.env.srcdir) / folder
        notebooks = sorted(src.glob("*.ipynb"))
        if not notebooks:
            self._problem(f"example-gallery: no notebooks in {folder}")

        cards = []
        for nb_path in notebooks:
            self.env.note_dependency(str(nb_path))
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            stem = nb_path.stem
            rel = f"{folder}/{nb_path.name}"

            thumb = next(
                (
                    f"{stem}{ext}"
                    for ext in THUMBNAIL_EXTENSIONS
                    if _find_static(self.env.app, f"{THUMBNAIL_DIR}/{stem}{ext}")
                ),
                None,
            )
            tags = nb.get("metadata", {}).get("zea", {}).get("tags")
            title = _notebook_title(nb)
            if thumb is None:
                self._thumbnail_src(f"{stem}.webp")  # reread this page until it exists
                self._problem(
                    f"{rel}: add a thumbnail at docs/_static/{THUMBNAIL_DIR}/{stem}.webp "
                    "(the image only: no axes, labels or colorbars)"
                )
            if not tags:
                self._problem(f'{rel}: add tags to the notebook metadata: "zea": {{"tags": [...]}}')
            if title is None:
                self._problem(f"{rel}: the notebook has no markdown heading to use as title")
            if thumb is None or not tags or title is None:
                continue

            docname = f"{folder}/{stem}"
            href = self._uri_from_here(self.env.app.builder.get_target_uri(docname))
            cards.append(_card_html(href, self._thumbnail_src(thumb), title, tags))
        return self._raw('<div class="example-gallery">' + "".join(cards) + "</div>")


class ExternalExample(_GalleryDirective):
    """A card that links to an example outside this repository."""

    required_arguments = 1
    final_argument_whitespace = True
    option_spec = {
        "link": directives.uri,
        "thumbnail": directives.unchanged_required,
        "tags": directives.unchanged_required,
        "source": directives.unchanged,
    }

    def run(self):
        title = self.arguments[0]
        missing = [k for k in ("link", "thumbnail", "tags") if not self.options.get(k)]
        if missing:
            self._problem(
                f"external-example '{title}' is missing " + ", ".join(f":{k}:" for k in missing)
            )
            return []
        thumb_src = self._thumbnail_src(self.options["thumbnail"])
        if thumb_src is None:
            self._problem(
                f"external-example '{title}': thumbnail not found: "
                f"docs/_static/{THUMBNAIL_DIR}/{self.options['thumbnail']}"
            )
            return []
        tags = [t.strip() for t in self.options["tags"].split(",") if t.strip()]
        card = _card_html(
            self.options["link"],
            thumb_src,
            title,
            tags,
            source=self.options.get("source"),
            external=True,
        )
        return self._raw(card)


def _purge(app, env, docname):
    getattr(env, "example_gallery_problems", {}).pop(docname, None)


def _merge(app, env, docnames, other):
    problems = getattr(env, "example_gallery_problems", {})
    for docname, msgs in getattr(other, "example_gallery_problems", {}).items():
        if docname in docnames:
            problems[docname] = msgs
    setattr(env, "example_gallery_problems", problems)


def _report_problems(app, exception):
    """Fail the build, with one error per problem, if any card is incomplete.

    Problems are kept per page in the environment, so an incremental build
    that does not reread the page still reports them.
    """
    if exception is not None:
        return
    problems = getattr(app.env, "example_gallery_problems", {})
    for docname, msgs in sorted(problems.items()):
        for msg in msgs:
            logger.error(msg, location=docname)
    if any(problems.values()):
        logger.error(
            "Every example needs a thumbnail and tags, see 'Adding an example notebook' "
            "in docs/source/contributing.rst"
        )
        app.statuscode = 1


def setup(app):
    app.connect("env-purge-doc", _purge)
    app.connect("env-merge-info", _merge)
    app.connect("build-finished", _report_problems)
    app.add_directive("example-gallery", ExampleGallery)
    app.add_directive("external-example", ExternalExample)
    app.add_js_file("example-gallery.js")
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
