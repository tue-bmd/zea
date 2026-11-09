// Tag filter for the examples page (see docs/source/_ext/example_gallery.py).
// Builds one button per tag found on the cards, hides cards without the
// selected tag, and hides a section once none of its cards are left. The
// selected tag is kept in ?tag= so a filtered view can be linked to.
document.addEventListener("DOMContentLoaded", () => {
  const bar = document.querySelector(".example-filter");
  const cards = [...document.querySelectorAll(".example-card")];
  if (!bar || cards.length === 0) return;

  const tagsOf = (card) => (card.dataset.tags || "").split("|").filter(Boolean);
  const counts = new Map();
  cards.forEach((c) => tagsOf(c).forEach((t) => counts.set(t, (counts.get(t) || 0) + 1)));
  const tags = [...counts.keys()].sort((a, b) => counts.get(b) - counts.get(a) || a.localeCompare(b));

  const button = (label, value) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.dataset.tag = value;
    bar.appendChild(b);
    return b;
  };
  button("all", "");
  tags.forEach((t) => button(t, t));

  const select = (tag) => {
    bar.querySelectorAll("button").forEach((b) => {
      b.setAttribute("aria-pressed", String(b.dataset.tag === tag));
    });
    cards.forEach((c) => {
      c.hidden = Boolean(tag) && !tagsOf(c).includes(tag);
    });
    document.querySelectorAll(".example-gallery").forEach((g) => {
      const section = g.closest("section");
      const empty = ![...g.querySelectorAll(".example-card")].some((c) => !c.hidden);
      if (section && section.querySelector(".example-filter") === null) section.hidden = empty;
    });
    const url = new URL(window.location);
    if (tag) url.searchParams.set("tag", tag);
    else url.searchParams.delete("tag");
    history.replaceState(null, "", url);
  };

  bar.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) select(b.getAttribute("aria-pressed") === "true" ? "" : b.dataset.tag);
  });

  const initial = new URL(window.location).searchParams.get("tag");
  select(tags.includes(initial) ? initial : "");
});
