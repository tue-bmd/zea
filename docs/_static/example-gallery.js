// Filters for the examples page (see docs/source/_ext/example_gallery.py).
// Builds a row of level buttons and a row of tag buttons from the cards,
// hides cards that do not match both, and hides a section once none of its
// cards are left. The selection is kept in ?level= and ?tag= so a filtered
// view can be linked to.
document.addEventListener("DOMContentLoaded", () => {
  const bar = document.querySelector(".example-filter");
  const cards = [...document.querySelectorAll(".example-card")];
  if (!bar || cards.length === 0) return;

  const LEVELS = ["beginner", "intermediate", "advanced"];
  const tagsOf = (card) => (card.dataset.tags || "").split("|").filter(Boolean);
  const counts = new Map();
  cards.forEach((c) => tagsOf(c).forEach((t) => counts.set(t, (counts.get(t) || 0) + 1)));
  const tags = [...counts.keys()].sort((a, b) => counts.get(b) - counts.get(a) || a.localeCompare(b));
  const levels = LEVELS.filter((l) => cards.some((c) => c.dataset.level === l));

  const state = { level: "", tag: "" };

  const row = (key, values, title) => {
    const div = document.createElement("div");
    div.className = `example-filter-row example-filter-${key}`;
    const name = document.createElement("span");
    name.className = "example-filter-label";
    name.textContent = title;
    div.appendChild(name);
    const options = document.createElement("span");
    options.className = "example-filter-options";
    div.appendChild(options);
    [["", "all"], ...values.map((v) => [v, v])].forEach(([value, label]) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      b.dataset.key = key;
      b.dataset.value = value;
      if (key === "level" && value) b.dataset.level = value;
      options.appendChild(b);
    });
    bar.appendChild(div);
  };
  row("level", levels, "Level");
  row("tag", tags, "Tags");

  const apply = () => {
    bar.querySelectorAll("button").forEach((b) => {
      b.setAttribute("aria-pressed", String(state[b.dataset.key] === b.dataset.value));
    });
    cards.forEach((c) => {
      const tagOk = !state.tag || tagsOf(c).includes(state.tag);
      const levelOk = !state.level || c.dataset.level === state.level;
      c.hidden = !(tagOk && levelOk);
    });
    document.querySelectorAll(".example-gallery").forEach((g) => {
      const section = g.closest("section");
      const empty = ![...g.querySelectorAll(".example-card")].some((c) => !c.hidden);
      if (section && section.querySelector(".example-filter") === null) section.hidden = empty;
    });
    const url = new URL(window.location);
    for (const key of ["level", "tag"]) {
      if (state[key]) url.searchParams.set(key, state[key]);
      else url.searchParams.delete(key);
    }
    history.replaceState(null, "", url);
  };

  bar.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    const { key, value } = b.dataset;
    state[key] = state[key] === value ? "" : value;
    apply();
  });

  const params = new URL(window.location).searchParams;
  if (levels.includes(params.get("level"))) state.level = params.get("level");
  if (tags.includes(params.get("tag"))) state.tag = params.get("tag");
  apply();
});
