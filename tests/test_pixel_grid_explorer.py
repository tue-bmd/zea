"""The pixel grid explorer in the docs must reproduce zea's grid geometry.

``docs/explorer/pixel_grid_explorer.html`` re-implements the default grid of
:class:`zea.Parameters` in JavaScript. This runs that code in QuickJS against reference
values computed by the current zea, so a change to the geometry that is not carried over
to the page fails here. Skipped when the ``quickjs`` package is not installed.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

quickjs = pytest.importorskip("quickjs")

EXPLORER = Path(__file__).parents[1] / "docs" / "explorer"

COMPARE = r"""
function compare(reference) {
  const near = (a, b, rtol, atol) => Math.abs(a - b) <= atol + rtol * Math.abs(b);
  const failures = [];
  reference.forEach((fx, n) => {
    const cfg = fx.config, e = fx.expected, bad = [];
    const scan = cfg.enable_scanline ? " scanline" : "";
    const label = `#${n + 1} ${cfg.probe} ${cfg.grid_type}${scan}`;
    const R = resolve(cfg);
    if (e.scanline_error || R.scanlineError) {
      if (!e.scanline_error || !R.scanlineError) {
        failures.push(`${label}: scanline accepted by only one side`);
      }
      return;
    }
    const num = (name, got, exp, rtol = 1e-4, atol = 1e-6) => {
      const ok = Number.isFinite(got) && Number.isFinite(exp) && near(got, exp, rtol, atol);
      if (!ok) bad.push(`${name}: page ${got} zea ${exp}`);
    };
    const int = (name, got, exp) => {
      if (got !== exp) bad.push(`${name}: page ${got} zea ${exp}`);
    };
    num("grid_apex[2]", -R.d, e.grid_apex[2]);
    num("f_number", R.fNumber, e.f_number);
    e.polar_limits.forEach((v, k) => num(`polar_limits[${k}]`, R.polarLimits[k], v));
    e.xlims.forEach((v, k) => num(`xlims[${k}]`, R.xlims[k], v));
    e.zlims.forEach((v, k) => num(`zlims[${k}]`, R.zlims[k], v));
    e.rlims.forEach((v, k) => num(`rlims[${k}]`, R.rLims[k], v));
    int("grid_size_x", R.gx, e.grid_size_x);
    int("grid_size_z", R.gz, e.grid_size_z);
    [[0, 0], [1, 1], [2, 4], [3, 5]].forEach(([k, m]) => {
      num(`extent[${m}]`, R.extent[k], e.extent[m]);
    });
    Object.entries(e.grid_samples).forEach(([key, xz]) => {
      const [i, j] = key.split(",").map(Number);
      const p = R.point(i, j);
      num(`grid[${key}].x`, p[0], xz[0]);
      num(`grid[${key}].z`, p[1], xz[1]);
    });
    if (e.scan_convert) {
      e.scan_convert.x_lim.forEach((v, k) => num(`scan_convert x_lim[${k}]`, R.sc.x_lim[k], v));
      e.scan_convert.z_lim.forEach((v, k) => num(`scan_convert z_lim[${k}]`, R.sc.z_lim[k], v));
      num("scan_convert resolution", R.sc.resolution, e.scan_convert.resolution, 1e-3, 1e-7);
    }
    if (bad.length) failures.push(`${label}: ${bad.join("; ")}`);
  });
  return JSON.stringify(failures);
}
"""


def _load_reference_module():
    spec = importlib.util.spec_from_file_location("make_reference", EXPLORER / "make_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _page_script():
    html = (EXPLORER / "pixel_grid_explorer.html").read_text()
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert len(scripts) == 1, "expected one inline script in the explorer"
    return scripts[0]


def test_explorer_script_parses():
    context = quickjs.Context()
    context.set("SRC", _page_script())
    assert context.eval("try { new Function(SRC); 'ok' } catch (e) { String(e) }") == "ok"


def test_explorer_matches_zea():
    script = _page_script()
    # Only the ported geometry is needed; the rest of the script drives the page.
    port = script[script.index("const f32") : script.index("/* ======================= end of")]
    reference = _load_reference_module().reference()

    context = quickjs.Context()
    context.eval(port + COMPARE)
    context.set("REFERENCE", json.dumps(reference))
    failures = json.loads(context.eval("compare(JSON.parse(REFERENCE))"))
    assert not failures, "\n".join(failures)
