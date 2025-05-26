const fs = require("fs");
const path = require("path");
const fse = require("fs-extra");

const src = path.resolve(__dirname, "../_build/html");
const dest = path.resolve(__dirname, "../functions/_build/html");
const indexHtml = path.join(src, "index.html");

if (!fs.existsSync(indexHtml)) {
  console.error("Docs not built! Run `make docs-build` in docs/");
  process.exit(1);
}

// Remove old docs if they exist
if (fs.existsSync(dest)) {
  fse.removeSync(dest);
}

// Copy docs to functions/_build/html
fse.copySync(src, dest, { overwrite: true });

console.log("Docs copied to functions/_build/html");