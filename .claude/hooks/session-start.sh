#!/bin/bash
# Sets up a local dev environment for Claude Code on the web so linters, pre-commit,
# and pytest all work out of the box. Only runs in remote sessions — local
# contributors are expected to already have their own environment set up (see
# docs/source/contributing.rst).
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Mirrors contributing.rst; --frozen keeps this from touching uv.lock. `dev` comes along
# by default; jax-cpu is the cheapest backend group to exercise zea on a GPU-less runner.
uv sync --frozen --group jax-cpu

uv run --no-sync pre-commit install >/dev/null

# Make the venv (and its console scripts: pytest, ruff, ty, pre-commit, ...)
# available on PATH for the rest of the session, and select the JAX backend.
{
  echo "export VIRTUAL_ENV=\"$CLAUDE_PROJECT_DIR/.venv\""
  echo "export PATH=\"$CLAUDE_PROJECT_DIR/.venv/bin:\$PATH\""
  echo "export KERAS_BACKEND=jax"
} >> "$CLAUDE_ENV_FILE"
