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

# Mirrors `uv sync --extra dev` from docs/source/contributing.rst. --frozen keeps
# this from touching uv.lock. jax[cpu] is installed separately (not part of the
# `dev` extra) as the cheapest backend to exercise `zea`, matching the JAX-only
# jobs in CI.
export UV_TORCH_BACKEND=cpu
uv sync --frozen --extra dev
uv pip install --python .venv/bin/python "jax[cpu]"

uv run --no-sync pre-commit install >/dev/null

# Make the venv (and its console scripts: pytest, ruff, ty, pre-commit, ...)
# available on PATH for the rest of the session, and select the JAX backend.
{
  echo "export VIRTUAL_ENV=\"$CLAUDE_PROJECT_DIR/.venv\""
  echo "export PATH=\"$CLAUDE_PROJECT_DIR/.venv/bin:\$PATH\""
  echo "export KERAS_BACKEND=jax"
} >> "$CLAUDE_ENV_FILE"
