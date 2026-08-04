#!/usr/bin/env bash
# Check which CUDA generation jax / tensorflow / torch agree on, and print the PyTorch
# index block for zea's pyproject.toml (`[[tool.uv.index]]` + `[tool.uv.sources]`).
#
# The backend *versions* themselves are no longer produced here: they live in the
# `jax-*` / `torch-*` / `tf-*` dependency-groups in pyproject.toml and are pinned by
# uv.lock, so `uv lock --upgrade-package jax --upgrade-package tensorflow ...` is what
# moves them. The one thing uv cannot pick on its own is the CUDA generation of the
# PyTorch index, which is what this script resolves.
#
# jax[cuda12] and tensorflow[and-cuda] only ship CUDA-12 wheels, so we resolve those
# first, read the CUDA minor they pull (via nvidia-cuda-runtime-cu12, e.g. 12.9 -> cu129)
# and check that torch has wheels on the matching index. That way all three share one
# CUDA runtime instead of torch grabbing a newer CUDA generation and doubling the
# install. Nothing is installed here.
#
# Usage:
#   ./scripts/resolve_backend_versions.sh            # derive the CUDA backend from jax/tf
#   ./scripts/resolve_backend_versions.sh cu126      # or check a specific CUDA backend
set -euo pipefail

PYTHON_VERSION="3.12"
CU_BACKEND="${1:-}"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found -- install from https://docs.astral.sh/uv/" >&2
  exit 1
fi

reqs="$(mktemp)"
lock="$(mktemp)"
trap 'rm -f "$reqs" "$lock"' EXIT

compile() { uv pip compile "$1" --python-version "$PYTHON_VERSION" "${@:2}" -o "$lock" -q; }

# Derive the CUDA backend from jax/tf unless one was passed explicitly.
if [ -z "$CU_BACKEND" ]; then
  printf 'jax[cuda12]\ntensorflow[and-cuda]\n' > "$reqs"
  echo "Resolving jax/tensorflow to detect their CUDA version..." >&2
  compile "$reqs"
  cuda_ver="$(grep -iE '^nvidia-cuda-runtime-cu[0-9]+==' "$lock" | head -1 \
              | sed -E 's/.*==([0-9]+)\.([0-9]+).*/\1\2/')"
  if [ -z "$cuda_ver" ]; then
    echo "error: could not detect CUDA version from jax/tensorflow resolution" >&2
    exit 1
  fi
  CU_BACKEND="cu${cuda_ver}"
  echo "Detected CUDA backend: ${CU_BACKEND}" >&2
fi

cat > "$reqs" <<'EOF'
jax[cuda12]
tensorflow[and-cuda]
torch
torchvision
torchaudio
EOF

echo "Resolving full stack with --torch-backend=${CU_BACKEND} (python ${PYTHON_VERSION})..." >&2
compile "$reqs" --torch-backend="$CU_BACKEND"

# Informational: the newest stack this CUDA generation supports, ignoring the rest of
# zea's dependencies. `uv lock` resolves against those too, so it may land lower.
ver() { grep -iE "^$1==" "$lock" | head -1 | sed -E 's/^[^=]+==([^ ;+]+).*/\1/'; }

current="$(grep -oE 'pytorch-cu[0-9]+' "$(dirname "$0")/../pyproject.toml" | head -1 || true)"

cat <<EOF

# Newest stack available on ${CU_BACKEND} (informational -- uv.lock is the source of truth):
#   jax==$(ver jax)  tensorflow==$(ver tensorflow)
#   torch==$(ver torch)  torchvision==$(ver torchvision)  torchaudio==$(ver torchaudio)
EOF

if [ "$current" = "pytorch-${CU_BACKEND}" ]; then
  cat <<EOF
#
# pyproject.toml already points at pytorch-${CU_BACKEND}: no index change needed. To pick
# up the versions above (as far as zea's other dependencies allow), re-lock with
#   uv lock --upgrade-package jax --upgrade-package tensorflow --upgrade-package torch \\
#           --upgrade-package torchvision --upgrade-package torchaudio
EOF
else
  cat <<EOF
#
# pyproject.toml points at ${current:-<none>}; replace that index with:

[[tool.uv.index]]
name = "pytorch-${CU_BACKEND}"
url = "https://download.pytorch.org/whl/${CU_BACKEND}"
explicit = true

# ...renaming it in [tool.uv.sources] as well, then re-lock:
#   uv lock --upgrade-package torch --upgrade-package torchvision --upgrade-package torchaudio
EOF
fi
