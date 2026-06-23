#!/usr/bin/env bash
# Resolve a mutually-compatible jax / tensorflow / torch stack with uv and print the
# version pins for zea's Dockerfile (the `ENV JAX_VERSION=... TF_VERSION=...` block).
#
# The CUDA version is derived automatically: jax[cuda12] and tensorflow[and-cuda] only
# ship CUDA-12 wheels, so we first resolve those, read the CUDA minor they pull (via
# nvidia-cuda-runtime-cu12, e.g. 12.9 -> cu129), and point torch at the matching PyTorch
# index with --torch-backend. That way all three share one CUDA runtime instead of torch
# grabbing a newer CUDA generation and doubling the install. Nothing is installed here.
#
# Usage:
#   ./scripts/resolve_backend_versions.sh            # derive the CUDA backend from jax/tf
#   ./scripts/resolve_backend_versions.sh cu126      # or force a specific torch CUDA backend
#
# After running, copy the printed ENV block into the Dockerfile, and set the CU_BACKEND
# build-arg default there to the same tag.
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

# Pull the version out of each pin, dropping any local CUDA tag (e.g. 2.12.1+cu129 -> 2.12.1),
# since the Dockerfile re-applies the CUDA build via --torch-backend at install time.
ver() { grep -iE "^$1==" "$lock" | head -1 | sed -E 's/^[^=]+==([^ ;+]+).*/\1/'; }

cat <<EOF

# Resolved on ${CU_BACKEND} -- paste into zea/Dockerfile:
ENV JAX_VERSION=$(ver jax) \\
    TORCH_VERSION=$(ver torch) \\
    TORCHVISION_VERSION=$(ver torchvision) \\
    TORCHAUDIO_VERSION=$(ver torchaudio) \\
    TF_VERSION=$(ver tensorflow)

ARG CU_BACKEND=${CU_BACKEND}
EOF
