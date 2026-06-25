# syntax=docker/dockerfile:1
# By default no backend is installed. Explicitly pass build-args to enable each one.
# INSTALL_{BACKEND} accepts: cpu | gpu | false (default: false = not installed)
# Example – all backends with GPU:
# docker build -t zeahub/all:latest \
#   --build-arg INSTALL_JAX=gpu --build-arg INSTALL_TORCH=gpu --build-arg INSTALL_TF=gpu .
# Example – JAX only (GPU):
# docker build -t zeahub/jax:latest --build-arg INSTALL_JAX=gpu .

##############################
# 0) Declare build-time args
##############################
ARG INSTALL_JAX=false
ARG INSTALL_TORCH=false
ARG INSTALL_TF=false
ARG DEV=true

##############################
# 1) Builder: all deps (non-backend + selected backends)
##############################
FROM python:3.12-slim-bullseye AS builder

# Backend versions, to re-resolve to newer versions, run ./scripts/resolve_backend_versions.sh
# and paste its output here.
ENV JAX_VERSION=0.10.2 \
    TORCH_VERSION=2.12.1 \
    TORCHVISION_VERSION=0.27.1 \
    TORCHAUDIO_VERSION=2.11.0 \
    TF_VERSION=2.21.0

ARG CU_BACKEND=cu129

ARG DEBIAN_FRONTEND=noninteractive
# Install into the system env (/usr/local) so the runtime stage can copy it. UV_LINK_MODE=copy
# lets uv copy out of the --mount=type=cache on each uv RUN (a separate filesystem).
ENV PYTHONDONTWRITEBYTECODE=1 \
    LC_ALL=C \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv

WORKDIR /zea

COPY pyproject.toml uv.lock README.md ./

# Install all non-backend dependencies from the lockfile, installing dev extras only if
# DEV is true. --no-install-project skips installing zea itself (added later as an
# editable install), and --inexact keeps pip/setuptools available in the image.
ARG DEV
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$DEV" = "true" ]; then \
    uv sync --frozen --no-install-project --inexact --extra dev; \
    else \
    uv sync --frozen --no-install-project --inexact; \
    fi

# Install the selected backends in a single resolve. uv's --torch-backend selects the
# right PyTorch index (CPU vs CUDA), so there is no need for per-variant build stages or
# manual index URLs / +cpu suffixes. CPU vs GPU is uniform across backends per build.
ARG INSTALL_JAX
ARG INSTALL_TORCH
ARG INSTALL_TF
RUN --mount=type=cache,target=/root/.cache/uv \
    set -e; \
    case "${INSTALL_JAX}${INSTALL_TORCH}${INSTALL_TF}" in \
    *gpu*) TORCH_BACKEND="${CU_BACKEND}" ;; \
    *)     TORCH_BACKEND="cpu" ;; \
    esac; \
    PKGS=""; \
    [ "$INSTALL_TORCH" != "false" ] && PKGS="$PKGS torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION}"; \
    [ "$INSTALL_TF"  = "cpu" ] && PKGS="$PKGS tensorflow==${TF_VERSION}"; \
    [ "$INSTALL_TF"  = "gpu" ] && PKGS="$PKGS tensorflow[and-cuda]==${TF_VERSION}"; \
    [ "$INSTALL_JAX" = "cpu" ] && PKGS="$PKGS jax==${JAX_VERSION}"; \
    [ "$INSTALL_JAX" = "gpu" ] && PKGS="$PKGS jax[cuda12]==${JAX_VERSION}"; \
    if [ -n "$PKGS" ]; then \
    uv pip install --system --torch-backend="$TORCH_BACKEND" $PKGS; \
    fi

##############################
# 2) Final runtime image
##############################
FROM python:3.12-slim-bullseye AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    python3-tk \
    ffmpeg imagemagick \
    make pandoc \
    openssh-client git sudo && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /zea

# Copy over installed Python packages and entrypoints from builder (includes uv)
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy over Jupyter configuration and kernelspecs
COPY --from=builder /usr/local/share/jupyter /usr/local/share/jupyter

# preserve runtime flags
ARG INSTALL_JAX
ARG INSTALL_TORCH
ARG INSTALL_TF
ARG DEV
ENV INSTALL_JAX=${INSTALL_JAX} \
    INSTALL_TORCH=${INSTALL_TORCH} \
    INSTALL_TF=${INSTALL_TF} \
    DEV=${DEV}

ENV PYTHONDONTWRITEBYTECODE=1 \
    LC_ALL=C

# Install zea

# Copy source code to /zea (needed for editable install)
COPY . .
# in editable mode WITHOUT installing dependencies (which are already installed by uv)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps -e .

# Set KERAS_BACKEND in bashrc before motd.sh is called
RUN echo 'export KERAS_BACKEND=$( \
    if [ "$INSTALL_JAX" != "false" ]; then \
    echo jax; \
    elif [ "$INSTALL_TORCH" != "false" ]; then \
    echo torch; \
    elif [ "$INSTALL_TF" != "false" ]; then \
    echo tf; \
    else \
    echo numpy; \
    fi )' >> /etc/bash.bashrc && \
    echo '[ ! -z "$TERM" -a -r /etc/motd.sh ] && KERAS_BACKEND=$KERAS_BACKEND INSTALL_JAX=$INSTALL_JAX INSTALL_TORCH=$INSTALL_TORCH INSTALL_TF=$INSTALL_TF DEV=$DEV bash /etc/motd.sh' \
    >> /etc/bash.bashrc

# Source working/installation directory and add motd (message of the day)
COPY scripts/motd.sh /etc/motd.sh
RUN chmod +x /etc/motd.sh

CMD ["/bin/bash"]
