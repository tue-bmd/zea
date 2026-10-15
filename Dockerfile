# syntax=docker/dockerfile:1
# Backends are opt-in: INSTALL_{JAX,TORCH,TF} take cpu | gpu | false (default false).
#   docker build -t zeahub/jax:latest --build-arg INSTALL_JAX=gpu .
#   docker build -t zeahub/all:latest --build-arg INSTALL_JAX=gpu \
#     --build-arg INSTALL_TORCH=gpu --build-arg INSTALL_TF=gpu .
#
# Single stage: uv installs wheels and caches to --mount=type=cache, so a builder stage
# would have nothing to strip. Installing deps before `COPY . .` gives the same layer
# caching, and keeps the backends in their own layer so the variants share the rest.

# uv's Debian image, not python:*-slim: it ships uv and no system Python, so /opt/venv
# holds the only interpreter and the only pip. One pinned tag covers both the OS and uv.
FROM ghcr.io/astral-sh/uv:0.12.1-trixie-slim

ARG DEBIAN_FRONTEND=noninteractive

# No tk package: uv's managed Python bundles Tcl/Tk, so tkinter (and matplotlib's TkAgg
# backend) already work.
# ca-certificates is not in the base image but is needed for pre-commit hooks and other HTTPS requests.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    ca-certificates \
    ffmpeg \
    make pandoc \
    openssh-client git sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# /opt/venv rather than .venv: the workspace is bind-mounted over in dev containers, which
# would shadow it. PATH + VIRTUAL_ENV let `python`/`pip` find it unaided;
# UV_PROJECT_ENVIRONMENT points `uv sync`/`uv run` at it too. UV_PYTHON_INSTALL_DIR must
# sit outside any cache mount so the venv's interpreter symlink survives.
# UV_LINK_MODE=copy: the cache mount is a separate filesystem.
# PYTHON_VERSION drives UV_PYTHON and the site-packages path in LD_LIBRARY_PATH below.
# Give it as X.Y: it also names the venv's lib/pythonX.Y directory.
ARG PYTHON_VERSION=3.12
ENV UV_PYTHON=${PYTHON_VERSION} \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    LC_ALL=C

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install

WORKDIR /zea

COPY pyproject.toml uv.lock README.md ./

# Non-backend deps from the lockfile. uv installs the `dev` group by default, so DEV=false
# opts out to keep tests/docs/lint tooling out of production images. --no-install-project:
# zea itself goes in last. Depends on DEV alone, so all backend variants share this layer.
ARG DEV=true
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project \
    $([ "$DEV" = "true" ] || echo --no-dev)

# One dependency-group per backend. Versions and CPU/CUDA wheels come from uv.lock, so a
# later `uv sync` inside the image installs the same stack -- no rebuild to pick up a
# lockfile change. --inexact keeps the layer above in place.
ARG INSTALL_JAX=false
ARG INSTALL_TORCH=false
ARG INSTALL_TF=false
RUN --mount=type=cache,target=/root/.cache/uv \
    set -e; \
    GROUPS=""; \
    [ "$INSTALL_JAX" != "false" ] && GROUPS="$GROUPS --group jax-${INSTALL_JAX}"; \
    [ "$INSTALL_TORCH" != "false" ] && GROUPS="$GROUPS --group torch-${INSTALL_TORCH}"; \
    [ "$INSTALL_TF" != "false" ] && GROUPS="$GROUPS --group tf-${INSTALL_TF}"; \
    if [ -n "$GROUPS" ]; then \
    uv sync --frozen --no-install-project --no-default-groups --inexact $GROUPS; \
    fi

# TF 2.21.0's libtensorflow_framework.so.2 lost its cusolver RUNPATH entry (2.19 had it),
# so TF silently falls back to CPU. Harmless without TF/CUDA; drop once fixed upstream.
RUN test -d "/opt/venv/lib/python${PYTHON_VERSION}/site-packages" || { \
    echo "PYTHON_VERSION=${PYTHON_VERSION} does not match the venv in /opt/venv/lib" >&2; \
    exit 1; \
    }
ENV LD_LIBRARY_PATH=/opt/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cusolver/lib

# Kept for the motd and the KERAS_BACKEND default below.
ENV INSTALL_JAX=${INSTALL_JAX} \
    INSTALL_TORCH=${INSTALL_TORCH} \
    INSTALL_TF=${INSTALL_TF} \
    DEV=${DEV}

# KERAS_BACKEND cannot be a plain ENV: it is derived from the INSTALL_* args and ENV takes
# no conditionals. zea-backend.sh holds that logic once; it is sourced from two places
# because neither hook alone covers every entry path -- ENTRYPOINT is skipped by
# `docker exec` (how dev containers open every terminal), and /etc/bash.bashrc is read only
# by interactive bash, not by `docker run <img> python ...`.
COPY scripts/zea-backend.sh /etc/zea-backend.sh
COPY scripts/entrypoint.sh /usr/local/bin/zea-entrypoint

# Message of the day, shown on every interactive shell
COPY scripts/motd.sh /etc/motd.sh
RUN chmod +x /etc/motd.sh /usr/local/bin/zea-entrypoint && \
    echo '. /etc/zea-backend.sh' >> /etc/bash.bashrc && \
    echo '[ ! -z "$TERM" -a -r /etc/motd.sh ] && KERAS_BACKEND=$KERAS_BACKEND INSTALL_JAX=$INSTALL_JAX INSTALL_TORCH=$INSTALL_TORCH INSTALL_TF=$INSTALL_TF DEV=$DEV bash /etc/motd.sh' \
    >> /etc/bash.bashrc

# Last, so a source-only change rebuilds nothing above. --no-deps: already installed.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps -e .

ENTRYPOINT ["zea-entrypoint"]
CMD ["/bin/bash"]
