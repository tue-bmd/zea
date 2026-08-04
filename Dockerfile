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
FROM python:3.12-slim-trixie AS builder

ARG DEBIAN_FRONTEND=noninteractive
# Install into a venv at /opt/venv: a single self-contained tree the runtime stage copies
# in one go, at a fixed path outside any bind-mounted workspace. UV_LINK_MODE=copy lets uv
# copy out of the --mount=type=cache on each uv RUN (a separate filesystem).
ENV PYTHONDONTWRITEBYTECODE=1 \
    LC_ALL=C \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv

WORKDIR /zea

COPY pyproject.toml uv.lock README.md ./

# Install all non-backend dependencies from the lockfile, including the dev
# dependency-group (tests + docs + lint + dev-only runtime pkgs) only if DEV is true.
# uv installs the `dev` group by default, so the DEV=false branch passes
# --no-default-groups to keep dev tooling out of the production image.
# --no-install-project skips installing zea itself (added later as an editable install).
# uv does not seed pip into the venv; the dev group carries it, so only non-dev images
# need it installed explicitly. This layer depends on DEV alone, so all the per-backend
# image variants share it.
ARG DEV
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$DEV" = "true" ]; then \
    uv sync --frozen --no-install-project --group dev; \
    else \
    uv sync --frozen --no-install-project --no-default-groups && \
    uv pip install --python /opt/venv/bin/python pip; \
    fi

# Install the selected backends, one dependency-group each. Their versions and their
# CPU/CUDA wheels are pinned by pyproject.toml + uv.lock (see the `*-cpu` / `*-gpu`
# groups and the PyTorch indexes there), so the image and a later `uv sync` inside it
# install exactly the same stack -- no rebuild needed to pick up a lockfile change.
# --inexact keeps the packages installed by the layer above (and pip) in place.
ARG INSTALL_JAX
ARG INSTALL_TORCH
ARG INSTALL_TF
RUN --mount=type=cache,target=/root/.cache/uv \
    set -e; \
    GROUPS=""; \
    [ "$INSTALL_JAX" != "false" ] && GROUPS="$GROUPS --group jax-${INSTALL_JAX}"; \
    [ "$INSTALL_TORCH" != "false" ] && GROUPS="$GROUPS --group torch-${INSTALL_TORCH}"; \
    [ "$INSTALL_TF" != "false" ] && GROUPS="$GROUPS --group tf-${INSTALL_TF}"; \
    if [ -n "$GROUPS" ]; then \
    uv sync --frozen --no-install-project --no-default-groups --inexact $GROUPS; \
    fi

##############################
# 2) Final runtime image
##############################
FROM python:3.12-slim-trixie AS runtime

# tk8.6 (not python3-tk) supplies the Tcl/Tk shared libraries matplotlib's TkAgg backend
# needs; python3-tk would additionally pull in Debian's own Python, whose tkinter module
# this image's interpreter cannot import anyway.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    tk8.6 \
    ffmpeg imagemagick \
    make pandoc \
    openssh-client git sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /zea

# The environment is one self-contained tree, so a single copy brings over site-packages,
# console scripts and Jupyter kernelspecs alike.
COPY --from=builder /opt/venv /opt/venv
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv

# Putting the venv first on PATH and exporting VIRTUAL_ENV is what lets `python`, `pip` and
# `uv pip install` find it unaided; UV_PROJECT_ENVIRONMENT points uv's project commands at
# it too, so `uv sync`/`uv run` in a bind-mounted workspace update this env instead of
# silently creating a shadow .venv next to the source.
ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy

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

# TF 2.21.0's libtensorflow_framework.so.2 has RUNPATH entries for every bundled CUDA lib
# except cusolver (2.19 still had it), so TF silently falls back to CPU with a "Cannot
# dlopen some GPU libraries" warning. Put that one directory on the loader path; harmless
# when TF/CUDA is not installed. Drop this once the upstream wheel is fixed.
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cusolver/lib

# Install zea

# Copy source code to /zea (needed for editable install)
COPY . .
# in editable mode WITHOUT installing dependencies (which are already installed by uv)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps -e .

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
