# syntax=docker/dockerfile:1
# By default no backend is installed. Explicitly pass build-args to enable each one.
# INSTALL_{BACKEND} accepts: cpu | gpu | false (default: false = not installed)
# Example – all backends with GPU:
# docker build -t zeahub/all:latest \
#   --build-arg INSTALL_JAX=gpu --build-arg INSTALL_TORCH=gpu --build-arg INSTALL_TF=gpu .
# Example – JAX only (GPU):
# docker build -t zeahub/jax:latest --build-arg INSTALL_JAX=gpu .
#
# Single stage on purpose. A builder stage would have nothing to strip: uv installs
# wheels rather than compiling from source, and its caches live in --mount=type=cache,
# which never lands in a layer. Ordering the dependency installs ahead of `COPY . .`
# gives the same layer caching a stage split would, while keeping the (large) backend
# install in its own layer instead of squashing it into one `COPY --from` blob -- so the
# backend image variants share their common layers in the registry.

FROM python:3.12-slim-trixie

ARG DEBIAN_FRONTEND=noninteractive

# tk8.6 (not python3-tk) supplies the Tcl/Tk shared libraries matplotlib's TkAgg backend
# needs; python3-tk would additionally pull in Debian's own Python, whose tkinter module
# this image's interpreter cannot import anyway.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    tk8.6 \
    ffmpeg \
    make pandoc \
    openssh-client git sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv

# The environment lives at /opt/venv: a fixed path outside the workspace, which is
# bind-mounted over in dev containers and would otherwise shadow a .venv next to the
# source. Putting it first on PATH and exporting VIRTUAL_ENV is what lets `python` and
# `uv pip install` find it unaided; UV_PROJECT_ENVIRONMENT points uv's project commands
# at it too, so `uv sync`/`uv run` in a bind-mounted workspace update this env instead of
# silently creating that shadow .venv. UV_LINK_MODE=copy lets uv copy out of the
# --mount=type=cache below (a separate filesystem).
ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    LC_ALL=C

WORKDIR /zea

COPY pyproject.toml uv.lock README.md ./

# Install all non-backend dependencies from the lockfile, including the dev
# dependency-group (tests + docs + lint + dev-only runtime pkgs) only if DEV is true.
# uv installs the `dev` group by default, so the DEV=false branch passes
# --no-default-groups to keep dev tooling out of the production image.
# --no-install-project skips installing zea itself (added later as an editable install).
# This layer depends on DEV alone, so all the per-backend image variants share it.
ARG DEV=true
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project \
    $([ "$DEV" = "true" ] || echo --no-default-groups)

# uv does not seed pip into the venvs it creates. Without this the base image's own
# /usr/local/bin/pip wins the PATH lookup and installs into the *system* interpreter,
# so a `pip install` inside the container would silently miss /opt/venv entirely.
# No-op for DEV=true, where the dev group already brings pip in.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install pip

# Install the selected backends, one dependency-group each. Their versions and their
# CPU/CUDA wheels are pinned by pyproject.toml + uv.lock (see the `*-cpu` / `*-gpu`
# groups and the PyTorch indexes there), so the image and a later `uv sync` inside it
# install exactly the same stack -- no rebuild needed to pick up a lockfile change.
# --inexact keeps the packages installed by the layer above in place.
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

# TF 2.21.0's libtensorflow_framework.so.2 has RUNPATH entries for every bundled CUDA lib
# except cusolver (2.19 still had it), so TF silently falls back to CPU with a "Cannot
# dlopen some GPU libraries" warning. Put that one directory on the loader path; harmless
# when TF/CUDA is not installed. Drop this once the upstream wheel is fixed.
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cusolver/lib

# preserve the build flags for the motd and the KERAS_BACKEND default below
ENV INSTALL_JAX=${INSTALL_JAX} \
    INSTALL_TORCH=${INSTALL_TORCH} \
    INSTALL_TF=${INSTALL_TF} \
    DEV=${DEV}

# KERAS_BACKEND is derived from the installed backends. It has to be set from bashrc
# rather than an ENV: `docker exec` (how dev containers open every terminal) bypasses
# both ENTRYPOINT and CMD, and ENV cannot hold a value computed from build args.
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

# Message of the day, shown on every interactive shell
COPY scripts/motd.sh /etc/motd.sh
RUN chmod +x /etc/motd.sh

# Install zea itself, last so that a source-only change rebuilds nothing above.
# Editable and --no-deps: the dependencies are already installed from the lockfile.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps -e .

CMD ["/bin/bash"]
