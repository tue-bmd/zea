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
# 1) Base builder: non-backend deps
##############################
FROM python:3.12-slim-bullseye AS builder-base

# Backend versions
ENV JAX_VERSION=0.5.2 \
    TORCH_VERSION=2.6.0 \
    TORCHVISION_VERSION=0.21.0 \
    TORCHAUDIO_VERSION=2.6.0 \
    TF_VERSION=2.19.0

ARG DEBIAN_FRONTEND=noninteractive
# Install non-backend dependencies into the system environment (/usr/local) with uv,
# so they can be copied to the runtime stage just like before. UV_PROJECT_ENVIRONMENT
# points uv at the system interpreter instead of creating a separate .venv.
ENV PYTHONDONTWRITEBYTECODE=1 \
    LC_ALL=C \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv

WORKDIR /zea

COPY pyproject.toml uv.lock README.md ./

# Install all non-backend dependencies from the lockfile, installing dev extras only
# if DEV is true. --no-install-project skips installing zea itself (added later as an
# editable install), and --inexact keeps pip/setuptools available for the backend wheels.
ARG DEV
RUN if [ "$DEV" = "true" ]; then \
      uv sync --frozen --no-install-project --inexact --extra dev; \
    else \
      uv sync --frozen --no-install-project --inexact; \
    fi

##############################
# 2) JAX variants
##############################
FROM builder-base AS builder-jax-cpu
WORKDIR /wheels
RUN pip wheel --no-cache-dir --wheel-dir=/wheels jax==${JAX_VERSION}

FROM builder-base AS builder-jax-gpu
WORKDIR /wheels
RUN pip wheel --no-cache-dir --wheel-dir=/wheels "jax[cuda12]==${JAX_VERSION}"

FROM builder-base AS builder-jax-false
RUN mkdir /wheels

##############################
# 3) PyTorch variants
##############################
FROM builder-base AS builder-torch-cpu
WORKDIR /wheels
RUN pip wheel --no-cache-dir --wheel-dir=/wheels \
      torch==${TORCH_VERSION}+cpu torchvision==${TORCHVISION_VERSION}+cpu torchaudio==${TORCHAUDIO_VERSION}+cpu \
      --index-url https://download.pytorch.org/whl/cpu

FROM builder-base AS builder-torch-gpu
WORKDIR /wheels
RUN pip wheel --no-cache-dir --wheel-dir=/wheels \
      torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
      --index-url https://download.pytorch.org/whl/cu124

FROM builder-base AS builder-torch-false
RUN mkdir /wheels

##############################
# 4) TensorFlow variants
##############################
FROM builder-base AS builder-tf-cpu
WORKDIR /wheels
RUN pip wheel --no-cache-dir --wheel-dir=/wheels tensorflow==${TF_VERSION}

FROM builder-base AS builder-tf-gpu
WORKDIR /wheels
RUN pip wheel --no-cache-dir --wheel-dir=/wheels \
      --extra-index-url https://pypi.nvidia.com \
      "tensorflow[and-cuda]==${TF_VERSION}"

FROM builder-base AS builder-tf-false
RUN mkdir /wheels

##############################
# 5) Pick the right variant via ARG+FROM
##############################
ARG INSTALL_JAX
ARG INSTALL_TORCH
ARG INSTALL_TF

FROM builder-jax-${INSTALL_JAX}     AS jax-selected
FROM builder-torch-${INSTALL_TORCH} AS torch-selected
FROM builder-tf-${INSTALL_TF}       AS tf-selected

##############################
# 6) Install all wheels in a single layer
##############################
FROM builder-base AS builder-backends

# Copy in all wheels from each backend
COPY --from=jax-selected   /wheels /wheels
COPY --from=torch-selected /wheels /wheels
COPY --from=tf-selected    /wheels /wheels

# preserve runtime flags
ARG INSTALL_JAX
ARG INSTALL_TORCH
ARG INSTALL_TF

# Install all wheels at once in a clean Python environment
# We install jax last to avoid a version conflict with cuda
RUN set -e; \
    for pkg in \
        $( [ "$INSTALL_TORCH" = "cpu" ] && echo "torch==${TORCH_VERSION}+cpu torchvision==${TORCHVISION_VERSION}+cpu torchaudio==${TORCHAUDIO_VERSION}+cpu" ) \
        $( [ "$INSTALL_TORCH" = "gpu" ] && echo "torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION}" ) \
        $( [ "$INSTALL_TF" = "cpu" ] && echo "tensorflow==${TF_VERSION}" ) \
        $( [ "$INSTALL_TF" = "gpu" ] && echo "tensorflow[and-cuda]==${TF_VERSION}" ) \
        $( [ "$INSTALL_JAX" = "cpu" ] && echo "jax==${JAX_VERSION}" ) \
        $( [ "$INSTALL_JAX" = "gpu" ] && echo "jax[cuda12]==${JAX_VERSION}" ); \
    do \
        pip install --no-cache-dir --no-index --find-links=/wheels $pkg; \
    done

##############################
# 7) Final runtime image
##############################
FROM python:3.12-slim-bullseye AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
      python3-tk \
      ffmpeg imagemagick \
      make pandoc \
      openssh-client git sudo && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /zea

# Copy over installed Python packages and entrypoints from builder
COPY --from=builder-backends /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder-backends /usr/local/bin /usr/local/bin

# Copy over Jupyter configuration and kernelspecs
COPY --from=builder-backends /usr/local/share/jupyter /usr/local/share/jupyter

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools

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
RUN pip install --no-deps -e .

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
COPY motd.sh /etc/motd.sh
RUN chmod +x /etc/motd.sh

CMD ["/bin/bash"]
