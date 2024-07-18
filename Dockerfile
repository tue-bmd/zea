# base image
FROM ubuntu:22.04

ENV POETRY_VERSION=1.8.3
ENV POETRY_VENV=/opt/poetry-venv

# Do not prompt for input when installing packages
ARG DEBIAN_FRONTEND=noninteractive

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Install sudo
RUN apt-get update && apt-get install -y sudo

# Add non-root users
ARG BASE_UID=1000
ARG NUM_USERS=51

# Create users in a loop
RUN for i in $(seq 0 $NUM_USERS); do \
        USER_UID=$((BASE_UID + i)); \
        USERNAME="devcontainer$i"; \
        groupadd --gid $USER_UID $USERNAME && \
        useradd --uid $USER_UID --gid $USER_UID -m --shell /bin/bash $USERNAME && \
        echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
        chmod 0440 /etc/sudoers.d/$USERNAME; \
    done

# Install python, pip, git, opencv dependencies, ffmpeg, imagemagick, and ssh keyscan github
RUN apt-get install -y python3 python3-pip git python3-tk python3.10-venv \
                       libsm6 libxext6 libxrender-dev libqt5gui5 \
                       ffmpeg imagemagick && \
    python3 -m pip install pip setuptools -U && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    ln -s /usr/bin/python3 /usr/bin/python

# Creating a virtual environment just for poetry and install it with pip
RUN python3 -m venv $POETRY_VENV && \
    $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

ENV PATH="${PATH}:${POETRY_VENV}/bin"

# Set poetry environment variables
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_VIRTUALENVS_PATH=/opt/

# Set working directory
WORKDIR /ultrasound-toolbox
COPY . /ultrasound-toolbox/

ARG DEV=False
# Install additional packages if DEV=True
RUN --mount=type=cache,target=$POETRY_CACHE_DIR if [ "$DEV" = "True" ]; then \
        # Create a virtual environment for jax
        cd /ultrasound-toolbox/envs/dev-tf-jax && \
        poetry install && \
        # Create a virtual environment for torch
        cd /ultrasound-toolbox/envs/dev-torch && \
        poetry install; \
    else \
        # Just install usbmd without ml libraries
        poetry install; \
    fi

# Make VSCode discover the VENVs
ENV WORKON_HOME=$POETRY_VIRTUALENVS_PATH