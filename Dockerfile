# base image
FROM ubuntu:22.04

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
RUN apt-get install -y python3 python3-pip git python3-tk pipx \
                       libsm6 libxext6 libxrender-dev libqt5gui5 \
                       ffmpeg imagemagick && \
    python3 -m pip install pip -U && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    ln -s /usr/bin/python3 /usr/bin/python

# Install poetry and set environment variables
RUN pipx install poetry==1.8.3
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_CACHE_DIR=/opt/.cache

# Set working directory
WORKDIR /ultrasound-toolbox
COPY . /ultrasound-toolbox/

# Create a symbolic link to the ultrasound-toolbox directory
RUN mkdir /usbmd && ln -s /ultrasound-toolbox /usbmd

ARG KERAS3=False
# Install additional packages if KERAS3=True
RUN --mount=type=cache,target=$POETRY_CACHE_DIR if [ "$KERAS3" = "True" ]; then \
        poetry install; \
    else \
        poetry install --with torch,tensorflow,jax; \
    fi