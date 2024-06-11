# base image
FROM ubuntu:22.04

# Do not prompt for input when installing packages
ARG DEBIAN_FRONTEND=noninteractive

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Install python, pip, git, opencv dependencies, ffmpeg, imagemagick, and ssh keyscan github
RUN apt-get update && \
    apt-get install -y python3 python3-pip git \
                       libsm6 libxext6 libxrender-dev tk libqt5gui5 \
                       ffmpeg imagemagick sudo && \
    python3 -m pip install pip -U && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    ln -s /usr/bin/python3 /usr/bin/python

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

# Set working directory
WORKDIR /usbmd
COPY . /usbmd

# Install usbmd
RUN pip install --no-cache-dir -e .[test,linter]

ARG KERAS3=False
# Install additional packages if KERAS3=True
RUN if [ "$KERAS3" = "True" ]; then \
        pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorflow[and-cuda]==2.15.0 && \
        pip install --no-cache-dir --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html jax[cuda12_pip]==0.4.26 && \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.2.2+cu121 torchvision && \
        pip install --no-cache-dir --upgrade keras==3.1.1 && \
        pip install --no-cache-dir --upgrade keras-cv && \
        pip install wandb albumentations torchmetrics ax-platform; \
    fi