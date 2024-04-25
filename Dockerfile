# base image
FROM ubuntu:22.04

# Do not prompt for input when installing packages
ARG DEBIAN_FRONTEND=noninteractive

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Install python, pip, git, opencv dependencies, ffmpeg, imagemagick, and ssh keyscan github
RUN apt-get update && \
    apt-get install -y python3 python3-pip git \
                       libsm6 libxext6 libxrender-dev tk \
                       ffmpeg imagemagick sudo && \
    python3 -m pip install pip -U && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    ln -s /usr/bin/python3 /usr/bin/python

# Add non root user
ARG USERNAME=devcontainer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m --shell /bin/bash $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set working directory
WORKDIR /usbmd
COPY . /usbmd

# Install usbmd
RUN pip install --no-cache-dir -e . [test,linter]

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]