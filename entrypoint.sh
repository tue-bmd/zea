#!/bin/sh

set -e  # Exit on error
set -x  # Enable debugging

# Get uid/gid
USER_UID=$(ls -nd /home/devcontainer | awk '{print $3}')
USER_GID=$(ls -nd /home/devcontainer | awk '{print $4}')
USERNAME=devcontainer

# Modify group and user
groupmod --gid "$USER_GID" "$USERNAME"
usermod --uid "$USER_UID" --gid "$USER_GID" "$USERNAME"
chown -R "$USER_UID:$USER_GID" "/home/$USERNAME"