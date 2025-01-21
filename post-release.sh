#!/bin/bash

# PRIOR TO RUNNING THIS SCRIPT:
# Update usbmd version in pyproject.toml
# Update usbmd version in __init__.py

# Run this script on the linux server where the docker images are built.
# Depends on: docker, apptainer, scp, awk and poetry
# Run this script using `./post-release.sh <SNELLIUS_USER> &`

# Constants
TMP_USBMD_IMAGE_TAR=/tmp/usbmd.tar
TMP_USBMD_IMAGE_SIF=/tmp/usbmd.sif
SNELLIUS_ADDRESS=snellius.surf.nl

# Read SNELLIUS_USER from arguments
if [ -z "$1" ]; then
    echo "Please provide the username for snellius as an argument."
    exit 1
fi
SNELLIUS_USER=$1

# Check if snellius is reachable
if ssh -q -o BatchMode=yes -o ConnectTimeout=5 "$SNELLIUS_USER@$SNELLIUS_ADDRESS" exit; then
    echo "Snellius is reachable with username '$SNELLIUS_USER'."
else
    echo "Snellius is not reachable or authentication failed for username '$SNELLIUS_USER'."
fi

# Change directory to the script's location (so repo root)
cd "$(dirname "$0")"
echo "Current directory: $(pwd)"

# Update poetry lockfile
echo "Updating poetry lockfile..."
poetry lock --no-interaction

# Get usbmd version
version=$(awk -F'"' '/__version__/ {print $2}' usbmd/__init__.py)

# Build images
echo "Building docker images..."
docker build . -t usbmd/base:v$version
docker build --build-arg KERAS3=True . -t usbmd/keras3:v$version

# Tag images
docker tag usbmd/base:v$version usbmd/base:latest
docker tag usbmd/keras3:v$version usbmd/keras3:latest

# Update image on snellius
echo "Updating images on snellius..."
docker save -o $TMP_USBMD_IMAGE_TAR usbmd/keras3:latest # save docker image to file.
apptainer build $TMP_USBMD_IMAGE_SIF docker-archive://$TMP_USBMD_IMAGE_TAR # convert docker image to apptainer image
scp $TMP_USBMD_IMAGE_SIF $SNELLIUS_USER@$SNELLIUS_ADDRESS:/projects/0/prjs0966/usbmd-keras3-v$version.sif # copy apptainer image to snellius