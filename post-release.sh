#!/bin/bash
# Run this script on the linux server where the docker images are built.
# Depends on: docker, apptainer, scp, awk and poetry
# Run this script using `./post-release.sh <USBMD_VERSION> <SNELLIUS_USER>`

shopt -s expand_aliases

# Constants
TMP_USBMD_IMAGE_TAR=/tmp/usbmd.tar
TMP_USBMD_IMAGE_SIF=/tmp/usbmd.sif
SNELLIUS_ADDRESS=snellius.surf.nl
PREPEND_ECHO="[post-release.sh]"
alias precho='echo $PREPEND_ECHO'

# Check if version argument is provided
if [ -z "$1" ]; then
    precho "Please provide the version number as first argument (format: vX.X.X)"
    exit 1
fi

# Check if version format is correct (vX.X.X)
if ! [[ $1 =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    precho "Version format incorrect. Should be vX.X.X (e.g., v2.1.1)"
    exit 1
fi

VERSION=$1
VERSION_WITHOUT_V="${VERSION#v}"

# previous version
PREVIOUS_VERSION=$(awk -F'"' '/__version__/ {print $2}' usbmd/__init__.py)
PREVIOUS_VERSION="v$PREVIOUS_VERSION"

# Update version in pyproject.toml
sed -i "s/^version = .*/version = \"$VERSION_WITHOUT_V\"/" pyproject.toml

# Update version in __init__.py
sed -i "s/__version__ = .*/__version__ = \"$VERSION_WITHOUT_V\"/" usbmd/__init__.py

# Shift arguments to maintain compatibility with rest of script
shift

# print here that we are updating the version
precho "Updating version from $PREVIOUS_VERSION to $VERSION"

# Read SNELLIUS_USER from arguments
if [ -z "$1" ]; then
    precho "Please provide the username for snellius as an argument."
    exit 1
fi
SNELLIUS_USER=$1

# Check if snellius is reachable
if ssh -q -o BatchMode=yes -o ConnectTimeout=5 "$SNELLIUS_USER@$SNELLIUS_ADDRESS" exit; then
    precho "Snellius is reachable with username '$SNELLIUS_USER'."
else
    precho "Snellius is not reachable or authentication failed for username '$SNELLIUS_USER'."
fi

# Change directory to the script's location (so repo root)
cd "$(dirname "$0")"
precho "Current directory: $(pwd)"

# Update poetry lockfile
precho "Updating poetry lockfile..."
if command -v poetry &> /dev/null; then
    # If poetry is installed, use it
    poetry lock --no-interaction
else
    # If poetry is not installed, use docker
    docker run --rm -v $(pwd):/ultrasound-toolbox --user "$(id -u):$(id -g)" usbmd/base:latest sudo /opt/poetry-venv/bin/python3 -m poetry lock --no-interaction
fi

# Build images
precho "Building docker images..."
docker build . -t usbmd/base:$VERSION
docker build --build-arg KERAS3=True . -t usbmd/keras3:$VERSION

# Tag images
docker tag usbmd/base:$VERSION usbmd/base:latest
docker tag usbmd/keras3:$version usbmd/keras3:latest

# Update image on snellius
precho "Updating images on snellius..."
docker save -o $TMP_USBMD_IMAGE_TAR usbmd/keras3:latest # save docker image to file.
apptainer build $TMP_USBMD_IMAGE_SIF docker-archive://$TMP_USBMD_IMAGE_TAR # convert docker image to apptainer image
scp $TMP_USBMD_IMAGE_SIF $SNELLIUS_USER@$SNELLIUS_ADDRESS:/projects/0/prjs0966/usbmd-keras3-$version.sif # copy apptainer image to snellius
