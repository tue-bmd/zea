#!/bin/bash
# Run this script on the linux server where the docker images are built.
# Depends on: git, docker, apptainer, scp, awk and poetry
# 1. Make sure `./post-release.sh` is executable: `chmod +x post-release.sh`
# 2. Also make sure to checkout the usbmd version you want to release before running this script.
# 3. Also make sure your git working directory is clean (no uncommitted changes).
# 4. Run this script using `./post-release.sh <USBMD_VERSION> <SNELLIUS_USER>`
#    e.g. `./post-release.sh v2.1.1 tstevens`

shopt -s expand_aliases

TMP_USBMD_IMAGE_TAR=/tmp/usbmd.tar
TMP_USBMD_IMAGE_SIF=/tmp/usbmd.sif
SNELLIUS_ADDRESS=snellius.surf.nl
PREPEND_ECHO="[post-release.sh]"
alias precho='echo $PREPEND_ECHO'

# Check if required commands are installed
for cmd in git docker apptainer scp awk; do
    if ! command -v "$cmd" &> /dev/null; then
        precho "$cmd is not installed. Please install $cmd to proceed."
        exit 1
    fi
done

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

# Check if Git working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    precho "Error: Git working directory is not clean. Please commit or stash your changes before running this script."
    exit 1
fi

# Check if the current version in pyproject.toml matches the requested version
CURRENT_TOML_VERSION=$(grep -E "^version\s*=" pyproject.toml | sed -E 's/version\s*=\s*"([^"]+)"/\1/' | tr -d '\r\n')

# Debug output to see what's being compared with explicit length
precho "Debug: Comparing '${VERSION_WITHOUT_V}' (${#VERSION_WITHOUT_V} chars) with '${CURRENT_TOML_VERSION}' (${#CURRENT_TOML_VERSION} chars)"

# Strip all non-printing characters for a clean comparison
if [ "$VERSION_WITHOUT_V" != "$CURRENT_TOML_VERSION" ]; then
    precho "Error: Requested version ($VERSION) does not match the version in pyproject.toml (v$CURRENT_TOML_VERSION)."
    precho "Make sure you're on the correct branch/tag that matches the version you want to release."
    exit 1
fi

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
docker build -f Dockerfile.base --build-arg BACKEND=all . -t usbmd/all:latest
docker build -f Dockerfile.base --build-arg BACKEND=numpy . -t usbmd/base:latest

# Tag images
docker tag usbmd/all:latest usbmd/all:$VERSION # tag latest image with version
docker tag usbmd/base:latest usbmd/base:$VERSION # tag latest image with version

# Build private image
precho "Building private docker image..."
docker build . -t usbmd/private:$VERSION
docker tag usbmd/private:$VERSION usbmd/private:latest

# Update image on snellius
precho "Updating images on snellius..."
docker save -o $TMP_USBMD_IMAGE_TAR usbmd/private:latest # save docker image to file.
apptainer build $TMP_USBMD_IMAGE_SIF docker-archive://$TMP_USBMD_IMAGE_TAR # convert docker image to apptainer image
scp $TMP_USBMD_IMAGE_SIF $SNELLIUS_USER@$SNELLIUS_ADDRESS:/projects/0/prjs0966/usbmd-private-$VERSION.sif # copy apptainer image to snellius
