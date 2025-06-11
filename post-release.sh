#!/bin/bash
# Run this script on the linux server where the docker images are built.
# Depends on: git, docker, apptainer, scp, awk and poetry
# 1. Make sure `./post-release.sh` is executable: `chmod +x post-release.sh`
# 2. Also make sure to checkout the zea version you want to release before running this script.
# 3. Also make sure your git working directory is clean (no uncommitted changes).
# 4. Run this script using `./post-release.sh <ZEA_VERSION> <SNELLIUS_USER>`
#    e.g. `./post-release.sh v2.1.1 tstevens`
# 5. Sit back and relax, this might take a while...

shopt -s expand_aliases

TMP_ZEA_IMAGE_TAR=/tmp/zea.tar
TMP_ZEA_IMAGE_SIF=/tmp/zea.sif
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

# previous version
PREVIOUS_VERSION=$(awk -F'"' '/__version__/ {print $2}' zea/__init__.py)
PREVIOUS_VERSION="v$PREVIOUS_VERSION"

## Update version in pyproject.toml
sed -i "s/^version = .*/version = \"$VERSION_WITHOUT_V\"/" pyproject.toml

# Update version in __init__.py
sed -i "s/__version__ = .*/__version__ = \"$VERSION_WITHOUT_V\"/" zea/__init__.py

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

## Check if snellius is reachable
if ssh -q -o BatchMode=yes -o ConnectTimeout=5 "$SNELLIUS_USER@$SNELLIUS_ADDRESS" exit; then
    precho "Snellius is reachable with username '$SNELLIUS_USER'."
else
    precho "Snellius is not reachable or authentication failed for username '$SNELLIUS_USER'."
fi

## Change directory to the script's location (so repo root)
cd "$(dirname "$0")"
precho "Current directory: $(pwd)"

## Update poetry lockfile
precho "Updating poetry lockfile..."
if command -v poetry &> /dev/null; then
    # If poetry is installed, use it
    poetry lock --no-interaction
else
    # If poetry is not installed, use docker
    docker run --rm -v "$(pwd):/ultrasound-toolbox" --user "$(id -u):$(id -g)" zeahub/base:latest sudo /opt/poetry-venv/bin/python3 -m poetry lock --no-interaction
fi

## Build images
precho "Building docker images..."
docker build -f Dockerfile.base --build-arg BACKEND=all . -t zeahub/all:latest
docker build -f Dockerfile.base --build-arg BACKEND=numpy . -t zeahub/base:latest

## Tag images
# tag latest image with version
docker tag zeahub/all:latest zeahub/all:$VERSION
# tag latest image with version
docker tag zeahub/base:latest zeahub/base:$VERSION

## Build private image
precho "Building private docker image..."
docker build -t "zeahub/private:$VERSION" .
docker tag "zeahub/private:$VERSION" zeahub/private:latest

## Update image on snellius
precho "Updating images on snellius..."
# save docker image to file.
docker save -o $TMP_ZEA_IMAGE_TAR zeahub/private:latest
# convert docker image to apptainer image
apptainer build $TMP_ZEA_IMAGE_SIF docker-archive://$TMP_ZEA_IMAGE_TAR
# copy apptainer image to snellius
scp $TMP_ZEA_IMAGE_SIF $SNELLIUS_USER@$SNELLIUS_ADDRESS:/projects/0/prjs0966/zea-private-$VERSION.sif
