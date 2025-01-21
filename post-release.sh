# PRIOR TO RUNNING THIS SCRIPT:
# Update usbmd version in pyproject.toml
# Update usbmd version in __init__.py

# Run this script on the linux server where the docker images are built.
# Depends on: docker, apptainer, scp and poetry
# ADVICE: run this script using `nohup ./post-release.sh &`

TMP_USBMD_IMAGE_TAR=/tmp/usbmd.tar
TMP_USBMD_IMAGE_SIF=/tmp/usbmd.sif
SNELLIUS_USER=wvannierop
REPO_DIR=~/ultrasound-toolbox

# Update lockfile
cd $REPO_DIR
poetry lock

# Get usbmd version
version=$(pip show usbmd | grep Version | cut -d ' ' -f 2)

# Build images
docker build . -t usbmd/base:v$version
docker build --build-arg KERAS3=True . -t usbmd/keras3:v$version

# Tag images
docker tag usbmd/base:v$version usbmd/base:latest
docker tag usbmd/keras3:v$version usbmd/keras3:latest

# Update image on snellius
docker save -o $TMP_USBMD_IMAGE_TAR usbmd/keras3:latest # save docker image to file.
apptainer build $TMP_USBMD_IMAGE_SIF docker-archive://$TMP_USBMD_IMAGE_TAR # convert docker image to apptainer image
scp $TMP_USBMD_IMAGE_SIF $SNELLIUS_USER@snellius.surf.nl:/projects/0/prjs0966/usbmd-keras3-v$version.sif # copy apptainer image to snellius