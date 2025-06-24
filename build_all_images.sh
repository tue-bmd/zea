#!/usr/bin/env bash
set -e

BACKENDS=("numpy" "jax" "torch" "tensorflow" "all")

IMAGE_PREFIX="zeahub"

echo "Building CPU images..."
for backend in "${BACKENDS[@]}"; do
  tag="${IMAGE_PREFIX}/${backend}-cpu"
  # Set build args per backend for CPU
  case "$backend" in
    all)
      BUILD_ARGS="--build-arg INSTALL_JAX=cpu --build-arg INSTALL_TORCH=cpu --build-arg INSTALL_TF=cpu --build-arg DEV=true"
      ;;
    jax)
      BUILD_ARGS="--build-arg INSTALL_JAX=cpu --build-arg INSTALL_TORCH=false --build-arg INSTALL_TF=false"
      ;;
    torch)
      BUILD_ARGS="--build-arg INSTALL_JAX=false --build-arg INSTALL_TORCH=cpu --build-arg INSTALL_TF=false"
      ;;
    tensorflow)
      BUILD_ARGS="--build-arg INSTALL_JAX=false --build-arg INSTALL_TORCH=false --build-arg INSTALL_TF=cpu"
      ;;
    numpy)
      BUILD_ARGS="--build-arg INSTALL_JAX=false --build-arg INSTALL_TORCH=false --build-arg INSTALL_TF=false"
      ;;
  esac
  docker build -f Dockerfile \
    $BUILD_ARGS \
    -t "$tag" .
done

echo "Building GPU images..."
for backend in "${BACKENDS[@]}"; do
  # No GPU image for numpy
  if [[ "$backend" == "numpy" ]]; then
    continue
  fi
  tag="${IMAGE_PREFIX}/${backend}"
  # Set build args per backend for GPU
  case "$backend" in
    all)
      BUILD_ARGS="--build-arg INSTALL_JAX=gpu --build-arg INSTALL_TORCH=gpu --build-arg INSTALL_TF=gpu"
      ;;
    jax)
      BUILD_ARGS="--build-arg INSTALL_JAX=gpu --build-arg INSTALL_TORCH=false --build-arg INSTALL_TF=false"
      ;;
    torch)
      BUILD_ARGS="--build-arg INSTALL_JAX=false --build-arg INSTALL_TORCH=gpu --build-arg INSTALL_TF=false"
      ;;
    tensorflow)
      BUILD_ARGS="--build-arg INSTALL_JAX=false --build-arg INSTALL_TORCH=false --build-arg INSTALL_TF=gpu"
      ;;
  esac
  docker build -f Dockerfile \
    $BUILD_ARGS \
    -t "$tag" .
done

echo
echo "Image sizes (uncompressed):"
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep -E "zeahub/"

echo
echo "Image sizes (compressed):"
for backend in "${BACKENDS[@]}"; do
  for tag in "${IMAGE_PREFIX}/${backend}-cpu" "${IMAGE_PREFIX}/${backend}"; do
    if docker image inspect "$tag" > /dev/null 2>&1; then
      size=$(docker image save "$tag" | gzip -c | wc -c)
      echo "$tag: $((size / 1048576)) MB (compressed)"
    fi
  done
done
