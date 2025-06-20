#!/usr/bin/env bash
set -e

BACKENDS=("numpy" "jax" "torch" "tensorflow" "all")

IMAGE_PREFIX="zeahub"

echo "Building CPU images..."
for backend in "${BACKENDS[@]}"; do
  tag="${IMAGE_PREFIX}/${backend}-cpu"
  docker build -f Dockerfile --target cpu \
    --build-arg BACKEND="$backend" \
    -t "$tag" .
done

echo "Building GPU images..."
for backend in "${BACKENDS[@]}"; do
  tag="${IMAGE_PREFIX}/${backend}"
  docker build -f Dockerfile --target gpu \
    --build-arg BACKEND="$backend" \
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
