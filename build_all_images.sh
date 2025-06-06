#!/usr/bin/env bash
set -e

BACKENDS=("numpy" "jax" "torch" "tensorflow" "all")

echo "Building CPU images..."
for backend in "${BACKENDS[@]}"; do
  tag="usbmd/${backend}-cpu"
  docker build -f Dockerfile.base --target cpu \
    --build-arg BACKEND="$backend" \
    -t "$tag" .
done

echo "Building GPU images..."
for backend in "${BACKENDS[@]}"; do
  tag="usbmd/${backend}"
  docker build -f Dockerfile.base --target gpu \
    --build-arg BACKEND="$backend" \
    -t "$tag" .
done

echo
echo "Image sizes (uncompressed):"
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep -E "usbmd/"

echo
echo "Image sizes (compressed):"
for backend in "${BACKENDS[@]}"; do
  for tag in "usbmd/${backend}-cpu" "usbmd/${backend}"; do
    if docker image inspect "$tag" > /dev/null 2>&1; then
      size=$(docker image save "$tag" | gzip -c | wc -c)
      echo "$tag: $((size / 1048576)) MB (compressed)"
    fi
  done
done
