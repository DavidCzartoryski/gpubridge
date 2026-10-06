#!/usr/bin/env bash
# Reproduce the mixed-build rendezvous experiments with one command:
#
#   scripts/mixed_build_rendezvous.sh                  # build the image, run everything
#   scripts/mixed_build_rendezvous.sh --group 2.14.1   # extra args go to the driver
#
# Builds a linux/amd64 image with CUDA and ROCm builds of PyTorch (about 20 GB
# of downloads the first time), then runs scripts/mixed_build_rendezvous.py in
# it with the repo mounted. Results land in results/mixed_build/. On Apple
# Silicon this runs under emulation and takes a while.
set -euo pipefail

cd "$(dirname "$0")/.."
IMAGE=gpubridge-mixed:dev

docker build --platform linux/amd64 -f scripts/Dockerfile -t "$IMAGE" .
docker run --rm --platform linux/amd64 --shm-size=2g \
    -v "$PWD":/gpubridge -w /gpubridge "$IMAGE" \
    /envs/cu-2.14.1/bin/python scripts/mixed_build_rendezvous.py "$@"
