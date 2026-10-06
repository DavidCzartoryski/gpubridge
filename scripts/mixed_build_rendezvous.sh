#!/usr/bin/env bash
# Reproduce the mixed-build rendezvous experiments with one command:
#
#   scripts/mixed_build_rendezvous.sh                  # every venv, every experiment
#   scripts/mixed_build_rendezvous.sh --group 2.14.1   # extra args go to the driver
#   GPUBRIDGE_ENVS=2.14.1 scripts/mixed_build_rendezvous.sh --group 2.14.1
#                                                      # only fetch and build the 2.14.1 pair
#
# 1. Fills the wheel cache on the host (scripts/fetch_wheels.sh). It defaults to
#    ~/.cache/gpubridge/wheels; set GPUBRIDGE_WHEEL_CACHE to move it. The first
#    run downloads about 20 GB for all venvs (about 10 GB for the 2.14.1 pair);
#    later runs reuse the cache.
# 2. Builds a linux/amd64 image with one venv per PyTorch build, installed
#    offline from that cache.
# 3. Runs scripts/mixed_build_rendezvous.py in it with the repo mounted. Results
#    land in results/mixed_build/ unless you pass --results.
#
# On Apple Silicon the image runs under emulation.
set -euo pipefail

cd "$(dirname "$0")/.."
IMAGE=gpubridge-mixed:dev
CACHE="${GPUBRIDGE_WHEEL_CACHE:-$HOME/.cache/gpubridge/wheels}"
ENVS="${GPUBRIDGE_ENVS:-all}"

all_venvs=(cu-2.14.1 rocm-2.14.1 cu-2.13.0 cu-2.9.1 rocm-2.9.1)
case $ENVS in
2.14.1) venvs=(cu-2.14.1 rocm-2.14.1) ;;
all) venvs=("${all_venvs[@]}") ;;
*) echo "GPUBRIDGE_ENVS must be 2.14.1 or all, got $ENVS" >&2; exit 2 ;;
esac

scripts/fetch_wheels.sh "$CACHE" "${venvs[@]}"

# Every wheels-<venv> context the Dockerfile names must exist, even for stages
# this build skips, so venvs that weren't fetched get an empty directory.
mkdir -p "$CACHE/.empty"
contexts=()
for venv in "${all_venvs[@]}"; do
    dir="$CACHE/.empty"
    [[ " ${venvs[*]} " == *" $venv "* ]] && dir="$CACHE/$venv"
    contexts+=(--build-context "wheels-$venv=$dir")
done
docker build --platform linux/amd64 -f scripts/Dockerfile --build-arg "ENVS=$ENVS" \
    "${contexts[@]}" -t "$IMAGE" .
docker run --rm --platform linux/amd64 --shm-size=2g \
    -v "$PWD":/gpubridge -w /gpubridge "$IMAGE" \
    /envs/cu-2.14.1/bin/python scripts/mixed_build_rendezvous.py "$@"
