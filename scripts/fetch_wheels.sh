#!/usr/bin/env bash
# Fill the permanent wheel cache for the mixed-build image, one directory per venv:
#
#   scripts/fetch_wheels.sh CACHE_DIR VENV...     e.g. ~/.cache/gpubridge/wheels cu-2.14.1
#
# The cache lives on the host, outside BuildKit. Docker Desktop's BuildKit caps
# cache mounts at about 2.76 GiB and evicted the 6+ GB ROCm wheels, so the image
# no longer downloads anything itself; scripts/Dockerfile installs offline from
# these directories.
#
# - ROCm torch wheels (6+ GB) are fetched on the host with scripts/fetch_wheel.sh:
#   curl with resume, checked against the index's sha256. Installing one straight
#   from the index with uv once failed with "Failed to read from zip file".
# - Everything else (CUDA torch, NVIDIA libraries, triton, numpy): `uv pip
#   compile` on the host pins the exact packages for Linux x86_64 / Python 3.12
#   (uv evaluates dependency markers for the target, not for the host). Then 8
#   parallel `pip download --no-deps` runs in a throwaway linux/amd64 container
#   fetch one pin each. A single pip download fetched only about 1 MB/s.
#   pip skips files already in the directory, and a .complete stamp skips
#   finished venvs.
#
# Needs uv and Docker on the host.
set -euo pipefail

cache=$1
shift
here="$(cd "$(dirname "$0")" && pwd)"
image=ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# venv -> "torch version|index|ROCm wheel URL|sha256" (the last two only for ROCm).
# Must match scripts/Dockerfile and ENVS in scripts/mixed_build_rendezvous.py.
spec() {
    case $1 in
    cu-2.14.1) echo "2.14.1+cu130|https://download.pytorch.org/whl/cu130||" ;;
    rocm-2.14.1) echo "2.14.1+rocm7.2|https://download.pytorch.org/whl/rocm7.2|https://download-r2.pytorch.org/whl/rocm7.2/torch-2.14.1%2Brocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl|fc35e48fd83329f5d60ab925951d138edb5a6fce84323600703637fbc693541c" ;;
    cu-2.13.0) echo "2.13.0+cu130|https://download.pytorch.org/whl/cu130||" ;;
    cu-2.9.1) echo "2.9.1+cu128|https://download.pytorch.org/whl/cu128||" ;;
    rocm-2.9.1) echo "2.9.1+rocm6.4|https://download.pytorch.org/whl/rocm6.4|https://download-r2.pytorch.org/whl/rocm6.4/torch-2.9.1%2Brocm6.4-cp312-cp312-manylinux_2_28_x86_64.whl|43471c9e4520402b5feeebd6c6d180e8bf2314402925798fe219c32fecd1ef95" ;;
    *) echo "fetch_wheels: unknown venv $1" >&2; return 1 ;;
    esac
}

for env in "$@"; do
    line=$(spec "$env")
    IFS='|' read -r version index rocm_url rocm_sha <<<"$line"
    dir="$cache/$env"
    mkdir -p "$dir"
    if [ "$(cat "$dir/.complete" 2>/dev/null)" = "$line" ]; then
        echo "[$env] cached: $(du -sh "$dir" | cut -f1) in $dir"
        continue
    fi
    if [ -n "$rocm_url" ]; then
        echo "[$env] fetching the ROCm torch wheel on the host ..."
        "$here/fetch_wheel.sh" "$rocm_url" "$rocm_sha" "$dir" >/dev/null
    fi
    echo "[$env] pinning torch==$version and its dependencies for Linux x86_64 ..."
    printf 'torch==%s\nnumpy\n' "$version" | uv pip compile - --quiet --no-header \
        --no-annotate --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
        --index-url "$index" -o "$dir/.requirements.txt"
    pins=$(grep -v -e '^#' -e '^$' "$dir/.requirements.txt")
    if [ -n "$rocm_url" ]; then
        pins=$(grep -v '^torch==' <<<"$pins")  # already here from fetch_wheel.sh
    fi
    echo "[$env] downloading $(echo "$pins" | wc -l | tr -d ' ') wheels from $index ..."
    echo "$pins" | docker run --rm -i --platform linux/amd64 -v "$dir":/wheels \
        -e PIP_ROOT_USER_ACTION=ignore -e PIP_DISABLE_PIP_VERSION_CHECK=1 "$image" \
        xargs -P 8 -I{} python -m pip download --quiet --no-deps --dest /wheels \
        --index-url "$index" "{}"
    echo "$line" >"$dir/.complete"
    echo "[$env] done: $(du -sh "$dir" | cut -f1) in $dir"
done
