# Linux x86_64 image with several PyTorch builds side by side, one venv each,
# for testing whether CUDA-build and ROCm-build processes can join one job.
# No GPU is needed: every process runs on CPU with GPUBRIDGE_CPU_ONLY=1.
#
# Built by scripts/mixed_build_rendezvous.sh. Nothing is downloaded here. Each
# venv installs offline from its directory in the host's permanent wheel cache
# (~/.cache/gpubridge/wheels by default, filled by scripts/fetch_wheels.sh),
# passed in as a `wheels-<venv>` build context. BuildKit's own cache mounts
# can't hold those wheels: Docker Desktop caps them at about 2.76 GiB, and the
# ROCm torch wheels alone are 6+ GB.
#
# --build-arg ENVS=2.14.1 builds only the 2.14.1 pair; the default builds all
# five venvs. Each venv is its own layer and does not depend on the repo, so
# editing gpubridge never reinstalls torch.

ARG ENVS=all

# The platform (linux/amd64) comes from `docker build --platform`.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ENV UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

# The ROCm build of torch links against libatomic, which the slim image lacks.
RUN apt-get update && apt-get install -y --no-install-recommends libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Each RUN: <venv>, with the exact torch version that scripts/fetch_wheels.sh cached.
FROM base AS envs-2.14.1
RUN --mount=type=bind,from=wheels-cu-2.14.1,target=/wheels \
    uv venv -q /envs/cu-2.14.1 && uv pip install -q --python /envs/cu-2.14.1/bin/python \
        --no-index --find-links /wheels "torch==2.14.1+cu130" numpy
RUN --mount=type=bind,from=wheels-rocm-2.14.1,target=/wheels \
    uv venv -q /envs/rocm-2.14.1 && uv pip install -q --python /envs/rocm-2.14.1/bin/python \
        --no-index --find-links /wheels "torch==2.14.1+rocm7.2" numpy

# Older and neighbouring releases, for the version-matrix experiments.
FROM envs-2.14.1 AS envs-all
RUN --mount=type=bind,from=wheels-cu-2.13.0,target=/wheels \
    uv venv -q /envs/cu-2.13.0 && uv pip install -q --python /envs/cu-2.13.0/bin/python \
        --no-index --find-links /wheels "torch==2.13.0+cu130" numpy
RUN --mount=type=bind,from=wheels-cu-2.9.1,target=/wheels \
    uv venv -q /envs/cu-2.9.1 && uv pip install -q --python /envs/cu-2.9.1/bin/python \
        --no-index --find-links /wheels "torch==2.9.1+cu128" numpy
RUN --mount=type=bind,from=wheels-rocm-2.9.1,target=/wheels \
    uv venv -q /envs/rocm-2.9.1 && uv pip install -q --python /envs/rocm-2.9.1/bin/python \
        --no-index --find-links /wheels "torch==2.9.1+rocm6.4" numpy

# gpubridge, editable in every venv. The repo is mounted over /gpubridge at run
# time, so code changes apply without rebuilding.
FROM envs-${ENVS} AS final
COPY pyproject.toml README.md LICENSE /gpubridge/
COPY src /gpubridge/src
RUN for env in /envs/*/; do \
        uv pip install -q --python "$env/bin/python" --no-deps -e /gpubridge || exit 1; \
    done

WORKDIR /gpubridge
