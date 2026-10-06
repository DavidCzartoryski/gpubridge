# Linux x86_64 image with several PyTorch builds side by side, one venv each,
# for testing whether CUDA-build and ROCm-build processes can join one job.
# No GPU is needed: every process runs on CPU with GPUBRIDGE_CPU_ONLY=1.
#
# Built by scripts/mixed_build_rendezvous.sh. Each venv is its own layer and
# does not depend on the repo, so editing gpubridge never re-downloads torch.

FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# The torch layers skip uv's cache so each multi-GB wheel is stored only once.
ENV UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

# Each line: <venv name> <exact torch version> <PyTorch wheel index>.
# Versions and index URLs were checked against https://download.pytorch.org/whl/torch/.
RUN uv venv -q /envs/cu-2.14.1 && uv pip install -q --python /envs/cu-2.14.1/bin/python \
        --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.1+cu130" numpy

# ROCm wheels are 6+ GB. Installing one straight from the index failed once
# with uv's "Failed to read from zip file" (a cut-off stream or uv's streaming
# unzip; not pinned down). Instead, fetch it with curl (resumable, kept in a
# BuildKit cache across failed builds), check the index's sha256, and install
# the verified file.
# The ROCm build of torch also links against libatomic, which the slim image lacks.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates libatomic1 \
    && rm -rf /var/lib/apt/lists/*
COPY scripts/fetch_wheel.sh /usr/local/bin/fetch_wheel
RUN --mount=type=cache,target=/wheels \
    wheel=$(fetch_wheel https://download-r2.pytorch.org/whl/rocm7.2/torch-2.14.1%2Brocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl \
        fc35e48fd83329f5d60ab925951d138edb5a6fce84323600703637fbc693541c /wheels) \
    && uv venv -q /envs/rocm-2.14.1 && uv pip install --python /envs/rocm-2.14.1/bin/python \
        --index-url https://download.pytorch.org/whl/rocm7.2 "$wheel" numpy

# gpubridge, editable in every venv. The repo is mounted over /gpubridge at run
# time, so code changes apply without rebuilding.
COPY pyproject.toml README.md LICENSE /gpubridge/
COPY src /gpubridge/src
RUN for env in /envs/*/; do \
        uv pip install -q --python "$env/bin/python" --no-deps -e /gpubridge || exit 1; \
    done

WORKDIR /gpubridge
