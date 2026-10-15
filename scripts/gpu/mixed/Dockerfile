# A CPU-only Linux machine for mixed/containers.sh: the CPU build of torch,
# gpubridge (editable; the repo is mounted over /gpubridge at run time) and the
# tools node.sh calls (ip, tar, git). Built by containers.sh; no GPU.
FROM python:3.12-slim

ARG TORCH_VERSION=2.14.1
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1

# git: the records' commit; the mounted repo belongs to another user, so mark it safe.
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 procps git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'
RUN python -m venv /venv \
    && /venv/bin/pip install -q "torch==${TORCH_VERSION}" \
        --index-url https://download.pytorch.org/whl/cpu \
    && /venv/bin/pip install -q numpy

COPY pyproject.toml README.md LICENSE /gpubridge/
COPY src /gpubridge/src
RUN /venv/bin/pip install -q --no-deps -e /gpubridge

WORKDIR /gpubridge
