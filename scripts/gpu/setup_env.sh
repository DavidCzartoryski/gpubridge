#!/usr/bin/env bash
# Create (or reuse) the kit's Python environment. Idempotent: reruns skip
# whatever is already in place, so it costs seconds on a machine set up before.
#
#   scripts/gpu/setup_env.sh --flavor cuda|rocm|system [--venv DIR] [--dry-run]
#
#   cuda    CUDA build of PyTorch from the PyTorch index (Explorer, NVIDIA clouds)
#   rocm    ROCm build of PyTorch from the PyTorch index (AMD clouds, Explorer AMD nodes)
#   system  reuse a PyTorch already installed in the machine's Python, e.g. a
#           ROCm container image; saves a 6+ GB download on a paid machine
#
# Needs internet access. On a Slurm cluster it refuses to run outside a job, so
# installs never land on a login node; on Explorer use explorer/submit.sh setup.
set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "$0")/common.sh" "$@"

# ---- Settings (override with environment variables) ---------------------------
TORCH_VERSION="${TORCH_VERSION:-2.14.1}"
# CUDA 13 builds dropped V100 (sm_70) support, so cu126 is the safe default for
# a mixed V100/A100/H100 fleet. probe.py reports if the build can't run your GPU.
CUDA_INDEX="${CUDA_INDEX:-cu126}"
ROCM_INDEX="${ROCM_INDEX:-rocm7.2}"
# Where the torch wheel comes from; default: the PyTorch index for CUDA_INDEX/ROCM_INDEX.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV="${VENV:-${WORK_DIR:-$HOME/gpubridge-gpu}/venv}"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-python3}"
# -------------------------------------------------------------------------------

flavor=""
set -- "${ARGS[@]+"${ARGS[@]}"}"
while [ $# -gt 0 ]; do
    case $1 in
    --flavor) flavor=$2; shift 2 ;;
    --venv) VENV=$2; shift 2 ;;
    *) die "unknown argument: $1" ;;
    esac
done
case $flavor in
cuda) index_tag=$CUDA_INDEX ;;
rocm) index_tag=$ROCM_INDEX ;;
system) index_tag="" ;;
*) die "--flavor must be cuda, rocm or system" ;;
esac
index_url=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/$index_tag}

# sbatch on the PATH but no job: this is a cluster's login node.
if [ "$DRY_RUN" = 0 ] && [ -z "${SLURM_JOB_ID:-}" ] && command -v sbatch >/dev/null 2>&1; then
    die "this looks like a Slurm login node (sbatch is here, but no job is running)." \
        "Install on a compute node: scripts/gpu/explorer/submit.sh setup on Explorer," \
        "or run this inside an interactive job (srun --pty bash)."
fi

# uv's cache next to the venv keeps multi-GB wheels out of small home quotas.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$(dirname "$VENV")/uv-cache}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    log "installing uv into ~/.local/bin"
    if [ "$DRY_RUN" = 1 ]; then
        echo "+ curl -LsSf https://astral.sh/uv/install.sh | sh"
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
fi
export PATH="$HOME/.local/bin:$PATH"

if [ ! -x "$VENV/bin/python" ]; then
    if [ "$flavor" = system ]; then
        log "creating $VENV on top of $SYSTEM_PYTHON (reusing its PyTorch)"
        run uv venv --system-site-packages --python "$SYSTEM_PYTHON" "$VENV"
    else
        log "creating $VENV with Python $PYTHON_VERSION"
        run uv venv --python "$PYTHON_VERSION" "$VENV"
    fi
fi

current=""
if [ "$DRY_RUN" = 0 ]; then
    current=$("$VENV/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || true)
fi
if [ "$flavor" = system ]; then
    [ "$DRY_RUN" = 1 ] || [ -n "$current" ] || die "$SYSTEM_PYTHON has no PyTorch to reuse"
elif [ "$current" = "$TORCH_VERSION+$index_tag" ]; then
    log "torch $current already installed"
else
    log "installing torch $TORCH_VERSION+$index_tag (current: ${current:-none})"
    for attempt in 1 2 3; do
        if run uv pip install --python "$VENV/bin/python" \
            --index-url "$index_url" \
            "torch==$TORCH_VERSION+$index_tag" numpy; then
            break
        fi
        [ "$attempt" = 3 ] && die "torch install failed 3 times"
        log "install failed (attempt $attempt); retrying in 10s"
        sleep 10
    done
fi

log "installing gpubridge from $REPO_DIR"
run uv pip install --python "$VENV/bin/python" -e "$REPO_DIR"
run "$VENV/bin/python" -c "import torch, gpubridge; print('torch', torch.__version__, \
'cuda', torch.version.cuda, 'hip', torch.version.hip, 'gpubridge', gpubridge.__version__)"
log "environment ready: $VENV"
