# shellcheck shell=bash
# Shared helpers for the GPU validation kit. Source it, don't run it:
#
#   source "$(dirname "$0")/../common.sh" "$@"
#
# Provides:
#   --dry-run   handled here: commands are printed, not run (DRY_RUN=1 works too)
#   ARGS        the script's arguments with --dry-run removed
#   log, die    messages on stderr
#   run CMD     run a command, or print it in dry-run mode
#   capture F CMD    save a command's output to file F; never fails the script
#   step NAME CMD    run a named step, log to $RESULTS/NAME.log, record pass/fail in
#                    $RESULTS/steps.jsonl and carry on, so one failure doesn't
#                    waste the rest of a paid session
#   finish      print the step table; exit non-zero if any step failed
#   load_env_file F  read NAME=value settings from file F, if it exists

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$KIT_DIR/../.." && pwd)"
export KIT_DIR REPO_DIR

DRY_RUN="${DRY_RUN:-0}"
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then DRY_RUN=1; else ARGS+=("$arg"); fi
done
export DRY_RUN

FAILED_STEPS=()
PASSED_STEPS=()

# load_env_file FILE: read NAME=value lines from FILE (if it exists) for settings
# not already set, so the environment wins. The file is parsed, not run: other
# lines are ignored, and a value may be wrapped in double quotes.
load_env_file() {
    local line name value
    [ -f "$1" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        [[ $line =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
        name=${BASH_REMATCH[1]}
        value=${BASH_REMATCH[2]}
        value=${value#\"}
        value=${value%\"}
        [ -n "${!name:-}" ] || printf -v "$name" '%s' "$value"
    done <"$1"
}

log() { printf '[gpubridge-kit %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

quote() { printf '%q ' "$@"; }

run() {
    if [ "$DRY_RUN" = 1 ]; then
        printf '+ %s\n' "$(quote "$@")"
        return 0
    fi
    "$@"
}

capture() {
    local out=$1
    shift
    if [ "$DRY_RUN" = 1 ]; then
        printf '+ %s> %s\n' "$(quote "$@")" "$out"
        return 0
    fi
    mkdir -p "$(dirname "$out")"
    if command -v "$1" >/dev/null 2>&1; then
        "$@" >"$out" 2>&1 || true
    else
        echo "$1: not installed on $(hostname)" >"$out"
    fi
}

step() {
    local name=$1
    shift
    : "${RESULTS:?set RESULTS before calling step}"
    log "step $name: $(quote "$@")"
    if [ "$DRY_RUN" = 1 ]; then
        if declare -F "$1" >/dev/null; then
            printf '+ [%s] (via %s)\n' "$name" "$1"
            "$@"  # shell functions print their own commands through run
        else
            printf '+ [%s] %s\n' "$name" "$(quote "$@")"
        fi
        PASSED_STEPS+=("$name")
        return 0
    fi
    mkdir -p "$RESULTS"
    local start code
    start=$(date +%s)
    set +e
    "$@" 2>&1 | tee "$RESULTS/$name.log"
    code=${PIPESTATUS[0]}
    set -e
    printf '{"step": "%s", "exit_code": %d, "seconds": %d, "log": "%s.log"}\n' \
        "$name" "$code" "$(($(date +%s) - start))" "$name" >>"$RESULTS/steps.jsonl"
    if [ "$code" = 0 ]; then
        PASSED_STEPS+=("$name")
    else
        FAILED_STEPS+=("$name")
        log "step $name FAILED (exit $code); continuing. Log: $RESULTS/$name.log"
    fi
    return 0
}

finish() {
    log "passed: ${PASSED_STEPS[*]:-none}"
    if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
        log "FAILED: ${FAILED_STEPS[*]}"
        return 1
    fi
    log "all steps passed"
}

# Activate the kit's venv (created by setup_env.sh). In dry-run mode, just say so.
activate_venv() {
    local venv=$1
    if [ "$DRY_RUN" = 1 ]; then
        printf '+ source %s/bin/activate\n' "$venv"
        return 0
    fi
    [ -f "$venv/bin/activate" ] || die "no venv at $venv; run setup first"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
}

# Number of GPUs PyTorch sees. GPU_COUNT overrides (and is required in dry-run mode).
gpu_count() {
    if [ -n "${GPU_COUNT:-}" ]; then
        echo "$GPU_COUNT"
    elif [ "$DRY_RUN" = 1 ]; then
        echo 4
    else
        python -c "import torch; print(torch.cuda.device_count())"
    fi
}

# package_results DIR: tar DIR next to itself and print exactly what to download.
package_results() {
    local dir=$1 parent name tarball
    parent=$(dirname "$dir")
    name=$(basename "$dir")
    tarball="$parent/$name.tgz"
    run tar -czf "$tarball" -C "$parent" "$name"
    if [ "$DRY_RUN" = 1 ]; then
        return 0
    fi
    local size host port
    size=$(du -h "$tarball" | cut -f1)
    # SSH_CONNECTION is "client_ip client_port server_ip server_port" in an SSH session.
    read -r _ _ host port <<<"${SSH_CONNECTION:-}"
    cat >&2 <<MSG

================================================================================
Results: $tarball ($size)
Download it before stopping this machine, e.g. from your laptop:

    scp -P ${port:-22} ${USER:-root}@${host:-<this-machine-address>}:$tarball .

(If the provider proxies SSH, use the host and port from its connect dialog.)
================================================================================
MSG
}

# A free TCP port on this machine (a fixed placeholder in dry-run mode).
free_port() {
    if [ "$DRY_RUN" = 1 ]; then
        echo 29500
    else
        python -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])"
    fi
}

# torchrun on this machine only. Pins the rendezvous to 127.0.0.1 on a free port
# instead of --standalone, which hangs on hosts whose own name doesn't resolve
# (seen on macOS).
local_torchrun() {
    run torchrun --nnodes=1 --master-addr=127.0.0.1 --master-port="$(free_port)" "$@"
}

# env_local_torchrun NAME=VALUE... ARGS: local_torchrun with extra environment variables.
env_local_torchrun() {
    local assignments=()
    while [ $# -gt 0 ] && [[ $1 =~ ^[A-Z_][A-Z0-9_]*= ]]; do
        assignments+=("$1")
        shift
    done
    run env "${assignments[@]}" torchrun --nnodes=1 --master-addr=127.0.0.1 \
        --master-port="$(free_port)" "$@"
}
