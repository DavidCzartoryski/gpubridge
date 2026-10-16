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
#                    waste the rest of a paid session. NCCL/RCCL write their
#                    NCCL_DEBUG=INFO output to $RESULTS/nccl/NAME/; if the step
#                    fails, the end of each of those logs is added to NAME.log
#   negative_step NAME SECONDS REGEX CMD
#                    a step that checks something fails as it should; passes only
#                    if CMD exits 0 and REGEX appears in its output, within SECONDS
#   limited CMD      run CMD under coreutils timeout (STEP_TIMEOUT), or print it
#                    If the step fails, the job's own output gets the last 20
#                    lines of NAME.log after the FAILED line (before torchrun's
#                    failure report and the NCCL/RCCL logs)
#   job_started      start $RESULTS/job.jsonl, the job's start, errors and end
#   rocm_asic_ids PYTHON  point a ROCm wheel's libdrm at the amdgpu.ids it ships
#   triton_cache_begin, triton_cache_end   a node-local Triton cache at a fixed path,
#                    filled from and saved to TRITON_CACHE_STORE
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

# ---- Hard limits, so a hang costs minutes instead of the allocation ------------
# KIT_TIMEOUT: the process-group timeout, in seconds, that every kit tool passes
# to gpubridge.init unless given --timeout. It bounds the rendezvous and every
# collective: Gloo on the bridge, and NCCL/RCCL in the islands, where
# TORCH_NCCL_ASYNC_ERROR_HANDLING=3 ends the process once a collective exceeds it.
export KIT_TIMEOUT="${KIT_TIMEOUT:-300}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-3}"
# STEP_TIMEOUT: wall-clock limit, in seconds, on each torchrun or srun launch
# (coreutils timeout, where installed; 0 = none). It is above KIT_TIMEOUT, so a
# stuck collective fails with its own error first. Slurm's --time caps each job.
export STEP_TIMEOUT="${STEP_TIMEOUT:-1200}"
# NCCL/RCCL say which version, transport and interfaces they picked at INFO;
# step sends it to files, so it costs nothing until a step fails.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

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
die() {
    log "ERROR: $*"
    job_event error "\"message\": \"$(json_string "$(redacted "$*")")\""
    exit 1
}

# redacted TEXT: TEXT with what the kit's JSON records never hold replaced, as gpukit
# does: /projects/<name> and PROJECT_DIR become $PROJECT_DIR, the home directory ~,
# and the user name as a path segment $USER.
redacted() {
    # shellcheck disable=SC2016  # the literal $PROJECT_DIR and $USER, not their values
    local text=$1 project='$PROJECT_DIR' home='~' user
    [ -z "${PROJECT_DIR:-}" ] || text=${text//"$PROJECT_DIR"/$project}
    [ -z "${HOME:-}" ] || [ "$HOME" = / ] || text=${text//"$HOME"/$home}
    # shellcheck disable=SC2016
    text=$(printf '%s' "$text" | sed -E 's#/projects/[^/[:space:]]+#$PROJECT_DIR#g')
    if [ -n "${USER:-}" ]; then
        user=${USER//./\\.}
        text=$(printf '%s' "$text" | sed -E "s#/${user}(/|\$|[[:space:]:;,])#/\$USER\1#g")
    fi
    printf '%s' "$text"
}

# json_string TEXT: TEXT escaped for a JSON string (quotes, backslashes, control characters).
json_string() {
    local text=$1
    text=${text//\\/\\\\}
    text=${text//\"/\\\"}
    printf '%s' "${text//[$'\001'-$'\037']/ }"
}

# job_started: begin $RESULTS/job.jsonl for this job (one line per event, appended,
# so reruns of a step keep their history): started, error (die, redacted), finished
# (finish) and exit (the script's exit code). triage.py lists a job whose last start
# has no end as unfinished, with its error or exit code: a job that stops before or
# during its first step otherwise leaves no steps.jsonl line to find. An empty
# steps.jsonl marks the directory as a job's from the start.
job_started() {
    [ "$DRY_RUN" = 0 ] || return 0
    JOB_EVENTS="$RESULTS/job.jsonl"
    touch "$RESULTS/steps.jsonl"
    job_event started
    # Also how the script ended: a command failing under set -e leaves no error line.
    trap 'job_event exit "\"code\": $?"' EXIT
}

# rocm_asic_ids PYTHON: export AMDGPU_ASIC_ID_TABLE_PATHS (unless it is set) as the
# directory of the GPU-name table that PYTHON's torch ships, torch/share/libdrm/amdgpu.ids
# (ROCm wheels only; nothing for CUDA ones). The libdrm_amdgpu bundled in the ROCm wheels
# looks for /opt/amdgpu/share/libdrm/amdgpu.ids, then walks the directory tree from the
# Python executable, and when that fails too it prints "(null): No such file or
# directory" for every GPU each process opens. torch is located, not imported.
rocm_asic_ids() {
    [ -z "${AMDGPU_ASIC_ID_TABLE_PATHS:-}" ] || return 0
    if [ "$DRY_RUN" = 1 ]; then
        echo "+ export AMDGPU_ASIC_ID_TABLE_PATHS=<torch>/share/libdrm (if the wheel ships it)"
        return 0
    fi
    local dir
    dir=$("$1" -c 'import importlib.util, os
spec = importlib.util.find_spec("torch")
d = os.path.join(os.path.dirname(spec.origin), "share", "libdrm") if spec and spec.origin else ""
print(d if d and os.path.isfile(os.path.join(d, "amdgpu.ids")) else "")' 2>/dev/null) || return 0
    [ -n "$dir" ] || return 0
    export AMDGPU_ASIC_ID_TABLE_PATHS="$dir"
    log "AMDGPU_ASIC_ID_TABLE_PATHS=$dir: the GPU-name table this ROCm build ships"
}

# triton_cache_begin: point Triton's kernel cache at a node-local directory, the same
# path on every node and in every job (TRITON_CACHE_LOCAL, default
# /tmp/gpubridge-triton-$USER), filled from TRITON_CACHE_STORE (persistent, e.g. in the
# project space); triton_cache_end (called by finish) adds the kernels the job built
# to the store. The path must not change: Triton's group files name their kernels by
# absolute path and rebuild a kernel whose path is gone. Triton's default cache is
# ~/.triton/cache, often network storage that each kernel build writes file by file.
# Only the batch node's kernels are kept; other nodes of a job build their own. A
# TRITON_CACHE_DIR you set wins, and an empty TRITON_CACHE_STORE turns this off. Never
# fails a job: without the directory, Triton keeps its default.
triton_cache_begin() {
    [ -z "${TRITON_CACHE_DIR:-}" ] && [ -n "${TRITON_CACHE_STORE:-}" ] || return 0
    local dir="${TRITON_CACHE_LOCAL:-/tmp/gpubridge-triton-${USER:-$(id -un)}}"
    if [ "$DRY_RUN" = 1 ]; then
        printf '+ TRITON_CACHE_DIR=%s, filled from %s\n' "$dir" "$TRITON_CACHE_STORE"
        return 0
    fi
    # Only a directory of our own: Triton imports compiled code from its cache, and
    # /tmp is shared with every user of the node.
    if ! mkdir -p "$dir" 2>/dev/null || [ -L "$dir" ] || [ ! -O "$dir" ] ||
        [ ! -w "$dir" ]; then
        log "warning: $dir isn't a directory of your own; Triton keeps its default cache"
        return 0
    fi
    chmod 700 "$dir" 2>/dev/null || true
    # Entries are whole directories, added to the store by rename (triton_cache_end),
    # so these are complete ones; another job may be adding some meanwhile.
    local entry
    for entry in "$TRITON_CACHE_STORE"/*/; do
        [ -d "$entry" ] || continue
        [ -e "$dir/$(basename "$entry")" ] || cp -a "$entry" "$dir/" 2>/dev/null || true
    done
    TRITON_CACHE_SAVE="$dir"
    export TRITON_CACHE_DIR="$dir"
    log "Triton kernel cache: $dir, node-local; $(find "$dir" -mindepth 1 -maxdepth 1 \
        ! -name '.*' | wc -l | tr -d ' ') entries, filled from $TRITON_CACHE_STORE"
}

triton_cache_end() {
    [ -n "${TRITON_CACHE_SAVE:-}" ] && [ -d "$TRITON_CACHE_SAVE" ] || return 0
    local entry name incoming
    if mkdir -p "$TRITON_CACHE_STORE" 2>/dev/null; then
        for entry in "$TRITON_CACHE_SAVE"/*/; do
            [ -d "$entry" ] || continue
            name=$(basename "$entry")
            [ -e "$TRITON_CACHE_STORE/$name" ] && continue
            # Finished kernels only: Triton writes the group file last, and another job
            # on this node may still be building into the same directory.
            compgen -G "$entry/__grp__*" >/dev/null || continue
            # Copied under a temporary name, then renamed: never seen half-written.
            incoming=$(mktemp -d "$TRITON_CACHE_STORE/.incoming-XXXXXX" 2>/dev/null) || continue
            if cp -a "$entry" "$incoming/$name" 2>/dev/null; then
                mv -T "$incoming/$name" "$TRITON_CACHE_STORE/$name" 2>/dev/null || true
            fi
            rm -rf "$incoming"
        done
    fi
    TRITON_CACHE_SAVE=""  # the local directory stays: other jobs on this node may use it
}

# job_event EVENT [JSON MEMBERS]: append an event to job.jsonl, once job_started ran.
job_event() {
    [ -n "${JOB_EVENTS:-}" ] || return 0
    printf '{"event": "%s", "job_id": "%s", "time": "%s"%s}\n' "$1" "${SLURM_JOB_ID:-local}" \
        "$(date +%Y-%m-%dT%H:%M:%S%z)" "${2:+, $2}" >>"$JOB_EVENTS"
}

quote() { printf '%q ' "$@"; }

run() {
    if [ "$DRY_RUN" = 1 ]; then
        printf '+ %s\n' "$(quote "$@")"
        return 0
    fi
    "$@"
}

# limited CMD...: CMD under coreutils timeout, which signals CMD's whole process
# group after STEP_TIMEOUT seconds (KILL 30 s later); exit code 124 means it ran
# out of time. Without timeout installed (macOS), CMD just runs.
limited() {
    if [ "$STEP_TIMEOUT" != 0 ] && { [ "$DRY_RUN" = 1 ] || command -v timeout >/dev/null 2>&1; }
    then
        run timeout --kill-after=30 "$STEP_TIMEOUT" "$@"
    else
        run "$@"
    fi
}

# nccl_report DIR: the last lines of each NCCL/RCCL log in DIR, for a failed step.
nccl_report() {
    local dir=$1 file found=0
    for file in "$dir"/*; do
        [ -f "$file" ] || continue
        found=1
        printf '\n==> NCCL/RCCL log %s (last 40 lines) <==\n' "$file"
        tail -n 40 "$file"
    done
    [ "$found" = 1 ] || printf '\n(no NCCL/RCCL log in %s: no communicator was created)\n' "$dir"
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
    if [ "$name" != triton_check ] && needs_triton "$name" "$@" &&
        { [ "$TRITON_GATE_MIXED" != 0 ] || [ "${1:-}" != mixed_torchrun ]; } && ! triton_ready; then
        not_run "$name" "triton_check failed: Triton can't run gpubridge's compression" \
            "kernels on some side of this job (triton_check.log)"
        return 0
    fi
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
    local start code timed_out=false nccl_dir="$RESULTS/nccl/$name"
    mkdir -p "$nccl_dir"
    start=$(date +%s)
    set +e
    # %h and %p: NCCL/RCCL fill in the host and process id, one file per rank.
    NCCL_DEBUG_FILE="$nccl_dir/%h.%p.log" "$@" 2>&1 | tee "$RESULTS/$name.log"
    code=${PIPESTATUS[0]}
    set -e
    local result=""
    if [ -n "${STEP_EXPECT:-}" ]; then  # set by negative_step
        if [ "$code" = 0 ] && grep -q -E -e "$STEP_EXPECT" "$RESULTS/$name.log"; then
            result="expected failure confirmed"
        else
            result="check did not fire"
            [ "$code" != 0 ] || code=1
        fi
        log "step $name: $result" 2>&1 | tee -a "$RESULTS/$name.log"
    fi
    if [ "$code" = 124 ]; then
        timed_out=true
        log "step $name ran out of time (STEP_TIMEOUT=${STEP_TIMEOUT}s)" 2>&1 |
            tee -a "$RESULTS/$name.log"
    fi
    local own_tail=""
    if [ "$code" != 0 ]; then
        # The step's own last lines: before torchrun's failure report, which names the
        # ranks that failed but not why, and before the NCCL/RCCL logs added below.
        own_tail=$(awk '/failed \(exitcode: -?[0-9]+\) local_rank:/ {exit} {print}' \
            "$RESULTS/$name.log" | tail -n 20)
        nccl_report "$nccl_dir" >>"$RESULTS/$name.log"
    fi
    rmdir "$nccl_dir" "$RESULTS/nccl" 2>/dev/null || true  # only if empty (CPU runs)
    printf '{"step": "%s", "exit_code": %d, "seconds": %d, "log": "%s.log", "timed_out": %s%s}\n' \
        "$name" "$code" "$(($(date +%s) - start))" "$name" "$timed_out" \
        "${result:+, \"result\": \"$result\"}" >>"$RESULTS/steps.jsonl"
    if grep -q -F "gpubridge-kit: import check failed" "$RESULTS/$name.log"; then
        # probe.py on some side of the job found that side's gpubridge isn't this
        # checkout's: every later step would measure other code.
        FAILED_STEPS+=("$name")
        die "step $name: a rank imported gpubridge from outside this checkout; stopping" \
            "before any other step. See $RESULTS/$name.log."
    fi
    if [ "$code" = 0 ]; then
        PASSED_STEPS+=("$name")
    else
        FAILED_STEPS+=("$name")
        log "step $name FAILED (exit $code); continuing. Log, with the end of any" \
            "NCCL/RCCL logs: $RESULTS/$name.log"
        log "the last 20 lines of $name.log, before any torchrun failure report:"
        printf '%s\n' "${own_tail:-(empty)}" | sed 's/^/    | /' >&2
    fi
    return 0
}

# ---- Triton kernels --------------------------------------------------------------
# gpubridge compresses GPU tensors with Triton kernels when Triton is installed
# (GPUBRIDGE_COMPRESSION_KERNELS=auto, the default) or forced (=triton). Where Triton
# can't compile them (venv-rocm63's 3.5.1 on an MI50's gfx906), only the steps that use
# them would fail. So before the first such step, step triton_check runs
# triton_check.py once through TRITON_CHECK_LAUNCH (each side of a mixed job; the jobs
# set it); if it fails, each such step is recorded as failed without running, and the
# rest of the job goes on. Uncompressed steps never check.

#: Steps that use the kernels whatever the compression setting.
TRITON_STEPS="${TRITON_STEPS:-gpu_tests bench_compression_kernels bench_compressed}"
#: How to run triton_check.py: a command prefix (words), default this node's python.
TRITON_CHECK_LAUNCH="${TRITON_CHECK_LAUNCH:-limited python}"
#: 0: never skip a mixed_torchrun step. For runners whose machines each decide alone
#: (mixed/node.sh): a step skipped on one machine and started on the other would leave
#: the two a step apart for the rest of the run. Ungated, it fails on both together.
TRITON_GATE_MIXED="${TRITON_GATE_MIXED:-1}"
TRITON_CHECKED_WITH="" TRITON_CHECK_RESULT=""

# needs_triton NAME CMD...: whether step NAME uses the kernels: one of TRITON_STEPS, or
# a torchrun launch (collectives) with GPUBRIDGE_BRIDGE_COMPRESSION set and the PyTorch
# kernels not forced. Probes, preflights and one-process runs never use them, even with
# compression exported for the whole job.
needs_triton() {
    local name=$1 arg compression=${GPUBRIDGE_BRIDGE_COMPRESSION:-} kernels
    shift
    case " $TRITON_STEPS " in *" $name "*) return 0 ;; esac
    kernels=$(printf '%s' "${GPUBRIDGE_COMPRESSION_KERNELS:-auto}" | tr '[:upper:]' '[:lower:]')
    [ -n "$compression" ] && [ "$compression" != none ] && [ "$kernels" != torch ] || return 1
    for arg in "$@"; do
        case $arg in *torchrun*) return 0 ;; esac
    done
    return 1
}

# triton_ready: step triton_check, once per TRITON_CHECK_LAUNCH; whether it passed.
triton_ready() {
    if [ "$TRITON_CHECKED_WITH" != "$TRITON_CHECK_LAUNCH" ]; then
        local failed_before=${#FAILED_STEPS[@]}
        # shellcheck disable=SC2086 # TRITON_CHECK_LAUNCH is a command prefix: split it
        step triton_check $TRITON_CHECK_LAUNCH "$KIT_DIR/triton_check.py"
        TRITON_CHECKED_WITH=$TRITON_CHECK_LAUNCH TRITON_CHECK_RESULT=ok
        [ "${#FAILED_STEPS[@]}" = "$failed_before" ] || TRITON_CHECK_RESULT=failed
    fi
    [ "$TRITON_CHECK_RESULT" = ok ]
}

# not_run NAME REASON...: record step NAME as failed without running it.
not_run() {
    local name=$1
    shift
    log "step $name not run: $*" 2>&1 | tee "$RESULTS/$name.log"
    printf '{"step": "%s", "exit_code": 1, "seconds": 0, "log": "%s.log", "timed_out": false, "not_run": "%s"}\n' \
        "$name" "$name" "$(json_string "$*")" >>"$RESULTS/steps.jsonl"
    FAILED_STEPS+=("$name")
}

# negative_step NAME SECONDS REGEX CMD...: a step whose CMD checks that something
# fails the way it should, e.g. check.py --expect-init-error REGEX. It passes only
# if CMD exits 0 and REGEX appears in its output: failing for another reason, or
# hanging, doesn't count. CMD gets its own limit of SECONDS instead of STEP_TIMEOUT.
# steps.jsonl gives the step a "result": "expected failure confirmed", or "check
# did not fire".
negative_step() {
    local name=$1 STEP_EXPECT=$3
    local -x STEP_TIMEOUT=$2
    shift 3
    step "$name" "$@"
}

finish() {
    triton_cache_end
    log "passed: ${PASSED_STEPS[*]:-none}"
    job_event finished "\"failed\": ${#FAILED_STEPS[@]}"
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
    if [ ! -f "$venv/bin/activate" ] && [ -x "$venv/bin/python" ]; then
        # A directory of wrappers around another venv's python and torchrun (e.g.
        # rocm_triton_overlay.sh's): put them first on PATH.
        log "$venv has no bin/activate; using its bin/ wrappers"
        export PATH="$venv/bin:$PATH"
        return 0
    fi
    [ -f "$venv/bin/activate" ] || die "no venv at $venv; run setup first"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
}

# fresh_results DIR: DIR, or DIR_job<id> when DIR already holds another run's
# records (job.jsonl or steps.jsonl), so a run never writes into, or over,
# another's folder. The id is SLURM_JOB_ID, else RUN_ID, else this shell's pid.
fresh_results() {
    local dir=$1 base n=1
    if [ -e "$dir/job.jsonl" ] || [ -e "$dir/steps.jsonl" ]; then
        base="${dir}_job${SLURM_JOB_ID:-${RUN_ID:-$$}}"
        dir=$base
        while [ -e "$dir/job.jsonl" ] || [ -e "$dir/steps.jsonl" ]; do
            n=$((n + 1))
            dir="${base}_$n"
        done
        log "$1 already holds another run's records; this run's go to $dir"
    fi
    printf '%s\n' "$dir"
}

# require_kit_import: stop unless the active python imports this checkout's
# gpubridge (import_check.py). Run after activate_venv.
require_kit_import() {
    if [ "$DRY_RUN" = 1 ]; then
        printf '+ python %s/import_check.py\n' "$KIT_DIR"
        return 0
    fi
    python "$KIT_DIR/import_check.py" >&2 ||
        die "the venv's gpubridge isn't this checkout's (above); stopping before any step"
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

# preflight_ifname FILE: the interface a preflight record (preflight.py --out)
# chose for its machine, or nothing if there is none.
preflight_ifname() {
    [ "$DRY_RUN" = 1 ] && return 0
    python -c 'import json, sys; print(json.load(open(sys.argv[1])).get("chosen_ifname") or "")' \
        "$1" 2>/dev/null || true
}

# After a preflight, Gloo uses the interfaces it chose, unless GLOO_SOCKET_IFNAME
# was set already (the preflight then only checks that one) or USE_PREFLIGHT_IFNAME=0.
USE_PREFLIGHT_IFNAME="${USE_PREFLIGHT_IFNAME:-1}"
PREFLIGHT_SECONDS="${PREFLIGHT_SECONDS:-2}"  # of raw TCP data each way, per pair of interfaces

# use_preflight_ifname DIR: after a preflight between two nodes (preflight.py --role
# auto, so node 0 is the master), make Gloo use the interfaces it chose. If both
# nodes chose the same name, export GLOO_SOCKET_IFNAME for the rest of the job.
# Otherwise export IFNAME_BY_NODE ("<node 0's> <node 1's>"): job steps started on
# both nodes through node_ifname.sh then set it to each node's own choice.
use_preflight_ifname() {
    [ -z "${GLOO_SOCKET_IFNAME:-}" ] && [ "$USE_PREFLIGHT_IFNAME" = 1 ] || return 0
    local master worker
    master=$(preflight_ifname "$1/preflight-master.json")
    worker=$(preflight_ifname "$1/preflight-worker.json")
    if [ -n "$master" ] && [ "$master" = "$worker" ]; then
        export GLOO_SOCKET_IFNAME="$master"
        log "GLOO_SOCKET_IFNAME=$master, as the preflight chose"
    elif [ -n "$master" ] && [ -n "$worker" ]; then
        export IFNAME_BY_NODE="$master $worker"
        log "GLOO_SOCKET_IFNAME per node, as the preflight chose: $master on node 0" \
            "(the master), $worker on node 1"
    elif [ -n "$master$worker" ]; then
        log "the preflight chose an interface on only one node (${master:-none} on the" \
            "master, ${worker:-none} on the worker); GLOO_SOCKET_IFNAME stays unset, so Gloo picks"
    fi
}

# torchrun on this machine only. Pins the rendezvous to 127.0.0.1 on a free port
# instead of --standalone, which hangs on hosts whose own name doesn't resolve
# (seen on macOS).
local_torchrun() {
    limited torchrun --nnodes=1 --master-addr=127.0.0.1 --master-port="$(free_port)" "$@"
}

# dp_run_options RUN: what a data-parallel run's name (step 07's and the mixed runner's
# DP_RUNS) means for train_dp.py, in DP_MODE, DP_COMPRESSION (for
# GPUBRIDGE_BRIDGE_COMPRESSION) and DP_RUN_ARGS: ddp and fsdp2; ddp-bf16 and ddp-fp8 (the
# bridge compressed); ddp-speed (batch shares by each GPU's speed); ddp-bucket-<MB>
# (DDP's bucket size).
# shellcheck disable=SC2034  # the caller reads them
dp_run_options() {
    DP_MODE=${1%%-*} DP_COMPRESSION='' DP_RUN_ARGS=()
    case $1 in
    ddp | fsdp2) ;;
    ddp-bf16 | ddp-fp8) DP_COMPRESSION=${1#ddp-} ;;
    ddp-speed) DP_RUN_ARGS=(--shares speed) ;;
    ddp-bucket-*) DP_RUN_ARGS=(--bucket-mb "${1#ddp-bucket-}") ;;
    *) die "DP_RUNS: unknown run '$1'" ;;
    esac
}

# env_local_torchrun NAME=VALUE... ARGS: local_torchrun with extra environment variables.
env_local_torchrun() {
    local assignments=()
    while [ $# -gt 0 ] && [[ $1 =~ ^[A-Z_][A-Z0-9_]*= ]]; do
        assignments+=("$1")
        shift
    done
    limited env "${assignments[@]}" torchrun --nnodes=1 --master-addr=127.0.0.1 \
        --master-port="$(free_port)" "$@"
}
