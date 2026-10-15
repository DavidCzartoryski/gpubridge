#!/usr/bin/env bash
# One venv's PyTorch with another venv's Triton: an env-only venv (bin/python, python3,
# torchrun) that puts TRITON_VENV's Triton in front of BASE_VENV's on the path. For a
# GPU whose BLAS kernels only an older ROCm wheel has, and whose Triton only a newer
# one compiles gpubridge's compression kernels with (an MI50, gfx906:
# docs/GPU_VALIDATION_RUNBOOK.md, "Choosing the ROCm build").
#
#   scripts/gpu/rocm_triton_overlay.sh BASE_VENV TRITON_VENV OUT_DIR
#
#   scripts/gpu/rocm_triton_overlay.sh $WORK_DIR/venv-rocm63 $WORK_DIR/venv-rocm \
#       $WORK_DIR/venv-rocm63-triton38
#   VENV_ROCM=$WORK_DIR/venv-rocm63-triton38 scripts/gpu/explorer/submit.sh 07
#
# OUT_DIR/overlay holds symbolic links to TRITON_VENV's triton package and its
# dist-info, nothing else, so rebuilding TRITON_VENV changes what the overlay runs.
# No Python runs here (fine on a login node); nothing in either venv changes.
set -euo pipefail
[ $# -eq 3 ] || { sed -n '2,16p' "$0" >&2; exit 2; }
die() { echo "rocm_triton_overlay: $*" >&2; exit 1; }
base=$(cd "$1" && pwd -P) || die "$1: no base venv directory"
triton_venv=$(cd "$2" && pwd -P) || die "$2: no Triton venv directory"
out=$3
[ -x "$base/bin/python" ] || die "$base/bin/python: not a venv's python"
for tool in python python3 torchrun; do
    [ -x "$base/bin/$tool" ] || die "$base/bin/$tool is missing"
done
shopt -s nullglob
packages=("$triton_venv"/lib/python3*/site-packages/triton)
[ ${#packages[@]} -eq 1 ] || die "expected one triton package under $triton_venv/lib/python3*/site-packages, found ${#packages[@]}"
site=$(dirname "${packages[0]}")
infos=("$site"/triton-*.dist-info "$site"/triton_rocm-*.dist-info "$site"/pytorch_triton_rocm-*.dist-info)
[ ${#infos[@]} -eq 1 ] || die "expected one Triton dist-info in $site, found ${#infos[@]}"
# The overlay has to sit in front of the same Python minor version as the base venv's.
base_sites=("$base"/lib/python3*/site-packages)
[ ${#base_sites[@]} -eq 1 ] || die "expected one site-packages under $base/lib"
[ "$(basename "$(dirname "$site")")" = "$(basename "$(dirname "${base_sites[0]}")")" ] ||
    die "$triton_venv is $(basename "$(dirname "$site")"), $base is $(basename "$(dirname "${base_sites[0]}")")"

# Resolve aliases before touching bin/: an existing venv's python commonly
# points at a shared interpreter, which must never become a wrapper script.
mkdir -p "$out"
out=$(cd "$out" && pwd -P)
case "$out/" in
    "$base/"* | "$triton_venv/"*) die "OUT_DIR must be outside both source venvs" ;;
esac
[ ! -e "$out/pyvenv.cfg" ] || die "OUT_DIR is an existing Python venv"
for directory in bin overlay; do
    [ ! -L "$out/$directory" ] || die "OUT_DIR/$directory must not be a symbolic link"
done
mkdir -p "$out/overlay" "$out/bin"
rm -f "$out/overlay/triton" "$out"/overlay/*.dist-info
ln -s "${packages[0]}" "$out/overlay/triton"
ln -s "${infos[0]}" "$out/overlay/$(basename "${infos[0]}")"
for tool in python python3 torchrun; do
    # Replace links rather than following them, including on reruns.
    rm -f "$out/bin/$tool"
    printf -v overlay_path '%q' "$out/overlay"
    printf -v executable '%q' "$base/bin/$tool"
    cat >"$out/bin/$tool" <<EOF
#!/usr/bin/env bash
# $(basename "$base")'s $tool with $(basename "${infos[0]}" .dist-info) from $(basename "$triton_venv") first on the path
# (scripts/gpu/rocm_triton_overlay.sh).
export PYTHONPATH=$overlay_path\${PYTHONPATH:+:\$PYTHONPATH}
exec $executable "\$@"
EOF
    chmod +x "$out/bin/$tool"
done
echo "$out: $(basename "$base")'s torch with $(basename "${infos[0]}" .dist-info) (VENV_ROCM=$out)"
