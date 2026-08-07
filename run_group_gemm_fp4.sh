#!/bin/bash
set -euo pipefail

script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

if [[ ! -d third-party/cutlass/include || ! -d third-party/fmt/include ]]; then
    echo "[group-gemm] Initializing git submodules..."
    git submodule update --init --recursive
fi

shopt -s nullglob
extension_files=(deep_gemm/_C*.so)
if [[ "${FORCE_BUILD:-0}" == "1" || ${#extension_files[@]} -eq 0 ]]; then
    echo "[group-gemm] Building DeepGEMM..."
    ./develop.sh
fi

existing_pythonpath="${PYTHONPATH:-}"
existing_pythonpath="${existing_pythonpath#:}"
if [[ -n "$existing_pythonpath" ]]; then
    export PYTHONPATH="$script_dir:$existing_pythonpath"
else
    export PYTHONPATH="$script_dir"
fi

# The benchmark is single-process. CUDA_VISIBLE_DEVICES may be overridden by the caller.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python3 scripts/bench_group_gemm_fp4.py "$@"
