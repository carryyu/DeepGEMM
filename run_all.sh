#!/bin/bash
# Build DeepGEMM and run a selected SM100 Mega MoE datatype path.
set -euo pipefail

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

NUM_PROCESSES=${NUM_PROCESSES:-}
NUM_MAX_TOKENS=${NUM_MAX_TOKENS:-8192}
NUM_TOKENS=${NUM_TOKENS:-8192}
NUM_SHARED_EXPERTS=${NUM_SHARED_EXPERTS:-0}
HIDDEN=${HIDDEN:-7168}
INTERMEDIATE_HIDDEN=${INTERMEDIATE_HIDDEN:-2048}
NUM_EXPERTS=${NUM_EXPERTS:-96}
NUM_TOPK=${NUM_TOPK:-8}
MMA_TYPE=${MMA_TYPE:-fp8xfp4}
# MMA_TYPE=${MMA_TYPE:-nvfp4xnvfp4}
MASTER_PORT=${MASTER_PORT:-18361}
SKIP_BUILD=${SKIP_BUILD:-1}
SMOKE=${SMOKE:-0}

# NUM_PROCESSES=${NUM_PROCESSES:-}
# NUM_MAX_TOKENS=${NUM_MAX_TOKENS:-1}
# NUM_TOKENS=${NUM_TOKENS:-1}
# NUM_SHARED_EXPERTS=${NUM_SHARED_EXPERTS:-0}
# HIDDEN=${HIDDEN:-4096}
# INTERMEDIATE_HIDDEN=${INTERMEDIATE_HIDDEN:-2048}
# NUM_EXPERTS=${NUM_EXPERTS:-128}
# NUM_TOPK=${NUM_TOPK:-6}
# MMA_TYPE=${MMA_TYPE:-fp8xfp4}
# # MMA_TYPE=${MMA_TYPE:-nvfp4xnvfp4}
# MASTER_PORT=${MASTER_PORT:-18361}
# SKIP_BUILD=${SKIP_BUILD:-1}
# SMOKE=${SMOKE:-0}

# NUM_PROCESSES=${NUM_PROCESSES:-}
# NUM_MAX_TOKENS=${NUM_MAX_TOKENS:-1}
# NUM_TOKENS=${NUM_TOKENS:-1}
# NUM_SHARED_EXPERTS=${NUM_SHARED_EXPERTS:-0}
# HIDDEN=${HIDDEN:-7168}
# INTERMEDIATE_HIDDEN=${INTERMEDIATE_HIDDEN:-3072}
# NUM_EXPERTS=${NUM_EXPERTS:-192}
# NUM_TOPK=${NUM_TOPK:-6}
# # MMA_TYPE=${MMA_TYPE:-fp8xfp4}
# MMA_TYPE=${MMA_TYPE:-nvfp4xnvfp4}
# MASTER_PORT=${MASTER_PORT:-18361}
# SKIP_BUILD=${SKIP_BUILD:-1}
# SMOKE=${SMOKE:-0}

usage() {
    cat <<'EOF'
Usage: ./run_all.sh [options]

Options:
  --smoke              Quick smoke test (256 tokens, no shared experts)
  --skip-build         Skip develop.sh if already built
  --num-processes N    Local ranks / GPUs (default: auto from nvidia-smi)
  --num-tokens N       Tokens per rank (default: 32)
  --num-experts N      Total experts, must be divisible by num-processes (default: 96)
  --mma-type TYPE      nvfp4xnvfp4 (default), fp8xfp4, or bf16xbf16
  -h, --help           Show this help

Env overrides: NUM_PROCESSES, NUM_TOKENS, NUM_MAX_TOKENS, NUM_EXPERTS,
               NUM_SHARED_EXPERTS, HIDDEN, INTERMEDIATE_HIDDEN, NUM_TOPK,
               MMA_TYPE, MASTER_PORT, SKIP_BUILD=1, SMOKE=1
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --num-processes) NUM_PROCESSES="$2"; shift 2 ;;
        --num-tokens) NUM_TOKENS="$2"; NUM_MAX_TOKENS="$2"; shift 2 ;;
        --num-experts) NUM_EXPERTS="$2"; shift 2 ;;
        --mma-type) MMA_TYPE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

case "$MMA_TYPE" in
    fp8xfp4|nvfp4xnvfp4|bf16xbf16) ;;
    *) echo "Error: unsupported MMA_TYPE '$MMA_TYPE'" >&2; usage; exit 1 ;;
esac

if [[ "$SMOKE" == "1" ]]; then
    NUM_MAX_TOKENS=256
    NUM_TOKENS=256
    NUM_SHARED_EXPERTS=0
fi

if [[ -z "$NUM_PROCESSES" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_PROCESSES=$(nvidia-smi -L 2>/dev/null | wc -l)
    else
        NUM_PROCESSES=1
    fi
fi
if [[ "$NUM_PROCESSES" -lt 1 ]]; then
    echo "Error: no GPU detected (NUM_PROCESSES=$NUM_PROCESSES)" >&2
    exit 1
fi
if (( NUM_EXPERTS % NUM_PROCESSES != 0 )); then
    echo "Error: NUM_EXPERTS ($NUM_EXPERTS) must be divisible by NUM_PROCESSES ($NUM_PROCESSES)" >&2
    exit 1
fi

need_submodules=0
for path in third-party/cutlass/include third-party/fmt/include; do
    if [[ ! -d "$path" ]]; then
        need_submodules=1
        break
    fi
done
if [[ "$need_submodules" == "1" ]]; then
    echo "[run_all] Initializing git submodules..."
    git submodule update --init --recursive
fi

so_glob=(deep_gemm/_C*.so)
if [[ "$SKIP_BUILD" != "1" ]] || [[ ! -e "${so_glob[0]}" ]]; then
    echo "[run_all] Building via develop.sh..."
    ./develop.sh
else
    echo "[run_all] Skip build (found ${so_glob[0]})"
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$script_dir:$PYTHONPATH"
else
    export PYTHONPATH="$script_dir"
fi
export MASTER_PORT

echo "[run_all] PYTHONPATH=$PYTHONPATH"
echo "[run_all] Running Mega MoE: mma=$MMA_TYPE processes=$NUM_PROCESSES tokens=$NUM_TOKENS experts=$NUM_EXPERTS"

python3 tests/test_mega_moe.py \
    --num-processes "$NUM_PROCESSES" \
    --mma-type "$MMA_TYPE" \
    --num-max-tokens-per-rank "$NUM_MAX_TOKENS" \
    --num-tokens "$NUM_TOKENS" \
    --num-shared-experts "$NUM_SHARED_EXPERTS" \
    --hidden "$HIDDEN" \
    --intermediate-hidden "$INTERMEDIATE_HIDDEN" \
    --num-experts "$NUM_EXPERTS" \
    --num-topk "$NUM_TOPK"

echo "[run_all] Done."
