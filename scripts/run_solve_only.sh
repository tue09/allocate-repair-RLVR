#!/usr/bin/env bash
# Solve-only GRPO baseline (M=0): same pipeline, data, decoding and logging, no repair phase.
#   bash scripts/run_solve_only.sh
#   G=12 bash scripts/run_solve_only.sh      # e.g. a compute-matched baseline with a larger G
set -euo pipefail
export M=0
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXP_NAME="${EXP_NAME:-solve_only_G${G}_bs${TRAIN_BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)}"
launch_training "$EXP_NAME" "$@"
