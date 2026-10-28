#!/usr/bin/env bash
# Two-phase allocate-repair training: G solves per question, M repairs per failed solve, L = L_solve + LAMBDA*L_repair.
#   bash scripts/run_allocate_repair.sh
#   G=8 M=8 LAMBDA=0.5 bash scripts/run_allocate_repair.sh trainer.total_training_steps=200
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if (( M < 1 )); then
    echo "M must be >= 1 for repair training (use scripts/run_solve_only.sh for M=0; M=1 means advantage = reward, see allocate_repair/batching.py)" >&2
    exit 1
fi
EXP_NAME="${EXP_NAME:-ar_G${G}_M${M}_lam${LAMBDA}_bs${TRAIN_BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)}"
launch_training "$EXP_NAME" "$@"
