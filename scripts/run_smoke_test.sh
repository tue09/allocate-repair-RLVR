#!/usr/bin/env bash
# End-to-end smoke test on the 4 GPUs (~15-30 min): 4 steps on 8 questions, G=4, M=2, 1024-token generations,
# validation before training and at step 2/4, examples every step, one checkpoint at the last step.
#   bash scripts/run_smoke_test.sh
set -euo pipefail
export TRAIN_FILE="${TRAIN_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/processed/smoke_train.parquet}"
export VAL_FILES="${VAL_FILES:-[$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/processed/smoke_val.parquet]}"
export G="${G:-4}" M="${M:-2}" LAMBDA="${LAMBDA:-1.0}" TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export SOLVE_MAX_TOKENS="${SOLVE_MAX_TOKENS:-1024}" REPAIR_MAX_TOKENS="${REPAIR_MAX_TOKENS:-1024}"
export VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-1024}" VAL_N="${VAL_N:-1}"
export TOTAL_STEPS="${TOTAL_STEPS:-4}" TEST_FREQ="${TEST_FREQ:-2}" SAVE_FREQ="${SAVE_FREQ:-4}" EXAMPLES_INTERVAL=1
export PROJECT_NAME="${PROJECT_NAME:-allocate_repair_smoke}"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXP_NAME="${EXP_NAME:-smoke_$(date +%Y%m%d_%H%M%S)}"
launch_training "$EXP_NAME" "$@"

OUT="$OUTPUT_ROOT/$EXP_NAME"
python - "$OUT" <<'EOF'
import json, os, sys
out = sys.argv[1]
logs = os.path.join(out, "allocate_repair_logs")
rows = [json.loads(l) for l in open(os.path.join(logs, "metrics.jsonl"))]
train = [r for r in rows if "train/total_loss" in r]
assert len(train) >= 1, "no training steps logged"
for r in train:
    print(f"step {r['step']}: solve reward {r['solve/reward_mean']:.3f}  repair available {r['repair/available']}  "
          f"gens {r['compute/total_generations']}  total loss {r['train/total_loss']}")
assert os.path.exists(os.path.join(logs, "examples.jsonl")), "examples.jsonl missing"
assert os.path.exists(os.path.join(logs, "run_info.json")), "run_info.json missing"
assert any(name.startswith("global_step_") for name in os.listdir(out)), "no checkpoint saved"
assert os.path.isdir(os.path.join(out, "wandb")), "offline W&B directory missing"
print("smoke test passed:", out)
EOF
