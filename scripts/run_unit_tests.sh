#!/usr/bin/env bash
# Focused CPU checks: reward conversion, parent-specific grouping, input masking, no ground-truth leakage,
# zero-variance groups, empty repair batches, gradient accumulation, M / lambda weighting, trainer flow.
#   bash scripts/run_unit_tests.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_ROOT"
# also exercise </think> extraction with the real tokenizer when the model is available
if [[ -f "$MODEL_PATH/tokenizer_config.json" ]]; then
    export AR_TEST_TOKENIZER="$MODEL_PATH"
fi
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests -p no:warnings "$@"
