#!/usr/bin/env bash
# Shared launch settings. Every variable below can be overridden from the environment, e.g.
#   M=8 LAMBDA=0.5 TRAIN_BATCH_SIZE=32 bash scripts/run_allocate_repair.sh
# Extra Hydra overrides can be appended to any run_*.sh command line.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------- environment
CONDA_ENV_NAME="${CONDA_ENV_NAME:-arrlvr}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]] && command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
N_GPUS="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
export WANDB_MODE=offline
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER:-ar}}"

# ---------------------------------------------------------------- paths
MODEL_PATH="${MODEL_PATH:-/cm/archive/tue09/model_hub/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/processed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILES="${VAL_FILES:-[$DATA_DIR/val_dapo_heldout.parquet,$DATA_DIR/eval_aime24.parquet,$DATA_DIR/eval_aime25.parquet,$DATA_DIR/eval_math500.parquet]}"
PROJECT_NAME="${PROJECT_NAME:-allocate_repair_rlvr}"

# ---------------------------------------------------------------- method / optimization
G="${G:-8}"                                   # phase-1 samples per question
M="${M:-4}"                                   # repairs per failed response (0 = solve-only)
LAMBDA="${LAMBDA:-1.0}"                       # L_total = L_solve + LAMBDA * L_repair
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"    # questions per optimizer step
LR="${LR:-1e-6}"
SEED="${SEED:-42}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-}"                # empty = epochs * len(dataloader)
TEST_FREQ="${TEST_FREQ:-10}"                  # evaluation interval (steps)
SAVE_FREQ="${SAVE_FREQ:-50}"                  # checkpoint interval (steps)
EXAMPLES_INTERVAL="${EXAMPLES_INTERVAL:-10}"

# ---------------------------------------------------------------- per-phase decoding and token limits
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"           # phase-1 prompt budget (dataset filter)
ROLLOUT_PROMPT_LENGTH="${ROLLOUT_PROMPT_LENGTH:-2560}"   # covers phase-2 prompts too
SOLVE_MAX_TOKENS="${SOLVE_MAX_TOKENS:-8192}"
REPAIR_MAX_TOKENS="${REPAIR_MAX_TOKENS:-8192}"
VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-$SOLVE_MAX_TOKENS}"
MAX_PREV_ANSWER_TOKENS="${MAX_PREV_ANSWER_TOKENS:-1024}"
SOLVE_TEMPERATURE="${SOLVE_TEMPERATURE:-1.0}"
SOLVE_TOP_P="${SOLVE_TOP_P:-1.0}"
REPAIR_TEMPERATURE="${REPAIR_TEMPERATURE:-1.0}"
REPAIR_TOP_P="${REPAIR_TOP_P:-1.0}"
VAL_N="${VAL_N:-4}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.6}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"

# ---------------------------------------------------------------- memory (4x A100 40GB, 1.5B model)
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.6}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-False}"

max_of() { local m=$1; shift; for x in "$@"; do (( x > m )) && m=$x; done; echo "$m"; }

launch_training() {
    local exp_name="$1"
    shift
    local out_dir="$OUTPUT_ROOT/$exp_name"
    mkdir -p "$out_dir"
    local response_length seq_len ppo_tokens logprob_tokens
    response_length="$(max_of "$SOLVE_MAX_TOKENS" "$REPAIR_MAX_TOKENS" "$VAL_MAX_TOKENS")"
    seq_len=$((ROLLOUT_PROMPT_LENGTH + response_length))
    ppo_tokens="$(max_of "${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}" "$seq_len")"
    logprob_tokens="$(max_of "${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}" "$seq_len")"
    local steps_arg=()
    [[ -n "$TOTAL_STEPS" ]] && steps_arg=("trainer.total_training_steps=$TOTAL_STEPS")

    local cmd=(
        python -m allocate_repair.main
        "actor_rollout_ref.model.path=$MODEL_PATH"
        "data.train_files=$TRAIN_FILE"
        "data.val_files=$VAL_FILES"
        "data.train_batch_size=$TRAIN_BATCH_SIZE"
        "data.max_prompt_length=$MAX_PROMPT_LENGTH"
        "data.seed=$SEED"
        "actor_rollout_ref.rollout.n=$G"
        "allocate_repair.M=$M"
        "allocate_repair.repair_lambda=$LAMBDA"
        "allocate_repair.solve.max_tokens=$SOLVE_MAX_TOKENS"
        "allocate_repair.solve.temperature=$SOLVE_TEMPERATURE"
        "allocate_repair.solve.top_p=$SOLVE_TOP_P"
        "allocate_repair.repair.max_tokens=$REPAIR_MAX_TOKENS"
        "allocate_repair.repair.temperature=$REPAIR_TEMPERATURE"
        "allocate_repair.repair.top_p=$REPAIR_TOP_P"
        "allocate_repair.repair.max_previous_answer_tokens=$MAX_PREV_ANSWER_TOKENS"
        "allocate_repair.validation.max_tokens=$VAL_MAX_TOKENS"
        "allocate_repair.logging.examples_interval=$EXAMPLES_INTERVAL"
        "allocate_repair.data_manifest=$DATA_DIR/manifest.json"
        "actor_rollout_ref.rollout.prompt_length=$ROLLOUT_PROMPT_LENGTH"
        "actor_rollout_ref.rollout.response_length=$response_length"
        "actor_rollout_ref.rollout.max_model_len=$seq_len"
        "actor_rollout_ref.rollout.seed=$SEED"
        "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL"
        "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$logprob_tokens"
        "actor_rollout_ref.rollout.val_kwargs.n=$VAL_N"
        "actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE"
        "actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P"
        "actor_rollout_ref.rollout.agent.agent_loop_config_path=$REPO_ROOT/allocate_repair/config/agent_loops.yaml"
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$logprob_tokens"
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_tokens"
        "actor_rollout_ref.actor.optim.lr=$LR"
        "actor_rollout_ref.actor.data_loader_seed=$SEED"
        "actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD"
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD"
        "trainer.n_gpus_per_node=$N_GPUS"
        "trainer.project_name=$PROJECT_NAME"
        "trainer.experiment_name=$exp_name"
        "trainer.default_local_dir=$out_dir"
        "trainer.total_epochs=$TOTAL_EPOCHS"
        "trainer.test_freq=$TEST_FREQ"
        "trainer.save_freq=$SAVE_FREQ"
        "+ray_kwargs.ray_init.address=local"
        ${steps_arg[@]+"${steps_arg[@]}"}
        "$@"
    )
    {
        echo "# $(date -Iseconds) on $(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        printf '%q ' "${cmd[@]}"
        echo
    } >"$out_dir/command.sh"
    echo "[launch] $exp_name -> $out_dir (log: $out_dir/train.log)"
    cd "$REPO_ROOT"
    set -o pipefail
    "${cmd[@]}" 2>&1 | tee -a "$out_dir/train.log"
}
