#!/usr/bin/env bash
# Allocate-repair (two-phase GRPO) on DAPO-Math-17k, DeepSeek-R1-Distill-Qwen-1.5B, 4x A100-40GB: per question, G solves;
# for every wrong solve, M repairs; L = L_solve + LAMBDA * L_repair, one optimizer step per training step.
#
#   bash scripts/train_allocate_repair_r1distill_dapo.sh                       # G=4, M=1 (defaults below)
#   M=2 LAMBDA=0.5 bash scripts/train_allocate_repair_r1distill_dapo.sh        # change settings with env variables
#   M=0 bash scripts/train_allocate_repair_r1distill_dapo.sh                   # solve-only baseline, same trainer
#   DRY_RUN=1 bash scripts/train_allocate_repair_r1distill_dapo.sh             # print the resolved config, no training
#   EXP_NAME=<existing run> bash scripts/train_allocate_repair_r1distill_dapo.sh   # resume from its latest checkpoint
#
# M=1: each failed response gets one repair, so its repair group has a single sample. verl's GRPO estimator gives a
# single-sample group mean 0 and std 1 -> the repair advantage is the raw 0/1 reward: successful repairs are pushed up,
# failed repairs get no gradient, and the repair term is not group-normalized like the solve term (allocate_repair/
# batching.py). M >= 2 gives group-relative repair advantages.
#
# Not directly comparable with scripts/train_grpo_r1distill_dapo.sh: this trainer uses seq-mean-token-mean (not the
# DAPO token-mean) and one optimizer step per training step, so every PPO ratio is 1 and the clip range (incl.
# clip-higher) has no effect. For a like-for-like baseline run M=0 with this script.
#
# Every trajectory of every step (exact prompt + full raw output) -> $TRAJECTORY_DIR/step_NNNNN.jsonl.gz.
# Everything else of a run goes to $CKPT_ROOT/$EXP_NAME: checkpoints (global_step_N/), train.log, command.sh, wandb/,
# metrics/*.jsonl, allocate_repair_logs/ (per-step metrics, repair examples, run info).
# Progress figures: python scripts/plot_run.py <EXP_NAME>  (-> result/<EXP_NAME>/)
# HF model from a checkpoint:
#   python -m verl.model_merger merge --backend fsdp --local_dir <run>/global_step_N/actor --target_dir <out dir>
# Default: 250 training steps, then one evaluation on all VAL_FILES. A run is long: start it inside tmux.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data/processed"

# ---------------------------------------------------------------- model / data / output
MODEL_PATH="${MODEL_PATH:-/cm/archive/tue09/model_hub/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"   # DAPO-Math-17k (deduplicated), 16,373 questions
# first-attempt accuracy (original question, no feedback) on: held-out DAPO questions, AIME24, AIME25, MATH-500
VAL_FILES="${VAL_FILES:-[$DATA_DIR/val_dapo_heldout.parquet,$DATA_DIR/eval_aime24.parquet,$DATA_DIR/eval_aime25.parquet,$DATA_DIR/eval_math500.parquet]}"
PROJECT_NAME="${PROJECT_NAME:-allocate_repair_r1distill_dapo}"
CKPT_ROOT="${CKPT_ROOT:-/cm/archive/tue09_new/model_hub/$PROJECT_NAME}"

# ---------------------------------------------------------------- method
G="${G:-4}"                                       # phase-1 solves per question (GRPO group)
M="${M:-1}"                                       # repairs per failed solve (0 = solve-only baseline)
LAMBDA="${LAMBDA:-1.0}"                           # L_total = L_solve + LAMBDA * L_repair
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"        # questions per step -> 64*4 solves + (#wrong solves)*M repairs
REPAIR_PROMPT_FORMAT="${REPAIR_PROMPT_FORMAT:-single_turn}"   # single_turn | multi_turn
# v3: wrong answer -> "do not solve from scratch, repair the previous answer" + "# REASON WHY ..." format;
#     no final answer -> "solve again more concisely". v1: the original "explain why it is incorrect and solve again".
REPAIR_PROMPT_VERSION="${REPAIR_PROMPT_VERSION:-v3}"
MAX_PREV_ANSWER_TOKENS="${MAX_PREV_ANSWER_TOKENS:-1024}"      # previous answer shown in the repair prompt (tail kept)
# passed for completeness; with one optimizer step per training step the ratio is always 1 (see header)
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.24}"
USE_KL_LOSS="${USE_KL_LOSS:-False}"               # off: no reference model is built
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
LR="${LR:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TOTAL_STEPS="${TOTAL_STEPS:-250}"                 # training steps (one epoch = 16,373 // 64 = 255 steps)

# ---------------------------------------------------------------- lengths and sampling
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"    # phase-1 prompts (data has none over 1024 tokens)
ROLLOUT_PROMPT_LENGTH="${ROLLOUT_PROMPT_LENGTH:-2560}"   # covers phase-2 prompts (question + previous answer + feedback)
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"       # same as the GRPO script; R1-distill often needs more
SOLVE_MAX_TOKENS="${SOLVE_MAX_TOKENS:-$MAX_RESPONSE_LENGTH}"
REPAIR_MAX_TOKENS="${REPAIR_MAX_TOKENS:-$MAX_RESPONSE_LENGTH}"
VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-$MAX_RESPONSE_LENGTH}"
SOLVE_TEMPERATURE="${SOLVE_TEMPERATURE:-1.0}"
SOLVE_TOP_P="${SOLVE_TOP_P:-1.0}"
REPAIR_TEMPERATURE="${REPAIR_TEMPERATURE:-1.0}"
REPAIR_TOP_P="${REPAIR_TOP_P:-1.0}"
VAL_N="${VAL_N:-4}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.6}"         # R1-distill's recommended decoding
VAL_TOP_P="${VAL_TOP_P:-0.95}"
# evaluate only once, after the last training step (verl always validates at the last step when test_freq > 0)
TEST_FREQ="${TEST_FREQ:-$((TOTAL_STEPS + 1))}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
SAVE_FREQ="${SAVE_FREQ:-100}"                     # checkpoint every N steps + at the last step (~27 GB each)
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-3}"
EXAMPLES_INTERVAL="${EXAMPLES_INTERVAL:-10}"      # dump solve/repair examples every N steps (allocate_repair_logs/)
SEED="${SEED:-42}"

# ---------------------------------------------------------------- GPU memory / speed
# Always on: sequence packing, gradient checkpointing, token-budget micro-batches, DP-balanced batches, chunked
# entropy, vLLM prefix caching + chunked prefill + CUDA graphs, KV cache freed while the actor trains, no ref model.
# (No bypass of old log-probs: the trainer requires them recomputed by the actor under each phase's prompt.)
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.6}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-False}"
CALCULATE_ENTROPY="${CALCULATE_ENTROPY:-True}"

# ---------------------------------------------------------------- GPUs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
N_GPUS="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"

# ---------------------------------------------------------------- environment
if [[ "${CONDA_DEFAULT_ENV:-}" != "arrlvr" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate arrlvr
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER:-ar}}"

EXP_NAME="${EXP_NAME:-ar_r1d1.5b_G${G}_M${M}_lam${LAMBDA}_bs${TRAIN_BATCH_SIZE}_lr${LR}_p${REPAIR_PROMPT_VERSION}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$CKPT_ROOT/$EXP_NAME"
TRAJECTORY_DIR="${TRAJECTORY_DIR:-/cm/archive/tue09_new/trajectories/$PROJECT_NAME/$EXP_NAME}"   # every trajectory, every step
RESPONSE_LENGTH=$(( SOLVE_MAX_TOKENS > REPAIR_MAX_TOKENS ? SOLVE_MAX_TOKENS : REPAIR_MAX_TOKENS ))
(( VAL_MAX_TOKENS > RESPONSE_LENGTH )) && RESPONSE_LENGTH=$VAL_MAX_TOKENS
SEQ_LEN=$((ROLLOUT_PROMPT_LENGTH + RESPONSE_LENGTH))
(( PPO_MAX_TOKEN_LEN_PER_GPU < SEQ_LEN )) && PPO_MAX_TOKEN_LEN_PER_GPU=$SEQ_LEN
(( LOG_PROB_MAX_TOKEN_LEN_PER_GPU < SEQ_LEN )) && LOG_PROB_MAX_TOKEN_LEN_PER_GPU=$SEQ_LEN

CMD=(
    # config: allocate_repair/config/allocate_repair_trainer.yaml (verl's ppo_trainer + the allocate_repair section)
    python -m allocate_repair.main
    # ---- data
    "data.train_files=$TRAIN_FILE"
    "data.val_files=$VAL_FILES"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.seed=$SEED"
    # ---- method
    "actor_rollout_ref.rollout.n=$G"
    "allocate_repair.M=$M"
    "allocate_repair.repair_lambda=$LAMBDA"
    "allocate_repair.solve.max_tokens=$SOLVE_MAX_TOKENS"
    "allocate_repair.solve.temperature=$SOLVE_TEMPERATURE"
    "allocate_repair.solve.top_p=$SOLVE_TOP_P"
    "allocate_repair.repair.max_tokens=$REPAIR_MAX_TOKENS"
    "allocate_repair.repair.temperature=$REPAIR_TEMPERATURE"
    "allocate_repair.repair.top_p=$REPAIR_TOP_P"
    "allocate_repair.repair.prompt_format=$REPAIR_PROMPT_FORMAT"
    "allocate_repair.repair.prompt_version=$REPAIR_PROMPT_VERSION"
    "allocate_repair.repair.max_previous_answer_tokens=$MAX_PREV_ANSWER_TOKENS"
    "allocate_repair.validation.max_tokens=$VAL_MAX_TOKENS"
    "allocate_repair.logging.examples_interval=$EXAMPLES_INTERVAL"
    "allocate_repair.logging.trajectory_dir=$TRAJECTORY_DIR"
    "allocate_repair.data_manifest=$DATA_DIR/manifest.json"
    # ---- actor
    "actor_rollout_ref.model.path=$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    "actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW"
    "actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH"
    "actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS"
    "actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF"
    "actor_rollout_ref.actor.calculate_entropy=$CALCULATE_ENTROPY"
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.use_dynamic_bsz=True
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU"
    "actor_rollout_ref.actor.optim.lr=$LR"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS"
    "actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY"
    "actor_rollout_ref.actor.grad_clip=$GRAD_CLIP"
    "actor_rollout_ref.actor.data_loader_seed=$SEED"
    "actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD"
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU"
    # ---- rollout (vLLM); the agent loop applies each phase's sampling settings and the R1 tokenization fix
    "actor_rollout_ref.rollout.agent.agent_loop_config_path=$REPO_ROOT/allocate_repair/config/agent_loops.yaml"
    "actor_rollout_ref.rollout.prompt_length=$ROLLOUT_PROMPT_LENGTH"
    "actor_rollout_ref.rollout.response_length=$RESPONSE_LENGTH"
    "actor_rollout_ref.rollout.max_model_len=$SEQ_LEN"
    "actor_rollout_ref.rollout.seed=$SEED"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS"
    "actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU"
    "actor_rollout_ref.rollout.val_kwargs.n=$VAL_N"
    "actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE"
    "actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P"
    # ---- trainer
    # file: per-step metrics as JSONL in $RUN_DIR/metrics/ (read by scripts/plot_run.py)
    "trainer.logger=[console,wandb,file]"
    "trainer.project_name=$PROJECT_NAME"
    "trainer.experiment_name=$EXP_NAME"
    "trainer.default_local_dir=$RUN_DIR"
    trainer.resume_mode=auto
    trainer.nnodes=1
    "trainer.n_gpus_per_node=$N_GPUS"
    "trainer.total_training_steps=$TOTAL_STEPS"
    # epochs only need to be enough to reach total_training_steps (too few would skip the final evaluation)
    "trainer.total_epochs=$TOTAL_STEPS"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.max_actor_ckpt_to_keep=$MAX_CKPT_TO_KEEP"
    "trainer.val_before_train=$VAL_BEFORE_TRAIN"
    +ray_kwargs.ray_init.address=local
    # keep Hydra's own run folder out of the repo
    "hydra.run.dir=$RUN_DIR/hydra"
    "$@"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    cd "$REPO_ROOT"
    "${CMD[@]}" --cfg job --resolve
    exit 0
fi

# ---------------------------------------------------------------- checks before spending GPU hours
[[ -f "$MODEL_PATH/config.json" ]] || { echo "model not found: $MODEL_PATH" >&2; exit 1; }
[[ -f "$TRAIN_FILE" ]] || { echo "train data not found: $TRAIN_FILE" >&2; exit 1; }
(( G >= 2 && M >= 0 )) || { echo "need G >= 2 and M >= 0 (got G=$G, M=$M)" >&2; exit 1; }
# vLLM needs ROLLOUT_GPU_MEM_UTIL of each GPU free at startup; stop now instead of crashing minutes later
BUSY_GPUS="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" |
    awk -F', ' '$2 >= 1024 {printf "%s(%s MiB) ", $1, $2}')"
if [[ -n "$BUSY_GPUS" ]]; then
    echo "GPUs already in use: $BUSY_GPUS-> wait for them to be free, or pick others with CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi

mkdir -p "$RUN_DIR/metrics"
# verl's file logger truncates its file when it starts, so every launch (first start or resume) gets its own file
export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics/metrics_$(date +%Y%m%d_%H%M%S).jsonl"
{
    echo "# $(date -Iseconds) on $(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${CMD[@]}"
    echo
} >>"$RUN_DIR/command.sh"
echo "[allocate-repair] $EXP_NAME -> $RUN_DIR"
echo "[allocate-repair] GPUs=$CUDA_VISIBLE_DEVICES  G=$G  M=$M  lambda=$LAMBDA  repair_prompt=$REPAIR_PROMPT_VERSION  batch=$TRAIN_BATCH_SIZE  lr=$LR" \
    "max_tokens(solve/repair/val)=$SOLVE_MAX_TOKENS/$REPAIR_MAX_TOKENS/$VAL_MAX_TOKENS"
cd "$REPO_ROOT"
"${CMD[@]}" 2>&1 | tee -a "$RUN_DIR/train.log"
