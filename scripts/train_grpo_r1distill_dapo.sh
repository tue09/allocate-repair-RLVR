#!/usr/bin/env bash
# GRPO with the DAPO token-level loss + clip-higher on DAPO-Math-17k, DeepSeek-R1-Distill-Qwen-1.5B, 4x A100-40GB.
# Plain verl (verl.trainer.main_ppo); no repair phase.
#
#   bash scripts/train_grpo_r1distill_dapo.sh                                   # defaults below
#   LR=2e-6 G=16 bash scripts/train_grpo_r1distill_dapo.sh                      # change settings with env variables
#   bash scripts/train_grpo_r1distill_dapo.sh trainer.save_freq=50              # extra Hydra overrides at the end
#   DRY_RUN=1 bash scripts/train_grpo_r1distill_dapo.sh                         # print the resolved config, no training
#   EXP_NAME=<existing run> bash scripts/train_grpo_r1distill_dapo.sh           # resume from its latest checkpoint
#
# Everything of a run goes to $CKPT_ROOT/$EXP_NAME: checkpoints (global_step_N/), train.log, command.sh, wandb/,
# metrics/*.jsonl. Progress figures: python scripts/plot_run.py  (-> result/<EXP_NAME>/)
# Checkpoints are FSDP shards (model + optimizer); to get a normal HF model from one:
#   python -m verl.model_merger merge --backend fsdp --local_dir <run>/global_step_N/actor --target_dir <out dir>
# Default: 250 training steps, then one evaluation on all VAL_FILES (no evaluation before / during training).
# A run is long: start it inside tmux.
#
# Why not scripts/run_solve_only.sh: the allocate_repair trainer only supports seq-mean-token-mean
# (allocate_repair/losses.py; trainer.py asserts it) and does one optimizer step per training step.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data/processed"

# ---------------------------------------------------------------- model / data / output
MODEL_PATH="${MODEL_PATH:-/cm/archive/tue09/model_hub/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"   # DAPO-Math-17k (deduplicated), 16,373 questions
# first-attempt accuracy on: held-out DAPO questions, AIME24, AIME25, MATH-500
VAL_FILES="${VAL_FILES:-[$DATA_DIR/val_dapo_heldout.parquet,$DATA_DIR/eval_aime24.parquet,$DATA_DIR/eval_aime25.parquet,$DATA_DIR/eval_math500.parquet]}"
PROJECT_NAME="${PROJECT_NAME:-grpo_r1distill_dapo}"
CKPT_ROOT="${CKPT_ROOT:-/cm/archive/tue09_new/model_hub/$PROJECT_NAME}"
# agent loop from allocate_repair/config/agent_loops.yaml: r1_single_turn_agent selects verl's DeepSeek tokenization
# builder (R1-distill's config.json says "qwen2", but its tokenizer has no <|im_end|>). Real Qwen models:
# AGENT_LOOP=single_turn_agent
AGENT_LOOP="${AGENT_LOOP:-r1_single_turn_agent}"

# ---------------------------------------------------------------- GRPO + DAPO tricks
G="${G:-8}"                                       # responses per question (group size)
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"        # questions per step -> 64 * 8 = 512 responses
# 4 optimizer updates per step. Must be < TRAIN_BATCH_SIZE for the clip to matter: with one update per step the new
# policy equals the rollout policy, every ratio is 1 and clip-higher never acts.
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"      # DAPO token-level loss: mean over all tokens of the mini-batch
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"           # DAPO clip-higher: ratio clipped to [1-0.2, 1+0.24]
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.24}"
# DAPO drops the KL term; with it off verl builds no reference model (less memory, one forward pass fewer per step)
USE_KL_LOSS="${USE_KL_LOSS:-False}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"
LR="${LR:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TOTAL_STEPS="${TOTAL_STEPS:-250}"                 # training steps (one epoch = 16,373 // 64 = 255 steps)

# ---------------------------------------------------------------- lengths and sampling
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"    # data has no prompt over 1024 tokens
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"  # R1-distill thinks long; 1-2k truncates almost every answer
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
VAL_N="${VAL_N:-4}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.6}"         # R1-distill's recommended decoding
VAL_TOP_P="${VAL_TOP_P:-0.95}"
# evaluate only once, after the last training step: verl always validates at the last step when test_freq > 0, so a
# test_freq larger than the run never triggers in between. TEST_FREQ=N also evaluates every N steps.
TEST_FREQ="${TEST_FREQ:-$((TOTAL_STEPS + 1))}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
SAVE_FREQ="${SAVE_FREQ:-100}"                     # checkpoint every N steps + at the last step (~27 GB each)
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-3}"
SEED="${SEED:-42}"

# ---------------------------------------------------------------- GPU memory / speed
# Always on: sequence packing (use_remove_padding), gradient checkpointing, token-budget micro-batches
# (use_dynamic_bsz), DP-balanced batches, chunked entropy, vLLM prefix caching (the 8 samples share a prompt) +
# chunked prefill + CUDA graphs, vLLM KV cache freed while the actor trains (free_cache_engine), no ref model (KL off).
ROLLOUT_TP="${ROLLOUT_TP:-1}"                     # 1.5B fits one GPU: 4 independent vLLM replicas are fastest
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.6}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"
# tokens per GPU per micro-batch (>= prompt + response). Values of the repo's tested config; raise for speed if
# memory allows, lower if out of memory
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"           # True = actor weights on CPU between uses (less GPU memory, slower)
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-False}"   # True = Adam states on CPU (saves ~3 GB/GPU here, slower)
SP_SIZE="${SP_SIZE:-1}"                           # Ulysses sequence parallel: 2 halves activation memory per GPU
CALCULATE_ENTROPY="${CALCULATE_ENTROPY:-True}"    # log policy entropy (what clip-higher is meant to keep up)
# Opt-in, not tested on this setup yet -- try them in a short run first:
USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-False}"   # log-probs without materializing the full 152k-vocab logits
USE_LIGER="${USE_LIGER:-False}"                   # Liger fused kernels (RMSNorm, SwiGLU, RoPE) for Qwen2
# reuse vLLM's log-probs as the PPO "old" log-probs instead of recomputing them with the actor: skips one forward
# pass over all 512 responses per step, but the ratio is then taken against the vLLM numerics
BYPASS_OLD_LOGPROB="${BYPASS_OLD_LOGPROB:-False}"

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
export WANDB_MODE="${WANDB_MODE:-offline}"        # upload later: wandb sync <run>/wandb/wandb/offline-run-*
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER:-grpo}}"

EXP_NAME="${EXP_NAME:-r1d1.5b_G${G}_bs${TRAIN_BATCH_SIZE}_mb${PPO_MINI_BATCH_SIZE}_lr${LR}_clip${CLIP_RATIO_LOW}-${CLIP_RATIO_HIGH}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$CKPT_ROOT/$EXP_NAME"
TRAJECTORY_DIR="${TRAJECTORY_DIR:-/cm/archive/tue09_new/trajectories/$PROJECT_NAME/$EXP_NAME}"   # every trajectory, every step
SEQ_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
(( PPO_MAX_TOKEN_LEN_PER_GPU < SEQ_LEN )) && PPO_MAX_TOKEN_LEN_PER_GPU=$SEQ_LEN
(( LOG_PROB_MAX_TOKEN_LEN_PER_GPU < SEQ_LEN )) && LOG_PROB_MAX_TOKEN_LEN_PER_GPU=$SEQ_LEN

CMD=(
    python -m verl.trainer.main_ppo
    # legacy single-controller trainer (the v1 trainer needs TransferQueue, not installed)
    trainer.use_v1=False
    # ---- data
    "data.train_files=$TRAIN_FILE"
    "data.val_files=$VAL_FILES"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=True
    "data.seed=$SEED"
    data.dataloader_num_workers=4
    # ---- algorithm: GRPO advantages (group-normalized), no KL in the reward
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=True
    algorithm.use_kl_in_reward=False
    "algorithm.rollout_correction.bypass_mode=$BYPASS_OLD_LOGPROB"
    # ---- model
    "actor_rollout_ref.model.path=$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    "actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS"
    "actor_rollout_ref.model.use_liger=$USE_LIGER"
    # ---- actor: DAPO token-level loss + clip-higher
    "actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE"
    "actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW"
    "actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH"
    "actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS"
    "actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    "actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF"
    "actor_rollout_ref.actor.calculate_entropy=$CALCULATE_ENTROPY"
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.ppo_epochs=1
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU"
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE"
    "actor_rollout_ref.actor.optim.lr=$LR"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS"
    "actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY"
    "actor_rollout_ref.actor.grad_clip=$GRAD_CLIP"
    "actor_rollout_ref.actor.data_loader_seed=$SEED"
    "actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD"
    # ---- reference model (only built when USE_KL_LOSS=True)
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU"
    "actor_rollout_ref.ref.ulysses_sequence_parallel_size=$SP_SIZE"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    # ---- rollout (vLLM)
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    "actor_rollout_ref.rollout.agent.agent_loop_config_path=$REPO_ROOT/allocate_repair/config/agent_loops.yaml"
    "actor_rollout_ref.rollout.agent.default_agent_loop=$AGENT_LOOP"
    "actor_rollout_ref.rollout.n=$G"
    "actor_rollout_ref.rollout.temperature=$TEMPERATURE"
    "actor_rollout_ref.rollout.top_p=$TOP_P"
    actor_rollout_ref.rollout.top_k=-1
    "actor_rollout_ref.rollout.seed=$SEED"
    "actor_rollout_ref.rollout.max_model_len=$SEQ_LEN"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS"
    "actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU"
    "actor_rollout_ref.rollout.val_kwargs.n=$VAL_N"
    "actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE"
    "actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P"
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    # ---- reward: math_dapo checker, 0/1 (same as the allocate-repair runs)
    reward.num_workers=8
    reward.custom_reward_function.path=pkg://allocate_repair.reward_fn
    reward.custom_reward_function.name=compute_score
    # ---- trainer
    trainer.balance_batch=True
    trainer.critic_warmup=0
    # file: per-step metrics as JSONL in $RUN_DIR/metrics/ (read by scripts/plot_run.py)
    "trainer.logger=[console,wandb,file]"
    "trainer.project_name=$PROJECT_NAME"
    "trainer.experiment_name=$EXP_NAME"
    "trainer.default_local_dir=$RUN_DIR"
    # verl's rollout dump: every sample of every step (prompt + full output, special tokens dropped) -> <step>.jsonl
    "trainer.rollout_data_dir=$TRAJECTORY_DIR"
    trainer.resume_mode=auto
    trainer.nnodes=1
    "trainer.n_gpus_per_node=$N_GPUS"
    "trainer.total_training_steps=$TOTAL_STEPS"
    # the run stops at total_training_steps; epochs only need to be enough to get there (every epoch has >= 1 step).
    # Too few epochs would end the loop before the "last step", skipping the final evaluation and checkpoint.
    "trainer.total_epochs=$TOTAL_STEPS"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.max_actor_ckpt_to_keep=$MAX_CKPT_TO_KEEP"
    "trainer.val_before_train=$VAL_BEFORE_TRAIN"
    trainer.log_val_generations=0
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
(( TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE == 0 )) || {
    echo "TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE) must be a multiple of PPO_MINI_BATCH_SIZE ($PPO_MINI_BATCH_SIZE)" >&2
    exit 1
}
# vLLM needs ROLLOUT_GPU_MEM_UTIL of each GPU free at startup; stop now instead of crashing minutes later
BUSY_GPUS="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" |
    awk -F', ' '$2 >= 1024 {printf "%s(%s MiB) ", $1, $2}')"
if [[ -n "$BUSY_GPUS" ]]; then
    echo "GPUs already in use: $BUSY_GPUS-> wait for them to be free, or pick others with CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi

mkdir -p "$RUN_DIR/metrics"
export WANDB_DIR="$RUN_DIR"
# verl's file logger truncates its file when it starts, so every launch (first start or resume) gets its own file
export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics/metrics_$(date +%Y%m%d_%H%M%S).jsonl"
{
    echo "# $(date -Iseconds) on $(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${CMD[@]}"
    echo
} >>"$RUN_DIR/command.sh"
echo "[grpo] $EXP_NAME -> $RUN_DIR"
echo "[grpo] GPUs=$CUDA_VISIBLE_DEVICES  G=$G  batch=$TRAIN_BATCH_SIZE  mini=$PPO_MINI_BATCH_SIZE  lr=$LR" \
    "loss=$LOSS_AGG_MODE  clip=[$CLIP_RATIO_LOW,$CLIP_RATIO_HIGH]  kl=$USE_KL_LOSS  max_response=$MAX_RESPONSE_LENGTH"
cd "$REPO_ROOT"
"${CMD[@]}" 2>&1 | tee -a "$RUN_DIR/train.log"
