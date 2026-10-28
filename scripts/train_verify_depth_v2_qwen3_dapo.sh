#!/usr/bin/env bash
# Scaling Verification Depth, VERSION 2 of the note (/cm/shared/tue09_new/idea/Scaling_Verification_Depth.pdf, uploaded
# 27 Oct 2026 22:59) on DAPO-Math-17k, Qwen3-1.7B (thinking mode, no SFT), 4x A100-40GB. Same configuration as
# scripts/train_verify_depth_qwen3_dapo.sh (version 1); the method follows version 2 of the note:
#   G = solutions per question (phase 1, GRPO), M = verification depth (phase 2: one chain of M rounds per solution).
#   Round 1 sees (question, solution); round d >= 2 sees (question, solution, A_{d-1}), A_{d-1} = the previous round's
#   text after </think> + the running conclusion c_{d-1} about the solution (c_1 = v_1; c_d = c_{d-1} if v_d = 1
#   else 1 - c_{d-1}; computed, never generated). No reasoning trace, earlier round or reference answer is shown.
#   Verdicts: v_1 judges the solution, v_d judges v_{d-1}; targets v*_1 = Verify(S, R), v*_d = 1[v_{d-1} = v*_{d-1}].
#   Every round is its own training example: reward r_d = 1[v_d = v*_d], advantage r_d - 1/2 (fixed baseline, no
#   group normalization) on all its generated tokens. J = J_GRPO + J_verify, one optimizer step per training step.
# Kept from version 1 (env variables below): FORCED_VERDICT (a round without a verdict gets a forced verdict header),
# FORMAT_PENALTY (format reward: a round whose verdict the policy did not write itself in the requested form gets
# reward FORMAT_PENALTY_REWARD = -1/M instead of 1[v_d = v*_d]), UNEXTRACTABLE_POLICY=skip.
#
#   bash scripts/train_verify_depth_v2_qwen3_dapo.sh                          # G=4, M=3 (defaults below)
#   G=8 M=5 bash scripts/train_verify_depth_v2_qwen3_dapo.sh                  # change settings with env variables
#   DRY_RUN=1 bash scripts/train_verify_depth_v2_qwen3_dapo.sh                # print the resolved config, no training
#   EXP_NAME=<existing run> bash scripts/train_verify_depth_v2_qwen3_dapo.sh  # resume from its latest checkpoint
#
# Results go to the HF dataset repo every 10 minutes (HF_SYNC=1, HF_SYNC_INTERVAL_MIN; scripts/hf_sync_results.py,
# token from HF_TOKEN_FILE): <repo>/results/$PROJECT_NAME/$EXP_NAME (figures, metrics.csv, key_metrics.md).
#
# Cost: phase 2 adds up to G * M generations per question (here 4 * 3 = 12 on top of 4 solutions), only for
# solutions that finished with a final answer. Code: allocate_repair/verify_depth.py (pure logic),
# allocate_repair/verify_trainer_v2.py (VerifyDepthV2Trainer on top of verify_trainer.py).
# Training metrics: verify/* (depth-wise match rates, verdict accuracy on correct / wrong solutions, ...); phase-2 loss
# terms are logged as train/repair/* (the shared phase-2 slot of the two-phase trainer).
#
# Every trajectory of every step (exact prompt + full raw output incl. thinking, verdicts, targets, rewards) goes to
# $TRAJECTORY_DIR/step_NNNNN.jsonl.gz (default /cm/archive/tue09_new/trajectories/$PROJECT_NAME/$EXP_NAME).
# Everything else of a run goes to $CKPT_ROOT/$EXP_NAME: checkpoints (global_step_N/), train.log, command.sh, wandb/,
# metrics/*.jsonl, allocate_repair_logs/ (per-step metrics, verification-chain examples, run info).
# Progress figures: python scripts/plot_run.py <EXP_NAME>  (-> result/<EXP_NAME>/)
# Default: 250 training steps, then one evaluation on all VAL_FILES. A run is long: start it inside tmux.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data/processed"

# ---------------------------------------------------------------- model / data / output
MODEL_PATH="${MODEL_PATH:-/cm/archive/tue09/model_hub/Qwen/Qwen3-1.7B}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"   # DAPO-Math-17k (deduplicated), 16,373 questions
# question-only (first-attempt) accuracy on: held-out DAPO questions, AIME24, AIME25, MATH-500
VAL_FILES="${VAL_FILES:-[$DATA_DIR/val_dapo_heldout.parquet,$DATA_DIR/eval_aime24.parquet,$DATA_DIR/eval_aime25.parquet,$DATA_DIR/eval_math500.parquet]}"
PROJECT_NAME="${PROJECT_NAME:-verify_depth_qwen3_dapo}"
CKPT_ROOT="${CKPT_ROOT:-/cm/archive/tue09_new/model_hub/$PROJECT_NAME}"

# ---------------------------------------------------------------- method
G="${G:-4}"                                       # phase-1 solutions per question (GRPO group)
M="${M:-3}"                                       # verification depth: rounds of the one chain per solution
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"        # questions per step
# solutions without a final-response section (no </think>, e.g. truncated): skip = not verified | placeholder
UNEXTRACTABLE_POLICY="${UNEXTRACTABLE_POLICY:-skip}"
# a round without a verdict after </think> gets "**Verdict (CORRECT or INCORRECT):**" appended (masked out of the
# loss) and the model writes the verdict with the VERDICT_RESERVE_TOKENS it held back
FORCED_VERDICT="${FORCED_VERDICT:-True}"
VERDICT_RESERVE_TOKENS="${VERDICT_RESERVE_TOKENS:-16}"
# format reward: a round with a format violation (no verdict written by the policy itself: missing, or only after the
# forced header) gets reward FORMAT_PENALTY_REWARD (default -1/M) instead of 1[v_d = v*_d]; False: the note's 1 or 0
FORMAT_PENALTY="${FORMAT_PENALTY:-True}"
FORMAT_PENALTY_REWARD="${FORMAT_PENALTY_REWARD:-null}"   # null = -1/M
ADVANTAGE_BASELINE="${ADVANTAGE_BASELINE:-0.5}"          # eq. (13): advantage = round reward - 1/2
MAX_SOLUTION_TOKENS="${MAX_SOLUTION_TOKENS:-1024}"   # solution text after </think> shown to the verifier (tail kept)
MAX_ROUND_TOKENS="${MAX_ROUND_TOKENS:-512}"          # previous round's text after </think> shown (tail kept)
USE_KL_LOSS="${USE_KL_LOSS:-False}"               # off: no reference model is built
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
LR="${LR:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TOTAL_STEPS="${TOTAL_STEPS:-250}"                 # training steps (one epoch = 16,373 // 64 = 255 steps)

# ---------------------------------------------------------------- lengths and sampling
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"    # phase-1 prompts (data has none over 1024 tokens)
# covers verification prompts: question + solution (<= MAX_SOLUTION_TOKENS) + previous round (<= MAX_ROUND_TOKENS)
ROLLOUT_PROMPT_LENGTH="${ROLLOUT_PROMPT_LENGTH:-3584}"
# Qwen3-1.7B thinks long: offline (40 DAPO questions x 4) 8192 tokens -> 56% of solutions finish, 38% correct;
# 4096 -> 28% finish, 16% correct. Verification rounds get 4096 (a cut-off round still ends with a forced verdict).
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
SOLVE_MAX_TOKENS="${SOLVE_MAX_TOKENS:-$MAX_RESPONSE_LENGTH}"
VERIFY_MAX_TOKENS="${VERIFY_MAX_TOKENS:-4096}"
VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-$MAX_RESPONSE_LENGTH}"
SOLVE_TEMPERATURE="${SOLVE_TEMPERATURE:-1.0}"
SOLVE_TOP_P="${SOLVE_TOP_P:-1.0}"
VERIFY_TEMPERATURE="${VERIFY_TEMPERATURE:-1.0}"
VERIFY_TOP_P="${VERIFY_TOP_P:-1.0}"
VAL_N="${VAL_N:-4}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.6}"         # Qwen3's recommended thinking-mode decoding
VAL_TOP_P="${VAL_TOP_P:-0.95}"
VAL_TOP_K="${VAL_TOP_K:-20}"
# evaluate only once, after the last training step (verl always validates at the last step when test_freq > 0)
TEST_FREQ="${TEST_FREQ:-$((TOTAL_STEPS + 1))}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
SAVE_FREQ="${SAVE_FREQ:-100}"                     # checkpoint every N steps + at the last step (~27 GB each)
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-3}"
EXAMPLES_INTERVAL="${EXAMPLES_INTERVAL:-10}"      # dump solution / verification-chain examples every N steps
SEED="${SEED:-42}"

# ---------------------------------------------------------------- GPU memory / speed
# Same as the allocate-repair script: sequence packing, gradient checkpointing, token-budget micro-batches,
# DP-balanced batches, chunked entropy, vLLM prefix caching + chunked prefill + CUDA graphs, KV cache freed while the
# actor trains, no ref model. All M rounds are generated with the phase-1 weights before the replicas sleep.
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
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER:-vd}}"
HF_SYNC="${HF_SYNC:-1}"                                   # push results to the HF repo while training
HF_SYNC_INTERVAL_MIN="${HF_SYNC_INTERVAL_MIN:-10}"
HF_REPO="${HF_REPO:-LeTue09/allocate-repair-RLVR}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/cm/shared/tue09_new/key.txt}"

EXP_NAME="${EXP_NAME:-vd2_qwen3-1.7b_G${G}_M${M}_bs${TRAIN_BATCH_SIZE}_lr${LR}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$CKPT_ROOT/$EXP_NAME"
TRAJECTORY_DIR="${TRAJECTORY_DIR:-/cm/archive/tue09_new/trajectories/$PROJECT_NAME/$EXP_NAME}"   # every trajectory, every step
RESPONSE_LENGTH=$(( SOLVE_MAX_TOKENS > VERIFY_MAX_TOKENS ? SOLVE_MAX_TOKENS : VERIFY_MAX_TOKENS ))
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
    # ---- method: phase 2 = verification chains, version 2 of the note
    allocate_repair.method=verify_depth_v2
    "actor_rollout_ref.rollout.n=$G"
    # the note's M = verification depth; J = J_GRPO + J_verify (weight 1)
    "allocate_repair.M=$M"
    allocate_repair.repair_lambda=1.0
    "allocate_repair.verify.max_tokens=$VERIFY_MAX_TOKENS"
    "allocate_repair.verify.temperature=$VERIFY_TEMPERATURE"
    "allocate_repair.verify.top_p=$VERIFY_TOP_P"
    "allocate_repair.verify.max_round_tokens=$MAX_ROUND_TOKENS"
    "allocate_repair.verify.unextractable_solution_policy=$UNEXTRACTABLE_POLICY"
    "allocate_repair.verify.forced_verdict=$FORCED_VERDICT"
    "allocate_repair.verify.verdict_reserve_tokens=$VERDICT_RESERVE_TOKENS"
    "allocate_repair.verify.format_penalty=$FORMAT_PENALTY"
    "allocate_repair.verify.format_penalty_reward=$FORMAT_PENALTY_REWARD"
    "allocate_repair.verify.advantage_baseline=$ADVANTAGE_BASELINE"
    # the solution shown to the verifier is cut like a repair's previous answer (text after </think>, tail kept)
    "allocate_repair.repair.max_previous_answer_tokens=$MAX_SOLUTION_TOKENS"
    "allocate_repair.repair.max_tokens=$VERIFY_MAX_TOKENS"
    "allocate_repair.solve.max_tokens=$SOLVE_MAX_TOKENS"
    "allocate_repair.solve.temperature=$SOLVE_TEMPERATURE"
    "allocate_repair.solve.top_p=$SOLVE_TOP_P"
    "allocate_repair.validation.max_tokens=$VAL_MAX_TOKENS"
    "allocate_repair.logging.examples_interval=$EXAMPLES_INTERVAL"
    "allocate_repair.logging.trajectory_dir=$TRAJECTORY_DIR"
    "allocate_repair.data_manifest=$DATA_DIR/manifest.json"
    # Qwen3: config.json model_type "qwen3" selects verl's Qwen prompt builder, which matches its chat template
    # (the config default deepseek_v3 is the R1-distill fix)
    allocate_repair.token_builder_model_type=null
    # ---- actor
    "actor_rollout_ref.model.path=$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
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
    # ---- rollout (vLLM); the agent loop applies each phase's sampling settings and the forced verdicts
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
    "actor_rollout_ref.rollout.val_kwargs.top_k=$VAL_TOP_K"
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
(( G >= 2 && M >= 1 )) || { echo "need G >= 2, M >= 1 (got G=$G, M=$M)" >&2; exit 1; }
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
echo "[verify-depth-v2] $EXP_NAME -> $RUN_DIR  (trajectories: $TRAJECTORY_DIR)"
echo "[verify-depth-v2] GPUs=$CUDA_VISIBLE_DEVICES  G=$G (solutions/question)  M=$M (verification depth, 1 chain/solution)" \
    "forced_verdict=$FORCED_VERDICT  format_penalty=$FORMAT_PENALTY (reward $FORMAT_PENALTY_REWARD, null = -1/M)" \
    "baseline=$ADVANTAGE_BASELINE  batch=$TRAIN_BATCH_SIZE  lr=$LR" \
    "max_tokens(solve/verify/val)=$SOLVE_MAX_TOKENS/$VERIFY_MAX_TOKENS/$VAL_MAX_TOKENS"
cd "$REPO_ROOT"
# results -> HF every HF_SYNC_INTERVAL_MIN minutes; the sync stops by itself after the final upload (or when no new
# step was logged for 6 h). Not started twice for the same run (e.g. on resume).
if [[ "$HF_SYNC" == "1" ]] && ! pgrep -f "hf_sync_results.py $RUN_DIR" >/dev/null; then
    # the sync needs the Hub: not HF_HUB_OFFLINE (set above for training)
    nohup env -u HF_HUB_OFFLINE python scripts/hf_sync_results.py "$RUN_DIR" --repo "$HF_REPO" --token-file "$HF_TOKEN_FILE" \
        --interval-min "$HF_SYNC_INTERVAL_MIN" >>"$RUN_DIR/hf_sync.log" 2>&1 &
    echo "[verify-depth-v2] HF sync every $HF_SYNC_INTERVAL_MIN min -> $HF_REPO (pid $!, log: $RUN_DIR/hf_sync.log)"
fi
"${CMD[@]}" 2>&1 | tee -a "$RUN_DIR/train.log"
