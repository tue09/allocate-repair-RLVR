# allocate-repair RLVR

Two-phase GRPO on [verl](https://github.com/verl-project/verl) (pinned to **v0.9.1**, commit `1876b06`):

1. **Solve**: sample G responses per question, score with a binary 0/1 reward.
2. **Explain and repair**: for every incorrect response, show the model the question, its previous final answer
   (reasoning inside the think block removed) and "this answer is incorrect", then sample M repairs.
3. **One update**: GRPO advantages per phase (question groups / failed-parent groups) and a single optimizer step
   on `L_total = L_solve + λ · L_repair`.

Defaults: DeepSeek-R1-Distill-Qwen-1.5B, DAPO-Math-17k, 4× A100-40GB (GPUs 0–3), W&B offline.
The full method spec is in [allocate-repair-prompt.md](allocate-repair-prompt.md).

## Quick start on the server

Copy this whole folder (it includes the prepared data and the pinned verl checkout) to the server, then:

```bash
cd /cm/shared/tue09_new/allocate_repair_RLVR   # wherever you copied it

bash scripts/setup_env.sh            # once, on the GPU node: creates conda env "arrlvr" (~10-40 min)
bash scripts/run_unit_tests.sh       # CPU checks (~1 min)
bash scripts/run_smoke_test.sh       # 4-step end-to-end run on GPUs 0-3 (~15-30 min)

bash scripts/run_solve_only.sh       # baseline:  G=8, M=0
bash scripts/run_allocate_repair.sh  # method:    G=8, M=4, lambda=1.0
```

The run scripts activate `arrlvr` themselves, use `CUDA_VISIBLE_DEVICES=0,1,2,3`, and read the model from
`/cm/archive/tue09/model_hub/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`. Change anything with environment variables
or extra Hydra overrides:

```bash
M=8 LAMBDA=0.5 TRAIN_BATCH_SIZE=32 bash scripts/run_allocate_repair.sh trainer.total_training_steps=200
MODEL_PATH=/path/to/other/model bash scripts/run_smoke_test.sh
```

Long runs: start them inside `tmux`/`screen` (or `nohup ... &`); everything is also written to `outputs/<exp>/train.log`.

If you clone from GitHub instead of copying the folder, the data is not in git:

```bash
git clone --recursive https://github.com/tue09/allocate-repair-RLVR.git
cd allocate-repair-RLVR && bash scripts/setup_env.sh && conda activate arrlvr
HF_HUB_OFFLINE=0 bash scripts/download_data.sh   # needs huggingface.co access
```

## Environment

`scripts/setup_env.sh` reads the NVIDIA driver version (shown in the `nvidia-smi` header) and picks a build:

| driver | variant | what gets installed |
|---|---|---|
| ≥ 580 | `cu130` | verl's tested stack: torch 2.11.0+cu130, vLLM 0.24.0, prebuilt flash-attn 2.8.3 (verl wheelhouse). No compiling. |
| 525–579 | `cu129` | torch 2.11.0+cu129, vLLM 0.24.0+cu129 release wheel, flash-attn 2.8.3 built from source for sm80 (~10–30 min; nvcc 12.9 from conda unless `CUDA_HOME` has nvcc 12.x). |

Force a variant with `CUDA_VARIANT=cu129 bash scripts/setup_env.sh`. Every other package is pinned to verl's
`uv.lock` through `requirements/constraints-common.txt` (regenerate with `scripts/make_constraints.py`). The script
ends with `python scripts/check_env.py`, which verifies versions, the GPUs, a flash-attn kernel and the verl commit.
Run setup on a GPU node; on a login node without GPUs the final CUDA check fails.

## Data (included under `data/`, git-ignored)

`scripts/prepare_data.py` downloads pinned dataset revisions and writes `data/processed/` plus `manifest.json`
(revisions, file hashes, counts, policies).

| file | rows | source |
|---|---|---|
| `train.parquet` | 16,373 | DAPO-Math-17k @ `6587709` |
| `val_dapo_heldout.parquet` | 500 | DAPO-Math-17k, held out |
| `eval_aime24.parquet` | 30 | BytedTsinghua-SIA/AIME-2024 @ `aa49075` |
| `eval_aime25.parquet` | 30 | MathArena/aime_2025 @ `c94da77` |
| `eval_math500.parquet` | 500 | HuggingFaceH4/MATH-500 @ `6e4ed1a` |
| `smoke_train.parquet` / `smoke_val.parquet` | 64 / 24 | subsets for the smoke test |

How duplicates in DAPO-Math-17k are handled:
- 1,791,700 raw rows = 17,917 question ids × exactly 100 copies. The copies are collapsed.
- 17,398 unique question texts remain, or 16,901 after normalization (NFKC, lower-case, whitespace removed).
- 1,001 ids duplicate another id's normalized question and are merged.
- 15 normalized groups (30 ids) have ground truths that disagree; these groups are dropped.
- 5 questions overlapping MATH-500 and 8 prompts over 1,024 tokens are removed.
- The held-out split is drawn at the normalized-question level, so it is disjoint from train by construction.

All prompts use the DAPO answer format (`Answer: $Answer`). Training and evaluation are both scored by verl's
`math_dapo` checker.

## What one training step does

| step | where |
|---|---|
| Phase 1: G samples per question via verl's agent loop / vLLM; 0/1 reward streamed by verl's reward loop | `trainer.py::_rollout_two_phases`, `reward_fn.py` |
| Answer extraction after the last `</think>` (token id from the tokenizer); a response without it is an *extraction failure* (logged, placeholder text shown, never the reasoning) | `repair.py::AnswerExtractor` |
| Repair prompt = original user message verbatim + previous answer + feedback; built without access to the ground truth or other responses | `repair.py::RepairPromptBuilder`, `build_repair_requests` |
| Phase 2: M samples per failed response, generated **before** the vLLM replicas sleep (same weights as phase 1) | `trainer.py::_rollout_two_phases` |
| Advantages: verl's GRPO estimator (unbiased std, ε=1e-6) with group ids `solve::<question>` and `repair::<parent response>`; identical-reward groups get 0 | `trainer.py::fit`, `batching.py` |
| old log-probs (and optional ref log-probs) recomputed by the FSDP actor under each trajectory's own prompt | verl `_compute_old_log_prob` |
| Loss: verl's clipped policy loss, `seq-mean-token-mean` within each phase, normalized by the *global* phase trajectory count; `L_solve + λ·L_repair`; optional KL once per phase; one mini-batch = one optimizer step | `losses.py`, `workers.py`, `trainer.py::_update_actor_two_phase` |

The integration subclasses verl's `RayPPOTrainer` (the legacy single-controller path, `trainer.use_v1=false`). The
v1 trainer streams prompts through a TransferQueue replay buffer, which does not fit a second generation phase that
depends on the first. verl's rollout, agent loop, reward loop, advantage estimator, policy loss, FSDP engine,
checkpointing and tracking are reused unchanged. The only additions are the worker subclass that installs the
phase-weighted loss and a small agent loop that applies per-phase sampling settings.

## Configuration

Main knobs (environment variables in `scripts/common.sh`, defaults in brackets):

| variable | meaning |
|---|---|
| `G` [8], `M` [4], `LAMBDA` [1.0] | samples per question, repairs per failed response (0 = solve-only), repair loss weight |
| `TRAIN_BATCH_SIZE` [64], `LR` [1e-6], `SEED` [42] | questions per optimizer step, learning rate, seed |
| `SOLVE_MAX_TOKENS` / `REPAIR_MAX_TOKENS` / `VAL_MAX_TOKENS` [8192] | per-phase generation limits |
| `SOLVE_TEMPERATURE`, `SOLVE_TOP_P`, `REPAIR_TEMPERATURE`, `REPAIR_TOP_P` [1.0] | per-phase sampling |
| `MAX_PREV_ANSWER_TOKENS` [1024], `ROLLOUT_PROMPT_LENGTH` [2560] | previous-answer cap in the repair prompt (tail kept), prompt budget |
| `VAL_N` [4], `VAL_TEMPERATURE` [0.6], `VAL_TOP_P` [0.95] | evaluation sampling (first attempt only) |
| `TEST_FREQ` [10], `SAVE_FREQ` [50], `TOTAL_EPOCHS` [1], `TOTAL_STEPS` | evaluation / checkpoint interval, run length |
| `MODEL_PATH`, `DATA_DIR`, `OUTPUT_ROOT`, `CUDA_VISIBLE_DEVICES` | paths and GPUs |
| `ROLLOUT_GPU_MEM_UTIL` [0.6], `OPTIMIZER_OFFLOAD`, `PARAM_OFFLOAD` [False] | memory |

Everything else (repair prompt format and wording, extraction mode, failure policy, KL, clipping, …) is in
`allocate_repair/config/allocate_repair_trainer.yaml` and can be overridden on the command line, e.g.
`allocate_repair.repair.prompt_format=multi_turn` or `actor_rollout_ref.actor.use_kl_loss=true`.

## Outputs and metrics

Each run writes to `outputs/<experiment>/`:

- `train.log`, `command.sh` (exact command line)
- `wandb/`: offline W&B run. Upload later with `wandb sync outputs/<exp>/wandb/wandb/offline-run-*`
- `allocate_repair_logs/metrics.jsonl`: every step's metrics, with raw counts next to rates and `null` when unavailable
- `allocate_repair_logs/examples.jsonl`: bounded samples (step 1 and every `EXAMPLES_INTERVAL`). Each holds question
  id and text, phase-1 outputs/rewards, the exact repair prompt (messages and rendered), phase-2 outputs/rewards,
  parent / group ids and token counts.
- `allocate_repair_logs/run_info.json`: resolved config, software versions, verl commit, data manifest
- `global_step_N/`: checkpoints (re-running with the same `EXP_NAME` resumes from the latest one)

Metric groups:
- `solve/*`: mean r, fraction of questions with ≥1 correct, all-zero / all-one / mixed group fractions and counts
- `repair/*`: mean e, fraction of failed parents with ≥1 successful repair, group fractions, extraction failures;
  `repair/available=0` when no repairs exist
- `train/*`: solve / repair / weighted repair / total loss, per-phase advantage stats, clip fractions, PPO-KL,
  optional KL, entropy, learning rate, gradient norm
- `compute/*`, `time/*`, `cumulative/*`: per-phase generation counts, prompt and generated tokens, response-length
  distribution, truncation and answer-parse-failure rates, wall time per phase, cumulative tokens and time
- `eval/first_attempt_acc/<set>`: held-out first-attempt accuracy (original question only, no feedback). verl's full
  `val-core/*` / `val-aux/*` metrics are kept, including an evaluation-only `acc_math_verify` symbolic check.

## Tests

`bash scripts/run_unit_tests.sh` (CPU) covers:
- reward conversion (±1 → 0/1)
- think-block extraction, including with the real tokenizer
- no ground-truth or sibling leakage
- parent-specific grouping
- zero-variance groups
- input / padding masking
- empty repair batches and ranks
- micro-batch and DP-sharded gradient accumulation (equals the full-batch gradient)
- invariance of the update to M
- λ linearity and once-per-phase KL
- a scripted end-to-end check of the two-phase rollout: counts G + F_q·M, both phases generated before the replicas
  sleep, batch assembly into the loss

## Notes

- **Token budget.** R1-distill often thinks for more than 8k tokens on hard problems. Truncated responses get
  reward 0, and their repair prompt shows the extraction-failure placeholder (they are counted in
  `compute/*/frac_truncated` and `repair/num_extraction_failures`). Raise `SOLVE_MAX_TOKENS` / `REPAIR_MAX_TOKENS`
  (e.g. 16384) if compute allows; the per-GPU token limits follow automatically.
- **Out of memory.** Lower `ROLLOUT_GPU_MEM_UTIL` (e.g. 0.5), set `OPTIMIZER_OFFLOAD=True`, or lower
  `TRAIN_BATCH_SIZE`.
- **Temperature.** Sampling uses T=1.0, top-p=1.0 by default, so phase samples come from the policy whose log-probs
  are trained. If the two phases use different temperatures, log-probs are recomputed per row at each phase's
  temperature.
- **Scope.** Fixed G and M only; the adaptive allocation scheduler is intentionally not part of this implementation.
