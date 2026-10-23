# allocate-repair: implementation prompt

Implement the following two-phase RLVR training method in **verl**: https://github.com/verl-project/verl. Use **BytedTsinghua-SIA/DAPO-Math-17k** for training: https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k. Use **Weights & Biases in offline mode**, with WANDB_MODE=offline.

Use one shared policy model and GRPO. This first implementation uses fixed, configurable G and M. Phase 2 applies only to incorrect phase-1 responses, following the current method definition.

1. **Phase 1: solve.**

   For each question q in a training batch, sample G complete responses from the current rollout policy:

   \[
   y_{qi}\sim\pi_{\mathrm{old}}(\cdot\mid q),\qquad i=1,\ldots,G.
   \]

   Retain each complete generated reasoning trajectory and final answer for RL training. Score each final answer using the dataset's ground-truth answer:

   \[
   r_{qi}=\mathbf{1}[\text{final answer in }y_{qi}\text{ is correct}].
   \]

   Rewards must be exactly 0 or 1. Inspect the selected verl scorer: the inspected math_dapo scorer returns a dictionary with an accuracy field and scores of -1/+1, so explicitly convert its correctness result to 0/1. Use the same checker and answer format in both phases. Keep ground truth in verifier metadata only.

   Keep correct responses in the phase-1 training data. Do not discard all-zero phase-1 groups before generating repairs, since those groups are part of the experiment.

2. **Phase 2: explain and repair.**

   For each response with r_qi=0, construct a prompt c_qi containing the original question, the previous answer, and feedback stating that the answer is incorrect. Exclude the original reasoning inside the think block. Use model-aware extraction of the answer portion after the closing think delimiter; log extraction failures rather than silently substituting the full reasoning trace.

   The instruction should be equivalent to: "The previous answer to this question is incorrect. Explain why it is incorrect and solve the problem again. End with your corrected final answer."

   Do not expose the ground-truth answer or another successful response. Sample M independent explanation-and-repair responses from this same prompt:

   \[
   z_{qij}\sim\pi_{\mathrm{old}}(\cdot\mid c_{qi}),\qquad j=1,\ldots,M.
   \]

   Score each repaired answer:

   \[
   e_{qij}=\mathbf{1}[\text{final answer in }z_{qij}\text{ is correct}].
   \]

   These are M parallel repair samples, not M sequential repair rounds. Train on the complete generated phase-2 trajectory. The reward checks the repaired answer; it does not independently establish that the explanation is correct.

3. **Compute advantages separately for the two phases.**

   For phase 1, group the G responses by their original question:

   \[
   A^{\mathrm{solve}}_{qi}
   =
   \frac{r_{qi}-\operatorname{mean}_{k=1}^{G}r_{qk}}
   {\operatorname{std}_{k=1}^{G}r_{qk}+\epsilon}.
   \]

   For phase 2, group the M repairs by their specific failed parent response:

   \[
   A^{\mathrm{repair}}_{qij}
   =
   \frac{e_{qij}-\operatorname{mean}_{k=1}^{M}e_{qik}}
   {\operatorname{std}_{k=1}^{M}e_{qik}+\epsilon}.
   \]

   Use verl's standard deviation convention consistently. Assign distinct phase and group identifiers; do not mix repairs of different failed responses into one group. Identical-reward groups have zero correctness advantage. Require G>=2 and M>=2 when repair training is enabled; allow M=0 for the solve-only baseline.

4. **Final loss and optimizer update.**

   Let ell_GRPO(c,o,A) denote verl's clipped policy loss for output o under its actual prompt c and scalar advantage A. Use generated-token masks, mean over valid generated tokens within each trajectory, then mean over trajectories within each phase.

   \[
   \mathcal L_{\mathrm{solve}}
   =\operatorname{mean}_{q,i}
   \ell_{\mathrm{GRPO}}(q,y_{qi},A^{\mathrm{solve}}_{qi}),
   \]

   \[
   \mathcal L_{\mathrm{repair}}
   =\operatorname{mean}_{q,i:r_{qi}=0,j}
   \ell_{\mathrm{GRPO}}(c_{qi},z_{qij},A^{\mathrm{repair}}_{qij}),
   \]

   \[
   \boxed{
   \mathcal L_{\mathrm{total}}
   =
   \mathcal L_{\mathrm{solve}}
   +
   \lambda\mathcal L_{\mathrm{repair}}
   }.
   \]

   Expose lambda as a configuration parameter, with 1.0 as the initial default. Optional KL regularization should use verl's existing implementation, be included once per phase, and be logged separately. Do not also apply KL reward shaping when it would duplicate that penalty.

   Generate both phases from the same policy snapshot before updating weights. Cache old log probabilities under each trajectory's actual generation context. Accumulate both phase losses and take one optimizer step per logical training batch; microbatching is allowed. Do not perform a solve-only optimizer step before generating or training the repair phase.

   Normalize each phase separately across the full distributed batch, so increasing M does not automatically increase the repair loss weight. Set the repair loss to zero if no repairs exist, and handle ranks with empty repair data safely.

   Mask all input and padding tokens from the loss. Apply the same outcome advantage to the generated trajectory, without additional token-level credit assignment. Use pure RL: no SFT, distillation, or training repaired trajectories under question-only prompts. Keep r_qi as the phase-1 reward rather than averaging it with repair rewards.

5. **Budget and configuration.**

   M means repairs per failed response. For a question with F_q failures, the generation count is G+F_q*M, not G+M.

   Expose G, M, lambda, model path, per-phase decoding settings and token limits, learning rate, batch sizes, seed, evaluation interval, and checkpoint interval. Keep the original question unchanged. Record actual prompt and generation tokens because sample counts alone do not measure compute.

   Inspect and report dataset row counts, unique question counts, and duplicate handling. Keep any validation split disjoint by question identity or normalized question text. Record the dataset revision and verl commit.

6. **Record these metrics in offline W&B and a local structured log.**

   - **Solve outcomes:** mean r_qi; fraction of questions with at least one correct initial response; fractions of groups with all-zero, all-one, and mixed rewards.
   - **Repair outcomes:** mean e_qij; fraction of failed parent responses with at least one successful repair; fractions of repair groups with all-zero, all-one, and mixed rewards. Record denominators; use null or an explicit unavailable flag when no repairs exist.
   - **Training:** solve loss, repair loss, weighted repair loss, total loss, per-phase advantage statistics, clipping fractions, optional KL, learning rate, and gradient norm.
   - **Compute:** generation counts, input and generated tokens, response-length distributions, truncation and parsing-failure rates, wall time by phase and update, and cumulative tokens and elapsed time.
   - **Evaluation:** held-out first-attempt accuracy using only the original question, without feedback or repair. Keep this separate from training repair success and any oracle-assisted post-repair metrics.
   - **Examples:** periodically save a bounded sample of question IDs, original questions, phase-1 outputs and rewards, exact repair prompts, phase-2 outputs and rewards, parent IDs, group IDs, and token counts.

   Log the resolved configuration and software versions once per run. Record raw counts alongside rates so they can be aggregated correctly.

7. **Implementation and validation.**

   Reuse verl's rollout, advantage, loss, checkpoint, and logging infrastructure where practical. Inspect the checked-out version before choosing integration points, especially group identifiers and distributed loss normalization.

   Provide runnable configuration and launch commands, a solve-only baseline, and a small smoke test. Add focused checks for reward conversion, parent-specific grouping, input masking, absence of ground-truth leakage, zero-variance groups, empty repair batches, and combined gradient accumulation. Verify that changing M does not silently change lambda through sample-count weighting.

   Keep the adaptive allocation scheduler outside this first implementation. The fixed-G/M runs will provide the observations needed to decide how to allocate the budget later.

