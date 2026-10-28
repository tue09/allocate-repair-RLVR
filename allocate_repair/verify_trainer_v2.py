"""Scaling Verification Depth, version 2 of the note: ``allocate_repair.method=verify_depth_v2``.

Note: /cm/shared/tue09_new/idea/Scaling_Verification_Depth.pdf (version uploaded to the HF repo on 27 Oct 2026, 22:59).
Script: scripts/train_verify_depth_v2_qwen3_dapo.sh. G = ``actor_rollout_ref.rollout.n`` solutions per question
(phase 1, GRPO), M = ``allocate_repair.M`` verification depth (phase 2: one chain of M rounds per solution). Reuses the
rollout loop of ``VerifyDepthTrainer`` (version 1); what changes:

* context, eq. (5)-(7): round 1 sees (Q, S); round d >= 2 sees (Q, S, A_{d-1}) with A_{d-1} = the previous round's text
  after </think> + the running conclusion c_{d-1} about S (c_1 = v_1, c_d = c_{d-1} if v_d = 1 else 1 - c_{d-1}),
  computed on the driver from the sampled verdicts, never generated (so it gets no loss). Earlier rounds, every
  reasoning trace and the reference answer are never shown.
* verdicts are relative: v_1 judges S, v_d judges v_{d-1}; targets v*_1 = Verify(S, R), v*_d = 1[v_{d-1} = v*_{d-1}].
* every round is its own training example: reward r_d = 1[v_d = v*_d] (eq. 10), advantage r_d - 1/2 (eq. 13, fixed
  baseline: no group normalization, no reward shared across depths) on all its generated tokens. The phase-2 loss is
  the clipped objective with a token mean per round and a mean over the G*M rounds (seq-mean-token-mean, eq. 15);
  J = J_GRPO + J_verify (repair_lambda = 1, eq. 16).
* format reward (``verify.format_penalty``): a round whose verdict was not written by the policy itself in the
  requested form (missing, or only obtained after the forced verdict header) gets reward
  ``verify.format_penalty_reward`` (null = -1/M) instead of 1[v_d = v*_d]. Its verdict, if any, still drives c_d and
  the next target.

Phase-2 training metrics are logged under train/repair/* (the shared phase-2 slot), rollout metrics under verify/*.
"""

import numpy as np

from allocate_repair.batching import advantage_stats, outcome_group_stats
from allocate_repair.verify_depth import (
    chain_builder_from_config,
    fixed_baseline_advantages,
    round_rewards,
    running_conclusions,
    target_chain,
)
from allocate_repair.verify_trainer import VerifyDepthTrainer


class VerifyDepthV2Trainer(VerifyDepthTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verify_builder = chain_builder_from_config(self.ar.verify)

    def _check_config(self):
        super()._check_config()
        assert 0.0 <= float(self.ar.verify.advantage_baseline) <= 1.0, "verify.advantage_baseline must be in [0, 1]"

    def format_penalty_reward(self) -> float:
        value = self.ar.verify.get("format_penalty_reward", None)
        return -1.0 / self.depth if value is None else float(value)

    # ------------------------------------------------------------------ prompts: (Q, S) or (Q, S, A_{d-1})
    def _chain_messages(self, ch: dict, stats: dict) -> list[dict]:
        question, build = ch["question"], self.verify_builder.build
        if not ch["rounds"]:
            return self._fit_messages([ch["solution"]], lambda t: build(question, t[0]), stats)
        conclusion = running_conclusions(ch["verdicts"])[-1]  # c_{d-1}
        return self._fit_messages([ch["solution"], ch["rounds"][-1]],
                                  lambda t: build(question, t[0], t[1], conclusion), stats)

    # ------------------------------------------------------------------ targets, round rewards, advantages
    def _score_chains(self, chains: list[dict], rows_q, r1) -> None:
        v = self.ar.verify
        penalty = bool(v.get("format_penalty", False))
        baseline = float(v.advantage_baseline)
        for ch in chains:
            truth = ch["solution_correct"] = int(r1[ch["row"]])
            ch["relative_verdicts"] = list(ch["verdicts"])
            ch["targets"], ch["matches"] = target_chain(truth, ch["verdicts"])
            ch["conclusions"] = running_conclusions(ch["verdicts"])
            ch["beliefs"] = ch["conclusions"]
            # format ok = the policy wrote a parseable verdict itself (no forced verdict header needed)
            ch["format_ok"] = [int(x is not None and not f) for x, f in zip(ch["verdicts"], ch["round_forced"])]
            ch["scores"] = round_rewards(ch["matches"], ch["format_ok"] if penalty else None,
                                         self.format_penalty_reward())
            ch["reward"] = float(np.mean(ch["scores"]))  # eq. (12) summary score, logging only
            ch["advantage"] = fixed_baseline_advantages(ch["scores"], baseline)

    def _row_advantage(self, ch: dict, d: int) -> float:
        return ch["advantage"][d]

    def _row_reward(self, ch: dict, d: int) -> float:
        return ch["scores"][d]

    # ------------------------------------------------------------------ trajectory dump
    def _phase2_row_info(self, info: dict) -> list[dict]:
        """Phase-2 rows are depth-major: row (d - 1) * n + i is round d of chain i."""
        chains, depth = info["chains"], self.depth
        n, row_adv = len(chains), info["verify_row_adv"]
        out = []
        for r in range(depth * n):
            d, ch = r // n, chains[r % n]
            out.append({
                "depth": d + 1, "solution_sample_index": ch["sample"], "solution_correct": ch["solution_correct"],
                "conclusion_in_prompt": ch["conclusions"][d - 1] if d > 0 else None,
                "verdict": ch["verdicts"][d], "conclusion": ch["conclusions"][d],
                "target": ch["targets"][d], "match": ch["matches"][d], "forced_verdict": ch["round_forced"][d],
                "format_ok": ch["format_ok"][d], "round_reward": ch["scores"][d],
                "final_text_forwarded": ch["rounds"][d],
                "reward": ch["scores"][d], "chain_summary_reward": ch["reward"], "row_advantage": float(row_adv[r]),
            })
        return out

    # ------------------------------------------------------------------ metrics
    def _rollout_metrics(self, info: dict) -> dict:
        phase1, phase2, r1, chains, depth = info["phase1"], info["phase2"], info["r1"], info["chains"], self.depth
        stats = info["verify_stats"]
        out = {"solve/num_questions": info["num_questions"], "solve/G": self.G, "verify/M": depth}
        out.update(outcome_group_stats(r1, phase1.non_tensor_batch["uid"], "solve"))
        out["solve/num_failed_responses"] = len(info["failed_rows"])
        verified = sorted({ch["row"] for ch in chains})
        out.update({
            "verify/num_solutions": stats["solutions"],
            "verify/num_solutions_no_final_answer": stats["no_final_answer"],
            "verify/num_solutions_skipped": stats["skipped"],
            "verify/num_solutions_verified": len(verified),
            "verify/num_solutions_verified_correct": int(sum(r1[j] for j in verified)),
            "verify/num_chains": len(chains),
            "verify/num_rounds": depth * len(chains),
            "verify/num_prompts_shortened": stats["prompts_shortened"],
            "verify/format_penalty_reward": self.format_penalty_reward(),
        })
        if chains:
            out["verify/reward_mean"] = float(np.mean([ch["reward"] for ch in chains]))
            out["verify/final_belief_acc"] = float(np.mean([ch["conclusions"][-1] == ch["solution_correct"]
                                                            for ch in chains]))
            reserve = int(self.ar.verify.get("verdict_reserve_tokens", 0)) if self.ar.verify.get("forced_verdict") else 0
            max_tok = int(self.ar.verify.max_tokens) - reserve
            for d in range(depth):
                pre = f"verify/d{d + 1}"
                verdicts = [ch["verdicts"][d] for ch in chains]
                parsed = [x for x in verdicts if x is not None]
                out[f"{pre}/match_rate"] = float(np.mean([ch["matches"][d] for ch in chains]))
                out[f"{pre}/conclusion_acc"] = float(np.mean([ch["conclusions"][d] == ch["solution_correct"]
                                                               for ch in chains]))
                out[f"{pre}/frac_target_correct"] = float(np.mean([ch["targets"][d] for ch in chains]))
                out[f"{pre}/frac_no_verdict"] = float(np.mean([x is None for x in verdicts]))
                out[f"{pre}/frac_verdict_correct"] = float(np.mean(parsed)) if parsed else None
                out[f"{pre}/frac_no_final_response"] = float(np.mean([not ch["round_ok"][d] for ch in chains]))
                out[f"{pre}/frac_truncated"] = float(np.mean([ch["round_len"][d] >= max_tok for ch in chains]))
                out[f"{pre}/frac_forced_verdict"] = float(np.mean([ch["round_forced"][d] for ch in chains]))
                out[f"{pre}/frac_format_violation"] = float(np.mean([1 - ch["format_ok"][d] for ch in chains]))
                out[f"{pre}/score_mean"] = float(np.mean([ch["scores"][d] for ch in chains]))
                out[f"{pre}/mean_tokens"] = float(np.mean([ch["round_tokens"][d] for ch in chains]))
            for value, name in ((1, "correct"), (0, "wrong")):
                sel = [ch for ch in chains if ch["solution_correct"] == value]
                out[f"verify/d1/acc_on_{name}_solutions"] = (
                    float(np.mean([ch["matches"][0] for ch in sel])) if sel else None)
                out[f"verify/reward_mean_on_{name}_solutions"] = (
                    float(np.mean([ch["reward"] for ch in sel])) if sel else None)
            out.update(advantage_stats([a for ch in chains for a in ch["advantage"]], "verify/round_advantage"))
        out.update(self._phase_compute_metrics(phase1, "solve", int(self.ar.solve.max_tokens)))
        out.update(self._phase_compute_metrics(phase2, "verify", int(self.ar.verify.max_tokens)))
        expected = self.G * info["num_questions"] + depth * len(verified)
        out["compute/total_generations"] = out["compute/solve/num_generations"] + out["compute/verify/num_generations"]
        assert out["compute/total_generations"] == expected, "generations must equal G * questions + M * verified"
        out["compute/total_prompt_tokens"] = out["compute/solve/prompt_tokens"] + out["compute/verify/prompt_tokens"]
        out["compute/total_gen_tokens"] = out["compute/solve/gen_tokens"] + out["compute/verify/gen_tokens"]
        return out

    def _examples(self, info: dict) -> list[dict]:
        examples = super()._examples(info)
        by_traj = {str(info["phase1"].non_tensor_batch["ar_traj_uid"][ch["row"]]): ch for ch in info["chains"]}
        for ex in examples:
            for item in ex["solutions"]:
                ch = by_traj.get(item["traj_uid"])
                if item["chain"] is not None and ch is not None:
                    item["chain"]["conclusions"] = ch["conclusions"]
                    item["chain"]["solution_correct"] = ch["solution_correct"]
        return examples

    def _run_info(self) -> dict:
        info = super()._run_info()
        info.update({"method": "verify_depth_v2", "verify_advantage": "per round: reward - advantage_baseline",
                     "verify_advantage_baseline": float(self.ar.verify.advantage_baseline),
                     "verify_format_penalty_reward": self.format_penalty_reward()})
        info.pop("verify_advantage_group", None)
        return info
