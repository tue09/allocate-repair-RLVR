"""Scaling Verification Depth trainer: phase 1 = GRPO solutions, phase 2 = one depth-M verification chain per solution.

Selected with ``allocate_repair.method=verify_depth`` (scripts/train_verify_depth_qwen3_dapo.sh). Hyperparameters as in
the note: G = ``actor_rollout_ref.rollout.n`` solutions per question, M = ``allocate_repair.M`` verification depth.
Reuses the two-phase machinery of ``AllocateRepairTrainer``: phase-2 rows (the verification rounds) carry the
PHASE_REPAIR code, so the loss is L_solve + lambda * L_verify (the note: lambda = ``allocate_repair.repair_lambda`` = 1)
and the training metrics of phase 2 are logged under ``train/repair/*``.

Phase 2, for every phase-1 solution S with a final-response section (``verify.unextractable_solution_policy``): one
chain of M rounds. Round d is one generation conditioned on (question text, S's text after </think>, rounds 1..d-1's
text after </think>); all M rounds are generated with the phase-1 weights before the rollout replicas sleep. Chain
advantages are normalized over the chains of one question (the note's eq. 9: one chain per solution, G chains when
every solution is verified). Chain rewards: ``allocate_repair.verify_depth``.
"""

import numpy as np
import torch
from verl import DataProto
from verl.utils.debug import marked_timer

from allocate_repair.batching import (
    PHASE_REPAIR,
    advantage_stats,
    outcome_group_stats,
    trajectory_uid,
)
from allocate_repair.repair import TRUNCATION_MARKER, AnswerExtractor, count_prompt_tokens
from allocate_repair.trainer import PROMPT_TOKEN_MARGIN, AllocateRepairTrainer, _object_array
from allocate_repair.verify_depth import (
    absolute_matches,
    builder_from_config,
    chain_advantages,
    depth_reward,
    round_scores,
    implied_beliefs,
    parse_verdict,
    relative_from_absolute,
    round_advantage_scale,
    target_chain,
)


def verify_group_id(question_uid: str) -> str:
    """Advantage group for phase 2: the verification chains of one question (one per verified solution)."""
    return f"verify::{question_uid}"


class VerifyDepthTrainer(AllocateRepairTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        v = self.ar.verify
        self.depth = self.M  # the note's M: rounds of the one verification chain per solution
        self.round_extractor = AnswerExtractor(
            self.tokenizer,
            mode=self.ar.repair.extraction_mode,
            think_end=self.ar.repair.think_end,
            max_answer_tokens=int(v.max_round_tokens),
        )
        self.verify_builder = builder_from_config(v)
        self.per_row_temperature = float(self.ar.solve.temperature) != float(v.temperature)

    def _check_config(self):
        super()._check_config()
        v = self.ar.verify
        assert self.M >= 1, "verify_depth: allocate_repair.M is the verification depth, need M >= 1"
        assert v.get("depth") is None, "verify.depth is gone: the verification depth is allocate_repair.M (the note's M)"
        assert self.config.actor_rollout_ref.rollout.response_length >= int(v.max_tokens), (
            f"rollout.response_length must be >= verify.max_tokens={v.max_tokens}"
        )
        assert v.unextractable_solution_policy in ("skip", "placeholder")

    def _phase2_temperature(self) -> float:
        return float(self.ar.verify.temperature)

    # ------------------------------------------------------------------ prompts
    def _round_messages(self, question: str, solution: str, rounds: list[str], stats: dict) -> list[dict]:
        """Round prompt; if it is too long, shorten the longest shown text (tail kept) until it fits."""
        return self._fit_messages([solution] + list(rounds),
                                  lambda texts: self.verify_builder.build(question, texts[0], texts[1:]), stats)

    def _fit_messages(self, texts: list[str], build, stats: dict) -> list[dict]:
        """``build(texts)``; while the prompt is too long, shorten the longest of ``texts`` (tail kept)."""
        max_tokens = int(self.config.actor_rollout_ref.rollout.prompt_length) - PROMPT_TOKEN_MARGIN
        texts = list(texts)
        messages = build(texts)
        n = count_prompt_tokens(self.tokenizer, messages, self.chat_template_kwargs)
        shortened = False
        while n > max_tokens:
            ids = [self.tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
            longest = int(np.argmax([len(x) for x in ids]))
            if len(ids[longest]) <= 16:
                raise ValueError(f"verification prompt needs {n} tokens > {max_tokens}; raise rollout.prompt_length")
            keep = ids[longest][len(ids[longest]) // 4 :]
            texts[longest] = TRUNCATION_MARKER + self.tokenizer.decode(keep, skip_special_tokens=True).strip()
            messages = build(texts)
            n = count_prompt_tokens(self.tokenizer, messages, self.chat_template_kwargs)
            shortened = True
        stats["prompts_shortened"] += int(shortened)
        return messages

    def _chain_messages(self, ch: dict, stats: dict) -> list[dict]:
        """Prompt of the chain's next round (version 1: question, solution and every earlier round)."""
        return self._round_messages(ch["question"], ch["solution"], ch["rounds"], stats)

    @staticmethod
    def _question_text(phase1: DataProto, row: int, messages: list[dict]) -> str:
        extra = phase1.non_tensor_batch["extra_info"][row]
        question = extra.get("question") if isinstance(extra, dict) else None
        return question if question else messages[-1]["content"]

    # ------------------------------------------------------------------ rollout
    def _rollout_two_phases(self, batch_dict: dict, metrics: dict, timing_raw: dict):
        p1 = self._rollout_phase1(batch_dict, timing_raw)
        phase1, traj1, rows_q, r1 = p1["phase1"], p1["traj1"], p1["rows_q"], p1["r1"]
        depth, v = self.depth, self.ar.verify
        nt1 = phase1.non_tensor_batch
        stats = {"solutions": len(phase1), "no_final_answer": 0, "skipped": 0, "prompts_shortened": 0}

        # ---------------- which solutions are verified, and what the verifier sees of them
        chains = []
        with marked_timer("build_verify", timing_raw, color="cyan"):
            response_ids = self._valid_response_ids(phase1, range(len(phase1)))
            for j in range(len(phase1)):
                ex = self.extractor.extract(response_ids[j])
                if not ex.ok:
                    stats["no_final_answer"] += 1
                    if v.unextractable_solution_policy == "skip":
                        stats["skipped"] += 1
                        continue
                solution = ex.text if ex.ok else v.no_solution_text
                question = self._question_text(phase1, j, p1["question_messages"][rows_q[j]])
                chains.append({"row": j, "sample": int(nt1["ar_sample_index"][j]), "question": question,
                               "solution": solution, "rounds": [], "verdicts": [], "round_ok": [], "round_tokens": [],
                               "round_len": [], "round_forced": []})

        # ---------------- M sequential rounds (same weights: replicas are still awake)
        parts = []
        for d in range(1, depth + 1):
            if not chains:
                break
            n = len(chains)
            rows = np.array([ch["row"] for ch in chains])
            with marked_timer("build_verify", timing_raw, color="cyan"):
                messages = [self._chain_messages(ch, stats) for ch in chains]
            meta = DataProto.from_dict(
                tensors={"dummy_tensor": torch.zeros(n, 1, dtype=torch.uint8)},
                non_tensors={
                    "data_source": nt1["data_source"][rows],
                    "reward_model": nt1["reward_model"][rows],
                    "extra_info": nt1["extra_info"][rows],
                    "uid": _object_array([verify_group_id(q) for q in nt1["ar_question_uid"][rows]]),
                    "ar_question_uid": nt1["ar_question_uid"][rows],
                    "ar_traj_uid": _object_array(
                        [f"{trajectory_uid(traj1[ch['row']], 0, PHASE_REPAIR)}/d{d}" for ch in chains]
                    ),
                    "ar_parent_uid": traj1[rows],
                    "ar_sample_index": np.array([ch["sample"] for ch in chains], dtype=np.int64),
                },
            )
            gen = DataProto.from_dict(
                tensors={"dummy_tensor": torch.zeros(n, 1, dtype=torch.uint8)},
                non_tensors={
                    "raw_prompt": _object_array(messages),
                    "data_source": meta.non_tensor_batch["data_source"],
                    "reward_model": meta.non_tensor_batch["reward_model"],
                    "extra_info": meta.non_tensor_batch["extra_info"],
                    "uid": meta.non_tensor_batch["uid"],
                    "ar_phase": _object_array(["verify"] * n),
                },
                meta_info={"global_steps": self.global_steps},
            )
            with marked_timer(f"gen_verify_d{d}", timing_raw, color="red"):
                out = self._generate(gen, timing_raw, f"gen_verify_d{d}")
            part = meta.union(out)
            ids = self._valid_response_ids(part, range(n))
            trained = part.batch["response_mask"].sum(-1).tolist()  # excludes forced (masked) text
            forced = part.non_tensor_batch.get("forced_verdict", np.full(n, None, dtype=object))
            for i, ch in enumerate(chains):
                ex = self.round_extractor.extract(ids[i])
                ch["rounds"].append(ex.text if ex.ok else v.round_failure_text)
                header = str(v.forced_verdict_text).strip() if v.get("forced_verdict") else None
                ch["verdicts"].append(parse_verdict(ex.text, forced_header=header) if ex.ok else None)
                ch["round_ok"].append(ex.ok)
                ch["round_tokens"].append(int(trained[i]))
                ch["round_len"].append(len(ids[i]))
                ch["round_forced"].append(int(forced[i] or 0))
                if d == 1:
                    ch["first_prompt"] = messages[i]
                ch["last_prompt"] = messages[i]
            parts.append(part)

        with marked_timer("sleep_replicas", timing_raw):
            self.checkpoint_manager.sleep_replicas()

        # ---------------- targets, depth rewards, advantages
        self._score_chains(chains, rows_q, r1)

        phase2, row_adv, row_reward = None, np.zeros(0), np.zeros(0)
        if parts:
            phase2 = DataProto.concat(parts)  # depth-major: row (d - 1) * n + i is round d of chain i
            row_adv = np.array([self._row_advantage(ch, d) for d in range(depth) for ch in chains])
            row_reward = np.array([self._row_reward(ch, d) for d in range(depth) for ch in chains])
            # rm_scores of phase-2 rows: the row's reward on the last response token (logging only;
            # the advantages are set from the chains in _compute_advantages)
            scores = torch.zeros_like(phase2.batch["rm_scores"])
            _, response_len = self._lengths(phase2)
            for r, (length, value) in enumerate(zip(response_len, row_reward, strict=True)):
                if length > 0:
                    scores[r, int(length) - 1] = float(value)
            phase2.batch["rm_scores"] = scores

        return {
            "phase1": phase1,
            "phase2": phase2,
            "r1": r1,
            "r2": row_reward,
            "requests": [],
            "repair_stats": None,
            "failed_rows": p1["failed_rows"],
            "question_messages": p1["question_messages"],
            "rows_q": rows_q,
            "num_questions": p1["num_questions"],
            "chains": chains,
            "verify_stats": stats,
            "verify_row_adv": row_adv,
        }

    def _row_advantage(self, ch: dict, d: int) -> float:
        """Advantage of round d + 1 of a chain (version 1: chain advantage x token-weight scale)."""
        return ch["advantage"] * ch["round_scale"][d]

    def _row_reward(self, ch: dict, d: int) -> float:
        """Reward logged on round d + 1 of a chain (version 1: the chain's depth reward)."""
        return ch["reward"]

    def _score_chains(self, chains: list[dict], rows_q, r1) -> None:
        """Version 1: targets, depth reward per chain, advantages normalized over the chains of a question."""
        v, depth = self.ar.verify, self.depth
        absolute = str(v.get("judgment", "relative")) == "absolute"
        penalty = bool(v.get("format_penalty", False))
        for ch in chains:
            truth = ch["solution_correct"] = int(r1[ch["row"]])
            if absolute:  # verdicts judge S itself; the note's relative chain is derived for logging
                ch["relative_verdicts"] = relative_from_absolute(ch["verdicts"])
                ch["targets"], _ = target_chain(truth, ch["relative_verdicts"])
                ch["matches"] = absolute_matches(truth, ch["verdicts"])
                ch["beliefs"] = list(ch["verdicts"])
            else:
                ch["relative_verdicts"] = list(ch["verdicts"])
                ch["targets"], ch["matches"] = target_chain(truth, ch["verdicts"])
                ch["beliefs"] = implied_beliefs(ch["verdicts"])
            # format ok = the policy wrote a parseable verdict itself (no forced verdict header needed)
            ch["format_ok"] = [int(x is not None and not f) for x, f in zip(ch["verdicts"], ch["round_forced"])]
            fmt = ch["format_ok"] if penalty else None
            ch["scores"] = round_scores(ch["matches"], fmt if fmt is not None else [1] * depth)
            ch["reward"] = depth_reward(ch["matches"], fmt)
        for idxs in self._question_groups(chains, rows_q).values():
            for idx, adv in zip(idxs, chain_advantages([chains[i]["reward"] for i in idxs]), strict=True):
                chains[idx]["advantage"] = adv
                chains[idx]["round_scale"] = round_advantage_scale(chains[idx]["round_tokens"])

    @staticmethod
    def _question_groups(chains: list[dict], rows_q) -> dict:
        """Chain indices per question: the advantage groups of phase 2."""
        groups = {}
        for idx, ch in enumerate(chains):
            groups.setdefault(int(rows_q[ch["row"]]), []).append(idx)
        return groups

    # ------------------------------------------------------------------ trajectory dump
    PHASE2_NAME = "verify"

    def _phase2_row_info(self, info: dict) -> list[dict]:
        """Phase-2 rows are depth-major: row (d - 1) * n + i is round d of chain i."""
        chains, depth = info["chains"], self.depth
        n, row_adv = len(chains), info["verify_row_adv"]
        out = []
        for r in range(depth * n):
            d, ch = r // n, chains[r % n]
            out.append({
                "depth": d + 1, "solution_sample_index": ch["sample"], "solution_correct": ch["solution_correct"],
                "verdict": ch["verdicts"][d], "relative_verdict": ch["relative_verdicts"][d],
                "target": ch["targets"][d], "match": ch["matches"][d], "forced_verdict": ch["round_forced"][d],
                "format_ok": ch["format_ok"][d], "round_score": ch["scores"][d],
                "round_text_shown_to_later_rounds": ch["rounds"][d],
                "reward": ch["reward"], "chain_advantage": ch["advantage"], "row_advantage": float(row_adv[r]),
            })
        return out

    # ------------------------------------------------------------------ advantages
    def _compute_advantages(self, batch: DataProto, info: dict) -> DataProto:
        """Phase 1: verl's GRPO over question groups. Phase 2: chain advantages (normalized over the chains of a
        question) times the per-round token-weight scale, set on every response token of the round."""
        batch = super()._compute_advantages(batch, info)
        rows = torch.nonzero(batch.batch["ar_phase"] == PHASE_REPAIR, as_tuple=True)[0]
        row_adv = torch.as_tensor(info["verify_row_adv"], dtype=batch.batch["advantages"].dtype)
        assert len(rows) == len(row_adv), f"{len(rows)} phase-2 rows but {len(row_adv)} verification advantages"
        if len(rows):
            mask = batch.batch["response_mask"][rows].to(row_adv.dtype)
            values = row_adv[:, None].to(mask.device) * mask
            batch.batch["advantages"][rows] = values
            batch.batch["returns"][rows] = values
        return batch

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
            "verify/num_prompts_shortened": stats["prompts_shortened"],
        })
        if chains:
            rewards = np.array([ch["reward"] for ch in chains])
            out["verify/reward_mean"] = float(rewards.mean())
            beliefs = [ch["beliefs"][-1] for ch in chains]
            out["verify/final_belief_acc"] = float(np.mean([b == ch["solution_correct"]
                                                            for b, ch in zip(beliefs, chains)]))
            # the first generation of a forced-verdict round stops verdict_reserve_tokens early
            reserve = int(self.ar.verify.get("verdict_reserve_tokens", 0)) if self.ar.verify.get("forced_verdict") else 0
            max_tok = int(self.ar.verify.max_tokens) - reserve
            for d in range(depth):
                pre = f"verify/d{d + 1}"
                verdicts = [ch["verdicts"][d] for ch in chains]
                parsed = [x for x in verdicts if x is not None]
                out[f"{pre}/match_rate"] = float(np.mean([ch["matches"][d] for ch in chains]))
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
            groups = [[chains[i]["reward"] for i in idxs] for idxs in self._question_groups(chains, info["rows_q"]).values()]
            out["verify/num_groups"] = len(groups)
            out["verify/mean_chains_per_group"] = float(np.mean([len(g) for g in groups]))
            out["verify/frac_groups_single_chain"] = float(np.mean([len(g) == 1 for g in groups]))
            out["verify/frac_groups_zero_advantage"] = float(np.mean([len(set(g)) == 1 for g in groups]))
            out.update(advantage_stats([ch["advantage"] for ch in chains], "verify/chain_advantage"))
        out.update(self._phase_compute_metrics(phase1, "solve", int(self.ar.solve.max_tokens)))
        out.update(self._phase_compute_metrics(phase2, "verify", int(self.ar.verify.max_tokens)))
        expected = self.G * info["num_questions"] + depth * len(verified)
        out["compute/total_generations"] = out["compute/solve/num_generations"] + out["compute/verify/num_generations"]
        assert out["compute/total_generations"] == expected, "generations must equal G * questions + M * verified"
        out["compute/total_prompt_tokens"] = out["compute/solve/prompt_tokens"] + out["compute/verify/prompt_tokens"]
        out["compute/total_gen_tokens"] = out["compute/solve/gen_tokens"] + out["compute/verify/gen_tokens"]
        return out

    def _examples(self, info: dict) -> list[dict]:
        cfg = self.ar.logging
        max_chars = int(cfg.examples_max_chars)
        phase1, r1, chains = info["phase1"], info["r1"], info["chains"]
        _, p1_resp = self._lengths(phase1)

        def clip(text):
            return {"text": text[:max_chars], "num_chars": len(text), "clipped": len(text) > max_chars}

        by_row = {ch["row"]: ch for ch in chains}
        chosen = []
        for q in range(info["num_questions"]):  # questions with both a correct and a wrong verified solution first
            rows = np.flatnonzero(info["rows_q"] == q)
            kinds = {int(r1[j]) for j in rows if j in by_row}
            chosen.append((-len(kinds), q))
        examples = []
        for _, q in sorted(chosen)[: int(cfg.examples_num_questions)]:
            rows = np.flatnonzero(info["rows_q"] == q)
            extra = phase1.non_tensor_batch["extra_info"][rows[0]]
            ex = {"question_uid": str(phase1.non_tensor_batch["ar_question_uid"][rows[0]]),
                  "question_id": extra.get("question_id"), "question": extra.get("question"), "solutions": []}
            for j in rows:
                ids = phase1.batch["responses"][j, : int(p1_resp[j])].tolist()
                item = {"traj_uid": str(phase1.non_tensor_batch["ar_traj_uid"][j]), "reward": float(r1[j]),
                        "output": clip(self._decode(ids)), "verified": j in by_row, "chain": None}
                ch = by_row.get(j)
                if ch is not None:
                    rendered = self.tokenizer.apply_chat_template(
                        ch["last_prompt"], add_generation_prompt=True, tokenize=False, **self.chat_template_kwargs
                    )
                    item["chain"] = {
                        "verdicts": ch["verdicts"], "relative_verdicts": ch["relative_verdicts"],
                        "round_forced": ch["round_forced"], "targets": ch["targets"],
                        "matches": ch["matches"], "format_ok": ch["format_ok"], "round_scores": ch["scores"],
                        "reward": ch["reward"], "advantage": ch["advantage"],
                        "round_tokens": ch["round_tokens"], "round_final_texts": [clip(t) for t in ch["rounds"]],
                        "round1_prompt_messages": ch["first_prompt"],
                        f"round{len(ch['rounds'])}_prompt_rendered": clip(rendered),
                    }
                ex["solutions"].append(item)
            examples.append(ex)
        return examples

    def _run_info(self) -> dict:
        info = super()._run_info()
        info.update({"method": "verify_depth", "verify_depth_M": self.depth, "verify_chains_per_solution": 1,
                     "verify_advantage_group": "question"})
        return info
