"""CPU integration test of the trainer's two-phase rollout and batch assembly with a scripted rollout manager.

Checks: generation count G + F_q * M, both phases generated before the replicas sleep (same policy snapshot),
parent-specific group ids, absence of ground-truth / sibling leakage in phase-2 prompts, padding rows, and that the
assembled batch feeds the phase-weighted loss.
"""

import numpy as np
import pytest
import torch
from conftest import actor_config, compose_config
from verl import DataProto
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import left_right_2_no_padding

from allocate_repair.batching import PHASE_PAD, PHASE_REPAIR, PHASE_SOLVE
from allocate_repair.losses import phase_weighted_ppo_loss
from allocate_repair.repair import AnswerExtractor, RepairPromptBuilder
from allocate_repair.reward_fn import compute_score
from allocate_repair.trainer import AllocateRepairTrainer

GT = {"q0": "11", "q1": "22", "q2": "33"}


class Recorder:
    def __init__(self):
        self.events = []


class FakeCheckpointManager:
    def __init__(self, rec):
        self.rec = rec

    def sleep_replicas(self):
        self.rec.events.append("sleep")


class FakeRolloutManager:
    """Scripted policy. Phase 1: question q0 always right, q1 right on even samples, q2 always wrong
    (q2's last sample never closes its think block). Phase 2: a repair is right iff it saw the parent's answer."""

    def __init__(self, tok, cfg, rec):
        self.tok, self.cfg, self.rec = tok, cfg, rec
        self.seen_prompts = []

    def _respond(self, qid, phase, k, prompt_text):
        gt = GT[qid]
        if phase == "solve":
            correct = qid == "q0" or (qid == "q1" and k % 2 == 0)
            if qid == "q2" and k == 3:
                return "SECRET_REASONING never ends"  # truncated: no think close
            ans = gt if correct else "0"
            return f"SECRET_REASONING_{qid}_{k} </think> SIBLING_TEXT_{qid}_{k} Answer: {ans}"
        right = "Answer: 0" in prompt_text or "ended without" in prompt_text
        return f"redo </think> fixed. Answer: {gt if right and k == 0 else '9'}"

    def generate_sequences(self, batch: DataProto) -> DataProto:
        nt = batch.non_tensor_batch
        n = len(batch)
        phase = nt["ar_phase"][0]
        self.rec.events.append(f"gen_{phase}")
        P = self.cfg.actor_rollout_ref.rollout.prompt_length
        R = self.cfg.actor_rollout_ref.rollout.response_length
        counter = {}
        prompts = torch.zeros(n, P, dtype=torch.long)
        responses = torch.zeros(n, R, dtype=torch.long)
        attn = torch.zeros(n, P + R, dtype=torch.long)
        resp_mask = torch.zeros(n, R, dtype=torch.long)
        rm = torch.zeros(n, R)
        extra = {"score": [], "acc": [], "pred": [], "answer_parsed": [], "math_dapo_score": []}
        for i in range(n):
            msgs = list(nt["raw_prompt"][i])
            text = self.tok.apply_chat_template(msgs, tokenize=False)
            self.seen_prompts.append((phase, text))
            qid = nt["extra_info"][i]["question_id"]
            key = (phase, nt["uid"][i])
            k = counter.get(key, 0)
            counter[key] = k + 1
            out = self._respond(qid, phase, k, text)
            p_ids = self.tok.encode(text)[-P:]
            r_ids = (self.tok.encode(out) + [self.tok.eos_token_id])[:R]
            prompts[i, P - len(p_ids) :] = torch.tensor(p_ids)
            responses[i, : len(r_ids)] = torch.tensor(r_ids)
            attn[i, P - len(p_ids) : P + len(r_ids)] = 1
            resp_mask[i, : len(r_ids)] = 1
            score = compute_score(nt["data_source"][i], self.tok.decode(r_ids), nt["reward_model"][i]["ground_truth"])
            rm[i, len(r_ids) - 1] = score["score"]
            for key2 in extra:
                extra[key2].append(score[key2])
        position_ids = (attn.cumsum(-1) - 1).clamp(min=0) * attn
        out = DataProto.from_dict(
            tensors={
                "prompts": prompts,
                "responses": responses,
                "response_mask": resp_mask,
                "input_ids": torch.cat([prompts, responses], 1),
                "attention_mask": attn,
                "position_ids": position_ids,
                "rm_scores": rm,
            },
            non_tensors={"__num_turns__": np.full(n, 2, dtype=np.int32), **{k: np.array(v) for k, v in extra.items()}},
            meta_info={"reward_extra_keys": list(extra), "timing": {"agent_loop/generate_sequences/max": 0.0}},
        )
        return out


def make_trainer(tok, G=4, M=3, dp=4, policy="placeholder"):
    cfg = compose_config(
        [
            f"actor_rollout_ref.rollout.n={G}",
            f"allocate_repair.M={M}",
            "actor_rollout_ref.rollout.prompt_length=600",
            "actor_rollout_ref.rollout.response_length=160",
            "actor_rollout_ref.rollout.agent.num_workers=4",
            "allocate_repair.solve.max_tokens=160",
            "allocate_repair.repair.max_tokens=160",
            "allocate_repair.validation.max_tokens=160",
            f"allocate_repair.repair.extraction_failure_policy={policy}",
        ]
    )
    rec = Recorder()
    t = object.__new__(AllocateRepairTrainer)
    t.config, t.ar, t.tokenizer = cfg, cfg.allocate_repair, tok
    t.G, t.M, t.repair_lambda, t.global_steps = G, M, 1.0, 1
    t.extractor = AnswerExtractor(tok)
    t.prompt_builder = RepairPromptBuilder(
        failure_placeholder=cfg.allocate_repair.repair.extraction_failure_placeholder
    )
    t.chat_template_kwargs = {}
    t.per_row_temperature = False
    t.async_rollout_manager = FakeRolloutManager(tok, cfg, rec)
    t.checkpoint_manager = FakeCheckpointManager(rec)
    t._get_dp_size = lambda wg, role: dp
    t.actor_rollout_wg = None
    return t, rec


def batch_dict():
    rows = []
    for qid in ("q0", "q1", "q2"):
        rows.append(
            {
                "raw_prompt": [{"role": "user", "content": f"Question {qid}: compute it. End with Answer: $Answer."}],
                "data_source": "math_dapo",
                "reward_model": {"ground_truth": GT[qid], "style": "rule"},
                "extra_info": {"question_id": qid, "question": f"Question {qid}"},
                "index": qid,
            }
        )
    out = {"dummy_tensor": torch.zeros(len(rows), 1, dtype=torch.uint8)}
    for key in rows[0]:
        arr = np.empty(len(rows), dtype=object)
        for i, r in enumerate(rows):
            arr[i] = r[key]
        out[key] = arr
    return out


def test_two_phase_rollout(fake_tokenizer):
    G, M = 4, 3
    trainer, rec = make_trainer(fake_tokenizer, G=G, M=M)
    info = trainer._rollout_two_phases(batch_dict(), {}, {})
    # both phases are generated before the replicas sleep (same weights)
    assert rec.events == ["gen_solve", "gen_repair", "sleep"]
    r1 = info["r1"]
    assert r1.tolist() == [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    failures_per_q = [0, 2, 4]
    assert len(info["requests"]) == sum(failures_per_q)
    metrics = trainer._rollout_metrics(info)
    assert metrics["compute/total_generations"] == sum(G + f * M for f in failures_per_q)
    assert metrics["solve/num_groups_all_one"] == 1 and metrics["solve/num_groups_all_zero"] == 1
    assert metrics["solve/num_groups_mixed"] == 1
    assert metrics["repair/num_extraction_failures"] == 1
    assert metrics["repair/num_groups"] == 6 and metrics["repair/num_samples"] == 18

    # parent-specific groups: each repair group = M repairs of one failed parent
    p2 = info["phase2"].non_tensor_batch
    for gid in set(p2["uid"]):
        members = np.flatnonzero(p2["uid"] == gid)
        assert len(members) == M
        assert len(set(p2["ar_parent_uid"][members])) == 1
        assert gid == "repair::" + p2["ar_parent_uid"][members[0]]

    # no leakage: phase-2 prompts never contain the ground truth, reasoning, or other responses
    for phase, text in trainer.async_rollout_manager.seen_prompts:
        if phase != "repair":
            continue
        assert "SECRET_REASONING" not in text and "redo" not in text
        siblings = [t for t in ("SIBLING_TEXT_q1_0", "SIBLING_TEXT_q1_2", "SIBLING_TEXT_q0_0") if t in text]
        assert not siblings, f"successful sibling responses leaked: {siblings}"
        qid = text.split("Question ")[1].split(":")[0]
        assert f"Answer: {GT[qid]}" not in text

    examples = trainer._examples(info)
    assert examples and any("repairs" in item for ex in examples for item in ex["phase1"])


def test_no_repair_when_M_is_zero(fake_tokenizer):
    trainer, rec = make_trainer(fake_tokenizer, M=0)
    info = trainer._rollout_two_phases(batch_dict(), {}, {})
    assert rec.events == ["gen_solve", "sleep"] and info["phase2"] is None
    metrics = trainer._rollout_metrics(info)
    assert metrics["repair/available"] == 0 and metrics["repair/reward_mean"] is None
    assert metrics["compute/total_generations"] == 12


def test_skip_policy_drops_unextractable_parent(fake_tokenizer):
    trainer, _ = make_trainer(fake_tokenizer, policy="skip")
    info = trainer._rollout_two_phases(batch_dict(), {}, {})
    assert len(info["requests"]) == 5 and info["repair_stats"]["skipped"] == 1


@pytest.mark.parametrize("M", [0, 3])
def test_combined_batch_feeds_the_loss(fake_tokenizer, M):
    from verl.trainer.ppo.ray_trainer import compute_advantage

    trainer, _ = make_trainer(fake_tokenizer, M=M, dp=4)
    info = trainer._rollout_two_phases(batch_dict(), {}, {})
    batch = trainer._combine(info)
    ns = int((batch.batch["ar_phase"] == PHASE_SOLVE).sum())
    nr = int((batch.batch["ar_phase"] == PHASE_REPAIR).sum())
    assert (ns, nr) == (12, 18 if M else 0)
    batch.batch["token_level_scores"] = batch.batch["rm_scores"]
    batch.batch["token_level_rewards"] = batch.batch["rm_scores"]
    batch = compute_advantage(batch, adv_estimator="grpo", norm_adv_by_std_in_grpo=True)
    batch, pad = trainer._pad_for_dp(batch)
    assert len(batch) % 4 == 0 and pad == (-(ns + nr)) % 4
    pad_rows = batch.batch["ar_phase"] == PHASE_PAD
    assert int(pad_rows.sum()) == pad
    assert batch.batch["response_mask"][pad_rows].sum() == 0 and batch.batch["advantages"][pad_rows].abs().sum() == 0
    # all-correct question q0 has zero advantage; failed-only q2 too
    adv = batch.batch["advantages"][:, 0]
    assert torch.all(adv[:4] == 0) and torch.all(adv[8:12] == 0)
    metrics = trainer._training_metrics(batch, None)
    assert metrics["train/solve/advantage/count"] == 12

    # feed the assembled batch to the loss in verl's no-padding layout
    batch.batch["old_log_probs"] = -torch.rand(batch.batch["responses"].shape) * batch.batch["response_mask"]
    td = left_right_2_no_padding(
        batch.select(
            batch_keys=[
                "prompts",
                "responses",
                "attention_mask",
                "response_mask",
                "input_ids",
                "position_ids",
                "old_log_probs",
                "advantages",
                "ar_phase",
            ]
        ).to_tensordict()
    )
    tu.assign_non_tensor(td, dp_size=1, ar_num_solve=ns, ar_num_repair=nr, ar_repair_lambda=1.0)
    nnz = int(batch.batch["attention_mask"].sum())
    log_probs = (-torch.rand(nnz)).requires_grad_(True)
    loss, metrics = phase_weighted_ppo_loss(actor_config(), {"log_probs": log_probs}, td)
    loss.backward()
    assert torch.isfinite(loss) and log_probs.grad is not None
    assert metrics["ar/repair/num_seqs_x_dp"].aggregate() == nr


def test_full_trajectory_dump_repair(fake_tokenizer, tmp_path):
    import gzip
    import json

    trainer, _ = make_trainer(fake_tokenizer)
    trainer.ar.logging.trajectory_dir = str(tmp_path)
    info = trainer._rollout_two_phases(batch_dict(), {}, {})
    rows = [json.loads(line) for line in gzip.open(trainer._dump_trajectories(info), "rt")]
    assert [r["phase"] for r in rows].count("solve") == 12 and [r["phase"] for r in rows].count("repair") == 18
    assert all(r["reward"] in (0.0, 1.0) and r["prompt"] and r["response"] for r in rows)
