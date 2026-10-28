"""Scaling Verification Depth, version 2 of the note (method=verify_depth_v2): running conclusion, context of only the
previous round's final answer, per-round rewards / fixed-baseline advantages, format reward -1/M, and a scripted CPU
run of VerifyDepthV2Trainer's rollout."""

import itertools
import random
import re

import numpy as np
import pytest
import torch
from conftest import compose_config
from test_trainer_flow import GT, FakeCheckpointManager, FakeRolloutManager, Recorder, batch_dict
from test_verify_depth import FakeVerifyRolloutManager

from allocate_repair import verify_depth as V
from allocate_repair.batching import PHASE_REPAIR, PHASE_SOLVE
from allocate_repair.repair import AnswerExtractor
from allocate_repair.verify_trainer_v2 import VerifyDepthV2Trainer

PREV = V.DEFAULT_PREVIOUS_ROUND_HEADER


# ---------------------------------------------------------------- pure logic
def test_running_conclusion_eq7_and_eq11():
    assert V.running_conclusions([1, 0, 1, 0]) == [1, 0, 0, 1]
    assert V.running_conclusions([0, None, 1]) == [0, None, None]
    rng = random.Random(0)
    for _ in range(500):  # eq. (11): a round matches exactly when its running conclusion is right
        k, truth = rng.randint(1, 6), rng.randint(0, 1)
        verdicts = [rng.randint(0, 1) for _ in range(k)]
        _, matches = V.target_chain(truth, verdicts)
        assert matches == [int(c == truth) for c in V.running_conclusions(verdicts)]


def test_figure_2_example():
    # S wrong, policy chain (1, 0, 1, 0): targets (0, 0, 1, 1), matches x v v x, summary 2/4
    targets, matches = V.target_chain(0, [1, 0, 1, 0])
    assert targets == [0, 0, 1, 1] and matches == [0, 1, 1, 0]
    rewards = V.round_rewards(matches)
    assert rewards == [0.0, 1.0, 1.0, 0.0] and np.mean(rewards) == 0.5
    assert V.fixed_baseline_advantages(rewards) == [-0.5, 0.5, 0.5, -0.5]


def test_round_rewards_with_format_reward():
    M = 3
    assert V.round_rewards([1, 0, 1, 0], [1, 1, 0, 0], penalty=-1 / M) == [1.0, 0.0, -1 / M, -1 / M]
    assert V.fixed_baseline_advantages([1.0, 0.0, -1 / M]) == pytest.approx([0.5, -0.5, -1 / M - 0.5])


def test_chain_prompt_shows_only_previous_final_answer_and_conclusion():
    b = V.ChainPromptBuilder()
    m1 = b.build("What is 6*7?", "6*7 = 41.\nAnswer: 41")[0]["content"]
    assert m1 == "Question:\nWhat is 6*7?\n\nProposed solution:\n6*7 = 41.\nAnswer: 41\n\n" + V.DEFAULT_ROUND1_INSTRUCTION
    m2 = b.build("Q", "S", "round text. Verdict: INCORRECT", 0)[0]["content"]
    assert m2 == ("Question:\nQ\n\nProposed solution:\nS\n\n" + PREV + "\nround text. Verdict: INCORRECT\n\n"
                  "Running conclusion: the proposed solution is INCORRECT.\n\n" + V.DEFAULT_LATER_ROUND_INSTRUCTION_V2)
    assert "the proposed solution is CORRECT." in b.build("Q", "S", "x", 1)[0]["content"]
    assert "unknown" in b.build("Q", "S", "x", None)[0]["content"]


def test_v2_config_defaults():
    cfg = compose_config(["allocate_repair.method=verify_depth_v2"]).allocate_repair
    v = cfg.verify
    assert cfg.method == "verify_depth_v2" and v.format_penalty and v.forced_verdict
    assert v.later_round_instruction_v2 == V.DEFAULT_LATER_ROUND_INSTRUCTION_V2
    assert v.previous_round_header == PREV and v.conclusion_text == V.DEFAULT_CONCLUSION_TEXT
    assert v.format_penalty_reward is None and v.advantage_baseline == 0.5
    b = V.chain_builder_from_config(v)
    assert b.build("Q", "S", "r", 1)[0]["content"] == V.ChainPromptBuilder().build("Q", "S", "r", 1)[0]["content"]
    lines = V.DEFAULT_LATER_ROUND_INSTRUCTION_V2.splitlines()
    assert V.parse_verdict(lines[-3]) == 1 and V.parse_verdict(lines[-1]) == 0  # the requested lines are parseable


# ---------------------------------------------------------------- scripted rollout through the real trainer
class FakeV2RolloutManager(FakeVerifyRolloutManager):
    """Phase 1 as in test_trainer_flow. Verification, k = the chain's index among the question's verified solutions
    (= its sample index here): round 1 judges S right on even k, wrong on odd k. Later rounds read the running
    conclusion of their prompt: even k say CORRECT iff that conclusion is right; odd k agree in round 2 and disagree
    in round 3. Verification reasoning is SECRET_V; the forwarded text names its depth ("analysis d.")."""

    def _respond(self, qid, phase, k, prompt_text):
        if phase != "verify":
            return FakeRolloutManager._respond(self, qid, phase, k, prompt_text)
        solution = prompt_text.split("Proposed solution:")[1].split(PREV)[0]
        correct = f"Answer: {GT[qid]}" in solution
        if PREV in prompt_text:
            depth = int(re.search(r"analysis (\d+)\.", prompt_text).group(1)) + 1
            word = re.search(r"Running conclusion: the proposed solution is (\w+)", prompt_text).group(1)
            v = ((word == "CORRECT") == correct) if k % 2 == 0 else depth == 2
        else:
            depth, v = 1, (correct if k % 2 == 0 else not correct)
        self._forced_flags.append(int((k, depth) in self.forced))
        if (k, depth) in self.forced:
            return (f"SECRET_V_{depth} </think> analysis {depth}.{V.DEFAULT_FORCED_VERDICT_TEXT} "
                    f"{'CORRECT' if v else 'INCORRECT'}")
        if (k, depth) in self.no_verdict:
            return f"SECRET_V_{depth} </think> analysis {depth}. The answer is \\boxed{{{GT[qid]}}}"
        return f"SECRET_V_{depth} </think> analysis {depth}. Verdict: {'CORRECT' if v else 'INCORRECT'}"


def make_v2_trainer(tok, G=4, M=3, overrides=()):
    cfg = compose_config([
        "allocate_repair.method=verify_depth_v2", f"actor_rollout_ref.rollout.n={G}", f"allocate_repair.M={M}",
        "actor_rollout_ref.rollout.prompt_length=3000",
        "actor_rollout_ref.rollout.response_length=160", "actor_rollout_ref.rollout.agent.num_workers=4",
        "allocate_repair.solve.max_tokens=160", "allocate_repair.repair.max_tokens=160",
        "allocate_repair.validation.max_tokens=160", "allocate_repair.verify.max_tokens=160", *overrides,
    ])
    rec = Recorder()
    t = object.__new__(VerifyDepthV2Trainer)
    t.config, t.ar, t.tokenizer = cfg, cfg.allocate_repair, tok
    t.G, t.M, t.depth, t.repair_lambda, t.global_steps = G, M, M, 1.0, 1
    t.extractor = AnswerExtractor(tok)
    t.round_extractor = AnswerExtractor(tok, max_answer_tokens=int(cfg.allocate_repair.verify.max_round_tokens))
    t.verify_builder = V.chain_builder_from_config(cfg.allocate_repair.verify)
    t.chat_template_kwargs, t.per_row_temperature = {}, False
    t.async_rollout_manager = FakeV2RolloutManager(tok, cfg, rec)
    t.checkpoint_manager = FakeCheckpointManager(rec)
    t._get_dp_size = lambda wg, role: 4
    return t, rec


def test_v2_rollout_end_to_end(fake_tokenizer):
    G, M = 4, 3
    trainer, rec = make_v2_trainer(fake_tokenizer, G, M)
    info = trainer._rollout_two_phases(batch_dict(), {}, {})
    assert rec.events == ["gen_solve"] + ["gen_verify"] * M + ["sleep"]  # all rounds before the replicas sleep
    chains = info["chains"]
    assert len(chains) == 11 and 11 not in [ch["row"] for ch in chains]  # q2's unfinished solution is skipped
    for ch in chains:
        t = ch["solution_correct"]
        if ch["sample"] % 2 == 0:  # right at every depth
            assert ch["matches"] == [1, 1, 1] and ch["conclusions"] == [t, t, t]
            assert ch["advantage"] == [0.5, 0.5, 0.5]
        else:  # v1 wrong (c1 = 1 - t), agree (target 0: wrong), disagree (target 0: right, c3 = t)
            assert ch["verdicts"] == [1 - t, 1, 0] and ch["targets"] == [t, 0, 0]
            assert ch["matches"] == [0, 0, 1] and ch["conclusions"] == [1 - t, 1 - t, t]
            assert ch["scores"] == [0.0, 0.0, 1.0] and ch["advantage"] == [-0.5, -0.5, 0.5]
        assert ch["matches"] == [int(c == t) for c in ch["conclusions"]]  # eq. (11)
        assert ch["reward"] == pytest.approx(np.mean(ch["scores"]))

    # context: (Q, S) at depth 1, (Q, S, A_{d-1}) at depth d >= 2 -- only the previous round, with c_{d-1}
    # (the fake also sees the rows that pad each request to a multiple of the agent workers: count unique prompts)
    verify_prompts = sorted({t for p, t in trainer.async_rollout_manager.seen_prompts if p == "verify"})
    assert len(verify_prompts) == M * 11
    for text in verify_prompts:
        assert "SECRET_REASONING" not in text and "SECRET_V" not in text  # no reasoning trace is ever shown
        qid = text.split("Question ")[1].split("\n")[0].strip()
        solution = text.split("Proposed solution:")[1].split(PREV)[0]
        if f"Answer: {GT[qid]}" not in solution:
            assert f"Answer: {GT[qid]}" not in text  # reference answer never shown for a wrong solution
        assert text.count(PREV) <= 1 and text.count("analysis ") <= 1
        if PREV in text:
            d_prev = int(re.search(r"analysis (\d+)\.", text).group(1))
            assert "Running conclusion: the proposed solution is " in text
            assert text.rstrip("<|assistant|>").endswith(V.DEFAULT_LATER_ROUND_INSTRUCTION_V2)
            assert d_prev in (1, 2)
        else:
            assert "Running conclusion" not in text and V.DEFAULT_ROUND1_INSTRUCTION in text

    metrics = trainer._rollout_metrics(info)
    assert metrics["compute/total_generations"] == G * 3 + M * 11 and metrics["verify/num_rounds"] == M * 11
    assert metrics["verify/d1/match_rate"] == pytest.approx(6 / 11) and metrics["verify/d3/match_rate"] == 1.0
    assert metrics["verify/d2/conclusion_acc"] == pytest.approx(6 / 11) and metrics["verify/final_belief_acc"] == 1.0
    assert metrics["verify/format_penalty_reward"] == pytest.approx(-1 / M)
    assert metrics["verify/round_advantage/count"] == M * 11

    batch = trainer._combine(info)
    batch = trainer._compute_advantages(batch, info)
    phase = batch.batch["ar_phase"]
    assert int((phase == PHASE_SOLVE).sum()) == 12 and int((phase == PHASE_REPAIR).sum()) == M * 11
    rows2 = torch.nonzero(phase == PHASE_REPAIR, as_tuple=True)[0]
    mask = batch.batch["response_mask"][rows2].bool()
    adv = batch.batch["advantages"][rows2]
    expected = torch.tensor([ch["advantage"][d] for d in range(M) for ch in chains], dtype=adv.dtype)
    # eq. (13): every generated token of a round carries reward - 1/2 (no token-weight scaling, no normalization)
    assert torch.allclose(adv, expected[:, None] * mask) and torch.all(adv[~mask] == 0)
    scores = batch.batch["rm_scores"][rows2].sum(-1)
    assert torch.allclose(scores, torch.tensor([ch["scores"][d] for d in range(M) for ch in chains],
                                                dtype=scores.dtype))
    examples = trainer._examples(info)
    assert any(item["chain"] and "conclusions" in item["chain"] for ex in examples for item in ex["solutions"])


def test_v2_format_reward_and_missing_verdict(fake_tokenizer):
    M = 3
    trainer, _ = make_v2_trainer(fake_tokenizer, M=M)
    trainer.async_rollout_manager.no_verdict = frozenset({(0, 2)})  # even chains: no verdict at depth 2
    trainer.async_rollout_manager.forced = frozenset({(2, 3)})  # chain k=2: verdict only after the forced header
    info = trainer._rollout_two_phases(batch_dict(), {}, {})
    unknown = {t for p, t in trainer.async_rollout_manager.seen_prompts if p == "verify" and "unknown" in t}
    assert len(unknown) == 3  # depth 3 of the three k=0 chains: c_2 is unknown after the missing verdict
    for ch in info["chains"]:
        t = ch["solution_correct"]
        if ch["sample"] == 0:
            assert ch["verdicts"][1] is None and ch["format_ok"] == [1, 0, 1]
            assert ch["conclusions"] == [t, None, None] and ch["targets"][:2] == [t, 1]
            assert ch["matches"][:2] == [1, 0] and ch["scores"][:2] == [1.0, -1 / M]
        elif ch["sample"] == 2:  # forced verdict: right, but a format violation -> -1/M
            assert ch["round_forced"] == [0, 0, 1] and ch["matches"] == [1, 1, 1]
            assert ch["scores"] == [1.0, 1.0, -1 / M] and ch["advantage"][2] == pytest.approx(-1 / M - 0.5)
        else:
            assert ch["format_ok"] == [1, 1, 1]
    m = trainer._rollout_metrics(info)
    assert m["verify/d2/frac_format_violation"] == pytest.approx(3 / 11)
    assert m["verify/d3/frac_forced_verdict"] == pytest.approx(3 / 11)

    trainer, _ = make_v2_trainer(fake_tokenizer, M=M, overrides=["allocate_repair.verify.format_penalty=false"])
    trainer.async_rollout_manager.forced = frozenset({(2, 3)})
    info = trainer._rollout_two_phases(batch_dict(), {}, {})
    assert all(ch["scores"] == [1.0, 1.0, 1.0] for ch in info["chains"] if ch["sample"] == 2)


def test_v2_trajectory_dump(fake_tokenizer, tmp_path):
    import gzip
    import json

    trainer, _ = make_v2_trainer(fake_tokenizer)
    trainer.ar.logging.trajectory_dir = str(tmp_path)
    info = trainer._rollout_two_phases(batch_dict(), {}, {})
    rows = [json.loads(line) for line in gzip.open(trainer._dump_trajectories(info), "rt")]
    verify = [r for r in rows if r["phase"] == "verify"]
    assert len(verify) == 3 * 11 and {r["depth"] for r in verify} == {1, 2, 3}
    for r in verify:
        assert r["prompt"].count(PREV) == int(r["depth"] > 1)
        assert r["row_advantage"] == pytest.approx(r["round_reward"] - 0.5)
        assert (r["conclusion_in_prompt"] is None) == (r["depth"] == 1)
        if r["depth"] > 1:
            word = "CORRECT" if r["conclusion_in_prompt"] == 1 else "INCORRECT"
            assert f"Running conclusion: the proposed solution is {word}." in r["prompt"]


def test_v2_phase2_loss_is_round_level_token_mean(fake_tokenizer):
    """J_verify, eq. (15): mean over rounds of each round's token mean, with A = r - 1/2 (no chain-level scaling)."""
    from verl.trainer.ppo.core_algos import agg_loss

    lengths, rewards = [5, 9, 2, 7], [1.0, 0.0, -1 / 3, 1.0]
    adv = V.fixed_baseline_advantages(rewards)
    g = torch.randn(len(lengths), 10, dtype=torch.float64)
    loss_mat, mask = torch.zeros_like(g), torch.zeros_like(g)
    for i, n in enumerate(lengths):
        loss_mat[i, :n], mask[i, :n] = adv[i] * g[i, :n], 1
    rows = agg_loss(loss_mat, mask, "seq-mean-token-mean", dp_size=1, global_batch_size=len(lengths))
    target = sum(adv[i] * g[i, :n].mean() for i, n in enumerate(lengths)) / len(lengths)
    assert torch.allclose(rows, target)


@pytest.mark.parametrize("truth,v1", list(itertools.product((0, 1), (0, 1))))
def test_always_agree_keeps_first_judgment(truth, v1):
    _, matches = V.target_chain(truth, [v1, 1, 1])
    assert matches == [int(v1 == truth)] * 3 and V.running_conclusions([v1, 1, 1]) == [v1] * 3
