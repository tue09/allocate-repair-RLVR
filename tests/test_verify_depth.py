"""Scaling Verification Depth: verdict parsing, target chain / depth reward, advantages and their token weighting,
prompts (no reasoning, no reference answer), and a scripted CPU run of VerifyDepthTrainer's rollout."""

import inspect
import itertools
import random

import numpy as np
import pytest
import torch
from conftest import compose_config
from test_trainer_flow import GT, FakeCheckpointManager, FakeRolloutManager, Recorder, batch_dict

from allocate_repair import verify_depth as V
from allocate_repair.batching import PHASE_REPAIR, PHASE_SOLVE
from allocate_repair.repair import AnswerExtractor
from allocate_repair.verify_trainer import VerifyDepthTrainer


# ---------------------------------------------------------------- verdicts
@pytest.mark.parametrize("text,expected", [
    ("analysis ...\nVerdict: CORRECT", 1),
    ("analysis ...\nVerdict: INCORRECT", 0),
    ("Verdict: **Incorrect** (the verdict of round 1 is wrong)", 0),
    ("verdict: correct (the verdict of round 2 is right)", 1),
    ("Verdict: CORRECT\n... on reflection\nVerdict: INCORRECT", 0),  # last one wins
    ("The verdict of round 1 is correct.", None),  # no 'Verdict:' line
    ("", None),
    (None, None),
])
def test_parse_verdict(text, expected):
    assert V.parse_verdict(text) == expected


# ---------------------------------------------------------------- targets and depth reward
def test_figure_1_example():
    # S is wrong; policy chain (1, 0, 1, 0) -> targets (0, 0, 1, 1), matches x v v x, reward 2/4
    targets, matches = V.target_chain(0, [1, 0, 1, 0])
    assert targets == [0, 0, 1, 1] and matches == [0, 1, 1, 0] and V.depth_reward(matches) == 0.5


def test_depth_one_is_self_verification_and_missing_verdicts_never_match():
    for truth, v in itertools.product((0, 1), (0, 1)):
        assert V.depth_reward(V.target_chain(truth, [v])[1]) == float(v == truth)
    targets, matches = V.target_chain(1, [None, 0, 1])
    assert targets == [1, 0, 1] and matches == [0, 1, 1]


def test_match_iff_implied_belief_is_right():
    rng = random.Random(0)
    for _ in range(500):
        k, truth = rng.randint(1, 6), rng.randint(0, 1)
        verdicts = [rng.randint(0, 1) for _ in range(k)]
        _, matches = V.target_chain(truth, verdicts)
        assert matches == [int(b == truth) for b in V.implied_beliefs(verdicts)]


def test_always_agree_after_round_one_equals_self_verification():
    for truth, v1 in itertools.product((0, 1), (0, 1)):
        assert V.depth_reward(V.target_chain(truth, [v1, 1, 1])[1]) == float(v1 == truth)


# ---------------------------------------------------------------- advantages
def test_chain_advantages_match_verl_grpo():
    from test_grouping_advantage import grpo

    rewards = [1.0, 1 / 3, 2 / 3, 1 / 3]
    np.testing.assert_allclose(V.chain_advantages(rewards), grpo(rewards, ["g"] * 4), atol=1e-6)
    assert V.chain_advantages([0.5, 0.5]) == [0.0, 0.0] and V.chain_advantages([1.0]) == [0.0]


def test_round_scaling_equals_token_mean_over_the_chain():
    from verl.trainer.ppo.core_algos import agg_loss

    rng = np.random.default_rng(0)
    k, chains = 3, [[5, 9, 2], [7, 1, 4], [3, 3, 8]]  # tokens per round
    chain_adv = [0.9, -0.4, 1.3]
    width = 10
    g = torch.tensor(rng.normal(size=(len(chains), k, width)), dtype=torch.float64)  # per-token gradient factor
    # target: eq. (1) applied to whole chains = mean over chains of the token mean over all k rounds
    target = sum(chain_adv[c] * sum(g[c, d, :n].sum() for d, n in enumerate(chains[c])) / sum(chains[c])
                 for c in range(len(chains))) / len(chains)
    # rows = rounds, seq-mean-token-mean over N_rows rows, with round advantages scaled
    loss_mat = torch.zeros(len(chains) * k, width, dtype=torch.float64)
    mask = torch.zeros_like(loss_mat)
    for c, counts in enumerate(chains):
        for d, (n, s) in enumerate(zip(counts, V.round_advantage_scale(counts))):
            loss_mat[c * k + d, :n] = chain_adv[c] * s * g[c, d, :n]
            mask[c * k + d, :n] = 1
    rows = agg_loss(loss_mat, mask, "seq-mean-token-mean", dp_size=1, global_batch_size=len(chains) * k)
    assert torch.allclose(rows, target)


# ---------------------------------------------------------------- prompts
def test_prompt_builder_rounds_and_isolation():
    b = V.VerifyPromptBuilder()
    m1 = b.build("What is 6*7?", "6*7 = 41.\nAnswer: 41", [])[0]["content"]
    assert m1.startswith("Question:\nWhat is 6*7?\n\nProposed solution:\n6*7 = 41.\nAnswer: 41\n\n")
    assert m1.endswith(V.DEFAULT_ROUND1_INSTRUCTION)
    m3 = b.build("Q", "S", ["first check. Verdict: CORRECT", "second. Verdict: INCORRECT"])[0]["content"]
    assert "Verification round 1:\nfirst check. Verdict: CORRECT\n\nVerification round 2:\nsecond." in m3
    assert m3.endswith(V.DEFAULT_LATER_ROUND_INSTRUCTION.format(prev=2)) and "round 2 is right" in m3
    assert not {"reference", "ground_truth", "answer", "reward", "target"} & set(
        inspect.signature(b.build).parameters)


def test_config_matches_defaults_and_agent_loop_phase():
    cfg = compose_config(["allocate_repair.method=verify_depth",
                          "allocate_repair.verify.judgment=relative"]).allocate_repair
    assert cfg.method == "verify_depth" and "depth" not in cfg.verify  # the depth is allocate_repair.M
    assert cfg.verify.round1_instruction == V.DEFAULT_ROUND1_INSTRUCTION
    assert cfg.verify.later_round_instruction == V.DEFAULT_LATER_ROUND_INSTRUCTION
    b = V.builder_from_config(cfg.verify)
    assert b.build("Q", "S", ["r1"])[0]["content"] == V.VerifyPromptBuilder().build("Q", "S", ["r1"])[0]["content"]
    for text in (V.DEFAULT_ROUND1_INSTRUCTION, V.DEFAULT_LATER_ROUND_INSTRUCTION.format(prev=1)):
        assert V.parse_verdict(text.splitlines()[-1]) is not None  # the requested line is parseable


# ---------------------------------------------------------------- scripted rollout through the real trainer
class FakeVerifyRolloutManager(FakeRolloutManager):
    """Phase 1 as in test_trainer_flow. Verification (one chain per solution; k = the solution's index among the
    question's verified solutions = its sample index here): even k judges S correctly, odd k wrongly; round 2 always
    agrees, round 3 agrees on even k and disagrees on odd k. Verification reasoning is SECRET_V."""

    judgment = "relative"
    no_verdict = frozenset()  # (chain, depth) pairs whose round ends without a verdict
    forced = frozenset()  # (chain, depth) pairs whose verdict comes only after the forced verdict header

    def generate_sequences(self, batch):
        self._forced_flags = []
        out = super().generate_sequences(batch)
        if self._forced_flags:
            out.non_tensor_batch["forced_verdict"] = np.array(self._forced_flags, dtype=object)
        return out

    def _respond(self, qid, phase, k, prompt_text):
        if phase != "verify":
            return super()._respond(qid, phase, k, prompt_text)
        depth = prompt_text.count("Verification round ") + 1
        solution = prompt_text.split("Proposed solution:")[1].split("Verification round")[0]
        correct = f"Answer: {GT[qid]}" in solution
        if self.judgment == "absolute":  # even k always judges S right; odd k wrong in round 1 only
            v = correct if (k % 2 == 0 or depth > 1) else not correct
        elif depth == 1:
            v = correct if k % 2 == 0 else not correct
        elif depth == 2:
            v = True
        else:
            v = k % 2 == 0
        self._forced_flags.append(int((k, depth) in self.forced))
        if (k, depth) in self.forced:
            return (f"SECRET_V_{depth} </think> analysis {depth}.{V.DEFAULT_FORCED_VERDICT_TEXT} "
                    f"{'CORRECT' if v else 'INCORRECT'}")
        if (k, depth) in self.no_verdict:
            return f"SECRET_V_{depth} </think> analysis {depth}. The answer is \\boxed{{{GT[qid]}}}"
        return f"SECRET_V_{depth} </think> analysis {depth}. Verdict: {'CORRECT' if v else 'INCORRECT'}"


def make_verify_trainer(tok, G=4, M=3, judgment="relative"):
    cfg = compose_config([
        "allocate_repair.method=verify_depth", f"actor_rollout_ref.rollout.n={G}", f"allocate_repair.M={M}",
        "actor_rollout_ref.rollout.prompt_length=3000",
        "actor_rollout_ref.rollout.response_length=160", "actor_rollout_ref.rollout.agent.num_workers=4",
        "allocate_repair.solve.max_tokens=160", "allocate_repair.repair.max_tokens=160",
        "allocate_repair.validation.max_tokens=160", "allocate_repair.verify.max_tokens=160",
        f"allocate_repair.verify.judgment={judgment}",
    ])
    rec = Recorder()
    t = object.__new__(VerifyDepthTrainer)
    t.config, t.ar, t.tokenizer = cfg, cfg.allocate_repair, tok
    t.G, t.M, t.depth, t.repair_lambda, t.global_steps = G, M, M, 1.0, 1
    t.extractor = AnswerExtractor(tok)
    t.round_extractor = AnswerExtractor(tok, max_answer_tokens=int(cfg.allocate_repair.verify.max_round_tokens))
    t.verify_builder = V.builder_from_config(cfg.allocate_repair.verify)
    t.chat_template_kwargs, t.per_row_temperature = {}, False
    t.async_rollout_manager = FakeVerifyRolloutManager(tok, cfg, rec)
    t.async_rollout_manager.judgment = judgment
    t.checkpoint_manager = FakeCheckpointManager(rec)
    t._get_dp_size = lambda wg, role: 4
    return t, rec


def test_verify_rollout_end_to_end(fake_tokenizer):
    G, M = 4, 3
    trainer, rec = make_verify_trainer(fake_tokenizer, G, M)
    info = trainer._rollout_two_phases(batch_dict(), {}, {})
    assert rec.events == ["gen_solve"] + ["gen_verify"] * M + ["sleep"]  # all rounds before the replicas sleep
    chains = info["chains"]
    verified = [ch["row"] for ch in chains]
    assert len(chains) == 11 and len(set(verified)) == 11 and 11 not in verified  # one chain per solution; q2's
    # last solution never closes </think>: skipped
    for ch in chains:  # scripted chains: even k always right -> 1; odd k: v1 wrong, agree, flip -> 1/3
        assert len(ch["verdicts"]) == M
        assert ch["matches"] == ([1, 1, 1] if ch["sample"] % 2 == 0 else [0, 0, 1])
        assert ch["reward"] == pytest.approx(1.0 if ch["sample"] % 2 == 0 else 1 / 3)
    # advantages: normalized over the chains of one question (eq. 9 of the note), not over chains of one solution
    groups = {}
    for ch in chains:
        groups.setdefault(int(info["rows_q"][ch["row"]]), []).append(ch)
    assert sorted(len(g) for g in groups.values()) == [3, 4, 4]
    for g in groups.values():
        np.testing.assert_allclose([ch["advantage"] for ch in g], V.chain_advantages([ch["reward"] for ch in g]))
    q2 = min(groups.values(), key=len)  # 3 chains: rewards 1, 1/3, 1
    assert [round(ch["advantage"], 3) for ch in q2] == [0.577, -1.155, 0.577]

    for phase, text in trainer.async_rollout_manager.seen_prompts:
        if phase != "verify":
            continue
        assert "SECRET_REASONING" not in text and "SECRET_V" not in text  # only text after </think> is shown
        qid = text.split("Question ")[1].split("\n")[0].strip()
        solution = text.split("Proposed solution:")[1].split("Verification round")[0]
        if f"Answer: {GT[qid]}" not in solution:
            assert f"Answer: {GT[qid]}" not in text  # reference answer never shown for a wrong solution
    round3 = [t for p, t in trainer.async_rollout_manager.seen_prompts if p == "verify" and "Verification round 2:" in t]
    assert round3 and all("analysis 1." in t and "analysis 2." in t for t in round3)

    metrics = trainer._rollout_metrics(info)
    assert metrics["compute/total_generations"] == G * 3 + M * 11
    assert metrics["verify/M"] == M and metrics["verify/num_groups"] == 3
    assert metrics["verify/d1/match_rate"] == pytest.approx(6 / 11) and metrics["verify/d3/match_rate"] == 1.0
    assert metrics["verify/num_solutions_skipped"] == 1 and metrics["verify/frac_groups_zero_advantage"] == 0.0

    batch = trainer._combine(info)
    batch = trainer._compute_advantages(batch, info)
    phase = batch.batch["ar_phase"]
    assert int((phase == PHASE_SOLVE).sum()) == 12 and int((phase == PHASE_REPAIR).sum()) == M * 11
    rows2 = torch.nonzero(phase == PHASE_REPAIR, as_tuple=True)[0]
    mask = batch.batch["response_mask"][rows2].bool()
    first_tok = batch.batch["advantages"][rows2, 0].numpy()
    np.testing.assert_allclose(first_tok, info["verify_row_adv"], atol=1e-6)
    assert torch.all(batch.batch["advantages"][rows2][~mask] == 0)
    examples = trainer._examples(info)
    assert examples and any(item["chain"] for ex in examples for item in ex["solutions"])


def test_verify_rollout_absolute_judgments(fake_tokenizer):
    trainer, rec = make_verify_trainer(fake_tokenizer, judgment="absolute")
    info = trainer._rollout_two_phases(batch_dict(), {}, {})
    for ch in info["chains"]:  # even k: right every round -> 1; odd k: wrong in round 1 only -> 2/3
        assert ch["matches"] == ([1, 1, 1] if ch["sample"] % 2 == 0 else [0, 1, 1])
        _, paper_matches = V.target_chain(ch["solution_correct"], ch["relative_verdicts"])
        assert paper_matches == ch["matches"]  # identical to the note's recursion on the derived relative chain
    later = [t for p, t in trainer.async_rollout_manager.seen_prompts if p == "verify" and "Verification round 1:" in t]
    assert later and all("check again whether the proposed solution is correct" in t for t in later)


# ---------------------------------------------------------------- absolute judgments and forced verdicts
def test_boxed_verdicts_and_last_one_wins():
    assert V.parse_verdict("so \\boxed{CORRECT}") == 1
    assert V.parse_verdict("\\boxed{\\text{Incorrect}}") == 0
    assert V.parse_verdict("Verdict: CORRECT ... finally \\boxed{INCORRECT}") == 0
    assert V.parse_verdict("\\boxed{INCORRECT} then **Verdict:** CORRECT") == 1
    assert V.parse_verdict("the answer is \\boxed{204}") is None


def test_absolute_mode_equals_note_reward_when_all_verdicts_present():
    rng = random.Random(1)
    for _ in range(500):
        k, truth = rng.randint(1, 6), rng.randint(0, 1)
        judgments = [rng.randint(0, 1) for _ in range(k)]
        _, paper = V.target_chain(truth, V.relative_from_absolute(judgments))
        assert paper == V.absolute_matches(truth, judgments)
    # a missing judgment only costs its own round
    assert V.absolute_matches(1, [1, None, 1]) == [1, 0, 1]
    assert V.target_chain(1, V.relative_from_absolute([1, None, 1]))[1] == [1, 0, 0]


def test_config_default_judgment_and_prompts():
    cfg = compose_config(["allocate_repair.method=verify_depth"]).allocate_repair.verify
    assert cfg.judgment == "absolute" and cfg.forced_verdict and cfg.forced_verdict_text == V.DEFAULT_FORCED_VERDICT_TEXT
    assert cfg.later_round_instruction_absolute == V.DEFAULT_LATER_ROUND_INSTRUCTION_ABSOLUTE
    text = V.builder_from_config(cfg).build("Q", "S", ["r1"])[0]["content"]
    assert text.endswith(V.DEFAULT_LATER_ROUND_INSTRUCTION_ABSOLUTE)


def test_forced_verdict_suffix_and_assembly():
    from allocate_repair.agent_loop import assemble_forced, forced_verdict_suffix

    forced = V.DEFAULT_FORCED_VERDICT_TEXT
    assert forced_verdict_suffix("think </think> fine. Verdict: CORRECT", forced) is None
    assert forced_verdict_suffix("think </think> the answer is \\boxed{4}", forced) == forced
    assert forced_verdict_suffix("thinking that never ends", forced) == "\n</think>" + forced
    ids, mask, lps = assemble_forced([5, 6, 1], [-.1, -.2, -.3], [7, 8], [9, 1], [-.4, -.5], eos_id=1)
    assert ids == [5, 6, 7, 8, 9, 1] and mask == [1, 1, 0, 0, 1, 1] and lps == [-.1, -.2, 0.0, 0.0, -.4, -.5]
    assert assemble_forced([5], None, [7], [9], None, eos_id=1)[2] is None


def test_forced_verdict_agent_loop(fake_tokenizer):
    import asyncio
    from types import SimpleNamespace

    from allocate_repair.agent_loop import PhaseAwareSingleTurnAgentLoop

    tok = fake_tokenizer
    calls = []

    class FakeServer:
        async def generate(self, request_id, *, prompt_ids, sampling_params, **kw):
            calls.append((request_id, list(prompt_ids), dict(sampling_params)))
            text = "long reasoning that is cut off" if len(calls) == 1 else " INCORRECT"
            ids = tok.encode(text) + ([tok.eos_token_id] if len(calls) == 2 else [])
            return SimpleNamespace(token_ids=ids, log_probs=[-0.5] * len(ids), num_preempted=0, extra_fields={})

    async def build(messages, **kw):
        return tok.encode("PROMPT")

    cfg = compose_config(["allocate_repair.method=verify_depth"]).allocate_repair
    loop = object.__new__(PhaseAwareSingleTurnAgentLoop)
    loop.tokenizer, loop.server_manager, loop.response_length = tok, FakeServer(), 400
    loop.ct_build_initial_tokens = build
    loop.phase_config = {"verify": cfg.verify}
    out = asyncio.run(loop.run({"temperature": 1.0}, ar_phase="verify", raw_prompt=[{"role": "user", "content": "x"}]))
    assert len(calls) == 2 and calls[0][0] == calls[1][0]  # same request id (sticky server)
    reserve = int(cfg.verify.verdict_reserve_tokens)
    assert calls[0][2]["max_tokens"] == int(cfg.verify.max_tokens) - reserve
    suffix = tok.encode("\n</think>" + cfg.verify.forced_verdict_text)
    own = tok.encode("long reasoning that is cut off")
    assert calls[1][1] == tok.encode("PROMPT") + own + suffix  # prompt + own tokens + forced text
    assert out.response_ids == own + suffix + tok.encode(" INCORRECT") + [tok.eos_token_id]
    assert out.response_mask == [1] * len(own) + [0] * len(suffix) + [1] * (len(tok.encode(" INCORRECT")) + 1)
    assert out.extra_fields["forced_verdict"] == 1
    final = tok.decode(out.response_ids).rpartition("</think>")[2]
    assert V.parse_verdict(final) == 0


def test_forced_header_continuation_is_parsed_leniently():
    h = V.DEFAULT_FORCED_VERDICT_TEXT.strip()
    assert V.parse_verdict(h) is None  # the header alone is not a verdict
    assert V.parse_verdict(f"analysis\n\n{h} The proposed solution is incorrect. The correct value is 7") == 0
    assert V.parse_verdict(f"{h} The solution is correct.") == 1
    assert V.parse_verdict(f"{h} CORRECT") == 1 and V.parse_verdict(f"{h} not correct") == 0
    assert V.parse_verdict(f"{h} 正确无误。") == 1 and V.parse_verdict(f"{h} 不正确") == 0
    assert V.parse_verdict(f"Verdict: INCORRECT ... {h} \\boxed{{4}}") == 0  # nothing after the header -> strict
    assert V.parse_verdict(f"{h} \\boxed{{4}}") is None


def test_full_trajectory_dump(fake_tokenizer, tmp_path):
    import gzip
    import json

    trainer, _ = make_verify_trainer(fake_tokenizer, judgment="absolute")
    trainer.ar.logging.trajectory_dir = str(tmp_path)
    info = trainer._rollout_two_phases(batch_dict(), {}, {})
    path = trainer._dump_trajectories(info)
    rows = [json.loads(line) for line in gzip.open(path, "rt")]
    solve = [r for r in rows if r["phase"] == "solve"]
    verify = [r for r in rows if r["phase"] == "verify"]
    assert len(solve) == 12 and len(verify) == 3 * 11  # every trajectory of the step (M = 3 rounds x 11 chains)
    assert all("SECRET_REASONING" in r["response"] for r in solve)  # full raw output, thinking included
    assert all(r["prompt"].startswith("<|user|>Question q") for r in solve)  # full prompt
    assert {r["depth"] for r in verify} == {1, 2, 3} and all("SECRET_V_" in r["response"] for r in verify)
    for r in verify:
        assert r["prompt"].count("Verification round ") == r["depth"] - 1  # one header per earlier round
        assert set(r) >= {"verdict", "target", "match", "reward", "row_advantage", "forced_verdict", "ground_truth"}
    trainer.ar.logging.trajectory_dir = None
    assert trainer._dump_trajectories(info) is None


# ---------------------------------------------------------------- format penalty: +1/k, 0, -1/k per round
def test_round_scores_three_values():
    assert V.round_scores([1, 0, 0, 1], [1, 1, 0, 0]) == [1, 0, -1, -1]
    assert V.depth_reward([1, 0, 0], [1, 1, 0]) == 0.0
    assert V.depth_reward([1, 1, 1], [0, 0, 0]) == -1.0 and V.depth_reward([1, 1, 1], [1, 1, 1]) == 1.0
    assert V.depth_reward([1, 0, 0]) == pytest.approx(1 / 3)  # no format_ok: the note's reward


def test_format_penalty_in_rollout(fake_tokenizer):
    trainer, _ = make_verify_trainer(fake_tokenizer, judgment="absolute")
    assert trainer.ar.verify.format_penalty  # default on
    trainer.async_rollout_manager.no_verdict = frozenset({(1, 2)})
    info = trainer._rollout_two_phases(batch_dict(), {}, {})
    for ch in info["chains"]:
        if ch["sample"] % 2 == 0:  # right in every round
            assert ch["format_ok"] == [1, 1, 1] and ch["scores"] == [1, 1, 1] and ch["reward"] == 1.0
        elif ch["sample"] == 1:  # wrong in round 1, no verdict in round 2, right in round 3
            assert ch["verdicts"][1] is None and ch["format_ok"] == [1, 0, 1]
            assert ch["scores"] == [0, -1, 1] and ch["reward"] == 0.0
        else:
            assert ch["scores"] == [0, 1, 1]
    m = trainer._rollout_metrics(info)  # 11 chains: 6 even, 3 with sample 1 (one per question), 2 with sample 3
    assert m["verify/d2/frac_format_violation"] == pytest.approx(3 / 11) and m["verify/d1/frac_format_violation"] == 0
    assert m["verify/d2/score_mean"] == pytest.approx((6 - 3 + 2) / 11)


def test_forced_round_is_a_format_violation(fake_tokenizer):
    trainer, _ = make_verify_trainer(fake_tokenizer, judgment="absolute")
    trainer.async_rollout_manager.forced = frozenset({(0, 2)})
    info = trainer._rollout_two_phases(batch_dict(), {}, {})
    for ch in info["chains"]:
        if ch["sample"] == 0:  # the forced verdict is parsed (and right) but still scores -1
            assert ch["round_forced"] == [0, 1, 0] and ch["verdicts"][1] == ch["solution_correct"]
            assert ch["matches"] == [1, 1, 1] and ch["scores"] == [1, -1, 1] and ch["reward"] == pytest.approx(1 / 3)
        else:
            assert ch["round_forced"] == [0, 0, 0]
            assert ch["scores"] == ([1, 1, 1] if ch["sample"] % 2 == 0 else [0, 1, 1])
    m = trainer._rollout_metrics(info)  # one sample-0 chain per question: 3 of 11
    assert m["verify/d2/frac_forced_verdict"] == pytest.approx(3 / 11) and m["verify/d2/frac_no_verdict"] == 0
    assert m["verify/d2/frac_format_violation"] == pytest.approx(3 / 11)


def test_format_penalty_off_is_the_note_reward(fake_tokenizer):
    trainer, _ = make_verify_trainer(fake_tokenizer, judgment="absolute")
    trainer.ar.verify.format_penalty = False
    trainer.async_rollout_manager.no_verdict = frozenset({(1, 2)})
    info = trainer._rollout_two_phases(batch_dict(), {}, {})
    for ch in info["chains"]:
        assert ch["reward"] == pytest.approx(V.depth_reward(ch["matches"])) and min(ch["scores"]) >= 0
