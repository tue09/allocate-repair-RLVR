"""Scaling Verification Depth: the pure logic of the verification-chain phase (no Ray / GPU code).

Idea note: /cm/shared/tue09_new/idea/Scaling_Verification_Depth.pdf (Tue Le, Oct 2026).

Notation follows the note: G solutions per question, verification depth M (written k inside the functions below).
For a phase-1 solution S of question Q, its one verification chain of depth k = M has rounds d = 1..k. Round d sees the
question, S's final-response section (text after </think>; the solution's reasoning is never shown) and the
final-response sections of rounds 1..d-1, and ends with a binary verdict:
    v_1 = 1 claims S is correct;  v_d = 1 (d >= 2) claims v_{d-1} is correct.
Target chain (built after generation, never shown to the policy):
    v*_1 = Verify(S, R),  v*_d = 1[v_{d-1} = v*_{d-1}]
Depth reward: r_ver = (1/k) * sum_d 1[v_d = v*_d]; an unparseable / missing verdict never matches.
With verify.format_penalty (default) each round scores s_d = +1 (verdict matches), 0 (own verdict, does not match)
or -1 (format violation: the policy wrote no parseable verdict itself -- missing, or only obtained through the forced
verdict header), and r_ver = (1/k) * sum_d s_d in [-1, 1] (``round_scores``).
Advantages are normalized over the chains of one question (one chain per solution, eq. 9 of the note) and shared by
every generated token of the chain.

Nothing here receives the reference answer except ``target_chain`` (through the phase-1 reward of S).

Version 2 of the note (allocate_repair.method=verify_depth_v2, ``ChainPromptBuilder`` and the functions at the end):
round d >= 2 sees only the question, S and the previous round's forwarded final answer A_{d-1} = (its text after
</think>, running conclusion c_{d-1} about S), eq. (5)-(7); c_1 = v_1, c_d = c_{d-1} if v_d = 1 else 1 - c_{d-1}
(computed deterministically, never generated). Every round is its own training example with reward
r_d = 1[v_d = v*_d] (eq. 10) and fixed-baseline advantage r_d - 1/2 (eq. 13); with the format penalty a round
without a verdict of the policy's own gets reward -1/M instead.

Judgment modes (``verify.judgment``, version 1 only):
  relative  the note's prompts: round d >= 2 is asked whether round d-1's verdict is right (v_d as above).
  absolute  every round is asked whether S itself is correct (a_d), taking the earlier rounds into account; the note's
            relative verdicts are v_1 = a_1, v_d = 1[a_d = a_{d-1}]. With every verdict present the depth reward is
            identical in both modes (a match at depth d <=> the belief about S after round d is right, see
            ``implied_beliefs``); a missing verdict only costs its own round instead of also turning the next round's
            target around.
"""

from __future__ import annotations

import re

import numpy as np

# "Verdict: X", also "**Verdict (CORRECT or INCORRECT):** X" (the forced header followed by the verdict)
VERDICT_RE = re.compile(r"verdict\s*(?:\([^)]*\))?\s*[:：]?\s*[*_`\"'\s]*(incorrect|correct)\b", re.IGNORECASE)
# right after the forced header the model's verdict is often prose ("The proposed solution is incorrect.")
FORCED_CONTINUATION_RE = re.compile(r"\b(not\s+correct|incorrect|wrong|correct|right)\b|不正确|错误|正确", re.IGNORECASE)
BOXED_VERDICT_RE = re.compile(r"\\boxed\s*\{\s*(?:\\text(?:bf)?\s*\{)?\s*[*\s]*(incorrect|correct)\b", re.IGNORECASE)

DEFAULT_QUESTION_HEADER = "Question:"
DEFAULT_SOLUTION_HEADER = "Proposed solution:"
DEFAULT_ROUND_HEADER = "Verification round {d}:"
DEFAULT_ROUND1_INSTRUCTION = (
    "Check whether the proposed solution above is correct: verify its steps and whether its final answer correctly "
    "answers the question. Do not solve the problem from scratch. Write a short analysis, then end with exactly one "
    "line:\nVerdict: CORRECT\nor\nVerdict: INCORRECT"
)
DEFAULT_LATER_ROUND_INSTRUCTION = (
    "The verification rounds above may contain mistakes. Check whether the verdict of verification round {prev} is "
    "right, i.e. whether its conclusion about the proposed solution is correct. Write a short analysis, then end with "
    "exactly one line:\nVerdict: CORRECT (the verdict of round {prev} is right)\nor\n"
    "Verdict: INCORRECT (the verdict of round {prev} is wrong)"
)
DEFAULT_LATER_ROUND_INSTRUCTION_ABSOLUTE = (
    "The verification rounds above may contain mistakes. Taking them into account, check again whether the proposed "
    "solution is correct. Do not solve the problem from scratch. Write a short analysis, then end with exactly one "
    "line:\nVerdict: CORRECT (the proposed solution is correct)\nor\nVerdict: INCORRECT (the proposed solution is wrong)"
)
DEFAULT_FORCED_VERDICT_TEXT = "\n\n**Verdict (CORRECT or INCORRECT):**"
DEFAULT_ROUND_FAILURE_TEXT = "(This verification round ended without a final response.)"
DEFAULT_NO_SOLUTION_TEXT = "(The solution ended without a final answer.)"


def parse_verdict(final_text: str | None, forced_header: str | None = "**Verdict (CORRECT or INCORRECT):**") -> int | None:
    """1 for CORRECT, 0 for INCORRECT, None when the final-response section has no verdict.

    If ``forced_header`` (the forced verdict text) occurs, the verdict is the first correct / incorrect / wrong /
    not correct word in the 120 characters after its last occurrence. Otherwise a "Verdict: X" line or \\boxed{X}
    is accepted, and the last one in the text wins.
    """
    if not final_text:
        return None
    if forced_header and forced_header in final_text:
        m = FORCED_CONTINUATION_RE.search(final_text.rpartition(forced_header)[2][:120])
        if m:
            word = m.group(0).lower()
            return 1 if word in ("correct", "right", "正确") else 0
    matches = [m for regex in (VERDICT_RE, BOXED_VERDICT_RE) for m in regex.finditer(final_text)]
    if not matches:
        return None
    return int(max(matches, key=lambda m: m.end()).group(1).lower() == "correct")


def relative_from_absolute(judgments: list[int | None]) -> list[int | None]:
    """The note's relative verdicts from absolute judgments of S: v_1 = a_1, v_d = 1[a_d = a_{d-1}] (None if either
    judgment is missing)."""
    out = []
    for d, a in enumerate(judgments):
        if a is None or (d > 0 and judgments[d - 1] is None):
            out.append(None)
        else:
            out.append(a if d == 0 else int(a == judgments[d - 1]))
    return out


def absolute_matches(solution_correct: int, judgments: list[int | None]) -> list[int]:
    """m_d = 1[a_d = Verify(S, R)]: equals the note's matches whenever every judgment is present."""
    return [int(a is not None and a == int(solution_correct)) for a in judgments]


def target_chain(solution_correct: int, verdicts: list[int | None]) -> tuple[list[int], list[int]]:
    """(targets v*_1..v*_k, matches m_1..m_k) for one chain."""
    targets, matches = [], []
    target = int(solution_correct)
    for v in verdicts:
        targets.append(target)
        m = int(v is not None and v == target)
        matches.append(m)
        target = m  # v*_{d+1} = 1[v_d = v*_d]
    return targets, matches


def round_scores(matches: list[int], format_ok: list[int]) -> list[int]:
    """Per-round scores: +1 verdict matches, 0 the policy's own verdict does not match, -1 format violation."""
    return [(1 if m else 0) if ok else -1 for m, ok in zip(matches, format_ok, strict=True)]


def depth_reward(matches: list[int], format_ok: list[int] | None = None) -> float:
    """(1/k) * sum_d s_d with s_d from ``round_scores``; without ``format_ok``, the note's (1/k) * sum_d m_d."""
    scores = list(matches) if format_ok is None else round_scores(matches, format_ok)
    return float(sum(scores)) / len(scores)


def implied_beliefs(verdicts: list[int | None]) -> list[int | None]:
    """Belief about S implied by the chain after each round (v_d = 1 keeps it, v_d = 0 flips it); None once a verdict
    is missing. A match at depth d happens exactly when this belief equals Verify(S, R)."""
    beliefs, b = [], None
    for d, v in enumerate(verdicts):
        if v is None or (d > 0 and b is None):
            b = None
        else:
            b = v if d == 0 else (b if v == 1 else 1 - b)
        beliefs.append(b)
    return beliefs


def chain_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """GRPO advantages over the chains of one question (verl convention: unbiased std, epsilon; one chain -> 0)."""
    r = np.asarray(rewards, dtype=np.float64)
    if r.size < 2:
        return [0.0] * r.size
    return list((r - r.mean()) / (r.std(ddof=1) + eps))


def round_advantage_scale(round_token_counts: list[int]) -> list[float]:
    """Per-round multipliers that make seq-mean-token-mean over rounds equal a token mean over the whole chain.

    The phase loss averages each round's token-mean over N_rounds = k * N_chains rows. Scaling round d's advantage by
    k * |o_d| / |chain| turns that into (1 / N_chains) * (1 / |chain|) * sum over all chain tokens, i.e. eq. (1) of the
    note applied to whole chains. Exact for the one-optimizer-step-per-batch update (ratio = 1, linear in A).
    """
    k, total = len(round_token_counts), sum(round_token_counts)
    if total == 0:
        return [0.0] * k
    return [k * n / total for n in round_token_counts]


class VerifyPromptBuilder:
    """Round-d prompt from (question text, solution final-response section, earlier rounds' final-response sections).

    It never receives the reference answer, the solution's reasoning, other solutions or other chains.
    """

    def __init__(
        self,
        question_header: str = DEFAULT_QUESTION_HEADER,
        solution_header: str = DEFAULT_SOLUTION_HEADER,
        round_header: str = DEFAULT_ROUND_HEADER,
        round1_instruction: str = DEFAULT_ROUND1_INSTRUCTION,
        later_round_instruction: str = DEFAULT_LATER_ROUND_INSTRUCTION,
    ):
        self.question_header = question_header
        self.solution_header = solution_header
        self.round_header = round_header
        self.round1_instruction = round1_instruction
        self.later_round_instruction = later_round_instruction

    def build(self, question: str, solution: str, previous_rounds: list[str]) -> list[dict]:
        parts = [f"{self.question_header}\n{question.strip()}", f"{self.solution_header}\n{solution.strip()}"]
        for d, text in enumerate(previous_rounds, start=1):
            parts.append(f"{self.round_header.format(d=d)}\n{text.strip()}")
        d = len(previous_rounds) + 1
        parts.append(self.round1_instruction if d == 1 else self.later_round_instruction.format(prev=d - 1))
        return [{"role": "user", "content": "\n\n".join(parts)}]


def builder_from_config(verify_cfg) -> VerifyPromptBuilder:
    judgment = str(verify_cfg.get("judgment", "relative"))
    if judgment not in ("relative", "absolute"):
        raise ValueError(f"unknown verify.judgment {judgment!r} (relative | absolute)")
    later = verify_cfg.later_round_instruction_absolute if judgment == "absolute" else verify_cfg.later_round_instruction
    return VerifyPromptBuilder(
        question_header=verify_cfg.question_header,
        solution_header=verify_cfg.solution_header,
        round_header=verify_cfg.round_header,
        round1_instruction=verify_cfg.round1_instruction,
        later_round_instruction=later,
    )


# ---------------------------------------------------------------- version 2: running conclusion (note, eq. 5-13)
DEFAULT_PREVIOUS_ROUND_HEADER = "Previous verification round:"
DEFAULT_CONCLUSION_TEXT = "Running conclusion: the proposed solution is {conclusion}."
DEFAULT_CONCLUSION_CORRECT = "CORRECT"
DEFAULT_CONCLUSION_INCORRECT = "INCORRECT"
DEFAULT_CONCLUSION_UNKNOWN = "unknown (an earlier verification round gave no verdict)"
DEFAULT_LATER_ROUND_INSTRUCTION_V2 = (
    "The previous verification round above may contain mistakes. Check whether its verdict is right, i.e. whether "
    "the running conclusion about the proposed solution is correct. Write a short analysis, then end with exactly one "
    "line:\nVerdict: CORRECT (the previous verdict is right)\nor\nVerdict: INCORRECT (the previous verdict is wrong)"
)


def running_conclusions(verdicts: list[int | None]) -> list[int | None]:
    """Eq. (7): c_1 = v_1, c_d = c_{d-1} if v_d = 1 else 1 - c_{d-1}; None once a verdict is missing.
    Uses no reference answer. Whenever every verdict is present, 1[v_d = v*_d] = 1[c_d = Verify(S, R)] (eq. 11)."""
    return implied_beliefs(verdicts)


def round_rewards(matches: list[int], format_ok: list[int] | None = None, penalty: float = 0.0) -> list[float]:
    """Eq. (10): r_d = m_d in {0, 1}. With ``format_ok``, a round with a format violation gets ``penalty``
    (the format reward, -1/M) instead of m_d."""
    if format_ok is None:
        return [float(m) for m in matches]
    return [float(m) if ok else float(penalty) for m, ok in zip(matches, format_ok, strict=True)]


def fixed_baseline_advantages(rewards: list[float], baseline: float = 0.5) -> list[float]:
    """Eq. (13): A_d = r_d - 1/2 for every round on its own (no comparison across branches or depths)."""
    return [float(r) - baseline for r in rewards]


class ChainPromptBuilder:
    """Round-d prompt of version 2, eq. (5): (Q, S) at d = 1; (Q, S, A_{d-1}) at d >= 2, where A_{d-1} is the previous
    round's text after </think> followed by the running conclusion c_{d-1}. Earlier rounds, any reasoning and the
    reference answer are never shown."""

    def __init__(
        self,
        question_header: str = DEFAULT_QUESTION_HEADER,
        solution_header: str = DEFAULT_SOLUTION_HEADER,
        previous_round_header: str = DEFAULT_PREVIOUS_ROUND_HEADER,
        conclusion_text: str = DEFAULT_CONCLUSION_TEXT,
        conclusion_words: tuple[str, str, str] = (
            DEFAULT_CONCLUSION_INCORRECT, DEFAULT_CONCLUSION_CORRECT, DEFAULT_CONCLUSION_UNKNOWN),
        round1_instruction: str = DEFAULT_ROUND1_INSTRUCTION,
        later_round_instruction: str = DEFAULT_LATER_ROUND_INSTRUCTION_V2,
    ):
        self.question_header = question_header
        self.solution_header = solution_header
        self.previous_round_header = previous_round_header
        self.conclusion_text = conclusion_text
        self.conclusion_words = conclusion_words  # (c = 0, c = 1, c unknown)
        self.round1_instruction = round1_instruction
        self.later_round_instruction = later_round_instruction

    def conclusion_line(self, conclusion: int | None) -> str:
        word = self.conclusion_words[2] if conclusion is None else self.conclusion_words[int(conclusion)]
        return self.conclusion_text.format(conclusion=word)

    def build(self, question: str, solution: str, previous_final: str | None = None,
              conclusion: int | None = None) -> list[dict]:
        parts = [f"{self.question_header}\n{question.strip()}", f"{self.solution_header}\n{solution.strip()}"]
        if previous_final is None:
            parts.append(self.round1_instruction)
        else:
            parts.append(f"{self.previous_round_header}\n{previous_final.strip()}\n\n{self.conclusion_line(conclusion)}")
            parts.append(self.later_round_instruction)
        return [{"role": "user", "content": "\n\n".join(parts)}]


def chain_builder_from_config(verify_cfg) -> ChainPromptBuilder:
    return ChainPromptBuilder(
        question_header=verify_cfg.question_header,
        solution_header=verify_cfg.solution_header,
        previous_round_header=verify_cfg.previous_round_header,
        conclusion_text=verify_cfg.conclusion_text,
        conclusion_words=(verify_cfg.conclusion_incorrect, verify_cfg.conclusion_correct,
                          verify_cfg.conclusion_unknown),
        round1_instruction=verify_cfg.round1_instruction,
        later_round_instruction=verify_cfg.later_round_instruction_v2,
    )
