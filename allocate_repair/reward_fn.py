"""Binary correctness reward used identically for phase 1, phase 2 and evaluation.

verl's ``math_dapo.compute_score`` returns ``{"score": +1/-1, "acc": bool, "pred": str}``. Training needs r in {0, 1},
so the correctness bit is converted explicitly here. The ground truth arrives through verifier metadata
(``reward_model.ground_truth``) and is never part of a prompt.

Configure with::

    reward.custom_reward_function.path=pkg://allocate_repair.reward_fn
    reward.custom_reward_function.name=compute_score
"""

import re

from verl.utils.reward_score import math_dapo

INVALID_PRED = "[INVALID]"
_ANSWER_LINE = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")


def correctness(solution_str: str, ground_truth: str) -> dict:
    result = math_dapo.compute_score(solution_str, ground_truth)
    correct = bool(result["acc"])
    pred = result.get("pred")
    pred = INVALID_PRED if pred is None else str(pred)
    return {
        "score": 1.0 if correct else 0.0,
        "acc": 1.0 if correct else 0.0,
        "pred": pred,
        "answer_parsed": 0.0 if pred == INVALID_PRED else 1.0,
        "math_dapo_score": float(result["score"]),
    }


def _raw_final_answer(solution_str: str) -> str | None:
    tail = solution_str[-300:]
    matches = _ANSWER_LINE.findall(tail)
    if matches:
        return matches[-1].strip().strip("$").strip()
    boxed = math_dapo.last_boxed_only_string(tail)
    return math_dapo.remove_boxed(boxed) if boxed is not None else None


def aux_math_verify(solution_str: str, ground_truth: str) -> float:
    """Evaluation-only auxiliary check (symbolic equivalence) on the extracted final answer line."""
    raw = _raw_final_answer(solution_str)
    if raw is None:
        return 0.0
    try:
        from verl.utils.reward_score import math_verify
    except Exception:
        return float("nan")
    return float(math_verify.compute_score("\\boxed{" + raw + "}", ground_truth))


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
    out = correctness(solution_str, ground_truth)
    if extra_info is not None and extra_info.get("aux_math_verify", False):
        out["acc_math_verify"] = aux_math_verify(solution_str, ground_truth)
    return out
