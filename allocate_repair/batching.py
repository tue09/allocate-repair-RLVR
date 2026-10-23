"""Pure helpers for phase bookkeeping: identifiers, outcome statistics and budget checks.

Kept free of Ray / GPU dependencies so the focused unit tests can exercise them directly.
"""

from collections import OrderedDict

import numpy as np

PHASE_SOLVE = 0
PHASE_REPAIR = 1
PHASE_PAD = -1
PHASE_NAMES = {PHASE_SOLVE: "solve", PHASE_REPAIR: "repair"}


def question_uid(step: int, index: int) -> str:
    return f"s{step}-q{index}"


def trajectory_uid(parent: str, sample: int, phase: int) -> str:
    return f"{parent}/{'y' if phase == PHASE_SOLVE else 'z'}{sample}"


def solve_group_id(q_uid: str) -> str:
    """GRPO group for phase 1: the G responses to one question."""
    return f"solve::{q_uid}"


def repair_group_id(parent_traj_uid: str) -> str:
    """GRPO group for phase 2: the M repairs of one specific failed response."""
    return f"repair::{parent_traj_uid}"


def validate_budget(G: int, M: int) -> None:
    if G < 2:
        raise ValueError(f"G (rollout.n) must be >= 2 for GRPO, got {G}")
    if M != 0 and M < 2:
        raise ValueError(f"M must be 0 (solve-only baseline) or >= 2 when repair training is enabled, got {M}")


def generation_budget(num_failures_per_question: list[int], G: int, M: int) -> int:
    """Number of generations for a batch: sum_q (G + F_q * M)."""
    return int(sum(G + f * M for f in num_failures_per_question))


def outcome_group_stats(rewards, group_ids, prefix: str) -> dict:
    """Mean reward and all-zero / all-one / mixed group fractions, with raw counts as denominators.

    Returns explicit ``{prefix}/available = 0`` and ``None`` rates when there are no samples.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    out = {f"{prefix}/num_samples": int(rewards.size)}
    groups: "OrderedDict[str, list[float]]" = OrderedDict()
    for gid, r in zip(group_ids, rewards, strict=True):
        groups.setdefault(gid, []).append(float(r))
    n_groups = len(groups)
    out[f"{prefix}/num_groups"] = n_groups
    if rewards.size == 0:
        out[f"{prefix}/available"] = 0
        for key in (
            "reward_mean",
            "frac_groups_any_correct",
            "frac_groups_all_zero",
            "frac_groups_all_one",
            "frac_groups_mixed",
        ):
            out[f"{prefix}/{key}"] = None
        for key in (
            "num_correct",
            "num_groups_any_correct",
            "num_groups_all_zero",
            "num_groups_all_one",
            "num_groups_mixed",
        ):
            out[f"{prefix}/{key}"] = 0
        return out
    all_zero = sum(1 for v in groups.values() if max(v) == 0.0)
    all_one = sum(1 for v in groups.values() if min(v) == 1.0)
    mixed = n_groups - all_zero - all_one
    any_correct = n_groups - all_zero
    out.update(
        {
            f"{prefix}/available": 1,
            f"{prefix}/num_correct": int(rewards.sum()),
            f"{prefix}/reward_mean": float(rewards.mean()),
            f"{prefix}/num_groups_any_correct": any_correct,
            f"{prefix}/num_groups_all_zero": all_zero,
            f"{prefix}/num_groups_all_one": all_one,
            f"{prefix}/num_groups_mixed": mixed,
            f"{prefix}/frac_groups_any_correct": any_correct / n_groups,
            f"{prefix}/frac_groups_all_zero": all_zero / n_groups,
            f"{prefix}/frac_groups_all_one": all_one / n_groups,
            f"{prefix}/frac_groups_mixed": mixed / n_groups,
        }
    )
    return out


def scalar_stats(values, prefix: str) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}/count": 0,
            f"{prefix}/mean": None,
            f"{prefix}/std": None,
            f"{prefix}/min": None,
            f"{prefix}/max": None,
        }
    return {
        f"{prefix}/count": int(values.size),
        f"{prefix}/mean": float(values.mean()),
        f"{prefix}/std": float(values.std()),
        f"{prefix}/min": float(values.min()),
        f"{prefix}/max": float(values.max()),
    }


def advantage_stats(advantages, prefix: str) -> dict:
    values = np.asarray(advantages, dtype=np.float64)
    out = scalar_stats(values, prefix)
    out[f"{prefix}/num_zero"] = int((np.abs(values) < 1e-8).sum())
    out[f"{prefix}/frac_zero"] = float((np.abs(values) < 1e-8).mean()) if values.size else None
    out[f"{prefix}/abs_mean"] = float(np.abs(values).mean()) if values.size else None
    return out


def length_stats(lengths, prefix: str) -> dict:
    values = np.asarray(lengths, dtype=np.float64)
    out = scalar_stats(values, prefix)
    out[f"{prefix}/sum"] = int(values.sum()) if values.size else 0
    for q in (50, 90, 99):
        out[f"{prefix}/p{q}"] = float(np.percentile(values, q)) if values.size else None
    return out
