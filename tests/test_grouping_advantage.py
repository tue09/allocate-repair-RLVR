"""Parent-specific grouping, zero-variance groups, outcome statistics and budget checks."""

import numpy as np
import pytest
import torch
from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import compute_advantage

from allocate_repair.batching import (
    generation_budget,
    outcome_group_stats,
    repair_group_id,
    solve_group_id,
    validate_budget,
)


def grpo(rewards, uids, response_len=3):
    n = len(rewards)
    scores = torch.zeros(n, response_len)
    scores[:, -1] = torch.tensor(rewards, dtype=torch.float32)
    data = DataProto.from_dict(
        tensors={"token_level_rewards": scores, "response_mask": torch.ones(n, response_len, dtype=torch.long)},
        non_tensors={"uid": np.array(uids, dtype=object)},
    )
    data = compute_advantage(data, adv_estimator=AdvantageEstimator.GRPO, norm_adv_by_std_in_grpo=True)
    return data.batch["advantages"][:, 0].numpy()


def test_repairs_of_different_parents_are_not_mixed():
    # question q: 4 solve samples (1 correct). two failed parents, each with M=3 repairs.
    solve = [1, 0, 0, 0]
    rep_a = [1, 0, 0]  # parent y1
    rep_b = [1, 1, 0]  # parent y2
    uids = [solve_group_id("q")] * 4 + [repair_group_id("q/y1")] * 3 + [repair_group_id("q/y2")] * 3
    adv = grpo(solve + rep_a + rep_b, uids)

    def ref(r):
        r = torch.tensor(r, dtype=torch.float32)
        return ((r - r.mean()) / (r.std() + 1e-6)).numpy()  # verl convention: unbiased std, eps=1e-6

    np.testing.assert_allclose(adv[:4], ref(solve), atol=1e-6)
    np.testing.assert_allclose(adv[4:7], ref(rep_a), atol=1e-6)
    np.testing.assert_allclose(adv[7:], ref(rep_b), atol=1e-6)
    pooled = ref(rep_a + rep_b)
    assert not np.allclose(adv[4:], pooled)  # pooling the parents would give different advantages


def test_phase_one_advantages_unaffected_by_repairs():
    solve = [1, 0, 1, 0]
    alone = grpo(solve, [solve_group_id("q")] * 4)
    with_repairs = grpo(solve + [1, 0], [solve_group_id("q")] * 4 + [repair_group_id("q/y1")] * 2)
    np.testing.assert_allclose(alone, with_repairs[:4], atol=1e-7)


def test_group_ids_are_phase_specific():
    assert solve_group_id("x") != repair_group_id("x")


@pytest.mark.parametrize("rewards", [[0, 0, 0, 0], [1, 1, 1, 1]])
def test_zero_variance_groups_have_zero_advantage(rewards):
    assert np.all(grpo(rewards, ["g"] * 4) == 0.0)


def test_outcome_stats_and_denominators():
    rewards = [1, 0, 0, 0, 0, 0, 1, 1]
    groups = ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2
    s = outcome_group_stats(rewards, groups, "solve")
    assert s["solve/num_groups"] == 4 and s["solve/num_samples"] == 8
    assert s["solve/reward_mean"] == 3 / 8
    assert (s["solve/num_groups_all_zero"], s["solve/num_groups_all_one"], s["solve/num_groups_mixed"]) == (2, 1, 1)
    assert s["solve/frac_groups_any_correct"] == 0.5


def test_outcome_stats_when_no_repairs_exist():
    s = outcome_group_stats([], [], "repair")
    assert s["repair/available"] == 0 and s["repair/num_groups"] == 0
    assert s["repair/reward_mean"] is None and s["repair/frac_groups_any_correct"] is None


def test_single_repair_advantage_is_its_reward():
    # M=1: every repair group has one sample -> verl uses mean 0, std 1 -> advantage = raw 0/1 reward
    solve = [1, 0, 0, 0]
    repairs = [1, 0, 0]  # one repair for each of the 3 failed parents
    uids = [solve_group_id("q")] * 4 + [repair_group_id(f"q/y{i}") for i in (1, 2, 3)]
    adv = grpo(solve + repairs, uids)
    np.testing.assert_allclose(adv[4:], [1 / (1 + 1e-6), 0.0, 0.0], atol=1e-6)


def test_budget_rules():
    validate_budget(2, 0)
    validate_budget(4, 1)
    validate_budget(8, 4)
    for G, M in [(1, 4), (8, -1)]:
        with pytest.raises(ValueError):
            validate_budget(G, M)
    # G + F_q * M generations per question, not G + M
    assert generation_budget([0, 3, 8], G=8, M=4) == 3 * 8 + (3 + 8) * 4
