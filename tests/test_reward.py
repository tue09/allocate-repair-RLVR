"""Reward conversion: verl's math_dapo returns +/-1 with an `acc` bool; training must see exactly 0/1."""

import pytest
from verl.utils.reward_score import math_dapo

from allocate_repair.reward_fn import compute_score, correctness


@pytest.mark.parametrize(
    "solution, gt, expected",
    [
        ("Thinking...\n</think>\nSo the result is 34.\n\nAnswer: 34", "34", 1.0),
        ("Thinking...\n</think>\nAnswer: 35", "34", 0.0),
        ("I am not sure.", "34", 0.0),  # no parsable answer
        ("... therefore \\boxed{12}", "12", 1.0),  # math_dapo's boxed fallback
        ("Answer: \\frac{1}{2}", "\\frac{1}{2}", 1.0),
    ],
)
def test_binary_reward(solution, gt, expected):
    out = compute_score("math_dapo", solution, gt, extra_info={})
    assert out["score"] == expected
    assert out["acc"] == expected
    assert out["score"] in (0.0, 1.0)


def test_conversion_matches_math_dapo_accuracy():
    for solution, gt in [("Answer: 7", "7"), ("Answer: 8", "7"), ("nothing", "7")]:
        raw = math_dapo.compute_score(solution, gt)
        assert raw["score"] in (-1.0, 1.0)
        out = correctness(solution, gt)
        assert out["score"] == (1.0 if raw["acc"] else 0.0)
        assert out["math_dapo_score"] == raw["score"]


def test_parse_failure_flag():
    assert compute_score("math_dapo", "no final line", "3")["answer_parsed"] == 0.0
    assert compute_score("math_dapo", "Answer: 4", "3")["answer_parsed"] == 1.0


def test_same_checker_for_every_data_source():
    sol, gt = "Answer: 42", "42"
    scores = {ds: compute_score(ds, sol, gt, extra_info={})["score"] for ds in ("math_dapo", "aime24", "math500")}
    assert set(scores.values()) == {1.0}


def test_aux_metric_only_for_eval_rows():
    train = compute_score("math_dapo", "Answer: 1", "1", extra_info={"aux_math_verify": False})
    assert "acc_math_verify" not in train
    # keys must be identical for all training rows (verl stacks them into arrays)
    assert set(train) == set(compute_score("math_dapo", "Answer: 2", "1", extra_info={}))
