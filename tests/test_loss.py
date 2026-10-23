"""Phase-separated loss: exact value, input masking, empty repair data, gradient accumulation, M / lambda checks."""

import pytest
import torch
from conftest import actor_config, flat_log_probs_from_padded, make_padded_batch
from tensordict import TensorDict
from verl.trainer.ppo.core_algos import kl_penalty
from verl.utils import tensordict_utils as tu

from allocate_repair.batching import PHASE_PAD, PHASE_REPAIR, PHASE_SOLVE
from allocate_repair.losses import phase_weighted_ppo_loss

PW, RW = 6, 8  # padded prompt / response widths


def reference_loss(cur, old, adv, mask, phases, lam, cfg, ref=None):
    """Independent re-implementation: per-trajectory token mean, then mean over each phase's trajectories."""
    eps_lo, eps_hi, c = cfg.clip_ratio_low, cfg.clip_ratio_high, cfg.clip_ratio_c
    ratio = torch.exp(torch.clamp(cur - old, -20, 20))
    pg1, pg2 = -adv * ratio, -adv * torch.clamp(ratio, 1 - eps_lo, 1 + eps_hi)
    clip1 = torch.maximum(pg1, pg2)
    tok = torch.where(adv < 0, torch.minimum(-adv * c, clip1), clip1)
    if ref is not None:
        tok = tok + cfg.kl_loss_coef * kl_penalty(cur, ref, cfg.kl_loss_type)
    per_traj = (tok * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    total = 0.0
    for code, w in ((PHASE_SOLVE, 1.0), (PHASE_REPAIR, lam)):
        sel = phases == code
        if sel.any():
            total = total + w * per_traj[sel].mean()
    return total


def build(phases, seed=0, prompt_lens=None, response_lens=None, mask_tail=None, theta=None, features=None):
    """Synthetic batch; current log-probs = old + 0.1 * features @ theta (theta is the shared 'model')."""
    g = torch.Generator().manual_seed(seed)
    bs = len(phases)
    prompt_lens = prompt_lens or [int(x) for x in torch.randint(2, PW + 1, (bs,), generator=g)]
    response_lens = response_lens or [int(x) for x in torch.randint(3, RW + 1, (bs,), generator=g)]
    padded = make_padded_batch(prompt_lens, response_lens, PW, RW, seed=seed)
    mask = padded["response_mask"].clone()
    if mask_tail:
        for row, k in mask_tail.items():
            r = response_lens[row]
            mask[row, r - k : r] = 0
    old = -torch.rand(bs, RW, generator=g) * 2
    adv_row = torch.randn(bs, generator=g)
    adv = adv_row[:, None] * mask
    if theta is None:
        theta = torch.zeros(4, requires_grad=True)
    if features is None:
        features = torch.randn(bs, RW, 4, generator=g)
    cur = old + 0.3 * torch.randn(bs, RW, generator=g) + 0.1 * (features @ theta)
    values, _ = flat_log_probs_from_padded(cur, padded["attention_mask"], PW)
    data = {
        **padded,
        "response_mask": mask,
        "old_log_probs": old * mask,
        "advantages": adv,
        "ar_phase": torch.tensor(phases, dtype=torch.long),
    }
    return data, values, cur, theta, features


def run_loss(cfg, data, values, num_solve, num_repair, lam, dp_size=1, rows=None, ref=None):
    rows = list(range(len(data["ar_phase"]))) if rows is None else rows
    batch = {k: v[rows] for k, v in data.items()}
    if ref is not None:
        batch["ref_log_prob"] = ref[rows]
    td = TensorDict(batch, batch_size=[len(rows)])
    tu.assign_non_tensor(td, dp_size=dp_size, ar_num_solve=num_solve, ar_num_repair=num_repair, ar_repair_lambda=lam)
    # repack the selected rows' log-probs
    seq_lens = data["attention_mask"].sum(-1).tolist()
    offsets = [0]
    for n in seq_lens:
        offsets.append(offsets[-1] + n)
    packed = torch.cat([values[offsets[i] : offsets[i + 1]] for i in rows])
    return phase_weighted_ppo_loss(cfg, {"log_probs": packed}, td)


PHASES = [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]


def counts(phases):
    return sum(p == PHASE_SOLVE for p in phases), sum(p == PHASE_REPAIR for p in phases)


def test_matches_reference_formula():
    cfg = actor_config()
    data, values, cur, *_ = build(PHASES)
    ns, nr = counts(PHASES)
    loss, metrics = run_loss(cfg, data, values, ns, nr, lam=0.7)
    ref = reference_loss(
        cur, data["old_log_probs"], data["advantages"], data["response_mask"].float(), data["ar_phase"], 0.7, cfg
    )
    torch.testing.assert_close(loss, ref, rtol=1e-5, atol=1e-6)
    parts = metrics["ar/solve/loss"].aggregate() + 0.7 * metrics["ar/repair/loss"].aggregate()
    assert abs(parts - loss.item()) < 1e-5


def test_input_masked_and_padding_tokens_get_no_gradient():
    cfg = actor_config()
    phases = [0, 1, 0, PHASE_PAD]
    data, values, *_ = build(phases, prompt_lens=[4, 5, 3, 4], response_lens=[6, 7, 5, 6], mask_tail={0: 2})
    values = values.detach().requires_grad_(True)
    loss, _ = run_loss(cfg, data, values, num_solve=2, num_repair=1, lam=1.0)
    loss.backward()
    grad = values.grad
    offset = 0
    for i, (p, r) in enumerate(zip([4, 5, 3, 4], [6, 7, 5, 6], strict=True)):
        seq = grad[offset : offset + p + r]
        resp = seq[p - 1 : p - 1 + r]
        assert torch.all(seq[: p - 1] == 0), "prompt positions must not receive gradient"
        assert torch.all(seq[p - 1 + r :] == 0)
        if phases[i] == PHASE_PAD:
            assert torch.all(resp == 0), "padding rows must not receive gradient"
        elif i == 0:
            assert torch.all(resp[-2:] == 0), "masked response tokens must not receive gradient"
            assert resp[:-2].abs().sum() > 0
        else:
            assert resp.abs().sum() > 0
        offset += p + r


def test_empty_repair_batch():
    cfg = actor_config()
    phases = [0, 0, 0, 0]
    data, values, cur, *_ = build(phases)
    loss, metrics = run_loss(cfg, data, values, num_solve=4, num_repair=0, lam=1.0)
    ref = reference_loss(
        cur, data["old_log_probs"], data["advantages"], data["response_mask"].float(), data["ar_phase"], 1.0, cfg
    )
    torch.testing.assert_close(loss, ref)
    assert torch.isfinite(loss)
    assert metrics["ar/repair/loss"].aggregate() == 0.0 and metrics["ar/repair/num_seqs_x_dp"].aggregate() == 0.0


def test_micro_batch_without_repair_rows_is_safe():
    cfg = actor_config()
    data, values, *_ = build(PHASES)
    solve_rows = [i for i, p in enumerate(PHASES) if p == PHASE_SOLVE]
    loss, metrics = run_loss(cfg, data, values, *counts(PHASES), lam=1.0, rows=solve_rows)
    assert torch.isfinite(loss)
    assert metrics["ar/repair/loss"].aggregate() == 0.0
    assert metrics["ar/repair/clipped_tokens_x_dp"].aggregate() == 0.0


def _theta_grad(cfg, phases, lam, splits, dp_size, seed=3):
    theta = torch.randn(4, generator=torch.Generator().manual_seed(11)).requires_grad_(True)
    feats = torch.randn(len(phases), RW, 4, generator=torch.Generator().manual_seed(12))
    data, values, *_ = build(phases, seed=seed, theta=theta, features=feats)
    ns, nr = counts(phases)
    total = torch.zeros(4)
    for rank_rows in splits:  # each rank: list of micro-batches
        rank_grad = torch.zeros(4)
        for mb in rank_rows:
            theta.grad = None
            loss, _ = run_loss(cfg, data, values, ns, nr, lam, dp_size=dp_size, rows=mb)
            (g,) = torch.autograd.grad(loss, theta, retain_graph=True)
            rank_grad += g
        total += rank_grad
    return total / len(splits)  # FSDP averages gradients over DP ranks


def test_gradient_accumulation_and_dp_sharding_match_full_batch():
    cfg = actor_config()
    full = _theta_grad(cfg, PHASES, lam=0.5, splits=[[list(range(12))]], dp_size=1)
    sharded = _theta_grad(
        cfg,
        PHASES,
        lam=0.5,
        dp_size=2,
        splits=[[[0, 5], [3], [9, 1, 7]], [[2, 4], [6, 8, 10, 11]]],  # uneven micro-batches; rank-local phases differ
    )
    torch.testing.assert_close(full, sharded, rtol=1e-5, atol=1e-6)


def test_repair_weight_does_not_grow_with_M():
    """Duplicating every repair trajectory (M -> 2M, identical rewards) leaves the update unchanged."""
    cfg = actor_config()
    theta = torch.randn(4, generator=torch.Generator().manual_seed(5)).requires_grad_(True)
    data, values, *_ = build(PHASES, seed=7, theta=theta)
    rep = [i for i, p in enumerate(PHASES) if p == PHASE_REPAIR]
    ns, nr = counts(PHASES)
    loss_m, _ = run_loss(cfg, data, values, ns, nr, lam=1.0)
    (g_m,) = torch.autograd.grad(loss_m, theta, retain_graph=True)
    rows_2m = list(range(len(PHASES))) + rep  # every repair appears twice
    loss_2m, _ = run_loss(cfg, data, values, ns, 2 * nr, lam=1.0, rows=rows_2m)
    (g_2m,) = torch.autograd.grad(loss_2m, theta, retain_graph=True)
    torch.testing.assert_close(loss_m, loss_2m)
    torch.testing.assert_close(g_m, g_2m)
    # sanity: a pooled sample count (no per-phase normalization) would have changed the repair weight
    data2 = {k: v.clone() for k, v in data.items()}
    data2["ar_phase"][:] = PHASE_SOLVE
    pooled_m, _ = run_loss(cfg, data2, values, ns + nr, 0, lam=1.0)
    pooled_2m, _ = run_loss(cfg, data2, values, ns + 2 * nr, 0, lam=1.0, rows=rows_2m)
    assert not torch.allclose(pooled_m, pooled_2m)


def test_lambda_scales_only_the_repair_term():
    cfg = actor_config()
    theta = torch.randn(4, generator=torch.Generator().manual_seed(9)).requires_grad_(True)
    data, values, *_ = build(PHASES, seed=9, theta=theta)
    ns, nr = counts(PHASES)
    grads = {}
    for lam in (0.0, 1.0, 2.0):
        loss, _ = run_loss(cfg, data, values, ns, nr, lam=lam)
        (grads[lam],) = torch.autograd.grad(loss, theta, retain_graph=True)
    torch.testing.assert_close(grads[2.0] - grads[0.0], 2 * (grads[1.0] - grads[0.0]), rtol=1e-5, atol=1e-6)
    solve_rows = [i for i, p in enumerate(PHASES) if p == PHASE_SOLVE]
    loss_solve, _ = run_loss(cfg, data, values, ns, nr, lam=0.0, rows=solve_rows)
    (g_solve,) = torch.autograd.grad(loss_solve, theta, retain_graph=True)
    torch.testing.assert_close(grads[0.0], g_solve, rtol=1e-5, atol=1e-6)


def test_kl_is_included_once_per_phase():
    cfg = actor_config(["actor_rollout_ref.actor.use_kl_loss=true", "actor_rollout_ref.actor.kl_loss_coef=0.05"])
    data, values, cur, *_ = build(PHASES, seed=4)
    ref = data["old_log_probs"] + 0.05 * torch.randn_like(data["old_log_probs"])
    ns, nr = counts(PHASES)
    loss, metrics = run_loss(cfg, data, values, ns, nr, lam=0.6, ref=ref)
    expected = reference_loss(
        cur,
        data["old_log_probs"],
        data["advantages"],
        data["response_mask"].float(),
        data["ar_phase"],
        0.6,
        cfg,
        ref=ref,
    )
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
    assert metrics["ar/solve/kl_loss"].aggregate() > 0 and metrics["ar/repair/kl_loss"].aggregate() > 0


def test_clip_fraction_metrics_are_counts():
    cfg = actor_config()
    data, values, *_ = build(PHASES, seed=2)
    _, metrics = run_loss(cfg, data, values, *counts(PHASES), lam=1.0)
    for name in ("solve", "repair"):
        tokens = metrics[f"ar/{name}/num_tokens_x_dp"].aggregate()
        clipped = metrics[f"ar/{name}/clipped_tokens_x_dp"].aggregate()
        assert tokens > 0 and 0 <= clipped <= tokens
        assert abs(clipped - round(clipped)) < 1e-3


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_total_metric_equals_returned_loss(lam):
    cfg = actor_config()
    data, values, *_ = build(PHASES, seed=1)
    loss, metrics = run_loss(cfg, data, values, *counts(PHASES), lam=lam)
    assert abs(metrics["ar/total_loss"].aggregate() - loss.item()) < 1e-6
