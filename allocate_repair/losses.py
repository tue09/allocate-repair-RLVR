"""Phase-separated GRPO loss: L_total = L_solve + lambda * L_repair.

Each phase reuses verl's registered policy loss (clipped PPO objective for ``loss_mode=vanilla``) with
``seq-mean-token-mean`` aggregation: mean over valid generated tokens of each trajectory, then mean over the
trajectories of that phase in the *full distributed batch*. The per-phase denominators N_solve / N_repair are
global trajectory counts computed on the driver, so

* micro-batching and DP sharding do not change the gradient (verl's FSDP averages gradients over DP ranks and
  ``agg_loss`` multiplies by ``dp_size`` to undo that average),
* increasing M does not increase the weight of the repair phase -- only lambda does,
* ranks / micro-batches without repair rows contribute an exact zero to the repair term.

Optional KL (``actor.use_kl_loss``) and entropy terms use verl's ``kl_penalty`` / ``agg_loss`` and are added once
inside each phase, so the repair KL is weighted by lambda together with the rest of L_repair.
"""

from tensordict import TensorDict
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig
from verl.workers.utils.padding import no_padding_2_padding

from allocate_repair.batching import PHASE_REPAIR, PHASE_SOLVE

LOSS_AGG_MODE = "seq-mean-token-mean"
PHASE_COUNT_KEYS = {"solve": "ar_num_solve", "repair": "ar_num_repair"}
LAMBDA_KEY = "ar_repair_lambda"


def _sum_metric(value) -> Metric:
    return Metric(aggregation=AggregationType.SUM, value=value)


def phase_weighted_ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    dp_size = data["dp_size"]
    repair_lambda = float(tu.get_non_tensor_data(data=data, key=LAMBDA_KEY, default=None))
    num_global = {
        name: int(tu.get_non_tensor_data(data=data, key=key, default=None)) for name, key in PHASE_COUNT_KEYS.items()
    }

    fields = ["response_mask", "old_log_probs", "advantages", "ar_phase"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if config.use_kl_loss:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    phase = data["ar_phase"].to(response_mask.device)
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)
    policy_loss_fn = get_policy_loss_fn(config.policy_loss.get("loss_mode", "vanilla"))

    total_loss = None
    metrics = {}
    for name, code, weight in (("solve", PHASE_SOLVE, 1.0), ("repair", PHASE_REPAIR, repair_lambda)):
        mask = response_mask & (phase == code).unsqueeze(-1)
        n_global = num_global[name]
        local_tokens = mask.sum()
        local_seqs = (mask.sum(-1) > 0).sum()
        zero = log_prob.sum() * 0.0
        if n_global == 0:
            phase_loss = pg_loss = kl_loss = entropy_loss = zero
            clipfrac = clipfrac_lower = ppo_kl = 0.0
        else:
            # ActorConfig fields are frozen; like verl's ppo_loss, update the dict in place
            config.global_batch_info.update(
                dp_size=dp_size,
                batch_num_tokens=None,
                global_batch_size=n_global,
                loss_scale_factor=config.loss_scale_factor,
            )
            pg_loss, pg_metrics = policy_loss_fn(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                response_mask=mask,
                loss_agg_mode=LOSS_AGG_MODE,
                config=config,
                rollout_is_weights=rollout_is_weights,
            )
            clipfrac = pg_metrics.get("actor/pg_clipfrac", 0.0)
            clipfrac_lower = pg_metrics.get("actor/pg_clipfrac_lower", 0.0)
            ppo_kl = pg_metrics.get("actor/ppo_kl", 0.0)
            phase_loss = pg_loss
            entropy_loss = zero
            if entropy is not None:
                entropy_loss = agg_loss(entropy, mask, LOSS_AGG_MODE, dp_size=dp_size, global_batch_size=n_global)
                if config.entropy_coeff != 0:
                    phase_loss = phase_loss - config.entropy_coeff * entropy_loss
            kl_loss = zero
            if config.use_kl_loss:
                kld = kl_penalty(logprob=log_prob, ref_logprob=data["ref_log_prob"], kl_penalty=config.kl_loss_type)
                kl_loss = agg_loss(kld, mask, LOSS_AGG_MODE, dp_size=dp_size, global_batch_size=n_global)
                phase_loss = phase_loss + config.kl_loss_coef * kl_loss

        total_loss = weight * phase_loss if total_loss is None else total_loss + weight * phase_loss

        # Losses are already scaled by dp_size / N_phase: SUM over micro-batches of the DP-mean is the global value.
        # Raw counts are multiplied by dp_size for the same reason (they come back as global sums).
        tokens = float(local_tokens.item())
        metrics[f"ar/{name}/loss"] = _sum_metric(phase_loss.detach())
        metrics[f"ar/{name}/weighted_loss"] = _sum_metric((weight * phase_loss).detach())
        metrics[f"ar/{name}/pg_loss"] = _sum_metric(pg_loss.detach())
        metrics[f"ar/{name}/kl_loss"] = _sum_metric(kl_loss.detach())
        metrics[f"ar/{name}/entropy"] = _sum_metric(entropy_loss.detach())
        metrics[f"ar/{name}/num_tokens_x_dp"] = _sum_metric(tokens * dp_size)
        metrics[f"ar/{name}/num_seqs_x_dp"] = _sum_metric(float(local_seqs.item()) * dp_size)
        metrics[f"ar/{name}/clipped_tokens_x_dp"] = _sum_metric(float(clipfrac) * tokens * dp_size)
        metrics[f"ar/{name}/clipped_lower_tokens_x_dp"] = _sum_metric(float(clipfrac_lower) * tokens * dp_size)
        metrics[f"ar/{name}/ppo_kl_token_sum_x_dp"] = _sum_metric(float(ppo_kl) * tokens * dp_size)

    metrics["ar/total_loss"] = _sum_metric(total_loss.detach())
    return total_loss, metrics


def finalize_loss_metrics(actor_metrics: dict, repair_lambda: float) -> dict:
    """Turn the summed worker metrics into per-phase training metrics (driver side)."""
    out = {}
    for name in ("solve", "repair"):
        tokens = actor_metrics.get(f"actor/ar/{name}/num_tokens_x_dp", 0.0)
        out[f"train/{name}/loss"] = actor_metrics.get(f"actor/ar/{name}/loss")
        out[f"train/{name}/pg_loss"] = actor_metrics.get(f"actor/ar/{name}/pg_loss")
        out[f"train/{name}/kl_loss"] = actor_metrics.get(f"actor/ar/{name}/kl_loss")
        out[f"train/{name}/entropy"] = actor_metrics.get(f"actor/ar/{name}/entropy")
        out[f"train/{name}/num_loss_tokens"] = tokens
        out[f"train/{name}/num_loss_trajectories"] = actor_metrics.get(f"actor/ar/{name}/num_seqs_x_dp", 0.0)
        for key in ("clipped_tokens", "clipped_lower_tokens"):
            count = actor_metrics.get(f"actor/ar/{name}/{key}_x_dp", 0.0)
            out[f"train/{name}/{key}"] = count
            out[f"train/{name}/{key.replace('_tokens', '')}_frac"] = count / tokens if tokens > 0 else None
        kl_sum = actor_metrics.get(f"actor/ar/{name}/ppo_kl_token_sum_x_dp", 0.0)
        out[f"train/{name}/ppo_kl"] = kl_sum / tokens if tokens > 0 else None
    out["train/repair/weighted_loss"] = actor_metrics.get("actor/ar/repair/weighted_loss")
    out["train/repair_lambda"] = repair_lambda
    out["train/total_loss"] = actor_metrics.get("actor/ar/total_loss")
    return out
