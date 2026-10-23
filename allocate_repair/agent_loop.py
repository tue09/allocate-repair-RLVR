"""Single-turn agent loop with per-phase decoding settings.

Phase-1 and phase-2 rows carry ``ar_phase`` ("solve" / "repair") in their non-tensor batch; validation rows carry
none and use ``allocate_repair.validation`` (temperature / top-p / top-k come from ``rollout.val_kwargs``).

Registered for the rollout workers through ``allocate_repair/config/agent_loops.yaml``
(``actor_rollout_ref.rollout.agent.agent_loop_config_path``).
"""

from typing import Any

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop


class PhaseAwareSingleTurnAgentLoop(SingleTurnAgentLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ar_config = self.config.allocate_repair
        self.phase_config = {"solve": ar_config.solve, "repair": ar_config.repair, "validation": ar_config.validation}

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        phase = kwargs.get("ar_phase", None)
        phase = "validation" if phase is None else str(phase)
        cfg = self.phase_config[phase]
        params = dict(sampling_params)
        if phase != "validation":
            params.update(temperature=float(cfg.temperature), top_p=float(cfg.top_p), top_k=int(cfg.top_k))
        params["max_tokens"] = int(cfg.max_tokens)
        return await super().run(params, **kwargs)
