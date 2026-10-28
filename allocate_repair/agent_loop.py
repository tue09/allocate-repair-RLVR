"""Single-turn agent loops registered through ``allocate_repair/config/agent_loops.yaml``
(``actor_rollout_ref.rollout.agent.agent_loop_config_path``).

``TokenBuilderSingleTurnAgentLoop``: verl's single-turn loop, but the ``model_type`` that verl uses to pick its prompt
tokenization builder can be replaced (R1-Distill-Qwen's config.json says "qwen2", which selects the Qwen builder, but
its tokenizer uses DeepSeek chat tokens and has no ``<|im_end|>``). Used by plain verl GRPO (scripts/train_grpo_*.sh).

``PhaseAwareSingleTurnAgentLoop``: the allocate-repair loop with per-phase decoding settings. Phase-1 and phase-2 rows
carry ``ar_phase`` ("solve" / "repair") in their non-tensor batch; validation rows carry none and use
``allocate_repair.validation`` (temperature / top-p / top-k come from ``rollout.val_kwargs``). Its builder model_type
comes from ``allocate_repair.token_builder_model_type``.

Verification rounds (``ar_phase`` "verify", allocate_repair.method=verify_depth) with ``verify.forced_verdict``: the
model first generates with ``verify.verdict_reserve_tokens`` fewer tokens; if the text after its </think> holds no
verdict, ``verify.forced_verdict_text`` (preceded by a forced </think> when the thinking never closed) is appended and
the model generates the verdict with the reserved tokens. The forced text is masked out of the loss (response_mask 0,
like a tool response); the model's own tokens before and after it are trained.
"""

from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler import simple_timer

from allocate_repair.verify_depth import parse_verdict

THINK_CLOSE = "</think>"


def forced_verdict_suffix(text: str, forced_text: str) -> str | None:
    """None if the text after </think> already holds a verdict, otherwise the text to append before the verdict."""
    if THINK_CLOSE in text:
        return None if parse_verdict(text.rpartition(THINK_CLOSE)[2]) is not None else forced_text
    return "\n" + THINK_CLOSE + forced_text


def assemble_forced(ids1, logprobs1, suffix_ids, ids2, logprobs2, eos_id):
    """(response_ids, response_mask, response_logprobs) for: own tokens (EOS dropped) + forced suffix + own tokens."""
    ids1, logprobs1 = list(ids1), (list(logprobs1) if logprobs1 is not None else None)
    while ids1 and ids1[-1] == eos_id:
        ids1.pop()
        if logprobs1 is not None:
            logprobs1.pop()
    response = ids1 + list(suffix_ids) + list(ids2)
    mask = [1] * len(ids1) + [0] * len(suffix_ids) + [1] * len(ids2)
    logprobs = None
    if logprobs1 is not None and logprobs2 is not None:
        logprobs = logprobs1 + [0.0] * len(suffix_ids) + list(logprobs2)
    return response, mask, logprobs


class TokenBuilderSingleTurnAgentLoop(SingleTurnAgentLoop):
    def __init__(self, *args, token_builder_model_type: str | None = None, **kwargs):
        if token_builder_model_type is not None:
            kwargs["hf_model_type"] = str(token_builder_model_type)
        super().__init__(*args, **kwargs)


class PhaseAwareSingleTurnAgentLoop(TokenBuilderSingleTurnAgentLoop):
    def __init__(self, *args, **kwargs):
        builder_model_type = kwargs["trainer_config"].config.allocate_repair.get("token_builder_model_type")
        super().__init__(*args, token_builder_model_type=builder_model_type, **kwargs)
        ar_config = self.config.allocate_repair
        self.phase_config = {"solve": ar_config.solve, "repair": ar_config.repair, "validation": ar_config.validation}
        if ar_config.get("verify") is not None:  # verification rounds of allocate_repair.method=verify_depth
            self.phase_config["verify"] = ar_config.verify

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        phase = kwargs.get("ar_phase", None)
        phase = "validation" if phase is None else str(phase)
        cfg = self.phase_config[phase]
        params = dict(sampling_params)
        if phase != "validation":
            params.update(temperature=float(cfg.temperature), top_p=float(cfg.top_p), top_k=int(cfg.top_k))
        params["max_tokens"] = int(cfg.max_tokens)
        if phase == "verify" and bool(cfg.get("forced_verdict", False)):
            return await self._run_forced_verdict(params, cfg, **kwargs)
        return await super().run(params, **kwargs)

    async def _run_forced_verdict(self, params: dict[str, Any], cfg, **kwargs) -> AgentLoopOutput:
        """Single-turn verification round that always ends with a verdict (see module docstring). Text only."""
        messages = list(kwargs["raw_prompt"])
        prompt_ids = await self.ct_build_initial_tokens(messages)
        reserve = int(cfg.verdict_reserve_tokens)
        request_id = uuid4().hex  # same id for both calls: sticky server, prefix cache reused
        metrics = {}
        first = dict(params, max_tokens=max(1, int(params["max_tokens"]) - reserve))
        with simple_timer("generate_sequences", metrics):
            out1 = await self.server_manager.generate(request_id=request_id, prompt_ids=prompt_ids, sampling_params=first)
            ids1, logprobs1 = list(out1.token_ids), (list(out1.log_probs) if out1.log_probs else None)
            suffix = forced_verdict_suffix(self.tokenizer.decode(ids1, skip_special_tokens=True), cfg.forced_verdict_text)
            if suffix is None:
                response, mask, logprobs = ids1, [1] * len(ids1), logprobs1
            else:
                suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
                eos = self.tokenizer.eos_token_id
                own = list(ids1)
                while own and own[-1] == eos:
                    own.pop()
                second = dict(params, max_tokens=max(1, reserve - len(suffix_ids)))
                out2 = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids + own + suffix_ids, sampling_params=second
                )
                response, mask, logprobs = assemble_forced(
                    ids1, logprobs1, suffix_ids, out2.token_ids, out2.log_probs if out2.log_probs else None, eos
                )
        metrics["num_preempted"] = out1.num_preempted if out1.num_preempted is not None else -1
        n = self.response_length
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response[:n],
            response_mask=mask[:n],
            response_logprobs=logprobs[:n] if logprobs else None,
            multi_modal_data={},
            num_turns=2,
            metrics=metrics,
            extra_fields=dict(out1.extra_fields),
        )
        output.extra_fields.update({"turn_scores": [], "tool_rewards": [], "forced_verdict": int(suffix is not None)})
        return output
