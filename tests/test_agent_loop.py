"""The configured tokenization-builder model_type must reach verl's agent loop (R1-distill has no <|im_end|>)."""

import os
from types import SimpleNamespace

from conftest import CONFIG_DIR, compose_config
from omegaconf import OmegaConf
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

from allocate_repair.agent_loop import PhaseAwareSingleTurnAgentLoop, TokenBuilderSingleTurnAgentLoop


def _capture_verl_kwargs(monkeypatch):
    captured = {}

    def fake_init(self, *args, **kwargs):
        captured.update(kwargs)
        self.config = kwargs["trainer_config"].config

    monkeypatch.setattr(SingleTurnAgentLoop, "__init__", fake_init)
    return captured


def test_registry_entries():
    entries = {e.name: e for e in OmegaConf.load(os.path.join(CONFIG_DIR, "agent_loops.yaml"))}
    assert entries["allocate_repair_agent"]._target_.endswith("PhaseAwareSingleTurnAgentLoop")
    r1 = entries["r1_single_turn_agent"]
    assert r1._target_.endswith("TokenBuilderSingleTurnAgentLoop")
    assert r1.token_builder_model_type == compose_config().allocate_repair.token_builder_model_type == "deepseek_v3"


def test_token_builder_loop_overrides_model_type(monkeypatch):
    captured = _capture_verl_kwargs(monkeypatch)
    trainer_config = SimpleNamespace(config=compose_config())
    TokenBuilderSingleTurnAgentLoop(trainer_config=trainer_config, hf_model_type="qwen2",
                                    token_builder_model_type="deepseek_v3")
    assert captured["hf_model_type"] == "deepseek_v3" and "token_builder_model_type" not in captured

    captured.clear()
    TokenBuilderSingleTurnAgentLoop(trainer_config=trainer_config, hf_model_type="qwen2")
    assert captured["hf_model_type"] == "qwen2"


def test_phase_aware_loop_uses_allocate_repair_config(monkeypatch):
    captured = _capture_verl_kwargs(monkeypatch)
    PhaseAwareSingleTurnAgentLoop(trainer_config=SimpleNamespace(config=compose_config()), hf_model_type="qwen2")
    assert captured["hf_model_type"] == "deepseek_v3"

    captured.clear()
    cfg = compose_config(["allocate_repair.token_builder_model_type=null"])
    loop = PhaseAwareSingleTurnAgentLoop(trainer_config=SimpleNamespace(config=cfg), hf_model_type="qwen2")
    assert captured["hf_model_type"] == "qwen2"
    assert set(loop.phase_config) == {"solve", "repair", "validation", "verify"}
    assert loop.phase_config["verify"].max_tokens == cfg.allocate_repair.verify.max_tokens
