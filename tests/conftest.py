import os

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "allocate_repair", "config")


class FakeTokenizer:
    """Deterministic character-level tokenizer with a single-token think delimiter (like R1-distill / Qwen3)."""

    pad_token_id = 0
    eos_token_id = 1
    think_end_id = 2
    eos_token = "<eos>"
    chat_template = "{{ messages }} </think>"
    _offset = 10

    def get_vocab(self):
        return {"<pad>": 0, "<eos>": 1, "</think>": 2}

    def encode(self, text):
        ids, i = [], 0
        while i < len(text):
            if text.startswith("</think>", i):
                ids.append(self.think_end_id)
                i += len("</think>")
            else:
                ids.append(ord(text[i]) + self._offset)
                i += 1
        return ids

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text)}

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for t in ids:
            t = int(t)
            if t == self.think_end_id:
                out.append("</think>")
            elif t in (self.pad_token_id, self.eos_token_id):
                if not skip_special_tokens:
                    out.append(self.eos_token)
            else:
                out.append(chr(t - self._offset))
        return "".join(out)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        text = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<|assistant|>"
        return text


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


def compose_config(overrides=()):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    base = [
        "actor_rollout_ref.model.path=/tmp/model",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path=" + os.path.join(CONFIG_DIR, "agent_loops.yaml"),
        "data.train_files=/tmp/train.parquet",
        "data.val_files=/tmp/val.parquet",
    ]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="allocate_repair_trainer", overrides=base + list(overrides))
    OmegaConf.resolve(cfg)
    return cfg


@pytest.fixture
def config():
    return compose_config()


def actor_config(overrides=()):
    from verl.utils.config import omega_conf_to_dataclass

    return omega_conf_to_dataclass(compose_config(overrides).actor_rollout_ref.actor)


def make_padded_batch(prompt_lens, response_lens, prompt_width, response_width, seed=0):
    """Left-padded prompts / right-padded responses in verl's layout."""
    g = torch.Generator().manual_seed(seed)
    bs = len(prompt_lens)
    prompts = torch.zeros(bs, prompt_width, dtype=torch.long)
    responses = torch.zeros(bs, response_width, dtype=torch.long)
    attention = torch.zeros(bs, prompt_width + response_width, dtype=torch.long)
    response_mask = torch.zeros(bs, response_width, dtype=torch.long)
    for i, (p, r) in enumerate(zip(prompt_lens, response_lens, strict=True)):
        prompts[i, prompt_width - p :] = torch.randint(10, 100, (p,), generator=g)
        responses[i, :r] = torch.randint(10, 100, (r,), generator=g)
        attention[i, prompt_width - p : prompt_width + r] = 1
        response_mask[i, :r] = 1
    input_ids = torch.cat([prompts, responses], dim=1)
    position_ids = (attention.cumsum(-1) - 1).clamp(min=0) * attention
    return {
        "prompts": prompts,
        "responses": responses,
        "attention_mask": attention,
        "response_mask": response_mask,
        "input_ids": input_ids,
        "position_ids": position_ids,
    }


def flat_log_probs_from_padded(response_log_probs, attention_mask, prompt_width):
    """Inverse of verl's no_padding_2_padding: place per-response log-probs into a packed (total_nnz,) tensor.

    Prompt positions get their own independent values so tests can check they receive no gradient.
    """
    values, owners = [], []
    for i in range(attention_mask.shape[0]):
        p = int(attention_mask[i, :prompt_width].sum())
        r = int(attention_mask[i, prompt_width:].sum())
        seq = torch.zeros(p + r)
        # model output at position t predicts token t+1: response token k lives at index p - 1 + k
        seq[p - 1 : p - 1 + r] = response_log_probs[i, :r]
        seq[: p - 1] = -0.5
        seq[p - 1 + r :] = -0.25
        values.append(seq)
        owners.append((p, r))
    return torch.cat(values), owners


@pytest.fixture
def rng():
    return np.random.RandomState(0)
