#!/usr/bin/env python3
"""Sanity-check the training environment: versions, GPUs, flash-attn kernel, vLLM import, verl commit."""

import importlib
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED = {"torch": "2.11.0", "vllm": "0.24.0", "transformers": "5.9.0", "verl": "0.9.1", "flash_attn": "2.8.3"}
EXPECTED_VERL_COMMIT = "1876b06d0a3e4e71e06230be10af14492ca8a75b"

failures = []


def check(name, fn):
    try:
        msg = fn()
        print(f"[ok]   {name}: {msg}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        failures.append(name)


def version(module_name):
    def _inner():
        module = importlib.import_module(module_name)
        v = getattr(module, "__version__", "unknown")
        want = EXPECTED.get(module_name)
        note = "" if want is None or v.split("+")[0] == want else f"  (expected {want})"
        return f"{v}{note}"

    return _inner


def cuda():
    import torch

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return f"CUDA {torch.version.cuda}, {len(names)} GPU(s): {names}"


def flash_attn_kernel():
    import torch
    from flash_attn import flash_attn_varlen_func

    q = torch.randn(16, 2, 64, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 7, 16], device="cuda", dtype=torch.int32)
    out = flash_attn_varlen_func(q, q, q, cu, cu, 9, 9, causal=True)
    return f"varlen kernel ok, out shape {tuple(out.shape)}"


def verl_commit():
    import verl

    root = os.path.dirname(os.path.dirname(os.path.abspath(verl.__file__)))
    commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert commit == EXPECTED_VERL_COMMIT, f"verl at {commit}, expected {EXPECTED_VERL_COMMIT} (v0.9.1)"
    return f"{commit} ({root})"


def allocate_repair():
    from allocate_repair.reward_fn import compute_score

    assert compute_score("math_dapo", "</think> Answer: 5", "5")["score"] == 1.0
    assert compute_score("math_dapo", "</think> Answer: 6", "5")["score"] == 0.0
    import allocate_repair.losses  # noqa: F401
    import allocate_repair.trainer  # noqa: F401

    return "imports ok, binary reward ok"


def math_verify():
    from verl.utils.reward_score import math_verify as mv

    score = mv.compute_score("\\boxed{\\frac{1}{2}}", "0.5")
    assert score == 1.0, f"expected 1.0, got {score}"
    return "symbolic check ok"


# guard needed: verl's math_verify uses a "spawn" process pool, which re-imports this module in the child
if __name__ == "__main__":
    for mod in ("torch", "vllm", "transformers", "ray", "tensordict", "flash_attn", "verl", "wandb"):
        check(mod, version(mod))
    check("cuda", cuda)
    check("flash-attn kernel", flash_attn_kernel)
    check("verl commit", verl_commit)
    check("allocate_repair", allocate_repair)
    check("math-verify (eval-only aux metric)", math_verify)

    if failures:
        print(f"\n{len(failures)} check(s) failed: {failures}")
        sys.exit(1)
    print("\nenvironment looks good")
