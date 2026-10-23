"""Local structured logs: per-step metrics JSONL, bounded example dumps and a one-off run-info record."""

import datetime
import json
import math
import os
import platform
import socket
import subprocess
import sys

import numpy as np


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if hasattr(value, "item") and callable(value.item) and getattr(value, "numel", lambda: 2)() == 1:
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


def _git_commit(path: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _git_dirty(path: str) -> bool | None:
    try:
        out = subprocess.run(["git", "-C", path, "status", "--porcelain"], capture_output=True, text=True, timeout=10)
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def software_versions() -> dict:
    versions = {"python": sys.version.split()[0], "platform": platform.platform(), "hostname": socket.gethostname()}
    for name in ("torch", "vllm", "transformers", "ray", "tensordict", "flash_attn", "numpy", "wandb", "verl"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    try:
        import torch

        versions["torch_cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        import verl

        verl_root = os.path.dirname(os.path.dirname(os.path.abspath(verl.__file__)))
        versions["verl_path"] = verl_root
        versions["verl_commit"] = _git_commit(verl_root)
        versions["verl_dirty"] = _git_dirty(verl_root)
    except Exception:
        pass
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    versions["allocate_repair_commit"] = _git_commit(repo_root)
    versions["allocate_repair_dirty"] = _git_dirty(repo_root)
    return versions


class RunLogger:
    """Writes ``run_info.json``, ``metrics.jsonl`` and ``examples.jsonl`` under ``output_dir``."""

    def __init__(self, output_dir: str):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.output_dir, "metrics.jsonl")
        self.examples_path = os.path.join(self.output_dir, "examples.jsonl")

    def write_run_info(self, info: dict) -> None:
        info = {"written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **info}
        with open(os.path.join(self.output_dir, "run_info.json"), "w") as f:
            json.dump(_jsonable(info), f, indent=2)
        with open(os.path.join(self.output_dir, "run_info_history.jsonl"), "a") as f:
            f.write(json.dumps(_jsonable(info)) + "\n")

    def log_metrics(self, step: int, metrics: dict) -> None:
        record = {"step": step, "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **metrics}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(_jsonable(record)) + "\n")

    def log_examples(self, step: int, examples: list[dict]) -> None:
        with open(self.examples_path, "a") as f:
            for example in examples:
                f.write(json.dumps(_jsonable({"step": step, **example})) + "\n")
