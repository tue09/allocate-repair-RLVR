#!/usr/bin/env python3
"""Upload a run's key results (figures, numbers) to a Hugging Face dataset repo every N minutes (or N steps).

    python scripts/hf_sync_results.py <EXP_NAME>                  # poll; upload every 10 minutes and at the end
    python scripts/hf_sync_results.py <EXP_NAME> --interval-min 0 # upload at steps 10, 20, ... (--every) instead
    python scripts/hf_sync_results.py <EXP_NAME> --once           # upload the current state now and exit

Runs next to training (started by the train_*.sh scripts, or in its own tmux session). Every --interval-min minutes,
if the run logged a new step since the last upload, it reruns scripts/plot_run.py and uploads result/<EXP_NAME>/
(figures, summary.txt, metrics.csv) plus key_metrics.md to <repo>/results/<project>/<EXP_NAME>/. Trajectories,
checkpoints and logs are never uploaded. It waits for the run to start, and stops after the final upload (last
training step + evaluation logged), or after --idle-hours without a new step.

Token: the last whitespace-separated field of --token-file (a line "hf_write: <token>" works), read again before every
upload (a replaced token is picked up without a restart); it is never printed.
"""

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
from huggingface_hub import HfApi
from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_ROOT = "/cm/archive/tue09_new/model_hub"
UPLOAD_PATTERNS = ["*.png", "summary.txt", "metrics.csv", "key_metrics.md"]


def find_run(name: str) -> str:
    if os.path.isdir(name):
        return os.path.abspath(name)
    hits = glob.glob(os.path.join(RUNS_ROOT, "*", name))
    if len(hits) != 1:
        sys.exit(f"run {name!r}: {len(hits)} matches under {RUNS_ROOT}")
    return hits[0]


def read_metrics(run_dir: str) -> dict[int, dict]:
    """step -> metrics (a later launch overrides the steps it repeats); half-written lines are skipped."""
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "metrics", "*.jsonl"))):
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[int(row["step"])] = row.get("data", row)
    return out


def fmt(x, digits=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def block_mean(metrics: dict, steps: list[int], key: str):
    vals = [metrics[s][key] for s in steps if isinstance(metrics[s].get(key), (int, float))]
    return float(np.mean(vals)) if vals else None


def config_path(run_dir: str) -> str:
    return os.path.join(run_dir, "hydra", ".hydra", "config.yaml")


def key_metrics_md(run_dir: str, metrics: dict, every: int) -> str:
    cfg = OmegaConf.load(config_path(run_dir))
    ar, v = cfg.allocate_repair, cfg.allocate_repair.verify
    G, M, total = int(cfg.actor_rollout_ref.rollout.n), int(ar.M), int(cfg.trainer.total_training_steps)
    v2 = str(ar.get("method")) == "verify_depth_v2"
    steps = sorted(metrics)
    last = steps[-1]
    depths = range(1, M + 1)
    cols = [
        ("solve reward", lambda b: block_mean(metrics, b, "solve/reward_mean")),
        ("mean round reward" if v2 else "verify reward (-1..1)", lambda b: block_mean(metrics, b, "verify/reward_mean")),
        *[(f"match d{d}", (lambda d: lambda b: block_mean(metrics, b, f"verify/d{d}/match_rate"))(d)) for d in depths],
        ("format viol. (mean over rounds)",
         lambda b: _mean([block_mean(metrics, b, f"verify/d{d}/frac_format_violation") for d in depths])),
        ("d1 acc. on correct sol.", lambda b: block_mean(metrics, b, "verify/d1/acc_on_correct_solutions")),
        ("d1 acc. on wrong sol.", lambda b: block_mean(metrics, b, "verify/d1/acc_on_wrong_solutions")),
    ]
    if v2:
        cols.append(("final conclusion right", lambda b: block_mean(metrics, b, "verify/final_belief_acc")))
    else:
        cols.append(("questions w/o verify signal",
                     lambda b: block_mean(metrics, b, "verify/frac_groups_zero_advantage")))
    if v2:
        penalty = v.get("format_penalty_reward")
        method_lines = [
            f"- G = {G} solutions per question; M = {M} verification depth (one chain per solution, note version 2): "
            "round d >= 2 sees only the previous round's final answer + the running conclusion c_(d-1); "
            "J = J_GRPO + J_verify",
            f"- every round is its own example: reward 1[v_d = v*_d], advantage reward - "
            f"{v.get('advantage_baseline')} (fixed baseline); format violation -> reward "
            f"{'-1/M' if penalty is None else penalty} (format_penalty={v.get('format_penalty')}); "
            f"forced_verdict={v.get('forced_verdict')}",
        ]
    else:
        method_lines = [
            f"- G = {G} solutions per question; M = {M} verification depth (one chain per solution); "
            "chain advantages normalized over the chains of a question; J = J_GRPO + J_verify",
            f"- round score +1/M right verdict, 0 wrong verdict, -1/M format violation "
            f"(format_penalty={v.get('format_penalty')}); judgment={v.get('judgment')}; "
            f"forced_verdict={v.get('forced_verdict')}",
        ]
    L = [f"# {os.path.basename(run_dir)}", "",
         f"Updated {dt.datetime.now():%Y-%m-%d %H:%M} at training step **{last} / {total}**.", "",
         "Scaling Verification Depth (note: `Scaling_Verification_Depth.pdf` in this repo), "
         f"model `{os.path.basename(str(cfg.actor_rollout_ref.model.path))}`, DAPO-Math-17k.", "",
         *method_lines,
         f"- batch {cfg.data.train_batch_size} questions, lr {cfg.actor_rollout_ref.actor.optim.lr}, "
         f"max tokens solve/verify {ar.solve.max_tokens}/{v.max_tokens}", "",
         f"## Means over blocks of {every} steps", "",
         "| steps | " + " | ".join(c for c, _ in cols) + " | s / step |",
         "|" + "---|" * (len(cols) + 2)]
    for start in range(1, last + 1, every):
        block = [s for s in steps if start <= s < start + every]
        if block:
            step_s = block_mean(metrics, block, "timing_s/step") or block_mean(metrics, block, "time/step_s")
            row = [fmt(f(block)) for _, f in cols] + [fmt(step_s, 0)]
            L.append(f"| {block[0]}-{block[-1]} | " + " | ".join(row) + " |")
    val = {k: x for k, x in metrics[last].items() if k.startswith("val") and isinstance(x, (int, float))}
    L += ["", "## Evaluation (question-only, after the last training step)", ""]
    L += [f"- `{k}`: {fmt(x)}" for k, x in sorted(val.items())] if val else ["Not run yet."]
    L += ["", "Figures: overview.png (solving), verify.png (verification chains), training.png, lengths.png, "
          "speed.png; all per-step numbers: metrics.csv; latest values and ETA: summary.txt.", ""]
    return "\n".join(L)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def read_token(path: str) -> str:
    with open(path) as f:
        return f.read().split()[-1]


def upload(api: HfApi, repo: str, run_dir: str, metrics: dict, every: int) -> None:
    name = os.path.basename(run_dir)
    project = os.path.basename(os.path.dirname(run_dir))
    out_dir = os.path.join(REPO_ROOT, "result", name)
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "plot_run.py"), run_dir], check=True,
                   stdout=subprocess.DEVNULL)
    with open(os.path.join(out_dir, "key_metrics.md"), "w") as f:
        f.write(key_metrics_md(run_dir, metrics, every))
    step = max(metrics)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=out_dir,
                      path_in_repo=f"results/{project}/{name}", allow_patterns=UPLOAD_PATTERNS,
                      commit_message=f"{name}: results at step {step}")
    print(f"[{dt.datetime.now():%H:%M:%S}] uploaded step {step} -> https://huggingface.co/datasets/{repo}/tree/main/"
          f"results/{project}/{name}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="EXP_NAME or run directory")
    p.add_argument("--repo", default="LeTue09/allocate-repair-RLVR")
    p.add_argument("--every", type=int, default=10, help="steps per block in key_metrics.md (and the upload "
                   "interval in steps when --interval-min is 0)")
    p.add_argument("--interval-min", type=float, default=10.0, help="upload every N minutes (0: every --every steps)")
    p.add_argument("--token-file", default="/cm/shared/tue09_new/key.txt")
    p.add_argument("--poll-s", type=int, default=60)
    p.add_argument("--wait-start-min", type=float, default=60.0, help="how long to wait for the run to start")
    p.add_argument("--idle-hours", type=float, default=6.0)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()

    run_dir = find_run(a.run)
    deadline = time.time() + a.wait_start_min * 60
    while not os.path.exists(config_path(run_dir)):  # the trainer writes its config when it starts
        if time.time() > deadline:
            sys.exit(f"{run_dir}: no {config_path(run_dir)} after {a.wait_start_min} min")
        time.sleep(10)
    total = int(OmegaConf.load(config_path(run_dir)).trainer.total_training_steps)
    if a.once:
        upload(HfApi(token=read_token(a.token_file)), a.repo, run_dir, read_metrics(run_dir), a.every)
        return
    when = f"every {a.interval_min:g} min" if a.interval_min > 0 else f"every {a.every} steps"
    print(f"watching {run_dir}: upload {when} to {a.repo}", flush=True)
    uploaded, last_change, last_step, last_upload_t = 0, time.time(), 0, time.time()
    while True:
        metrics = read_metrics(run_dir)
        step = max(metrics) if metrics else 0
        if step != last_step:
            last_step, last_change = step, time.time()
        finished = step >= total and any(k.startswith("val") for k in metrics.get(step, {}))
        idle = time.time() - last_change > a.idle_hours * 3600
        if a.interval_min > 0:
            due = step > uploaded and time.time() - last_upload_t >= a.interval_min * 60
        else:
            due = step // a.every > uploaded // a.every
        if step and (due or ((finished or idle) and step > uploaded)):
            try:
                upload(HfApi(token=read_token(a.token_file)), a.repo, run_dir, metrics, a.every)
                uploaded, last_upload_t = step, time.time()
            except Exception as e:  # network / hub hiccup: retry at the next poll
                print(f"[{dt.datetime.now():%H:%M:%S}] upload failed ({type(e).__name__}: {e}); retrying", flush=True)
        if (finished or idle) and uploaded == step:
            print("run finished" if finished else f"no new step for {a.idle_hours} h", "- stopping", flush=True)
            return
        time.sleep(a.poll_s)


if __name__ == "__main__":
    main()
