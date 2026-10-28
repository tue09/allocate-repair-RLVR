#!/usr/bin/env python3
"""Progress figures for runs of scripts/train_grpo_r1distill_dapo.sh and scripts/train_allocate_repair_r1distill_dapo.sh.

    python scripts/plot_run.py                 # newest run (any project under --runs-root)
    python scripts/plot_run.py <EXP_NAME>      # a run by name (or give the full run directory)
    python scripts/plot_run.py --all           # every run

Reads <run>/metrics/*.jsonl (verl's file logger, one file per launch; a resumed launch overrides the steps it
repeats), or <run>/train.log (console logger lines) for runs without metrics files, and writes result/<EXP_NAME>/:
overview.png, training.png, lengths.png, speed.png, repair.png (allocate-repair runs), validation.png (once an
evaluation has run), summary.txt and metrics.csv. Running it again replaces the files of that run. Safe to run while
training is writing.

Each panel shows the raw per-step values (faint) and a centered moving average (solid, --smooth); the number at the
end of a line is its smoothed value. summary.txt lists the raw latest values, progress and the time left.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# runs live in <runs root>/<project>/<EXP_NAME> (grpo_r1distill_dapo, allocate_repair_r1distill_dapo, ...)
DEFAULT_RUNS_ROOT = "/cm/archive/tue09_new/model_hub"
DEFAULT_OUT_ROOT = os.path.join(REPO_ROOT, "result")
OUTPUT_FILES = ("overview.png", "training.png", "lengths.png", "speed.png", "repair.png", "verify.png",
                "validation.png", "summary.txt", "metrics.csv")

# chart palette (light): categorical slots in fixed order, ink, chrome
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

def either(*keys):
    """Metric spec matching whichever of these names the run logged (plain verl vs. the allocate_repair trainer)."""
    return "re:^(" + "|".join(re.escape(k) for k in keys) + ")$"


# panel = (title, [(metric key or regex, series label)], number format). For allocate-repair runs the main panels show
# the solve phase (first attempts); repair.png has the repair phase.
CLIP = [(either("actor/pg_clipfrac", "train/solve/clipped_frac"), "upper clip (ratio > 1+high)"),
        (either("actor/pg_clipfrac_lower", "train/solve/clipped_lower_frac"), "lower clip (ratio < 1-low)")]
ENTROPY = either("actor/entropy", "train/solve/entropy")
STEP_TIME = either("timing_s/step", "time/step_s")
OVERVIEW = [
    ("Train reward (fraction of correct first attempts)", [(either("critic/score/mean", "solve/reward_mean"), None)],
     "{:.3f}"),
    ("Mean response length (tokens)", [(either("response_length/mean", "compute/solve/response_length/mean"), None)],
     "{:.0f}"),
    ("Truncated responses (hit max length)",
     [(either("response_length/clip_ratio", "compute/solve/frac_truncated"), None)], "{:.3f}"),
    ("Policy entropy", [(ENTROPY, None)], "{:.3f}"),
    ("Clipped token fraction", CLIP, "{:.4f}"),
    ("Time per training step (s)", [(STEP_TIME, None)], "{:.0f}"),
]
TRAINING = [
    ("Policy loss", [(either("actor/pg_loss", "train/total_loss"), None)], "{:.4f}"),
    ("Gradient norm", [(either("actor/grad_norm", "train/grad_norm"), None)], "{:.3f}"),
    ("PPO KL (new vs. rollout policy)", [(either("actor/ppo_kl", "train/solve/ppo_kl"), None)], "{:.5f}"),
    ("Policy entropy", [(ENTROPY, None)], "{:.3f}"),
    ("Clipped token fraction", CLIP, "{:.4f}"),
    ("Learning rate", [(either("actor/lr", "actor/optim/lr", "train/lr"), None)], "{:.2e}"),
]
LENGTHS = [
    ("Response length (tokens)",
     [(either(f"response_length/{s}", f"compute/solve/response_length/{s}"), s) for s in ("max", "mean", "min")],
     "{:.0f}"),
    ("Truncated responses (hit max length)",
     [(either("response_length/clip_ratio", "compute/solve/frac_truncated"), None)], "{:.3f}"),
    ("Mean prompt length (tokens)", [(either("prompt_length/mean", "compute/solve/prompt_length/mean"), None)],
     "{:.0f}"),
]
SPEED = [
    ("Time per training step (s)", [(STEP_TIME, None)], "{:.0f}"),
    ("Time per stage (s)", [(either("timing_s/gen", "time/gen_solve_s"), "generation"),
                            (either("timing_s/update_actor", "time/update_actor_s"), "actor update"),
                            (either("timing_s/old_log_prob", "time/old_log_prob_s"), "old log-probs")], "{:.0f}"),
    ("Throughput (tokens / s / GPU)", [(either("perf/throughput", "perf/train_tokens_per_s_per_gpu"), None)],
     "{:.0f}"),
    ("Peak GPU memory allocated (GB)", [(r"re:^(actor/)?perf/max_memory_allocated_gb$", None)], "{:.1f}"),
]
# allocate-repair only (drawn when the run logged repair/* metrics)
REPAIR = [
    ("Reward: first attempts vs. repairs", [("solve/reward_mean", "solve"), ("repair/reward_mean", "repair")],
     "{:.3f}"),
    ("Failed responses with a successful repair", [("repair/frac_groups_any_correct", None)], "{:.3f}"),
    ("Generations per step", [("compute/solve/num_generations", "solve"),
                              ("compute/repair/num_generations", "repair")], "{:.0f}"),
    ("Solve groups (fraction of questions)", [("solve/frac_groups_all_zero", "all wrong"),
                                              ("solve/frac_groups_mixed", "mixed"),
                                              ("solve/frac_groups_all_one", "all correct")], "{:.2f}"),
    ("Loss per phase", [("train/solve/loss", "solve"), ("train/repair/weighted_loss", "repair (x lambda)")],
     "{:.4f}"),
    ("Truncated responses per phase", [("compute/solve/frac_truncated", "solve"),
                                       ("compute/repair/frac_truncated", "repair")], "{:.3f}"),
    ("Repair response length (tokens)", [("compute/repair/response_length/mean", None)], "{:.0f}"),
    ("Previous answer not extractable", [("repair/frac_extraction_failures", None)], "{:.3f}"),
    ("Generation time per phase (s)", [("time/gen_solve_s", "solve"), ("time/gen_repair_s", "repair")], "{:.0f}"),
]
# verify_depth only (drawn when the run logged verify/* metrics); depth panels show rounds 1-3
DEPTHS = (1, 2, 3)
VERIFY = [
    ("Depth reward (mean over chains, -1..1)", [("verify/reward_mean", None)], "{:.3f}"),
    ("Correct judgments per round (match rate)", [(f"verify/d{d}/match_rate", f"round {d}") for d in DEPTHS], "{:.3f}"),
    ("Rounds without a verdict", [(f"verify/d{d}/frac_no_verdict", f"round {d}") for d in DEPTHS], "{:.3f}"),
    ("Round-1 accuracy by solution", [("verify/d1/acc_on_correct_solutions", "correct solutions"),
                                      ("verify/d1/acc_on_wrong_solutions", "wrong solutions")], "{:.3f}"),
    ("Solutions verified per step", [("verify/num_solutions_verified", "verified"),
                                     ("verify/num_solutions_verified_correct", "of which correct")], "{:.0f}"),
    ("Solution groups with zero advantage", [("verify/frac_groups_zero_advantage", None)], "{:.3f}"),
    ("Truncated verification rounds", [(f"verify/d{d}/frac_truncated", f"round {d}") for d in DEPTHS], "{:.3f}"),
    ("Verification round length (tokens)", [(f"verify/d{d}/mean_tokens", f"round {d}") for d in DEPTHS], "{:.0f}"),
    ("Final belief about the solution is right", [("verify/final_belief_acc", None)], "{:.3f}"),
    ("Rounds with a format violation (score -1/k)",
     [(f"verify/d{d}/frac_format_violation", f"round {d}") for d in DEPTHS], "{:.3f}"),
    ("Rounds that needed the forced verdict header", [(f"verify/d{d}/frac_forced_verdict", f"round {d}") for d in DEPTHS],
     "{:.3f}"),
]
# method=verify_depth_v2 (per-round rewards, fixed baseline): no question groups; show the round rewards instead
VERIFY_V2 = [
    ("Round reward per depth (format violation: -1/M)", [(f"verify/d{d}/score_mean", f"round {d}") for d in DEPTHS],
     "{:.3f}") if title == "Solution groups with zero advantage" else
    (("Mean round reward over chains", [("verify/reward_mean", None)], "{:.3f}")
     if title.startswith("Depth reward") else (title, wanted, fmt))
    for title, wanted, fmt in VERIFY
]
VAL_RE = re.compile(r"^val-core/(?P<ds>[^/]+)/(?P<var>[^/]+)/(?P<kind>mean|maj|best)@(?P<n>\d+)(?P<stat>/mean)?$")


# ---------------------------------------------------------------- loading
def metric_sources(run_dir):
    files = glob.glob(os.path.join(run_dir, "metrics", "*.jsonl"))
    log = os.path.join(run_dir, "train.log")
    return files or ([log] if os.path.exists(log) else [])


def find_runs(runs_root):
    """Run directories (with metrics files or a train.log) directly under runs_root or one project level below,
    oldest activity first."""
    candidates = glob.glob(os.path.join(runs_root, "*")) + glob.glob(os.path.join(runs_root, "*", "*"))
    runs = [d for d in candidates if os.path.isdir(d) and metric_sources(d)]
    return sorted(runs, key=lambda d: max(os.path.getmtime(f) for f in metric_sources(d)))


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CONSOLE_STEP_RE = re.compile(r"\bstep:(\d+) - (.*)$")
NUMPY_SCALAR_RE = re.compile(r"^np\.\w+\((.*)\)$")  # actor metrics are printed as np.float64(0.12)


def load_console_log(path):
    """{step: {key: float}} from verl's console logger lines in train.log ("step:N - key:value - key:value ...").

    Fallback for runs started without the file logger. Lines are in time order (train.log is appended on resume),
    so a repeated step keeps its latest values.
    """
    steps = {}
    with open(path, "rb") as f:
        for raw in f:
            m = CONSOLE_STEP_RE.search(ANSI_RE.sub("", raw.decode("utf-8", "replace")).rstrip())
            if not m:
                continue
            values = {}
            for item in m.group(2).split(" - "):
                key, sep, value = item.rpartition(":")
                value = NUMPY_SCALAR_RE.sub(r"\1", value.strip())
                try:
                    v = float(value)
                except ValueError:
                    continue
                if sep and math.isfinite(v):
                    values[key.strip()] = v
            steps.setdefault(int(m.group(1)), {}).update(values)
    return steps


def load_metrics(run_dir):
    """{step: {key: float}} merged over all launches (later launch wins for repeated steps)."""
    files = sorted(glob.glob(os.path.join(run_dir, "metrics", "*.jsonl")))
    log = os.path.join(run_dir, "train.log")
    if not files and os.path.exists(log):
        return dict(sorted(load_console_log(log).items()))
    steps = {}
    for path in files:
        launch = {}
        with open(path, "rb") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:  # half-written last line while training is running
                    continue
                values = {k: float(v) for k, v in row.get("data", {}).items()
                          if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}
                launch.setdefault(int(row["step"]), {}).update(values)
        steps.update(launch)
    return dict(sorted(steps.items()))


def planned_steps(run_dir):
    try:
        lines = open(os.path.join(run_dir, "command.sh")).read().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        m = re.search(r"trainer\.total_training_steps=(\d+)", line)
        if m:
            return int(m.group(1))
    return None


def is_verify_depth(metrics):
    return any(k.startswith("verify/") for row in metrics.values() for k in row)


def is_allocate_repair(metrics):
    return any(k.startswith("repair/") for row in metrics.values() for k in row)


def resolve(key, keys):
    if key.startswith("re:"):
        pattern = re.compile(key[3:])
        return next((k for k in sorted(keys) if pattern.search(k)), None)
    return key if key in keys else None


def series(metrics, key):
    xs = [s for s, row in metrics.items() if key in row]
    return xs, [metrics[s][key] for s in xs]


def moving_average(values, window):
    """Centered moving average (no lag, unlike an EMA dragged by a slow first step); shorter window at the ends."""
    if window is None:
        window = 2 * max(1, len(values) // 25) + 1   # ~4% of the run, at least 3 points
    half = window // 2
    return [sum(values[max(0, i - half):i + half + 1]) / len(values[max(0, i - half):i + half + 1])
            for i in range(len(values))]


# ---------------------------------------------------------------- drawing
def style_axes(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.set_xlabel("training step", color=MUTED, fontsize=8.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def draw_panel(ax, metrics, panel, smooth):
    title, wanted, fmt = panel
    style_axes(ax, title)
    keys = set().union(*metrics.values()) if metrics else set()
    drawn = 0
    for i, (key, label) in enumerate(wanted):
        key = resolve(key, keys)
        if key is None:
            continue
        xs, ys = series(metrics, key)
        color = SERIES[i]
        if len(xs) > 4 and smooth != 0:
            ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.25)  # raw
            ys_line = moving_average(ys, smooth)
        else:
            ys_line = ys
        ax.plot(xs, ys_line, color=color, linewidth=1.6, label=label,
                marker="o" if len(xs) <= 4 else None, markersize=4)
        # direct label at the line end: its (smoothed) value, in ink (the line carries the identity)
        ax.annotate(fmt.format(ys_line[-1]), (xs[-1], ys_line[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8.5, color=INK_2, va="center")
        drawn += 1
    if drawn == 0:
        ax.text(0.5, 0.5, "not logged yet", transform=ax.transAxes, ha="center", va="center", color=MUTED)
        ax.set_xticks([])
        ax.set_yticks([])
    elif drawn > 1:
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="best")
    ax.margins(x=0.08)


def save_grid(path, metrics, panels, suptitle, smooth, ncols=3):
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.4 * nrows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, panel in zip(axes.flat, panels):
        draw_panel(ax, metrics, panel, smooth)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def kind_order(kind):
    """mean@N, maj@N, best@N in that order."""
    return ("mean", "maj", "best").index(kind.split("@")[0])


def validation_results(metrics):
    """{step: {dataset: {"mean@N": v, "maj@N": v, "best@N": v}}} from val-core metrics."""
    out = {}
    for step, row in metrics.items():
        for key, value in row.items():
            m = VAL_RE.match(key)
            if m and (m["kind"] == "mean" or m["stat"]):
                out.setdefault(step, {}).setdefault(m["ds"], {})[f"{m['kind']}@{m['n']}"] = value
    return out


def save_validation(path, val, suptitle):
    datasets = sorted({ds for per_step in val.values() for ds in per_step})
    steps = sorted(val)
    kinds = sorted({k for per_step in val.values() for r in per_step.values() for k in r}, key=kind_order)
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.6 * len(kinds), 0.55 * len(datasets) + 1.8), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    titles = {"mean": "accuracy, mean over samples", "maj": "majority-vote accuracy", "best": "best-of-N accuracy"}
    last = val[steps[-1]]
    for ax, kind in zip(axes.flat, kinds):
        ax.set_facecolor(SURFACE)
        ax.set_title(f"{kind}: {titles[kind.split('@')[0]]}", loc="left", fontsize=10.5, color=INK, pad=8)
        vals = [last.get(ds, {}).get(kind, float("nan")) for ds in datasets]
        ax.barh(datasets, vals, color=SERIES[0], height=0.6, edgecolor=SURFACE, linewidth=2)
        for y, v in enumerate(vals):
            if math.isfinite(v):
                ax.text(v, y, f" {v:.3f}", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, 1.12)
        ax.invert_yaxis()
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.tick_params(axis="y", colors=INK_2)
    fig.suptitle(f"{suptitle} - validation at step {steps[-1]}", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- text outputs
def write_csv(path, metrics):
    keys = sorted(set().union(*metrics.values()))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + keys)
        for step, row in metrics.items():
            w.writerow([step] + [row.get(k, "") for k in keys])


def summary_text(name, run_dir, metrics, total, val):
    lines = [f"run:   {name}", f"dir:   {run_dir}"]
    done = max(metrics) if metrics else 0
    lines.append(f"steps: {done} / {total if total else '?'}" + (f"  ({100 * done / total:.0f}%)" if total else ""))
    step_time_key = resolve(STEP_TIME, set().union(*metrics.values()))
    step_times = series(metrics, step_time_key)[1] if step_time_key else []
    if step_times:
        recent = step_times[-10:]
        per_step = sum(recent) / len(recent)
        lines.append(f"time:  {per_step:.0f} s per step (mean of last {len(recent)}), "
                     f"{sum(step_times) / 3600:.1f} h of training so far")
        if total and done < total:
            lines.append(f"ETA:   ~{(total - done) * per_step / 3600:.1f} h for the remaining {total - done} steps"
                         " (+ final evaluation)")
    lines.append("")
    lines.append(f"latest values (step {done}, not smoothed):")
    shown = set()
    panels = OVERVIEW + TRAINING[:4] + (REPAIR[:3] if is_allocate_repair(metrics) else []) + (
        VERIFY[:4] if is_verify_depth(metrics) else [])
    for title, wanted, fmt in panels:
        for key, label in wanted:
            key = resolve(key, set(metrics[done]))
            if key is not None and key not in shown:
                shown.add(key)
                lines.append(f"  {(title + (' - ' + label if label else '')):58s} {fmt.format(metrics[done][key])}")
    if val:
        step = max(val)
        kinds = sorted({k for r in val[step].values() for k in r}, key=kind_order)
        lines += ["", f"validation (step {step}):", "  " + f"{'dataset':16s}" + "".join(f"{k:>10s}" for k in kinds)]
        for ds in sorted(val[step]):
            lines.append("  " + f"{ds:16s}" + "".join(f"{val[step][ds].get(k, float('nan')):10.3f}" for k in kinds))
    else:
        lines += ["", "validation: not run yet (runs once, after the last training step)"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main
def plot_run(run_dir, out_root, smooth):
    name = os.path.basename(os.path.normpath(run_dir))
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    for f in OUTPUT_FILES:  # replace this run's previous outputs (and drop figures that no longer apply)
        if os.path.exists(os.path.join(out_dir, f)):
            os.remove(os.path.join(out_dir, f))
    metrics = load_metrics(run_dir)
    total = planned_steps(run_dir)
    val = validation_results(metrics)
    if metrics:
        train_metrics = {s: r for s, r in metrics.items() if any(not k.startswith("val-") for k in r)}
        title = f"{name}  (step {max(metrics)}" + (f" / {total})" if total else ")")
        save_grid(os.path.join(out_dir, "overview.png"), train_metrics, OVERVIEW, title, smooth)
        save_grid(os.path.join(out_dir, "training.png"), train_metrics, TRAINING, f"{title} - optimization", smooth)
        save_grid(os.path.join(out_dir, "lengths.png"), train_metrics, LENGTHS, f"{title} - lengths", smooth)
        save_grid(os.path.join(out_dir, "speed.png"), train_metrics, SPEED, f"{title} - speed and memory", smooth,
                  ncols=2)
        if is_allocate_repair(metrics):
            save_grid(os.path.join(out_dir, "repair.png"), train_metrics, REPAIR, f"{title} - solve vs. repair",
                      smooth)
        if is_verify_depth(metrics):
            v2 = any("verify/round_advantage/mean" in row for row in metrics.values())
            save_grid(os.path.join(out_dir, "verify.png"), train_metrics, VERIFY_V2 if v2 else VERIFY,
                      f"{title} - verification chains", smooth)
        if val:
            save_validation(os.path.join(out_dir, "validation.png"), val, name)
        write_csv(os.path.join(out_dir, "metrics.csv"), metrics)
    text = summary_text(name, run_dir, metrics, total, val) if metrics else f"run: {name}\nno metrics logged yet\n"
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text)
    print(text + f"-> {out_dir}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="run name (EXP_NAME) or run directory; default: the newest run")
    p.add_argument("--all", action="store_true", help="plot every run under --runs-root")
    p.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--smooth", type=int, default=None,
                   help="moving-average window in steps (default: ~4%% of the run; 0 = raw only)")
    args = p.parse_args()

    if args.all:
        runs = find_runs(args.runs_root)
    elif args.run:
        if os.path.isdir(args.run):
            runs = [args.run]
        else:
            runs = [d for d in find_runs(args.runs_root) if os.path.basename(d) == args.run][-1:]
    else:
        runs = find_runs(args.runs_root)[-1:]
    if not runs:
        sys.exit(f"no run with metrics found ({args.run or args.runs_root})")
    for run in runs:
        plot_run(run, args.out_root, args.smooth)


if __name__ == "__main__":
    main()
