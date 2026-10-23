#!/usr/bin/env python3
"""Download (pinned revisions) and prepare the training / evaluation parquet files.

Outputs (default ``data/processed``)::

    train.parquet             DAPO-Math-17k, de-duplicated, minus held-out and decontaminated questions
    val_dapo_heldout.parquet  held-out DAPO questions (disjoint from train by normalized question text)
    eval_aime24.parquet       AIME 2024 (30)
    eval_aime25.parquet       AIME 2025 (30)
    eval_math500.parquet      MATH-500 (500)
    smoke_train.parquet       small subset of train for the smoke test
    smoke_val.parquet         small subset of the eval sets for the smoke test
    manifest.json             revisions, file hashes, row counts, duplicate handling, overlap statistics

Every prompt uses the DAPO answer format ("... The last line of your response should be of the form
Answer: $Answer ..."), so the same ``math_dapo`` checker is used for training and evaluation.

Only pandas, pyarrow and huggingface_hub are required (transformers is optional, for token lengths).
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCES = {
    "dapo": {
        "repo_id": "BytedTsinghua-SIA/DAPO-Math-17k",
        "revision": "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
        "filename": "data/dapo-math-17k.parquet",
    },
    "aime24": {
        "repo_id": "BytedTsinghua-SIA/AIME-2024",
        "revision": "aa49075e24ad594b79fdf0bdcefa735c2181be67",
        "filename": "data/aime-2024.parquet",
    },
    "aime25": {
        "repo_id": "MathArena/aime_2025",
        "revision": "c94da77eb22bbd6439e62a323bec18493a421302",
        "filename": "data/train-00000-of-00001.parquet",
    },
    "math500": {
        "repo_id": "HuggingFaceH4/MATH-500",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "filename": "test.jsonl",
    },
}

# The DAPO-Math-17k prompt template (verified to wrap every row of the pinned revision).
DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'
REWARD_STYLE = "rule-lighteval/MATH_v2"


def load_math_dapo():
    """Load verl's math_dapo module by file path (it only depends on ``re``), avoiding the heavy verl import."""
    path = os.path.join(REPO_ROOT, "third_party", "verl", "verl", "utils", "reward_score", "math_dapo.py")
    spec = importlib.util.spec_from_file_location("_math_dapo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_path(raw_dir: str, key: str) -> str:
    src = SOURCES[key]
    return os.path.join(raw_dir, src["repo_id"].replace("/", "__"), src["filename"])


def download_raw(raw_dir: str) -> None:
    from huggingface_hub import hf_hub_download

    for key, src in SOURCES.items():
        if os.path.exists(raw_path(raw_dir, key)):
            continue
        print(f"[download] {src['repo_id']}@{src['revision']} :: {src['filename']}")
        hf_hub_download(
            repo_id=src["repo_id"],
            repo_type="dataset",
            revision=src["revision"],
            filename=src["filename"],
            local_dir=os.path.join(raw_dir, src["repo_id"].replace("/", "__")),
        )


def normalize_question(text: str) -> str:
    """Question identity key: NFKC, lower-case, all whitespace removed.

    Deliberately aggressive so that near-identical copies (spacing / case variants) collapse to one key,
    which makes the train / held-out split conservatively disjoint.
    """
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", "", text.lower())


def strip_dapo_template(prompt_text: str) -> str:
    assert prompt_text.startswith(DAPO_PREFIX) and prompt_text.endswith(DAPO_SUFFIX), prompt_text[:200]
    return prompt_text[len(DAPO_PREFIX) : -len(DAPO_SUFFIX)]


def make_row(
    data_source, question, ground_truth, question_id, split, source_dataset, aux_math_verify, prompt_text=None
):
    if prompt_text is None:
        prompt_text = DAPO_PREFIX + question + DAPO_SUFFIX
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "MATH",
        # Ground truth lives only in verifier metadata; it never enters a prompt.
        "reward_model": {"ground_truth": str(ground_truth), "style": REWARD_STYLE},
        "extra_info": {
            "index": str(question_id),
            "question_id": str(question_id),
            "question": question,
            "split": split,
            "source_dataset": source_dataset,
            "aux_math_verify": bool(aux_math_verify),
        },
    }


def dedupe_dapo(raw_file: str, math_dapo, stats: dict) -> pd.DataFrame:
    df = pd.read_parquet(raw_file)
    stats["raw_rows"] = int(len(df))
    df["prompt_text"] = df["prompt"].map(lambda m: m[0]["content"])
    df["qid"] = df["extra_info"].map(lambda e: str(e["index"]))
    df["gt"] = df["reward_model"].map(lambda r: str(r["ground_truth"]))

    repeats = df.groupby("qid").size()
    stats["unique_question_ids"] = int(repeats.shape[0])
    stats["rows_per_question_id"] = {str(k): int(v) for k, v in repeats.value_counts().items()}
    per_id = df.groupby("qid").agg(n_text=("prompt_text", "nunique"), n_gt=("gt", "nunique"))
    stats["question_ids_with_inconsistent_rows"] = int(((per_id.n_text > 1) | (per_id.n_gt > 1)).sum())

    # 1) collapse the exact per-id repetitions
    uniq = df.drop_duplicates("qid")[["qid", "prompt_text", "gt"]].copy()
    uniq["question"] = uniq["prompt_text"].map(strip_dapo_template)
    stats["unique_exact_question_texts"] = int(uniq["question"].nunique())
    uniq["qkey"] = uniq["question"].map(normalize_question)
    uniq["gt_norm"] = uniq["gt"].map(lambda g: math_dapo.normalize_final_answer(g))
    stats["unique_normalized_questions"] = int(uniq["qkey"].nunique())

    # 2) merge different ids that share the normalized question text
    groups = uniq.groupby("qkey")
    multi = groups.size()
    stats["normalized_groups_with_multiple_ids"] = int((multi > 1).sum())
    n_gt = groups["gt_norm"].nunique()
    conflicting_keys = set(n_gt[n_gt > 1].index)
    stats["normalized_groups_with_conflicting_ground_truth"] = len(conflicting_keys)
    stats["question_ids_dropped_for_conflicting_ground_truth"] = int(uniq["qkey"].isin(conflicting_keys).sum())

    kept = uniq[~uniq["qkey"].isin(conflicting_keys)].sort_values("qid")
    stats["question_ids_merged_as_duplicates"] = int(len(kept) - kept["qkey"].nunique())
    kept = kept.drop_duplicates("qkey", keep="first").reset_index(drop=True)
    stats["unique_questions_after_dedup"] = int(len(kept))
    return kept


def load_eval_sets(raw_dir: str) -> dict:
    out = {}
    a = pd.read_parquet(raw_path(raw_dir, "aime24"))
    a["qid"] = a["extra_info"].map(lambda e: e["index"])
    a = a.drop_duplicates("qid").sort_values("qid")
    out["aime24"] = [
        dict(
            question=r.extra_info["raw_problem"],
            gt=r.reward_model["ground_truth"],
            qid=f"aime24-{r.qid}",
            prompt_text=r.prompt[0]["content"],
        )
        for r in a.itertuples()
    ]
    b = pd.read_parquet(raw_path(raw_dir, "aime25")).sort_values("problem_idx")
    out["aime25"] = [dict(question=r.problem, gt=str(r.answer), qid=f"aime25-{r.problem_idx}") for r in b.itertuples()]
    m = pd.read_json(raw_path(raw_dir, "math500"), lines=True)
    out["math500"] = [dict(question=r.problem, gt=str(r.answer), qid=f"math500-{r.unique_id}") for r in m.itertuples()]
    return out


def find_overlaps(train_keys: list, eval_keys: list, min_len: int = 40) -> set:
    """Indices of train questions that equal, contain, or are contained in an eval question (normalized)."""
    eval_set = set(eval_keys)
    long_eval = [k for k in eval_keys if len(k) >= min_len]
    hits = set()
    for i, k in enumerate(train_keys):
        if k in eval_set:
            hits.add(i)
            continue
        if len(k) >= min_len and any((e in k) or (k in e) for e in long_eval):
            hits.add(i)
    return hits


def prompt_token_lengths(tokenizer_path: str, prompts: list) -> np.ndarray:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]
    return np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts])


def length_summary(lengths: np.ndarray) -> dict:
    return {
        "min": int(lengths.min()),
        "mean": float(lengths.mean()),
        "p50": float(np.percentile(lengths, 50)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", default=os.path.join(REPO_ROOT, "data", "raw"))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "data", "processed"))
    parser.add_argument("--num_heldout", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_train", type=int, default=64)
    parser.add_argument("--tokenizer", default=None, help="optional HF tokenizer path for prompt-length stats/filter")
    parser.add_argument("--max_prompt_tokens", type=int, default=1024, help="drop train questions above this")
    parser.add_argument("--no_download", action="store_true")
    args = parser.parse_args()

    if not args.no_download:
        download_raw(args.raw_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    math_dapo = load_math_dapo()
    rng = np.random.RandomState(args.seed)

    stats = {}
    dapo = dedupe_dapo(raw_path(args.raw_dir, "dapo"), math_dapo, stats)
    evals = load_eval_sets(args.raw_dir)

    # Decontaminate the DAPO pool against every external benchmark.
    keys = dapo["qkey"].tolist()
    contamination = {}
    drop = set()
    for name, items in evals.items():
        hits = find_overlaps(keys, [normalize_question(x["question"]) for x in items])
        contamination[name] = len(hits)
        drop |= hits
    stats["dapo_questions_overlapping_eval_benchmarks"] = contamination
    stats["dapo_questions_removed_for_overlap"] = len(drop)
    dapo = dapo.drop(index=sorted(drop)).reset_index(drop=True)

    # Optional prompt-length filter for the phase-1 prompt budget (keeps every question unchanged or drops it).
    length_stats = None
    if args.tokenizer:
        lengths = prompt_token_lengths(args.tokenizer, dapo["prompt_text"].tolist())
        too_long = lengths > args.max_prompt_tokens
        stats["dapo_questions_removed_over_max_prompt_tokens"] = int(too_long.sum())
        length_stats = {"tokenizer": args.tokenizer, "dapo_prompt_tokens": length_summary(lengths)}
        dapo = dapo[~too_long].reset_index(drop=True)

    # Held-out split at the level of normalized question groups (each group is one row after dedup).
    perm = rng.permutation(len(dapo))
    heldout_idx = set(perm[: args.num_heldout].tolist())
    heldout = dapo.iloc[sorted(heldout_idx)]
    train = dapo.iloc[sorted(set(range(len(dapo))) - heldout_idx)]
    assert not (set(train["qkey"]) & set(heldout["qkey"])), "train / held-out overlap"

    def dapo_rows(frame, data_source, split, aux):
        return [
            make_row(
                data_source, r.question, r.gt, r.qid, split, SOURCES["dapo"]["repo_id"], aux, prompt_text=r.prompt_text
            )
            for r in frame.itertuples()
        ]

    outputs = {
        "train": pd.DataFrame(dapo_rows(train, "math_dapo", "train", False)),
        "val_dapo_heldout": pd.DataFrame(dapo_rows(heldout, "dapo_heldout", "val", True)),
    }
    for name, items in evals.items():
        outputs[f"eval_{name}"] = pd.DataFrame(
            [
                make_row(
                    name, x["question"], x["gt"], x["qid"], "eval", SOURCES[name]["repo_id"], True, x.get("prompt_text")
                )
                for x in items
            ]
        )
    outputs["smoke_train"] = outputs["train"].sample(n=args.smoke_train, random_state=args.seed).reset_index(drop=True)
    outputs["smoke_val"] = pd.concat(
        [
            outputs["val_dapo_heldout"].sample(n=16, random_state=args.seed),
            outputs["eval_math500"].sample(n=8, random_state=args.seed),
        ]
    ).reset_index(drop=True)

    for name, frame in outputs.items():
        frame.to_parquet(os.path.join(args.out_dir, f"{name}.parquet"), index=False)
        print(f"[write] {name}.parquet  rows={len(frame)}")

    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sources": {k: {**v, "sha256": sha256_file(raw_path(args.raw_dir, k))} for k, v in SOURCES.items()},
        "dedup_policy": (
            "collapse repeated rows by extra_info.index; merge ids whose question text matches after "
            "NFKC + lower-case + whitespace removal; drop merged groups whose ground truths disagree after "
            "math_dapo.normalize_final_answer; keep the smallest id per group"
        ),
        "split_policy": (
            f"random held-out of {args.num_heldout} normalized-question groups (seed={args.seed}); "
            "train and held-out share no normalized question key"
        ),
        "decontamination_policy": (
            "remove DAPO questions equal to / containing / contained in an eval question "
            "(normalized, >=40 chars for containment)"
        ),
        "dapo_stats": stats,
        "prompt_lengths": length_stats,
        "row_counts": {name: int(len(frame)) for name, frame in outputs.items()},
        "unique_question_counts": {
            name: int(frame["extra_info"].map(lambda e: normalize_question(e["question"])).nunique())
            for name, frame in outputs.items()
        },
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["dapo_stats"], indent=2))
    print(json.dumps(manifest["row_counts"], indent=2))


if __name__ == "__main__":
    main()
