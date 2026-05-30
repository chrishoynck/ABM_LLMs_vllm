"""Generate test posts using an optimizer-aligned (or otherwise tuned) instruction.

Two entry paths:

  ``--instruction-file <one.txt>``
      Run that single instruction. Typical when comparing a hand-tuned variant
      against a baseline.

  ``--instruction-dir <dir>``
      Walk the directory recursively for ``best_instruction*.txt`` files and
      run each one. Skipped files can be filtered by ``--filename-pattern``.

The rules-substitution / fixed-tail structure is preserved (the instruction is
spliced into the ``tweet_gen.system_forced`` prompt at the
``### RULES ### / ### CONSTRAINTS ###`` markers — exact mirror of how
``prompt_optimizer.py`` evaluates a student instruction).

Neighbour posts are sampled from the Qwen3.5-27B ``test_post/`` tree by
default (every ``seed_*/tweets_with_phq9.csv`` under both inter and no_inter),
mirroring ``prompt_optimizer._generate_file_path`` + ``_sample_neighbor_tweets``.

Output layout (sibling SA_prompt/ folder of the input dir):

    <parent>/SA_prompt/
        <instr_id>_<safe_model>.csv     # posts per (variant, model)
        scores.csv                       # appended row per (variant, model)

Usage
-----
    python -m utils.create_data.generate_test_data \\
        --instruction-dir data/prompt_optimization_h/prompt_variants \\
        --persona-phq9-file data/personas_eval_1000_phq9.csv \\
        --model qwen27 --num_agents 12 --seed 42
"""

from __future__ import annotations

import argparse
import csv as _csv
import gc
import glob
import os
import re
import sys

import numpy as np
import torch
from transformers import set_seed

from utils.tools.path_manager import TestPathManager
from utils.create_data.test_phq9_llms import TestLLMs

from utils.create_data.tools import (
    DEFAULT_NEIGHBOR_ROOTS,
    SEED,
    build_aligned_context,
    gather_neighbor_pool,
    get_llm,
    get_tokenizer,
    load_persona_phq9,
    load_well_being_zeros,
    resolve_model_id,
    sanitize_model_name,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--instruction-file", help="Single .txt instruction to evaluate.")
    src.add_argument("--instruction-dir",
                     help="Directory scanned (recursive) for instruction files.")
    parser.add_argument("--filename-pattern", default="best_instruction*.txt",
                        help="Glob applied within --instruction-dir.")

    parser.add_argument("--persona-phq9-file", required=True,
                        help="(persona, phq9) CSV from utils.tools.build_persona_phq9_eval.")
    parser.add_argument("--model", required=True,
                        help="Short alias (qwen27, gemma12, ...) or full HF ID.")
    parser.add_argument("--num_agents", type=int, default=12,
                        help="Persona sample size per variant.")
    parser.add_argument("--seed", type=int, default=42,
                        help="vLLM / sampling seed (NOT the per-variant agent-sampling seed; "
                             "that is derived from each instruction filename's integer).")
    parser.add_argument("--check_point", type=int, default=10,
                        help="Posts per persona block.")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--neighbor-source-dir", action="append", default=None,
                        help="Directory containing seed_*/tweets_with_phq9.csv. "
                             "Repeatable. Defaults to both inter+no_inter Qwen3.5-27B trees.")
    parser.add_argument("--num-neighbors", type=int, default=5,
                        help="Max neighbour posts per inference. Set 0 to disable.")
    parser.add_argument("--neighbor-seed", type=int, default=42,
                        help="Sub-RNG seed for per-(agent, round) neighbour reproducibility.")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable chat-template <think> blocks during generation.")
    return parser.parse_args()


def _resolve_instruction_paths(args) -> tuple[list[str], str]:
    """Return (paths, sweep_root). sweep_root is the dir we name variants relative to."""
    if args.instruction_file:
        path = args.instruction_file
        if not os.path.isfile(path):
            sys.exit(f"instruction file not found: {path}")
        return [path], os.path.dirname(os.path.abspath(path))

    sweep_root = args.instruction_dir
    paths = sorted(glob.glob(os.path.join(sweep_root, "**", args.filename_pattern),
                             recursive=True))
    if not paths:
        sys.exit(f"no instruction files matched "
                 f"{args.filename_pattern!r} under {sweep_root}")
    return paths, sweep_root


def _instr_identity(instr_path: str, sweep_root: str, fallback_idx: int) -> tuple[str, int]:
    """Return (instr_id, variant_idx) — mirrors the legacy SA naming."""
    rel = os.path.relpath(instr_path, sweep_root)
    instr_id = rel.replace(os.sep, "_")
    if instr_id.lower().endswith(".txt"):
        instr_id = instr_id[:-4]
    m = re.search(r"\d+", os.path.basename(instr_path))
    variant_idx = int(m.group(0)) if m else fallback_idx
    return instr_id, variant_idx


def _sa_output_dir(sweep_root: str, instruction_file: str | None) -> str:
    """SA_prompt/ sibling folder; same layout the legacy test_llms_sa produced."""
    if instruction_file:
        # File mode: SA_prompt/ at grandparent (sibling of the prompt's parent dir).
        anchor = os.path.dirname(os.path.dirname(os.path.abspath(instruction_file)))
    else:
        anchor = os.path.dirname(os.path.normpath(sweep_root))
    out = os.path.join(anchor, "SA_prompt")
    os.makedirs(out, exist_ok=True)
    return out


def _merge_scores_csv(path: str, new_rows: list[dict]) -> None:
    """Append/refresh rows; preserve manually-entered training_score / test_score."""
    fieldnames = ["variant_id", "model", "csv", "n_agents", "sample_seed",
                  "training_score", "test_score"]
    existing: dict[tuple[str, str], dict] = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                existing[(row.get("variant_id"), row.get("model"))] = row

    merged = dict(existing)
    for row in new_rows:
        key = (row["variant_id"], row["model"])
        prev = merged.get(key, {})
        is_test = row.pop("_is_test", False)
        if is_test:
            row["test_score"] = prev.get("test_score", "") or row["test_score"]
            row["training_score"] = prev.get("training_score", "")
        else:
            row["training_score"] = prev.get("training_score", "") or row["training_score"]
            row["test_score"] = prev.get("test_score", "")
        merged[key] = row

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged.values():
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    args = _parse_args()
    set_seed(args.seed)

    instr_paths, sweep_root = _resolve_instruction_paths(args)
    sa_dir = _sa_output_dir(sweep_root, args.instruction_file)
    print(f"[sa] {len(instr_paths)} instruction(s); output -> {sa_dir}")

    import pandas as pd
    df_pp = pd.read_csv(args.persona_phq9_file)
    if len(df_pp) < args.num_agents:
        sys.exit(f"--persona-phq9-file has {len(df_pp)} rows but "
                 f"--num_agents={args.num_agents}.")

    well_being = load_well_being_zeros(args.num_agents, seed=args.seed)

    neighbor_roots = args.neighbor_source_dir or list(DEFAULT_NEIGHBOR_ROOTS)
    neighbor_pool = gather_neighbor_pool(neighbor_roots) if args.num_neighbors > 0 else None

    model_id = resolve_model_id(args.model)
    safe_model = sanitize_model_name(model_id)
    tok = get_tokenizer(model_id)
    pipe = get_llm(model_id, seed=SEED)

    score_rows: list[dict] = []
    try:
        for sorted_idx, instr_path in enumerate(instr_paths):
            instr_id, variant_idx = _instr_identity(instr_path, sweep_root, sorted_idx)
            personas, phq9_assignments, _ = load_persona_phq9(
                args.persona_phq9_file, n_rows=args.num_agents,
                sample_seed=variant_idx,
            )

            instruction, fmt_block, prompts_json = build_aligned_context(instr_path)
            out_csv = os.path.join(sa_dir, f"{instr_id}_{safe_model}.csv")
            print(f"[sa] variant={instr_id} idx={variant_idx} -> {out_csv}")

            tester = TestLLMs(
                well_being=well_being, num_agents=args.num_agents, seed=args.seed,
                personas=personas, agents=None, interaction=False,
                tweet_instruction=instruction, tweet_format_block=fmt_block,
                prompts=prompts_json, thinking=args.thinking,
                phq9_assignments=phq9_assignments,
                neighbor_pool=neighbor_pool,
                num_neighbors=args.num_neighbors,
                neighbor_seed=args.neighbor_seed,
            )
            tester.run_simulation(
                tokenizer=tok, pipe=pipe,
                n_rounds=args.check_point, check_point=args.check_point,
                temp=args.temp, top_p=args.top_p, model_name=model_id,
                time_info=False, mistake_dict=None,
                test_performance=False, checkpoint_every=0,
            )
            tester.export_tweets_with_phq9(
                file_path=out_csv, check_point=args.check_point,
                temp=args.temp, top_p=args.top_p,
                model_name=model_id, interaction=False,
            )
            score_rows.append({
                "variant_id": instr_id, "model": model_id,
                "csv": os.path.relpath(out_csv, sa_dir),
                "n_agents": args.num_agents, "sample_seed": variant_idx,
                "training_score": "", "test_score": "",
                "_is_test": os.path.basename(instr_path).lower().startswith("test_"),
            })
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores_csv = os.path.join(sa_dir, "scores.csv")
    _merge_scores_csv(scores_csv, score_rows)
    print(f"[sa] scores -> {scores_csv} ({len(score_rows)} row(s))")
    print(f"[sa] done; {len(instr_paths)} CSV(s) under {sa_dir}/")


if __name__ == "__main__":
    main()
