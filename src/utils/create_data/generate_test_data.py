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

from utils.create_data.loaders import (
    DEFAULT_NEIGHBOR_ROOTS,
    SEED,
    build_aligned_context,
    gather_neighbor_pool,
    get_llm,
    get_tokenizer,
    load_persona_phq9,
    load_persona_phq9_stratified,
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
                        help="(persona, phq9) CSV from utils.create_data.build_persona_phq9_eval.")
    parser.add_argument("--model", required=True,
                        help="Short alias (qwen27, gemma12, ...) or full HF ID.")
    parser.add_argument("--num_agents", type=int, default=12,
                        help="Persona sample size per variant.")
    parser.add_argument("--seed", type=int, default=42,
                        help="vLLM / sampling seed (NOT the per-variant agent-sampling seed; "
                             "that is derived from each instruction filename's integer).")
    parser.add_argument("--check_point", type=int, default=10,
                        help="Posts per persona block.")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Tweet-generation sampling temperature (default 0.7, the "
                             "optimizer-student value). Varied by the decoding SA sweep.")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Tweet-generation sampling top_p (default 0.9, the "
                             "optimizer-student value). Varied by the decoding SA sweep.")

    parser.add_argument("--neighbor-source-dir", action="append", default=None,
                        help="Directory containing seed_*/tweets_with_phq9.csv. "
                             "Repeatable. Defaults to both inter+no_inter Qwen3.5-27B trees.")
    parser.add_argument("--num-neighbors", type=int, default=5,
                        help="Max neighbour posts per inference. Set 0 to disable.")
    parser.add_argument("--neighbor-seed", type=int, default=42,
                        help="Sub-RNG seed for per-(agent, round) neighbour reproducibility.")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable chat-template <think> blocks during generation.")
    parser.add_argument("--first-n", action="store_true",
                        help="Take the first --num_agents rows of --persona-phq9-file "
                             "in CSV order (deterministic). Default behavior randomises "
                             "via sample_seed=variant_idx parsed from the filename.")
    parser.add_argument("--agent-seed", type=int, default=None,
                        help="Override the agent-sampling seed (otherwise parsed from the "
                             "filename integer). Used for sensitivity analysis.")
    parser.add_argument("--nondeterministic", action="store_true",
                        help="Disable LLM engine + per-call sampling seeds — generations "
                             "vary across reruns. Pair with --agent-seed for replicate-based "
                             "variance estimation.")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Full path to the posts CSV output. Overrides the default "
                             "<sweep_root>/SA_prompt/<instr_id>_<model>.csv layout.")
    parser.add_argument("--stratify-phq9", action="store_true",
                        help="Stratified persona sampling: the PHQ-9 vector at the slot "
                             "level is fixed by --stratify-ref-seed, and --agent-seed only "
                             "picks which persona fills each slot. Enables paired (slot, "
                             "round) cross-setting comparisons for the agent axis with "
                             "PHQ-9 held constant per slot.")
    parser.add_argument("--stratify-ref-seed", type=int, default=0,
                        help="Reference seed that fixes the per-slot target PHQ-9 vector "
                             "when --stratify-phq9 is set. Must be the SAME value across "
                             "all runs in the same SA sweep.")
    parser.add_argument("--phq9-band-range", type=int, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Force every persona's PHQ-9 to a value cycled uniformly "
                             "across [LO, HI] inclusive (slot i → LO + i mod (HI-LO+1)). "
                             "For PHQ-9 conditioning: fix agent + neighbour seeds and run "
                             "once per band (e.g. 0 4 / 5 9 / 10 14 / 15 19 / 20 27) so "
                             "every persona visits every band exactly once across settings "
                             "AND every band-setting has a distribution over its scores.")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="If >0, generate in chunks of this many personas, appending each "
                             "chunk's blocks to --output-csv as it finishes (live-watchable, "
                             "crash-resumable: a re-run skips agent_ids already in the file). "
                             "Requires --output-csv. Default 0 = single-shot write at the end.")
    return parser.parse_args()


def _existing_agent_ids(path: str) -> set[int]:
    """Agent IDs already written to `path` (empty if the file does not exist)."""
    ids: set[int] = set()
    if os.path.isfile(path):
        with open(path, newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                try:
                    ids.add(int(row["agent_id"]))
                except (TypeError, ValueError, KeyError):
                    pass
    return ids


def _append_blocks(path: str, agents, global_ids, interaction: bool = False) -> None:
    """Append each agent's (step, phq9, tweet) rows to `path` under its GLOBAL id.

    Mirrors TestLLMs.export_tweets_with_phq9 row-for-row, but appends (so chunks
    accumulate) and stamps agent_id from `global_ids` so ids stay unique across
    chunks. Writes the header only when the file is new.
    """
    write_header = not os.path.isfile(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = _csv.writer(fh)
        if write_header:
            writer.writerow(["agent_id", "persona", "age", "step", "phq9", "tweet", "interaction"])
        for agent, gid in zip(agents, global_ids):
            phq_series = list(agent.all_phq9_sumscores)
            tweets = list(agent.tweethistory)
            for idx, value in enumerate(phq_series):
                tweet = tweets[idx] if idx < len(tweets) else ""
                tweet = tweet.replace("\n", " ").replace("\r", " ").strip()
                writer.writerow([gid, agent.persona, getattr(agent, "age", None),
                                 idx, value, tweet, interaction])


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
    pipe = get_llm(model_id, seed=None if args.nondeterministic else SEED)

    if args.output_csv and len(instr_paths) > 1:
        sys.exit("--output-csv is only valid with --instruction-file (single instruction); "
                 "got --instruction-dir with multiple matches.")

    score_rows: list[dict] = []
    try:
        for sorted_idx, instr_path in enumerate(instr_paths):
            instr_id, variant_idx = _instr_identity(instr_path, sweep_root, sorted_idx)
            if args.agent_seed is not None:
                sample_seed = args.agent_seed
            elif args.first_n:
                sample_seed = None
            else:
                sample_seed = variant_idx
            if args.stratify_phq9:
                if sample_seed is None:
                    sys.exit("--stratify-phq9 requires --agent-seed (or a filename "
                             "integer); cannot stratify with --first-n.")
                personas, phq9_assignments, _ = load_persona_phq9_stratified(
                    args.persona_phq9_file, n_rows=args.num_agents,
                    sample_seed=sample_seed,
                    reference_seed=args.stratify_ref_seed,
                )
            else:
                personas, phq9_assignments, _ = load_persona_phq9(
                    args.persona_phq9_file, n_rows=args.num_agents,
                    sample_seed=sample_seed,
                )

            if args.phq9_band_range is not None:
                lo, hi = args.phq9_band_range
                if hi < lo:
                    sys.exit(f"--phq9-band-range: HI ({hi}) < LO ({lo})")
                width = hi - lo + 1
                phq9_assignments = [lo + (i % width) for i in range(len(phq9_assignments))]
                from collections import Counter
                ct = Counter(phq9_assignments)
                print(f"[sa] --phq9-band-range [{lo},{hi}]: PHQ-9 spread = "
                      f"{dict(sorted(ct.items()))}")

            instruction, fmt_block, prompts_json = build_aligned_context(instr_path)
            if args.output_csv:
                out_csv = args.output_csv
                os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
            else:
                out_csv = os.path.join(sa_dir, f"{instr_id}_{safe_model}.csv")
            print(f"[sa] variant={instr_id} agent_seed={sample_seed} "
                  f"neighbor_seed={args.neighbor_seed} "
                  f"nondet={args.nondeterministic} -> {out_csv}")

            def _build_and_run(personas_slice, phq9_slice, wb_slice, global_ids):
                """Construct a TestLLMs over a persona slice, renumber to GLOBAL ids,
                generate, and return the agents (in slice order)."""
                t = TestLLMs(
                    well_being=wb_slice, num_agents=len(personas_slice), seed=args.seed,
                    personas=personas_slice, agents=None, interaction=False,
                    tweet_instruction=instruction, tweet_format_block=fmt_block,
                    prompts=prompts_json, thinking=args.thinking,
                    phq9_assignments=phq9_slice,
                    neighbor_pool=neighbor_pool,
                    num_neighbors=args.num_neighbors,
                    neighbor_seed=args.neighbor_seed,
                    nondeterministic_sampling=args.nondeterministic,
                    gen_temp=args.temp, gen_top_p=args.top_p,
                )
                # phq9_assignments keeps persona order (no permutation), so all_agents[i]
                # corresponds to personas_slice[i]; renumber ID -> global so neighbour
                # seeding (SeedSequence[..,int(agent.ID),..]) and output ids match a
                # single-shot run exactly. TestLLMs keys phq9_sequences/phq9_indices by
                # the ORIGINAL (local) ID, so remap those dict keys in lockstep — else
                # the PHQ-9 update step hits KeyError on chunks past the first.
                old_ids = [ag.ID for ag in t.all_agents]
                t.phq9_sequences = {gid: t.phq9_sequences[old] for old, gid in zip(old_ids, global_ids)}
                t.phq9_indices = {gid: t.phq9_indices[old] for old, gid in zip(old_ids, global_ids)}
                for ag, gid in zip(t.all_agents, global_ids):
                    ag.ID = gid
                t.run_simulation(
                    tokenizer=tok, pipe=pipe,
                    n_rounds=args.check_point, check_point=args.check_point,
                    temp=args.temp, top_p=args.top_p, model_name=model_id,
                    time_info=False, mistake_dict=None,
                    test_performance=False, checkpoint_every=0,
                )
                return t.all_agents

            if args.chunk_size and args.chunk_size > 0:
                if not args.output_csv:
                    sys.exit("--chunk-size requires --output-csv.")
                done_ids = _existing_agent_ids(out_csv)
                if done_ids:
                    print(f"[sa] resume: {len(done_ids)} agent_id(s) already in {out_csv}; "
                          f"generating only the rest.")
                n_total = len(personas)
                for start in range(0, n_total, args.chunk_size):
                    end = min(start + args.chunk_size, n_total)
                    ids = [i for i in range(start, end) if i not in done_ids]
                    if not ids:
                        print(f"[sa] chunk {start}:{end} already complete; skipping.")
                        continue
                    p_slice = [personas[i] for i in ids]
                    q_slice = [phq9_assignments[i] for i in ids]
                    wb_slice = [well_being[i % len(well_being)] for i in ids]
                    print(f"[sa] chunk {start}:{end} → generating {len(ids)} block(s)")
                    agents = _build_and_run(p_slice, q_slice, wb_slice, ids)
                    _append_blocks(out_csv, agents, ids, interaction=False)
                    print(f"[sa] chunk {start}:{end} appended → {out_csv}")
            else:
                # Single-shot: overwrite any existing file (original "w" semantics).
                if os.path.isfile(out_csv):
                    os.remove(out_csv)
                agents = _build_and_run(personas, phq9_assignments, well_being,
                                        list(range(len(personas))))
                _append_blocks(out_csv, agents, list(range(len(personas))),
                               interaction=False)
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
