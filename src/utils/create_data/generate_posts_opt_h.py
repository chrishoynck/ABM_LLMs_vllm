"""Human-in-the-loop prompt-iteration tool.

You play the teacher role: the prompt for each iteration lives at
``<run>/iter_<N>/prompt.txt``. Edit it in place, run the script, read the
posts.csv that lands next to it, then create ``iter_<N+1>/prompt.txt`` (cp +
edit) and re-run. The script never moves or copies the prompt — it just
generates ``posts.csv`` and ``feedback.md`` next to the prompt you pointed at.

The iteration index N is read from the parent dir's basename (``iter_<N>``),
not from the prompt filename, so the prompt itself can stay named ``prompt.txt``
across every iteration.

Output layout::

    data/prompt_optimization_h/<run-name>/iter_<N>/
        prompt.txt    # you edit this; the script reads it
        posts.csv     # generated; overwritten on each run
        feedback.md   # scaffold written once; you fill in scores

Usage
-----
    python -m utils.create_data.generate_posts_opt_h \\
        --prompt-file data/prompt_optimization_h/qwen27_baseline/iter_0/prompt.txt \\
        --persona-phq9-file data/personas_eval_1000_phq9.csv \\
        --num_agents 7
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

import torch
from transformers import set_seed

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
    parse_iter_from_prompt_path,
    resolve_model_id,
)


FEEDBACK_TEMPLATE = """# Iter {iter_n} — {run_name}

model:    {model_id}
agents:   {num_agents}  (sample seed = {sample_seed})

## What you saw
<read posts.csv; jot down patterns, repetition, off-tone outputs, etc.>

## What to try next
<concrete edits for iter_{next_iter}/prompt.txt>

## Scores (fill in by hand)
training_score:
test_score:
"""


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-file", required=True,
                        help="Path to <run>/iter_<N>/prompt.txt. The iter index "
                             "is read from the parent dir's name.")
    parser.add_argument("--persona-phq9-file", required=True,
                        help="(persona, phq9) CSV from utils.tools.build_persona_phq9_eval.")
    parser.add_argument("--model", default="qwen27",
                        help="Short alias (qwen27, gemma12, ...) or full HF ID.")
    parser.add_argument("--num_agents", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check_point", type=int, default=10,
                        help="Posts per persona block (one block per persona).")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--neighbor-source-dir", action="append", default=None,
                        help="Repeatable. Defaults to both inter+no_inter Qwen3.5-27B trees.")
    parser.add_argument("--num-neighbors", type=int, default=5,
                        help="Max neighbour posts per inference. Set 0 to disable.")
    parser.add_argument("--neighbor-seed", type=int, default=42)
    parser.add_argument("--thinking", action="store_true",
                        help="Enable chat-template <think> blocks.")
    return parser.parse_args()


def main():
    args = _parse_args()
    if not os.path.isfile(args.prompt_file):
        sys.exit(f"prompt-file not found: {args.prompt_file}")

    iter_n = parse_iter_from_prompt_path(args.prompt_file)
    iter_dir = os.path.dirname(os.path.abspath(args.prompt_file))
    run_name = os.path.basename(os.path.dirname(iter_dir))
    print(f"[opt-h] run={run_name} iter={iter_n} -> {iter_dir}")

    set_seed(args.seed)
    personas, phq9_assignments, _ = load_persona_phq9(
        args.persona_phq9_file, n_rows=args.num_agents,
        sample_seed=iter_n,
    )
    well_being = load_well_being_zeros(args.num_agents, seed=args.seed)
    instruction, fmt_block, prompts_json = build_aligned_context(args.prompt_file)

    neighbor_roots = args.neighbor_source_dir or list(DEFAULT_NEIGHBOR_ROOTS)
    neighbor_pool = gather_neighbor_pool(neighbor_roots) if args.num_neighbors > 0 else None

    model_id = resolve_model_id(args.model)
    tok = get_tokenizer(model_id)
    # Human-loop tool: fresh randomness each invocation (engine + per-call).
    pipe = get_llm(model_id, seed=None)

    try:
        tester = TestLLMs(
            well_being=well_being, num_agents=args.num_agents, seed=args.seed,
            personas=personas, agents=None, interaction=False,
            tweet_instruction=instruction, tweet_format_block=fmt_block,
            prompts=prompts_json, thinking=args.thinking,
            phq9_assignments=phq9_assignments,
            neighbor_pool=neighbor_pool,
            num_neighbors=args.num_neighbors,
            neighbor_seed=args.neighbor_seed,
            nondeterministic_sampling=True,
        )
        tester.run_simulation(
            tokenizer=tok, pipe=pipe,
            n_rounds=args.check_point, check_point=args.check_point,
            temp=args.temp, top_p=args.top_p, model_name=model_id,
            time_info=False, mistake_dict=None,
            test_performance=False, checkpoint_every=0,
        )
        posts_csv = os.path.join(iter_dir, "posts.csv")
        tester.export_tweets_with_phq9(
            file_path=posts_csv, check_point=args.check_point,
            temp=args.temp, top_p=args.top_p,
            model_name=model_id, interaction=False,
        )
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    feedback_path = os.path.join(iter_dir, "feedback.md")
    if not os.path.exists(feedback_path):
        with open(feedback_path, "w", encoding="utf-8") as fh:
            fh.write(FEEDBACK_TEMPLATE.format(
                iter_n=iter_n,
                run_name=run_name,
                model_id=model_id,
                num_agents=args.num_agents,
                sample_seed=iter_n,
                next_iter=iter_n + 1,
            ))
        print(f"[opt-h] wrote feedback scaffold -> {feedback_path}")
    else:
        print(f"[opt-h] feedback.md already exists; leaving it alone")

    next_iter_dir = os.path.join(os.path.dirname(iter_dir), f"iter_{iter_n + 1}")
    print(f"[opt-h] done. read {posts_csv}; for the next round:\n"
          f"        mkdir -p {next_iter_dir} && "
          f"cp {args.prompt_file} {next_iter_dir}/prompt.txt && "
          f"edit + rerun.")


if __name__ == "__main__":
    main()
