"""Generate a tweets-with-PHQ-9 dataset under the *old* framework.

Old framework = the agent-driven prompt builder in :class:`Agent.step_llm_tweet`
(no optimizer-aligned instruction, no externally-sampled neighbour pool):

  * ``--interaction``   build a network and sample neighbours from it.
  * (no flag)           no network, no neighbours — each agent posts in isolation.

Neighbour sampling from an external tweets_with_phq9 pool (``--rand_interaction``
in the legacy CLI) lives in :mod:`utils.create_data.generate_test_data` now,
since it belongs with the optimizer-aligned pipeline.

The output schema (``tweets_with_phq9.csv``) is identical to the legacy
``--test_llms`` output, so downstream consumers (prompt_optimizer.py,
BERT trainer, ...) need no changes.

Usage
-----
    python -m utils.create_data.generate_synthetic_dataset r \\
        --model llama8 --num_agents 12 --seeds 42 43 \\
        --check_point 10
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

import torch
from transformers import set_seed

import utils.tools.load_personas as lp
from classes.network import RandomNetwork, SocialDistanceAttachment
from utils.tools.format_config import FC
from utils.tools.path_manager import TestPathManager
from utils.create_data.test_phq9_llms import TestLLMs

from utils.create_data.loaders import (
    DEFAULT_MODELS,
    SEED,
    get_llm,
    get_tokenizer,
    load_persona_phq9,
    load_well_being_zeros,
    resolve_model_id,
)


def _build_network(args, personas, well_being, happy_personas=None):
    """Same network-builder as llama_activate, duplicated to avoid the heavy import."""
    if args.net in {"sda", "sdc"}:
        return SocialDistanceAttachment(
            alpha=args.alpha, degree=args.degree, dim=args.dim,
            num_agents=args.num_agents, seed=args.seed, plot=False,
            well_being=well_being, personas=personas,
            sdc=(args.net == "sdc"),
            happy_personas=happy_personas, directed=args.directed,
        )
    return RandomNetwork(
        p=args.p, k=args.k, num_agents=args.num_agents, seed=args.seed,
        personas=personas, well_being=well_being,
        happy_personas=happy_personas, directed=args.directed,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("net", nargs="?", choices=["sf", "r", "sda", "sdc"], default="r",
                        help="Network type (only used when --interaction is set).")
    parser.add_argument("--model", type=str, required=True,
                        help="Short alias (qwen27, gemma12, llama8, ...) or full HF ID. "
                             "Pass 'all' to iterate over every default model.")
    parser.add_argument("--num_agents", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--interaction", action="store_true",
                        help="Build a network and sample neighbours from it.")
    parser.add_argument("--directed", action="store_true")

    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--check_point", type=int, default=10,
                        help="Posts per PHQ-9 block.")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Total rounds. Defaults to check_point * 28 (every PHQ-9 score "
                             "seen once per agent), or check_point if --persona-phq9-file is set.")

    # Network shape (only consulted when --interaction).
    parser.add_argument("--p", type=float, default=0.5, help="Edge probability (random network)")
    parser.add_argument("--k", type=int, default=0, help="Regular degree (Watts–Strogatz if >0)")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--happy", action="store_true")

    # Persona source: at most one.
    parser.add_argument("--persona-pool", default="",
                        help="Shared eval-persona CSV (load_or_build_persona_pool). "
                             "Mutually exclusive with --persona-phq9-file.")
    parser.add_argument("--persona-phq9-file", default=None,
                        help="(persona, phq9) CSV from utils.create_data.build_persona_phq9_eval. "
                             "One block per persona at the assigned score; rounds default "
                             "to --check_point.")
    return parser.parse_args()


def _resolve_personas(args, seed):
    """Return (personas, phq9_assignments, rounds) for the requested persona source."""
    if args.persona_phq9_file:
        personas, phq9_assignments, _ = load_persona_phq9(
            args.persona_phq9_file, n_rows=args.num_agents,
        )
        rounds = args.rounds if args.rounds is not None else args.check_point
        print(f"[persona-phq9] one block per persona; rounds={rounds} "
              f"(from {args.persona_phq9_file})")
        return personas, phq9_assignments, rounds

    if args.persona_pool:
        personas = lp.load_or_build_persona_pool(
            n_needed=args.num_agents, pool_path=args.persona_pool,
        )
    else:
        personas = lp.load_personas_from_file(
            "data/personas_short_10k.csv", args.num_agents, seed=seed,
        )
    rounds = args.rounds if args.rounds is not None else args.check_point * 28
    return personas, None, rounds


def _run_one(args, model_id: str):
    """Run one model across every --seeds value."""
    tok = get_tokenizer(model_id)
    pipe = get_llm(model_id, seed=SEED)
    try:
        for seed in args.seeds:
            args.seed = seed
            set_seed(seed)
            print(f"\n{'='*50}\nModel={model_id}  seed={seed}\n{'='*50}")

            personas, phq9_assignments, rounds = _resolve_personas(args, seed)
            well_being = load_well_being_zeros(args.num_agents, seed=seed)

            if args.happy:
                happy = lp.load_happy_personas(
                    "data/happy_persona.csv", personass_to_load=1, seed=seed,
                )
            else:
                happy = None

            if args.interaction:
                network = _build_network(args, personas=personas,
                                         well_being=well_being,
                                         happy_personas=happy)
                agents = network.all_agents
            else:
                agents = None

            tester = TestLLMs(
                well_being=well_being,
                num_agents=args.num_agents,
                seed=seed,
                personas=personas,
                agents=agents,
                interaction=args.interaction,
                phq9_assignments=phq9_assignments,
            )

            tpm = TestPathManager(model_id, args.temp, args.top_p,
                                  args.check_point, seed=seed,
                                  interaction=args.interaction)

            os.makedirs(f"data/test{FC.DIR_SUFFIX}/", exist_ok=True)
            tester.run_simulation(
                tokenizer=tok, pipe=pipe,
                n_rounds=rounds, check_point=args.check_point,
                temp=args.temp, top_p=args.top_p,
                model_name=model_id,
                time_info=True, mistake_dict=None,
                test_performance=False, checkpoint_every=args.check_point,
            )
            tester.export_tweets_with_phq9(
                file_path=str(tpm.get_tweets_path()),
                check_point=args.check_point,
                temp=args.temp, top_p=args.top_p,
                model_name=model_id, interaction=args.interaction,
            )
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    args = _parse_args()
    if args.persona_pool and args.persona_phq9_file:
        sys.exit("--persona-pool and --persona-phq9-file are mutually exclusive.")

    if args.model.strip().lower() == "all":
        models_to_run = DEFAULT_MODELS
    else:
        models_to_run = [resolve_model_id(args.model)]

    for mid in models_to_run:
        _run_one(args, mid)
    print("\n[done] generate_synthetic_dataset finished.")


if __name__ == "__main__":
    main()
