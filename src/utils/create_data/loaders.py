"""Shared loaders used by every script under :mod:`utils.create_data`.

Includes:
  * MODEL_ALIASES / DEFAULT_MODELS used by the new CLIs.
  * Thin wrappers around tokenizer + vLLM construction (one place to keep the
    sampling / dtype defaults in sync).
  * Loaders for the eval persona-PHQ-9 file, zero-initialised well-being dicts,
    and the optimizer-aligned (instruction, format-block, prompts-json) tuple.
  * Neighbour-pool gatherer that mirrors prompt_optimizer._generate_file_path
    + _sample_neighbor_tweets: walk every ``seed_*/tweets_with_phq9.csv`` under
    a base dir and flatten into one (agent_id, post) pool.
"""

from __future__ import annotations

import csv
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM

import utils.tools.load_personas as lp
from utils.tools.format_config import FC
from utils.create_data.test_phq9_llms import (
    derive_tweet_format_block,
    load_instruction_file,
)

# Re-export for callers that import from this module.
__all__ = [
    "MODEL_ALIASES",
    "DEFAULT_MODELS",
    "DEFAULT_NEIGHBOR_ROOTS",
    "SEED",
    "resolve_model_id",
    "sanitize_model_name",
    "get_tokenizer",
    "get_llm",
    "load_persona_phq9",
    "load_persona_phq9_stratified",
    "load_well_being_zeros",
    "build_aligned_context",
    "gather_neighbor_pool",
    "parse_iter_from_prompt_path",
    "derive_tweet_format_block",
    "load_instruction_file",
]


SEED = 1234

MODEL_ALIASES = {
    "qwen397": "Qwen/Qwen3.5-397B-A17B",
    "qwen27": "Qwen/Qwen3.5-27B",
    "gemma12": "google/gemma-3-12b-it",
    "llama8": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral7": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama70": "meta-llama/Llama-3.3-70B-Instruct",
    "hermes70": "NousResearch/Hermes-3-Llama-3.1-70B",
    "dolphin72": "cognitivecomputations/dolphin-2.9.2-qwen2-72b",
}

DEFAULT_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "google/gemma-3-12b-it",
    "meta-llama/Llama-4-8B-Instruct",
    "mistralai/Mistral-Small-4-119B-2603",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen3.5-27B",
]

# Both interaction conditions under the Qwen3.5-27B output tree feed the
# neighbour pool by default (matches prompt_optimizer.__main__).
DEFAULT_NEIGHBOR_ROOTS = (
    "data/test_post/Qwen_Qwen3.5-27B/temp_0.8_top_p_0.6_cp_10_inter",
    "data/test_post/Qwen_Qwen3.5-27B/temp_0.8_top_p_0.6_cp_10_no_inter",
)


def resolve_model_id(alias_or_id: str) -> str:
    """Map a short alias (e.g. ``qwen27``) to a full HuggingFace ID; pass through unknowns."""
    if not alias_or_id:
        return alias_or_id
    return MODEL_ALIASES.get(alias_or_id.strip().lower(), alias_or_id.strip())


def sanitize_model_name(model_id: str) -> str:
    """Make a model ID safe for filesystem use (``org/name`` -> ``org_name``)."""
    return model_id.replace("/", "_").replace("\\", "_")


def get_tokenizer(model_id: str):
    """Load a left-padded tokenizer; matches the simulation's tokenizer setup."""
    cache_dir = os.environ.get("TRANSFORMERS_CACHE", None)
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def get_llm(model_id: str, seed: int | None = SEED, max_model_len: int = 8192) -> LLM:
    """Build the vLLM engine with the same defaults llama_activate.get_llm uses.

    ``seed=None`` skips the explicit seed kwarg, letting vLLM pick its own —
    use this when you want fresh randomness across invocations (the human-loop
    tool does this).
    """
    gpus_count = torch.cuda.device_count()
    print(f"Loading vLLM model: {model_id} on {gpus_count} GPU(s)...")
    kwargs = dict(
        model=model_id,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=gpus_count,
        gpu_memory_utilization=0.90,
        max_model_len=max_model_len,
    )
    if seed is not None:
        kwargs["seed"] = seed
    if "qwen3.5" in model_id.lower():
        kwargs["limit_mm_per_prompt"] = {"image": 0}
        kwargs["enable_prefix_caching"] = True
    return LLM(**kwargs)


def load_persona_phq9(path: str, n_rows: int | None = None,
                      sample_seed: int | None = None,
                      ) -> tuple[list, list[int], list[int]]:
    """Read a (persona, phq9) CSV.

    Returns ``(personas, phq9_assignments, sample_idx)``. When ``sample_seed`` is
    given, ``n_rows`` rows are sampled with rng(sample_seed) and ``sample_idx``
    is the sorted positional index of the kept rows; otherwise rows are taken in
    file order and ``sample_idx`` is ``list(range(n_rows))``.

    A persona-only pool (no ``phq9`` column, e.g. data/personas_short_10k.csv) is
    accepted directly: PHQ-9 is filled balanced (round-robin 0..27) so the pool
    can be used as-is for calibration data. Pair with
    ``generate_test_data --phq9-band-range 0 27`` for exactly balanced coverage.
    """
    df = pd.read_csv(path)
    if "phq9" not in df.columns:
        df = df.copy()
        df["phq9"] = [i % 28 for i in range(len(df))]
    if n_rows is None:
        n_rows = len(df)
    if len(df) < n_rows:
        raise ValueError(
            f"--persona-phq9-file has {len(df)} rows but {n_rows} requested."
        )

    if sample_seed is not None:
        rng = np.random.default_rng(sample_seed)
        sample_idx = sorted(rng.choice(len(df), size=n_rows, replace=False))
        personas = df["persona"].iloc[sample_idx].tolist()
        phq9 = df["phq9"].iloc[sample_idx].astype(int).tolist()
    else:
        sample_idx = list(range(n_rows))
        personas = df["persona"].head(n_rows).tolist()
        phq9 = df["phq9"].head(n_rows).astype(int).tolist()
    return personas, phq9, sample_idx


def load_persona_phq9_stratified(path: str, n_rows: int, sample_seed: int,
                                 reference_seed: int = 0,
                                 ) -> tuple[list, list[int], list[int]]:
    """Sample personas in a PHQ-9-balanced way that yields the SAME per-slot
    PHQ-9 vector across different ``sample_seed`` values.

    Mechanism: a fixed "reference" draw (via ``reference_seed``) determines the
    PHQ-9 score that occupies each slot ``i`` for all subsequent calls; each
    actual call then draws a fresh persona whose ``phq9`` matches that target,
    using ``sample_seed`` to vary *which* persona is picked.

    Effect: slot 0 always has the same PHQ-9 score (and thus the same severity
    band) across runs that vary only ``sample_seed`` — only the persona text
    differs. This lets the agent-axis SA do neighbour-style paired (slot,
    round) comparisons with PHQ-9 and neighbour input held constant, isolating
    the effect of swapping the persona.

    Returns ``(personas, phq9_assignments, sample_idx)`` aligned by slot.
    Raises if any reference PHQ-9 score has fewer than ``n_rows`` candidates
    (since we need at least one per slot, and ideally distinct ones per call).
    """
    df = pd.read_csv(path)
    if len(df) < n_rows:
        raise ValueError(
            f"--persona-phq9-file has {len(df)} rows but {n_rows} requested."
        )

    # Reference draw → fixed per-slot PHQ-9 target vector.
    ref_rng = np.random.default_rng(reference_seed)
    ref_idx = sorted(ref_rng.choice(len(df), size=n_rows, replace=False))
    target_phq9 = df["phq9"].iloc[ref_idx].astype(int).tolist()

    # Group the persona pool by PHQ-9 score for fast lookup.
    by_phq9: dict[int, list[int]] = {}
    for i, score in enumerate(df["phq9"].astype(int).tolist()):
        by_phq9.setdefault(int(score), []).append(i)

    # For each slot, pick one persona whose PHQ-9 matches the target.
    # Use a per-call RNG so different sample_seed values pick different
    # personas; track per-score usage to avoid the same row at different slots
    # within one call.
    call_rng = np.random.default_rng(sample_seed)
    used: set[int] = set()
    chosen_idx: list[int] = []
    for slot_target in target_phq9:
        candidates = [i for i in by_phq9.get(slot_target, []) if i not in used]
        if not candidates:
            # Fall back to the full candidate set (allows reuse) if exhausted —
            # rare unless n_rows is large relative to per-score pool.
            candidates = list(by_phq9.get(slot_target, []))
            if not candidates:
                raise ValueError(
                    f"no personas with PHQ-9={slot_target} in {path}; cannot "
                    f"satisfy stratified draw for slot."
                )
        pick = int(call_rng.choice(candidates))
        used.add(pick)
        chosen_idx.append(pick)

    personas = df["persona"].iloc[chosen_idx].tolist()
    phq9 = df["phq9"].iloc[chosen_idx].astype(int).tolist()
    assert phq9 == target_phq9, "stratification invariant violated"
    return personas, phq9, chosen_idx


def load_well_being_zeros(num_agents: int, seed: int) -> list:
    """Load PHQ-9 well-being dicts and zero each ``phq9_sumscore``.

    Same pattern test_llms() uses: the assignment is overridden later (by
    phq9_assignments or by the TestLLMs round-robin), so the starting score
    must be cleared to avoid leaking the .sav default into the first prompt.
    """
    well_being = lp.load_phq9("data/confidential/phq9.sav", num_agents, seed=seed)
    for wb in well_being:
        wb["phq9_sumscore"] = 0
    return well_being


def build_aligned_context(prompt_file: str,
                          prompts_file: str | None = None,
                          max_chars: int = 240,
                          ) -> tuple[str, str, dict]:
    """Return (instruction_text, tweet_format_block, prompts_json).

    ``instruction_text`` comes from ``prompt_file`` (with leading ``# `` metadata
    headers stripped); ``tweet_format_block`` is the constraints/format section
    spliced out of ``prompts_file``'s ``tweet_gen.system_forced`` so the rules
    section in the instruction can be substituted in cleanly downstream.
    """
    import json

    prompts_file = prompts_file or FC.PROMPTS_FILE
    with open(prompts_file, encoding="utf-8") as fh:
        prompts_json = json.load(fh)
    instruction = load_instruction_file(prompt_file)
    format_block = derive_tweet_format_block(prompts_json, max_chars=max_chars)
    return instruction, format_block, prompts_json


_PREFIX_RE = re.compile(r"^\s*(POST|TWEET)\s*:\s*", re.IGNORECASE)


def gather_neighbor_pool(roots: list[str] | tuple[str, ...] | None = None,
                         target_filename: str = "tweets_with_phq9.csv",
                         ) -> list[tuple[str, str]]:
    """Flatten every ``{root}/seed_*/{target_filename}`` into one (agent_id, post) pool.

    Mirrors prompt_optimizer._generate_file_path + the per-row stripping in
    generate_posts_grok.load_neighbor_pool, so the pool feeds straight into
    TestLLMs.neighbor_pool.
    """
    roots = roots or DEFAULT_NEIGHBOR_ROOTS
    paths: list[str] = []
    for base in roots:
        paths.extend(sorted(glob.glob(os.path.join(base, "seed_*", target_filename))))

    pool: list[tuple[str, str]] = []
    no_content = {"NO_POST", "NO_TWEET"}
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                aid = (row.get("agent_id") or "").strip()
                text = (row.get("tweet") or "").strip()
                text = _PREFIX_RE.sub("", text).strip().strip('"').strip()
                if text and text.upper() not in no_content:
                    pool.append((aid, text))
    if not pool:
        raise FileNotFoundError(
            f"neighbour pool is empty; no rows found in {paths or roots!r}"
        )
    print(f"[neighbour-pool] {len(pool)} posts from {len(paths)} file(s) under "
          f"{', '.join(roots)}")
    return pool


_ITER_DIR_RE = re.compile(r"^iter_(\d+)$")


def parse_iter_from_prompt_path(path: str) -> int:
    """Return ``N`` from a prompt-file path whose parent dir is ``iter_<N>``.

    The prompt lives at ``<run>/iter_<N>/prompt.txt``; this function reads N
    from the parent dir's basename, not from the filename, so the prompt file
    doesn't need to carry a version number itself.
    """
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    m = _ITER_DIR_RE.match(parent)
    if not m:
        raise ValueError(
            f"prompt-file parent dir {parent!r} does not match 'iter_<N>'; "
            "place the prompt at <run-name>/iter_<N>/prompt.txt"
        )
    return int(m.group(1))
