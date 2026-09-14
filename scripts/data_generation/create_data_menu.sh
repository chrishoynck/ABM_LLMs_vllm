#!/usr/bin/env bash
# Pipelines under utils/create_data/ (plus the teacher eval that lives in
# utils.prompt_optimizer). Invoke from the repo root with the venv activated
# and a GPU session. Uncomment ONE block at a time. ITER / RUN_NAME / NUM_AGENTS
# at the top drive the two iter-aware blocks (1 and 2).
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

ITER=10
RUN_NAME=qwen27_baseline
NUM_AGENTS=7


# ============================================================================
# 1. HUMAN-LOOP POST GENERATION  (generate_posts_opt_h.py)
# ----------------------------------------------------------------------------
# Reads iter_N/prompt.txt, samples NUM_AGENTS persona-PHQ9 pairs from the eval
# CSV with sample_seed=N, generates check_point (default 10) cold-start posts
# per agent, writes posts.csv + feedback.md scaffold into iter_N/. LLM sampling
# is non-deterministic by design, re-running produces fresh posts so you can
# eyeball variability before locking in a prompt change.
PYTHONPATH=src python -m utils.create_data.generate_posts_opt_h \
    --prompt-file "data/prompt_optimization_h/${RUN_NAME}/iter_${ITER}/prompt.txt" \
    --persona-phq9-file data/personas_eval_1000_phq9.csv \
    --num_agents "${NUM_AGENTS}"


# ============================================================================
# 2. TEACHER EVAL  (utils.prompt_optimizer  --mode tweets-rerun-test)
# ----------------------------------------------------------------------------
# Companion to block 1: scores iter_N/prompt.txt against 100 fresh agent-PHQ9
# pairs (sample_seed=999). Student generates 3 cold-start posts per agent with
# the prompt as system; teacher (same model) rates each set 0-10 using the
# textgrad training-time rubric. Writes test_raw_scores.csv, test_scores_phq9.csv,
# test_posts.csv, eval_meta.txt to iter_N/. Same --sample-seed across iters →
# same agents → directly comparable scores.
# PYTHONPATH=src python -m utils.prompt_optimizer \
#     --mode tweets-rerun-test \
#     --instruction-file "data/prompt_optimization_h/${RUN_NAME}/iter_${ITER}/prompt.txt" \
#     --persona-phq9-file data/personas_eval_1000_phq9.csv \
#     --num-agents 100 \
#     --sample-seed 999 \
#     --seeds 42 \
#     --model Qwen/Qwen3.5-27B


# ============================================================================
# 3. VARIANT COMPARISON  (generate_test_data.py)
# ----------------------------------------------------------------------------
# Sweeps best_instruction*.txt (or any *.txt) under --instruction-dir, OR runs
# one --instruction-file. For each variant, samples NUM agents with rng(variant_idx)
# so different variants see different agents (per-variant fresh subset). Runs
# deterministic generation, writes one CSV per (variant, model) + a scores.csv
# into a sibling SA_prompt/ folder where you hand-fill training_score / test_score.
# Neighbour pool defaults to data/test_post/Qwen_Qwen3.5-27B/{inter,no_inter}/.
# PYTHONPATH=src python -m utils.create_data.generate_test_data \
#     --instruction-dir data/prompt_optimization_h/prompt_variants \
#     --persona-phq9-file data/personas_eval_1000_phq9.csv \
#     --model qwen27 \
#     --num_agents 12 \
#     --seed 42

# The old-framework builder (generate_synthetic_dataset.py) and the Grok API
# builder (generate_posts_grok.py) were binned on 2026-09-14; see bin/NOTES.md.
