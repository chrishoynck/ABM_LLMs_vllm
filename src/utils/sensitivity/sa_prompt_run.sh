#!/usr/bin/env bash
# Prompt-axis sensitivity GENERATION driver (counterpart of sa_run.sh).
#
# Generates posts for a set of post-generation instruction prompts —
#   minimal (un-optimised), iter_10 (human-in-the-loop), and the TextGrad seeds —
# with EVERYTHING but the prompt held fixed:
#     * same personas       (same --agent-seed across every prompt)
#     * same neighbour posts (same --neighbor-seed across every prompt)
#     * same per-slot PHQ-9  (falls out of the fixed agent-seed)
#     * LLM left UNSEEDED    (--nondeterministic; one free draw per prompt)
# so a paired per-(agent, round) cosine in sa_prompt.py isolates the prompt.
#
# Guarded: any variant whose output CSV already exists is SKIPPED (no
# regeneration). Delete a CSV to force its variant to re-generate.
#
# Run from the repo root with the venv activated and a GPU session.
set -euo pipefail

# === Config (edit before running) =========================================
PERSONA_FILE="data/personas_eval_1000_phq9.csv"
MODEL="Qwen/Qwen3.5-27B"
NUM_AGENTS=120                 # 120 × 10 posts = 1200 anchors per variant
CHECK_POINT=10                 # posts per agent
AGENT_SEED=42                  # SAME for every variant → identical personas per slot
NEIGHBOR_SEED=42               # SAME for every variant → identical neighbours per anchor
NUM_NEIGHBORS=5
# Dedicated dir so discovery never collides with the shared SA_prompt/ outputs
# (e.g. prompt_Qwen_Qwen3.5-27B.csv, which other pipelines depend on by name).
SA_DIR="data/prompt_optimization_h/qwen27_baseline/prompt_sa"

# label -> instruction file. The label becomes the CSV stem, which sa_prompt.py
# uses for --box-against / --heatmap-exclude matching (so keep 'minimal' and
# 'iter_10' in the names). Edit this block to add/remove prompts.
LABELS=(minimal iter_10 textgrad_seed24 textgrad_seed25 textgrad_seed28 textgrad_seed29 textgrad_seed53)
declare -A PROMPTS=(
    [minimal]="data/prompt_optimization_h/qwen27_baseline/iter_0/prompt.txt"
    [iter_10]="data/prompt_optimization_h/qwen27_baseline/iter_10/prompt.txt"
    [textgrad_seed24]="data/test_post/optimized_tweets/Qwen3.5-27B_seed24/best_instruction_tweet.txt"
    [textgrad_seed25]="data/test_post/optimized_tweets/Qwen3.5-27B_seed25/best_instruction_tweet.txt"
    [textgrad_seed28]="data/test_post/optimized_tweets/Qwen3.5-27B_seed28/best_instruction_tweet.txt"
    [textgrad_seed29]="data/test_post/optimized_tweets/Qwen3.5-27B_seed29/best_instruction_tweet.txt"
    [textgrad_seed53]="data/test_post/optimized_tweets/Qwen3.5-27B_seed53/best_instruction_tweet.txt"
)

# === Run ===================================================================
mkdir -p "${SA_DIR}"
echo "================================================================"
echo "PROMPT axis: ${#LABELS[@]} prompts × ${NUM_AGENTS} agents"
echo "  agent_seed=${AGENT_SEED} neighbor_seed=${NEIGHBOR_SEED} (fixed) | LLM unseeded"
echo "================================================================"

for label in "${LABELS[@]}"; do
    instr="${PROMPTS[$label]}"
    out_csv="${SA_DIR}/${label}.csv"

    if [[ -f "${out_csv}" ]]; then
        echo "[skip] ${out_csv} exists"
        continue
    fi
    if [[ ! -f "${instr}" ]]; then
        echo "[warn] instruction file missing for '${label}': ${instr} — skipping"
        continue
    fi

    echo "[run] ${label}  <-  ${instr}"
    PYTHONPATH=src python -m utils.create_data.generate_test_data \
        --instruction-file "${instr}" \
        --persona-phq9-file "${PERSONA_FILE}" \
        --model "${MODEL}" \
        --num_agents "${NUM_AGENTS}" \
        --check_point "${CHECK_POINT}" \
        --agent-seed "${AGENT_SEED}" \
        --neighbor-seed "${NEIGHBOR_SEED}" \
        --num-neighbors "${NUM_NEIGHBORS}" \
        --output-csv "${out_csv}" \
        --nondeterministic
    echo "[ok] ${out_csv}"
done

echo
echo "[done] variants under ${SA_DIR}/"
echo "       next: PYTHONPATH=src python -m utils.sensitivity.sa_prompt \\"
echo "                  --sa-dir ${SA_DIR} --heatmap-exclude iter_10"
