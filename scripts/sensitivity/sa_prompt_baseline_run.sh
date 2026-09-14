#!/usr/bin/env bash
# Prompt-axis sensitivity with replicates: TOTAL_REPS unseeded draws of every prompt,
# everything else fixed (same agent and neighbour seeds as sa_prompt_run.sh). With
# reps, the heatmap diagonal is the within-prompt noise floor instead of a trivial 1.0.
# Layout: <ROOT>/<label>/rep_<N>/posts.csv; rep_1 reuses the single draw from
# prompt_sa/<label>.csv. Existing reps are skipped. Then run sa_embed and
# sa_analyze --prompt-reps. Run from the repo root, venv active, GPU session.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# === Config (edit before running) =========================================
PERSONA_FILE="data/prompt_optimization_h/qwen27_baseline/inputs/personas_eval_1000_phq9.csv"
MODEL="Qwen/Qwen3.5-27B"
NUM_AGENTS=120                 # MUST match sa_prompt_run.sh so anchors line up
CHECK_POINT=10
AGENT_SEED=42                  # MUST match sa_prompt_run.sh
NEIGHBOR_SEED=42               # MUST match sa_prompt_run.sh
NUM_NEIGHBORS=5
TOTAL_REPS=3                   # unseeded draws per prompt (rep_1 reuses existing run)

PROMPT_SA="data/prompt_optimization_h/qwen27_baseline/prompt_sa"        # existing single draws (rep_1)
ROOT="data/prompt_optimization_h/qwen27_baseline/prompt_sa_reps"        # replicate tree

# label -> instruction file (mirrors sa_prompt_run.sh; the prompts in the heatmap).
# iter_10 = the human-optimized post-gen prompt (10th iteration of prompt_optimization_h);
# its rep_1 reuses the existing single draw in prompt_sa/iter_10.csv, so only the extra
# unseeded reps are generated.
LABELS=(minimal textgrad_seed24 textgrad_seed25 textgrad_seed28 textgrad_seed29 textgrad_seed53 iter_10)
declare -A PROMPTS=(
    [minimal]="data/prompt_optimization_h/qwen27_baseline/inputs/prompt_iter_0.txt"
    [textgrad_seed24]="data/prompt_optimization_h/qwen27_baseline/inputs/textgrad_seed24.txt"
    [textgrad_seed25]="data/prompt_optimization_h/qwen27_baseline/inputs/textgrad_seed25.txt"
    [textgrad_seed28]="data/prompt_optimization_h/qwen27_baseline/inputs/textgrad_seed28.txt"
    [textgrad_seed29]="data/prompt_optimization_h/qwen27_baseline/inputs/textgrad_seed29.txt"
    [textgrad_seed53]="data/prompt_optimization_h/qwen27_baseline/inputs/textgrad_seed53.txt"
    [iter_10]="data/prompt_optimization_h/qwen27_baseline/inputs/prompt_iter_10.txt"
)

# === Run ===================================================================
echo "================================================================"
echo "PROMPT reps: ${#LABELS[@]} prompts × ${TOTAL_REPS} unseeded reps × ${NUM_AGENTS} agents"
echo "  agent_seed=${AGENT_SEED} neighbor_seed=${NEIGHBOR_SEED} (fixed) | LLM unseeded"
echo "================================================================"

for label in "${LABELS[@]}"; do
    instr="${PROMPTS[$label]}"
    if [[ ! -f "${instr}" ]]; then
        echo "[warn] instruction missing for '${label}': ${instr} — skipping prompt"
        continue
    fi

    # rep_1 := existing single draw, copied into its rep dir (original untouched).
    rep1="${ROOT}/${label}/rep_1/posts.csv"
    existing="${PROMPT_SA}/${label}.csv"
    if [[ ! -f "${rep1}" && -f "${existing}" ]]; then
        mkdir -p "${ROOT}/${label}/rep_1"
        cp "${existing}" "${rep1}"
        echo "[copy] ${existing} -> ${rep1}"
    fi

    for rep in $(seq 1 "${TOTAL_REPS}"); do
        out_csv="${ROOT}/${label}/rep_${rep}/posts.csv"
        if [[ -f "${out_csv}" ]]; then
            echo "[skip] ${out_csv}"
            continue
        fi
        mkdir -p "${ROOT}/${label}/rep_${rep}"
        echo "[run] ${label} rep_${rep}/${TOTAL_REPS}  <-  ${instr}"
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
done

echo
echo "[done] replicates under ${ROOT}/<label>/rep_*/posts.csv"
echo "       next, on a GPU (embed) then analyse:"
echo "         PYTHONPATH=src python -m utils.sensitivity.sa_embed --root ${ROOT}"
echo "         PYTHONPATH=src python -m utils.sensitivity.sa_analyze --prompt-reps --root ${ROOT}"
