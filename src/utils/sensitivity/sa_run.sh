#!/usr/bin/env bash
# Sensitivity-analysis generation sweep.
#   Two axes: NEIGHBOUR and AGENT sampling.
#   4 settings per axis × 3 unseeded LLM replicates = 12 runs per axis → 24 total.
#   Each replicate is a fresh non-deterministic generation (no seeded LLM),
#   so within-setting variance estimates baseline LLM noise.
#
# Invoke from the repo root with the venv activated and a GPU session.
set -euo pipefail

# === Config (edit before running) =========================================
PROMPT="data/prompt_optimization_h/qwen27_baseline/iter_10/prompt.txt"
PERSONA_FILE="data/personas_eval_1000_phq9.csv"
MODEL="Qwen/Qwen3.5-27B"
NUM_AGENTS=60                          # per run; 60 × 10 posts = 600 per run
CHECK_POINT=10                         # posts per agent
AXIS_SEEDS=(11 23 47 89)               # 4 distinct seeds drive the varying axis
FIXED_AGENT_SEED=42                    # used during neighbour-axis runs
FIXED_NEIGHBOR_SEED=42                 # used during agent-axis runs
NUM_REPS=3                             # unseeded replicates per setting
SA_ROOT="data/sensitivity"

# === Helper ================================================================
run_one() {
    local axis="$1"
    local setting_seed="$2"
    local rep="$3"
    local agent_seed="$4"
    local neighbor_seed="$5"
    local extra_flags="${6:-}"
    local out_dir="${SA_ROOT}/${axis}/setting_${setting_seed}/rep_${rep}"
    local out_csv="${out_dir}/posts.csv"

    if [[ -f "${out_csv}" ]]; then
        echo "[skip] ${out_csv} exists"
        return
    fi
    mkdir -p "${out_dir}"
    echo "[run] axis=${axis} setting=${setting_seed} rep=${rep}/${NUM_REPS} "\
"agent_seed=${agent_seed} neighbor_seed=${neighbor_seed} ${extra_flags}"

    PYTHONPATH=src python -m utils.create_data.generate_test_data \
        --instruction-file "${PROMPT}" \
        --persona-phq9-file "${PERSONA_FILE}" \
        --model "${MODEL}" \
        --num_agents "${NUM_AGENTS}" \
        --check_point "${CHECK_POINT}" \
        --agent-seed "${agent_seed}" \
        --neighbor-seed "${neighbor_seed}" \
        --output-csv "${out_csv}" \
        --nondeterministic ${extra_flags}
}

# === Neighbour axis: vary neighbor_seed, fix agent_seed ====================
echo "================================================================"
echo "NEIGHBOUR axis: ${#AXIS_SEEDS[@]} settings × ${NUM_REPS} reps"
echo "================================================================"
for setting in "${AXIS_SEEDS[@]}"; do
    for rep in $(seq 1 ${NUM_REPS}); do
        run_one "neighbor" "${setting}" "${rep}" \
                "${FIXED_AGENT_SEED}" "${setting}"
    done
done

# === Agent axis: vary agent_seed, fix neighbor_seed ========================
# Uses --stratify-phq9 so slot i has the same PHQ-9 across all agent settings
# (only the persona text differs). This enables neighbour-style paired (slot,
# round) cosine comparisons in sa_analyze.py. The reference seed below MUST
# stay constant across every run in this axis or the per-slot PHQ-9 vector
# desynchronises.
STRATIFY_REF_SEED=0
echo "================================================================"
echo "AGENT axis: ${#AXIS_SEEDS[@]} settings × ${NUM_REPS} reps  (--stratify-phq9, ref=${STRATIFY_REF_SEED})"
echo "================================================================"
for setting in "${AXIS_SEEDS[@]}"; do
    for rep in $(seq 1 ${NUM_REPS}); do
        run_one "agent" "${setting}" "${rep}" \
                "${setting}" "${FIXED_NEIGHBOR_SEED}" \
                "--stratify-phq9 --stratify-ref-seed ${STRATIFY_REF_SEED}"
    done
done

# === Joint axis: vary BOTH agent and neighbor with tied seeds ==============
# Tests whether the two axes are additive or interact. Same paired-per-slot
# comparison as the other two axes, thanks to --stratify-phq9 holding PHQ-9
# constant per slot across joint settings.
echo "================================================================"
echo "JOINT axis: ${#AXIS_SEEDS[@]} settings × ${NUM_REPS} reps  (tied seeds, stratified)"
echo "================================================================"
for setting in "${AXIS_SEEDS[@]}"; do
    for rep in $(seq 1 ${NUM_REPS}); do
        run_one "joint" "${setting}" "${rep}" \
                "${setting}" "${setting}" \
                "--stratify-phq9 --stratify-ref-seed ${STRATIFY_REF_SEED}"
    done
done

# === PHQ-9 conditioning: same agents + same neighbours, vary PHQ-9 band ====
# Tests how the PHQ-9 input (a DESIGNED input variable, not a nuisance) shapes
# the output distribution. Every agent visits every band exactly once across
# the 5 settings; within each band-setting, agents are spread uniformly over
# the band's PHQ-9 values (not all pinned to one value). 1 rep per setting,
# no LLM-noise baseline (per design — PHQ-9 effect dwarfs LLM noise).
PHQ9_BAND_LABELS=("minimal" "mild" "moderate" "modsevere" "severe")
PHQ9_BAND_LOS=(0  5  10 15 20)
PHQ9_BAND_HIS=(4  9  14 19 27)
echo "================================================================"
echo "PHQ-9 conditioning: ${#PHQ9_BAND_LABELS[@]} band settings × 1 rep"
echo "================================================================"
for i in "${!PHQ9_BAND_LABELS[@]}"; do
    label=${PHQ9_BAND_LABELS[$i]}
    lo=${PHQ9_BAND_LOS[$i]}
    hi=${PHQ9_BAND_HIS[$i]}
    out_dir="${SA_ROOT}/phq9/${label}"
    out_csv="${out_dir}/posts.csv"
    if [[ -f "${out_csv}" ]]; then
        echo "[skip] ${out_csv} exists"
        continue
    fi
    mkdir -p "${out_dir}"
    echo "[run] phq9-conditioning band=${label} range=[${lo},${hi}]  (agent_seed=${FIXED_AGENT_SEED}, neighbor_seed=${FIXED_NEIGHBOR_SEED})"

    PYTHONPATH=src python -m utils.create_data.generate_test_data \
        --instruction-file "${PROMPT}" \
        --persona-phq9-file "${PERSONA_FILE}" \
        --model "${MODEL}" \
        --num_agents "${NUM_AGENTS}" \
        --check_point "${CHECK_POINT}" \
        --agent-seed "${FIXED_AGENT_SEED}" \
        --neighbor-seed "${FIXED_NEIGHBOR_SEED}" \
        --phq9-band-range "${lo}" "${hi}" \
        --output-csv "${out_csv}" \
        --nondeterministic
done

echo
N_AXIS_RUNS=$((${#AXIS_SEEDS[@]} * NUM_REPS * 3))
N_PHQ9_RUNS=${#PHQ9_BAND_LABELS[@]}
echo "[done] ${N_AXIS_RUNS} axis runs + ${N_PHQ9_RUNS} PHQ-9 conditioning runs; output under ${SA_ROOT}/"
echo "       next: python -m utils.sensitivity.sa_embed"
