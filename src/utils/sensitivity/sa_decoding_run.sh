#!/usr/bin/env bash
# Decoding-parameter sensitivity GENERATION driver (temperature + top_p axis).
#
# Counterpart of sa_run.sh's NEIGHBOUR axis: structurally identical (same 60
# agents across every setting), but the swept axis is the LLM's decoding params
# (temperature, top_p) instead of a neighbour seed. Everything else is held
# fixed so a paired per-(agent, round) cosine in sa_analyze.py isolates the
# decoding effect:
#     * same personas        (--agent-seed 42, FIXED across all settings)
#     * same neighbour posts  (--neighbor-seed 42, FIXED across all settings)
#     * same per-slot PHQ-9   (falls out of the fixed agent-seed)
#     * LLM left UNSEEDED      (--nondeterministic; 3 fresh draws per setting)
#
# Each (temp, top_p) pair is a "setting"; each unseeded draw a "rep" — exactly
# sa_analyze.py's within/cross design (within-setting = LLM-noise floor at that
# decoding point; cross-setting = how much the decoding change moves outputs).
#
# Settings: the baseline (the operating point the simulation actually uses) plus
# six perturbations — temp +/-, top_p +/-, and both +/-:
#     baseline (0.7, 0.9)
#     temp_hi  (1.0, 0.9)   temp_lo  (0.4, 0.9)
#     topp_hi  (0.7, 1.0)   topp_lo  (0.7, 0.8)
#     both_hi  (1.0, 1.0)   both_lo  (0.4, 0.8)
# = 7 settings x 3 reps = 21 generations.
#
# Output layout (consumed by sa_embed + sa_analyze, auto-discovered):
#     data/sensitivity/decoding/setting_<label>/rep_<N>/posts.csv
#
# Guarded: any rep whose posts.csv already exists is SKIPPED (no overwrite), so
# this resumes across job submissions. Run from the repo root with the venv
# activated and a GPU session.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# === Config (edit before running) =========================================
PROMPT="data/sensitivity/inputs/prompt_iter_10.txt"   # copy of qwen27_baseline/iter_10/prompt.txt
PERSONA_FILE="data/sensitivity/inputs/personas_eval_1000_phq9.csv"
MODEL="Qwen/Qwen3.5-27B"
NUM_AGENTS=60                          # per run; 60 x 10 posts = 600 per run
CHECK_POINT=10                         # posts per agent
FIXED_AGENT_SEED=42                    # SAME for every setting -> identical personas + PHQ-9
FIXED_NEIGHBOR_SEED=42                 # SAME for every setting -> identical neighbours
NUM_REPS=3                             # unseeded replicates per setting
SA_ROOT="data/sensitivity"
ANCHOR="decoding"

# label  temp  top_p   (baseline = operating point; +/- = step temp 0.3, top_p 0.1)
SETTINGS=(
    "baseline 0.7 0.9"
    "temp_hi  1.0 0.9"
    "temp_lo  0.4 0.9"
    "topp_hi  0.7 1.0"
    "topp_lo  0.7 0.8"
    "both_hi  1.0 1.0"
    "both_lo  0.4 0.8"
)

# === Helper ================================================================
run_one() {
    local label="$1"
    local temp="$2"
    local top_p="$3"
    local rep="$4"
    local out_dir="${SA_ROOT}/${ANCHOR}/setting_${label}/rep_${rep}"
    local out_csv="${out_dir}/posts.csv"

    if [[ -f "${out_csv}" ]]; then
        echo "[skip] ${out_csv} exists"
        return
    fi
    mkdir -p "${out_dir}"
    echo "[run] anchor=${ANCHOR} setting=${label} temp=${temp} top_p=${top_p} "\
"rep=${rep}/${NUM_REPS} (agent_seed=${FIXED_AGENT_SEED} neighbor_seed=${FIXED_NEIGHBOR_SEED})"

    PYTHONPATH=src python -m utils.create_data.generate_test_data \
        --instruction-file "${PROMPT}" \
        --persona-phq9-file "${PERSONA_FILE}" \
        --model "${MODEL}" \
        --num_agents "${NUM_AGENTS}" \
        --check_point "${CHECK_POINT}" \
        --agent-seed "${FIXED_AGENT_SEED}" \
        --neighbor-seed "${FIXED_NEIGHBOR_SEED}" \
        --temp "${temp}" \
        --top_p "${top_p}" \
        --output-csv "${out_csv}" \
        --nondeterministic
}

# === Run ===================================================================
echo "================================================================"
echo "DECODING axis: ${#SETTINGS[@]} settings x ${NUM_REPS} reps  (temp/top_p swept)"
echo "  agent_seed=${FIXED_AGENT_SEED} neighbor_seed=${FIXED_NEIGHBOR_SEED} (fixed) | LLM unseeded"
echo "================================================================"
for spec in "${SETTINGS[@]}"; do
    read -r label temp top_p <<< "${spec}"
    for rep in $(seq 1 ${NUM_REPS}); do
        run_one "${label}" "${temp}" "${top_p}" "${rep}"
    done
done

echo
N_RUNS=$(( ${#SETTINGS[@]} * NUM_REPS ))
echo "[done] ${N_RUNS} decoding runs under ${SA_ROOT}/${ANCHOR}/"
echo "       next, on a GPU (embed) then analyse:"
echo "         PYTHONPATH=src python -m utils.sensitivity.sa_embed --root ${SA_ROOT}"
echo "         PYTHONPATH=src python -m utils.sensitivity.sa_analyze --root ${SA_ROOT}"
