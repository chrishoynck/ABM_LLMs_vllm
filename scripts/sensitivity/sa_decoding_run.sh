#!/usr/bin/env bash
# Decoding-axis sensitivity generation: 7 (temperature, top_p) settings x 3 reps,
# with personas, neighbour posts and PHQ-9 held fixed (same seeds as the neighbour
# axis in sa_run.sh). Settings: baseline (0.7, 0.9) plus temp and top_p up and
# down, alone and together. Output: <SA_ROOT>/decoding/setting_<label>/rep_<N>/
# posts.csv; existing reps are skipped, so it resumes.
#
# Every setting is an env var (defaults = the original unseeded Qwen run):
#   SA_MODEL SA_PROMPT SA_ROOT PYTHON   as in sa_phq9_minimal_run.sh
#   SA_SEED_MODE   nondet (default): --nondeterministic; agent: --seed
#                  <SA_BASE_SEED + 100*setting + rep> --per-agent-seed (no two runs
#                  share random numbers; see checks/README.md "Retired checks")
#   SA_BASE_SEED   base for the agent mode (default 1000)
#   SA_GRID        ';'-separated "label temp top_p" entries; keep the labels
#                  (baseline, temp_hi, ...) so sa_analyze groups them, the values are
#                  read back from the posts.csv.meta.json sidecars. Gemma-4 grid
#                  (jobs/sa_seeded.job): temp 0.7/1.0/1.3, top_p 0.95 (vendor) /
#                  0.975 (operating point) / 1.0, mirroring Qwen's 0.8/0.9/1.0.
# Run from the repo root, venv active, GPU session.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# === Config (env overrides; defaults reproduce the original Qwen run) ======
PROMPT="${SA_PROMPT:-data/sensitivity/inputs/prompt_iter_10.txt}"   # copy of qwen27_baseline/iter_10/prompt.txt
PERSONA_FILE="data/sensitivity/inputs/personas_eval_1000_phq9.csv"
MODEL="${SA_MODEL:-Qwen/Qwen3.5-27B}"
PYTHON="${PYTHON:-python}"
NUM_AGENTS=60                          # per run; 60 x 10 posts = 600 per run
CHECK_POINT=10                         # posts per agent
FIXED_AGENT_SEED=42                    # SAME for every setting -> identical personas + PHQ-9
FIXED_NEIGHBOR_SEED=42                 # SAME for every setting -> identical neighbours
NUM_REPS=3                             # replicates per setting
SA_ROOT="${SA_ROOT:-data/sensitivity}"
ANCHOR="decoding"
SA_SEED_MODE="${SA_SEED_MODE:-nondet}"
SA_BASE_SEED="${SA_BASE_SEED:-1000}"

# label  temp  top_p   (baseline = operating point; +/- = step temp 0.3, top_p 0.1)
SA_GRID="${SA_GRID:-baseline 0.7 0.9;temp_hi 1.0 0.9;temp_lo 0.4 0.9;topp_hi 0.7 1.0;topp_lo 0.7 0.8;both_hi 1.0 1.0;both_lo 0.4 0.8}"
IFS=';' read -r -a SETTINGS <<< "${SA_GRID}"

# === Helper ================================================================
run_one() {
    local label="$1"
    local temp="$2"
    local top_p="$3"
    local rep="$4"
    local idx="$5"
    local out_dir="${SA_ROOT}/${ANCHOR}/setting_${label}/rep_${rep}"
    local out_csv="${out_dir}/posts.csv"

    if [[ -f "${out_csv}" ]]; then
        echo "[skip] ${out_csv} exists"
        return
    fi
    mkdir -p "${out_dir}"
    local seed_flags
    if [[ "${SA_SEED_MODE}" == "agent" ]]; then
        seed_flags=(--seed "$((SA_BASE_SEED + 100 * idx + rep))" --per-agent-seed)
    else
        seed_flags=(--nondeterministic)
    fi
    echo "[run] anchor=${ANCHOR} setting=${label} temp=${temp} top_p=${top_p} "\
"rep=${rep}/${NUM_REPS} (agent_seed=${FIXED_AGENT_SEED} neighbor_seed=${FIXED_NEIGHBOR_SEED} ${seed_flags[*]})"

    PYTHONPATH=src "${PYTHON}" -m utils.create_data.generate_test_data \
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
        "${seed_flags[@]}"
}

# === Run ===================================================================
echo "================================================================"
echo "DECODING axis: ${#SETTINGS[@]} settings x ${NUM_REPS} reps  (temp/top_p swept)"
echo "  model=${MODEL} python=${PYTHON} | agent_seed=${FIXED_AGENT_SEED} neighbor_seed=${FIXED_NEIGHBOR_SEED} (fixed) | seed mode=${SA_SEED_MODE}"
echo "================================================================"
for idx in "${!SETTINGS[@]}"; do
    read -r label temp top_p <<< "${SETTINGS[$idx]}"
    for rep in $(seq 1 ${NUM_REPS}); do
        run_one "${label}" "${temp}" "${top_p}" "${rep}" "${idx}"
    done
done

echo
N_RUNS=$(( ${#SETTINGS[@]} * NUM_REPS ))
echo "[done] ${N_RUNS} decoding runs under ${SA_ROOT}/${ANCHOR}/"
echo "       next, on a GPU (embed) then analyse:"
echo "         PYTHONPATH=src python -m utils.sensitivity.sa_embed --root ${SA_ROOT}"
echo "         PYTHONPATH=src python -m utils.sensitivity.sa_analyze --root ${SA_ROOT}"
