#!/usr/bin/env bash
# PHQ-9 conditioning generation for the MINIMAL (un-optimised) prompt.
#
# Variant of the "PHQ-9 conditioning" block in sa_run.sh: one dataset per PHQ-9
# band, but generated with the minimal prompt (iter_0/prompt.txt) instead of the
# human-optimised iter_10 prompt. Everything but the prompt is held fixed:
#     * same personas        (--agent-seed 42, identical to sa_run.sh)
#     * same neighbour posts  (--neighbor-seed 42, identical to sa_run.sh)
#     * varying PHQ-9 band     (--phq9-band-range lo hi, one band per setting)
#     * LLM left UNSEEDED      (--nondeterministic)
#
# Writes to a DEDICATED subdir so it never clobbers the iter_10 band data under
# data/sensitivity/phq9/:
#     <SA_ROOT>/phq9_minimal_prompt/<band>/rep_1/posts.csv
#
# One rep per band. Guarded: any band whose posts.csv exists is SKIPPED.
# Invoke from the repo root with the venv activated and a GPU session.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# === Config (edit before running) =========================================
PROMPT="data/sensitivity/inputs/prompt_iter_0.txt"   # MINIMAL prompt (copy of qwen27_baseline/iter_0/prompt.txt)
PERSONA_FILE="data/sensitivity/inputs/personas_eval_1000_phq9.csv"
MODEL="Qwen/Qwen3.5-27B"
NUM_AGENTS=60                          # per run; 60 × 10 posts = 600 per run (matches sa_run.sh)
CHECK_POINT=10                         # posts per agent
FIXED_AGENT_SEED=42                    # MUST match sa_run.sh so personas line up
FIXED_NEIGHBOR_SEED=42                 # MUST match sa_run.sh so neighbours line up
NUM_PHQ9_REPS=1                        # one unseeded draw per band
SA_ROOT="data/sensitivity"
OUT_SUBDIR="phq9_minimal_prompt"       # dedicated dir; does NOT touch phq9/ (iter_10)

# === PHQ-9 bands ===========================================================
PHQ9_BAND_LABELS=("minimal" "mild" "moderate" "modsevere" "severe")
PHQ9_BAND_LOS=(0  5  10 15 20)
PHQ9_BAND_HIS=(4  9  14 19 27)

# === Run ===================================================================
if [[ ! -f "${PROMPT}" ]]; then
    echo "[fatal] minimal prompt not found: ${PROMPT}" >&2
    exit 1
fi

echo "================================================================"
echo "PHQ-9 conditioning (MINIMAL prompt): ${#PHQ9_BAND_LABELS[@]} band settings × ${NUM_PHQ9_REPS} rep"
echo "  prompt=${PROMPT}"
echo "  agent_seed=${FIXED_AGENT_SEED} neighbor_seed=${FIXED_NEIGHBOR_SEED} (fixed) | LLM unseeded"
echo "  output -> ${SA_ROOT}/${OUT_SUBDIR}/<band>/rep_<N>/posts.csv"
echo "================================================================"

for i in "${!PHQ9_BAND_LABELS[@]}"; do
    label=${PHQ9_BAND_LABELS[$i]}
    lo=${PHQ9_BAND_LOS[$i]}
    hi=${PHQ9_BAND_HIS[$i]}
    for rep in $(seq 1 ${NUM_PHQ9_REPS}); do
        out_dir="${SA_ROOT}/${OUT_SUBDIR}/${label}/rep_${rep}"
        out_csv="${out_dir}/posts.csv"
        if [[ -f "${out_csv}" ]]; then
            echo "[skip] ${out_csv} exists"
            continue
        fi
        mkdir -p "${out_dir}"
        echo "[run] phq9 band=${label} rep=${rep}/${NUM_PHQ9_REPS} range=[${lo},${hi}]  (agent_seed=${FIXED_AGENT_SEED}, neighbor_seed=${FIXED_NEIGHBOR_SEED})"

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
done

echo
echo "[done] ${#PHQ9_BAND_LABELS[@]} band datasets under ${SA_ROOT}/${OUT_SUBDIR}/"
echo "       next: PYTHONPATH=src python -m utils.sensitivity.sa_embed --root ${SA_ROOT}/${OUT_SUBDIR}"
