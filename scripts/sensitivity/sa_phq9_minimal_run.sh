#!/usr/bin/env bash
# PHQ-9 conditioning generation, one dataset per PHQ-9 band.
#
# By default the MINIMAL (un-optimised) prompt variant of the "PHQ-9 conditioning"
# block in sa_run.sh: generated with the minimal prompt (iter_0/prompt.txt) instead
# of the human-optimised iter_10 prompt. Everything but the prompt is held fixed:
#     * same personas        (--agent-seed 42, identical to sa_run.sh)
#     * same neighbour posts  (--neighbor-seed 42, identical to sa_run.sh; the Qwen
#                              teacher corpus, DEFAULT_NEIGHBOR_ROOTS, for every model)
#     * varying PHQ-9 band     (--phq9-band-range lo hi, one band per setting)
#     * LLM left UNSEEDED      (--nondeterministic)
#
# Writes to a DEDICATED subdir so it never clobbers the iter_10 band data under
# data/sensitivity/phq9/:
#     <SA_ROOT>/<OUT_SUBDIR>/<band>/rep_<N>/posts.csv
#
# Every setting is an env var (defaults = the original Qwen minimal-prompt run):
#   SA_MODEL       generator alias / HF id (loaders.MODEL_ALIASES; decoding comes
#                  from loaders.STUDENT_DECODING per model)
#   SA_PROMPT      instruction file
#   NUM_PHQ9_REPS  unseeded replicates per band (sa_run.sh uses 3 for iter_10)
#   SA_ROOT        sensitivity root; another generator gets its own root, e.g.
#                  data/sensitivity/gemma4, so sa_embed / sa_analyze --root work unchanged
#   OUT_SUBDIR     phq9_minimal_prompt or phq9
#   PYTHON         interpreter with the generator's vLLM (.venv_vllm_g4 for Gemma / Mistral)
#   SA_SEED_MODE   nondet (default): --nondeterministic, the original unseeded runs;
#                  agent: --seed <SA_BASE_SEED + 100*band + rep> --per-agent-seed, one
#                  seed per (agent, round) derived from a base seed that differs per
#                  band AND rep, so no two runs share random numbers (see
#                  checks/README.md, "Retired checks")
#   SA_BASE_SEED   base for the agent mode (default 1000)
# e.g. the Gemma-4 human-optimised run (jobs/sa_phq9_gemma4.job):
#   SA_MODEL=gemma4-31b SA_PROMPT=data/sensitivity/inputs/prompt_iter_10.txt NUM_PHQ9_REPS=3 \
#   SA_ROOT=data/sensitivity/gemma4 OUT_SUBDIR=phq9 PYTHON=.venv_vllm_g4/bin/python \
#     bash scripts/sensitivity/sa_phq9_minimal_run.sh
#
# Guarded: any band/rep whose posts.csv exists is SKIPPED.
# Invoke from the repo root with the venv activated and a GPU session.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# === Config (env overrides; defaults reproduce the Qwen minimal-prompt run) ===
PROMPT="${SA_PROMPT:-data/sensitivity/inputs/prompt_iter_0.txt}"   # MINIMAL prompt (copy of qwen27_baseline/iter_0/prompt.txt)
PERSONA_FILE="data/sensitivity/inputs/personas_eval_1000_phq9.csv"
MODEL="${SA_MODEL:-Qwen/Qwen3.5-27B}"
PYTHON="${PYTHON:-python}"
NUM_AGENTS=60                          # per run; 60 × 10 posts = 600 per run (matches sa_run.sh)
CHECK_POINT=10                         # posts per agent
FIXED_AGENT_SEED=42                    # MUST match sa_run.sh so personas line up
FIXED_NEIGHBOR_SEED=42                 # MUST match sa_run.sh so neighbours line up
NUM_PHQ9_REPS="${NUM_PHQ9_REPS:-1}"    # one unseeded draw per band
SA_ROOT="${SA_ROOT:-data/sensitivity}"
OUT_SUBDIR="${OUT_SUBDIR:-phq9_minimal_prompt}"   # dedicated dir; does NOT touch phq9/ (iter_10)
SA_SEED_MODE="${SA_SEED_MODE:-nondet}"
SA_BASE_SEED="${SA_BASE_SEED:-1000}"

# === PHQ-9 bands ===========================================================
PHQ9_BAND_LABELS=("minimal" "mild" "moderate" "modsevere" "severe")
PHQ9_BAND_LOS=(0  5  10 15 20)
PHQ9_BAND_HIS=(4  9  14 19 27)

# === Run ===================================================================
if [[ ! -f "${PROMPT}" ]]; then
    echo "[fatal] prompt not found: ${PROMPT}" >&2
    exit 1
fi

echo "================================================================"
echo "PHQ-9 conditioning: ${#PHQ9_BAND_LABELS[@]} band settings × ${NUM_PHQ9_REPS} rep(s)"
echo "  model=${MODEL} python=${PYTHON}"
echo "  prompt=${PROMPT}"
echo "  agent_seed=${FIXED_AGENT_SEED} neighbor_seed=${FIXED_NEIGHBOR_SEED} (fixed) | seed mode=${SA_SEED_MODE}"
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
        if [[ "${SA_SEED_MODE}" == "agent" ]]; then
            seed_flags=(--seed "$((SA_BASE_SEED + 100 * i + rep))" --per-agent-seed)
        else
            seed_flags=(--nondeterministic)
        fi
        echo "[run] phq9 band=${label} rep=${rep}/${NUM_PHQ9_REPS} range=[${lo},${hi}]  (agent_seed=${FIXED_AGENT_SEED}, neighbor_seed=${FIXED_NEIGHBOR_SEED}, ${seed_flags[*]})"

        PYTHONPATH=src "${PYTHON}" -m utils.create_data.generate_test_data \
            --instruction-file "${PROMPT}" \
            --persona-phq9-file "${PERSONA_FILE}" \
            --model "${MODEL}" \
            --num_agents "${NUM_AGENTS}" \
            --check_point "${CHECK_POINT}" \
            --agent-seed "${FIXED_AGENT_SEED}" \
            --neighbor-seed "${FIXED_NEIGHBOR_SEED}" \
            --phq9-band-range "${lo}" "${hi}" \
            --output-csv "${out_csv}" \
            "${seed_flags[@]}"
    done
done

echo
echo "[done] ${#PHQ9_BAND_LABELS[@]} band datasets under ${SA_ROOT}/${OUT_SUBDIR}/"
echo "       next: PYTHONPATH=src python -m utils.sensitivity.sa_embed --sbert --root ${SA_ROOT}/${OUT_SUBDIR}"
