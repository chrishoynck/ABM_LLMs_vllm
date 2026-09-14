#!/usr/bin/env bash
# Score one generator's 300-block held-out set with the Qwen3.5-27B LLM assessor,
# using BOTH the minimal and the TextGrad-optimized PHQ-9 prompt, per prompt seed.
#
# This is the cross-generator counterpart of run_minimal_shift.sh steps 2/2b:
# the same Qwen assessor, the same five prompt seeds, but the posts were written
# by another student generator (e.g. Gemma 4) via run_finetune.sh with
# GEN_TAG=<tag>. Together with the MentalBERT+MLP evals this gives the
# generator x estimator table (`utils.visualization multimodel`, run as the last step).
#
# Usage (GPU session, repo root; the Qwen venv, NOT .venv_vllm_g4):
#   bash scripts/assessment/run_llm_assessor_on_heldout.sh gemma4
#   POSTS=path/to/other.csv bash scripts/assessment/run_llm_assessor_on_heldout.sh <tag>
#
# Outputs (existing minimal_human300/ and eval_on_human300/ are untouched):
#   data/test_post/optimized_phq9/Qwen3.5-27B_seed<s>/minimal_<tag>300/test_raw_scores.csv
#   data/test_post/optimized_phq9/Qwen3.5-27B_seed<s>/eval_on_<tag>300/test_raw_scores.csv
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

TAG="${1:?usage: run_llm_assessor_on_heldout.sh <tag>   (e.g. gemma4)}"
PYTHON="${PYTHON:-.venv_vllm/bin/python}"          # interpreter with vLLM 0.17.1 + Qwen
MODEL="Qwen/Qwen3.5-27B"                             # the assessor (fixed across generators)
PROMPT_SEEDS="${PROMPT_SEEDS:-23 24 25 32 33}"       # same seeds the optimized group uses
POSTS="${POSTS:-data/finetune/${TAG}/test_posts_${TAG}.csv}"
OPT_DIR="data/test_post/optimized_phq9"
MIN_SUBDIR="minimal_${TAG}300"
OPT_SUBDIR="eval_on_${TAG}300"

[[ -f "${POSTS}" ]] || { echo "posts file not found: ${POSTS}" >&2; exit 1; }
# The minimal prompt is seed-independent (data/prompts_post_minimal.json); write it
# into any seed dir that lacks it (same as run_minimal_shift.sh step 0).
for s in ${PROMPT_SEEDS}; do
  d="${OPT_DIR}/Qwen3.5-27B_seed${s}"
  [[ -f "${d}/optimized_instruction.txt" ]] || { echo "missing optimized_instruction.txt in ${d}" >&2; exit 1; }
  if [[ ! -f "${d}/minimal_instruction.txt" ]]; then
    PYTHONPATH=src "${PYTHON}" -c "import json,sys; open(sys.argv[2],'w').write(json.load(open(sys.argv[1]))['phq9']['system_instruction'])" \
        data/prompts_post_minimal.json "${d}/minimal_instruction.txt"
    echo "  wrote ${d}/minimal_instruction.txt"
  fi
done

export PYTHONUNBUFFERED=1
mkdir -p logs
LOG="logs/llm_assessor_${TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
echo "[run] tag=${TAG} posts=${POSTS} seeds=${PROMPT_SEEDS}; logging to ${LOG}"

echo "[1] MINIMAL prompt on ${POSTS} -> ${MIN_SUBDIR}/"
PYTHONPATH=src "${PYTHON}" -m utils.prompt_optimizer --mode phq9-rerun-test \
    --model "${MODEL}" --seeds ${PROMPT_SEEDS} \
    --instruction-filename minimal_instruction.txt \
    --posts-file "${POSTS}" \
    --result-subdir "${MIN_SUBDIR}"

echo "[2] OPTIMIZED (TextGrad) prompt on ${POSTS} -> ${OPT_SUBDIR}/"
PYTHONPATH=src "${PYTHON}" -m utils.prompt_optimizer --mode phq9-rerun-test \
    --model "${MODEL}" --seeds ${PROMPT_SEEDS} \
    --instruction-filename optimized_instruction.txt \
    --posts-file "${POSTS}" \
    --result-subdir "${OPT_SUBDIR}"

echo ""
echo "DONE. Per-seed scores under:"
for s in ${PROMPT_SEEDS}; do
  echo "  ${OPT_DIR}/Qwen3.5-27B_seed${s}/${MIN_SUBDIR}/test_raw_scores.csv"
  echo "  ${OPT_DIR}/Qwen3.5-27B_seed${s}/${OPT_SUBDIR}/test_raw_scores.csv"
done

# === 3. Generator x estimator table + figures (CPU; missing inputs are skipped) =
echo "[3] multimodel summary"
PYTHONPATH=src "${PYTHON}" -m utils.visualization multimodel
