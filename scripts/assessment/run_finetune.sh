#!/usr/bin/env bash
# Fine-tune the BERT PHQ-9 regressor on posts generated with the iter_10
# (human-optimized) prompt, then compare baseline vs fine-tuned on a TEST_N-block
# test set (disjoint from the training personas).
#
# Every setting is an env var. GEN_TAG names the generator arm and picks the
# folder its posts live in, data/finetune/<tag>/. Empty (or `qwen`) = the
# original Qwen3.5-27B run, whose test set is assembled from 120 pre-existing +
# 180 fresh blocks; any other tag generates all TEST_N blocks in one go, e.g.
#
#   GEN_TAG=gemma4 GEN_MODEL=gemma4-31b PYTHON_GEN=.venv_vllm_g4/bin/python \
#       bash scripts/assessment/run_finetune.sh
#   GEN_TAG=mistral GEN_MODEL=mistral-small-24b PYTHON_GEN=.venv_vllm_g4/bin/python \
#       bash scripts/assessment/run_finetune.sh
#
# Run from the repo root in a GPU session (see jobs/run_finetune_*.job). All
# output is tee'd to logs/finetune[_<tag>]_<timestamp>.log.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# ============================== CONFIG ======================================
N_TRAIN="${N_TRAIN:-3000}"                 # training personas (<= 7736 available)
TEST_N="${TEST_N:-300}"                    # test blocks
CHUNK_SIZE="${CHUNK_SIZE:-100}"            # generate posts in chunks (appended + crash-resumable)
GEN_MODEL="${GEN_MODEL:-qwen27}"           # generator alias / HF id that writes the posts
GEN_TAG="${GEN_TAG:-}"                     # generator arm; "" == "qwen" (see the Qwen branch below)
[[ "${GEN_TAG}" == "qwen" ]] && GEN_TAG=""  # same arm, same paths as the default
INIT_MODEL="${INIT_MODEL:-Qwen/Qwen3.5-27B}"          # names the SOURCE (baseline) regressors
PYTHON_GEN="${PYTHON_GEN:-.venv_vllm/bin/python}"     # interpreter with the generator's vLLM
PYTHON_BERT="${PYTHON_BERT:-.venv_vllm/bin/python}"   # interpreter for the MentalBERT steps
SEEDS="${SEEDS:-34 35 36 37 38}"           # regressor seeds in <ARM_ROOT>/<arm>/models/
LR="${LR:-2e-5}"                           # low LR for fine-tuning
EPOCHS="${EPOCHS:-30}"
PROMPT="${PROMPT:-data/prompt_optimization_h/qwen27_baseline/iter_10/prompt.txt}"
TEMP="${TEMP:-}"                           # empty = per-model default (loaders.STUDENT_DECODING)
TOP_P="${TOP_P:-}"

# HF id that names the new regressor dirs and the embedding cache.
GEN_HF_ID="${GEN_HF_ID:-$(PYTHONPATH=src "${PYTHON_BERT}" -c \
    "from utils.create_data.loaders import resolve_model_id; print(resolve_model_id('${GEN_MODEL}'))" 2>/dev/null | tail -1)}"

DECODE_ARGS=()
[[ -n "${TEMP}" ]]  && DECODE_ARGS+=(--temp "${TEMP}")
[[ -n "${TOP_P}" ]] && DECODE_ARGS+=(--top_p "${TOP_P}")

# Paths
PERSONAS="data/personas_finetune_phq9.csv"

# Assessor tree (src/utils/assessors.py). An ARM is one training corpus, named
# <generator>_<generation prompt>; every arm holds models/, eval/on_<corpus>/ and
# meta.json. ARM is derived from GEN_TAG, so the legacy tags still work.
ARM_ROOT="data/assessors/bert"
BASELINE_ARM="teacher"                     # un-fine-tuned regressors every arm starts from
QWEN_FT_ARM="qwen27_optimized"             # the deployed Qwen arm (cross-generator transfer row)
if [[ -z "${GEN_TAG}" ]]; then
    ARM="${QWEN_FT_ARM}"
else
    ARM="$(PYTHONPATH=src "${PYTHON_BERT}" -c \
        "from utils.assessors import resolve; print(resolve('${GEN_TAG}'))")"
fi
BASELINE_DIR="${ARM_ROOT}/${BASELINE_ARM}/models"   # --init-from-dir / baseline --regressor-dir
QWEN_FT_DIR="${ARM_ROOT}/${QWEN_FT_ARM}/models"
FT_DIR="${ARM_ROOT}/${ARM}/models"                  # --bert-out-dir for this run

# The three eval cells this run fills: <assessor arm>/eval/on_<corpus>, corpus = ARM.
BASE_EVAL="${ARM_ROOT}/${BASELINE_ARM}/eval/on_${ARM}"
FT_EVAL="${ARM_ROOT}/${ARM}/eval/on_${ARM}"
XFER_EVAL="${ARM_ROOT}/${QWEN_FT_ARM}/eval/on_${ARM}"
# Qwen's own run has no transfer cell (it would be the same dir as FT_EVAL).
[[ "${ARM}" == "${QWEN_FT_ARM}" ]] && XFER_EVAL=""

if [[ -z "${GEN_TAG}" ]]; then
    # Qwen: the test set reuses the 120 SA_prompt blocks and tops up to TEST_N.
    TRAIN_POSTS="data/finetune/qwen/train_posts_qwen.csv"
    EXISTING_TEST_POSTS="data/prompt_optimization_h/qwen27_baseline/SA_prompt/prompt_Qwen_Qwen3.5-27B.csv"
    EXISTING_TEST_N=120                    # blocks already generated in EXISTING_TEST_POSTS
    TEST_EXTRA_PERSONAS="data/finetune/personas_test_extra.csv"
    TEST_EXTRA_POSTS="data/finetune/qwen/test_posts_extra_qwen.csv"
    TEST_POSTS="data/finetune/qwen/test_posts_qwen.csv"
    LOG_TAG=""
else
    TRAIN_POSTS="data/finetune/${GEN_TAG}/train_posts_${GEN_TAG}.csv"
    TEST_PERSONAS="data/finetune/personas_test_${TEST_N}.csv"   # same personas/PHQ-9 as Qwen's test set
    TEST_POSTS="data/finetune/${GEN_TAG}/test_posts_${GEN_TAG}.csv"
    LOG_TAG="_${GEN_TAG}"
fi

# Count finished blocks (unique agent_ids) in a posts CSV; 0 if missing.
n_blocks() {
    [[ -f "$1" ]] || { echo 0; return; }
    "${PYTHON_BERT}" -c "import pandas as pd, sys; print(pd.read_csv(sys.argv[1], usecols=['agent_id'])['agent_id'].nunique())" "$1"
}

# --- tee all output to a timestamped log ------------------------------------
export PYTHONUNBUFFERED=1
mkdir -p logs
LOG="logs/finetune${LOG_TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
echo "[run] generator=${GEN_MODEL} (${GEN_HF_ID}) tag='${GEN_TAG}' init=${INIT_MODEL}"
echo "[run] python_gen=${PYTHON_GEN} python_bert=${PYTHON_BERT}; logging to ${LOG}"

# === 1. Training personas (skip if already built) ===========================
if [[ ! -f "${PERSONAS}" ]]; then
    PYTHONPATH=src "${PYTHON_BERT}" -m utils.create_data.build_personas finetune --n "${N_TRAIN}" --out "${PERSONAS}"
fi

# === 2. Generate training posts (chunked + resumable; iter_10 prompt) ========
if [[ "$(n_blocks "${TRAIN_POSTS}")" -lt "${N_TRAIN}" ]]; then
    PYTHONPATH=src "${PYTHON_GEN}" -m utils.create_data.generate_test_data \
        --instruction-file "${PROMPT}" \
        --persona-phq9-file "${PERSONAS}" \
        --model "${GEN_MODEL}" --num_agents "${N_TRAIN}" --check_point 10 --first-n \
        --chunk-size "${CHUNK_SIZE}" "${DECODE_ARGS[@]}" \
        --output-csv "${TRAIN_POSTS}"
else
    echo "[2] ${TRAIN_POSTS} already has ${N_TRAIN} blocks; skipping generation"
fi

# === 3. Fine-tune each regressor on the training posts ======================
PYTHONPATH=src "${PYTHON_BERT}" -m utils.prompt_optimizer \
    --mode bert --model "${GEN_HF_ID}" --init-from-model "${INIT_MODEL}" --seeds ${SEEDS} \
    --posts-file "${TRAIN_POSTS}" \
    --init-from-dir "${BASELINE_DIR}" --bert-out-dir "${FT_DIR}" \
    --learning-rate "${LR}" --epochs "${EPOCHS}"

# === 4. Build the TEST_N-block test set =====================================
if [[ -z "${GEN_TAG}" ]]; then
    # Legacy Qwen layout: reuse the existing 120 blocks, generate the rest, concat.
    if [[ "${TEST_N}" -gt "${EXISTING_TEST_N}" && ! -f "${TEST_EXTRA_PERSONAS}" ]]; then
        PYTHONPATH=src "${PYTHON_BERT}" -m utils.create_data.build_personas test-extra \
            --n "${TEST_N}" --keep "${EXISTING_TEST_N}" --out "${TEST_EXTRA_PERSONAS}"
    fi
    if [[ "${TEST_N}" -gt "${EXISTING_TEST_N}" ]]; then
        PYTHONPATH=src "${PYTHON_GEN}" -m utils.create_data.generate_test_data \
            --instruction-file "${PROMPT}" \
            --persona-phq9-file "${TEST_EXTRA_PERSONAS}" \
            --model "${GEN_MODEL}" --num_agents "$((TEST_N - EXISTING_TEST_N))" \
            --check_point 10 --first-n \
            --chunk-size "${CHUNK_SIZE}" "${DECODE_ARGS[@]}" \
            --output-csv "${TEST_EXTRA_POSTS}"
        PYTHONPATH=src "${PYTHON_BERT}" - "${EXISTING_TEST_POSTS}" "${TEST_EXTRA_POSTS}" "${TEST_POSTS}" "${EXISTING_TEST_N}" <<'PY'
import sys, pandas as pd
existing, extra, out, offset = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
a = pd.read_csv(existing)
b = pd.read_csv(extra)
b["agent_id"] = b["agent_id"].astype(int) + offset      # no collision with existing ids
pd.concat([a, b], ignore_index=True).to_csv(out, index=False)
print(f"[test-set] combined {a['agent_id'].nunique()} + {b['agent_id'].nunique()} blocks -> {out}")
PY
    else
        cp "${EXISTING_TEST_POSTS}" "${TEST_POSTS}"
    fi
else
    # Tagged layout: all TEST_N blocks fresh from this generator, same personas as Qwen's set.
    if [[ ! -f "${TEST_PERSONAS}" ]]; then
        PYTHONPATH=src "${PYTHON_BERT}" -m utils.create_data.build_personas test-extra \
            --n "${TEST_N}" --keep 0 --out "${TEST_PERSONAS}"
    fi
    if [[ "$(n_blocks "${TEST_POSTS}")" -lt "${TEST_N}" ]]; then
        PYTHONPATH=src "${PYTHON_GEN}" -m utils.create_data.generate_test_data \
            --instruction-file "${PROMPT}" \
            --persona-phq9-file "${TEST_PERSONAS}" \
            --model "${GEN_MODEL}" --num_agents "${TEST_N}" --check_point 10 --first-n \
            --chunk-size "${CHUNK_SIZE}" "${DECODE_ARGS[@]}" \
            --output-csv "${TEST_POSTS}"
    else
        echo "[4] ${TEST_POSTS} already has ${TEST_N} blocks; skipping generation"
    fi
fi

# === 5. Evaluate on the test set ============================================
# (a) Qwen-trained baseline regressors
PYTHONPATH=src "${PYTHON_BERT}" -m utils.prompt_optimizer \
    --mode bert-eval --model "${INIT_MODEL}" --seeds ${SEEDS} \
    --posts-file "${TEST_POSTS}" --regressor-dir "${BASELINE_DIR}" \
    --bert-eval-out-dir "${BASE_EVAL}"

# (b) Qwen fine-tuned regressors on this generator's posts (cross-generator transfer)
if [[ -n "${XFER_EVAL}" ]]; then
    PYTHONPATH=src "${PYTHON_BERT}" -m utils.prompt_optimizer \
        --mode bert-eval --model "${INIT_MODEL}" --seeds ${SEEDS} \
        --posts-file "${TEST_POSTS}" --regressor-dir "${QWEN_FT_DIR}" \
        --bert-eval-out-dir "${XFER_EVAL}"
fi

# (c) regressors fine-tuned on this generator's posts
PYTHONPATH=src "${PYTHON_BERT}" -m utils.prompt_optimizer \
    --mode bert-eval --model "${GEN_HF_ID}" --seeds ${SEEDS} \
    --posts-file "${TEST_POSTS}" --regressor-dir "${FT_DIR}" \
    --bert-eval-out-dir "${FT_EVAL}"

# === 6. Provenance sidecar (so the arm dir describes itself) ================
PYTHONPATH=src "${PYTHON_BERT}" -c "
from utils import assessors
print('[meta]', assessors.write_meta('${ARM}',
    seeds=[int(s) for s in '${SEEDS}'.split()],
    learning_rate=float('${LR}'), epochs=int('${EPOCHS}'),
    generation_prompt_file='${PROMPT}',
    trained_by='scripts/assessment/run_finetune.sh', log='${LOG}'))
"

echo ""
echo "DONE. Compare MAE:"
echo "  baseline : ${BASE_EVAL}/aggregate.csv"
[[ -n "${XFER_EVAL}" ]] && echo "  qwen-ft  : ${XFER_EVAL}/aggregate.csv"
echo "  finetuned: ${FT_EVAL}/aggregate.csv"
