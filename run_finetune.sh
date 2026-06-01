#!/usr/bin/env bash
# Fine-tune the BERT PHQ-9 regressor on posts generated with the iter_10 prompt,
# then compare baseline vs fine-tuned on a TEST_N-block test set (disjoint from
# the training personas).
#
# Run from the repo root with the venv activated and a GPU session. All output is
# also tee'd to logs/finetune_<timestamp>.log so you can `tail -f` progress.
set -euo pipefail

# ============================== CONFIG ======================================
N_TRAIN=3000                       # training personas (<= 7736 available)
TEST_N=300                         # test blocks (reuses the existing 120, adds the rest)
CHUNK_SIZE=100                     # generate posts in chunks of this many personas;
                                   # each chunk is appended + crash-resumable (0 = single-shot)
GEN_MODEL="qwen27"                 # model that writes the training posts
REG_MODEL="Qwen/Qwen3.5-27B"       # names the regressor dirs
SEEDS="34 35 36 37 38"             # regressor seeds in data/test_post/bert_regression/
LR="2e-5"                          # low LR for fine-tuning
EPOCHS=30
PROMPT="data/prompt_optimization_h/qwen27_baseline/iter_10/prompt.txt"

# Paths
PERSONAS="data/personas_finetune_phq9.csv"
TRAIN_POSTS="data/finetune/train_posts.csv"
EXISTING_TEST_POSTS="data/prompt_optimization_h/qwen27_baseline/SA_prompt/prompt_Qwen_Qwen3.5-27B.csv"
EXISTING_TEST_N=120                # blocks already generated in EXISTING_TEST_POSTS
TEST_EXTRA_PERSONAS="data/finetune/personas_test_extra.csv"
TEST_EXTRA_POSTS="data/finetune/test_posts_extra.csv"
TEST_POSTS="data/finetune/test_posts.csv"   # combined TEST_N-block test set used for eval
BASELINE_DIR="data/test_post/bert_regression"
FT_DIR="data/test_post/bert_regression_finetuned"

# --- tee all output to a timestamped log so progress is watchable / recoverable ---
# PYTHONUNBUFFERED so prints stream to the log line-by-line (not in big buffered
# bursts) — otherwise a frozen-looking log can hide a process that's actually busy.
export PYTHONUNBUFFERED=1
mkdir -p logs
LOG="logs/finetune_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
echo "[run] logging to ${LOG}  (tail -f to watch)"

# === 1. Training personas (skip if already built) ===========================
if [[ ! -f "${PERSONAS}" ]]; then
    PYTHONPATH=src python -m utils.tools.build_finetune_personas --n "${N_TRAIN}" --out "${PERSONAS}"
fi

# === 2. Generate training posts (chunked + resumable; iter_10 prompt) ========
# With CHUNK_SIZE>0 the generator appends each chunk and skips agent_ids already
# in TRAIN_POSTS, so a re-run resumes instead of restarting.
# PYTHONPATH=src python -m utils.create_data.generate_test_data \
#     --instruction-file "${PROMPT}" \
#     --persona-phq9-file "${PERSONAS}" \
#     --model "${GEN_MODEL}" --num_agents "${N_TRAIN}" --check_point 10 --first-n \
#     --chunk-size "${CHUNK_SIZE}" \
#     --output-csv "${TRAIN_POSTS}"

# === 3. Fine-tune each regressor on the training posts ======================
PYTHONPATH=src python -m utils.prompt_optimizer \
    --mode bert --model "${REG_MODEL}" --seeds ${SEEDS} \
    --posts-file "${TRAIN_POSTS}" \
    --init-from-dir "${BASELINE_DIR}" --bert-out-dir "${FT_DIR}" \
    --learning-rate "${LR}" --epochs "${EPOCHS}"

# === 4. Build the TEST_N-block test set (reuse existing 120, generate the rest) =
# 4a. pick the extra test personas (eval personas not in training, after the first 120)
if [[ "${TEST_N}" -gt "${EXISTING_TEST_N}" && ! -f "${TEST_EXTRA_PERSONAS}" ]]; then
    PYTHONPATH=src python -m utils.tools.build_test_personas \
        --n "${TEST_N}" --keep "${EXISTING_TEST_N}" --out "${TEST_EXTRA_PERSONAS}"
fi
# 4b. generate posts for the extra personas (chunked + resumable)
if [[ "${TEST_N}" -gt "${EXISTING_TEST_N}" ]]; then
    PYTHONPATH=src python -m utils.create_data.generate_test_data \
        --instruction-file "${PROMPT}" \
        --persona-phq9-file "${TEST_EXTRA_PERSONAS}" \
        --model "${GEN_MODEL}" --num_agents "$((TEST_N - EXISTING_TEST_N))" \
        --check_point 10 --first-n \
        --chunk-size "${CHUNK_SIZE}" \
        --output-csv "${TEST_EXTRA_POSTS}"
fi
# 4c. combine existing 120 + extra into one test file (offset extra agent_ids to avoid collision)
if [[ "${TEST_N}" -gt "${EXISTING_TEST_N}" ]]; then
    PYTHONPATH=src python - "${EXISTING_TEST_POSTS}" "${TEST_EXTRA_POSTS}" "${TEST_POSTS}" "${EXISTING_TEST_N}" <<'PY'
import sys, pandas as pd
existing, extra, out, offset = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
a = pd.read_csv(existing)
b = pd.read_csv(extra)
b["agent_id"] = b["agent_id"].astype(int) + offset      # 0..N -> offset.. (no collision with existing)
pd.concat([a, b], ignore_index=True).to_csv(out, index=False)
print(f"[test-set] combined {a['agent_id'].nunique()} + {b['agent_id'].nunique()} "
      f"= {a['agent_id'].nunique() + b['agent_id'].nunique()} blocks -> {out}")
PY
else
    cp "${EXISTING_TEST_POSTS}" "${TEST_POSTS}"
fi

# === 5. Evaluate baseline vs fine-tuned on the test set =====================
PYTHONPATH=src python -m utils.prompt_optimizer \
    --mode bert-eval --model "${REG_MODEL}" --seeds ${SEEDS} \
    --posts-file "${TEST_POSTS}" --regressor-dir "${BASELINE_DIR}" \
    --bert-eval-out-dir "${BASELINE_DIR}/eval_baseline"

PYTHONPATH=src python -m utils.prompt_optimizer \
    --mode bert-eval --model "${REG_MODEL}" --seeds ${SEEDS} \
    --posts-file "${TEST_POSTS}" --regressor-dir "${FT_DIR}" \
    --bert-eval-out-dir "${FT_DIR}/eval_finetuned"

echo ""
echo "DONE. Compare MAE:"
echo "  baseline : ${BASELINE_DIR}/eval_baseline/aggregate.csv"
echo "  finetuned: ${FT_DIR}/eval_finetuned/aggregate.csv"
