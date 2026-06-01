#!/usr/bin/env bash
# Build the two estimator-comparison figures (MAE + signed bias, with error bars):
#   fig1 — BERT regressor: non-finetuned vs fine-tuned vs in-distribution (synthetic) test
#   fig2 — robustness to distribution shift: BERT vs the post-assessment prompt
# Reads the per-sample test_raw_scores.csv / seed<seed>.csv files already on disk
# (no GPU / model load needed) and writes PNGs + prints the underlying table.
#
# Run from the repo root with the venv activated.
set -euo pipefail

# ============================== CONFIG ======================================
OUT_DIR="data/test_post/method_comparison"
BERT_DIR="data/test_post/bert_regression"               # {MODEL}_seed*/ (synthetic) + eval_baseline/ (human-opt)
BERT_FT_DIR="data/test_post/bert_regression_finetuned"  # eval_finetuned/ (human-opt)
PROMPT_DIR="data/test_post/optimized_phq9"              # {MODEL}_seed*/ (synthetic) + eval-on-prompt subdir (human-opt)
PROMPT_EVAL_SUBDIR="eval_on_prompt_Qwen_Qwen3.5-27B"
BERT_SEEDS="34 35 36 37 38"
PROMPT_SEEDS="23 24 25 32 33"

export PYTHONUNBUFFERED=1
PYTHONPATH=src python -m utils.visualization \
    --out-dir "${OUT_DIR}" \
    --bert-dir "${BERT_DIR}" \
    --bert-ft-dir "${BERT_FT_DIR}" \
    --prompt-dir "${PROMPT_DIR}" \
    --prompt-eval-subdir "${PROMPT_EVAL_SUBDIR}" \
    --bert-seeds ${BERT_SEEDS} \
    --prompt-seeds ${PROMPT_SEEDS}

echo ""
echo "DONE. Figures:"
echo "  ${OUT_DIR}/fig1_bert_finetune.png"
echo "  ${OUT_DIR}/fig2_bert_vs_prompt_robustness.png"
