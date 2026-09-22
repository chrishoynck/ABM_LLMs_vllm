#!/usr/bin/env bash
# Print the estimator-comparison table (MAE + signed bias, with error bars) that
# backs Tables 1-2 / Results §PHQ-9 Assessment:
#   BERT regressor: non-finetuned vs fine-tuned vs in-distribution (synthetic) test
#   robustness to distribution shift: BERT vs the post-assessment prompt
# Reads the per-sample test_raw_scores.csv / seed<seed>.csv files already on disk
# (no GPU / model load needed). The two companion figures were retired 2026-09-22.
#
# Run from the repo root with the venv activated.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# ============================== CONFIG ======================================
BERT_ARM="teacher"                  # assessor arm for the non-fine-tuned bars
BERT_FT_ARM="qwen27_optimized"      # assessor arm for the fine-tuned bars
CORPUS="qwen27_optimized"           # held-out corpus both are scored on (eval/on_<corpus>/)
PROMPT_DIR="data/test_post/optimized_phq9"              # {MODEL}_seed*/ + eval-on-* subdirs (aligned tests)
PROMPT_EVAL_SUBDIR="eval_on_human300"                   # optimized prompt on the 300-block human-opt set (paired with BERT eval_baseline)
PROMPT_SYNTH_SUBDIR="eval_on_test_blocks_seed35"        # optimized prompt on the BERT test blocks (aligned synthetic, paired with BERT synthetic)
BERT_SEEDS="34 35 36 37 38"
PROMPT_SEEDS="23 24 25 32 33"

export PYTHONUNBUFFERED=1
PYTHONPATH=src python -m utils.visualization eval-comparison \
    --bert-arm "${BERT_ARM}" \
    --bert-ft-arm "${BERT_FT_ARM}" \
    --corpus "${CORPUS}" \
    --prompt-dir "${PROMPT_DIR}" \
    --prompt-eval-subdir "${PROMPT_EVAL_SUBDIR}" \
    --prompt-synth-subdir "${PROMPT_SYNTH_SUBDIR}" \
    --bert-seeds ${BERT_SEEDS} \
    --prompt-seeds ${PROMPT_SEEDS}

echo ""
echo "DONE. Table printed above (Tables 1-2 source); no files written."
