#!/usr/bin/env bash
# Score the LLM PHQ-9 prompts (minimal vs TextGrad-optimized) on the SAME held-out
# blocks the non-finetuned MentalBERT+MLP regressor was tested on, so all three
# estimators give a per-PHQ-9 bias curve over identical in-distribution data.
#
# Pipeline (student model only — no teacher / optimizer is loaded):
#   0. reconstruct the regressor's seed-<SEED> held-out test blocks -> a CSV
#      (deterministic replay of the agent-level 80/10/10 split; CPU-only)
#   1. dump the minimal_post instruction (the one wired into format_config.py)
#   2. score the minimal prompt on those blocks
#   3. score each TextGrad-optimized prompt (per seed) on the same blocks
#
# BERT's own per-PHQ-9 result on these blocks is already on disk:
#   data/test_post/bert_regression/Qwen3.5-27B_seed<SEED>/test_scores_phq9.csv
#
# Run from the repo root on a GPU session:  bash scripts/assessment/run_phq9_on_bert_testset.sh
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# ============================== CONFIG ======================================
PYTHON="${PYTHON:-.venv_vllm/bin/python}"   # interpreter with vLLM + torch (override via env)
MODEL="Qwen/Qwen3.5-27B"
MODEL_SHORT="Qwen3.5-27B"
BERT_SEED=35                                 # deployed regressor seed (its 10% holdout = the test set)
TEXTGRAD_SEEDS="23 24 25 32 33"              # optimized-instruction seeds in optimized_phq9/
MINIMAL_SEED=23                              # which optimized_phq9 seed dir hosts minimal_instruction.txt
                                             # (prompt is seed-independent; one run suffices)
MINIMAL_PROMPTS_JSON="data/prompts_post_minimal.json"

# Paths (derived)
CACHE="data/test/Qwen/${MODEL_SHORT}/mentalbert_embeddings/embeddings_and_labels.pt"
BERT_DIR="data/test_post/bert_regression"
TESTSET="${BERT_DIR}/test_blocks_seed${BERT_SEED}.csv"
# Minimal is scored on the SAME blocks but via a distinctly-named copy of the
# posts file, so its eval subdir (named from the posts-file stem) does not collide
# with the optimized seed-${MINIMAL_SEED} run, which also writes eval_on_test_blocks_seed${BERT_SEED}/.
MINIMAL_TESTSET="${BERT_DIR}/test_blocks_seed${BERT_SEED}_minimal.csv"
OPT_DIR="data/test_post/optimized_phq9"
MINIMAL_INSTR="${OPT_DIR}/${MODEL_SHORT}_seed${MINIMAL_SEED}/minimal_instruction.txt"
EVAL_SUBDIR="eval_on_test_blocks_seed${BERT_SEED}"            # optimized: rerun names it from posts-file stem
MINIMAL_EVAL_SUBDIR="eval_on_test_blocks_seed${BERT_SEED}_minimal"  # minimal: distinct stem -> no overwrite

# --- tee all output to a timestamped log (watchable / recoverable) ----------
export PYTHONUNBUFFERED=1
mkdir -p logs
LOG="logs/phq9_on_bert_testset_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
echo "[run] logging to ${LOG}  (tail -f to watch)"

# === 0. Reconstruct the regressor's held-out test blocks ====================
# Pull the regressor's recorded n_test so the rebuild is asserted against it.
EXPECT_N=$(PYTHONPATH=src "${PYTHON}" - "${BERT_DIR}/${MODEL_SHORT}_seed${BERT_SEED}/performance.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["n_test"])
PY
)
echo "[0] reconstructing seed-${BERT_SEED} test blocks (expect n_test=${EXPECT_N}) -> ${TESTSET}"
PYTHONPATH=src "${PYTHON}" -m utils.tools.build_bert_testset \
    --seed "${BERT_SEED}" --cache "${CACHE}" --out "${TESTSET}" --expect-n "${EXPECT_N}"
# Same blocks, distinct filename -> distinct minimal eval subdir (see note above).
cp -f "${TESTSET}" "${MINIMAL_TESTSET}"

# === 1. Dump the minimal_post instruction ===================================
echo "[1] writing minimal instruction -> ${MINIMAL_INSTR}"
mkdir -p "$(dirname "${MINIMAL_INSTR}")"
PYTHONPATH=src "${PYTHON}" - "${MINIMAL_PROMPTS_JSON}" "${MINIMAL_INSTR}" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(dst, "w", encoding="utf-8") as fh:
    fh.write(json.load(open(src))["phq9"]["system_instruction"])
print(f"  wrote {dst}")
PY

# === 2. Minimal prompt on the held-out blocks ===============================
echo "[2] scoring MINIMAL prompt on ${TESTSET}"
PYTHONPATH=src "${PYTHON}" -m utils.prompt_optimizer --mode phq9-rerun-test \
    --model "${MODEL}" --seeds "${MINIMAL_SEED}" \
    --instruction-filename minimal_instruction.txt \
    --posts-file "${MINIMAL_TESTSET}"

# === 3. Optimized prompts on the same blocks (per seed) =====================
# echo "[3] scoring OPTIMIZED prompts on ${TESTSET}  (seeds: ${TEXTGRAD_SEEDS})"
# PYTHONPATH=src "${PYTHON}" -m utils.prompt_optimizer --mode phq9-rerun-test \
#     --model "${MODEL}" --seeds ${TEXTGRAD_SEEDS} \
#     --instruction-filename optimized_instruction.txt \
#     --posts-file "${TESTSET}"

echo ""
echo "DONE. Per-PHQ-9 (avg_mae, avg_bias, std_bias) on the seed-${BERT_SEED} held-out set:"
echo "  minimal  : ${OPT_DIR}/${MODEL_SHORT}_seed${MINIMAL_SEED}/${MINIMAL_EVAL_SUBDIR}/test_scores_phq9.csv"
for s in ${TEXTGRAD_SEEDS}; do
  echo "  optimized: ${OPT_DIR}/${MODEL_SHORT}_seed${s}/${EVAL_SUBDIR}/test_scores_phq9.csv"
done
echo "  bert     : ${BERT_DIR}/${MODEL_SHORT}_seed${BERT_SEED}/test_scores_phq9.csv  (already on disk)"
