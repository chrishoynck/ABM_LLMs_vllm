#!/usr/bin/env bash
# Precompute the per-level PHQ-9 bias table the simulation subtracts.
#
# Pipeline (all reusing existing tooling — no bespoke generators):
#   1. generate balanced, UNSEEN calibration blocks   (generate_test_data.py)
#   2. score them with the DEPLOYED BERT+MLP regressor (prompt_optimizer --mode
#      bert-eval, same flag path run_finetune.sh uses)
#   3. aggregate to the 28-level table                 (utils.tools.phq9_bias)
#
# Output: phq9_bias_table.csv next to the regressor — the simulation loads it
# automatically (network.py), no flag.
#
# Run interactively or from a SLURM job:  bash run_bias_calibration.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ── config ──────────────────────────────────────────────────────────────────
GEN_MODEL="qwen27"                     # model that writes the posts (matches finetune data)
REG_MODEL="Qwen/Qwen3.5-27B"           # names the regressor dirs
SEED=35                                # deployed regressor seed — MUST match the
                                       # bert_regressor_path default in llama_activate.py
PER_CLASS=50                           # blocks per PHQ-9 level (28 levels). 50 -> 1400 blocks.
CHUNK_SIZE=100                         # chunked + resumable generation

FT_DIR="data/test_post/bert_regression_finetuned"
REG_SUBDIR="${FT_DIR}/Qwen3.5-27B_seed${SEED}"
PROMPT="data/prompt_optimization_h/qwen27_baseline/iter_10/prompt.txt"
POOL="data/finetune/personas_unseen_pool.csv"
POSTS="data/finetune/calibration_posts.csv"
EVAL_DIR="${FT_DIR}/eval_calibration"
TABLE_OUT="${REG_SUBDIR}/phq9_bias_table.csv"

NUM_AGENTS=$(( PER_CLASS * 28 ))

echo "========================================================"
echo "PHQ-9 bias-table calibration"
echo "  regressor   : ${REG_SUBDIR}/regressor.pt"
echo "  blocks      : ${NUM_AGENTS}  (${PER_CLASS}/level x 28)"
echo "  posts       : ${POSTS}"
echo "  table out   : ${TABLE_OUT}"
echo "========================================================"

# ── 0. unseen persona pool (full pool minus train/eval/corpus personas) ──────
if [[ ! -f "${POOL}" ]]; then
    echo "[calib] building unseen persona pool -> ${POOL}"
    PYTHONPATH=src python - <<'PY'
import pandas as pd, glob
seen = set(pd.read_csv('data/personas_finetune_phq9.csv')['persona'].astype(str)) \
     | set(pd.read_csv('data/personas_eval_1000_phq9.csv')['persona'].astype(str))
for p in glob.glob('data/test_post/Qwen_Qwen3.5-27B/**/*.csv', recursive=True):
    try:
        d = pd.read_csv(p)
    except Exception:
        continue
    if 'persona' in d.columns:
        seen |= set(d['persona'].astype(str))
pool = pd.read_csv('data/personas_short_10k.csv')
mask = ~pool['persona'].astype(str).isin(seen)
pool[mask].to_csv('data/finetune/personas_unseen_pool.csv', index=False)
print(f"[calib] {len(pool)} pool -> {int(mask.sum())} unseen personas")
PY
fi

# ── 1. generate balanced, unseen blocks (random persona sample, forced balance) ─
PYTHONPATH=src python -m utils.create_data.generate_test_data \
    --instruction-file "${PROMPT}" \
    --persona-phq9-file "${POOL}" \
    --model "${GEN_MODEL}" --num_agents "${NUM_AGENTS}" --check_point 10 \
    --phq9-band-range 0 27 \
    --chunk-size "${CHUNK_SIZE}" \
    --output-csv "${POSTS}"

# ── 2. score with the deployed BERT+MLP (same flag path as run_finetune.sh) ──
PYTHONPATH=src python -m utils.prompt_optimizer \
    --mode bert-eval --model "${REG_MODEL}" --seeds "${SEED}" \
    --posts-file "${POSTS}" \
    --regressor-dir "${FT_DIR}" \
    --bert-eval-out-dir "${EVAL_DIR}"

# ── 3. aggregate to the 28-level table next to the regressor ─────────────────
PYTHONPATH=src python -m utils.tools.phq9_bias \
    --scores "${EVAL_DIR}/seed${SEED}.csv" \
    --out "${TABLE_OUT}"

echo "========================================================"
echo "DONE. Bias table -> ${TABLE_OUT}"
echo "The simulation (phq9_mode=bert) loads it automatically."
echo "========================================================"
