#!/usr/bin/env bash
# Produce the per-prompt data for figure 2 (plot_eval_bert_vs_prompt / Fig 5.11),
# scoring BOTH prompts on the SAME test sets so all three method groups are
# comparable per distribution:
#
#   synthetic (in-distribution) : the deterministic per-seed 1200-block split
#                                   optimized -> test_raw_scores.csv (already on disk)
#                                   minimal   -> minimal_synth/
#   human-opt  (shifted)        : the 300-block set data/finetune/test_posts.csv
#                                   (the SAME blocks BERT's eval_baseline uses)
#                                   optimized -> eval_on_human300/
#                                   minimal   -> minimal_human300/
#
# So on the human-opt side BERT (eval_baseline), the optimized prompt and the
# minimal prompt are all scored on the identical 300 blocks -> a clean paired
# comparison. (Synthetic stays on the prompts' 1200-block split; BERT keeps its
# own holdout there, since it trained on ~80% of those blocks.)
#
# Everything is written to NON-clobbering subdirs (via --result-subdir), so the
# optimized prompt's test_raw_scores.csv / training_trajectory.csv are untouched.
# The minimal prompt text is seed-independent (data/prompts_post_minimal.json ->
# phq9.system_instruction); we still score it per seed for matching structure.
#
# Pipeline (student model only — no teacher / optimizer is loaded). GPU session:
#   bash scripts/assessment/run_minimal_shift.sh
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# ============================== CONFIG ======================================
PYTHON="${PYTHON:-.venv_vllm/bin/python}"   # interpreter with vLLM + torch (override via env)
MODEL="Qwen/Qwen3.5-27B"
MODEL_SHORT="Qwen3.5-27B"
PROMPT_SEEDS="${PROMPT_SEEDS:-23 24 25 32 33}"   # same seeds the optimized group uses
MINIMAL_PROMPTS_JSON="data/prompts_post_minimal.json"
SHIFTED_POSTS="data/finetune/test_posts.csv"     # 300-block human-opt set (== BERT eval_baseline blocks)

OPT_DIR="data/test_post/optimized_phq9"
SYNTH_SUBDIR="minimal_synth"        # must match _EVAL_MINIMAL_SYNTH_SUBDIR in utils/visualization.py
HUMAN_SUBDIR="minimal_human300"     # must match _EVAL_MINIMAL_HUMAN_SUBDIR in utils/visualization.py
OPT_HUMAN_SUBDIR="eval_on_human300" # must match --prompt-eval-subdir default in utils/visualization.py

# --- tee all output to a timestamped log ------------------------------------
export PYTHONUNBUFFERED=1
mkdir -p logs
LOG="logs/minimal_shift_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
echo "[run] logging to ${LOG}  (tail -f to watch)"

# === 0. Make sure each seed dir has the minimal instruction =================
# (the prompt is seed-independent; rerun_test_phq9 loads it from <seed_dir>/<name>)
for s in ${PROMPT_SEEDS}; do
  instr="${OPT_DIR}/${MODEL_SHORT}_seed${s}/minimal_instruction.txt"
  mkdir -p "$(dirname "${instr}")"
  PYTHONPATH=src "${PYTHON}" - "${MINIMAL_PROMPTS_JSON}" "${instr}" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(dst, "w", encoding="utf-8") as fh:
    fh.write(json.load(open(src))["phq9"]["system_instruction"])
print(f"  wrote {dst}")
PY
done

# === 1. Minimal prompt on the synthetic split (in-distribution) =============
echo "[1] scoring MINIMAL prompt on the per-seed synthetic test split -> ${SYNTH_SUBDIR}/"
PYTHONPATH=src "${PYTHON}" -m utils.prompt_optimizer --mode phq9-rerun-test \
    --model "${MODEL}" --seeds ${PROMPT_SEEDS} \
    --instruction-filename minimal_instruction.txt \
    --result-subdir "${SYNTH_SUBDIR}"

# === 2. Minimal prompt on the 300-block human-opt set =======================
echo "[2] scoring MINIMAL prompt on ${SHIFTED_POSTS} -> ${HUMAN_SUBDIR}/"
PYTHONPATH=src "${PYTHON}" -m utils.prompt_optimizer --mode phq9-rerun-test \
    --model "${MODEL}" --seeds ${PROMPT_SEEDS} \
    --instruction-filename minimal_instruction.txt \
    --posts-file "${SHIFTED_POSTS}" \
    --result-subdir "${HUMAN_SUBDIR}"

# === 2b. Optimized prompt on the SAME 300-block human-opt set ===============
# The figure now reads the optimized human-opt bar from this subdir, so it must
# be scored on the same 300 blocks (distinct subdir -> does not clobber the
# existing 120-block eval_on_prompt_Qwen_Qwen3.5-27B/).
echo "[2b] scoring OPTIMIZED prompt on ${SHIFTED_POSTS} -> ${OPT_HUMAN_SUBDIR}/"
PYTHONPATH=src "${PYTHON}" -m utils.prompt_optimizer --mode phq9-rerun-test \
    --model "${MODEL}" --seeds ${PROMPT_SEEDS} \
    --instruction-filename optimized_instruction.txt \
    --posts-file "${SHIFTED_POSTS}" \
    --result-subdir "${OPT_HUMAN_SUBDIR}"

# === 3. Regenerate the comparison figures (now with the Minimal prompt group) =
# (CPU-only — just reads the per-sample CSVs. Mirrors run_eval_comparison.sh
#  defaults but uses ${PYTHON} so it works without activating the venv.)
echo "[3] rebuilding comparison figures"
PYTHONPATH=src "${PYTHON}" -m utils.visualization \
    --out-dir data/test_post/method_comparison \
    --prompt-dir "${OPT_DIR}" \
    --prompt-seeds ${PROMPT_SEEDS}

echo ""
echo "DONE. Per-seed scores under:"
for s in ${PROMPT_SEEDS}; do
  echo "  ${OPT_DIR}/${MODEL_SHORT}_seed${s}/${SYNTH_SUBDIR}/test_raw_scores.csv        (minimal, synthetic 1200)"
  echo "  ${OPT_DIR}/${MODEL_SHORT}_seed${s}/${HUMAN_SUBDIR}/test_raw_scores.csv     (minimal, human-opt 300)"
  echo "  ${OPT_DIR}/${MODEL_SHORT}_seed${s}/${OPT_HUMAN_SUBDIR}/test_raw_scores.csv      (optimized, human-opt 300)"
done
echo "Figure: data/test_post/method_comparison/fig2_bert_vs_prompt_robustness.png"
