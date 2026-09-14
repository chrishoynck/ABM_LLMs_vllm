#!/usr/bin/env bash
# Per-seed lexical-entrainment trajectories (MentalBERT, PCA) for the four calibrated
# debiased topologies, no LLM. Writes the per-setting overlay, the per-seed grid and
# the SDA+SDC shared map under plots/lexical_entrainment/. Embeddings are cached per
# seed; figures on disk are skipped (OVERWRITE=1 redraws). Options: REDUCTION=umap,
# EMBEDDING=sbert. Driver: src/utils/analyses/lexical_entrainment/global/plot_lexical_entrainment.py.
#   bash scripts/plotting/run_lexical_entrainment.sh

set -euo pipefail

# This script lives at scripts/plotting/, so the repo root is two directories up.
# All paths below are relative to that root.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# Project venv (has sentence-transformers / torch; the system python does not).
PY=".venv_vllm/bin/python"

GRID_ROOT="${GRID_ROOT:-data/networks_post/basis}"
REDUCTION="${REDUCTION:-pca}"      # pca (default) or umap
NUM_STEPS="${NUM_STEPS:-35}"       # sliding-window size (tweets)
SHIFT="${SHIFT:-5}"                # sliding-window stride

# Embedding model: mentalbert (default) or sbert.
EMB_FLAG=(--mentalbert)
[[ "${EMBEDDING:-mentalbert}" == "sbert" ]] && EMB_FLAG=(--sbert)

OVERWRITE_FLAG=()
[[ "${OVERWRITE:-0}" == "1" ]] && OVERWRITE_FLAG=(--overwrite)

echo "========================================================"
echo "Lexical-entrainment grid (per-seed MentalBERT trajectories)"
echo "  grid root  : $GRID_ROOT"
echo "  reduction  : $REDUCTION   embedding: ${EMBEDDING:-mentalbert}"
echo "  num_steps  : $NUM_STEPS   shift: $SHIFT"
echo "========================================================"

PYTHONPATH=src "$PY" -m utils.analyses.lexical_entrainment.global.plot_lexical_entrainment \
    --scan       "$GRID_ROOT" \
    --reduction  "$REDUCTION" \
    --num_steps  "$NUM_STEPS" \
    --shift      "$SHIFT" \
    "${EMB_FLAG[@]}" \
    "${OVERWRITE_FLAG[@]}"

echo
echo "========================================================"
echo "Done. Figures under plots/lexical_entrainment/."
echo "========================================================"
