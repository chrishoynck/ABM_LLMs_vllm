#!/usr/bin/env bash
# Per-seed lexical-entrainment trajectories on MentalBERT embeddings — no LLM, no
# notebook. For each of the four calibrated, debiased topologies (SDA/SDC x
# directed/undirected) it embeds every seed's tweets with MentalBERT, traces the
# per-window mean-embedding trajectory in a 2D PCA (fit once per setting on its
# pooled seeds), and draws one dot per window coloured by mean PHQ-9 (green->red).
# Runs are NOT averaged.
#
# Writes, under plots/lexical_entrainment/:
#   * per-setting overlay (all seeds, one shared-PCA axes)   -> <net>_<dir>_..._overlay.png
#   * per-seed grid (topology x seed, each panel own PCA)    -> entrainment_grid_perseed_*.png
#   * SDA+SDC shared map per direction (calibrated+high-deg) -> entrainment_shared_sda_sdc_*.png
#
# Tweet embeddings are cached per-seed under plots/lexical_entrainment/cache/, so
# the first run encodes (GPU recommended) and later runs reuse the cache.
# Figures already on disk are skipped; set OVERWRITE=1 to redraw.
#
# See src/utils/analyses/lexical_entrainment/global/plot_lexical_entrainment.py and
# the plot_entrainment_* functions in src/utils/visualization.py.
#
#   bash run_lexical_entrainment.sh                 # default: PCA, MentalBERT
#   OVERWRITE=1 bash run_lexical_entrainment.sh     # redraw everything
#   REDUCTION=umap bash run_lexical_entrainment.sh  # UMAP instead of PCA
#   EMBEDDING=sbert bash run_lexical_entrainment.sh # plain SBERT instead

set -euo pipefail

# This script lives at src/utils/analyses/lexical_entrainment/global/, so the repo
# root is five directories up. All paths below are relative to that root.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
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
