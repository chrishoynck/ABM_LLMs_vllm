#!/usr/bin/env bash
# Regenerate the network-evolution figures for already-built runs — no LLM, no
# notebook. Two phases, both driven off a scan of GRID_ROOT so they cover the
# same universe of runs (no hand-maintained config list to keep in sync):
#
#   1. Per-seed figures (CDS-evolution graph + critical-slowing-down dot grid +
#      the 10-panel PHQ-9 network sequence, identical to experiment.ipynb's
#      vis.print_subnetworks_phq9) for every saved run under GRID_ROOT, written
#      into each run's own plots/ folder, plus a per-PHQ-9-band CDS validation
#      table to stdout. CDS is recomputed from the saved tweet text with the
#      validated detector (the stored distortion flags are all-False for basis
#      runs).
#   2. Per-combination grids: one combined 4-row figure per parameter
#      combination (mean PHQ-9 + assortativity lines on top, then SD / lag-1
#      autocorrelation / PHQ-9 dot-grids, one column per seed, shared colourbar
#      per row), written into each combo's plots/ folder.
#
# Both phases scan GRID_ROOT for fully-finished runs (rounds=$ROUNDS) and skip
# the old_debiased / old_pop / different_debias_settings sub-trees. Both the
# standard non_debiased/ AND the debiased/ runs are kept (see EXCLUDE below). To
# widen or narrow the scope, edit ROUNDS / GRID_ROOT / EXCLUDE below or pass the
# driver's --exclude / --scan_rounds / --grid_rounds flags directly.
#
# Figures are written idempotently: any figure already on disk is skipped (per
# figure, not per run), so re-running only fills in what's missing — e.g. adding
# the network sequence to runs that already have the other figures. Set
# OVERWRITE=1 to force a full redraw.
#
# See src/utils/network_evolution.py and src/utils/tools/plot_network_evolution.py.
#
#   bash run_plot_evolution.sh                # skip figures already on disk
#   OVERWRITE=1 bash run_plot_evolution.sh    # redraw everything

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Pass --overwrite through to the driver when OVERWRITE=1.
OVERWRITE_FLAG=()
[[ "${OVERWRITE:-0}" == "1" ]] && OVERWRITE_FLAG=(--overwrite)

# No-LLM script: use the project venv directly (system python lacks
# networkx/seaborn/SALib — see the network-sa-python-env note).
PY=".venv_vllm/bin/python"

# ── Shared run parameters (match run_simulation.sh) ─────────────────────────
ROUNDS=300              # only fully-finished runs are plotted (rounds filter)
CHECK_POINT=10          # PHQ-9 update cadence — must match the simulated run
CSD_WINDOW=8            # critical-slowing-down rolling window, in PHQ-9 updates
GRID_ROOT="data/networks_post/basis"   # both phases scan here for runs/combos

# Sub-trees skipped in BOTH phases. debiased/ is now INCLUDED — only the old /
# alternative runs are skipped (non_debiased/ and debiased/ are both kept). Set
# EXCLUDE=() to plot every sub-tree, including the old ones.
EXCLUDE=(old_debiased old_pop different_debias_settings)
EXCLUDE_FLAG=(--exclude)
[[ ${#EXCLUDE[@]} -gt 0 ]] && EXCLUDE_FLAG+=("${EXCLUDE[@]}")

echo "========================================================"
echo "Plotting network evolution for saved runs"
echo "  rounds      : $ROUNDS   check_point: $CHECK_POINT   csd_window: $CSD_WINDOW"
echo "  grid root   : $GRID_ROOT"
echo "========================================================"

# ── Phase 1: per-seed figures for every saved run under GRID_ROOT ────────────
# Scans the same universe as phase 2 (rounds=$ROUNDS, old_pop/old_debiased/… skipped),
# so every run that gets a combo grid also gets its per-seed figures, including
# the 10-panel PHQ-9 network sequence. Self-maintaining: new parameter sets are
# picked up automatically — no CONFIGS list to keep in sync. Directed vs
# undirected is read from each saved net.json, so both sub-trees are covered.
echo
echo "### Phase 1: per-seed figures"
PYTHONPATH=src "$PY" -m utils.tools.plot_network_evolution \
    --scan        "$GRID_ROOT" \
    --scan_rounds "$ROUNDS" \
    --check_point "$CHECK_POINT" \
    --csd_window  "$CSD_WINDOW" \
    "${EXCLUDE_FLAG[@]}" \
    "${OVERWRITE_FLAG[@]}"

# ── Phase 2: per-combination grids (all rounds=$ROUNDS combos under GRID_ROOT) ─
echo
echo "### Phase 2: per-combination grids"
PYTHONPATH=src "$PY" -m utils.tools.plot_network_evolution \
    --grid \
    --scan        "$GRID_ROOT" \
    --grid_rounds "$ROUNDS" \
    --check_point "$CHECK_POINT" \
    --csd_window  "$CSD_WINDOW" \
    "${EXCLUDE_FLAG[@]}" \
    "${OVERWRITE_FLAG[@]}"

echo
echo "========================================================"
echo "All done (per-seed figures + per-combination grids)."
echo "========================================================"
