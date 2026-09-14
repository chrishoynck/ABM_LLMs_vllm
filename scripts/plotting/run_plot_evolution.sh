#!/usr/bin/env bash
# Redraw the network-evolution figures for every saved run under GRID_ROOT (no LLM):
# phase 1 writes the per-seed figures (CDS evolution, critical-slowing-down grid,
# PHQ-9 network sequence) into each run's plots/, phase 2 the per-combination grids.
# Figures already on disk are skipped; OVERWRITE=1 redraws everything. Scope is set
# by ROUNDS / GRID_ROOT / EXCLUDE below. Driver: src/utils/tools/plot_network_evolution.py.
#   bash scripts/plotting/run_plot_evolution.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# Pass --overwrite through to the driver when OVERWRITE=1.
OVERWRITE_FLAG=()
[[ "${OVERWRITE:-0}" == "1" ]] && OVERWRITE_FLAG=(--overwrite)

# No-LLM script: use the project venv directly (system python lacks
# networkx/seaborn/SALib, see the network-sa-python-env note).
PY=".venv_vllm/bin/python"

# ── Shared run parameters (match run_simulation_sda.sh) ─────────────────────────
ROUNDS=300              # only fully-finished runs are plotted (rounds filter)
CHECK_POINT=10          # PHQ-9 update cadence — must match the simulated run
CSD_WINDOW=8            # critical-slowing-down rolling window, in PHQ-9 updates
# Settings to plot, one sub-dir of data/networks_post/ per setting. Each is
# scanned independently for runs/combos. Add new settings here (e.g. happy).
SETTINGS=(basis happy)

# Sub-trees skipped in BOTH phases. debiased/ is now INCLUDED, only the old /
# alternative runs are skipped (non_debiased/ and debiased/ are both kept). Set
# EXCLUDE=() to plot every sub-tree, including the old ones.
EXCLUDE=(old_debiased old_pop different_debias_settings)
EXCLUDE_FLAG=(--exclude)
[[ ${#EXCLUDE[@]} -gt 0 ]] && EXCLUDE_FLAG+=("${EXCLUDE[@]}")

for SETTING in "${SETTINGS[@]}"; do
GRID_ROOT="data/networks_post/$SETTING"   # both phases scan here for runs/combos

echo "========================================================"
echo "Plotting network evolution for saved runs"
echo "  setting     : $SETTING"
echo "  rounds      : $ROUNDS   check_point: $CHECK_POINT   csd_window: $CSD_WINDOW"
echo "  grid root   : $GRID_ROOT"
echo "========================================================"

# ── Phase 1: per-seed figures for every saved run under GRID_ROOT ────────────
# Scans the same universe as phase 2 (rounds=$ROUNDS, old_pop/old_debiased/… skipped),
# so every run that gets a combo grid also gets its per-seed figures, including
# the 10-panel PHQ-9 network sequence. Self-maintaining: new parameter sets are
# picked up automatically, no CONFIGS list to keep in sync. Directed vs
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

# ── Phase 3: phase portraits (dw PHQ-9 vs PHQ-9 assortativity) per net type ───
# One 2x2 grid per network type (sda / sdc): rows directed/undirected, cols
# non_debiased/debiased, one line per seed coloured by configuration. Written to
# data/networks_post/basis/<net>/plots/phase_dw_phq9_assort_<net>.png. Same scan
# universe (rounds=$ROUNDS, EXCLUDE applied) as the other phases.
echo
echo "### Phase 3: phase portraits per network type"
PYTHONPATH=src "$PY" -m utils.tools.plot_network_evolution \
    --phase \
    --scan        "$GRID_ROOT" \
    --grid_rounds "$ROUNDS" \
    --check_point "$CHECK_POINT" \
    "${EXCLUDE_FLAG[@]}" \
    "${OVERWRITE_FLAG[@]}"

done

echo
echo "========================================================"
echo "All done (per-seed figures + per-combination grids + phase portraits)."
echo "========================================================"
