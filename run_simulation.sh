#!/usr/bin/env bash
# Full ABM simulation — optimal SDA parameters, 5 seeds.
#
# Optimal network parameters (from sa_network multi-seed calibration):
#   alpha=3.6772  latent_weight=16.0962  dim=5  age_weight=2.7192  n_clusters=2
#   (C=0.1383  age_assort=0.2685  phq9_assort=0.0224  mean_loss=0.7963)
#
# Called via run_data_simulation.job (SLURM resources allocated there).
# Can also run interactively:  bash run_simulation.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# ── Silence watchdog ───────────────────────────────────────────────────────
# Alerts to stderr if the SLURM output file goes unmodified for 5 minutes.
_watchdog() {
    local outfile="${HOME}/slurm_output_${SLURM_JOB_ID:-0}.out"
    local silence=300   # seconds
    while true; do
        sleep 60
        if [[ -f "$outfile" ]]; then
            local age=$(( $(date +%s) - $(stat -c %Y "$outfile") ))
            if (( age >= silence )); then
                echo "[WATCHDOG $(date '+%H:%M:%S')] WARNING: no output for ${age}s — possible hang." >&2
            fi
        fi
    done
}
_watchdog &
WATCHDOG_PID=$!
trap 'kill "$WATCHDOG_PID" 2>/dev/null || true' EXIT

# ── Simulation parameters ──────────────────────────────────────────────────
ROUNDS=300
NUM_AGENTS=100
DEGREE=6
SEEDS=(14 15 16 17 18)   # 5 seeds for multi-run calibration

# Optimal SDA topology parameters
ALPHA=2.1655
LATENT_WEIGHT=7.9839
DIM=5
AGE_WEIGHT=2.3149
N_CLUSTERS=2            # match calibration (N_CLUSTERS_FIXED=2 in sa_network)

# Checkpointing / saving
CHECK_POINT=10          # PHQ-9 update cadence (every 10 rounds)
LOG=20                  # save network state snapshot every 30 iterations

# ── Run ───────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Starting full simulation"
echo "  rounds        : $ROUNDS"
echo "  agents        : $NUM_AGENTS"
echo "  degree        : $DEGREE"
echo "  seeds         : ${SEEDS[*]}"
echo "  alpha         : $ALPHA"
echo "  latent_weight : $LATENT_WEIGHT"
echo "  dim           : $DIM"
echo "  age_weight    : $AGE_WEIGHT"
echo "  n_clusters    : $N_CLUSTERS"
echo "  check_point   : $CHECK_POINT  (PHQ-9 update cadence)"
echo "  log every     : $LOG iterations"
echo "  cds_dynamic   : off"
echo "  GPUs          : $(nvidia-smi --query-gpu=name --format=noheader 2>/dev/null | head -2 || echo 'N/A')"
echo "========================================================"

LLAMA_ID="Qwen/Qwen3.5-27B" PYTHONPATH=src python src/llama_activate.py sda \
    --rounds        "$ROUNDS" \
    --num_agents    "$NUM_AGENTS" \
    --degree        "$DEGREE" \
    --seeds         "${SEEDS[@]}" \
    --alpha         "$ALPHA" \
    --latent_weight "$LATENT_WEIGHT" \
    --dim           "$DIM" \
    --age_weight    "$AGE_WEIGHT" \
    --n_clusters    "$N_CLUSTERS" \
    --check_point   "$CHECK_POINT" \
    --log           "$LOG" \
    --no-cds_dynamic \
    --phq9_mode bert \
    --time_info \
    --save

echo "========================================================"
echo "Simulation complete."
echo "========================================================"
