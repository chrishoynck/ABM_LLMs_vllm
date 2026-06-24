#!/usr/bin/env bash
# Full ABM simulation — calibrated SDA, undirected, debiased, HAPPY hub, 5 seeds.
# (run_simulation2.sh is the SDC counterpart with the same settings.)
#
# Optimal SDC network parameters (sa_network --net sdc, averaged_best.csv, N=200, 5 seeds):
#   alpha=4.9429  stub_gamma=1.6187  degree=8.2539  dim=3  n_clusters=2
#   latent_weight=18.3813  age_weight=2.2095
#   Achieved: mean_degree=5.67  gamma=2.13  ks=0.105  C=0.208  phq9_assort=0.075
#             lcc=0.868  (mean_loss=0.437)
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
SEEDS=(14 15 16 17 18)   # 5 seeds for multi-run calibration
NET="sda"                # sdc = stub-matched scale-free; sda = SocialDistanceAttachment

# ── Optimal SDC topology parameters (sa_network --net sdc, averaged_best.csv) ─
# Achieved: mean_degree=5.67  gamma=2.13  ks=0.105  C=0.208  phq9_assort=0.075  lcc=0.868
# ALPHA=4.9429
# STUB_GAMMA=1.6187        # power-law exponent for the SDC stub degree sequence
# DEGREE=10 #8.2539            # target mean fed in; realized mean degree ≈ 5.67 (stub shortfall)
# DIM=3
# N_CLUSTERS=2
# LATENT_WEIGHT=18.3813
# AGE_WEIGHT=2.2095


# High PHQ-0 SDC variant
# ALPHA=8
# STUB_GAMMA=1.6187        # power-law exponent for the SDC stub degree sequence
# DEGREE=8.2539            # target mean fed in; realized mean degree ≈ 5.67 (stub shortfall)
# DIM=2
# N_CLUSTERS=2
# LATENT_WEIGHT=2
# AGE_WEIGHT=2.2095


# # NET="sda" (low degree)
DEGREE=4.5
ALPHA=2.1655
LATENT_WEIGHT=7.9839 #1
DIM=5       #3
AGE_WEIGHT=2.3149
N_CLUSTERS=2
STUB_GAMMA=2.5          # ignored for sda (sdc-only); kept bound so the shared run command works
INIT_0=0               # 1 = start every agent at PHQ-9 0; 0 = sample the real distribution

# high phq-9
# DEGREE=4.5
# ALPHA=1.1655
# LATENT_WEIGHT=1.0
# DIM=3
# AGE_WEIGHT=2.3149
# N_CLUSTERS=2
# STUB_GAMMA=2.5          # ignored for sda (sdc-only); kept bound so the shared run command works
# INIT_0=0               # 1 = start every agent at PHQ-9 0; 0 = sample the real distribution
# DIRECTED=0             # 1 = directed SDA graph (asymmetric draw, no symmetrisation); 0 = undirected


# # SDA directed k=3 — directed counterpart of the undirected 1_1655_d3_dim5 set.
# # Same topology params as "SDA 2026-06-14 active", only DIRECTED flipped on.
# # To run: comment the "high phq-9" block above and uncomment this one.
# # Note: k=3 here is the OUT-degree, so the projected graph is ~2x denser than
# # the undirected k=3 (set DEGREE=1.5 to match undirected density instead).
# # Rebuilt stats (5 seeds): clustering_out≈0.036  reciprocity≈0.06  lcc≈1.00
# NET="sda"
# DEGREE=4.5
# ALPHA=2.1655
# LATENT_WEIGHT=1   #7.9839
# DIM=3   #5
# AGE_WEIGHT=2.3149
# N_CLUSTERS=2
# STUB_GAMMA=2.5          # ignored for sda (sdc-only); kept bound so the shared run command works
INIT_0=0               # 1 = start every agent at PHQ-9 0; 0 = sample the real distribution
DIRECTED=0             # undirected graph -> saved under data/networks_post/happy/sda/undirected/...


# ── Previous parameter sets (kept for reference) ─────────────────────────────
# SDA 2026-06-14 active : NET=sda DEGREE=3  ALPHA=1.1655  LATENT_WEIGHT=7.9839   DIM=5  AGE_WEIGHT=2.3149  N_CLUSTERS=2
# SDA 2026-06-10        : NET=sda DEGREE=3  ALPHA=2.1655  LATENT_WEIGHT=7.9839   DIM=5  AGE_WEIGHT=2.3149  N_CLUSTERS=2
# SDA C-prioritised     : NET=sda DEGREE=3  ALPHA=3.4312  LATENT_WEIGHT=19.9048  DIM=4  AGE_WEIGHT=2.5259  N_CLUSTERS=2  (C=0.1301 age=0.4189 phq9=0.0337)
# SDA loss-opt          : NET=sda DEGREE=3  ALPHA=3.6772  LATENT_WEIGHT=16.0962  DIM=5  AGE_WEIGHT=2.7192  N_CLUSTERS=2  (C=0.1383 age=0.2685 phq9=0.0224 loss=0.7963)


# Checkpointing / saving
CHECK_POINT=10          # PHQ-9 update cadence (every 10 rounds)
LOG=20                  # save network state snapshot every 30 iterations

# PHQ-9 bias correction (opt-in; logged to meta.json + net properties). Pick one:
#   "none"   -> uncorrected (raw regressor output)        -> saved under non_debiased/
#   full fit / interior fit -> a phq9_bias_table_*.csv exported from experiment.ipynb
#                                                         -> saved under debiased/
# The debiased/ vs non_debiased/ split (PathManager) keeps these runs from
# overwriting each other at the same network parameters.
_REG_DIR="data/test_post/bert_regression_finetuned/Qwen3.5-27B_seed35"

# BIAS_TABLE="none"                                          # uncorrected
BIAS_TABLE="${_REG_DIR}/phq9_bias_table_fullfit.csv"         # full-corrected (full plotted line)
# BIAS_TABLE="${_REG_DIR}/phq9_bias_table_interiorfit.csv"   # non-fully corrected (ends excluded)

# --init_phq9_zero is a boolean (store_true) flag: include it only when INIT_0=1.
INIT_ARGS=()
(( INIT_0 )) && INIT_ARGS+=(--init_phq9_zero)

# --directed is a boolean (store_true) flag: include it only when DIRECTED=1.
# Default to 0 so older config blocks that predate this flag still run undirected.
DIRECTED_ARGS=()
(( ${DIRECTED:-0} )) && DIRECTED_ARGS+=(--directed)

# ── Run ───────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Starting full simulation"
echo "  net           : $NET"
echo "  rounds        : $ROUNDS"
echo "  agents        : $NUM_AGENTS"
echo "  degree        : $DEGREE  (target fed in; SDC realized ≈ 5.67)"
echo "  stub_gamma    : $STUB_GAMMA"
echo "  seeds         : ${SEEDS[*]}"
echo "  alpha         : $ALPHA"
echo "  latent_weight : $LATENT_WEIGHT"
echo "  dim           : $DIM"
echo "  age_weight    : $AGE_WEIGHT"
echo "  n_clusters    : $N_CLUSTERS"
echo "  directed      : ${DIRECTED:-0}"
echo "  happy         : on  (hub persona = data/happy_persona.csv, PHQ-9 pinned at 0)"
echo "  check_point   : $CHECK_POINT  (PHQ-9 update cadence)"
echo "  log every     : $LOG iterations"
echo "  bias_table    : $BIAS_TABLE"
echo "  init_phq9_0   : $INIT_0  (1 = all agents start at PHQ-9 0)"
echo "  cds_dynamic   : off"
echo "  GPUs          : $(nvidia-smi --query-gpu=name --format=noheader 2>/dev/null | head -2 || echo 'N/A')"
echo "========================================================"

LLAMA_ID="Qwen/Qwen3.5-27B" PYTHONPATH=src python src/llama_activate.py "$NET" \
    --rounds        "$ROUNDS" \
    --num_agents    "$NUM_AGENTS" \
    --degree        "$DEGREE" \
    --stub_gamma    "$STUB_GAMMA" \
    --seeds         "${SEEDS[@]}" \
    --alpha         "$ALPHA" \
    --latent_weight "$LATENT_WEIGHT" \
    --dim           "$DIM" \
    --age_weight    "$AGE_WEIGHT" \
    --n_clusters    "$N_CLUSTERS" \
    --check_point   "$CHECK_POINT" \
    --log           "$LOG" \
    "${INIT_ARGS[@]}" \
    "${DIRECTED_ARGS[@]}" \
    --happy \
    --no-cds_dynamic \
    --phq9_mode bert \
    --bias_table_path "$BIAS_TABLE" \
    --time_info \
    --save

echo "========================================================"
echo "Simulation complete."
echo "========================================================"
