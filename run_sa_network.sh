#!/usr/bin/env bash
# Run the network topology Sobol SA from the terminal.
# Usage:
#   bash run_sa_network.sh [options]
#
# Options (all optional — defaults shown below):
#   --n-sobol   N      Sobol base size; total evals = N×12   (default: 512)
#   --n-jobs    N      Parallel workers; -1 = all CPUs        (default: -1)
#   --out-dir   PATH   Where to write CSVs + PNGs             (default: data/sensitivity/network)
#
# Example — quick run with 128 base samples (~1 500 builds):
#   bash run_sa_network.sh --n-sobol 128
#
# Example — publication run with all CPUs:
#   bash run_sa_network.sh --n-sobol 512 --n-jobs -1

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON="$REPO_DIR/.venv_vllm/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    exit 1
fi

OUT_DIR="data/sensitivity/network"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run_$(date +%Y%m%d_%H%M%S).log"

echo "[sa_network] logging to $LOG"

PYTHONPATH="$REPO_DIR/src" "$PYTHON" -m utils.sensitivity.sa_network \
    --well-being "data/confidential/phq9.sav" \
    --n-sobol    512 \
    --n-agents   200 \
    --degree     6   \
    --seeds      43 44 45 46 47 \
    --n-jobs     ${SLURM_CPUS_PER_TASK:-4} \
    --dist-type  gaussian_clusters \
    --out-dir    "$OUT_DIR" \
    "$@" 2>&1 | tee "$LOG"
