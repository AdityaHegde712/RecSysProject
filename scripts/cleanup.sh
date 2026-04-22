#!/bin/bash
# Clean up old venv/wheels and reset conda env for a fresh setup.
#
# Usage:
#   bash scripts/cleanup.sh          # remove old artifacts + conda env
#   bash scripts/cleanup.sh --all    # also remove processed data and logs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================"
echo "HotelRec — Cleanup"
echo "============================================"
echo ""

# ── Remove old venv (no longer used) ─────────────────────────────────
if [ -d "${PROJECT_DIR}/venv" ]; then
    echo "Removing old venv..."
    rm -rf "${PROJECT_DIR}/venv"
    echo "  Done."
else
    echo "[OK] No old venv found."
fi

# ── Remove pip wheel cache ───────────────────────────────────────────
if [ -d "${PROJECT_DIR}/.wheels" ]; then
    echo "Removing .wheels cache..."
    rm -rf "${PROJECT_DIR}/.wheels"
    echo "  Done."
else
    echo "[OK] No .wheels cache found."
fi

# ── Remove old conda installations ───────────────────────────────────
for conda_dir in "$HOME/miniconda3" "$HOME/mambaforge"; do
    if [ -d "$conda_dir" ]; then
        echo "Removing $conda_dir..."
        rm -rf "$conda_dir"
        echo "  Done."
    fi
done

# ── Optional: remove data and logs ───────────────────────────────────
if [ "${1:-}" = "--all" ]; then
    echo ""
    echo "Removing processed data..."
    rm -rf "${PROJECT_DIR}/data/processed"
    echo "  Done."

    echo "Removing SLURM logs..."
    rm -f "${PROJECT_DIR}"/logs/slurm_*.out "${PROJECT_DIR}"/logs/slurm_*.err
    echo "  Done."

    echo "Removing results..."
    rm -rf "${PROJECT_DIR}/results"
    echo "  Done."
fi

echo ""
echo "============================================"
echo "Cleanup complete. To set up fresh:"
echo "  bash scripts/setup_env.sh"
echo "============================================"
