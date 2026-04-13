#!/bin/bash
# HPC convenience aliases for HotelRec (ItemKNN baseline)
#
# Source this file on the HPC login node:
#   source scripts/hpc_aliases.sh
#
# Or add to your ~/.bashrc:
#   echo 'source ~/HotelRec-HPA/scripts/hpc_aliases.sh' >> ~/.bashrc

# ─── Job management ──────────────────────────────────────────────────
alias jobs='squeue -u $USER'
alias myjobs='squeue -u $USER --format="%.8i %.20j %.8T %.10M %.6D %R"'
alias killall='scancel -u $USER'

# ─── Activate venv (auto-activates if not already active) ────────────
_hpa_activate() {
    if [ -z "$VIRTUAL_ENV" ]; then
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
        elif [ -f ".venv/bin/activate" ]; then
            source .venv/bin/activate
        fi
    fi
}

_hpa_run() {
    _hpa_activate
    python "$@"
}

# ─── Quick submit ────────────────────────────────────────────────────
alias hpa-setup='bash scripts/run_hpc.sh setup'
alias hpa-verify='_hpa_run scripts/verify_env.py'
alias hpa-validate='_hpa_run scripts/validate_pipeline.py'
alias hpa-run='sbatch scripts/run_hpc.sh'
alias hpa-preprocess='_hpa_activate && python -m src.data.preprocess --kcore 20 --config configs/data.yaml'
alias hpa-split='_hpa_activate && python -m src.data.split --kcore 20 --config configs/data.yaml'
alias hpa-train='_hpa_activate && python -m src.train --config configs/itemknn.yaml --kcore 20'
alias hpa-eval='_hpa_activate && python -m src.evaluate --config configs/itemknn.yaml --kcore 20'
alias hpa-explore='_hpa_run scripts/explore_data.py --data_dir data/raw'
alias hpa-download='bash scripts/download_data.sh full'
alias hpa-download-sample='bash scripts/download_data.sh sample'

# ─── Log viewing ─────────────────────────────────────────────────────
alias lastlog='ls -t logs/slurm_*.out 2>/dev/null | head -1 | xargs tail -f'
alias lasterr='ls -t logs/slurm_*.err 2>/dev/null | head -1 | xargs tail -f'
alias alllogs='ls -lt logs/slurm_*.out 2>/dev/null | head -10'
alias clearlogs='rm -f logs/slurm_*.out logs/slurm_*.err && echo "Logs cleared"'

# ─── Cleanup ─────────────────────────────────────────────────────────
alias cleandata='rm -rf data/processed && echo "Processed data deleted. Will re-preprocess on next run."'
alias cleanvenv='rm -rf venv && echo "venv deleted. Run hpa-setup to recreate."'
alias cleanall='rm -rf venv .wheels logs/slurm_* data/processed && echo "Cleaned venv, wheels, logs, and processed data"'

# ─── Results ─────────────────────────────────────────────────────────
alias results='ls -la results/ 2>/dev/null'
alias checkpoints='ls -la results/itemknn/ 2>/dev/null'

# ─── Interactive session ─────────────────────────────────────────────
alias cpunode='srun -n 1 -N 1 -c 4 --mem=16G --time=01:00:00 --pty /bin/bash'
alias gpunode='srun -p gpu --gres=gpu -n 1 -N 1 -c 4 --mem=16G --time=01:00:00 --pty /bin/bash'

# ─── Cluster info ────────────────────────────────────────────────────
alias nodes='sinfo -N -l'
alias gpus='sinfo -p gpu -N -l'
alias quota='df -h /home/$USER'

echo "HotelRec HPC aliases loaded (venv auto-activates). Commands:"
echo ""
echo "  Setup & Data:"
echo "    hpa-setup              — one-time environment setup"
echo "    hpa-download           — download full HotelRec dataset"
echo "    hpa-download-sample    — download small sample for testing"
echo "    hpa-explore            — print dataset statistics"
echo ""
echo "  Pipeline (auto-activates venv):"
echo "    hpa-preprocess    — k-core filter raw data → parquet"
echo "    hpa-split         — split into train/val/test"
echo "    hpa-train         — fit ItemKNN on training data"
echo "    hpa-eval          — evaluate ItemKNN on test set"
echo "    hpa-run           — submit full pipeline via SLURM"
echo ""
echo "  Utilities:"
echo "    hpa-verify        — verify all deps are working"
echo "    hpa-validate      — test full pipeline (dry run)"
echo "    jobs / myjobs     — check job status"
echo "    killall           — cancel all your jobs"
echo "    lastlog / lasterr — tail latest log/error"
echo "    clearlogs         — delete all log files"
echo "    cleandata         — delete processed data (force re-preprocess)"
echo "    cpunode           — get interactive CPU session (ItemKNN doesn't need GPU)"
echo "    gpunode           — get interactive GPU session (for neural variants)"
