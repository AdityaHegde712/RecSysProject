#!/bin/bash
# HPC convenience aliases for HotelRec (GMF baseline)
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

# ─── Activate venv helper (CentOS 7 default is Python 2.7) ──────────
_hpa_python() {
    if [ -f "venv/bin/python" ]; then
        venv/bin/python "$@"
    else
        python3 "$@"
    fi
}

# ─── Quick submit ────────────────────────────────────────────────────
alias hpa-setup='bash scripts/run_hpc.sh setup'
alias hpa-verify='_hpa_python scripts/verify_env.py'
alias hpa-validate='_hpa_python scripts/validate_pipeline.py'
alias hpa-run='sbatch scripts/run_hpc.sh'
alias hpa-preprocess='sbatch scripts/run_hpc.sh preprocess'
alias hpa-rec='sbatch scripts/run_hpc.sh rec'
alias hpa-eval='sbatch scripts/run_hpc.sh eval'
alias hpa-explore='_hpa_python scripts/explore_data.py --data_dir data/raw'
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
alias checkpoints='ls -la results/gmf/best_model.pt results/gmf/last_model.pt 2>/dev/null'

# ─── GPU node interactive session ────────────────────────────────────
alias gpunode='srun -p gpu --gres=gpu -n 1 -N 1 -c 4 --mem=16G --time=01:00:00 --pty /bin/bash'

# ─── Cluster info ────────────────────────────────────────────────────
alias nodes='sinfo -N -l'
alias gpus='sinfo -p gpu -N -l'
alias quota='df -h /home/$USER'

echo "HotelRec HPC aliases loaded. Commands:"
echo ""
echo "  Setup & Data:"
echo "    hpa-setup              — one-time environment setup"
echo "    hpa-download           — download full HotelRec dataset"
echo "    hpa-download-sample    — download small sample for testing"
echo "    hpa-explore            — print dataset statistics"
echo ""
echo "  Training:"
echo "    hpa-run           — submit full pipeline (preprocess + train + eval)"
echo "    hpa-preprocess    — preprocess raw data"
echo "    hpa-rec           — train GMF"
echo ""
echo "  Evaluation:"
echo "    hpa-eval          — evaluate GMF on test set"
echo ""
echo "  Utilities:"
echo "    hpa-verify        — verify all deps are working"
echo "    hpa-validate      — test full pipeline (dry run)"
echo "    jobs / myjobs     — check job status"
echo "    killall           — cancel all your jobs"
echo "    lastlog / lasterr — tail latest log/error"
echo "    clearlogs         — delete all log files"
echo "    cleandata         — delete processed data (force re-preprocess)"
echo "    gpunode           — get interactive GPU session"
