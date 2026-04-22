#!/bin/bash
# HPC convenience aliases for HotelRec
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
_hpc_activate() {
    if [ -z "$VIRTUAL_ENV" ]; then
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
        else
            echo "ERROR: venv not found. Run: bash scripts/setup_env.sh"
            return 1
        fi
    fi
}

_hpc_run() {
    _hpc_activate
    python "$@"
}

# ─── Setup ────────────────────────────────────────────────────────────
alias hpc-setup='bash scripts/setup_env.sh'
alias hpc-cleanup='bash scripts/cleanup.sh'
alias hpc-verify='_hpc_run scripts/verify_env.py'
alias hpc-validate='_hpc_run scripts/validate_pipeline.py'

# ─── Data ─────────────────────────────────────────────────────────────
alias hpc-download='bash scripts/download_data.sh full'
alias hpc-download-sample='bash scripts/download_data.sh sample'
alias hpc-explore='_hpc_run scripts/explore_data.py --data_dir data/raw'
alias hpc-preprocess='_hpc_activate && python -m src.data.preprocess --kcore 20 --config configs/data.yaml'

# ─── Baselines ────────────────────────────────────────────────────────
alias hpc-baselines='_hpc_activate && python -m src.run_baselines --kcore 20'
alias hpc-run='sbatch scripts/run_hpc.sh'

# ─── TextNCF variant ─────────────────────────────────────────────────
alias hpc-encode='_hpc_activate && python scripts/encode_text.py --kcore 20'
alias hpc-train-ncf='_hpc_activate && python -m src.train_text_ncf --config configs/text_ncf.yaml --kcore 20'
alias hpc-train-mt='_hpc_activate && python -m src.train_text_ncf_mt --config configs/text_ncf_mt.yaml --kcore 20'
alias hpc-train-subrating='_hpc_activate && python -m src.train_text_ncf_subrating --config configs/text_ncf_subrating.yaml --kcore 20'
alias hpc-ensemble='_hpc_activate && python -m src.evaluate_ensemble --kcore 20'
alias hpc-two-stage='_hpc_activate && python -m src.evaluate_two_stage --kcore 20'
alias hpc-run-ncf='sbatch scripts/run_hpc.sh text-ncf'
alias hpc-run-all='sbatch scripts/run_hpc.sh run-all'

# ─── Log viewing ─────────────────────────────────────────────────────
alias lastlog='ls -t logs/slurm_*.out 2>/dev/null | head -1 | xargs tail -f'
alias lasterr='ls -t logs/slurm_*.err 2>/dev/null | head -1 | xargs tail -f'
alias alllogs='ls -lt logs/slurm_*.out 2>/dev/null | head -10'
alias clearlogs='rm -f logs/slurm_*.out logs/slurm_*.err && echo "Logs cleared"'

# ─── Cleanup ─────────────────────────────────────────────────────────
alias cleandata='rm -rf data/processed && echo "Processed data deleted."'
alias cleanvenv='rm -rf venv && echo "venv deleted. Run hpc-setup to recreate."'
alias cleanall='rm -rf venv .wheels logs/slurm_* data/processed && echo "Cleaned everything."'

# ─── Results ─────────────────────────────────────────────────────────
alias results='ls -la results/ 2>/dev/null'

# ─── Interactive session ─────────────────────────────────────────────
alias cpunode='srun -n 1 -N 1 -c 4 --mem=16G --time=01:00:00 --pty /bin/bash'
alias gpunode='srun -p gpu --gres=gpu -n 1 -N 1 -c 4 --mem=16G --time=01:00:00 --pty /bin/bash'

# ─── Cluster info ────────────────────────────────────────────────────
alias nodes='sinfo -N -l'
alias gpus='sinfo -p gpu -N -l'
alias quota='df -h /home/$USER'

echo "HotelRec aliases loaded. Commands:"
echo ""
echo "  Setup:"
echo "    hpc-setup         — one-time environment setup"
echo "    hpc-cleanup       — remove venv + old artifacts"
echo "    hpc-download      — download full dataset"
echo ""
echo "  Baselines:"
echo "    hpc-preprocess    — k-core filter raw data"
echo "    hpc-baselines     — run all baselines"
echo "    hpc-run           — submit baseline pipeline via SLURM"
echo ""
echo "  TextNCF (Pramod's variant):"
echo "    hpc-encode        — encode reviews → sentence embeddings"
echo "    hpc-train-ncf     — train base TextNCF"
echo "    hpc-train-mt      — train multi-task TextNCF (BPR + rating)"
echo "    hpc-train-subrating — train sub-rating TextNCF (per-aspect)"
echo "    hpc-ensemble      — ensemble scoring (TextNCF + GMF + ItemKNN)"
echo "    hpc-two-stage     — two-stage retrieval (ItemKNN → TextNCF)"
echo "    hpc-run-ncf       — submit TextNCF pipeline via SLURM"
echo "    hpc-run-all       — submit baselines + TextNCF via SLURM"
echo ""
echo "  Utilities:"
echo "    lastlog / lasterr — tail latest SLURM log"
echo "    cpunode / gpunode — get interactive session"
