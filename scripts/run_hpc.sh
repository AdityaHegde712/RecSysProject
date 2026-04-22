#!/bin/bash
#SBATCH --job-name=hotelrec
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#
# Email notifications (override with: sbatch --mail-user=other@sjsu.edu ...)
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=pramod.yadav@sjsu.edu
#
# HotelRec — SJSU CoE HPC Experiment Runner
#
# FIRST TIME SETUP (run on the login node — has internet):
#   bash scripts/setup_env.sh
#
# Then submit experiments:
#   sbatch scripts/run_hpc.sh            # full baseline pipeline
#   sbatch scripts/run_hpc.sh text-ncf   # full TextNCF pipeline
#   sbatch scripts/run_hpc.sh run-all    # both back-to-back
#   sbatch scripts/run_hpc.sh preprocess # preprocess only
#   sbatch scripts/run_hpc.sh encode     # encode text embeddings
#
# SJSU HPC notes:
#   - Login node: GLIBC 2.17 (CentOS 7), has internet, no GCC
#   - GPU nodes:  GLIBC 2.17, no internet, have GPU
#   - /home is shared across all nodes
#   - Setup downloads pre-built wheels on login node (no compilation)
#   - Batch jobs use the venv created during setup

# NOTE: not using 'set -euo pipefail' because it causes silent failures
# when optional commands fail (e.g., module load). We handle errors
# explicitly instead.
set -o pipefail

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
KCORE=20  # k-core filtering threshold

# ─────────────────────────────────────────────────────────────────────
# PROJECT DIR
# SLURM copies scripts to /var/spool, so BASH_SOURCE won't work.
# Use SLURM_SUBMIT_DIR (set by sbatch to the dir where sbatch was called).
# For direct execution (bash scripts/run_hpc.sh), fall back to pwd detection.
# ─────────────────────────────────────────────────────────────────────
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
elif [ -f "scripts/run_hpc.sh" ]; then
    PROJECT_DIR="$(pwd)"
elif [ -f "run_hpc.sh" ]; then
    PROJECT_DIR="$(cd .. && pwd)"
else
    PROJECT_DIR="$(pwd)"
fi
cd "$PROJECT_DIR"

mkdir -p logs results

VENV_DIR="${PROJECT_DIR}/venv"

echo "============================================"
echo "HotelRec — HPC Job"
echo "============================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURM_NODELIST:-$(hostname)}"
echo "Project dir: $PROJECT_DIR"
echo "Date:        $(date)"
echo "============================================"

# ─────────────────────────────────────────────────────────────────────
# ENVIRONMENT ACTIVATION (used by batch jobs — no internet needed)
# ─────────────────────────────────────────────────────────────────────
activate_env() {
    module load python3 2>/dev/null || true
    module load cuda 2>/dev/null || true
    module load cudnn 2>/dev/null || true

    if [ -f "${VENV_DIR}/bin/activate" ]; then
        source "${VENV_DIR}/bin/activate"
        echo "Activated venv: $(python --version)"
    else
        echo "ERROR: venv not found at ${VENV_DIR}"
        echo "Run setup first on the login node:"
        echo "  bash scripts/setup_env.sh"
        exit 1
    fi

    # project root on PYTHONPATH so 'import src' works
    export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

    # unbuffered Python output so SLURM logs show progress in real time
    export PYTHONUNBUFFERED=1

    # help PyTorch manage GPU memory fragmentation
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

    # GPU check
    echo ""
    echo "--- PyTorch Backend Check ---"
    echo "  Python: $(which python)"
    python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    print(f'  Memory: {mem / 1e9:.1f} GB')
else:
    print('  WARNING: No GPU detected. Training will be slow.')
" || {
    echo "  WARNING: PyTorch check failed"
    python -c "import torch; print(torch.__version__)" 2>&1 | head -5
    echo "  Try: bash scripts/setup_env.sh"
}
    echo "-----------------------------"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────
# STEP 0: One-time setup (run on LOGIN NODE — has internet)
# Delegates to setup_env.sh which handles venv + pip wheels.
# ─────────────────────────────────────────────────────────────────────
setup_environment() {
    echo ""
    echo ">>> Delegating to scripts/setup_env.sh ..."
    echo ""
    bash "${PROJECT_DIR}/scripts/setup_env.sh"
}

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Preprocessing
# ─────────────────────────────────────────────────────────────────────
run_preprocess() {
    echo ""
    echo ">>> Preprocessing HotelRec data (kcore=${KCORE})..."

    # check if already processed
    KCORE_DIR="data/processed/kcore_${KCORE}"
    if [ -d "$KCORE_DIR" ] && [ -f "${KCORE_DIR}/train.parquet" ]; then
        n=$(python -c "import pandas as pd; print(len(pd.read_parquet('${KCORE_DIR}/train.parquet')))" 2>/dev/null || echo "0")
        if [ "$n" -gt 0 ]; then
            echo "[OK] Processed data exists (${n} training rows) — skipping"
            return 0
        fi
    fi

    # check for raw data — use find instead of glob (more reliable)
    echo "  Checking for data in $(pwd)/data/raw/ ..."
    ls -la data/raw/ 2>/dev/null | head -10
    DATA_COUNT=$(find data/raw -maxdepth 1 -type f \( -name "*.zip" -o -name "*.tar.gz" -o -name "*.tgz" -o -name "*.tar.bz2" -o -name "*.json" -o -name "*.txt" \) 2>/dev/null | wc -l | tr -d ' ')
    echo "  Found ${DATA_COUNT} data file(s)"
    if [ "$DATA_COUNT" -eq 0 ]; then
        echo "ERROR: No data found in data/raw/"
        echo "Expected a .zip/.tar.gz archive or .json/.txt files."
        echo "Download the dataset first:"
        echo "  bash scripts/download_data.sh full"
        exit 1
    fi

    python -m src.data.preprocess --kcore "$KCORE" --config configs/data.yaml || {
        echo "ERROR: Preprocessing failed!"
        exit 1
    }

    echo "Preprocessing done."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Training — baselines
# ─────────────────────────────────────────────────────────────────────
train_rec() {
    echo ""
    echo ">>> Running baselines (kcore=${KCORE})..."
    python -m src.run_baselines --kcore "$KCORE" || {
        echo "ERROR: Baseline training failed!"
        exit 1
    }
    echo "Baseline training done."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 3: Encode review text → sentence embeddings (TextNCF)
# ─────────────────────────────────────────────────────────────────────
run_encode() {
    echo ""
    echo ">>> Encoding review text (kcore=${KCORE})..."

    EMB_DIR="data/processed/text_emb"
    if [ -f "${EMB_DIR}/user_text_emb.npy" ] && [ -f "${EMB_DIR}/item_text_emb.npy" ]; then
        echo "[OK] Text embeddings already exist — skipping"
        return 0
    fi

    python scripts/encode_text.py --kcore "$KCORE" || {
        echo "ERROR: Text encoding failed!"
        exit 1
    }
    echo "Text encoding done."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 4: Training — TextNCF variant
# ─────────────────────────────────────────────────────────────────────
EPOCH_FLAG=""  # set by run-sample to override epochs

train_text_ncf() {
    echo ""
    echo ">>> Training TextNCF (kcore=${KCORE})..."
    python -m src.train_text_ncf --config configs/text_ncf.yaml --kcore "$KCORE" $EPOCH_FLAG
    echo "TextNCF training done."
}

train_text_ncf_mt() {
    echo ""
    echo ">>> Training Multi-Task TextNCF (kcore=${KCORE})..."
    python -m src.train_text_ncf_mt --config configs/text_ncf_mt.yaml --kcore "$KCORE" $EPOCH_FLAG
    echo "Multi-Task TextNCF training done."
}

train_text_ncf_subrating() {
    echo ""
    echo ">>> Training Sub-Rating TextNCF (kcore=${KCORE})..."
    python -m src.train_text_ncf_subrating --config configs/text_ncf_subrating.yaml --kcore "$KCORE" $EPOCH_FLAG
    echo "Sub-Rating TextNCF training done."
}

run_ensemble() {
    echo ""
    echo ">>> Ensemble evaluation (kcore=${KCORE})..."
    python -m src.evaluate_ensemble --kcore "$KCORE"
    echo "Ensemble evaluation done."
}

run_two_stage() {
    echo ""
    echo ">>> Two-stage evaluation (kcore=${KCORE})..."
    python -m src.evaluate_two_stage --kcore "$KCORE"
    echo "Two-stage evaluation done."
}

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
MODE="${1:-all}"

case "$MODE" in
    setup)
        # setup runs on login node (has internet) — NOT via sbatch
        setup_environment
        ;;
    preprocess)
        activate_env
        run_preprocess
        ;;
    rec)
        activate_env
        train_rec
        ;;
    encode)
        activate_env
        run_encode
        ;;
    train-ncf)
        activate_env
        train_text_ncf
        ;;
    text-ncf)
        # full TextNCF pipeline: preprocess → encode → train
        activate_env
        run_preprocess
        run_encode
        train_text_ncf
        ;;
    train-mt)
        activate_env
        train_text_ncf_mt
        ;;
    train-subrating)
        activate_env
        train_text_ncf_subrating
        ;;
    ensemble)
        activate_env
        run_ensemble
        ;;
    two-stage)
        activate_env
        run_two_stage
        ;;
    run-all)
        # everything: preprocess → baselines → encode → all TextNCF variants → eval
        activate_env
        run_preprocess
        train_rec
        run_encode
        echo ""
        echo "=========================================="
        echo "  Training all TextNCF variants"
        echo "=========================================="
        train_text_ncf
        train_text_ncf_mt
        train_text_ncf_subrating
        echo ""
        echo "=========================================="
        echo "  Running ensemble + two-stage evaluation"
        echo "=========================================="
        run_ensemble
        run_two_stage
        echo ""
        echo "=========================================="
        echo "  ALL DONE — Results:"
        echo "    results/text_ncf/test_metrics.json"
        echo "    results/text_ncf_mt/test_metrics.json"
        echo "    results/text_ncf_subrating/test_metrics.json"
        echo "    results/text_ncf/ensemble_metrics.json"
        echo "    results/text_ncf/two_stage_metrics.json"
        echo "=========================================="
        ;;
        run-sample)
            # smoke test: full pipeline with 2 epochs — verifies everything works
            # before committing to a long training run
            EPOCH_FLAG="--epochs 2"
            activate_env
            run_preprocess
            run_encode
            echo ""
            echo "=========================================="
            echo "  SMOKE TEST — 2 epochs per variant"
            echo "=========================================="
            train_text_ncf
            train_text_ncf_mt
            train_text_ncf_subrating
            run_ensemble
            run_two_stage
            echo ""
            echo "=========================================="
            echo "  Smoke test done. Check results/ for output."
            echo "  If everything looks good, run:"
            echo "    sbatch scripts/run_hpc.sh run-all"
            echo "=========================================="
            ;;
        all)
            activate_env
            run_preprocess
            train_rec
            ;;
        *)
            echo "Usage:"
            echo "  bash scripts/setup_env.sh            # first time (login node)"
            echo "  sbatch scripts/run_hpc.sh            # baselines pipeline"
            echo "  sbatch scripts/run_hpc.sh preprocess # preprocess only"
            echo "  sbatch scripts/run_hpc.sh rec        # train baselines"
            echo ""
            echo "  TextNCF variants:"
            echo "  sbatch scripts/run_hpc.sh encode         # encode text → embeddings"
            echo "  sbatch scripts/run_hpc.sh train-ncf      # train base TextNCF"
            echo "  sbatch scripts/run_hpc.sh train-mt       # train multi-task TextNCF"
            echo "  sbatch scripts/run_hpc.sh train-subrating # train sub-rating TextNCF"
            echo "  sbatch scripts/run_hpc.sh ensemble       # ensemble evaluation"
            echo "  sbatch scripts/run_hpc.sh two-stage      # two-stage evaluation"
            echo "  sbatch scripts/run_hpc.sh text-ncf       # full base TextNCF pipeline"
            echo ""
            echo "  Full comparison:"
            echo "  sbatch scripts/run_hpc.sh run-all    # baselines + all TextNCF variants"
            echo "  sbatch scripts/run_hpc.sh run-sample # smoke test (2 epochs, all variants)"
            exit 1
            ;;
    esac

echo ""
echo "============================================"
echo "Job finished at $(date)"
echo ""
echo "Results:"
ls -la results/ 2>/dev/null || echo "  (no results directory)"
echo ""
echo "To copy results to your local machine:"
echo "  scp -r $(whoami)@$(hostname -f 2>/dev/null || hostname):${PROJECT_DIR}/results/ ./results/"
echo "============================================"
