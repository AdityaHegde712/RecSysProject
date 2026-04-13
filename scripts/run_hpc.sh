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
# HotelRec — SJSU CoE HPC Experiment Runner (GMF only)
#
# FIRST TIME SETUP (run on the login node — has internet):
#   bash scripts/run_hpc.sh setup
#
# Then submit experiments:
#   sbatch scripts/run_hpc.sh            # full pipeline
#   sbatch scripts/run_hpc.sh preprocess # preprocess only
#   sbatch scripts/run_hpc.sh rec        # train GMF
#   sbatch scripts/run_hpc.sh eval       # evaluate GMF
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
echo "HotelRec — HPC Job (ItemKNN baseline)"
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
        echo "  bash scripts/run_hpc.sh setup"
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
    echo "  WARNING: PyTorch check failed. Showing error:"
    python -c "import torch; print(torch.__version__)" 2>&1 | head -5
    echo "  Check that venv was set up correctly: bash scripts/run_hpc.sh setup"
}
    echo "-----------------------------"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────
# STEP 0: One-time setup (run on LOGIN NODE — has internet)
# ─────────────────────────────────────────────────────────────────────
setup_environment() {
    echo ""
    echo ">>> One-time environment setup"
    echo ">>> Run this on the LOGIN NODE (has internet access)"
    echo ""

    module load python3 2>/dev/null || true

    echo "Python: $(python3 --version 2>&1)"
    echo "Node:   $(hostname)"
    echo ""

    # Create venv
    if [ ! -d "${VENV_DIR}" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "${VENV_DIR}"
    fi

    source "${VENV_DIR}/bin/activate"
    echo "Activated venv: $(which python)"

    # Upgrade pip first — system pip may not handle manylinux2014 properly
    echo "Upgrading pip..."
    python -m pip install --upgrade pip setuptools wheel

    echo ""
    echo "Installing dependencies..."
    echo ""

    # Download all wheels first (binary only, no source builds).
    # This is critical on SJSU HPC — no GCC on login node, no internet
    # on GPU nodes. We download everything here and install from cache.
    WHEEL_DIR="${PROJECT_DIR}/.wheels"
    mkdir -p "$WHEEL_DIR"

    echo "  Downloading binary wheels to .wheels/ ..."

    # PyTorch with CUDA — large (~2GB) but only downloaded once.
    # Using CUDA 12.1 wheels which are compatible with SJSU HPC.
    pip download --only-binary=:all: --dest "$WHEEL_DIR" \
        torch==2.2.2 \
        --index-url https://download.pytorch.org/whl/cu121

    # Other dependencies — all have manylinux2014 wheels
    pip download --only-binary=:all: --dest "$WHEEL_DIR" \
        numpy==1.26.4 scipy==1.13.1 pandas==2.2.2 \
        scikit-learn==1.4.2 matplotlib==3.9.2 \
        pyyaml==6.0.1 tqdm==4.66.5 Pillow==10.4.0

    echo ""
    echo "  Installing from downloaded wheels..."

    # Install PyTorch first (from PyTorch index)
    pip install --no-index --find-links="$WHEEL_DIR" \
        torch==2.2.2

    # Install everything else
    pip install --no-index --find-links="$WHEEL_DIR" \
        numpy==1.26.4 scipy==1.13.1 pandas==2.2.2 \
        scikit-learn==1.4.2 matplotlib==3.9.2 \
        pyyaml==6.0.1 tqdm==4.66.5 Pillow==10.4.0

    # pyarrow for parquet support — try binary wheel, skip if unavailable
    pip install pyarrow 2>/dev/null || echo "  (pyarrow install skipped — will use pandas CSV fallback)"

    # Install project in editable mode
    pip install -e . 2>/dev/null || echo "  (editable install skipped)"

    # Download dataset
    echo ""
    echo "Downloading HotelRec dataset..."
    bash scripts/download_data.sh sample || echo "  (dataset download skipped — run manually)"

    # Verify environment
    echo ""
    echo "Verifying environment..."
    python scripts/verify_env.py || true

    echo ""
    echo "============================================"
    echo "Setup complete. Next steps:"
    echo "  python scripts/verify_env.py         # re-verify anytime"
    echo "  python scripts/validate_pipeline.py  # test full pipeline"
    echo "  sbatch scripts/run_hpc.sh            # submit full pipeline"
    echo "============================================"
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
# STEP 2: Training — GMF
# ─────────────────────────────────────────────────────────────────────
train_rec() {
    echo ""
    echo ">>> Training ItemKNN (kcore=${KCORE})..."
    python -m src.train --config configs/itemknn.yaml --kcore "$KCORE"
    echo "ItemKNN training done."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 3: Evaluation
# ─────────────────────────────────────────────────────────────────────
run_eval() {
    echo ""
    echo ">>> Evaluating ItemKNN (kcore=${KCORE})..."
    python -m src.evaluate --config configs/itemknn.yaml --kcore "$KCORE"
    echo "Evaluation done."
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
    eval)
        activate_env
        run_eval
        ;;
    all)
        activate_env
        run_preprocess
        train_rec
        run_eval
        ;;
    *)
        echo "Usage:"
        echo "  bash scripts/run_hpc.sh setup       # first time (login node)"
        echo "  sbatch scripts/run_hpc.sh            # full pipeline (GPU node)"
        echo "  sbatch scripts/run_hpc.sh preprocess # preprocess only"
        echo "  sbatch scripts/run_hpc.sh rec        # train GMF"
        echo "  sbatch scripts/run_hpc.sh eval       # evaluate GMF"
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
