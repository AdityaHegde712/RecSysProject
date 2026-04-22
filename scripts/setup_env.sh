#!/bin/bash
# Set up Python environment on SJSU HPC.
# Run this on the login node (it has internet, GPU nodes don't).
#
#   bash scripts/setup_env.sh
#
# What this does:
#   1. Creates a venv
#   2. Downloads pip wheels (login node has internet, GPU nodes don't)
#   3. Installs torch, numpy, pandas, etc from cached wheels
#   4. Installs sentence-transformers for TextNCF text encoding
#
# The tricky part: HPC runs CentOS 7 with GLIBC 2.17. Most pip wheels
# are manylinux2014 which works fine. But sentencepiece (a dep of
# sentence-transformers) needs GLIBCXX_3.4.29 which doesn't exist here.
# Luckily all-MiniLM-L6-v2 uses WordPiece (from the tokenizers package),
# not sentencepiece, so we just skip it.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"

echo "============================================"
echo "HotelRec environment setup"
echo "============================================"
echo "Project: $PROJECT_DIR"
echo ""

# clean up old conda stuff if it's lying around
for old in "$HOME/miniconda3" "$HOME/mambaforge"; do
    [ -d "$old" ] && echo "Removing leftover $old..." && rm -rf "$old"
done

module load python3 2>/dev/null || true

# create or recreate venv
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/pip" ]; then
    echo "venv already exists"
else
    [ -d "$VENV_DIR" ] && rm -rf "$VENV_DIR"
    echo "Creating venv..."
    python3 -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"
echo "Python: $(python --version)"

python -m pip install --upgrade pip setuptools wheel

# download wheels to .wheels/ so GPU nodes (no internet) can install
WHEEL_DIR="${PROJECT_DIR}/.wheels"
mkdir -p "$WHEEL_DIR"

echo ""
echo "Downloading wheels..."

# torch with CUDA 12.1
pip download --only-binary=:all: --dest "$WHEEL_DIR" \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121 || {
    echo "FAILED: torch download"; exit 1
}

# scientific stack
pip download --only-binary=:all: --dest "$WHEEL_DIR" \
    numpy==1.26.4 scipy==1.13.1 pandas==2.2.2 \
    scikit-learn==1.4.2 matplotlib==3.9.2 \
    pyyaml==6.0.1 tqdm==4.66.5 Pillow==10.4.0 || {
    echo "FAILED: scientific stack download"; exit 1
}

# pyarrow (optional, for parquet)
pip download --only-binary=:all: --dest "$WHEEL_DIR" \
    pyarrow==15.0.2 2>/dev/null || echo "  pyarrow download failed, will use CSV"

# sentence-transformers deps (skip sentencepiece — not needed for MiniLM)
pip download --only-binary=:all: --dest "$WHEEL_DIR" \
    tokenizers==0.15.2 huggingface-hub==0.21.4 \
    transformers==4.38.2 safetensors==0.4.2 regex || {
    echo "FAILED: sentence-transformers deps download"; exit 1
}

# sentence-transformers itself (pure python)
pip download --only-binary=:all: --dest "$WHEEL_DIR" \
    sentence-transformers==2.5.1 2>/dev/null || \
pip download --dest "$WHEEL_DIR" --no-deps \
    sentence-transformers==2.5.1 || {
    echo "FAILED: sentence-transformers download"; exit 1
}

echo ""
echo "Installing from cached wheels..."

pip install --no-index --find-links="$WHEEL_DIR" torch==2.2.2 || {
    echo "FAILED: torch install"; exit 1
}

pip install --no-index --find-links="$WHEEL_DIR" \
    numpy==1.26.4 scipy==1.13.1 pandas==2.2.2 \
    scikit-learn==1.4.2 matplotlib==3.9.2 \
    pyyaml==6.0.1 tqdm==4.66.5 Pillow==10.4.0 || {
    echo "FAILED: scientific stack install"; exit 1
}

pip install --no-index --find-links="$WHEEL_DIR" pyarrow==15.0.2 2>/dev/null || \
    echo "  pyarrow skipped (CSV fallback)"

pip install --no-index --find-links="$WHEEL_DIR" \
    tokenizers==0.15.2 huggingface-hub==0.21.4 \
    safetensors==0.4.2 regex || {
    echo "FAILED: tokenizers install"; exit 1
}

pip install --no-index --find-links="$WHEEL_DIR" --no-deps \
    transformers==4.38.2 || { echo "FAILED: transformers"; exit 1; }

pip install --no-index --find-links="$WHEEL_DIR" --no-deps \
    sentence-transformers==2.5.1 || { echo "FAILED: sentence-transformers"; exit 1; }

echo ""
echo "Checking imports..."
python -c "
import torch, numpy, pandas, scipy
from sentence_transformers import SentenceTransformer
print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
m = SentenceTransformer('all-MiniLM-L6-v2')
e = m.encode(['test'])
print(f'sentence-transformers OK, dim={e.shape[1]}')
print('All good.')
" || {
    echo "Import check failed — see error above"
    exit 1
}

echo ""
echo "============================================"
echo "Done. Run:"
echo "  source scripts/hpc_aliases.sh"
echo "  hpc-encode        # encode reviews"
echo "  hpc-train-ncf     # train TextNCF"
echo "============================================"
