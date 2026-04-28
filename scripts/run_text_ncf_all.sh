#!/bin/bash
# Run all TextNCF variants end-to-end.
#
# Prerequisites:
#   - data/processed/20core/*.parquet       (shared preprocessing)
#   - data/processed/text_emb/*.npy         (encode_text.py)
#   - results/baselines/itemknn.pkl         (scripts/fit_itemknn.py)
#   - results/gmf/best_model.pt             (shared GMF baseline)
#
# Usage:
#   bash scripts/run_text_ncf_all.sh
#
# Runs locally

set -e
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8       # Windows default cp1252 chokes on unicode in prints
mkdir -p logs

run() {
    local name=$1
    shift
    echo
    echo "==================================================================="
    echo "[run] $name"
    echo "==================================================================="
    "$@" 2>&1 | tee "logs/${name}.log"
}

# 1. Base TextNCF
run text_ncf_base \
    python -m src.train_text_ncf --config configs/text_ncf.yaml --kcore 20

# 2. Ablation: GMF-only (no text branch)
python - <<'PY'
import yaml, os
with open("configs/text_ncf.yaml") as f:
    c = yaml.safe_load(f)
c["model"]["use_gmf"] = True
c["model"]["use_text"] = False
c["paths"]["checkpoint_dir"] = "results/text_ncf_gmf_only"
c["paths"]["log_dir"] = "logs/text_ncf_gmf_only"
os.makedirs("configs/_ablations", exist_ok=True)
with open("configs/_ablations/text_ncf_gmf_only.yaml", "w") as f:
    yaml.safe_dump(c, f, sort_keys=False)
PY
run text_ncf_gmf_only \
    python -m src.train_text_ncf \
        --config configs/_ablations/text_ncf_gmf_only.yaml --kcore 20

# 3. Ablation: text-only (no GMF branch)
python - <<'PY'
import yaml, os
with open("configs/text_ncf.yaml") as f:
    c = yaml.safe_load(f)
c["model"]["use_gmf"] = False
c["model"]["use_text"] = True
c["paths"]["checkpoint_dir"] = "results/text_ncf_text_only"
c["paths"]["log_dir"] = "logs/text_ncf_text_only"
with open("configs/_ablations/text_ncf_text_only.yaml", "w") as f:
    yaml.safe_dump(c, f, sort_keys=False)
PY
run text_ncf_text_only \
    python -m src.train_text_ncf \
        --config configs/_ablations/text_ncf_text_only.yaml --kcore 20

# 4. Multi-Task TextNCF
run text_ncf_mt \
    python -m src.train_text_ncf_mt \
        --config configs/text_ncf_mt.yaml --kcore 20

# 5. Sub-Rating TextNCF
run text_ncf_subrating \
    python -m src.train_text_ncf_subrating \
        --config configs/text_ncf_subrating.yaml --kcore 20

# 6. Ensemble (TextNCF + GMF + ItemKNN)
run ensemble \
    python -m src.evaluate_ensemble \
        --text-ncf-ckpt results/text_ncf/best_model.pt \
        --gmf-ckpt results/gmf/best_model.pt \
        --knn-ckpt results/baselines/itemknn.pkl \
        --kcore 20 --grid-step 0.1

# 7. Two-stage (ItemKNN → TextNCF)
run two_stage \
    python -m src.evaluate_two_stage \
        --text-ncf-ckpt results/text_ncf/best_model.pt \
        --knn-ckpt results/baselines/itemknn.pkl \
        --kcore 20 --retrieve-k 200

# 8. Rating metrics (calibrated RMSE/MAE) for all three TextNCF variants
run rating_metrics \
    python scripts/compute_rmse.py --kcore 20 \
        --text-ncf-ckpt results/text_ncf/best_model.pt \
        --text-ncf-config configs/text_ncf.yaml \
        --text-ncf-mt-ckpt results/text_ncf_mt/best_model.pt \
        --text-ncf-mt-config configs/text_ncf_mt.yaml \
        --text-ncf-subrating-ckpt results/text_ncf_subrating/best_model.pt \
        --text-ncf-subrating-config configs/text_ncf_subrating.yaml

echo
echo "All TextNCF runs done."
