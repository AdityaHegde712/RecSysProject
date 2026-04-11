# Experiment Guide — HotelRec GMF Baseline

Step-by-step instructions for reproducing the GMF baseline results from Antognini & Faltings (LREC 2020). The experiment has three phases: data preparation, training, and evaluation.

---

## Phase 1: Data Preparation

### 1.1 Download the Dataset

The HotelRec data is hosted on the authors' GitHub. The full dump is ~50GB of JSON files (one per hotel).

```bash
# On the login node (has internet):
bash scripts/download_data.sh full       # full dataset (~50GB)
bash scripts/download_data.sh sample     # small sample for testing (~500MB)
```

The download script places files in `data/raw/`. Each JSON file contains an array of review objects for one hotel.

### 1.2 Preprocess and Filter

Convert raw JSON to a clean parquet file, build user/item ID mappings, and apply k-core filtering.

```bash
# 20-core (smaller, faster — start here)
python -m src.data.preprocess --kcore 20 --config configs/data.yaml

# 5-core (larger, closer to paper's main results)
python -m src.data.preprocess --kcore 5 --config configs/data.yaml
```

**Output** (in `data/processed/{kcore}/`):
- `interactions.parquet` — filtered interactions with integer IDs
- `user2id.json` — mapping from original user URLs to integer IDs
- `item2id.json` — mapping from original hotel URLs to integer IDs

**Expected sizes:**

| Subset | Interactions | Parquet Size | Processing Time |
|--------|-------------|-------------|-----------------|
| 20-core | 2.2M | ~200MB | ~5 min |
| 5-core | 21.1M | ~2GB | ~30 min |

### 1.3 Split into Train/Val/Test

```bash
python -m src.data.split --kcore 20 --config configs/data.yaml
python -m src.data.split --kcore 5 --config configs/data.yaml
```

Splits 80/10/10 with seed=42, stratified by user (each user's interactions are split proportionally). Output: `train.parquet`, `val.parquet`, `test.parquet` in the same directory.

### 1.4 Verify the Data

Quick sanity check that everything looks right:

```bash
python -c "
import pandas as pd
for split in ['train', 'val', 'test']:
    df = pd.read_parquet(f'data/processed/20/{split}.parquet')
    print(f'{split}: {len(df):,} rows, {df.user_id.nunique():,} users, {df.item_id.nunique():,} items')
"
```

You should see roughly 1.78M / 222K / 222K rows for 20-core.

---

## Phase 2: Train GMF

```bash
# 20-core
python -m src.train --config configs/gmf.yaml --kcore 20

# 5-core
python -m src.train --config configs/gmf.yaml --kcore 5
```

This trains GMF with BPR loss for 20 epochs using a cosine learning rate schedule. The model checkpoints the best weights (by HR@10 on the validation set) and the final weights.

**Timing estimates (20-core, single GPU):**

| Model | Epochs | Time per Epoch | Total |
|-------|--------|---------------|-------|
| GMF | 20 | ~3 min | ~1 hour |

### Run on HPC

If you're on the SJSU HPC, the SLURM script handles everything:

```bash
# Full pipeline: preprocess → train → eval
sbatch scripts/run_hpc.sh

# Or individual phases:
sbatch scripts/run_hpc.sh preprocess
sbatch scripts/run_hpc.sh rec
sbatch scripts/run_hpc.sh eval
```

---

## Phase 3: Evaluate

```bash
python -m src.evaluate --config configs/gmf.yaml --kcore 20
python -m src.evaluate --config configs/gmf.yaml --kcore 5
```

This loads the best checkpoint and runs the leave-one-out evaluation protocol (1 positive + 99 negatives per test user).

### Expected Results (Paper Table 5 — GMF row)

**20-core:**

| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-------|------|-------|-------|--------|---------|---------|
| GMF | 0.3705 | 0.5219 | 0.6913 | 0.2565 | 0.3047 | 0.3477 |

**5-core:**

| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-------|------|-------|-------|--------|---------|---------|
| GMF | 0.3899 | 0.5340 | 0.7055 | 0.2761 | 0.3237 | 0.3666 |

Small differences (±5%) from the paper are expected due to different random seeds for negative sampling and minor implementation differences.

---

## Troubleshooting

### Out of Memory (OOM)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `MemoryError` during preprocessing | 5-core has 21M rows | Use `--kcore 20` first, or increase `--mem` in SLURM |
| `CUDA out of memory` during training | Batch size too large for GPU | Reduce `batch_size` in `configs/gmf.yaml` (try 128 or 64) |
| `Killed` (no error message) | OS OOM killer | Request more RAM: `sbatch --mem=64G scripts/run_hpc.sh` |

### Slow Training

| Symptom | Cause | Fix |
|---------|-------|-----|
| Training is very slow | No GPU detected | Check `hpa-verify` — make sure CUDA is available |
| Data loading is the bottleneck | Parquet reads are slow | Make sure data is on local SSD, not NFS |

### HPC Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'torch'` | venv not activated | Run `hpa-setup` first, then `source scripts/hpc_aliases.sh` |
| Job stuck in `PENDING` | No GPU nodes available | Check `squeue -p gpu` — wait or try off-peak hours |
| `SyntaxError: invalid syntax` | Using system Python 2.7 | Activate venv: `source venv/bin/activate` |

---

## How to Add a New Model

1. **Create the model file** in `src/models/`. Follow the pattern in [`gmf.py`](../src/models/gmf.py) — your model needs `__init__` and `forward(user_ids, item_ids) → scores`.

2. **Register it** in [`src/models/common.py`](../src/models/common.py) — add a case to `build_model()` so the training script can instantiate it by name.

3. **Add a config** (optional) — create a new YAML in `configs/` or add your model's params to the existing config.

4. **Run it**:
   ```bash
   python -m src.train --config configs/your_config.yaml --kcore 20
   ```

5. **Evaluate** — the evaluation pipeline works with any model that follows the same checkpoint format.

---

## Experiment Checklist

Use this to track progress:

- [ ] Download HotelRec data
- [ ] Preprocess 20-core subset
- [ ] Preprocess 5-core subset
- [ ] GMF (20-core) — HR/NDCG
- [ ] GMF (5-core) — HR/NDCG
- [ ] Compare with paper Table 5
