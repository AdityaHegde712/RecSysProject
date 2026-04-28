# Experiment Guide - HotelRec ItemKNN Baseline

Step-by-step instructions for reproducing the ItemKNN baseline results from Antognini & Faltings (LREC 2020). The experiment has three phases: data preparation, fitting, and evaluation.

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
# 20-core (smaller, faster - start here)
python -m src.data.preprocess --kcore 20 --config configs/data.yaml

# 5-core (larger, closer to paper's main results)
python -m src.data.preprocess --kcore 5 --config configs/data.yaml
```

**Output** (in `data/processed/{kcore}/`):
- `interactions.parquet` - filtered interactions with integer IDs
- `user2id.json` - mapping from original user URLs to integer IDs
- `item2id.json` - mapping from original hotel URLs to integer IDs

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

## Phase 2: Fit ItemKNN

```bash
# 20-core
python -m src.train --config configs/itemknn.yaml --kcore 20

# 5-core
python -m src.train --config configs/itemknn.yaml --kcore 5
```

This fits ItemKNN by building the item-item cosine similarity matrix from the training interactions. No GPU needed - runs entirely on CPU. The model is saved via pickle and validation metrics are computed immediately after fitting.

**Timing estimates (20-core, CPU):**

| Model | What It Does | Time |
|-------|-------------|------|
| ItemKNN | Build similarity matrix + evaluate | ~5-10 min |

### Run on HPC

If you're on the SJSU HPC, the SLURM script handles everything:

```bash
# Full pipeline: preprocess → fit → eval
sbatch scripts/run_hpc.sh

# Or individual phases:
sbatch scripts/run_hpc.sh preprocess
sbatch scripts/run_hpc.sh rec
sbatch scripts/run_hpc.sh eval
```

---

## Phase 3: Evaluate

```bash
python -m src.evaluate --config configs/itemknn.yaml --kcore 20
python -m src.evaluate --config configs/itemknn.yaml --kcore 5
```

This loads the fitted ItemKNN model from pickle and runs the leave-one-out evaluation protocol (1 positive + 99 negatives per test user).

### Expected Results (Paper Table 5 - ItemKNN row)

**20-core:**

| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-------|------|-------|-------|--------|---------|---------|
| ItemKNN | 0.0236 | 0.0411 | 0.0682 | 0.0061 | 0.0084 | 0.0110 |

**5-core:**

| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-------|------|-------|-------|--------|---------|---------|
| ItemKNN | 0.0162 | 0.0238 | 0.0340 | 0.0072 | 0.0088 | 0.0103 |

Small differences (±10%) from the paper are expected due to different random seeds for negative sampling and minor implementation differences.

---

## Troubleshooting

### Out of Memory (OOM)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `MemoryError` during preprocessing | 5-core has 21M rows | Use `--kcore 20` first, or increase `--mem` in SLURM |
| `MemoryError` during similarity computation | Item-item matrix too large | Reduce `k_neighbors` in config, or use 20-core subset |
| `Killed` (no error message) | OS OOM killer | Request more RAM: `sbatch --mem=64G scripts/run_hpc.sh` |

### Slow Fitting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Similarity computation is slow | Large item catalog | Expected for 5-core (312K items). Use 20-core for development. |
| Data loading is slow | Parquet reads are slow | Make sure data is on local SSD, not NFS |

### HPC Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'sklearn'` | venv not activated | Run `hpa-setup` first, then `source scripts/hpc_aliases.sh` |
| Job stuck in `PENDING` | No nodes available | Check `squeue` - wait or try off-peak hours |
| `SyntaxError: invalid syntax` | Using system Python 2.7 | Activate venv: `source venv/bin/activate` |

---

## How to Add a New Model

1. **Create the model file** in `src/models/`. Your model needs `fit(train_df, num_users, num_items)`, `recommend(user_id, k)`, and `predict(user_ids, item_ids)` methods.

2. **Register it** in [`src/models/common.py`](../src/models/common.py) - add a case to `build_model()` so the training script can instantiate it by name.

3. **Add a config** - create a new YAML in `configs/` with your model's parameters.

4. **Run it**:
   ```bash
   python -m src.train --config configs/your_config.yaml --kcore 20
   ```

5. **Evaluate** - the evaluation pipeline works with any model that follows the same interface and can be pickled.

---

## Experiment Checklist

Use this to track progress:

- [ ] Download HotelRec data
- [ ] Preprocess 20-core subset
- [ ] Preprocess 5-core subset
- [ ] ItemKNN (20-core) - HR/NDCG
- [ ] ItemKNN (5-core) - HR/NDCG
- [ ] Compare with paper Table 5
