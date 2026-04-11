# Architecture Documentation

## Data Pipeline

The raw HotelRec data is one JSON file per hotel, each containing an array of review objects. We flatten everything into a single DataFrame, assign contiguous integer IDs, apply k-core filtering, and split into train/val/test parquets.

```
data/raw/*.json  (one file per hotel, ~365K files)
      │
      ▼
  preprocess.py
  ├─ parse each JSON → flat records
  ├─ build user/item ID maps (contiguous ints)
  ├─ k-core filter (iterative: drop users/items with < k reviews)
  └─ save ID mappings as JSON
      │
      ▼
data/processed/{kcore}/
  ├─ interactions.parquet   (user_id, item_id, rating, text, date, sub_ratings)
  ├─ user2id.json
  └─ item2id.json
      │
      ▼
  split.py  (seed=42, stratified by user)
      │
      ├─ train.parquet  (80%)
      ├─ val.parquet    (10%)
      └─ test.parquet   (10%)
      │
      ▼
  dataset.py
  ├─ InteractionDataset  → (user_id, pos_item, neg_item) with negative sampling
  └─ EvalInteractionDataset → (user_id, item_list, labels) for ranking eval
      │
      ▼
  DataLoader (batch_size from config, shuffle for train)
```

### k-core Filtering

This is an iterative process. You can't just filter users once and be done — removing a user might drop an item below the threshold, which then drops another user, and so on. The loop runs until convergence:

```
while True:
    user_counts = df.groupby("user_id").size()
    item_counts = df.groupby("item_id").size()
    before = len(df)
    df = df[df["user_id"].isin(user_counts[user_counts >= k].index)]
    df = df[df["item_id"].isin(item_counts[item_counts >= k].index)]
    if len(df) == before:
        break  # converged
```

### Dataset Sizes After Filtering

| Subset | Users | Items | Interactions | Sparsity |
|--------|-------|-------|-------------|----------|
| Full | 21,891,294 | 365,056 | 50,264,531 | 99.999% |
| 5-core | 2,012,162 | 312,081 | 21,108,245 | 99.997% |
| 20-core | 72,603 | 38,903 | 2,222,373 | 99.921% |

---

## GMF Architecture

GMF (Generalized Matrix Factorization) is the simplest model in the NCF framework from He et al. (2017). It takes the element-wise product of user and item embeddings, then passes the result through a linear layer and sigmoid.

```
user_id ──► Embedding(num_users, d) ──► u  (d,)
                                            │
                                            ├─ element-wise multiply ──► h  (d,)
                                            │                              │
item_id ──► Embedding(num_items, d) ──► i  (d,)                     Linear(d, 1)
                                                                           │
                                                                       Sigmoid
                                                                           │
                                                                     score ∈ (0,1)
```

This is essentially matrix factorization with a learned (non-uniform) kernel instead of a plain dot product. The linear layer lets the model weight different latent dimensions differently, which gives it more expressiveness than a standard dot product.

Embeddings are initialized with `N(0, 0.01)` and the output layer uses Xavier uniform initialization.

---

## Training Pipeline

```
Config (YAML)
      │
      ▼
  train.py --config configs/gmf.yaml --kcore <k>
      │
      ├─ load_config()
      ├─ set_seed(42)
      ├─ build DataLoaders (streaming from parquet)
      │
      └─ build_model() → move to GPU
            │
            for each epoch:
              train_one_epoch():
                for batch in train_loader:
                  sample negatives
                  forward → BPR loss → backward → Adam step
              │
              validate():
                for batch in val_loader:
                  forward → compute HR@k, NDCG@k
              │
              save checkpoint (best + last)
              log metrics to CSV
```

### Loss Function

We use BPR (Bayesian Personalized Ranking) pairwise loss — the model should score positive items higher than negative items:

```
loss = -mean(log(sigmoid(pos_score - neg_score)))
```

### Negative Sampling

During training, for each positive (user, item) pair, we sample 4 random items the user hasn't interacted with. During evaluation, we sample 99 negatives per positive — this is the standard NCF evaluation protocol.

---

## Evaluation Protocol

Following He et al. (2017) and the HotelRec paper:

1. For each user in the test set, take their one held-out positive item
2. Sample 99 random negative items (items the user never interacted with)
3. Score all 100 candidates (1 positive + 99 negatives)
4. Rank by score, compute HR@k and NDCG@k

```
HR@k  = 1 if the positive item appears in the top-k, else 0
        (averaged over all test users)

NDCG@k = 1/log2(rank+1) if the positive item is in top-k, else 0
         (averaged over all test users)
```

HR@k tells you "did we retrieve the right item?" NDCG@k also cares about *where* in the list it appears — ranking it #1 is better than #10.

---

## HPC Execution Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  Login Node (has internet, no GPU)                               │
│                                                                  │
│  1. git clone → cd HotelRec-HPA                                 │
│  2. bash scripts/run_hpc.sh setup                                │
│     ├─ create venv                                               │
│     ├─ pip install from pre-built wheels                         │
│     └─ download data (if needed)                                 │
│  3. source scripts/hpc_aliases.sh                                │
│  4. hpa-verify  (check env is correct)                           │
│  5. hpa-validate  (dry-run pipeline check)                       │
│                                                                  │
│  6. sbatch scripts/run_hpc.sh  ──────────────────────────┐       │
└──────────────────────────────────────────────────────────┼───────┘
                                                           │
                                                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  GPU Node (no internet, has GPU + 32GB RAM)                      │
│                                                                  │
│  SLURM activates venv, then runs:                                │
│  1. Preprocess: k-core filter → parquet splits                   │
│  2. Train GMF with BPR loss (20 epochs, cosine LR)               │
│  3. Evaluate: HR@k, NDCG@k on test set                          │
│  4. Save results to results/ and logs to logs/                   │
│                                                                  │
│  Email notification on completion/failure                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## File Responsibilities

| File | What it does |
|------|-------------|
| `src/data/preprocess.py` | Raw JSON → parquet with k-core filtering and ID mapping |
| `src/data/split.py` | Train/val/test splitting (80/10/10, seed=42) |
| `src/data/dataset.py` | PyTorch Datasets (InteractionDataset, EvalInteractionDataset), DataLoader factory |
| `src/models/gmf.py` | Generalized Matrix Factorization |
| `src/models/common.py` | `build_model()` factory |
| `src/metrics/ranking.py` | HR@k, NDCG@k computation |
| `src/train.py` | Training entry point (BPR loss, cosine scheduler, early stopping) |
| `src/evaluate.py` | Evaluation entry point (load checkpoint, compute ranking metrics) |
| `src/utils/io.py` | Config loading, checkpoint save/load |
| `src/utils/seed.py` | `set_seed()` for reproducibility |
| `src/utils/metrics_logger.py` | CSV metrics logging |
| `scripts/run_hpc.sh` | SLURM job script (setup + experiment phases) |
| `scripts/hpc_aliases.sh` | Shell aliases for common HPC commands |
| `scripts/download_data.sh` | Download HotelRec data from source |
| `scripts/verify_env.py` | Check Python version, packages, GPU availability |
| `scripts/validate_pipeline.py` | Dry-run pipeline validation |
| `configs/data.yaml` | Dataset paths, k-core values, split ratios |
| `configs/gmf.yaml` | GMF model config (embedding dim, lr, epochs, negative sampling) |
