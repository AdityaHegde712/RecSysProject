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
  ├─ load_split()  → DataFrame for any split
  └─ get_user_positive_items()  → user → set of items (for negative sampling)
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

## ItemKNN Architecture

ItemKNN (Sarwar et al., 2001) is a standard item-based collaborative filtering method. It doesn't learn parameters — it computes item-item similarity from the interaction matrix and uses that to score candidates.

```
train.parquet
      │
      ▼
  Build sparse interaction matrix  (num_users × num_items, binary)
      │
      ▼
  Compute item-item cosine similarity  (num_items × num_items)
      │
      ▼
  Keep only top-k neighbors per item  (sparsify for efficiency)
      │
      ▼
  sim_matrix  (sparse, num_items × num_items)
```

### Recommendation

To recommend items for a user:

```
user_history = interaction_matrix[user_id]   # sparse row, items they've seen
scores = sim_matrix.T @ user_history.T       # sum of similarities to history
scores[seen_items] = -inf                    # exclude already-seen items
top_k = argsort(scores, descending)[:k]      # return top-k item IDs
```

Each candidate item's score is the sum of its cosine similarities to all items the user has interacted with. Items more similar to the user's history get higher scores.

### Prediction

To predict a score for a specific (user, item) pair:

```
score = sum(sim_matrix[item, j] for j in user_history)
```

This is the same as the recommendation scoring, but for a single item.

### Why Top-k Neighbors?

The full similarity matrix is (num_items × num_items), which for the 20-core subset is ~39K × 39K = 1.5 billion entries. Most of these are near-zero. Keeping only the top-k (default 50) most similar items per item reduces memory by ~99.9% with minimal impact on recommendation quality.

---

## Training Pipeline

```
Config (YAML)
      │
      ▼
  train.py --config configs/itemknn.yaml --kcore <k>
      │
      ├─ load_config()
      ├─ set_seed(42)
      ├─ load training DataFrame from parquet
      │
      └─ build_model() → ItemKNN
            │
            model.fit(train_df, num_users, num_items)
              ├─ build sparse interaction matrix
              ├─ compute item-item cosine similarity (batched)
              └─ sparsify to top-k neighbors
            │
            save model via pickle
            │
            evaluate on validation set
              ├─ for each val user: sample 99 negatives
              ├─ score all 100 candidates
              └─ compute HR@k, NDCG@k
```

ItemKNN has no training loop — it fits in a single pass. No optimizer, no loss function, no epochs. The "training" step is just building the similarity matrix, which takes a few minutes on CPU.

### Negative Sampling (Evaluation Only)

During evaluation, for each test/val user, we sample 99 random items the user hasn't interacted with. We score all 100 candidates (1 positive + 99 negatives) and rank them. This is the standard protocol from He et al. (2017) and matches the HotelRec paper.

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
│  Compute Node (no internet, has CPU + RAM)                       │
│                                                                  │
│  SLURM activates venv, then runs:                                │
│  1. Preprocess: k-core filter → parquet splits                   │
│  2. Fit ItemKNN (build similarity matrix, ~5 min)                │
│  3. Evaluate: HR@k, NDCG@k on test set                          │
│  4. Save results to results/ and logs to results/logs/           │
│                                                                  │
│  Note: ItemKNN doesn't need GPU — runs entirely on CPU.          │
│  Email notification on completion/failure                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## File Responsibilities

| File | What it does |
|------|-------------|
| `src/data/preprocess.py` | Raw JSON → parquet with k-core filtering and ID mapping |
| `src/data/split.py` | Train/val/test splitting (80/10/10, seed=42) |
| `src/data/dataset.py` | Load splits from parquet, build user-positive-items mapping |
| `src/models/knn.py` | ItemKNN: item-based collaborative filtering with cosine similarity |
| `src/models/common.py` | `build_model()` factory |
| `src/metrics/ranking.py` | HR@k, NDCG@k computation with leave-one-out protocol |
| `src/train.py` | Training entry point (fit ItemKNN, save via pickle, evaluate) |
| `src/evaluate.py` | Evaluation entry point (load model, compute ranking metrics) |
| `src/utils/io.py` | Config loading, pickle save/load for models |
| `src/utils/seed.py` | `set_seed()` for reproducibility |
| `src/utils/metrics_logger.py` | CSV metrics logging |
| `scripts/run_hpc.sh` | SLURM job script (setup + experiment phases) |
| `scripts/hpc_aliases.sh` | Shell aliases for common HPC commands |
| `scripts/download_data.sh` | Download HotelRec data from source |
| `scripts/verify_env.py` | Check Python version, packages |
| `scripts/validate_pipeline.py` | Dry-run pipeline validation |
| `configs/data.yaml` | Dataset paths, k-core values, split ratios |
| `configs/itemknn.yaml` | ItemKNN model config (k_neighbors, similarity, evaluation) |
