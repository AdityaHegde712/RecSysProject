# Hotel Recommendation on HotelRec

CMPE 256 -- Recommender Systems, Spring 2026
Team: Aditya Hegde, Pramod Yadav, Hriday Ampavatina

---

## Overview

We build and compare multiple recommender system approaches on the [HotelRec](https://github.com/Diego999/HotelRec) dataset -- 50M TripAdvisor hotel reviews. The project has three phases:

1. **Shared foundation** (Phase 1): data pipeline, baselines, evaluation framework
2. **Individual variants** (Phase 2): each member builds one advanced method
3. **Integration** (Phase 3): compare all approaches, write report

---

## Dataset

**HotelRec** (Antognini & Faltings, LREC 2020) -- ~50M hotel reviews, 365K hotels, 22M users.
Full-dataset stats are in [`results/data_evaluation.json`](results/data_evaluation.json).

| Subset | Users | Items | Interactions | Sparsity |
|--------|-------|-------|-------------|----------|
| Full | 21.9M | 365K | 50.3M | 99.999% |
| 20-core | ~47K | ~27K | ~1.8M | 99.86% |

We use the **20-core** subset (users and items with >= 20 interactions each).

---

## Repository Structure

```
├── src/                         # All source code lives here
│   ├── data/
│   │   ├── preprocess.py        # Raw JSONL -> k-core filtered parquet (two-pass)
│   │   ├── split.py             # Train/val/test splitting (80/10/10)
│   │   └── dataset.py           # PyTorch Datasets + DataLoader factory
│   ├── models/
│   │   ├── knn.py               # ItemKNN (cosine similarity, scipy sparse)
│   │   ├── gmf.py               # Generalized Matrix Factorization (PyTorch)
│   │   ├── lightgcn.py          # LightGCN (graph CF, day 10)
│   │   ├── popularity.py        # Global popularity baseline
│   │   └── common.py            # Model factory (build_model)
│   ├── evaluation/
│   │   ├── ranking.py           # HR@k, NDCG@k (leave-one-out, 1+99 negatives)
│   │   └── rating.py            # RMSE / MAE + score->rating calibration (day 10)
│   ├── utils/
│   │   ├── io.py                # Config, checkpoint, pickle save/load
│   │   ├── seed.py              # Reproducibility
│   │   └── metrics_logger.py    # CSV logging
│   ├── run_baselines.py         # Run Popularity + ItemKNN, save results
│   ├── train_gmf.py             # Train GMF with BPR loss
│   └── train_lightgcn.py        # Train LightGCN with BPR loss (day 10)
│
├── variants/                    # Phase 2: one folder per team member
│   ├── hriday/                  # LightGCN (graph-based CF)
│   ├── aditya/                  # NeuMF + attention-weighted sub-ratings
│   └── pramod/                  # Review-text-enhanced NCF
│
├── configs/
│   ├── data.yaml                # Dataset paths, k-core, split ratios
│   ├── gmf.yaml                 # GMF hyperparameters
│   ├── itemknn.yaml             # ItemKNN config
│   ├── lightgcn.yaml            # LightGCN sweep config (K=3, dim=64)
│   └── lightgcn_best.yaml       # LightGCN extended config (K=1, dim=128)
│
├── scripts/                     # Utilities and HPC
│   ├── explore_data.py          # Full dataset EDA (streaming, O(1) memory)
│   ├── download_data.sh         # Dataset download helper
│   ├── validate_pipeline.py     # Smoke test: fit + predict on synthetic data
│   ├── verify_env.py            # Check all dependencies
│   ├── compute_rmse.py          # RMSE/MAE on baselines + calibrated LightGCN
│   ├── summarize_lightgcn.py    # Assemble results/lightgcn/summary.md
│   ├── run_hpc.sh               # SLURM job script for SJSU HPC
│   └── hpc_aliases.sh           # Shell shortcuts for HPC
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md
│   ├── ELI5.md
│   └── EXPERIMENT_GUIDE.md
│
├── results/
│   ├── data_evaluation.json     # Full 50M dataset statistics
│   └── baselines/
│       └── baseline_results_20core.json
│
└── data/                        # (gitignored) Raw & processed data
    ├── raw/HotelRec.txt
    └── processed/20core/
        ├── interactions.parquet
        ├── train.parquet
        ├── val.parquet
        ├── test.parquet
        ├── user2id.json
        └── item2id.json
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/AdityaHegde712/RecSysProject.git
cd RecSysProject
pip install -r requirements.txt
pip install -e .

# 2. Download HotelRec.txt into data/raw/
#    (from https://github.com/Diego999/HotelRec)

# 3. Preprocess (20-core, two-pass -- handles 50GB without OOM)
python -m src.data.preprocess --kcore 20

# 4. Split into train/val/test (80/10/10)
python -m src.data.split --kcore 20

# 5. Run baselines (Popularity + ItemKNN)
python -m src.run_baselines --kcore 20

# 6. Train GMF (uses GPU if available, falls back to CPU)
python -m src.train_gmf --config configs/gmf.yaml --kcore 20

# 7. Train LightGCN (Hriday's variant; best config matches HR@10 = 0.736)
python -m src.train_lightgcn --config configs/lightgcn_best.yaml --kcore 20

# 8. Compute RMSE (baselines + calibrated LightGCN)
python scripts/compute_rmse.py --kcore 20 --lightgcn-layers 1 --lightgcn-dim 128 \
    --lightgcn-ckpt results/lightgcn/best_model_L1_d128.pt
```

---

## Evaluation Protocol

Following He et al. (2017):

1. For each test interaction, take the positive item
2. Sample 99 random negatives the user never interacted with
3. Rank all 100 candidates by model score
4. Compute **HR@k** and **NDCG@k** at k = 5, 10, 20

---

## Results (20-core, 1-vs-99 test set)

**Ranking metrics** (higher is better):

| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-------|------|-------|-------|--------|---------|---------|
| Popularity | 0.3150 | 0.4215 | 0.5538 | 0.2318 | 0.2662 | 0.2995 |
| GMF | 0.5553 | 0.6685 | 0.7936 | 0.4498 | 0.4863 | 0.5179 |
| ItemKNN | 0.6835 | 0.6870 | 0.7091 | 0.6082 | 0.6093 | 0.6150 |
| **LightGCN** (K=1, dim=128) | **0.6135** | **0.7364** | **0.8538** | **0.4985** | **0.5383** | **0.5681** |

**Rating-prediction metrics** (lower is better):

| Model | RMSE | MAE |
|-------|------|-----|
| GlobalMean (sanity) | 0.9315 | 0.7048 |
| Popularity (item mean) | **0.8685** | **0.6749** |
| ItemKNN (weighted neighbors) | 0.9703 | 0.7162 |
| LightGCN (calibrated) | 0.9311 | 0.7022 |

> The ranking protocol (1-vs-99, HR@k / NDCG@k) is the primary comparison since
> all models are trained with BPR-style ranking objectives. RMSE/MAE are
> included as a traditional secondary metric; Popularity (item-mean rating)
> wins because HotelRec ratings are heavily skewed toward 4-5 stars.

See [`results/lightgcn/summary.md`](results/lightgcn/summary.md) for the full
LightGCN sweep and analysis.

---

## Phase 2 Variants

| Member | Variant | Key Idea |
|--------|---------|----------|
| Hriday | LightGCN | Graph convolution over user-item bipartite graph |
| Aditya | NeuMF + sub-ratings | Attention over hotel quality dimensions (Service, Location, ...) |
| Pramod | Text-enhanced NCF | Sentence-transformer embeddings of review text |

---

## References

- Antognini & Faltings (2020). *HotelRec*. LREC 2020.
- He et al. (2017). *Neural Collaborative Filtering*. WWW 2017.
- He et al. (2020). *LightGCN*. SIGIR 2020.
- Sarwar et al. (2001). *Item-Based Collaborative Filtering*. WWW 2001.

## Tools

- **Claude** (Anthropic) -- scaffolding, code generation, troubleshooting
