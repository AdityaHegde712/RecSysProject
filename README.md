# Hotel Recommendation on HotelRec

CMPE 256 - Recommender Systems, Spring 2026
Team: Aditya Hegde, Pramod Yadav, Hriday Ampavatina

---

## Overview

We build and compare multiple recommender system approaches on the [HotelRec](https://github.com/Diego999/HotelRec) dataset - 50M TripAdvisor hotel reviews. The project has three phases:

1. **Shared foundation** (Phase 1): data pipeline, baselines, evaluation framework.
2. **Individual variants** (Phase 2): each member builds one advanced method.
3. **Integration** (Phase 3): a LightGBM meta-learner trained on out-of-fold predictions from all three variants as the final submission model.

---

## Dataset

**HotelRec** (Antognini & Faltings, LREC 2020) - ~50M hotel reviews, 365K hotels, 22M users.
Full-dataset stats in [`results/data_evaluation.json`](results/data_evaluation.json).

| Subset | Users | Items | Interactions | Sparsity |
|--------|-------|-------|-------------|----------|
| Full | 21.9M | 365K | 50.3M | 99.999% |
| 20-core | ~47K | ~27K | ~1.8M | 99.86% |

We use the **20-core** subset (users and items with ≥ 20 interactions each).

---

## Repository Structure

```
├── src/                         # All source code
│   ├── data/
│   │   ├── preprocess.py        # Raw JSONL → k-core filtered parquet (two-pass)
│   │   ├── split.py             # Train/val/test splitting (80/10/10)
│   │   ├── dataset.py           # PyTorch Datasets + DataLoader factory
│   │   └── sequential.py        # Per-user chronological sequence dataset (SASRec)
│   ├── models/
│   │   ├── knn.py               # ItemKNN
│   │   ├── gmf.py               # Generalized Matrix Factorization
│   │   ├── lightgcn_hg.py       # LightGCN-HG (secondary variant: metadata-augmented graph)
│   │   ├── sasrec.py            # SASRec (primary variant: self-attentive sequential)
│   │   ├── popularity.py        # Popularity baseline
│   │   └── common.py            # Model factory
│   ├── graph/
│   │   └── hetero_adj.py        # Torch-free scipy builder for the HG adjacency
│   ├── evaluation/
│   │   ├── ranking.py           # HR@k, NDCG@k
│   │   └── rating.py            # RMSE / MAE + score-to-rating calibration
│   ├── utils/
│   ├── run_baselines.py         # Run Popularity + ItemKNN
│   ├── train_gmf.py             # GMF trainer
│   ├── train_lightgcn_hg.py     # LightGCN-HG trainer
│   └── train_sasrec.py          # SASRec trainer
│
├── variants/                    # Phase 2: one folder per team member
│   ├── hriday/                  # SASRec (primary) + LightGCN-HG (secondary)
│   ├── aditya/                  # NeuMF + attention-weighted sub-ratings
│   └── pramod/                  # Review-text-enhanced NCF
│
├── configs/
│   ├── data.yaml
│   ├── gmf.yaml
│   ├── itemknn.yaml
│   ├── lightgcn_hg.yaml         # LightGCN-HG (secondary)
│   └── sasrec.yaml              # SASRec (primary)
│
├── scripts/
│   ├── explore_data.py          # Full dataset EDA (streaming)
│   ├── extract_hotel_meta.py    # Parse TripAdvisor URL → g_id/region/country
│   ├── compute_rmse.py          # RMSE/MAE for baselines + GMF + LightGCN-HG
│   ├── summarize_lightgcn_hg.py # results/lightgcn_hg/summary.md
│   ├── summarize_sasrec.py      # results/sasrec/summary.md
│   ├── download_data.sh
│   ├── validate_pipeline.py
│   ├── verify_env.py
│   ├── run_hpc.sh
│   └── hpc_aliases.sh
│
├── notebooks/                   # Executed, reproducible narratives (root = shared)
│   ├── 01_preprocessing.ipynb   # Raw stats → k-core → splits, with all cells executed
│   ├── 02_baselines.ipynb       # Popularity / ItemKNN / GMF, live re-fits
│   └── 05_ensemble_and_summary.ipynb  # Final cross-model comparison tables + plots
│
├── docs/
├── results/
│   ├── data_evaluation.json
│   ├── baselines/
│   ├── gmf/
│   ├── lightgcn_hg/             # Secondary variant
│   ├── sasrec/                  # Primary variant
│   └── text_ncf/                # Pramod's TextNCF family (base + summary;
│                                #  see results/text_ncf_{mt,subrating,gmf_only,text_only}/ too)
└── data/                        # (gitignored) Raw & processed data
```

---

## Quick Start

```bash
# 1. Install
git clone https://github.com/AdityaHegde712/RecSysProject.git
cd RecSysProject
pip install -r requirements.txt
pip install -e .

# 2. Download HotelRec.txt into data/raw/
#    (from https://github.com/Diego999/HotelRec)

# 3. Preprocess (20-core, two-pass - handles 50 GB without OOM)
python -m src.data.preprocess --kcore 20
python -m src.data.split --kcore 20

# 4. Baselines (Popularity + ItemKNN)
python -m src.run_baselines --kcore 20

# 5. GMF (neural baseline, ~25 min on GPU)
python -m src.train_gmf --config configs/gmf.yaml --kcore 20

# 6. Hotel metadata (for LightGCN-HG; read-only pass over item2id.json)
python -m scripts.extract_hotel_meta --kcore 20

# 7. LightGCN-HG (secondary variant, ~53 min on GPU)
python -m src.train_lightgcn_hg --config configs/lightgcn_hg.yaml --kcore 20

# 8. SASRec (primary variant, ~15 min on GPU)
python -m src.train_sasrec --config configs/sasrec.yaml --kcore 20

# 9. TextNCF family (Pramod's variant; best run is Multi-Task)
#    Full pipeline = encode reviews once (~11 min), then 5 trainings +
#    ablations + ensemble + two-stage + RMSE (~70 min on GPU).
python scripts/encode_text.py --kcore 20 --device cuda
python scripts/fit_itemknn.py --kcore 20          # input to ensemble + two-stage
bash scripts/run_text_ncf_all.sh

# 10. RMSE for all models
python scripts/compute_rmse.py --kcore 20 \
    --gmf-ckpt results/gmf/best_model.pt --gmf-dim 64 \
    --lightgcn-hg-ckpt results/lightgcn_hg/best_model_L1_d256_grc.pt \
    --lightgcn-hg-dim 256 --lightgcn-hg-layers 1 \
    --text-ncf-ckpt          results/text_ncf/best_model.pt \
    --text-ncf-mt-ckpt       results/text_ncf_mt/best_model.pt \
    --text-ncf-subrating-ckpt results/text_ncf_subrating/best_model.pt
```

---

## Evaluation Protocol

Following He et al. (2017):

1. For each test interaction, take the positive item.
2. Sample 99 random negatives the user never interacted with.
3. Rank all 100 candidates by model score (using `torch.sort` for consistent tie-breaking across models).
4. Compute **HR@k** and **NDCG@k** at k = 5, 10, 20.

---

## Results (20-core test set, 1-vs-99)

**Ranking metrics** (higher is better):

| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-------|------|-------|-------|--------|---------|---------|
| Popularity | 0.3150 | 0.4215 | 0.5538 | 0.2318 | 0.2662 | 0.2995 |
| GMF | 0.5553 | 0.6685 | 0.7936 | 0.4498 | 0.4863 | 0.5179 |
| ItemKNN (k=20) | 0.6835 | 0.6870 | 0.7091 | 0.6082 | 0.6093 | 0.6150 |
| TextNCF — Multi-Task (Pramod) | 0.5742 | 0.6864 | 0.8031 | 0.4734 | 0.5097 | 0.5392 |
| LightGCN-HG (secondary, dim=256) | 0.6460 | 0.7591 | 0.8655 | 0.5352 | 0.5718 | 0.5988 |
| **SASRec (primary, dim=128, L=2)** | **0.8502** | **0.8808** | **0.9173** | **0.8294** | **0.8392** | **0.8484** |

**Rating-prediction metrics** (lower is better):

| Model | RMSE | MAE |
|-------|------|-----|
| GlobalMean (sanity) | 0.9315 | 0.7048 |
| **Popularity** (item mean) | **0.8685** | **0.6749** |
| ItemKNN (k=20, weighted neighbors) | 0.9590 | 0.7094 |
| GMF (calibrated) | 0.9302 | 0.7002 |
| TextNCF — Multi-Task (calibrated) | 0.9304 | 0.7035 |
| LightGCN-HG (calibrated) | 0.9312 | 0.7025 |
| SASRec (calibrated) | 0.9315 | 0.7048 |

Ranking-trained models (BPR) all land at RMSE ≈ 0.93 - the calibration slope is near zero because BPR scores encode pairwise ranking, not rating levels. Popularity wins RMSE because 78% of HotelRec ratings are 4–5 stars, so item-mean is near-optimal on this rating distribution.

See [`results/sasrec/summary.md`](results/sasrec/summary.md), [`results/lightgcn_hg/summary.md`](results/lightgcn_hg/summary.md), and [`results/text_ncf/summary.md`](results/text_ncf/summary.md) for the full variant writeups. Pramod's summary also documents two instructive negative results — a collapsed sub-rating attention head and a per-variant ensemble that degenerated to ItemKNN — that motivate the Phase 3 meta-ensemble.

---

## Phase 2 Variants

| Member | Variant | Key Idea |
|--------|---------|----------|
| Hriday | **SASRec** (primary) + LightGCN-HG (secondary) | Self-attentive sequential recommendation over time-ordered hotel sequences (uses `date`), plus a graph-based secondary with TripAdvisor location / region / country pivot nodes |
| Aditya | NeuMF + sub-ratings | Attention over hotel quality dimensions (Service, Location, ...) |
| Pramod | **TextNCF — Multi-Task** (best of family) | Frozen MiniLM review embeddings fused with GMF branch; MT head adds a rating-MSE regulariser (α=0.7) — NDCG@10 = 0.5097 |

## Phase 3 Integration (planned)

All three variants will feed into a **LightGBM meta-learner** trained on out-of-fold predictions from each base model, as the final submission model. Per-variant ensembles (e.g. with ItemKNN) are out of scope - the integration is done at the meta-learner level across all variants.

---

## Notebooks

Every reported number traces back to an executed notebook cell:

- [`notebooks/01_preprocessing.ipynb`](notebooks/01_preprocessing.ipynb) - raw scan stats, k-core filter, splits, leakage checks.
- [`notebooks/02_baselines.ipynb`](notebooks/02_baselines.ipynb) - Popularity / ItemKNN live re-fits + GMF saved metrics, side-by-side tables.
- [`variants/hriday/notebooks/04_lightgcn_hg.ipynb`](variants/hriday/notebooks/04_lightgcn_hg.ipynb) - LightGCN-HG graph construction + training + evaluation.
- [`variants/hriday/notebooks/06_sasrec.ipynb`](variants/hriday/notebooks/06_sasrec.ipynb) - SASRec model, training curves, final results.
- [`variants/pramod/notebooks/07_text_ncf.ipynb`](variants/pramod/notebooks/07_text_ncf.ipynb) - TextNCF family walkthrough: base + ablations + MT + sub-rating + ensemble + two-stage.
- [`notebooks/05_ensemble_and_summary.ipynb`](notebooks/05_ensemble_and_summary.ipynb) - final comparison across all models.

---

## References

- Antognini & Faltings (2020). *HotelRec*. LREC 2020.
- He et al. (2017). *Neural Collaborative Filtering*. WWW 2017.
- He et al. (2020). *LightGCN*. SIGIR 2020.
- Kang & McAuley (2018). *Self-Attentive Sequential Recommendation*. ICDM 2018.
- Sarwar et al. (2001). *Item-Based Collaborative Filtering*. WWW 2001.

## Tools

- **Claude** (Anthropic) - scaffolding, code generation, troubleshooting.
