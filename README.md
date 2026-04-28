# Team 3 - Recommender Systems on the HotelRec Dataset

Team Members: Hriday Ampavatina, Pramod Yadav, Aditya Hegde

CMPE 256 - Recommender Systems, Spring 2026

## Overview

We built and compared multiple recommender system approaches on the [HotelRec](https://github.com/Diego999/HotelRec) dataset - 50M TripAdvisor hotel reviews. The project has three phases:
1. Shared foundation (Phase 1): data pipeline, baselines, evaluation framework.
2. Individual variants (Phase 2): each member builds one advanced method.
3. Integration (Phase 3): a LightGBM meta-learner trained on predictions from all three variants. 

## Dataset

The HotelRec dataset (Antognini & Faltings, LREC 2020) has around 50M hotel reviews, 365K hotels, and 22M users.
The full dataset stats are in [`results/data_evaluation.json`](results/data_evaluation.json).

| Subset | Users | Items | Interactions | Sparsity |
|--------|-------|-------|-------------|----------|
| Full | 21.9M | 365K | 50.3M | 99.999% |
| 20-core | ~47K | ~27K | ~1.8M | 99.86% |

We use the 20-core subset (users and items with ≥ 20 interactions each).

## Repository Structure

```
├── src/                         # All source code
│   ├── data/
│   │   ├── preprocess.py        # Raw JSONL → k-core filtered parquet (two-pass)
│   │   ├── preprocess_zip.py    # Same pipeline streaming directly from the .zip
│   │   ├── split.py             # Train/val/test splitting (80/10/10)
│   │   ├── dataset.py           # PyTorch Datasets + DataLoader factory
│   │   ├── sequential.py        # Per-user chronological sequence dataset (SASRec)
│   │   ├── subratings.py        # Sub-rating loader (NeuMF-Attn + TextNCF Sub-Rating)
│   │   └── text_embeddings.py   # MiniLM review encoder + per-user/per-item aggregation
│   ├── models/
│   │   ├── knn.py               # ItemKNN
│   │   ├── gmf.py               # Generalized Matrix Factorization
│   │   ├── popularity.py        # Popularity baseline
│   │   ├── common.py            # Model factory
│   │   ├── lightgcn_hg.py       # LightGCN-HG (secondary variant: metadata-augmented graph)
│   │   ├── sasrec.py            # SASRec (primary variant: self-attentive sequential)
│   │   ├── neumf_attn.py        # NeuMF-Attn (Variant B: sub-rating attention)
│   │   ├── text_ncf.py          # TextNCF base hybrid (Variant C)
│   │   ├── text_ncf_mt.py       # TextNCF Multi-Task (BPR + rating MSE)
│   │   └── text_ncf_subrating.py # TextNCF Sub-Rating decomposition
│   ├── graph/
│   │   └── hetero_adj.py        # Torch-free scipy builder for the HG adjacency
│   ├── evaluation/
│   │   ├── ranking.py           # HR@k, NDCG@k
│   │   └── rating.py            # RMSE / MAE + score-to-rating calibration
│   ├── utils/
│   ├── run_baselines.py         # Run Popularity + ItemKNN
│   ├── train_gmf.py             # GMF trainer
│   ├── train_lightgcn_hg.py     # LightGCN-HG trainer (--tiers none gives vanilla LightGCN)
│   ├── train_sasrec.py          # SASRec trainer
│   ├── train_neumf_attn.py      # NeuMF-Attn trainer (model.use_attention toggles enhanced/vanilla)
│   ├── train_text_ncf.py        # TextNCF base + ablation trainer (use_gmf / use_text toggles)
│   ├── train_text_ncf_mt.py     # TextNCF Multi-Task trainer
│   ├── train_text_ncf_subrating.py # TextNCF Sub-Rating trainer
│   ├── evaluate_ensemble.py     # Per-user weighted blend of TextNCF + GMF + ItemKNN (Pramod)
│   ├── evaluate_two_stage.py    # ItemKNN top-200 retrieval -> TextNCF re-rank (Pramod)
│   └── phase3_meta_ensemble.py  # Phase 3 LightGBM meta-learner over 4 base models
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
│   ├── sasrec.yaml              # SASRec (primary)
│   ├── neumf_attn.yaml          # NeuMF-Attn (Variant B, enhanced)
│   ├── neumf_vanilla.yaml       # NeuMF-Attn (Variant B, vanilla ablation)
│   ├── text_ncf.yaml / text_ncf_mt.yaml / text_ncf_subrating.yaml   # TextNCF family
│   └── _ablations/              # Auto-derived GMF-only / text-only configs (run_text_ncf_all.sh)
│
├── scripts/                     # End-to-end reproducibility scripts
│   ├── download_data.sh         # Fetch HotelRec.txt into data/raw/
│   ├── explore_data.py          # Full dataset EDA (streaming)
│   ├── extract_hotel_meta.py    # Parse TripAdvisor URL → g_id/region/country
│   ├── encode_text.py           # MiniLM review encoding (offline)
│   ├── fit_itemknn.py           # Pickle a fitted ItemKNN (input to ensemble + two-stage)
│   ├── compute_rmse.py          # RMSE/MAE for baselines + every variant
│   ├── run_text_ncf_all.sh      # One-button driver for the TextNCF family
│   ├── validate_pipeline.py     # Sanity check after preprocessing
│   └── verify_env.py            # Env/version verification
│
├── notebooks/                   # Executed, reproducible narratives (root = shared)
│   ├── dataset_eda.ipynb     # HotelRec EDA on the raw stats
│   ├── preprocessing.ipynb   # Raw stats → k-core → splits, with all cells executed
│   ├── baselines.ipynb       # Popularity / ItemKNN / GMF, live re-fits
│   └── ensemble_and_summary.ipynb  # Final cross-model comparison + Phase 3 walkthrough
│
├── results/
│   ├── data_evaluation.json
│   ├── baselines/
│   ├── gmf/
│   ├── lightgcn_hg/             # Hriday - secondary variant (vanilla + HG)
│   ├── sasrec/                  # Hriday - primary variant
│   ├── text_ncf*/               # Pramod's TextNCF family (5 sub-result dirs)
│   ├── neumf_attn/              # Aditya - NeuMF enhanced
│   ├── neumf_vanilla/           # Aditya - NeuMF vanilla ablation
│   └── phase3_meta/             # Phase 3 LightGBM meta-ensemble
│
├── extras/                      # Off-path tooling (not required for repro)
│   ├── hpc/                     # Pramod's SLURM/HPC convenience layer
│   ├── dev_tooling/             # summary.md auto-generators (frozen since hand-edited)
│   └── docs_phase1/             # Archived Phase 1 onboarding docs
│
└── data/                        # (gitignored) Raw & processed data
```

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

# 7. LightGCN - vanilla bipartite + HG augmentation (Hriday secondary variant, ~50 min each)
python -m src.train_lightgcn_hg --config configs/lightgcn_hg.yaml --kcore 20 --tiers none  # vanilla
python -m src.train_lightgcn_hg --config configs/lightgcn_hg.yaml --kcore 20               # HG (g_id+region+country)

# 8. SASRec (primary variant, ~15 min on GPU)
python -m src.train_sasrec --config configs/sasrec.yaml --kcore 20

# 9. NeuMF-Attn (Aditya's variant, ~100 min per run on GPU)
#    Enhanced (default), then vanilla ablation for the day-10 vanilla-vs-enhanced ask.
python -m src.train_neumf_attn --config configs/neumf_attn.yaml    --kcore 20
python -m src.train_neumf_attn --config configs/neumf_vanilla.yaml --kcore 20

# 10. TextNCF family (Pramod's variant; best run is Multi-Task)
#    Full pipeline = encode reviews once (~11 min), then 5 trainings +
#    ablations + ensemble + two-stage + RMSE (~70 min on GPU).
python scripts/encode_text.py --kcore 20 --device cuda
python scripts/fit_itemknn.py --kcore 20          # input to ensemble + two-stage
bash scripts/run_text_ncf_all.sh

# 11. Phase 3 - LightGBM meta-ensemble over the 4 best Phase-2 variants
#    (~2 min on GPU; needs the four checkpoints from steps 7-10 above)
python -m src.phase3_meta_ensemble --kcore 20

# 12. RMSE for all models (NeuMF-Attn + Phase 3 save their own rating_metrics.json in-script)
python scripts/compute_rmse.py --kcore 20 \
    --gmf-ckpt results/gmf/best_model.pt --gmf-dim 64 \
    --lightgcn-hg-ckpt results/lightgcn_hg/best_model_L1_d256_grc.pt \
    --lightgcn-hg-dim 256 --lightgcn-hg-layers 1 \
    --text-ncf-ckpt          results/text_ncf/best_model.pt \
    --text-ncf-mt-ckpt       results/text_ncf_mt/best_model.pt \
    --text-ncf-subrating-ckpt results/text_ncf_subrating/best_model.pt
```

## Evaluation Protocol

We followed He et al. (2017) and used the 1-vs-99 evaluation (1 positive + 99 random negatives) methodology:

1. For each test interaction, take the positive item.
2. Sample 99 random negatives the user never interacted with.
3. Rank all 100 candidates by model score (using `torch.sort` for consistent tie-breaking across models).
4. Compute **HR@k** and **NDCG@k** at k = 5, 10, 20.

## Results (20-core test set, 1-vs-99)

**Ranking metrics** (higher is better):

| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-------|------|-------|-------|--------|---------|---------|
| Popularity | 0.3150 | 0.4215 | 0.5538 | 0.2318 | 0.2662 | 0.2995 |
| GMF | 0.5553 | 0.6685 | 0.7936 | 0.4498 | 0.4863 | 0.5179 |
| ItemKNN (k=20) | 0.6835 | 0.6870 | 0.7091 | 0.6082 | 0.6093 | 0.6150 |
| Vanilla TextNCF (Pramod, base two-branch hybrid) | 0.5688 | 0.6787 | 0.7951 | 0.4702 | 0.5057 | 0.5351 |
| TextNCF - Multi-Task (Pramod, enhanced) | 0.5742 | 0.6864 | 0.8031 | 0.4734 | 0.5097 | 0.5392 |
| Vanilla NeuMF (Aditya) | 0.5978 | 0.7254 | 0.8468 | 0.4815 | 0.5228 | 0.5536 |
| NeuMF-Attn (Aditya, enhanced) | 0.5970 | 0.7245 | 0.8465 | 0.4809 | 0.5221 | 0.5530 |
| Vanilla LightGCN (Hriday, bipartite) | 0.6414 | 0.7532 | 0.8612 | 0.5315 | 0.5677 | 0.5950 |
| LightGCN-HG (Hriday secondary, dim=256, +g_id/region/country) | 0.6460 | 0.7591 | 0.8655 | 0.5352 | 0.5718 | 0.5988 |
| **SASRec (Hriday primary, dim=128, L=2)** | **0.8502** | **0.8808** | **0.9173** | **0.8294** | **0.8392** | **0.8484** |
| Phase 3 - LGBMRanker meta-ensemble | 0.6600 | 0.7739 | 0.8782 | 0.5474 | 0.5843 | 0.6107 |

**Rating-prediction metrics** (lower is better):

| Model | RMSE | MAE |
|-------|------|-----|
| GlobalMean (sanity) | 0.9315 | 0.7048 |
| **Popularity** (item mean) | **0.8685** | **0.6749** |
| ItemKNN (k=20, weighted neighbors) | 0.9590 | 0.7094 |
| GMF (calibrated) | 0.9302 | 0.7002 |
| Vanilla TextNCF (Pramod, calibrated) | 0.9306 | 0.7014 |
| TextNCF - Multi-Task (Pramod, calibrated) | 0.9304 | 0.7035 |
| Vanilla NeuMF (Aditya, calibrated) | 0.9304 | 0.7035 |
| NeuMF-Attn (Aditya, calibrated) | 0.9304 | 0.7032 |
| Vanilla LightGCN (Hriday, calibrated) | 0.9312 | 0.7025 |
| LightGCN-HG (Hriday, calibrated) | 0.9312 | 0.7025 |
| SASRec (Hriday, calibrated) | 0.9315 | 0.7048 |
| **Phase 3 meta-ensemble** (calibrated) | **0.8350** | **0.6164** |

Ranking-trained models (BPR) all land at RMSE ≈ 0.93. The calibration slope is near zero because BPR scores encode pairwise ranking, not rating levels. Popularity wins RMSE because 78% of HotelRec ratings are 4-5 stars, so item-mean is near-optimal on this rating distribution. The Phase 3 LGBMRanker meta-ensemble is the only ranking-trained pipeline that beats Popularity on RMSE (0.8350 vs 0.8685) because its blended score has more usable variance for an lstsq calibrator than any single BPR base model. On ranking, however, the meta-ensemble lands well below SASRec alone, the strong-model dilution effect under naive per-user normalisation.

## Phase 2 Variants

| Member | Variant | Key Idea |
|--------|---------|----------|
| Hriday | **SASRec** (primary) + LightGCN-HG (secondary) | Self-attentive sequential recommendation over time-ordered hotel sequences (uses `date`), plus a graph-based secondary with TripAdvisor location / region / country pivot nodes |
| Aditya | **NeuMF-Attn** | NeuMF backbone (GMF + MLP) with an optional per-user attention layer over six sub-rating aspects. Vanilla HR@10 = 0.7254, enhanced 0.7245 - attention head adds around 0 on this dataset. |
| Pramod | **TextNCF - Multi-Task** | Frozen MiniLM review embeddings fused with GMF branch. MT head adds a rating-MSE regulariser (α=0.7) - NDCG@10 = 0.5097 |

## Phase 3 Integration

We implemented a `LGBMRanker` (lambdarank objective) over per-user min-max normalised scores from the four headline models - SASRec (Hriday primary), LightGCN-HG (Hriday secondary), NeuMF-Attn (Aditya enhanced), TextNCF Multi-Task (Pramod enhanced). Trained on val (1+99 candidates per user, label = held-out positive), evaluated on test.

**Mixed result:**
- The meta-ensemble lands at HR@10 = 0.7739 / NDCG@10 = 0.5843. It's better than every non-sequential base model, but around 10pp below SASRec alone. 
 - Strong-model dilution under naive per-user normalisation: split-gain feature importance (text_ncf_mt 2582 ≥ neumf_attn 2419 ≥ lightgcn_hg 2345 ≥ sasrec 1654) confirms the LGBMRanker treats all four columns democratically once they're squashed to [0, 1].
- Rating: the meta-ensemble's calibrated RMSE = 0.8350, MAE = 0.6164. Slope a = 0.0261 (around 30x any single BPR base model), beating Popularity (0.8685) and every individual variant. The blend's varied output distribution gives the lstsq calibrator material to work with that no single BPR ranker provides.

The full walkthrough + feature-importance plot + future-work suggestions are in [`notebooks/ensemble_and_summary.ipynb`](notebooks/ensemble_and_summary.ipynb)

## Notebooks

Every reported number traces back to an executed notebook cell:

- [`notebooks/preprocessing.ipynb`](notebooks/preprocessing.ipynb) - raw scan stats, k-core filter, splits, leakage checks.
- [`notebooks/baselines.ipynb`](notebooks/baselines.ipynb) - Popularity / ItemKNN live re-fits + GMF saved metrics, side-by-side tables.
- [`variants/hriday/notebooks/lightgcn_hg.ipynb`](variants/hriday/notebooks/lightgcn_hg.ipynb) - LightGCN-HG graph construction + training + evaluation.
- [`variants/hriday/notebooks/sasrec.ipynb`](variants/hriday/notebooks/sasrec.ipynb) - SASRec model, training curves, final results.
- [`variants/pramod/notebooks/text_ncf.ipynb`](variants/pramod/notebooks/text_ncf.ipynb) - TextNCF family walkthrough: base + ablations + MT + sub-rating + ensemble + two-stage.
- [`variants/aditya/notebooks/neumf_attn.ipynb`](variants/aditya/notebooks/neumf_attn.ipynb) - NeuMF-Attn model, training curves, and final comparison.
- [`notebooks/ensemble_and_summary.ipynb`](notebooks/ensemble_and_summary.ipynb) - final comparison across all models.

## Reproducibility

**The Canonical environment (what the shipped numbers were produced on):**
- Windows 11, Python 3.11
- PyTorch nightly with CUDA 12.8 (RTX 5070 Ti / Blackwell - `cu124` will not load on this GPU. The nightly is required for the SM_120 kernel).
  ```bash
  pip install --pre torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/nightly/cu128
  ```
- The remaining dependencies pin via `requirements.txt`.

**Total estimated time for a fresh end-to-end re-run (RTX 5070 Ti):**

| Step | Cost | Notes |
|---|---|---|
| Data download | network-bound | around 50 GB zip, single fetch |
| Preprocessing (k-core + split) | 25 min | streaming JSONL, two-pass |
| Baselines (Popularity + ItemKNN + GMF) | 30 min |  |
| LightGCN vanilla + HG | 100 min | |
| SASRec | 15 min | |
| NeuMF vanilla + enhanced | 200 min | |
| TextNCF family (5 trainings + ensemble + two-stage + RMSE) | 80 min | |
| Phase 3 meta-ensemble | 2 min | |
| RMSE pass for all | 5 min |  |
| **Total** | **≈ 7.5 hours** | probably overnight on a single GPU |

**HPC alternative (Optional - not tested for all variants):** Pramod's SLURM layer lives in extras/hpc/ (run_hpc.sh, aliases, requirements-hpc.txt). The canonical local path above is what was used for every shipped result.

## AI Tool Disclosure

Each variant section's design decision log captures the design and architectural choices each member made (when AI suggested an approach, what they tried, and what they concluded). AI tools were used as follows: 

- Claude and Claude Code (Anthropic) - code scaffolding, scripts generation, debugging assistance, literature recall, prose review. Specific contributions include shared evaluation framework boilerplate, variant scripts and skeleton codes, partial README and summary drafting from concrete numbers, Phase 3 LightGBM harness, help with commit messages, and PR analysis.
- GitHub Copilot - PR Analysis, inline completion during coding in VSCode.
- Grok - Helping with interpretation on notebook results and literature research on the recommender systems.

## References
- Antognini & Faltings (2020). *HotelRec.* LREC.
- He et al. (2017). *Neural Collaborative Filtering.* WWW.
- He et al. (2020). *LightGCN.* SIGIR.
- Kang & McAuley (2018). *Self-Attentive Sequential Recommendation.* ICDM.
- Liang et al. (2018). *Variational Autoencoders for Collaborative Filtering.* WWW.
- Hidasi et al. (2016). *Session-based Recommendations with RNNs.* ICLR.
- Sun et al. (2019). *BERT4Rec.* CIKM.
- Petrov & Macdonald (2022). *A Systematic Review and Replicability Study of BERT4Rec.* RecSys.
- Sarwar et al. (2001). *Item-Based Collaborative Filtering.* WWW.
- Reimers & Gurevych (2019). *Sentence-BERT.* EMNLP.
- Wang et al. (2020). *MiniLM.* NeurIPS.
