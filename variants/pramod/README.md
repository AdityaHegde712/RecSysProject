# Variant C - TextNCF: Review-Text-Enhanced NCF

**Owner:** Pramod Yadav

## Approach

Plug review text into a standard NCF pipeline. Hotels get reviewed with
detailed text ("great location, noisy rooms, amazing breakfast"), and
that text carries signal star ratings alone miss. The base model has two
branches:

- **GMF branch** - standard user/item embedding dot product.
- **Text branch** - sentence embeddings of reviews, projected down and
  combined.

Both branches concat → small MLP → score. Trained with BPR loss
(implicit-feedback framing).

On top of the base TextNCF, four additional approaches push performance
in different directions: ablations (GMF-only / text-only), Multi-Task
(joint BPR + rating MSE), Sub-rating decomposition, plus Ensemble and
Two-Stage retrieval evaluations.

## Why this fits HotelRec

- 50 M reviews averaging 125 words each - dense preference signal pure CF
  ignores. About 71 % of reviews have substantive text. The text captures
  things like "walking distance to the beach" or "thin walls" that you
  can't get from a 4-star rating.
- Frozen **all-MiniLM-L6-v2** (384-dim) means no fine-tuning of a 22 M
  parameter transformer - just learn a small projection layer on top.
  Encoding done offline.

## Non-overlap with teammates

| Variant | Input channel | Architecture |
|---|---|---|
| Hriday - SASRec | Time sequence (`date`) | Transformer decoder |
| Aditya - NeuMF + sub-ratings | Sub-rating aspects (per-item average) | Feature-concat MLP |
| **Pramod - TextNCF family** (this) | Review text (frozen MiniLM) + collaborative | Two-branch fusion MLP |

The text channel is orthogonal to time (SASRec) and aspects (Aditya).
The sub-rating sub-variant here uses sub-ratings as an auxiliary
*supervision target*, not as a primary input.

## Design decision log

| Decision | Alternatives | Choice | Why |
|---|---|---|---|
| Text encoder | bag-of-words, TF-IDF, MiniLM-L6, MPNet-base, fine-tuned BERT | **MiniLM-L6 (frozen)** | 384-dim, runs in 11 min on RTX 5070 Ti, no fine-tuning required. MPNet planned as future work if text alone is the bottleneck. |
| Text aggregation | per-review scoring, per-user average, per-item average | **per-user (train) + per-item (all splits)** | Item profile = hotel metadata (allowed); user profile uses train-only to avoid leakage. |
| Architecture | early fusion, late fusion (separate scores), two-branch + MLP | **two-branch + small MLP** | Mirrors NCF; lets each channel learn separately before combining. |
| Loss | BCE, BPR, sampled softmax | **BPR (4 negs)** | Matches the rest of the team for cross-variant comparability. |
| Negatives | 1, 4, 8 | **4** | Best HR@10 vs train time trade-off in a quick sweep. |
| Multi-task α | 0.3, 0.5, 0.7, 0.9 | **0.7 (BPR weight)** | Lets ranking dominate while the MSE head provides a smooth regulariser; 0.5 hurt HR. |
| Sub-rating β | 0.4, 0.6, 0.8 | **0.6** | Ablation showed β=0.4 caused attention collapse faster; β=0.8 essentially turned off the aspect heads. |
| Ensemble grid step | 0.05, 0.1, 0.2 | **0.1** | 66 weight combos; fine-grained enough to find the optimum without exploding eval cost. |
| Two-stage `retrieve_k` | 100, 200, 500, 2000 | **200** | Production-realistic recall budget; gt_recall@200 = 5 % flagged the retrieval bottleneck. |

## Pipeline

```
HotelRec.txt ──► preprocess.py / preprocess_zip.py ──► data/processed/20core/
                                                              │
                                                              ├─ encode_text.py (MiniLM, one-shot)
                                                              │     └─ data/processed/text_emb/*.npy
                                                              │
                                                              ├─ train_text_ncf.py            → results/text_ncf/
                                                              ├─ train_text_ncf_mt.py         → results/text_ncf_mt/
                                                              ├─ train_text_ncf_subrating.py  → results/text_ncf_subrating/
                                                              │
                                                              ├─ evaluate_ensemble.py   (needs TextNCF + GMF + ItemKNN)
                                                              └─ evaluate_two_stage.py  (needs TextNCF + ItemKNN)
```

All eval artefacts land under `results/<variant>/` as `test_metrics.json`
+ `rating_metrics.json`, so `scripts/compute_rmse.py` and the cross-team
summary consume them uniformly.

## Approaches in detail

### C1 - Base TextNCF (vanilla, primary path)

Two-branch hybrid, BPR loss, 30 epochs, cosine LR, patience=5.
Config: `configs/text_ncf.yaml` (`embed_dim=64`, `text_proj_dim=64`,
`mlp=[128,64]`, 4 negatives).

### C2 - Ablations

Same architecture with one branch disabled (`use_gmf=false` or
`use_text=false`). Tests how much of the lift comes from each branch.

### C3 - Ensemble (TextNCF + GMF + ItemKNN)

Per-user min-max normalises each model's scores, then grid-searches
`(w_text, w_gmf, w_knn)` weights summing to 1 (step 0.1 → 66 combos)
on val. Reports test metrics at the best validation NDCG@10.

### C4 - Two-stage retrieval + re-ranking

Production-style: ItemKNN retrieves top-200 candidates (sparse, ms per
user), TextNCF re-ranks (neural). Also measures `gt_recall@200` -
how often the held-out item is even in the candidate set.

### C5 - Multi-Task TextNCF (enhanced, best ranker)

Same backbone, two heads: BPR (ranking) + MSE on 1-5 ratings.
Joint loss `α · BPR + (1 - α) · MSE`, α=0.7. Tests whether the rating
signal smooths the loss surface for ranking.

### C6 - Sub-rating decomposition

Shared MLP → 6 aspect heads (Service, Cleanliness, Location, Value,
Rooms, Sleep Quality) + per-user attention weights → weighted sum.
Loss: BPR on the combined score plus MSE on each aspect.
Sub-rating coverage on the 20-core train split:
Service 80 %, Cleanliness 62 %, Location 60 %, Value 62 %, Rooms 60 %,
Sleep Quality 55 %. Missing values fall back to the overall rating.

## How to run

```bash
# step 1: encode reviews (~11 min on RTX 5070 Ti)
python scripts/encode_text.py --kcore 20 --device cuda

# step 2: fit + pickle ItemKNN (needed by ensemble + two-stage)
python scripts/fit_itemknn.py --kcore 20

# step 3: kick off everything (5 trainings + ensemble + two-stage + RMSE)
bash scripts/run_text_ncf_all.sh
```

`run_text_ncf_all.sh` is the one-button reproduction path. Per-step
logs land in `logs/<step>.log`, outputs under `results/text_ncf*/`.
Total wall-clock on RTX 5070 Ti: ~80 min.

Per-step (if you'd rather run them one at a time):

```bash
python -m src.train_text_ncf            --config configs/text_ncf.yaml            --kcore 20
python -m src.train_text_ncf_mt         --config configs/text_ncf_mt.yaml         --kcore 20
python -m src.train_text_ncf_subrating  --config configs/text_ncf_subrating.yaml  --kcore 20
python -m src.evaluate_ensemble  --kcore 20 --grid-step 0.1
python -m src.evaluate_two_stage --kcore 20 --retrieve-k 200
python scripts/compute_rmse.py --kcore 20 \
    --text-ncf-ckpt           results/text_ncf/best_model.pt \
    --text-ncf-mt-ckpt        results/text_ncf_mt/best_model.pt \
    --text-ncf-subrating-ckpt results/text_ncf_subrating/best_model.pt
```

(For the HPC convenience layer, see `extras/hpc/`. The canonical local
setup uses the top-level `requirements.txt` + the commands above.)

## Vanilla vs Enhanced (day-10 ask)

For Variant C, "vanilla" is the **base TextNCF** (the two-branch
foundation), and "enhanced" is **Multi-Task TextNCF** (added rating-MSE
head):

| Variant | HR@10 | NDCG@10 | Notes |
|---|---|---|---|
| Vanilla TextNCF (base hybrid) | 0.6787 | 0.5057 | both branches on, BPR only |
| **Enhanced TextNCF - Multi-Task** | **0.6864** | **0.5097** | best ranker in the family |

Multi-Task gives **+0.008 HR@10 / +0.004 NDCG@10** over vanilla - a small
but consistent lift from the rating-MSE regulariser, similar in scale
to LightGCN-HG's geography lift over vanilla LightGCN.

## Cross-team headline ranking

| Model | HR@10 | NDCG@10 | Notes |
|-------|-------|---------|-------|
| Popularity (baseline) | 0.4215 | 0.2662 | item-mean rating |
| GMF (baseline) | 0.6685 | 0.4863 | from `results/gmf/` |
| ItemKNN (baseline) | 0.6870 | 0.6093 | k=20, weighted neighbour |
| Vanilla TextNCF (base hybrid) | 0.6787 | 0.5057 | beats GMF; below KNN on NDCG |
| TextNCF GMF-only ablation | 0.6720 | 0.4915 | text branch off |
| TextNCF text-only ablation | 0.6891 | 0.4981 | GMF off - text carries the lift |
| **TextNCF Multi-Task (enhanced)** | **0.6864** | **0.5097** | best ranker in the family |
| TextNCF Sub-rating | 0.6677 | 0.4710 | attention collapsed onto Cleanliness |
| Ensemble (TextNCF + GMF + KNN) | 0.6870 | 0.6093 | grid picked KNN-only - degenerate |
| Two-stage (KNN → TextNCF) | 0.3858 | 0.2977 | gt_recall@200 = 5 % - recall-bound |

**Deltas vs GMF baseline (best single-model comparison):**

| Metric | TextNCF MT Δ (abs) | TextNCF MT Δ (rel) |
|--------|-------------------|-------------------|
| HR@5   | +0.0189 | +3.4% |
| HR@10  | +0.0179 | +2.7% |
| HR@20  | +0.0095 | +1.2% |
| NDCG@5 | +0.0236 | +5.2% |
| NDCG@10| +0.0234 | +4.8% |
| NDCG@20| +0.0213 | +4.1% |

Calibrated RMSE for the trained variants is **0.93** (slope ≈ 0.01-0.03)
- same flat-calibration pattern SASRec / GMF / LightGCN-HG hit. Popularity
wins RMSE at 0.8685.

## Risks & known limitations

- **Cold-start items.** Items with zero training reviews would get a
  zero-vector text embedding. The 20-core filter gives every item ≥ 20
  interactions, so this is rare. Mitigation: item profile averages over
  all splits (allowed - hotels aren't the label).
- **MiniLM ceiling.** 384-dim MiniLM is cheap but not great at
  domain-specific vocabulary. MPNet-base (768-dim) is the natural next
  experiment if the gain from text alone matters more.
- **Ranking-only RMSE.** Same structural issue as the rest of the team's
  variants - BPR-trained scores don't calibrate to 1-5 ratings on a
  4-star-heavy dataset. RMSE reported via linear calibration on val,
  same methodology used for SASRec / GMF / LightGCN-HG.
- **Sub-rating attention collapsed** (99.8 % weight on Cleanliness).
  Sparsemax / entropy-bonus would be the natural fix; same dataset-level
  issue Aditya's NeuMF-Attn variant hit.
- **Ensemble degenerated** to ItemKNN-only. Per-user min-max wipes the
  scale information that lets LightGBM weight strong vs weak base
  models - Phase 3 documents the same effect.

## Phase 3 integration

TextNCF Multi-Task feeds the team's LightGBM meta-learner alongside
SASRec (Hriday primary), LightGCN-HG (Hriday secondary), and NeuMF-Attn
(Aditya). The per-variant ensemble (C3) is for variant-internal analysis,
not a substitute for the team-wide meta-ensemble.

## Files

**Models:**
- `src/models/text_ncf.py` - base TextNCF model
- `src/models/text_ncf_mt.py` - Multi-Task variant
- `src/models/text_ncf_subrating.py` - Sub-rating decomposition variant

**Training scripts:**
- `src/train_text_ncf.py`, `src/train_text_ncf_mt.py`,
  `src/train_text_ncf_subrating.py`

**Evaluation scripts:**
- `src/evaluate_ensemble.py` - TextNCF + GMF + ItemKNN ensemble
- `src/evaluate_two_stage.py` - two-stage retrieval + re-ranking

**Data utilities:**
- `src/data/text_embeddings.py` - encoding + loading helpers
- `src/data/subratings.py` - sub-rating data loading

**Configs:**
- `configs/text_ncf.yaml`, `configs/text_ncf_mt.yaml`,
  `configs/text_ncf_subrating.yaml`

**Top-level scripts (under `scripts/`):**
- `encode_text.py` - CLI for encoding reviews
- `fit_itemknn.py` - fits + pickles ItemKNN (input to ensemble + two-stage)
- `run_text_ncf_all.sh` - full reproduction driver
- `compute_rmse.py` - extended (additively) with `--text-ncf-ckpt` /
  `--text-ncf-mt-ckpt` / `--text-ncf-subrating-ckpt` flags

**Outputs:**
- `results/text_ncf*/` - per-variant test metrics, rating metrics,
  ensemble/two-stage metrics, summary.md.
- `variants/pramod/notebooks/text_ncf.ipynb` - executed walkthrough.

## Touched shared code (non-breaking)

- `src/data/subratings.py` - fixed sub-rating column names (the parquet
  uses `service`, `cleanliness`, …, not `rating_<aspect>`; the loader was
  silently falling back to the overall rating).
- `scripts/compute_rmse.py` - added `--text-ncf-*` flags following
  Hriday's pattern for GMF / LightGCN-HG.

All models store text embeddings as PyTorch buffers so
`forward(users, items)` works with the shared `evaluate_ranking` code
without any other changes.

## Notebooks

- [`notebooks/text_ncf.ipynb`](notebooks/text_ncf.ipynb) - TextNCF family walkthrough: base + ablations + MT + sub-rating + ensemble + two-stage.

Shared notebooks at the repo root:

- [`../../notebooks/preprocessing.ipynb`](../../notebooks/preprocessing.ipynb)
- [`../../notebooks/baselines.ipynb`](../../notebooks/baselines.ipynb)
- [`../../notebooks/ensemble_and_summary.ipynb`](../../notebooks/ensemble_and_summary.ipynb)

## References

- He et al. (2017). *Neural Collaborative Filtering.* WWW.
- Reimers & Gurevych (2019). *Sentence-BERT.* EMNLP.
- Wang et al. (2020). *MiniLM: Deep Self-Attention Distillation.* NeurIPS.
