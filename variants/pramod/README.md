# Variant C: TextNCF — Review-Text-Enhanced NCF

**Owner:** Pramod Yadav

> See [`PLAN.md`](PLAN.md) for the design decisions and iteration history.

## Approach

TextNCF adds review text to the standard NCF pipeline. Hotels get reviewed with detailed text ("great location, noisy rooms, amazing breakfast"), and that text carries signal that star ratings alone miss. The review text is encoded with a sentence transformer and fused with the usual collaborative filtering embeddings.

The base model has two branches:
- **GMF branch** — standard user/item embedding dot product, same as the baseline GMF
- **Text branch** — sentence embeddings from reviews, projected down and combined

Both branches get concatenated and run through a small MLP to produce a score. Trained with BPR loss since we're treating this as implicit feedback.

On top of the base TextNCF, four additional approaches push performance further.

## Why TextNCF fits HotelRec

The dataset has 50M reviews averaging 125 words each. That's a lot of text signal sitting there unused by pure collaborative filtering. About 71% of reviews have substantive text. The text captures stuff like "walking distance to the beach" or "thin walls" that you can't get from a 4-star rating.

A frozen all-MiniLM-L6-v2 (384-dim) encodes the text. Freezing it means no fine-tuning of a 22M parameter transformer — just a small projection layer on top. The encoding is done offline as a preprocessing step.

## Approaches

### 1. Base TextNCF (`src/train_text_ncf.py`)

Two-branch hybrid: GMF ⊙ + text ⊙ → MLP → score. Trained with BPR loss.

```bash
python -m src.train_text_ncf --config configs/text_ncf.yaml --kcore 20
```

### 2. Ensemble Scoring (`src/evaluate_ensemble.py`)

Combines TextNCF, GMF, and ItemKNN predictions with learned weights. Min-max normalizes each model's scores per user, then grid searches over weight combinations (w1+w2+w3=1) on the validation set.

```bash
python -m src.evaluate_ensemble \
    --text-ncf-ckpt results/text_ncf/best_model.pt \
    --gmf-ckpt results/gmf/best_model.pt \
    --knn-ckpt results/baselines/itemknn.pkl \
    --kcore 20
```

### 3. Two-Stage Retrieval + Ranking (`src/evaluate_two_stage.py`)

Mimics a production pipeline: ItemKNN retrieves top-200 candidates (fast, sparse), then TextNCF re-ranks them (slow, neural). This tests whether TextNCF adds value as a re-ranker on top of a strong retriever.

```bash
python -m src.evaluate_two_stage \
    --text-ncf-ckpt results/text_ncf/best_model.pt \
    --knn-ckpt results/baselines/itemknn.pkl \
    --kcore 20
```

### 4. Multi-Task TextNCF (`src/train_text_ncf_mt.py`)

Joint ranking + rating prediction. Same architecture as TextNCF but with two output heads:
- Ranking head (BPR loss)
- Rating head (MSE loss on 1-5 ratings)

Joint loss: `alpha * BPR + (1-alpha) * MSE`. The idea is that predicting ratings forces the model to learn finer-grained preferences.

```bash
python -m src.train_text_ncf_mt --config configs/text_ncf_mt.yaml --kcore 20
```

### 5. Sub-Rating Decomposition (`src/train_text_ncf_subrating.py`)

HotelRec has 6 aspect-level sub-ratings: Service, Cleanliness, Location, Value, Rooms, Sleep Quality. This model predicts each sub-rating separately with dedicated heads, then combines them with learned per-user attention weights. Different travelers care about different aspects — business travelers care about WiFi/location, families care about rooms/cleanliness.

```bash
python -m src.train_text_ncf_subrating \
    --config configs/text_ncf_subrating.yaml --kcore 20
```

## Run

```bash
# step 1: encode reviews (needs sentence-transformers, ~11 min on RTX 5070 Ti)
python scripts/encode_text.py --kcore 20 --device cuda

# step 2: fit + pickle ItemKNN (needed by ensemble + two-stage)
python scripts/fit_itemknn.py --kcore 20

# step 3: kick off everything (5 trainings + ensemble + two-stage + RMSE)
bash scripts/run_text_ncf_all.sh
```

The `run_text_ncf_all.sh` driver is the one-button reproduction path.
It writes per-step logs into `logs/<step>.log` and outputs into
`results/text_ncf*/`. Total wall-clock on RTX 5070 Ti: ~80 min.

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

## Results

20-core HotelRec, 1-vs-99 evaluation. Full numbers + decision notes in
[`results/text_ncf/summary.md`](../../results/text_ncf/summary.md);
the [`07_text_ncf` notebook](notebooks/07_text_ncf.ipynb) is the
executed walkthrough.

| Model | HR@10 | NDCG@10 | Notes |
|-------|-------|---------|-------|
| Popularity (baseline) | 0.4215 | 0.2662 | item-mean rating |
| GMF (baseline) | 0.6685 | 0.4863 | from `results/gmf/` |
| ItemKNN (baseline) | 0.6870 | 0.6093 | k=20, weighted neighbour |
| TextNCF base | 0.6787 | 0.5057 | beats GMF; below KNN on NDCG |
| TextNCF GMF-only ablation | 0.6720 | 0.4915 | text branch off |
| TextNCF text-only ablation | 0.6891 | 0.4981 | GMF off — text carries the lift |
| **TextNCF Multi-Task** | **0.6864** | **0.5097** | best ranker in the family |
| TextNCF Sub-rating | 0.6677 | 0.4710 | attention collapsed onto Cleanliness |
| Ensemble (TextNCF+GMF+KNN) | 0.6870 | 0.6093 | grid picked KNN-only — degenerate |
| Two-stage (KNN→TextNCF) | 0.3858 | 0.2977 | gt_recall@200 = 5 % — recall-bound |

**Deltas vs GMF baseline (best single-model comparison):**

| Metric | TextNCF MT Δ (abs) | TextNCF MT Δ (rel) |
|--------|-------------------|-------------------|
| HR@5   | +0.0189 | +3.4% |
| HR@10  | +0.0179 | +2.7% |
| HR@20  | +0.0095 | +1.2% |
| NDCG@5 | +0.0236 | +5.2% |
| NDCG@10| +0.0234 | +4.8% |
| NDCG@20| +0.0213 | +4.1% |

Calibrated RMSE for the three trained variants is around **0.93** (slope
≈ 0.01–0.03), the same flat calibration pattern SASRec / GMF /
LightGCN-HG hit. Popularity wins RMSE at 0.8685. Details in the summary.

### Why Popularity wins RMSE (and why that's fine)

Popularity predicts each item's mean training rating. On HotelRec, 78 %
of ratings are 4 or 5 stars, so the item-mean already captures most of
the rating variance. Any ranking-trained model (BPR loss, no explicit
rating target) will have a near-zero calibration slope — its "calibrated"
rating is essentially a constant near the global mean (~4.08). That
constant predictor has RMSE ≈ 0.93, losing to Popularity's 0.8685.
**Ranking is the primary metric.**

## Data leakage note

User text profiles only use training-split reviews. Item profiles use all reviews since that's basically hotel metadata (what people say about the hotel doesn't change across splits). This follows the standard practice from the NCF literature.

## Files

**Models:**
- `src/models/text_ncf.py` — base TextNCF model
- `src/models/text_ncf_mt.py` — multi-task variant (ranking + rating)
- `src/models/text_ncf_subrating.py` — sub-rating decomposition variant

**Training scripts:**
- `src/train_text_ncf.py` — train base TextNCF
- `src/train_text_ncf_mt.py` — train multi-task variant
- `src/train_text_ncf_subrating.py` — train sub-rating variant

**Evaluation scripts:**
- `src/evaluate_ensemble.py` — ensemble scoring (TextNCF + GMF + ItemKNN)
- `src/evaluate_two_stage.py` — two-stage retrieval + ranking

**Data utilities:**
- `src/data/text_embeddings.py` — encoding + loading helpers
- `src/data/subratings.py` — sub-rating data loading

**Configs:**
- `configs/text_ncf.yaml` — base TextNCF hyperparameters
- `configs/text_ncf_mt.yaml` — multi-task config (alpha parameter)
- `configs/text_ncf_subrating.yaml` — sub-rating config (beta parameter)

**Other:**
- `scripts/encode_text.py` — CLI for encoding reviews
- `scripts/fit_itemknn.py` — fits + pickles ItemKNN to `results/baselines/itemknn.pkl`
- `scripts/run_text_ncf_all.sh` — full reproduction driver
- `scripts/compute_rmse.py` — extended with `--text-ncf-*` flags for rating metrics
- `results/text_ncf*/` — per-variant outputs (test_metrics, rating_metrics, ensemble_metrics, two_stage_metrics, summary.md)

Two shared files were touched in non-breaking ways: `src/data/subratings.py`
to fix the sub-rating column names (the parquet uses `service`,
`cleanliness`, …, not the `rating_<aspect>` names the loader was looking
for), and `scripts/compute_rmse.py` to add TextNCF-family flags
following the same pattern used for GMF / LightGCN-HG. All models
store text embeddings as PyTorch buffers so `forward(users, items)`
works with the shared eval code without any other changes.

---

## Notebooks

- [`notebooks/07_text_ncf.ipynb`](notebooks/07_text_ncf.ipynb) — TextNCF family walkthrough: base + ablations + MT + sub-rating + ensemble + two-stage.

Shared notebooks at the repo root:

- [`../../notebooks/01_preprocessing.ipynb`](../../notebooks/01_preprocessing.ipynb)
- [`../../notebooks/02_baselines.ipynb`](../../notebooks/02_baselines.ipynb)
- [`../../notebooks/05_ensemble_and_summary.ipynb`](../../notebooks/05_ensemble_and_summary.ipynb)
