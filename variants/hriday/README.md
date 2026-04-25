# Variant A: SASRec (primary) + LightGCN-HG (secondary)

**Owner:** Hriday Ampavatina

Two sub-variants under this folder:

- **A1 - SASRec** (primary): Self-Attentive Sequential Recommendation
  (Kang & McAuley, ICDM 2018). Causal-attention transformer over each
  user's time-ordered hotel sequence. Uses HotelRec's `date` column,
  which no teammate's variant consumes.
- **A2 - LightGCN-HG** (secondary): LightGCN extended with TripAdvisor
  geography nodes (g_id / region / country) parsed from `hotel_url`.
  Kept as a backup feature-rich angle in case the sequential signal
  doesn't translate.

> **Note on the journey:** This variant initially started as **LightGCN**
> (graph-based CF over the bipartite user-item graph). Instructor feedback
> pushed us to use the dataset's richer features, which produced
> **LightGCN-HG** (the secondary variant here). After some experimentation
> with sequential models, we pivoted to **SASRec** as the primary because
> it uses the `date` signal that neither teammate's variant touches and
> it's architecturally orthogonal to both (transformer, not graph conv or
> MLP). LightGCN-HG is retained as a secondary for breadth.
>
> See [`PLAN.md`](PLAN.md) for the full decision trail.

---

## A1. SASRec (primary)

### Approach

SASRec treats recommendation as **next-item prediction**: for each user,
sort their interactions by date, and ask the model to predict item `t`
given items `[0, 1, ..., t-1]`. The architecture is a small transformer
decoder - item + position embeddings → two causal self-attention layers →
dot-product scoring of the last-position representation against a
candidate item's embedding.

```
input:  [pad, pad, ..., i_1, i_2, ..., i_{t-1}]  ->  h  ->  score = <h, e_c>
```

Trained with BPR over sampled negatives at each position. Standard 1-vs-99
evaluation with `torch.sort` tie-breaking, matching every other model in
this repo.

### Why SASRec fits HotelRec

- **Uses a signal no one else does.** `date` has 100 % coverage on the
  20-core. Aditya's NeuMF uses sub-ratings; Pramod's Text-NCF uses
  review text; none of the baselines use time.
- **Trip structure.** Hotel bookings are inherently sequential -
  vacations cluster in time, chains get re-visited, seasonal patterns
  exist. Self-attention is a good inductive bias for exactly this
  pattern.

### Files

- `src/data/sequential.py` - chronological per-user sequence builder,
  `NextItemDataset` (training), `SequentialEvalDataset` (1-vs-99 eval).
- `src/models/sasrec.py` - SASRec model.
- `src/train_sasrec.py` - training driver (BPR + cosine LR + early stop).
- `configs/sasrec.yaml` - dim=128, max_seqlen=100, 2 layers, 2 heads,
  30 epochs, patience=5.
- `results/sasrec/` - test metrics, calibrated RMSE, checkpoints, log CSV.

### Run

```bash
python -m src.train_sasrec --config configs/sasrec.yaml --kcore 20
```

### Results (dim=128, L=2, max_seqlen=100)

Trained for 30 epochs max, early-stopped at epoch 20 (patience=5). Best
validation at epoch 15. Total training time 869 s (~14.5 min on one RTX 5070 Ti).

| Method | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|
| Popularity | 0.3150 | 0.4215 | 0.5538 | 0.2318 | 0.2662 | 0.2995 |
| GMF | 0.5553 | 0.6685 | 0.7936 | 0.4498 | 0.4863 | 0.5179 |
| ItemKNN (k=20) | 0.6835 | 0.6870 | 0.7091 | 0.6082 | 0.6093 | 0.6150 |
| LightGCN-HG (3-tier, secondary) | 0.6460 | 0.7591 | 0.8655 | 0.5352 | 0.5718 | 0.5988 |
| **SASRec (dim=128, L=2)** | **0.8502** | **0.8808** | **0.9173** | **0.8294** | **0.8392** | **0.8484** |

**Deltas vs best baseline (ItemKNN):**

| Metric | Δ (absolute) | Δ (relative) |
|---|---|---|
| HR@5   | +0.1667 | +24.4% |
| HR@10  | +0.1938 | +28.2% |
| HR@20  | +0.2082 | +29.4% |
| NDCG@5 | +0.2212 | +36.4% |
| NDCG@10| +0.2299 | +37.7% |
| NDCG@20| +0.2334 | +38.0% |

SASRec wins every ranking metric by wide margins. The NDCG gains are
especially striking - **+37.7 %** relative on NDCG@10 means SASRec
reliably places the correct hotel very high in the ranked list.

### Why Popularity wins RMSE (and why that's fine)

Popularity predicts each item's mean training rating. On HotelRec, 78 %
of ratings are 4 or 5 stars, so the item-mean already captures most of
the rating variance. Any ranking-trained model (BPR loss, no explicit
rating target) will have a near-zero calibration slope `a ≈ 0` - its
"calibrated" rating is essentially a constant near the global mean
(~4.08). That constant predictor has RMSE ≈ 0.93, losing to Popularity's
0.8685.

This pattern is consistent across every BPR model we trained (GMF,
LightGCN-HG, SASRec) - all at RMSE 0.93 ± 0.01. It's a property of the
loss function + label distribution, not of any particular model.
**Ranking is the primary metric.**

---

## A2. LightGCN-HG (secondary)

### Motivation

Retained as a secondary because:

1. It addresses the "use the features" critique through a different
   channel (geographic metadata, not time). If the sequential angle
   turns out to be dataset-specific, HG still stands as a valid
   feature-rich variant.
2. Different research angle for the 15–20 min presentation - one
   sequential model + one graph-based model makes a richer story.

### Approach

Extends the (user, item) bipartite graph from the LightGCN paper
(He et al., SIGIR 2020) with three tiers of TripAdvisor geography parsed
from `hotel_url`:

- `g_id`         - TripAdvisor location id (one per city / neighbourhood)
- `region_slug`  - last 2 underscore tokens of the URL tail
- `country_slug` - last 1 token

Each hotel adds one edge to each tier node. Tier nodes connect nowhere
else - they act as pivot hubs so co-located hotels share signal even
when they have no shared reviewer. Same BPR loop, cosine LR, early stop;
only the adjacency changes.

### Graph size

Metadata extracted by `scripts/extract_hotel_meta.py` (writes to a new
`data/processed/hotel_meta/`, never touches the shared 20-core splits):

| Tier         | Unique nodes | Singleton nodes |
|--------------|-------------:|----------------:|
| g_id         | 7,760 | 4,079 (53%) |
| region_slug  | 3,706 | 1,589 (6%)  |
| country_slug |   787 |   299 (1%)  |

Adding all three tiers grows the graph from 73,857 to **86,110 nodes**
and ~2.96M to **~3.12M directed edges**.

### Results (K=1, dim=256, num_negatives=2, bpr_reg=1e-5, bs=8192, patience=15)

Trained to early stop at epoch 52; best checkpoint from epoch 37.

| Method | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|
| Popularity | 0.3150 | 0.4215 | 0.5538 | 0.2318 | 0.2662 | 0.2995 |
| GMF | 0.5553 | 0.6685 | 0.7936 | 0.4498 | 0.4863 | 0.5179 |
| ItemKNN (k=20) | **0.6835** | 0.6870 | 0.7091 | **0.6082** | **0.6093** | **0.6150** |
| Vanilla LightGCN (bipartite, no tiers) | 0.6414 | 0.7532 | 0.8612 | 0.5315 | 0.5677 | 0.5950 |
| **LightGCN-HG (3-tier)** | 0.6460 | **0.7591** | **0.8655** | 0.5352 | 0.5718 | 0.5988 |

Beats every baseline on HR@10/20 by comfortable margins. Loses NDCG@5/10/20
to ItemKNN because ItemKNN is very concentrated on top-1 placement. SASRec
(primary) dominates this comparison on every metric - HG is kept as a
secondary angle, not the lead result.

**Vanilla vs HG A/B** (added for the day-10 vanilla-vs-enhanced ask):
LightGCN-HG beats the vanilla bipartite LightGCN by +0.0059 HR@10 and
+0.0041 NDCG@10 — small but consistent across every k. The geography
augmentation works as advertised; the lift is just modest because most
of the signal is already in the user-item bipartite graph. Calibrated
RMSE is identical (0.9312) — the augmentation moves ranking, not rating.

Run vanilla via `--tiers none`:
```bash
python -m src.train_lightgcn_hg --config configs/lightgcn_hg.yaml --kcore 20 --tiers none
```

### Files

- `src/graph/hetero_adj.py` - torch-free scipy builder.
- `src/models/lightgcn_hg.py` - heterogeneous LightGCN.
- `src/train_lightgcn_hg.py` - training driver with `--tiers` flag.
- `configs/lightgcn_hg.yaml` - K=1, dim=256, 3-tier default.
- `scripts/extract_hotel_meta.py` - URL parser → `hotel_meta.parquet`.

### Run

```bash
python -m scripts.extract_hotel_meta --kcore 20

# Vanilla bipartite LightGCN (vanilla-vs-enhanced A/B)
python -m src.train_lightgcn_hg --config configs/lightgcn_hg.yaml --kcore 20 --tiers none

# HG variant (default — uses g_id, region, country pivots)
python -m src.train_lightgcn_hg --config configs/lightgcn_hg.yaml --kcore 20

# Calibrated RMSE for both checkpoints
python scripts/compute_rmse.py --kcore 20 \
    --lightgcn-hg-ckpt results/lightgcn_hg/best_model_L1_d256_none.pt \
    --lightgcn-hg-dim 256 --lightgcn-hg-layers 1 --lightgcn-hg-tiers none
python scripts/compute_rmse.py --kcore 20 \
    --lightgcn-hg-ckpt results/lightgcn_hg/best_model_L1_d256_grc.pt \
    --lightgcn-hg-dim 256 --lightgcn-hg-layers 1
python scripts/summarize_lightgcn_hg.py
```

---

## Notebooks

- [`notebooks/04_lightgcn_hg.ipynb`](notebooks/04_lightgcn_hg.ipynb) - LightGCN-HG (A2, secondary): graph construction, training, evaluation.
- [`notebooks/06_sasrec.ipynb`](notebooks/06_sasrec.ipynb) - SASRec (A1, primary): model walkthrough, training curves, final results.

Shared notebooks at the repo root:

- [`../../notebooks/01_preprocessing.ipynb`](../../notebooks/01_preprocessing.ipynb)
- [`../../notebooks/02_baselines.ipynb`](../../notebooks/02_baselines.ipynb)
- [`../../notebooks/05_ensemble_and_summary.ipynb`](../../notebooks/05_ensemble_and_summary.ipynb)
