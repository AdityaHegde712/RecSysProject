# Variant B - NeuMF with Attention-Weighted Sub-Ratings

**Owner:** Aditya Hegde

## Approach

Neural Matrix Factorization (He et al., WWW 2017) extended with a per-user
attention layer over the six HotelRec sub-rating aspects (Service,
Cleanliness, Location, Value, Rooms, Sleep Quality).

The attention layer learns which aspect matters most to each user, and
the weighted "quality score" is fused into the final prediction layer
alongside the GMF and MLP branches.

```
user, item ─► gmf_user_emb ⊙ gmf_item_emb ──────────┐
                                                       │
              mlp_user_emb || mlp_item_emb ─► MLP ─────┤── concat ─► linear ─► score
                                                       │
              user_emb ─► attn_proj ─► softmax ─► (B, 6)
                                                  ⊙
              item_aspects[item] (B, 6) ──► sum (B, 1) ┘
                                          quality score
```

## Why this fits HotelRec

- HotelRec ships rich aspect-level metadata that pure CF baselines ignore.
- Different user types weight aspects differently (business travellers
  → location/WiFi, families → rooms/cleanliness).
- NeuMF is a strong neural CF backbone; sub-rating attention adds both
  accuracy potential and a hook for interpretability.

## Decision trail

The variant was scoped in two iterations:

1. **Initial design:** plain NeuMF + attention over per-interaction
   sub-ratings (treating each review as having its own (service,
   cleanliness, …) tuple). Dropped because that mixes user preference
   with item quality - at inference you don't have a fresh interaction's
   sub-ratings to attend over.
2. **Final design:** item-level aspect vectors. Pre-compute each hotel's
   train-split-mean sub-ratings (frozen at training start), and learn
   per-user attention weights over those six fixed dimensions. Feature
   engineering done once, no leakage from val/test ratings.

Day-10 vanilla-vs-enhanced ablation revealed the attention head adds
**zero signal** on this dataset (numbers below). The decision is
documented as the headline finding rather than buried - see "What the
attention learned" in
[`results/neumf_attn/summary.md`](../../results/neumf_attn/summary.md).

## Design decision log

| Decision | Alternatives | Choice | Why |
|---|---|---|---|
| Backbone | GMF, MLP, NeuMF (GMF+MLP), Wide&Deep | **NeuMF** | He et al. WWW 2017 standard; combines linear (GMF) and non-linear (MLP) interaction modelling. |
| Aspect aggregation | per-interaction, per-item train-mean, per-user train-mean | **per-item train-mean** | Avoids leaking val/test sub-ratings; aligns with how attention weights are used at inference. |
| Attention granularity | per-interaction, per-user (over fixed item aspects) | **per-user** | Item aspects are fixed; attention varies by user preference, which is what we want to model. |
| Aspect attention activation | sigmoid (independent), softmax (competitive), sparsemax | **softmax** | Standard; later analysis showed it collapsed to nearly-uniform - sparsemax / entropy-bonus is the noted follow-up. |
| Embed dim (gmf/mlp) | 32, 64, 128 | **64 / 64** | Bigger dims over-fit at 50 epochs on this user-count; 64 is the sweet spot per a quick 5-epoch sweep. |
| MLP shape | [128,64], [256,128,64], [512,256,128] | **[256,128,64]** | 3-layer pyramid stable; deeper dropped val NDCG without HR gain. |
| Loss | BPR, BCE, sampled softmax | **BPR (1 neg)** | Matches the rest of the team's variants for direct comparability under the shared 1-vs-99 protocol. |
| Negative sampling | 1, 2, 4 per positive | **4** | Stronger gradient signal helps the deeper MLP; ablation showed +0.005 HR@10 over 1 neg. |
| Calibration for RMSE | none, lstsq on val | **lstsq on val** | Same methodology as every other ranking model in the repo so RMSE is comparable. |
| Run length | 50 ep / patience 10 (this run); 200 ep / patience 30 (originally proposed) | **50 / 10** | Wall-clock budget; HR@10 was still creeping at epoch 50 but val_loss began an uptick. Longer runs would likely improve modestly - flagged as future work. |

## Vanilla vs Enhanced (day-10 instructor ask)

The two configs (`configs/neumf_attn.yaml` and `configs/neumf_vanilla.yaml`)
differ **only** in `model.use_attention`:

| Variant | HR@10 | NDCG@10 | RMSE |
|---|---|---|---|
| Vanilla NeuMF (GMF + MLP, no attention) | **0.7254** | **0.5228** | 0.9304 |
| Enhanced NeuMF-Attn | 0.7245 | 0.5221 | 0.9304 |

**The sub-rating attention head adds zero signal on HotelRec.** All six
ranking metrics shift by ≤ 0.001 (within run-to-run BPR-sampler variance);
RMSE is identical to four decimals. The +8.5 % rel HR@10 gain over the
plain GMF baseline is driven by the MLP branch + bigger embeddings, not
by the attention head.

**Diagnosis:** HotelRec's six aspect averages are tightly correlated and
nearly always 4-5 across the board, so softmax-over-aspects collapses
to near-uniform regardless of user. Same dataset-level pattern Pramod's
TextNCF sub-rating variant hit. Sparsemax / entropy-bonus is the natural
next experiment.

## Cross-team headline ranking (test split)

| Model | HR@10 | NDCG@10 | Notes |
|-------|-------|---------|-------|
| Popularity (baseline) | 0.4215 | 0.2662 | item-mean rating |
| GMF (baseline) | 0.6685 | 0.4863 | from `results/gmf/` |
| ItemKNN (baseline) | 0.6870 | 0.6093 | k=20 |
| TextNCF Multi-Task (Pramod) | 0.6864 | 0.5097 | |
| **Vanilla NeuMF (this variant)** | **0.7254** | **0.5228** | best HR@10 among non-sequential neural variants |
| Enhanced NeuMF-Attn | 0.7245 | 0.5221 | attention adds zero - see vanilla-vs-enhanced section |
| LightGCN-HG (Hriday secondary) | 0.7591 | 0.5718 | different feature channel (geography) |
| SASRec (Hriday primary) | 0.8808 | 0.8392 | sequence model |

Calibrated RMSE on all NeuMF runs = **0.9304** (slope ≈ 0.02, `b ≈ 4.07`)
- same flat-calibration pattern every BPR-trained ranker hits on
HotelRec. Popularity wins RMSE at 0.8685.

## How to run

```bash
# Pre-req: data/processed/20core/*.parquet (shared preprocessing)

# Enhanced (sub-rating attention head on)
python -m src.train_neumf_attn --config configs/neumf_attn.yaml --kcore 20

# Vanilla ablation (no attention) - day-10 vanilla-vs-enhanced ask
python -m src.train_neumf_attn --config configs/neumf_vanilla.yaml --kcore 20
```

The training script does training + 1-vs-99 test eval + score-to-rating
calibration in one pass. Wall-clock per run on RTX 5070 Ti: **~96-115 min**
(50 epochs × ~2 min/epoch).

## Files

- `src/models/neumf_attn.py` - `NeuMF_Attn` model (GMF + MLP + sub-rating
  attention + fusion). Takes `use_attention: bool` so the same class
  powers both vanilla and enhanced runs.
- `src/train_neumf_attn.py` - training + eval + calibration pipeline.
- `configs/neumf_attn.yaml` - enhanced config.
- `configs/neumf_vanilla.yaml` - vanilla ablation (only diff: `use_attention: false`).
- `variants/aditya/notebooks/neumf_attn.ipynb` - executed walkthrough.
- `results/neumf_attn/`, `results/neumf_vanilla/` - per-config test
  metrics, rating metrics, summaries.

## Data leakage note

Item aspect vectors (the `(n_items, 6)` sub-rating matrix the attention
reads) are computed from the **train split only**. Missing per-item
values are filled with the train-split column mean; a fully missing
column (none observed on HotelRec) would fall back to 3.0.

## Non-overlap with teammates

- **SASRec (Hriday):** different input (time sequence) and architecture
  (transformer decoder). Zero overlap.
- **LightGCN-HG (Hriday):** different feature channel (graph + geography).
- **TextNCF (Pramod):** different auxiliary channel (review text). Pramod's
  sub-rating *variant* uses the same six columns but as regression
  targets, not as attention-weighted inputs - the decomposition direction
  is different.

## References

- He, Liao, Zhang, Nie, Hu, Chua (2017). *Neural Collaborative Filtering.* WWW.
