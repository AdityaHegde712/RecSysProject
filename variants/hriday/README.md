# Hriday's Variant - SASRec (primary) + LightGCN-HG (secondary)

There are 2 sub-variants under my variant folder:

- My primary variant is SASRec: Self-Attentive Sequential Recommendation
  (Kang & McAuley, ICDM 2018). It applies a causal-attention transformer over each user's time-ordered hotel sequence. Uses HotelRec's `date` column and it doesn't overlap with other variants.
- My secondary variant is LightGCN-HG: This variation is LightGCN extended with TripAdvisor geography nodes (`g_id` / region / country) parsed from `hotel_url`. I kept it as a feature-rich angle orthogonal to SASRec's temporal channel.

## Decision trail

My variant went through three iterations before settling on the final
SASRec (primary) + LightGCN-HG (secondary) structure.

1. I first started with vanilla LightGCN (He et al., SIGIR 2020) - symmetric-normalised graph convolution on the user-item bipartite graph. It was able to beat Popularity, GMF, and matched and beat ItemKNN on HR@10/20, but lost NDCG@5/10/20 to ItemKNN. Final HR@10 = 0.7532, NDCG@10 = 0.5677.
2. Based on Professor Eirinaki's feedback, I switched from LightGCN to LightGCN-HG. Using only (user, item, rating) makes the variant essentially a baseline against a feature-rich dataset. I built LightGCN-HG - bipartite graph extended with TripAdvisor location nodes. HG beats vanilla on every metric (HR@10 0.7591 vs 0.7532) but only by small margins. The graph-based ceiling looked low.
3. I later pivoted to SASRec as primary. I did a small model bakeoff (at `dim=64`, 8 epochs, matched 1-vs-99 eval) that showed SASRec dominated GRU4Rec and Mult-VAE by >2× HR@10 at the same compute. I scaled up to `dim=128` / 30 epochs, SASRec reached HR@10 = 0.8808 and NDCG@10 = 0.8392 (+28% / +38% relative vs ItemKNN).

## SASRec (primary)

### Approach

SASRec treats recommendation as next-item prediction: for each user,
sort their interactions by date, and ask the model to predict item `t`
given items `[0, 1, ..., t-1]`. The architecture is a small transformer
decoder: item + position embeddings --> two causal self-attention layers --> dot-product scoring of the last-position representation against a candidate item's embedding.

```
input: [pad, pad, ..., i_1, i_2, ..., i_{t-1}]  ->  h  ->  score = <h, e_c>
```

I trained with BPR over sampled negatives at each position. Then did standard 1-vs-99 evaluation with `torch.sort` tie-breaking, matching every other model in this repo.

### Why SASRec fits HotelRec
- It uses a signal no one else does, date. It has 100% coverage on the
  20-core.
- Trip structure - Hotel bookings are inherently sequential because vacations cluster in time, chains get re-visited, and seasonal patterns exist. Self-attention is a good inductive bias for exactly this pattern.

### Design decision log
| Decision | Alternatives considered | Choice | Why |
|---|---|---|---|
| Primary architecture | Vanilla LightGCN, GRU4Rec, Mult-VAE, BERT4Rec, SASRec | **SASRec** | Bakeoff at `dim=64` / 8 epochs: SASRec HR@10 0.83 vs GRU4Rec 0.49, Mult-VAE 0.46 (>2× margin). BERT4Rec skipped after Petrov & Macdonald (RecSys 2022) replicability study showed it doesn't consistently beat SASRec under normalised eval and bidirectional masked-LM is awkward for next-future-booking framing. |
| Embedding dim | 32, 64, 128, 256 | **128** | Bakeoff at 64 already beat LightGCN-HG; 128 captured slightly more without over-fitting at 30 epochs. 256 was tried and over-fit. |
| Number of attention layers | 1, 2, 4 | **2** | Per Kang & McAuley 2018 ablation; 1 layer underfits, 4 over-fits with our seq length. |
| Max sequence length | 50, 100, 200 | **100** | Covers the 95th-percentile user history on the 20-core (around 60 items); 200 just adds padding. |
| Loss | BCE, BPR, sampled softmax | **BPR (per-position)** | Matches the paper; 1-vs-99 ranking eval aligns naturally with the pairwise objective. |
| Negative sampling | 1, 2, 4 per position | **1** | Bakeoff showed no improvement from more at this scale; SASRec paper's default. |
| LR schedule | constant, step, cosine | **cosine to 1e-5** | Smooth annealing helped late-epoch NDCG without manual step tuning. |
| Early stopping | none, patience=3, 5, 10 | **patience=5** | Best val NDCG@10 hit at epoch 15; early stopped at 20 to save compute. |

### Results

Training: 30 epochs max, early-stopped at epoch 20 (best val at epoch 15).

| Method | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|
| Popularity | 0.3150 | 0.4215 | 0.5538 | 0.2318 | 0.2662 | 0.2995 |
| GMF | 0.5553 | 0.6685 | 0.7936 | 0.4498 | 0.4863 | 0.5179 |
| ItemKNN (k=20) | 0.6835 | 0.6870 | 0.7091 | 0.6082 | 0.6093 | 0.6150 |
| LightGCN-HG (3-tier, secondary) | 0.6460 | 0.7591 | 0.8655 | 0.5352 | 0.5718 | 0.5988 |
| **SASRec (dim=128, L=2)** | **0.8502** | **0.8808** | **0.9173** | **0.8294** | **0.8392** | **0.8484** |

Deltas vs best baseline (ItemKNN): HR@10 +28.2 % rel, NDCG@10 +37.7 % rel.
NDCG gains are especially striking because SASRec reliably places the correct hotel very high in the ranked list.

### Why Popularity wins RMSE
Popularity predicts each item's mean training rating. On HotelRec, 78%
of ratings are 4 or 5 stars, so the item-mean already captures most of
the rating variance. Any ranking-trained model (BPR loss, no explicit
rating target) has a near-zero calibration slope a ≈ 0. Its
"calibrated" rating is essentially a constant near the global mean
(~4.08). That constant predictor has RMSE ≈ 0.93, losing to Popularity's
0.8685. Ranking is the primary metric for this variant.

## LightGCN-HG (secondary)

### Approach

It extends the (user, item) bipartite graph from the LightGCN paper with
three tiers of TripAdvisor geography parsed from hotel_url:
- `g_id`         - TripAdvisor location id (one per city / neighbourhood)
- `region_slug`  - last 2 underscore tokens of the URL tail
- `country_slug` - last 1 token

Each hotel adds one edge to each tier node. Tier nodes connect nowhere
else, they act as pivot hubs so co-located hotels share signal even
when they have no shared reviewer. Same BPR loop, cosine LR, early stop;
only the adjacency changes.

### Design decision log

| Decision | Alternatives | Choice | Why |
|---|---|---|---|
| Graph tiers | bipartite only, +g_id, +g_id+region, +g_id+region+country | **+g_id+region+country (3-tier)** | Each adds incremental edges. 3-tier gives the most pivot connectivity without redundant noise. |
| LightGCN layers `K` | 1, 2, 3 | **1** | Higher K oversmoothed on this graph. |
| Embed dim | 64, 128, 256 | **256** | Larger dim helps graph CF more than seq models as per the LightGCN paper. |
| Negative sampling | 1, 2, 4 | **2** | Sweet spot - 1 was unstable, 4 slowed training without metric lift. |
| BPR L2 reg | 0, 1e-5, 1e-4 | **1e-5** | LightGCN authors' recommendation. Light reg works on a sparse graph. |
| Patience | 5, 10, 15 | **15** | Graph CF plateaus then dips. Longer patience caught the late peak at epoch 37. |

### Graph size

Metadata extracted by `scripts/extract_hotel_meta.py`:
| Tier         | Unique nodes | Singleton nodes |
|--------------|-------------:|----------------:|
| g_id         | 7,760 | 4,079 (53%) |
| region_slug  | 3,706 | 1,589 (6%)  |
| country_slug |   787 |   299 (1%)  |

Adding all three tiers grows the graph from 73,857 to 86,110 nodes
and around 2.96M to around 3.12M directed edges.

### Results

K=1, dim=256, num_negatives=2, bpr_reg=1e-5, bs=8192, patience=15.
Trained to early stop at epoch 52; best checkpoint from epoch 37.

| Method | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|
| Popularity | 0.3150 | 0.4215 | 0.5538 | 0.2318 | 0.2662 | 0.2995 |
| GMF | 0.5553 | 0.6685 | 0.7936 | 0.4498 | 0.4863 | 0.5179 |
| ItemKNN (k=20) | **0.6835** | 0.6870 | 0.7091 | **0.6082** | **0.6093** | **0.6150** |
| Vanilla LightGCN (bipartite) | 0.6414 | 0.7532 | 0.8612 | 0.5315 | 0.5677 | 0.5950 |
| **LightGCN-HG (3-tier)** | 0.6460 | **0.7591** | **0.8655** | 0.5352 | 0.5718 | 0.5988 |

It beat every baseline on HR@10/20 but lost NDCG@5/10/20 to ItemKNN (very
concentrated on top-1 placement). SASRec dominates on every metric. HG
is kept as a secondary angle, not the lead result.

**Vanilla vs HG A/B** (vanilla-vs-enhanced ask):
LightGCN-HG beats vanilla bipartite by +0.0059 HR@10 and +0.0041 NDCG@10
- It's small but consistent. The calibrated RMSE is identical (0.9312) because the geography augmentation moves ranking, not rating.

## Files
- `src/data/sequential.py` - chronological per-user sequence builder,
  `NextItemDataset` (training), `SequentialEvalDataset` (1-vs-99 eval).
- `src/models/sasrec.py` - SASRec model.
- `src/models/lightgcn_hg.py` - heterogeneous LightGCN model.
- `src/graph/hetero_adj.py` - torch-free scipy graph builder.
- `src/train_sasrec.py` - SASRec trainer (BPR + cosine LR + early stop).
- `src/train_lightgcn_hg.py` - LightGCN-HG trainer with `--tiers` flag.
- `configs/sasrec.yaml` - `dim=128`, `max_seqlen=100`, 2 layers, 30 epochs.
- `configs/lightgcn_hg.yaml` - `K=1`, `dim=256`, 3-tier default.
- `scripts/extract_hotel_meta.py` - URL parser → `hotel_meta.parquet`.
- `results/sasrec/`, `results/lightgcn_hg/` - test metrics, calibrated
  RMSE, checkpoints, log CSVs, summaries.

## Run

```bash
# SASRec primary
python -m src.train_sasrec --config configs/sasrec.yaml --kcore 20

# LightGCN-HG (extract geography metadata first)
python -m scripts.extract_hotel_meta --kcore 20

# Vanilla bipartite ablation
python -m src.train_lightgcn_hg --config configs/lightgcn_hg.yaml \
    --kcore 20 --tiers none

# HG variant (default - uses g_id + region + country pivots)
python -m src.train_lightgcn_hg --config configs/lightgcn_hg.yaml --kcore 20

# Calibrated RMSE for both LightGCN checkpoints
python scripts/compute_rmse.py --kcore 20 \
    --lightgcn-hg-ckpt results/lightgcn_hg/best_model_L1_d256_none.pt \
    --lightgcn-hg-dim 256 --lightgcn-hg-layers 1 --lightgcn-hg-tiers none

python scripts/compute_rmse.py --kcore 20 \
    --lightgcn-hg-ckpt results/lightgcn_hg/best_model_L1_d256_grc.pt \
    --lightgcn-hg-dim 256 --lightgcn-hg-layers 1
```

## Notebooks

- [`notebooks/lightgcn_hg.ipynb`](notebooks/lightgcn_hg.ipynb) -
  LightGCN-HG: graph construction, training, evaluation, vanilla-vs-HG A/B.
- [`notebooks/sasrec.ipynb`](notebooks/sasrec.ipynb) -
  SASRec: bakeoff justification, model walkthrough, training curves, final results.

## References

- Kang & McAuley (2018). *Self-Attentive Sequential Recommendation.* ICDM.
- He et al. (2020). *LightGCN.* SIGIR.
- Hidasi et al. (2016). *Session-based Recommendations with RNNs.* ICLR.
- Liang et al. (2018). *Variational Autoencoders for Collaborative Filtering.* WWW.
- Sun et al. (2019). *BERT4Rec.* CIKM.
- Petrov & Macdonald (2022). *A Systematic Review and Replicability Study of BERT4Rec.* RecSys.
