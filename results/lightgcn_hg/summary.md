# LightGCN-HG Results Summary

Metadata-augmented LightGCN variant (secondary). Extends the (user, item) bipartite graph from the LightGCN paper (He et al., SIGIR 2020) with TripAdvisor location nodes parsed from the hotel URL (g_id, region, country), giving hotels a way to share signal through geographic pivots rather than only through common reviewers. Same BPR loop as the standard LightGCN formulation; only the adjacency changes.

See `variants/hriday/PLAN.md` for the design doc and the decision trail that led to SASRec as the primary variant. Walkthrough in `variants/hriday/notebooks/04_lightgcn_hg.ipynb`.

## Run

K=1, dim=256, num_negatives=2, bpr_reg=1e-5, bs=8192, cosine LR, patience=15.

| Tiers | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | nodes | edges (dir) | best ep | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| g_id, region, country | 0.6460 | 0.7591 | 0.8655 | 0.5352 | 0.5718 | 0.5988 | 86,110 | 3,122,500 | 37 | 3160.23 |

## Ranking: LightGCN-HG vs Phase-1 baselines

| Method | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|
| Popularity | 0.3150 | 0.4215 | 0.5538 | 0.2318 | 0.2662 | 0.2995 |
| ItemKNN | 0.6835 | 0.6870 | 0.7091 | 0.6082 | 0.6093 | 0.6150 |
| GMF | 0.5553 | 0.6685 | 0.7936 | 0.4498 | 0.4863 | 0.5179 |
| **LightGCN-HG (g_id, region, country)** | **0.6460** | **0.7591** | **0.8655** | **0.5352** | **0.5718** | **0.5988** |

*LightGCN-HG improves HR@10 over ItemKNN by **+10.5%** relative.*

## Rating prediction (calibrated RMSE / MAE)

BPR doesn't output ratings, so we fit `rating = a * score + b` on the val split and report test RMSE / MAE. The slope ends up near zero, so RMSE ≈ GlobalMean (0.93). Popularity wins RMSE on HotelRec because 78 %% of ratings are 4-5 stars and item-mean is near-optimal.

| Method | RMSE | MAE | a | b |
|---|---|---|---|---|
| LightGCN-HG (g_id, region, country) | 0.9312 | 0.7025 | 0.0009 | 4.0655 |

## Run metadata

- `test_metrics_L1_d256_grc.json`: tiers=grc, K=1, dim=256, best_val_HR@10=0.7582, best_epoch=37, time=3160.23s, nodes=86110, edges_dir=3122500
