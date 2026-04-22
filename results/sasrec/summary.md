# SASRec Results Summary

Self-Attentive Sequential Recommendation (Kang & McAuley, ICDM 2018). Causal self-attention over per-user time-ordered hotel sequences, BPR loss, shared 1-vs-99 evaluation.

See `variants/hriday/PLAN.md` for the decision trail from LightGCN-HG to SASRec as the primary variant.

## Full-scale runs

| dim | layers | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | best ep | time (s) |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 2 | 0.8502 | 0.8808 | 0.9173 | 0.8294 | 0.8392 | 0.8484 | 15 | 869.4 |

## SASRec vs Phase-1 baselines and LightGCN-HG (secondary)

| Method | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|
| Popularity | 0.3150 | 0.4215 | 0.5538 | 0.2318 | 0.2662 | 0.2995 |
| ItemKNN | 0.6835 | 0.6870 | 0.7091 | 0.6082 | 0.6093 | 0.6150 |
| GMF | 0.5553 | 0.6685 | 0.7936 | 0.4498 | 0.4863 | 0.5179 |
| LightGCN-HG (g_id+region+country) | 0.6460 | 0.7591 | 0.8655 | 0.5352 | 0.5718 | 0.5988 |
| **SASRec (dim=128, L=2)** | **0.8502** | **0.8808** | **0.9173** | **0.8294** | **0.8392** | **0.8484** |

*SASRec improves HR@10 over ItemKNN by **+28.2%** relative, NDCG@10 by **+37.7%**.*

## Rating prediction (calibrated)

BPR doesn't output ratings natively; we fit `rating = a * score + b` on val, evaluate on test. The slope ends up near zero, so RMSE ≈ GlobalMean (0.93). Popularity still wins RMSE on HotelRec because 78 %% of ratings are 4-5 stars.

| dim / L | RMSE | MAE | a | b | note |
|---|---|---|---|---|---|
| d128_L2 | 0.9315 | 0.7047 | 0.0045 | 4.0809 |  |

## Run metadata

- `test_metrics_d128_L2.json`: dim=128, L=2, heads=2, seqlen=100, best_ep=15, time=869.39s
