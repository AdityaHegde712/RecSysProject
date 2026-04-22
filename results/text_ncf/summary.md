# TextNCF Results Summary

## 20-core Dataset

### Baselines (for comparison)

| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-------|------|-------|-------|--------|---------|---------|
| Popularity | 0.315 | 0.422 | 0.554 | 0.232 | 0.266 | 0.300 |
| GMF | 0.555 | 0.669 | 0.794 | 0.450 | 0.486 | 0.518 |
| ItemKNN | 0.684 | 0.687 | 0.709 | 0.608 | 0.609 | 0.615 |

### TextNCF Variants

| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-------|------|-------|-------|--------|---------|---------|
| TextNCF (base) | — | — | — | — | — | — |
| GMF-only ablation | — | — | — | — | — | — |
| Text-only ablation | — | — | — | — | — | — |
| Multi-Task TextNCF | — | — | — | — | — | — |
| Sub-Rating TextNCF | — | — | — | — | — | — |

### Ensemble & Two-Stage

| Method | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | Notes |
|--------|------|-------|-------|--------|---------|---------|-------|
| Ensemble (TextNCF+GMF+KNN) | — | — | — | — | — | — | weights TBD |
| Two-Stage (KNN→TextNCF) | — | — | — | — | — | — | top-200 retrieval |

_Results pending — run the training scripts to populate._

## How to reproduce

```bash
# base TextNCF
python -m src.train_text_ncf --config configs/text_ncf.yaml --kcore 20

# multi-task variant
python -m src.train_text_ncf_mt --config configs/text_ncf_mt.yaml --kcore 20

# sub-rating variant
python -m src.train_text_ncf_subrating --config configs/text_ncf_subrating.yaml --kcore 20

# ensemble (needs all three base models trained first)
python -m src.evaluate_ensemble --kcore 20

# two-stage (needs TextNCF + ItemKNN trained)
python -m src.evaluate_two_stage --kcore 20
```

## Key hyperparameters

| Variant | Key Param | Value | Description |
|---------|-----------|-------|-------------|
| Base TextNCF | embed_dim | 64 | Collaborative embedding dimension |
| Base TextNCF | text_proj_dim | 64 | Projected text dimension |
| Base TextNCF | mlp_layers | [128, 64] | Fusion MLP hidden sizes |
| Multi-Task | alpha | 0.7 | BPR weight (1-alpha = MSE weight) |
| Sub-Rating | beta | 0.6 | BPR weight (1-beta = aspect MSE weight) |
| Sub-Rating | aspect_hidden | 32 | Hidden dim per sub-rating head |
| Ensemble | grid_step | 0.1 | Weight search granularity |
| Two-Stage | retrieve_k | 200 | Candidates from ItemKNN |

## Sub-Rating Attention Analysis

_Will be populated after training. Shows which hotel aspects (Service, Cleanliness, Location, Value, Rooms, Sleep Quality) the model learns to weight most heavily._
