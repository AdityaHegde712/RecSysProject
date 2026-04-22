# TextNCF Results Summary

## 20-core Dataset

_Pending — run `python -m src.train_text_ncf --config configs/text_ncf.yaml --kcore 20` to generate._

### Baselines (for comparison)

| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-------|------|-------|-------|--------|---------|---------|
| Popularity | 0.315 | 0.422 | 0.554 | 0.232 | 0.266 | 0.300 |
| GMF | 0.555 | 0.669 | 0.794 | 0.450 | 0.486 | 0.518 |
| ItemKNN | 0.684 | 0.687 | 0.709 | 0.608 | 0.609 | 0.615 |

### TextNCF

| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-------|------|-------|-------|--------|---------|---------|
| TextNCF | — | — | — | — | — | — |
| GMF-only ablation | — | — | — | — | — | — |
| Text-only ablation | — | — | — | — | — | — |
