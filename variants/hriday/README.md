# Variant A: LightGCN (Graph-Based Collaborative Filtering)

**Owner:** Hriday Ampavatina
**Paper:** He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation," SIGIR 2020. https://arxiv.org/abs/2002.02126

## Approach

LightGCN removes feature transformation and non-linear activation from standard
GCNs, relying solely on symmetrically-normalized weighted neighborhood
aggregation over the user-item bipartite graph. At each layer `k`:

```
E^(k+1) = A_hat @ E^(k),     A_hat = D^(-1/2) A D^(-1/2)
```

where `A` is the `(N+M) x (N+M)` bipartite adjacency (users stacked above items)
and `D` is its degree matrix. The final user/item embeddings are the mean of
`E^(0), E^(1), …, E^(K)`. Scoring is a dot product between user and item
embeddings, and the model is trained with BPR loss plus L2 regularization on
the layer-0 embeddings, matching the original paper.

## Files

- `src/models/lightgcn.py` — model + `build_norm_adj()` utility.
- `configs/lightgcn.yaml` — default hyperparameters (K=3, dim=64, lr=1e-3).
- `src/train_lightgcn.py` — training + evaluation driver.
- `results/lightgcn/` — per-layer test metrics (`test_metrics_L{K}.json`),
  best checkpoints (`best_model_L{K}.pt`), and summary CSV.
- `logs/lightgcn/metrics_L{K}.csv` — per-epoch training curves.

## Run

```bash
# Single run (K=3 from config)
python -m src.train_lightgcn --config configs/lightgcn.yaml --kcore 20

# Sweep over layer count
for L in 1 2 3 4; do
  python -m src.train_lightgcn --config configs/lightgcn.yaml --kcore 20 --num-layers $L
done
```

## Key design choices

- **Sparse propagation on GPU.** `A_hat` is built once in scipy, converted to
  a `torch.sparse_coo_tensor`, and stored as a module buffer so it moves with
  `.to(device)`. Propagation is `torch.sparse.mm(A_hat, E)` per layer.
- **Eval-time embedding cache.** `propagate()` is expensive, so
  `model.cache_embeddings()` runs it once before evaluation; subsequent
  `forward()` calls reuse the cached user/item matrices. The cache is
  invalidated the moment `model.train()` is called.
- **BPR + L2 reg.** Loss is
  `BPR + reg * (||u0||^2 + ||p0||^2 + ||n0||^2)`, where `u0, p0, n0` are the
  layer-0 embeddings of the sampled triplet. `reg = 1e-4` per the paper.
- **Cosine LR schedule.** 30 epochs with a cosine decay to `1e-5`, early stop
  at patience=5 on validation HR@10.
- **Shared evaluation framework.** Uses the common 1-vs-99 protocol and
  `evaluate_ranking()` with no modifications.

## Why this fits HotelRec

- The 20-core interaction graph (47K users, 27K items, 1.85M edges) is dense
  enough for multi-hop message passing to capture collaborative signal that
  ItemKNN and vanilla GMF miss.
- LightGCN's parameterization is minimal (just the `E^(0)` embeddings), which
  keeps overfitting in check on a dataset where ~78% of ratings are ≥4 (heavy
  positive skew).
- Architecturally distinct from teammates' variants (NeuMF with sub-rating
  attention; review-text-enhanced NCF).

## Open questions / risks

- Does 3 layers trigger over-smoothing? The layer sweep (K=1..4) is the
  primary answer.
- The 1-vs-99 protocol with random negatives may overstate all methods;
  relative comparison is still informative.

## Results (20-core test set, 1-vs-99 protocol)

**K (layer count) sweep, dim=64, 30 epochs**:

| K | HR@5 | HR@10 | HR@20 | NDCG@10 | Time |
|---|---|---|---|---|---|
| **1** | **0.5584** | **0.6865** | **0.8205** | **0.4925** | 800 s |
| 2 | 0.5339 | 0.6623 | 0.8011 | 0.4724 | 910 s |
| 3 | 0.5162 | 0.6433 | 0.7844 | 0.4578 | 1067 s |
| 4 | 0.5015 | 0.6280 | 0.7711 | 0.4455 | 1158 s |

Clean monotonic decline confirms over-smoothing beyond 1 hop: the 20-core graph
is dense enough that direct neighbors already carry most of the useful signal
and additional propagation layers dilute user-specific preferences.

**Extended run (K=1, dim=128, 60 epochs, bs=4096)**:

| HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|
| **0.6135** | **0.7364** | **0.8538** | **0.4985** | **0.5383** | **0.5681** |

**vs. Phase-1 baselines (HR@10 / NDCG@10)**:

| Method | HR@10 | NDCG@10 |
|---|---|---|
| Popularity | 0.4215 | 0.2662 |
| GMF | 0.6685 | 0.4863 |
| ItemKNN (k=50) | 0.6870 | 0.6093 |
| **LightGCN (K=1, dim=128)** | **0.7364** | **0.5383** |

LightGCN improves HR@10 over ItemKNN by **+7.2%** relative. NDCG@10 is lower
than ItemKNN because ItemKNN's rating-weighted sparse cosine gives very
concentrated top-1 predictions, whereas LightGCN spreads hits more evenly
across positions 1-10; it compensates by much stronger top-20 recall (0.854
vs 0.709).

**Rating prediction (RMSE / MAE, traditional metric)**:

| Method | RMSE | MAE |
|---|---|---|
| GlobalMean (sanity) | 0.9315 | 0.7048 |
| **Popularity** (item mean) | **0.8685** | **0.6749** |
| ItemKNN | 0.9703 | 0.7162 |
| LightGCN (calibrated) | 0.9311 | 0.7022 |

Popularity wins RMSE because HotelRec ratings are dominated by 4-5 stars (78%)
so the item-level mean already captures most of the rating variance.
LightGCN's calibration slope is `a = 0.0013` (near zero), confirming that
BPR-trained embeddings carry ranking signal, not rating signal, by design.
See `results/lightgcn/summary.md` for a full writeup.

## Reproducing the results

```bash
# Build the junction-linked 20-core data (or re-run preprocess)
python -m src.data.preprocess --kcore 20
python -m src.data.split --kcore 20

# K=1..4 sweep (dim=64, 30 epochs)
for L in 1 2 3 4; do
  python -m src.train_lightgcn --config configs/lightgcn.yaml --kcore 20 --num-layers $L
done

# Best config (K=1, dim=128, 60 epochs)
python -m src.train_lightgcn --config configs/lightgcn_best.yaml --kcore 20

# RMSE (baselines + calibrated LightGCN)
python scripts/compute_rmse.py --kcore 20 --lightgcn-layers 1 --lightgcn-dim 128 \
    --lightgcn-ckpt results/lightgcn/best_model_L1_d128.pt

# Regenerate summary markdown
python scripts/summarize_lightgcn.py
```
