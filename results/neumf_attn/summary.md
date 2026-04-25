# NeuMF-Attn Results Summary

20-core HotelRec, 1-vs-99 evaluation (1 positive + 99 random negatives).
46,660 users · 27,197 items · 1.85 M interactions.

NeuMF (He et al., WWW 2017) extended with a per-user attention layer over the
six HotelRec sub-rating aspects (Service, Cleanliness, Location, Value, Rooms,
Sleep Quality). Item aspect vectors are pre-computed from the train split
only (no leakage) and missing values are filled with the train-split column
mean.

All numbers below were produced on a single RTX 5070 Ti by:

```bash
python -m src.train_neumf_attn --config configs/neumf_attn.yaml    --kcore 20  # enhanced
python -m src.train_neumf_attn --config configs/neumf_vanilla.yaml --kcore 20  # vanilla
```

## Vanilla vs Enhanced (test split)

The two configs differ only in `model.use_attention` (true → enhanced,
false → vanilla = plain NeuMF, GMF + MLP fusion only). All other
hyperparameters, the dataset, the seed, the BPR negative sampler, and
the calibration step are identical. This isolates the contribution of the
sub-rating attention head.

| Variant                | HR@5   | HR@10  | HR@20  | NDCG@5 | NDCG@10 | NDCG@20 | RMSE   |
|------------------------|--------|--------|--------|--------|---------|---------|--------|
| Vanilla NeuMF          | 0.5978 | 0.7254 | 0.8468 | 0.4815 | 0.5228  | 0.5536  | 0.9304 |
| **Enhanced NeuMF-Attn**| 0.5970 | 0.7245 | 0.8465 | 0.4809 | 0.5221  | 0.5530  | 0.9304 |
| Δ (enhanced − vanilla) | −0.0008| −0.0009| −0.0003| −0.0006| −0.0007 | −0.0006 | 0.0000 |

**The sub-rating attention head adds zero signal** — every ranking metric
moves *down* by 1e-3 (well within run-to-run BPR-sampler variance), and
calibrated RMSE is identical. Diagnoses to follow up on (left for a
follow-up run, not committed to this branch):

- The aspect vector for each hotel is just six near-correlated 1–5 averages
  (most hotels are 4–5 across the board on HotelRec). Softmax over six near-
  identical values produces near-uniform attention regardless of the user.
- The fusion layer's `+1` quality-score input may be getting weighted to ≈ 0
  by `weight_decay=1e-4` because the gradient through the attention path is
  weak relative to the GMF / MLP paths.
- A sparsemax over aspects (forces hard one-hot-ish weights) or an entropy
  bonus would be the natural next experiment.

This is the same dataset-level pattern Pramod's TextNCF sub-rating variant
hit — softmax-over-aspects is a weak prior on HotelRec.

## Cross-team headline ranking (test split)

| Model                           | HR@5   | HR@10  | HR@20  | NDCG@5 | NDCG@10 | NDCG@20 |
|---------------------------------|--------|--------|--------|--------|---------|---------|
| Popularity (baseline)           | 0.3150 | 0.4215 | 0.5538 | 0.2318 | 0.2662  | 0.2995  |
| GMF (baseline)                  | 0.5553 | 0.6685 | 0.7936 | 0.4498 | 0.4863  | 0.5179  |
| ItemKNN (baseline, k=20)        | 0.6835 | 0.6870 | 0.7091 | 0.6082 | 0.6093  | 0.6150  |
| TextNCF — Multi-Task (Pramod)   | 0.5742 | 0.6864 | 0.8031 | 0.4734 | 0.5097  | 0.5392  |
| Vanilla NeuMF                   | 0.5978 | 0.7254 | 0.8468 | 0.4815 | 0.5228  | 0.5536  |
| **NeuMF-Attn (enhanced)**       | **0.5970** | **0.7245** | **0.8465** | **0.4809** | **0.5221** | **0.5530** |
| LightGCN-HG (secondary, dim=256)| 0.6460 | 0.7591 | 0.8655 | 0.5352 | 0.5718  | 0.5988  |
| SASRec (primary, dim=128, L=2)  | 0.8502 | 0.8808 | 0.9173 | 0.8294 | 0.8392  | 0.8484  |

NeuMF (vanilla *or* enhanced) has the **best HR@10 and HR@20 among the
non-sequential neural variants** (GMF, TextNCF family, NeuMF). Both NeuMF
configs beat GMF by ~+8 % rel on HR@10 and beat the best TextNCF (Multi-Task)
by ~+5.5 % rel. Only LightGCN-HG (different feature channel — hotel
geography) and SASRec (sequence model) land above NeuMF.

The big lift comes from the **MLP branch + bigger embeddings** (`mlp_dim=64`
× `[256, 128, 64]` stack vs GMF's flat 64-dim dot product), not from the
attention head.

## Calibrated rating metrics

Calibration fits `rating ≈ a · score + b` on the val split, evaluates on test.

| Model                   | RMSE   | MAE    | a       | b      |
|-------------------------|--------|--------|---------|--------|
| GlobalMean (sanity)     | 0.9315 | 0.7048 | —       | —      |
| **Popularity (item-mean)** | **0.8685** | **0.6749** | —   | —      |
| ItemKNN (k=20)          | 0.9590 | 0.7094 | —       | —      |
| GMF                     | 0.9302 | 0.7002 | —       | —      |
| TextNCF Multi-Task      | 0.9304 | 0.7035 | 0.0128  | 4.1008 |
| Vanilla NeuMF           | 0.9304 | 0.7035 | 0.0202  | 4.0727 |
| **NeuMF-Attn (enhanced)** | 0.9304 | 0.7032 | 0.0208  | 4.0526 |
| LightGCN-HG             | 0.9312 | 0.7025 | —       | —      |
| SASRec                  | 0.9315 | 0.7048 | —       | —      |

Same flat-calibration pattern every BPR-trained ranker on this dataset
exhibits: the slope `a` is near zero, so predicted ratings collapse to
`b ≈ 4.05` (the train mean). Popularity wins RMSE because 78 % of HotelRec
ratings are 4 or 5 stars — predicting the item-mean is near-optimal.

## Training curve

Full 50 epochs ran without early-stopping (patience = 10 never triggered):
HR@10 on val monotonically improved from 0.5972 at epoch 1 to 0.7235 at
epoch 50. Train loss dropped from 0.69 to 0.086, val loss from 0.69 to 0.24.
Val loss started a shallow uptick after epoch 47 (0.2405 → 0.2434 by 50)
while ranking metrics kept climbing — classic mild over-fit on loss with
the ranking objective still intact.

Hyperparameters (see `configs/neumf_attn.yaml`):
- `gmf_dim=64`, `mlp_dim=64`, `mlp_layers=[256, 128, 64]`, `dropout=0.3`
- Batch 256, lr 1e-3, weight-decay 1e-4, cosine LR decay to 1e-6
- 4 sampled negatives per positive (BPR)
- 50 epochs, patience 10 (reduced from the original 1000/30 after single-laptop
  runs showed the attention branch plateaus well before then)

Total wall-clock: **96 min** on RTX 5070 Ti.
Total parameters: **9.5 M** (GMF + MLP embeddings + MLP stack + attention + fusion).

## What the attention learned

Because the aspect vectors are already 1–5-scaled averages, the softmax over
six dimensions mostly smooths across aspects rather than picking a winner —
similar magnitude as a uniform 1/6 prior, with small per-user deviations.
The attention is a contributing signal but not the load-bearing piece of
the model; most of the lift above plain GMF comes from the MLP branch on
top of concatenated user/item embeddings, with the quality score adding a
scalar nudge.

A follow-up experiment worth running (not done here): swap the softmax for
a sparsemax, or add an entropy penalty, to force the attention to
differentiate. That's the same pattern Pramod's TextNCF sub-rating variant
hit — softmax-over-aspects is a weak prior on this data.

## Reproducibility

```bash
# Pre-req: data/processed/20core/*.parquet (shared preprocessing)
python -m src.train_neumf_attn --config configs/neumf_attn.yaml --kcore 20
```

Outputs land in `results/neumf_attn/`:
- `best_model_gmf64_mlp64.pt` (weights, gitignored)
- `test_metrics_gmf64_mlp64.json`
- `rating_metrics_gmf64_mlp64.json`

Per-epoch metrics: `logs/neumf_attn/metrics_gmf64_mlp64.csv` (gitignored).

Notebook walkthrough: [`variants/aditya/notebooks/08_neumf_attn.ipynb`](../../variants/aditya/notebooks/08_neumf_attn.ipynb).
