# TextNCF Results Summary

20-core HotelRec, 1-vs-99 evaluation (1 ground-truth + 99 random negatives).
46,660 users · 27,197 items · 1.85 M interactions.

All numbers below are produced by `bash scripts/run_text_ncf_all.sh`
on a single RTX 5070 Ti box. Per-variant artefacts live under
`results/<variant>/test_metrics.json` and `results/<variant>/rating_metrics.json`.

## Headline ranking metrics (test split)

| Model                          | HR@5   | HR@10  | HR@20  | NDCG@5 | NDCG@10 | NDCG@20 |
|--------------------------------|--------|--------|--------|--------|---------|---------|
| Popularity (baseline)          | 0.3150 | 0.4215 | 0.5538 | 0.2318 | 0.2662  | 0.2995  |
| GMF (baseline)                 | 0.5553 | 0.6685 | 0.7936 | 0.4498 | 0.4863  | 0.5179  |
| ItemKNN (baseline)             | 0.6835 | 0.6870 | 0.7091 | 0.6082 | 0.6093  | 0.6150  |
| TextNCF — base                 | 0.5688 | 0.6787 | 0.7951 | 0.4702 | 0.5057  | 0.5351  |
| TextNCF — GMF-only ablation    | 0.5585 | 0.6720 | 0.7930 | 0.4548 | 0.4915  | 0.5221  |
| TextNCF — text-only ablation   | 0.5659 | 0.6891 | 0.8183 | 0.4583 | 0.4981  | 0.5308  |
| **TextNCF — Multi-Task (best)**| **0.5742** | **0.6864** | 0.8031 | **0.4734** | **0.5097** | **0.5392** |
| TextNCF — Sub-rating           | 0.5380 | 0.6677 | 0.8023 | 0.4291 | 0.4710  | 0.5050  |
| Ensemble (TextNCF+GMF+KNN)     | 0.6835 | 0.6870 | 0.7091 | 0.6082 | 0.6093  | 0.6150  |
| Two-stage (KNN→TextNCF)        | 0.3164 | 0.3858 | 0.4871 | 0.2754 | 0.2977  | 0.3231  |

## Calibrated rating metrics

Calibration fits `rating ≈ a · score + b` on the val split, then evaluates
RMSE / MAE on test. Same methodology used for SASRec / GMF / LightGCN-HG.

| Model                  | RMSE   | MAE    | a       | b      | Notes |
|------------------------|--------|--------|---------|--------|-------|
| GlobalMean (sanity)    | 0.9315 | 0.7048 | —       | —      | predicts the train mean (4.08) |
| Popularity (item-mean) | **0.8685** | **0.6749** | —   | —      | RMSE winner on a 78 %-4-or-5-star dataset |
| ItemKNN (k=20)         | 0.9590 | 0.7094 | —       | —      | weighted neighbour rating |
| TextNCF — base         | 0.9306 | 0.7014 | 0.0093  | 4.0941 | calibration slope ≈ 0 |
| TextNCF — Multi-Task   | 0.9304 | 0.7035 | 0.0128  | 4.1008 | rating head doesn't fix slope |
| TextNCF — Sub-rating   | 0.9309 | 0.7047 | 0.0326  | 4.0671 | aspect heads also calibrate flat |

The near-zero slopes are the now-familiar HotelRec pattern: a BPR-trained
ranker compresses scores into a tight band and the optimal linear map
collapses to "predict the mean". The Multi-Task head's MSE term doesn't
shift this — α=0.7 keeps BPR dominant. Popularity wins RMSE for the
same reason it does on every other variant.

## What worked

- **Adding text helps.** Base TextNCF beats GMF on every k
  (NDCG@10 0.5057 vs 0.4863, +4 % relative). The lift is driven by the
  text branch — the text-only ablation already matches the full model
  on HR@10/20, and the GMF-only ablation lands below GMF on NDCG.
  Frozen MiniLM embeddings + a small projection are enough to add
  signal without any fine-tuning.
- **Multi-Task TextNCF is the best ranker overall.** NDCG@10 0.5097 is
  the high-water mark in the family. The MSE head acts like a
  light regulariser — it doesn't shift HR much, but it nudges NDCG up.
- **Text-only HR@20 = 0.8183.** Highest HR@20 in the table. Useful when
  the downstream ranker only cares about hit-rate at large k
  (re-ranking, candidate filtering).

## What didn't work

- **Sub-rating decomposition collapsed.** The per-user attention
  network learned a degenerate solution: 99.8 % weight on Cleanliness,
  0 % on the other five aspects, on every user (std=0). The shared
  fusion MLP then carries all the signal and the aspect heads are
  effectively unused. NDCG@10 = 0.471 — worse than the GMF-only
  ablation. Likely fixes (untried for time): warm-start the attention
  uniform, add an entropy bonus on the attention logits, or regularise
  individual heads against the aspect labels with a stronger weight.
- **Three-way ensemble degenerated to ItemKNN.** Grid search over
  `(w_text, w_gmf, w_knn)` with step 0.1 picked `(0.0, 0.0, 1.0)`. After
  per-user min-max normalisation the spread of ItemKNN scores still
  encodes more useful per-user ranking information than blending in the
  weaker neural models adds — they net out as noise. Final ensemble
  numbers therefore reproduce ItemKNN exactly. Net signal: a per-variant
  blend doesn't help; the team-wide LightGBM meta-ensemble in Phase 3
  is the right home for combining these.
- **Two-stage retrieval is recall-bound.** ItemKNN's open top-200
  retrieves the held-out item only 5.0 % of the time across the full
  27 K-item space. With the GT injected into the candidate set for
  fairness, the re-ranker can't recover from a near-empty candidate
  pool — HR@10 drops to 0.39. This is a genuine production-relevance
  signal, not a code bug: ItemKNN's similarity assumes the item space
  matches the user history closely, which doesn't hold over a 1.85 M-row
  matrix. A larger retriever (`retrieve_k=2000`) or a different
  retriever (popularity blend, BM25 over hotel descriptions) would be
  the next experiment.

## Reproducibility

```bash
# 1. one-shot setup (idempotent)
python scripts/encode_text.py --kcore 20 --device cuda
python scripts/fit_itemknn.py --kcore 20

# 2. all five trainings + ablations + ensemble + two-stage + RMSE
bash scripts/run_text_ncf_all.sh
```

Per-step:

```bash
# trainings
python -m src.train_text_ncf            --config configs/text_ncf.yaml            --kcore 20
python -m src.train_text_ncf_mt         --config configs/text_ncf_mt.yaml         --kcore 20
python -m src.train_text_ncf_subrating  --config configs/text_ncf_subrating.yaml  --kcore 20

# ablations (configs auto-derived in run_text_ncf_all.sh)
python -m src.train_text_ncf --config configs/_ablations/text_ncf_gmf_only.yaml  --kcore 20
python -m src.train_text_ncf --config configs/_ablations/text_ncf_text_only.yaml --kcore 20

# ensemble + two-stage (need ItemKNN pickle + GMF + base TextNCF checkpoints)
python -m src.evaluate_ensemble  --kcore 20 --grid-step 0.1
python -m src.evaluate_two_stage --kcore 20 --retrieve-k 200

# rating metrics (extends the shared compute_rmse.py)
python scripts/compute_rmse.py --kcore 20 \
    --text-ncf-ckpt           results/text_ncf/best_model.pt \
    --text-ncf-mt-ckpt        results/text_ncf_mt/best_model.pt \
    --text-ncf-subrating-ckpt results/text_ncf_subrating/best_model.pt
```

## Hyperparameters

| Variant       | dim | text proj | MLP        | epochs | early-stop | aux loss weight |
|---------------|-----|-----------|------------|--------|------------|-----------------|
| Base TextNCF  | 64  | 64        | [128, 64]  | 30     | patience 5 | —               |
| Multi-Task    | 64  | 64        | [128, 64]  | 30     | patience 5 | α=0.7 (BPR)     |
| Sub-rating    | 64  | 64        | [128, 64]  | 8 (early-stopped) | patience 5 | β=0.6 (BPR), 6 aspect heads (hidden=32) |
| Ablation GMF  | 64  | —         | [128, 64]  | 30     | patience 5 | text branch off |
| Ablation Text | —   | 64        | [128, 64]  | 30     | patience 5 | gmf branch off  |

All variants: BPR loss, 4 negatives / positive, batch=256, lr=1e-3, cosine
LR decay to 1e-5, weight decay 1e-5, seed=42.

## Sub-rating attention (degenerate)

| Aspect          | Mean weight | Std |
|-----------------|-------------|-----|
| Service         | 0.000       | 0.000 |
| Cleanliness     | 0.998       | 0.000 |
| Location        | 0.000       | 0.000 |
| Value           | 0.000       | 0.000 |
| Rooms           | 0.000       | 0.000 |
| Sleep Quality   | 0.000       | 0.000 |

The attention logits collapsed within a few epochs onto Cleanliness for
every user. Treat this as the *current state of the model*, not a
finding about which aspect actually matters most on HotelRec.
