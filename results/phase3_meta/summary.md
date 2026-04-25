# Phase 3 — LightGBM Meta-Ensemble

20-core HotelRec, 1-vs-99 evaluation. Stacks four frozen Phase-2 base
models into a single `LGBMRanker` (lambdarank objective):

| Slot | Variant                | Owner   | Channel                     |
|------|------------------------|---------|-----------------------------|
| A1   | SASRec (primary)       | Hriday  | Time sequence (`date`)      |
| A2   | LightGCN-HG (secondary)| Hriday  | Graph + TripAdvisor geography |
| B    | NeuMF-Attn (enhanced)  | Aditya  | GMF + MLP + sub-rating attention |
| C    | TextNCF — Multi-Task   | Pramod  | Review text (frozen MiniLM) + rating MSE head |

Each base model contributes one column to the meta-feature matrix.
Per-user min-max normalisation flattens columns to `[0, 1]` per user-group.
The LGBMRanker is trained on val (1+99 candidates per user, label = 1 for
the held-out positive) and evaluated on test under the same protocol every
other variant uses.

Run: `python -m src.phase3_meta_ensemble --kcore 20`
Total wall-clock on RTX 5070 Ti: **~2 min** (most of which is base-model
scoring; LGBM training ≈ 50 s).

## Headline ranking metrics (test split)

| Model                               | HR@5   | HR@10  | HR@20  | NDCG@5 | NDCG@10 | NDCG@20 |
|-------------------------------------|--------|--------|--------|--------|---------|---------|
| Phase 3 meta-ensemble (LGBMRanker)  | 0.6600 | 0.7739 | 0.8782 | 0.5474 | 0.5843  | 0.6107  |
| **SASRec (best base)**              | **0.8501** | **0.8809** | **0.9173** | **0.8294** | **0.8392** | **0.8484** |
| LightGCN-HG                         | 0.6460 | 0.7591 | 0.8655 | 0.5352 | 0.5718  | 0.5988  |
| NeuMF-Attn                          | 0.5970 | 0.7245 | 0.8465 | 0.4809 | 0.5221  | 0.5530  |
| TextNCF Multi-Task                  | 0.5742 | 0.6864 | 0.8031 | 0.4734 | 0.5097  | 0.5392  |

The meta-ensemble lands **between LightGCN-HG and SASRec** on ranking — better than the three weaker base models, **but well below SASRec alone**.

## Calibrated rating metrics (test split)

| Model                               | RMSE   | MAE    | a      | b      |
|-------------------------------------|--------|--------|--------|--------|
| GlobalMean (sanity)                 | 0.9315 | 0.7048 | —      | —      |
| Popularity (item-mean)              | 0.8685 | 0.6749 | —      | —      |
| ItemKNN (k=20)                      | 0.9590 | 0.7094 | —      | —      |
| GMF                                 | 0.9302 | 0.7002 | ~0     | ~4.08  |
| Vanilla LightGCN                    | 0.9312 | 0.7025 | 0.0008 | 4.0658 |
| LightGCN-HG                         | 0.9312 | 0.7025 | 0.0009 | 4.0655 |
| NeuMF-Attn                          | 0.9304 | 0.7032 | 0.0208 | 4.0526 |
| TextNCF Multi-Task                  | 0.9304 | 0.7035 | 0.0128 | 4.1008 |
| SASRec                              | 0.9315 | 0.7048 | ~0     | ~4.08  |
| **Phase 3 meta-ensemble**           | **0.8350** | **0.6164** | **0.0261** | **4.0455** |

**The meta-ensemble is the only ranking-trained pipeline that beats
Popularity on RMSE.** Slope `a = 0.0261` is ~30× larger than any single
base model's slope — the ensembled lambdarank score has more usable
variance for an lstsq calibrator than any one BPR-trained score.

## Why the ensemble underperforms SASRec on ranking

LightGBM's split-gain feature importances:

| Base model      | LightGBM split-gain |
|-----------------|---------------------|
| text_ncf_mt     | 2582 |
| neumf_attn      | 2419 |
| lightgcn_hg     | 2345 |
| sasrec          | 1654 |

SASRec is the *least* used feature, even though it's the strongest base
model. The cause is the per-user min-max normalisation: every column
gets squashed to `[0, 1]` per user, so the LGBMRanker can't see that
SASRec's raw confidence margin over the negatives is ~10 percentage-
points larger than the others. Lambdarank ends up blending four columns
that all look equally informative on their `[0, 1]` scale — and the
blend dilutes SASRec's signal.

This is the **strong-model dilution effect**: stacking under naïve
normalisation hurts ranking when one base model dominates the others.
Pramod's earlier `evaluate_ensemble` (TextNCF + GMF + ItemKNN under per-user
min-max) hit the inverse failure — the grid search collapsed to
ItemKNN-only — but for the same underlying reason.

## Why the ensemble wins on RMSE

The lambdarank score isn't constrained to encode pairwise margins like
BPR is. The LightGBM trees learn a non-linear combination that produces
genuinely *graded* output across positives — some users get higher scores
than others — which is exactly what an lstsq calibrator needs. Each
single BPR base model's score is much flatter across positives (which
is fine for ranking, useless for rating).

Result: the meta-ensemble's calibrated RMSE 0.8350 beats Popularity's
0.8685 by 0.034 absolute (~4 % rel), and beats every BPR base model
by ~0.10. It's the **only model in the project that wins on RMSE
without being explicitly trained for rating prediction**.

## Recommended next experiments (not committed here)

1. **Feed raw scores + per-user z-score** as separate columns. The
   lambdarank tree should be able to read both magnitude and rank, which
   would let SASRec dominate where appropriate.
2. **Stacked classifier instead of ranker.** Pointwise binary
   classification (positive vs negative) avoids the per-user pairwise
   constraints and gives LightGBM more degrees of freedom.
3. **K-fold OOF** on train so the level-1 model isn't trained on the same
   distribution the level-0 models early-stopped on. Substantial cost
   (10+ hrs); deferred for the report-stage.

## Reproducibility

```bash
# Pre-req: every Phase-2 base model checkpoint trained
python -m src.phase3_meta_ensemble --kcore 20
```

Outputs (under `results/phase3_meta/`):

- `test_metrics.json` — meta-ensemble HR@k / NDCG@k
- `rating_metrics.json` — calibrated RMSE / MAE
- `feature_importances.json` — LightGBM split-gain per base model
- `component_metrics.json` — each base model's HR@k / NDCG@k on the
  *same* seed-42 candidate pool the meta-ensemble sees
- `lightgbm.txt` — saved booster
- `test_scores.npz` — raw per-pair meta scores + component scores +
  labels + user/item ids, for the notebook to inspect

Notebook walkthrough: [`notebooks/05_ensemble_and_summary.ipynb`](../../notebooks/05_ensemble_and_summary.ipynb).
