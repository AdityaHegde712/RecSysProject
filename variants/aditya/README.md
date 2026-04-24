# Variant B: NeuMF with Attention-Weighted Sub-Ratings

**Owner:** Aditya Hegde
**Branch:** `feature/neumf`

## Approach

Neural Matrix Factorization (NeuMF, He et al., WWW 2017) extended with a
per-user attention layer over the six HotelRec sub-rating aspects
(Service, Cleanliness, Location, Value, Rooms, Sleep Quality).

The attention layer learns which aspect matters most to each user, and the
weighted quality score is fused into the final prediction alongside the
GMF and MLP branches.

## Why this fits HotelRec

- HotelRec has rich sub-rating metadata that most baselines ignore.
- Different user types weight aspects differently (business travelers care
  about location/WiFi, families care about rooms/cleanliness).
- NeuMF is a strong neural CF backbone; sub-rating attention adds both
  accuracy and a hook for interpretability.

## Headline results (20-core, 1-vs-99)

Full numbers + decision notes in [`results/neumf_attn/summary.md`](../../results/neumf_attn/summary.md);
the [`08_neumf_attn` notebook](notebooks/08_neumf_attn.ipynb) is the
executed walkthrough.

| Model | HR@10 | NDCG@10 | Notes |
|-------|-------|---------|-------|
| Popularity (baseline) | 0.4215 | 0.2662 | item-mean rating |
| GMF (baseline) | 0.6685 | 0.4863 | from `results/gmf/` |
| ItemKNN (baseline) | 0.6870 | 0.6093 | k=20 |
| TextNCF Multi-Task (Pramod) | 0.6864 | 0.5097 | |
| **NeuMF-Attn (this variant)** | **0.7245** | **0.5221** | best HR@10 among non-sequential neural variants |
| LightGCN-HG (Hriday secondary) | 0.7591 | 0.5718 | different feature channel (geography) |
| SASRec (Hriday primary) | 0.8808 | 0.8392 | sequence model |

Calibrated RMSE = **0.9304**, MAE = **0.7032** — same flat-calibration
pattern (slope ≈ 0.02, `b ≈ 4.05`) every BPR-trained ranker hits on
HotelRec. Popularity wins RMSE at 0.8685.

## How to run

```bash
# Pre-req: data/processed/20core/*.parquet (shared preprocessing)
python -m src.train_neumf_attn --config configs/neumf_attn.yaml --kcore 20
```

One command; the training script does training + 1-vs-99 test eval +
score-to-rating calibration in one pass. Wall-clock: **~96 min** on an
RTX 5070 Ti (50 epochs × ~2 min/epoch, patience 10 — never triggered
because HR@10 on val was still improving at epoch 50).

## Files

- `src/models/neumf_attn.py` — `NeuMF_Attn` model (GMF + MLP + sub-rating attention + fusion).
- `src/train_neumf_attn.py` — training + eval + calibration pipeline.
- `configs/neumf_attn.yaml` — hyperparameters (renamed from `aditya_neumf.yaml` to match the variant naming convention used by `sasrec.yaml` / `text_ncf*.yaml`).
- `variants/aditya/PLAN.md` — design doc (written pre-implementation).
- `variants/aditya/notebooks/08_neumf_attn.ipynb` — executed walkthrough.
- `results/neumf_attn/` — test metrics, rating metrics, summary.

## Data leakage note

Item aspect vectors (the `(n_items, 6)` sub-rating matrix the attention
reads) are computed from the **train split only**. Missing per-item values
are filled with the train-split column mean; a fully missing column (none
observed on HotelRec) would fall back to 3.0.

## Non-overlap with teammates

- **SASRec (Hriday):** different input (time sequence) and architecture
  (transformer decoder). Zero overlap.
- **TextNCF (Pramod):** different auxiliary channel (review text). Pramod's
  sub-rating variant uses the *same* six columns but as regression targets,
  not as attention-weighted inputs — the decomposition direction is different.
