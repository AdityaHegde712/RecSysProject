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

### Vanilla vs Enhanced (instructor's day-10 ask)

Two configs differing only in `model.use_attention`:

| Variant | HR@10 | NDCG@10 | RMSE |
|---|---|---|---|
| Vanilla NeuMF (GMF + MLP) | **0.7254** | **0.5228** | 0.9304 |
| Enhanced NeuMF-Attn | 0.7245 | 0.5221 | 0.9304 |

**The sub-rating attention head is dead weight on HotelRec.** Across all
six ranking metrics the enhanced run lands ≤ 0.001 below the vanilla —
within run-to-run BPR-sampler variance — and the calibrated RMSE is
identical. The lift over plain GMF (+8.5 % rel HR@10) is the **MLP
branch + bigger embeddings**, not the attention.

Diagnosis: HotelRec's six aspect averages are tightly correlated and
nearly always 4–5 across the board, so softmax-over-aspects collapses to
near-uniform regardless of the user. Same dataset-level pattern Pramod's
TextNCF sub-rating variant hit. A sparsemax / entropy-bonus follow-up
is the natural next experiment (not committed here).

### Cross-team table

| Model | HR@10 | NDCG@10 | Notes |
|-------|-------|---------|-------|
| Popularity (baseline) | 0.4215 | 0.2662 | item-mean rating |
| GMF (baseline) | 0.6685 | 0.4863 | from `results/gmf/` |
| ItemKNN (baseline) | 0.6870 | 0.6093 | k=20 |
| TextNCF Multi-Task (Pramod) | 0.6864 | 0.5097 | |
| **Vanilla NeuMF (this variant)** | **0.7254** | **0.5228** | best HR@10 among non-sequential neural variants |
| Enhanced NeuMF-Attn | 0.7245 | 0.5221 | attention adds ~0 — see vanilla-vs-enhanced section |
| LightGCN-HG (Hriday secondary) | 0.7591 | 0.5718 | different feature channel (geography) |
| SASRec (Hriday primary) | 0.8808 | 0.8392 | sequence model |

Calibrated RMSE on all NeuMF runs = **0.9304** (slope ≈ 0.02, `b ≈ 4.07`)
— same flat-calibration pattern every BPR-trained ranker hits on
HotelRec. Popularity wins RMSE at 0.8685.

## How to run

```bash
# Pre-req: data/processed/20core/*.parquet (shared preprocessing)

# Enhanced (sub-rating attention head on)
python -m src.train_neumf_attn --config configs/neumf_attn.yaml --kcore 20

# Vanilla ablation (no attention) for the day-10 vanilla-vs-enhanced ask
python -m src.train_neumf_attn --config configs/neumf_vanilla.yaml --kcore 20
```

The training script does training + 1-vs-99 test eval + score-to-rating
calibration in one pass per config. Wall-clock: **~96–115 min** per run
on an RTX 5070 Ti (50 epochs × ~2 min/epoch, patience 10 — never triggered
because HR@10 on val was still improving at epoch 50 in both runs).

## Files

- `src/models/neumf_attn.py` — `NeuMF_Attn` model (GMF + MLP + sub-rating attention + fusion). Now takes `use_attention: bool` so the same class powers both vanilla and enhanced runs.
- `src/train_neumf_attn.py` — training + eval + calibration pipeline (reads `model.use_attention` from the config).
- `configs/neumf_attn.yaml` — enhanced config (renamed from `aditya_neumf.yaml` to match the variant naming convention used by `sasrec.yaml` / `text_ncf*.yaml`).
- `configs/neumf_vanilla.yaml` — vanilla ablation config (only diff vs enhanced: `use_attention: false`).
- `variants/aditya/PLAN.md` — design doc (written pre-implementation).
- `variants/aditya/notebooks/08_neumf_attn.ipynb` — executed walkthrough comparing vanilla vs enhanced.
- `results/neumf_attn/` and `results/neumf_vanilla/` — per-config test metrics, rating metrics, summary.

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
