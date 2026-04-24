# Variant C plan: Text-NCF family

**Owner:** Pramod Yadav
**Branch:** `pramod/text-ncf`

## Motivation

HotelRec pairs 1–5 star ratings with ~125-word reviews. About 71 % of
reviews have substantive text — a dense preference signal that pure
collaborative filtering ignores. The text captures aspects the overall
rating compresses away ("walking distance to the beach", "thin walls",
"breakfast was overcharged"). Our Variant C goal is to plug that signal
into a standard NCF pipeline and see where it helps.

## Non-overlap with teammates

| Variant | Input channel | Architecture | Overlap |
|---|---|---|---|
| **Hriday — SASRec** | Time sequence (`date`) | Transformer decoder | None |
| **Aditya — NeuMF + sub-ratings** | Sub-rating aspects | Feature-concat MLP | Sub-ratings head of TextNCF-SR could overlap, but the *channel* (text embeddings) is distinct. |
| **Pramod — TextNCF family (this)** | Review text + collaborative | Two-branch fusion MLP | n/a |

The text channel is orthogonal to time (SASRec) and aspects (Aditya's
headline feature). The sub-rating variant here borrows sub-ratings as an
auxiliary supervision target, not as a primary input.

## Approaches

### C1 — Base TextNCF (primary)

Two-branch hybrid: `GMF ⊙` + `text ⊙` → MLP → score.

- Encode every review with frozen **all-MiniLM-L6-v2** (384-dim) offline.
- Aggregate per user (train-split reviews only, to avoid leakage) and
  per item (all splits — hotels are not a target).
- Train a small projection + fusion MLP with BPR loss.

**Config** (`configs/text_ncf.yaml`): `embed_dim=64`, `text_proj_dim=64`,
`mlp=[128,64]`, 30 epochs, cosine LR, patience=5, 4 negatives.

### C2 — Ablations (GMF-only and text-only)

Same architecture, one branch disabled (`use_gmf=false` or
`use_text=false`). Tests how much of the full model's gain comes from
each branch. Runs with the same config overrides.

### C3 — Ensemble (TextNCF + GMF + ItemKNN)

Per-user min-max normalises each model's scores over the 99+1
candidates, then weights `w_text · TextNCF + w_gmf · GMF + w_knn · KNN`.
Grid-searches weights on the validation set (step 0.1 → 66 combos) and
reports test metrics at the best validation NDCG@10.

Rationale: if the three models make different mistakes, a weighted blend
should ratchet NDCG up without retraining.

### C4 — Two-stage retrieval + re-ranking

Production-style pipeline: ItemKNN retrieves top-200 candidates
(sparse, ~ms per user), TextNCF re-ranks (neural, ~ms per user at
small `retrieve_k`). Also measures `gt_recall@200` — how often the
held-out item is even in the candidate set.

### C5 — Multi-Task TextNCF

Same backbone, two heads: BPR (ranking) + MSE on 1–5 ratings
(regression). Joint loss `alpha · BPR + (1 − alpha) · MSE`, α=0.7.
Tests whether the rating signal gives the ranking head a smoother loss
surface.

### C6 — Sub-rating decomposition

Shared MLP → 6 aspect heads (Service, Cleanliness, Location, Value,
Rooms, Sleep Quality) + per-user attention weights → weighted sum.
Loss: BPR on the combined score plus MSE on each aspect. Surfaces
*which* aspect each user is sensitive to.

Sub-rating coverage on the 20-core train split:
Service 80 %, Cleanliness 62 %, Location 60 %, Value 62 %, Rooms 60 %,
Sleep Quality 55 %. Missing values fall back to the overall rating.

## Pipeline

```
HotelRec.txt  →  preprocess_zip.py  →  data/processed/20core/
                                           │
                                           ├─ encode_text.py (MiniLM, one-shot)
                                           │     └─ data/processed/text_emb/*.npy
                                           │
                                           ├─ train_text_ncf.py          → results/text_ncf/
                                           ├─ train_text_ncf_mt.py       → results/text_ncf_mt/
                                           ├─ train_text_ncf_subrating.py → results/text_ncf_subrating/
                                           │
                                           ├─ evaluate_ensemble.py   (needs TextNCF+GMF+ItemKNN)
                                           └─ evaluate_two_stage.py  (needs TextNCF+ItemKNN)
```

All eval artefacts land under `results/<variant>/` as `test_metrics.json`
+ `rating_metrics.json`, so `scripts/compute_rmse.py` and the shared
summary consume them uniformly.

## Risks

- **Cold-start items.** Items with zero training reviews get a
  zero-vector text embedding. The 20-core filter gives every item ≥ 20
  interactions, so this is rare but not impossible. Mitigation: item
  profile averages over all splits (allowed — hotels aren't the label).
- **MiniLM ceiling.** 384-dim MiniLM is cheap but not great at
  domain-specific vocabulary. An MPNet-base (768-dim) run is the
  natural next experiment if the gain from text alone is small.
- **Ranking-only RMSE.** Same structural issue as SASRec — a
  BPR-trained model's scores don't calibrate cleanly to 1–5 ratings on
  a 4-star-heavy dataset. RMSE reported via linear calibration on the
  val split, same methodology as the other ranking models.

## Phase 3 integration plan

TextNCF's per-user scores feed the team's LightGBM meta-learner
alongside SASRec (Hriday) and NeuMF + sub-ratings (Aditya). The
per-variant ensemble (C3) is for variant-internal analysis, not a
substitute for the team-wide meta-ensemble.
