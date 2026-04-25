# Variant A plan: SASRec (primary) + LightGCN-HG (secondary)

**Owner:** Hriday Ampavatina
**Branch:** `feature/sasrec`

## Decision trail

This variant went through three iterations before settling on the final
SASRec + LightGCN-HG structure.

### 1. Started with LightGCN (graph-based CF)

The first direction was **LightGCN** (He et al., SIGIR 2020) - symmetric-
normalised graph convolution on the user-item bipartite graph. Standard
paper implementation, BPR loss, 1-vs-99 evaluation. That beat Popularity,
GMF, and matched or beat ItemKNN on HR@10/20, but lost to ItemKNN on
NDCG@5/10/20.

### 2. Instructor feedback → LightGCN-HG

The feedback: using only `(user, item, rating)` makes the variant
essentially a baseline against a feature-rich dataset. In response, we
built **LightGCN-HG** - the bipartite graph extended with TripAdvisor
geography nodes (`g_id`, `region_slug`, `country_slug`) parsed from
`hotel_url`. Hotels in the same place share signal through these pivots
even when they have no co-reviewer.

LightGCN-HG beats vanilla LightGCN on every metric but only by small
margins (HR@10 0.7591 vs 0.7530, NDCG@10 0.5718 vs 0.5670). Still loses
NDCG@5/10/20 to ItemKNN. The feature-rich angle works, but the ceiling
on the graph-based direction looks low.

### 3. Pivoted to SASRec as primary

After evaluating sequential models at small scale, **SASRec**
(Kang & McAuley, ICDM 2018) dominated every ranking metric against
LightGCN-HG and every baseline, even at a 5-minute training budget.
Scaled up to dim=128 / 30 epochs, SASRec reaches HR@10=0.8808 and
NDCG@10=0.8392 - beating ItemKNN by +28 % and +38 % relative.

**Sequential bakeoff** (matched dim=64 / 8 epochs, 1-vs-99 eval):

| Model | HR@10 | NDCG@10 | Notes |
|---|---|---|---|
| GRU4Rec (Hidasi et al., 2016) | 0.49 | 0.29 | recurrent next-item predictor |
| Mult-VAE (Liang et al., 2018) | 0.46 | 0.29 | multinomial VAE on user×item matrix |
| **SASRec (Kang & McAuley, 2018)** | **0.83** | **0.78** | **chosen** |

SASRec wins by >2× HR@10 and ~3× NDCG@10 at the same compute. **BERT4Rec
not implemented** — Petrov & Macdonald (RecSys 2022, *A Systematic Review
and Replicability Study of BERT4Rec for Sequential Recommendation*) found
BERT4Rec doesn't consistently beat SASRec under normalised eval, and
bidirectional masked-LM is awkward for the "predict next future booking"
framing on hotel data. Cited the literature instead of running a fourth
model the published replication expects to lose. Bakeoff narrative + full
references in [`notebooks/06_sasrec.ipynb`](notebooks/06_sasrec.ipynb)
Section 1.

SASRec uses the `date` column (100 % coverage on the 20-core) which
neither Aditya (NeuMF + sub-ratings) nor Pramod (Text-NCF) consumes -
making it the best feature-rich angle for this role in the team split.

**Decision:** SASRec becomes the primary variant. LightGCN-HG is
retained as a secondary feature-rich option.

## A1 - SASRec (primary)

### Design

Causal self-attention over per-user time-ordered hotel sequences. Two
transformer decoder blocks, learned positional embeddings, BPR loss over
sampled negatives at each position, shared 1-vs-99 evaluation.

**Config (`configs/sasrec.yaml`):**
- dim=128, max_seqlen=100, 2 layers, 2 heads, dropout=0.2
- 30 epochs, patience=5, cosine LR decay
- batch_size=256, num_negatives=1

### Final results

- HR@10 = **0.8808** (+28 % relative vs ItemKNN)
- NDCG@10 = **0.8392** (+38 % relative vs ItemKNN)
- Training time: ~15 min on one RTX 5070 Ti, best val at epoch 15,
  early-stopped at epoch 20.

Full numbers in [`../../results/sasrec/summary.md`](../../results/sasrec/summary.md)
and the [SASRec notebook](notebooks/06_sasrec.ipynb).

### Non-overlaps with teammates

- **Aditya's NeuMF + sub-ratings:** different input (aspects vs time),
  different architecture (MLP over feature concat vs transformer over
  sequence). Zero overlap.
- **Pramod's Text-NCF:** different input (review text vs time),
  different architecture. Zero overlap.

### Risks

- **Cold-start users.** Short sequences have little context for
  self-attention. Mitigation: the 20-core filter already enforces
  ≥ 20 interactions per user.
- **Chronological evaluation pathology.** The 80/10/10 random split
  doesn't preserve time order, so some val/test items can be
  chronologically earlier than the user's training tail. This isn't a
  leak (val items are held out from training), but it means SASRec is
  being asked to "predict" a random hold-out rather than a strictly
  future item. Still apples-to-apples with every other model in the
  comparison.

## A2 - LightGCN-HG (secondary)

Kept as a secondary variant because (a) it addresses the "feature-rich"
critique through a channel orthogonal to SASRec (geography, not time),
and (b) graph-based CF is a genuinely different research angle worth
~5 minutes in the presentation.

All A2 artefacts (`src/models/lightgcn_hg.py`, `configs/lightgcn_hg.yaml`,
`results/lightgcn_hg/`) are committed.

**Final HG result:** HR@10=0.7591, NDCG@10=0.5718 (three-tier config,
dim=256, early-stopped at epoch 52, best checkpoint from epoch 37).

## Phase 3 integration plan

Ensembles at the per-variant level were considered (LightGCN-HG + ItemKNN
with weighted score blend) but dropped. The final integration step is a
**LightGBM meta-learner** trained on out-of-fold predictions from all
three team variants (SASRec, Aditya's NeuMF+sub-ratings, Pramod's
Text-NCF). Per-variant ensembles with ItemKNN would duplicate that work
without adding signal to the meta-learner.
