# Implementation Plan: NeuMF with Attention-Weighted Sub-Ratings

**Status**: Ready for Claude (Architecture & Pipeline Generation)

## 1. Objective

Implement **Variant B: NeuMF with Attention-Weighted Sub-Ratings**. This variant extends Neural Matrix Factorization by incorporating an attention mechanism over the interaction's sub-rating dimensions (Service, Cleanliness, Location, Value, Rooms, Sleep Quality). This goes "beyond user-item interactions" by modeling the _content aspects_ of the experience that drive the overall rating.

## 2. Assumptions & Pre-processed Data

The preprocessing pipeline (`src/data/preprocess.py` and `src/data/split.py`) has already been executed. The following files are available for training:

- **Interactions**: `data/processed/20core/train.parquet` (and `val`/`test`).
- **Sub-ratings**: Columns `service`, `cleanliness`, `location`, `value`, `rooms`, `sleep_quality` are present in the parquets.
- **Hotel Metadata**: `data/processed/hotel_meta/hotel_meta.parquet` (parsed from URLs).
- **ID Maps**: `data/processed/20core/user2id.json` and `item2id.json`.

## 3. Implementation Tasks (For Claude)

### A. Model Definition (`variants/aditya/models/neumf_attn.py`)

Implement the `NeuMF_Attn` class following the structural "layout" of `src/models/sasrec.py`.

- **Backbone**: NeuMF (GMF + MLP branches).
- **Attention Layer**:
  - Learns a per-user attention vector (weights) over the 6 sub-rating dimensions.
  - Aggregates the weighted sub-ratings into a "quality score."
- **Methods**:
  - `forward(users, pos_items, neg_items)` for BPR training.
  - `score_candidates(users, items)` for the 1-vs-99 ranking evaluation.

### B. Feature Engineering

- Pre-compute average sub-ratings per hotel using the **train split only** to avoid leakage.
- Handle missing values (NaN) in sub-ratings by filling with the global mean.

### C. Training Pipeline (`variants/aditya/train_neumf_attn.py`)

Adapt `src/train_sasrec.py` to support this new architecture:

- **Config**: Setup `configs/aditya_neumf.yaml` for hyper-parameters.
- **Loss**: BPR loss with one sampled negative per interaction.
- **Evaluation**: HR@10 and NDCG@10 (1-vs-99 protocol).
- **Calibration**: Score-to-rating calibration (RMSE/MAE) as implemented in `calibrate_sasrec`.
- **Metrics**: Log results using `src.utils.metrics_logger.MetricsLogger`.

## 4. Why This satisfies the "Beyond User-Item" Requirement

While standard collaborative filtering treats a review as a binary or scalar signal, this approach models the **multi-dimensional quality aspects** of the hotel. It explicitly learns which aspects (e.g., "Location" vs "Cleanliness") matter most to a specific user, providing both better accuracy and interpretability.

---

_Note: Once code generation is complete, transition back to Gemini for integration and final evaluation._
