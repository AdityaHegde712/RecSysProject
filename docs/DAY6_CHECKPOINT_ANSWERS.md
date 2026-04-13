# Day 6 Team Checkpoint — CMPE 256

**Team:** Aditya Hegde, Pramod Yadav, Hriday Ampavatina

**Dataset:** HotelRec (Antognini & Faltings, LREC 2020)

**GitHub:** https://github.com/AdityaHegde712/RecSysProject

---

## Section 1 — Dataset Summary

### 1a. Basic Statistics

| Stat | Value |
|------|-------|
| Number of users | 21,891,294 (full) / 72,603 (20-core) |
| Number of items | 365,056 (full) / 38,903 (20-core) |
| Number of interactions | 50,264,531 (full) / 2,222,373 (20-core) |
| Density | 0.00063% (full) / 0.079% (20-core) |
| Avg interactions per user | 2.24 (full) / 30.6 (20-core) |
| Available metadata | Overall rating (1–5), review text, date, up to 8 sub-ratings (Service, Cleanliness, Location, Value, Rooms, Sleep Quality, Check-In, Business Service), user profile URL, hotel URL |

We plan to use the **20-core** subset for development and hyperparameter tuning since it fits comfortably in RAM (~2.2M reviews), and the **5-core** subset (21M reviews) for final evaluation to match the paper's reported numbers.

### 1b. Key Challenges and Observations

- **Extreme sparsity (99.999%)** — the user-item matrix is almost entirely empty. Most collaborative filtering methods struggle at this level of sparsity because there's barely any overlap between users' interaction histories.

- **Severe cold start** — 67.55% of users wrote exactly one review, and 90.73% wrote fewer than five. This means the majority of users have almost no history to learn from, which is a fundamental problem for any user-embedding-based approach.

- **Power-law distribution** — both user activity and item popularity follow a heavy long tail. The median reviews per user is 1 while the mean is 2.24; for hotels, the median is 41 vs. mean 137.69. A small number of popular hotels dominate the dataset.

- **Positive rating bias** — the average overall rating is 4.15/5 with a median of 5. Most reviews are positive, which makes rating prediction less discriminative and pushes us toward ranking-based evaluation rather than RMSE.

- **Rich but optional metadata** — sub-ratings are available for a decent fraction of reviews, but coverage varies wildly. Service is present in 99.27% of reviews, but Business Service only in 1.69%. Any model that uses sub-ratings needs to handle missing values gracefully.

- **Scale** — 50M reviews is large enough that naive in-memory processing won't work. We need chunked I/O, parquet storage, and batched training. Even the 20-core subset at 2.2M rows requires some care with memory.

---

## Section 2 — Data Pipeline

### 2a. Preprocessing Steps

1. **Parse raw JSONL files** — the HotelRec dataset ships as JSONL (one JSON object per line per file). Each record has `hotel_url`, `author`, `date`, `rating`, `title`, `text`, and a nested `property_dict` containing sub-ratings like `service`, `cleanliness`, etc.

2. **Flatten and normalize** — we pop `property_dict` and promote each sub-rating to a top-level column. String fields are cleaned of non-UTF-8 characters. We convert the raw JSONL into Parquet for efficient columnar access (see `scripts/convert_parquet.py`).

3. **Build contiguous integer ID mappings** — user URLs and hotel URLs are mapped to `0..N-1` and `0..M-1` respectively. These mappings are saved as JSON files so we can reconstruct the original identifiers later.

4. **Apply iterative k-core filtering** — we remove users and items with fewer than `k` interactions, repeating until convergence. This is standard practice to ensure every user and item in the dataset has enough history for meaningful evaluation. We use `k=20` for development and `k=5` for final runs.

5. **Remap IDs after filtering** — after k-core filtering removes some users/items, we remap the remaining IDs back to a contiguous range to avoid sparse embedding tables.

6. **Text preprocessing** — lowercase review text, truncate to 512 tokens. This is mainly relevant for Pramod's text-enhanced variant but we do it in the shared pipeline so everyone has access to clean text.

7. **Save as Parquet** — the final output goes to `data/processed/{k}core/interactions.parquet` along with the ID mapping JSONs.

### 2b. Output Schema

The pipeline produces the following files under `data/processed/{k}core/`:

- **`interactions.parquet`** — columns: `user_id` (int), `item_id` (int), `rating` (float32), `text` (string), `date` (timestamp), plus sub-rating columns (`Service`, `Cleanliness`, `Location`, `Value`, `Rooms`, `Sleep Quality`, `Check-In`, `Business Service`) as float32 (nullable).

- **`user2id.json`** — dictionary mapping `{original_user_url: contiguous_int_id}`.

- **`item2id.json`** — dictionary mapping `{original_hotel_url: contiguous_int_id}`.

- **`train.parquet`**, **`val.parquet`**, **`test.parquet`** — 80/10/10 random split of `interactions.parquet`, produced by `src/data/split.py` with `seed=42`.

For the recommendation task, the `InteractionDataset` class in `src/data/dataset.py` consumes these splits and produces `(user_id, positive_item_id, negative_item_id)` triplets with on-the-fly negative sampling during training. For evaluation, `EvalInteractionDataset` pairs each test positive with 99 sampled negatives.

---

## Section 3 — Baselines

| Baseline method | Library / implementation | Key results (20-core) |
|----------------|------------------------|----------------------|
| GMF (He et al., 2017) | PyTorch (custom, `src/models/gmf.py`) | HR@10 = 0.5219, NDCG@10 = 0.3047 (paper-reported) |

**Notable observations:**

- GMF is a middle-ground model in the NCF family. It's significantly better than traditional CF approaches (the paper reports ItemKNN at HR@10 = 0.0411, which is basically useless) but leaves clear room for improvement compared to NeuMF (HR@10 = 0.5776).

- The ~21% relative gap between GMF and NeuMF on HR@10 tells us that adding an MLP path or attention mechanism on top of the element-wise product should yield meaningful gains. This is exactly the space our three variants will explore.

- We chose GMF as our shared baseline because (a) it's neural, so it uses the same training infrastructure (PyTorch, negative sampling, BPR loss) that our variants will build on, and (b) it's simple enough that we can be confident in our implementation — there's less room for bugs to hide compared to a multi-component model.

- We haven't yet reproduced these exact numbers on the full 20-core subset (training takes ~2h on a single GPU), but the pipeline is end-to-end runnable and we've validated the architecture against the paper's description. We expect our numbers to be within a small margin of the reported values.

---

## Section 4 — Evaluation Framework

### 4a. Split Strategy

We use a **random 80/10/10 split** (train/validation/test) with `seed=42` for reproducibility. The split is applied after k-core filtering, which ensures that every user and item in the test set has at least `k` interactions in the full dataset.

We considered a temporal split (train on older reviews, test on newer ones) but decided against it for two reasons: (1) the original paper uses random splits, and we want to reproduce their numbers before trying anything different, and (2) temporal splits introduce additional complexity around users who only appear in the test period.

### 4b. Metrics

| Metric | Parameter | Justification |
|--------|-----------|---------------|
| HR (Hit Ratio) | k = 5, 10, 20 | Measures whether the ground-truth item appears anywhere in the top-k list. It's the standard recall-style metric for implicit feedback recommendation — simple to interpret and widely reported. |
| NDCG (Normalized DCG) | k = 5, 10, 20 | Position-aware ranking metric that gives more credit when the correct item is ranked higher. Complements HR by measuring *where* in the list the hit occurs, not just whether it occurs. |

We report at k = 5, 10, and 20 to match the paper and to see how performance degrades as we ask for more precise recommendations (smaller k is harder).

### 4c. Dataset-Specific Evaluation Considerations

- **Implicit feedback formulation** — we treat all ratings as binary implicit feedback (interacted = 1, not interacted = 0), following the paper's evaluation protocol. We don't try to predict the actual rating value because the strong positive bias (avg 4.15/5) makes rating prediction less informative than ranking.

- **Leave-one-out with 99 negatives** — for each test user, we take one held-out positive item and sample 99 random items the user hasn't interacted with. We rank all 100 items and compute HR/NDCG on this list. This is the standard protocol from He et al. (2017) and matches what the HotelRec paper uses.

- **k-core filtering handles cold start** — by filtering to users/items with at least `k` interactions, we avoid cold-start artifacts in the metrics. Without this, a model could look good just by handling the easy cases (popular items) while failing on the long tail.

- **No item filtering in test** — we don't remove popular items from the test set. Some papers do this to focus on long-tail performance, but we follow the paper's protocol for fair comparison.

---

## Section 5 — Variant Plan

| Member | Planned variant direction | Why this approach fits the dataset |
|--------|--------------------------|-----------------------------------|
| Aditya Hegde | NeuMF with attention-weighted sub-ratings | HotelRec has rich sub-rating metadata (Service, Location, Cleanliness, etc.) that GMF completely ignores. By feeding sub-ratings through an attention layer, the model can learn which aspects matter most to each user — e.g., a business traveler might weight Location and Check-In heavily while a family might care more about Rooms and Cleanliness. |
| Pramod Yadav | Review-text-enhanced NCF using sentence embeddings | About 71% of reviews have meaningful text averaging 125 words. Encoding review semantics with a pretrained sentence-transformer and fusing those embeddings with the GMF user/item representations could capture nuanced preferences that aren't visible in ratings alone — like whether a user cares about "quiet rooms" or "friendly staff." |
| Hriday Ampavatina | Temporal-aware recommendation with time decay | The dataset spans 2001–2019. User preferences evolve over time — a hotel someone reviewed in 2010 may not reflect their current taste. Adding time-decay weighting or positional encoding to the interaction history could help the model focus on more recent preferences and improve relevance. |

---

## Section 6 — Team Self-Assessment

### 6a. Incomplete or Uncertain Areas

The GMF baseline code is complete and the full pipeline (raw data → preprocessing → k-core filtering → train/val/test split → training → evaluation) is end-to-end runnable. We've validated it with synthetic data and the architecture matches the paper exactly.

The main uncertainty is that we haven't yet reproduced the exact paper numbers on the real 20-core dataset — training takes about 2 hours on a single GPU and we're still waiting on compute time. We're fairly confident the numbers will land close to what the paper reports (HR@10 ≈ 0.52, NDCG@10 ≈ 0.30) since the architecture and training procedure match, but there could be small differences due to random seed, negative sampling strategy, or hyperparameter details not fully specified in the paper.

### 6b. What We Need Before Phase 2

- **Compute access** — training on the 5-core subset (21M reviews) will require either a multi-GPU setup or patience. We'd appreciate guidance on whether the 20-core results are sufficient for the final report or if 5-core is expected.

- **Text encoder strategy** — for Pramod's text-enhanced variant, we're debating whether to fine-tune the sentence-transformer end-to-end or freeze it and only train a projection layer. Fine-tuning would be more expressive but much more expensive. Any guidance on what's reasonable for a class project would help.
