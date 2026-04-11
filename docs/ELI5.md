# ELI5: HotelRec Explained Simply

> For the full technical README, see [README.md](../README.md)

---

## What Is This Project About?

We're reproducing the baseline experiments from a 2020 paper by Antognini & Faltings that introduced **HotelRec**, a massive dataset of 50 million TripAdvisor hotel reviews. The paper ran a bunch of standard recommendation algorithms on this data and reported how well they did. We focus on **GMF** (Generalized Matrix Factorization) as our baseline model.

The task is **top-K recommendation** — given a user, rank a set of candidate hotels and see if the one they actually liked ends up near the top. We measure this with Hit Rate and NDCG (higher is better).

---

## What Is the HotelRec Dataset?

The authors scraped TripAdvisor using Selenium (an automated web browser). They collected every publicly visible hotel review — about 50 million of them — covering 365K hotels and 22 million users.

Each review has:
- A user identifier (anonymized URL)
- A hotel identifier
- An overall rating (1–5 stars)
- Review text
- Date
- Up to 8 sub-ratings (Service, Cleanliness, Location, Value, Rooms, Sleep Quality, Check-In, Business Service)

The raw data is one JSON file per hotel. The full dump is roughly 50GB.

### Why Is This Dataset Interesting?

Most recommendation research uses MovieLens (27M ratings, 280K users) or Amazon reviews. HotelRec is an order of magnitude larger, and the domain is fundamentally different — most people review maybe 1–3 hotels total, while movie reviewers might rate hundreds of films. This makes the data extremely sparse.

---

## What Is k-core Filtering?

Raw data is messy. Tons of users wrote exactly one review, and some hotels have only a handful. Models can't learn anything useful from a user who reviewed one hotel — there's no pattern to find.

**k-core filtering** removes users and items with fewer than k interactions. The catch is that it's iterative: removing a user might drop a hotel below the threshold, which then drops another user, and so on. You keep looping until nothing changes.

| Subset | Users | Items | Interactions | Sparsity |
|--------|-------|-------|-------------|----------|
| Full | 21.9M | 365K | 50.3M | 99.999% |
| 5-core | 2.0M | 312K | 21.1M | 99.997% |
| 20-core | 73K | 39K | 2.2M | 99.921% |

Even after aggressive filtering (20-core), the data is still 99.92% sparse. That means for any given user-hotel pair, there's a 99.92% chance we have no data. This is way sparser than MovieLens (~93% sparse).

---

## What Is GMF?

**GMF (Generalized Matrix Factorization)** comes from the NCF framework by He et al. (2017). The idea is simple:

1. Learn a vector (embedding) for each user and each item
2. Multiply them element-wise (not dot product — each dimension gets its own weight)
3. Pass through a linear layer and sigmoid to get a score between 0 and 1

In classic matrix factorization, you'd just take the dot product of user and item vectors. GMF adds a learnable linear layer on top, which lets the model weight different latent dimensions differently. It's a small change, but it gives the model more flexibility.

### Why Does GMF Work Better Than Traditional Methods Here?

Traditional collaborative filtering (like ItemKNN or UserKNN) needs direct overlap between users — if User A and User B both rated the same hotels, we can compare them. But with 99.999% sparsity, most user pairs share zero hotels. The similarity matrix is almost entirely zeros, so the recommendations are basically random.

GMF sidesteps this by learning dense embeddings. Even if two users never rated the same hotel, their embeddings can still be similar if they rated *similar* hotels. The model learns these indirect patterns through backpropagation.

---

## What Are the Metrics?

The evaluation protocol (from He et al., 2017):

1. For each test user, take their one held-out positive item
2. Sample 99 random items the user never interacted with
3. Score all 100 candidates, rank them
4. Check if the positive item made it into the top k

| Metric | What It Measures | Intuition |
|--------|-----------------|-----------|
| **HR@k** (Hit Rate) | Did the positive item appear in the top k? | Binary: yes or no, averaged over users. |
| **NDCG@k** (Normalized DCG) | Where in the top k did it appear? | Rewards higher positions: #1 > #5 > #10. |

HR@10 = 0.52 means that for about half the test users, the hotel they actually liked was in the model's top-10 recommendations. NDCG@10 = 0.30 means it was in the top 10 but not always near the top.

---

## How to Run the Pipeline

### Quick Reference

```bash
# First time setup (on login node — has internet)
git clone <repo-url> HotelRec-HPA
cd HotelRec-HPA
bash scripts/run_hpc.sh setup
source scripts/hpc_aliases.sh

# Verify environment
hpa-verify

# Run everything (submits SLURM job to GPU node)
hpa-run

# Or step by step:
hpa-preprocess          # k-core filter + split
hpa-rec                 # train GMF
hpa-eval                # evaluate on test set
```

### What Each Step Does

| Command | What Happens | Time (20-core) |
|---------|-------------|----------------|
| `hpa-preprocess` | Parse JSON → k-core filter → train/val/test parquets | ~10 min |
| `hpa-rec` | Train GMF with BPR loss (20 epochs) | ~1 hour |
| `hpa-eval` | Evaluate GMF → HR@k, NDCG@k | ~5 min |

### Checking Results

```bash
# View latest SLURM log
lastlog

# List saved results
results

# Check model checkpoints
checkpoints
```

---

## Key Files

| File | What It Does |
|------|-------------|
| `configs/data.yaml` | Dataset paths, k-core values, split ratios |
| `configs/gmf.yaml` | GMF model config (embedding dim, lr, epochs) |
| `src/data/preprocess.py` | Raw JSON → filtered parquet |
| `src/data/split.py` | Train/val/test splitting |
| `src/data/dataset.py` | PyTorch Datasets with negative sampling |
| `src/models/gmf.py` | GMF model implementation |
| `src/train.py` | Training entry point |
| `src/evaluate.py` | Evaluation entry point |
| `scripts/run_hpc.sh` | SLURM job script for SJSU HPC |
| `scripts/hpc_aliases.sh` | Shell shortcuts (`hpa-run`, `hpa-eval`, etc.) |

---

## Glossary

| Term | Definition |
|------|-----------|
| **HotelRec** | Dataset of 50M TripAdvisor hotel reviews (Antognini & Faltings, 2020) |
| **k-core** | Subset where every user and item has at least k interactions |
| **Collaborative filtering** | Recommendation based on user-item interaction patterns (no content features) |
| **GMF** | Generalized Matrix Factorization — element-wise product + linear layer |
| **BPR** | Bayesian Personalized Ranking — pairwise loss for implicit feedback |
| **HR@k** | Hit Rate at k — fraction of users whose positive item is in top k |
| **NDCG@k** | Normalized Discounted Cumulative Gain — position-aware ranking metric |
| **Sparsity** | Fraction of empty cells in the user-item matrix |
| **HPC** | High-Performance Computing cluster (SJSU CoE) |
| **SLURM** | Job scheduler used on HPC clusters |
