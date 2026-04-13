# Benchmarking Recommendation Models on the HotelRec Dataset

CMPE 256 — Spring 2026
Team: Aditya Hegde, Pramod Yadav, Hriday Ampavatina

---

## Overview

We use **ItemKNN** (Sarwar et al., 2001) as our baseline — a standard item-based collaborative filtering method that recommends items similar to those the user has already interacted with, using cosine similarity. It builds a sparse user-item interaction matrix, computes item-item cosine similarity, and scores candidates by summing similarities to items in the user's history.

ItemKNN is intentionally simple and non-neural. It runs on CPU in minutes, doesn't need a GPU, and provides a clear lower bound that neural approaches should beat. The hotel domain has extreme sparsity (99.999% at 5-core), which makes neighbor-based methods struggle — exactly the gap we want our neural variants to close.

> **Note:** The full set of baselines (Mean, ItemKNN, UserKNN, PureSVD, GMF, MLP, NeuMF) is preserved on the [`feature/all-baselines`](../../tree/feature/all-baselines) branch.

---

## Dataset

**HotelRec** (Antognini & Faltings, 2020) — scraped from TripAdvisor, covering ~50M hotel reviews across 365K hotels and 22M users. Numbers below are from our own scan of the full dataset (`hpa-explore` on all 50M reviews); the paper reports 21,891,294 users and 365,056 items — the small difference is likely from deduplication.

| Subset | Users | Items | Interactions | Sparsity |
|--------|-------|-------|-------------|----------|
| Full | 21,891,404 | 365,057 | 50,264,531 | 99.999% |
| 5-core | 2,012,162 | 312,081 | 21,108,245 | 99.997% |
| 20-core | 72,603 | 38,903 | 2,222,373 | 99.921% |

5-core and 20-core numbers are from the paper (we haven't run preprocessing yet). Full-dataset stats are verified — see [`results/data_evaluation.json`](results/data_evaluation.json) for the complete breakdown including rating distribution, sub-rating coverage, user activity buckets, and temporal trends.

We use the **5-core** and **20-core** subsets. Each review provides a userID, itemID, overall rating (1–5), text, and date. We split 80/10/10 for train/validation/test, following the paper.

The k-core filtering ensures every user and every item has at least k interactions — standard practice to avoid cold-start noise in baseline evaluation.

---

## Project Structure

```
HotelRec-HPA/
├── setup.py
├── requirements.txt            # flexible versions for local dev
├── requirements-hpc.txt        # pinned for SJSU HPC (CentOS 7)
├── .gitignore
├── .python-version
├── README.md
│
├── configs/
│   ├── data.yaml               # dataset paths, k-core values, split ratios
│   └── itemknn.yaml            # ItemKNN model config
├── data/
│   ├── raw/                    # original HotelRec JSON files
│   └── processed/              # preprocessed interaction matrices
├── docs/                       # additional documentation
├── notebooks/                  # exploratory analysis
├── results/                    # evaluation outputs, checkpoints
├── scripts/                    # shell scripts, HPC job files
│
└── src/
    ├── __init__.py
    ├── train.py                # training entry point (fit ItemKNN)
    ├── evaluate.py             # evaluation entry point (HR@k, NDCG@k)
    ├── data/                   # data loading, preprocessing, splitting
    │   ├── __init__.py
    │   ├── dataset.py          # load_split(), get_n_users_items()
    │   ├── preprocess.py       # raw JSON → filtered parquet
    │   └── split.py            # train/val/test splitting
    ├── models/
    │   ├── __init__.py
    │   ├── common.py           # build_model() factory
    │   └── knn.py              # Item-based KNN collaborative filtering
    ├── metrics/
    │   ├── __init__.py
    │   └── ranking.py          # HR@k, NDCG@k, evaluate_ranking()
    └── utils/
        ├── __init__.py
        ├── io.py               # config loading, pickle save/load
        ├── seed.py             # set_seed() for reproducibility
        └── metrics_logger.py   # CSV metrics logging
```

---

## Quick Start (Local)

```bash
# clone and set up
git clone <repo-url> && cd HotelRec-HPA
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# download and preprocess data
python -m src.data.preprocess --kcore 20 --config configs/data.yaml

# train ItemKNN (no GPU needed — fits on CPU)
python -m src.train --config configs/itemknn.yaml --kcore 20

# evaluate
python -m src.evaluate --config configs/itemknn.yaml --kcore 20
```

---

## Running on SJSU HPC

The HPC cluster runs CentOS 7 with GLIBC 2.17. We pin all dependencies in `requirements-hpc.txt` to avoid binary compatibility issues.

### SSH key for GitHub (one-time, on login node)

The HPC login node has internet access — generate a key there so you can push/pull:

```bash
ssh <sjsu-id>@coe-hpc1.sjsu.edu

# generate key (no passphrase for batch jobs)
ssh-keygen -t ed25519 -C "your-email@sjsu.edu" -f ~/.ssh/id_ed25519_github -N ""

# print the public key — copy this
cat ~/.ssh/id_ed25519_github.pub

# add to GitHub: https://github.com/settings/keys → "New SSH key" → paste

# tell SSH to use this key for github.com
cat >> ~/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_github
    IdentitiesOnly yes
EOF

# test
ssh -T git@github.com
# → "Hi <username>! You've successfully authenticated..."
```

### Clone and setup (one-time)

```bash
git clone git@github.com:AdityaHegde712/RecSysProject.git HotelRec-HPA
cd HotelRec-HPA
bash scripts/run_hpc.sh setup
source scripts/hpc_aliases.sh
```

### Submit a job

```bash
# full pipeline: preprocess → train → evaluate
sbatch scripts/run_hpc.sh

# or individual steps
sbatch scripts/run_hpc.sh preprocess
sbatch scripts/run_hpc.sh rec
sbatch scripts/run_hpc.sh eval
```

### Monitor

```bash
squeue -u $USER          # check job status
tail -f logs/slurm_*.out # watch training output
scancel <job-id>         # cancel if needed
```

---

## Expected Results

These are the ItemKNN numbers reported in Antognini & Faltings (2020), Table 5. ItemKNN doesn't need a GPU — it runs on CPU in minutes.

**5-core subset:**

| Model | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|-------|------|--------|-------|---------|-------|---------|
| ItemKNN | 0.0162 | 0.0072 | 0.0238 | 0.0088 | 0.0340 | 0.0103 |

**20-core subset:**

| Model | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|-------|------|--------|-------|---------|-------|---------|
| ItemKNN | 0.0236 | 0.0061 | 0.0411 | 0.0084 | 0.0682 | 0.0110 |

---

## Tools

- **Claude** (Anthropic) — used for scaffolding the project structure, generating boilerplate code, and troubleshooting HPC pipeline issues

---

## References

- Antognini, D. & Faltings, B. (2020). *HotelRec: a Novel Very Large-Scale Hotel Recommendation Dataset*. In Proceedings of the 12th Language Resources and Evaluation Conference (LREC 2020), pages 4917–4924. [[Paper]](https://aclanthology.org/2020.lrec-1.605/) [[Dataset]](https://github.com/Diego999/HotelRec)
- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). *Item-based Collaborative Filtering Recommendation Algorithms*. In Proceedings of the 10th International Conference on World Wide Web, pages 285–295. [[Paper]](https://dl.acm.org/doi/10.1145/371920.372071)
