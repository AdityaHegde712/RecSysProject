# Benchmarking Recommendation Models on the HotelRec Dataset

CMPE 256 — Spring 2026
Team: Aditya Hegde, Pramod Yadav, Hriday Ampavatina

---

## Overview

We use **GMF** (Generalized Matrix Factorization, He et al., 2017) as our baseline for hotel recommendation on the [HotelRec](https://github.com/Diego999/HotelRec) dataset — 50M TripAdvisor hotel reviews. GMF learns user and item embeddings and scores interactions via element-wise product followed by a linear layer and sigmoid activation.

The hotel domain has much higher sparsity than typical recommendation datasets (99.999% sparse at 5-core), which makes collaborative filtering particularly challenging. GMF handles this well because it learns dense embeddings that capture indirect user-item relationships, rather than relying on direct overlap like neighbor-based methods.

> **Note:** The full set of baselines (Mean, ItemKNN, UserKNN, PureSVD, MLP, NeuMF) is preserved on the [`feature/all-baselines`](../../tree/feature/all-baselines) branch.

---

## Dataset

**HotelRec** (Antognini & Faltings, 2020) — scraped from TripAdvisor, covering ~50M hotel reviews across 365K hotels and 22M users.

| Subset | Users | Items | Interactions | Sparsity |
|--------|-------|-------|-------------|----------|
| Full | 21,891,294 | 365,056 | 50,264,531 | 99.999% |
| 5-core | 2,012,162 | 312,081 | 21,108,245 | 99.997% |
| 20-core | 72,603 | 38,903 | 2,222,373 | 99.921% |

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
│   └── gmf.yaml                # GMF model config
├── data/
│   ├── raw/                    # original HotelRec JSON files
│   └── processed/              # preprocessed interaction matrices
├── docs/                       # additional documentation
├── logs/                       # training/eval logs
├── notebooks/                  # exploratory analysis
├── results/                    # evaluation outputs, checkpoints
├── scripts/                    # shell scripts, HPC job files
│
└── src/
    ├── __init__.py
    ├── train.py                # training entry point (BPR loss)
    ├── evaluate.py             # evaluation entry point (HR@k, NDCG@k)
    ├── data/                   # data loading, preprocessing, splitting
    │   ├── __init__.py
    │   ├── dataset.py          # InteractionDataset, EvalInteractionDataset
    │   ├── preprocess.py       # raw JSON → filtered parquet
    │   └── split.py            # train/val/test splitting
    ├── models/
    │   ├── __init__.py
    │   ├── common.py           # build_model() factory
    │   └── gmf.py              # Generalized Matrix Factorization
    ├── metrics/
    │   ├── __init__.py
    │   └── ranking.py          # HR@k, NDCG@k, evaluate_ranking()
    └── utils/
        ├── __init__.py
        ├── io.py               # config loading, checkpoint save/load
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

# train GMF
python -m src.train --config configs/gmf.yaml --kcore 20

# evaluate
python -m src.evaluate --config configs/gmf.yaml --kcore 20
```

---

## Running on SJSU HPC

The HPC cluster runs CentOS 7 with GLIBC 2.17. We pin all dependencies in `requirements-hpc.txt` to avoid binary compatibility issues.

### Setup (one-time)

```bash
ssh <sjsu-id>@coe-hpc1.sjsu.edu
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

These are the GMF numbers reported in Antognini & Faltings (2020), Table 5.

**5-core subset:**

| Model | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|-------|------|--------|-------|---------|-------|---------|
| GMF | 0.3899 | 0.2761 | 0.5340 | 0.3237 | 0.7055 | 0.3666 |

**20-core subset:**

| Model | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|-------|------|--------|-------|---------|-------|---------|
| GMF | 0.3705 | 0.2565 | 0.5219 | 0.3047 | 0.6913 | 0.3477 |

---

## References

- Antognini, D. & Faltings, B. (2020). *HotelRec: a Novel Very Large-Scale Hotel Recommendation Dataset*. In Proceedings of the 12th Language Resources and Evaluation Conference (LREC 2020), pages 4917–4924. [[Paper]](https://aclanthology.org/2020.lrec-1.605/) [[Dataset]](https://github.com/Diego999/HotelRec)
- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). *Neural Collaborative Filtering*. In Proceedings of the 26th International Conference on World Wide Web, pages 173–182. [[Paper]](https://arxiv.org/abs/1708.05031)
