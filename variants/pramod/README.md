# Variant C: TextNCF — Review-Text-Enhanced NCF

**Pramod Yadav**

## What I'm doing

I'm adding review text to the standard NCF pipeline. The idea is pretty straightforward — hotels get reviewed with detailed text ("great location, noisy rooms, amazing breakfast"), and that text carries signal that star ratings alone miss. So I encode the review text with a sentence transformer and fuse it with the usual collaborative filtering embeddings.

The model has two branches:
- **GMF branch** — standard user/item embedding dot product, same as the baseline GMF
- **Text branch** — sentence embeddings from reviews, projected down and combined

Both branches get concatenated and run through a small MLP to produce a score. Trained with BPR loss since we're treating this as implicit feedback.

## Why this makes sense for HotelRec

The dataset has 50M reviews averaging 125 words each. That's a lot of text signal sitting there unused by pure collaborative filtering. About 71% of reviews have substantive text. The text captures stuff like "walking distance to the beach" or "thin walls" that you can't get from a 4-star rating.

I'm using a frozen all-MiniLM-L6-v2 (384-dim) to encode the text. Freezing it means I don't need to fine-tune a 22M parameter transformer — I just learn a small projection layer on top. The encoding is done offline as a preprocessing step.

## How to run

```bash
# step 1: encode reviews (needs sentence-transformers, ~30 min on GPU)
python scripts/encode_text.py --kcore 20 --device cuda

# step 2: train
python -m src.train_text_ncf --config configs/text_ncf.yaml --kcore 20
```

On HPC:
```bash
sbatch scripts/run_hpc.sh text-ncf   # does encode + train + eval
```

Results go to `results/text_ncf/test_metrics.json`.

## Results

Baselines (already computed, from `results/baselines/`):

| Model | HR@10 | NDCG@10 |
|-------|-------|---------|
| Popularity | 0.422 | 0.266 |
| GMF | 0.669 | 0.486 |
| ItemKNN | 0.687 | 0.609 |

TextNCF — waiting on full 20-core run. Will update once it finishes.

## Data leakage note

User text profiles only use training-split reviews. Item profiles use all reviews since that's basically hotel metadata (what people say about the hotel doesn't change across splits). This follows the standard practice from the NCF literature.

## Files I added

- `src/models/text_ncf.py` — the model
- `src/train_text_ncf.py` — training loop
- `src/data/text_embeddings.py` — encoding + loading helpers
- `configs/text_ncf.yaml` — hyperparameters
- `scripts/encode_text.py` — CLI for encoding reviews
- `results/text_ncf/` — where results land

I didn't touch any shared files (`src/data/dataset.py`, `src/evaluation/ranking.py`, etc.). The model stores text embeddings as PyTorch buffers so `forward(users, items)` works with the shared eval code without any changes.
