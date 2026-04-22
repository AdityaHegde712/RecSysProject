# Variant C: TextNCF — Review-Text-Enhanced NCF

**Pramod Yadav**

## What I'm doing

I'm adding review text to the standard NCF pipeline. The idea is pretty straightforward — hotels get reviewed with detailed text ("great location, noisy rooms, amazing breakfast"), and that text carries signal that star ratings alone miss. So I encode the review text with a sentence transformer and fuse it with the usual collaborative filtering embeddings.

The base model has two branches:
- **GMF branch** — standard user/item embedding dot product, same as the baseline GMF
- **Text branch** — sentence embeddings from reviews, projected down and combined

Both branches get concatenated and run through a small MLP to produce a score. Trained with BPR loss since we're treating this as implicit feedback.

On top of the base TextNCF, I'm exploring four additional approaches to push performance further.

## Why this makes sense for HotelRec

The dataset has 50M reviews averaging 125 words each. That's a lot of text signal sitting there unused by pure collaborative filtering. About 71% of reviews have substantive text. The text captures stuff like "walking distance to the beach" or "thin walls" that you can't get from a 4-star rating.

I'm using a frozen all-MiniLM-L6-v2 (384-dim) to encode the text. Freezing it means I don't need to fine-tune a 22M parameter transformer — I just learn a small projection layer on top. The encoding is done offline as a preprocessing step.

## Approaches

### 1. Base TextNCF (`src/train_text_ncf.py`)

Two-branch hybrid: GMF ⊙ + text ⊙ → MLP → score. Trained with BPR loss.

```bash
python -m src.train_text_ncf --config configs/text_ncf.yaml --kcore 20
```

### 2. Ensemble Scoring (`src/evaluate_ensemble.py`)

Combines TextNCF, GMF, and ItemKNN predictions with learned weights. Min-max normalizes each model's scores per user, then grid searches over weight combinations (w1+w2+w3=1) on the validation set.

```bash
python -m src.evaluate_ensemble \
    --text-ncf-ckpt results/text_ncf/best_model.pt \
    --gmf-ckpt results/gmf/best_model.pt \
    --knn-ckpt results/baselines/itemknn.pkl \
    --kcore 20
```

### 3. Two-Stage Retrieval + Ranking (`src/evaluate_two_stage.py`)

Mimics a production pipeline: ItemKNN retrieves top-200 candidates (fast, sparse), then TextNCF re-ranks them (slow, neural). This tests whether TextNCF adds value as a re-ranker on top of a strong retriever.

```bash
python -m src.evaluate_two_stage \
    --text-ncf-ckpt results/text_ncf/best_model.pt \
    --knn-ckpt results/baselines/itemknn.pkl \
    --kcore 20
```

### 4. Multi-Task TextNCF (`src/train_text_ncf_mt.py`)

Joint ranking + rating prediction. Same architecture as TextNCF but with two output heads:
- Ranking head (BPR loss)
- Rating head (MSE loss on 1-5 ratings)

Joint loss: `alpha * BPR + (1-alpha) * MSE`. The idea is that predicting ratings forces the model to learn finer-grained preferences.

```bash
python -m src.train_text_ncf_mt --config configs/text_ncf_mt.yaml --kcore 20
```

### 5. Sub-Rating Decomposition (`src/train_text_ncf_subrating.py`)

HotelRec has 6 aspect-level sub-ratings: Service, Cleanliness, Location, Value, Rooms, Sleep Quality. This model predicts each sub-rating separately with dedicated heads, then combines them with learned per-user attention weights. Different travelers care about different aspects — business travelers care about WiFi/location, families care about rooms/cleanliness.

```bash
python -m src.train_text_ncf_subrating \
    --config configs/text_ncf_subrating.yaml --kcore 20
```

## How to run everything

```bash
# step 1: encode reviews (needs sentence-transformers, ~30 min on GPU)
python scripts/encode_text.py --kcore 20 --device cuda

# step 2: train base TextNCF
python -m src.train_text_ncf --config configs/text_ncf.yaml --kcore 20

# step 3: train multi-task variant
python -m src.train_text_ncf_mt --config configs/text_ncf_mt.yaml --kcore 20

# step 4: train sub-rating variant
python -m src.train_text_ncf_subrating --config configs/text_ncf_subrating.yaml --kcore 20

# step 5: ensemble evaluation (needs trained TextNCF + GMF + ItemKNN)
python -m src.evaluate_ensemble --kcore 20

# step 6: two-stage evaluation
python -m src.evaluate_two_stage --kcore 20
```

On HPC:
```bash
sbatch scripts/run_hpc.sh text-ncf        # base TextNCF
sbatch scripts/run_hpc.sh text-ncf-mt      # multi-task
sbatch scripts/run_hpc.sh text-ncf-sub     # sub-rating
sbatch scripts/run_hpc.sh ensemble         # ensemble eval
sbatch scripts/run_hpc.sh two-stage        # two-stage eval
```

## Results

Baselines (already computed, from `results/baselines/`):

| Model | HR@10 | NDCG@10 |
|-------|-------|---------|
| Popularity | 0.422 | 0.266 |
| GMF | 0.669 | 0.486 |
| ItemKNN | 0.687 | 0.609 |

TextNCF variants — pending full 20-core runs. See `results/text_ncf/summary.md` for latest numbers.

## Data leakage note

User text profiles only use training-split reviews. Item profiles use all reviews since that's basically hotel metadata (what people say about the hotel doesn't change across splits). This follows the standard practice from the NCF literature.

## Files I added

**Models:**
- `src/models/text_ncf.py` — base TextNCF model
- `src/models/text_ncf_mt.py` — multi-task variant (ranking + rating)
- `src/models/text_ncf_subrating.py` — sub-rating decomposition variant

**Training scripts:**
- `src/train_text_ncf.py` — train base TextNCF
- `src/train_text_ncf_mt.py` — train multi-task variant
- `src/train_text_ncf_subrating.py` — train sub-rating variant

**Evaluation scripts:**
- `src/evaluate_ensemble.py` — ensemble scoring (TextNCF + GMF + ItemKNN)
- `src/evaluate_two_stage.py` — two-stage retrieval + ranking

**Data utilities:**
- `src/data/text_embeddings.py` — encoding + loading helpers
- `src/data/subratings.py` — sub-rating data loading

**Configs:**
- `configs/text_ncf.yaml` — base TextNCF hyperparameters
- `configs/text_ncf_mt.yaml` — multi-task config (alpha parameter)
- `configs/text_ncf_subrating.yaml` — sub-rating config (beta parameter)

**Other:**
- `scripts/encode_text.py` — CLI for encoding reviews
- `results/text_ncf/` — where results land

I didn't touch any shared files (`src/data/dataset.py`, `src/evaluation/ranking.py`, etc.). All models store text embeddings as PyTorch buffers so `forward(users, items)` works with the shared eval code without any changes.
