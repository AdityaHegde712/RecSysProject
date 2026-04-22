# TextNCF Design Notes

## The idea

Take the standard GMF model and bolt on review text. Hotels have tons of text reviews — 50M of them in HotelRec — and that text tells you things ratings can't. A 4-star review that says "perfect for business travel" is very different from one that says "great for families."

I encode each review with a frozen sentence transformer (all-MiniLM-L6-v2, 384-dim), average the embeddings per user and per item, then project them down to 64-dim and combine with the GMF embeddings through a small MLP.

## Architecture sketch

```
user_id → Embedding(64) ─┐
                          ├─ ⊙ → gmf_vec (64)
item_id → Embedding(64) ─┘                    ╲
                                                concat(128) → MLP → score
user_reviews → avg(MiniLM) → Linear(64) ─┐    ╱
                                          ├─ ⊙ → text_vec (64)
item_reviews → avg(MiniLM) → Linear(64) ─┘
```

The MLP is just two hidden layers (128 → 64) with ReLU and dropout. Nothing fancy.

## Why I made these choices

**Averaging instead of attention**: With 50M reviews, attention would be slow and hard to debug. Averaging is simple and gives a reasonable "profile" of what a user/hotel is about. Could try attention later as an extension.

**Frozen encoder**: Fine-tuning a 22M-param transformer would dominate the model. The whole point is to see if text helps on top of collaborative filtering, not to build a giant text model.

**BPR loss**: Standard for implicit feedback. User reviewed hotel = positive, random hotel = negative.

**Registered buffers for text embeddings**: The shared eval code calls `model(users, items)` with no text args. Storing the text embeddings as PyTorch buffers means they move to GPU automatically and get looked up inside forward(). No need to change the eval interface.

## Data leakage

This tripped me up initially. If I use test-set reviews to build user profiles, the model sees future information during training. So:
- User profiles: only training-split reviews
- Item profiles: all reviews (it's hotel metadata, not user behavior)

## What I need to run

1. `sentence-transformers` for encoding (only needed once, offline)
2. The usual PyTorch + numpy for training
3. Pre-computed embeddings stored as .npy files in `data/processed/text_emb/`

## Ablations I want to run

- Full model (GMF + text) — the main result
- GMF-only (text off) — is the text actually helping?
- Text-only (GMF off) — how far can text alone get?

These are controlled by `use_gmf` and `use_text` flags in the config.
