"""
Text embedding utilities for TextNCF variant.

Pre-computes sentence embeddings with all-MiniLM-L6-v2 and averages
per user/item. Stores results as .npy files in data/processed/text_emb/.

To avoid data leakage:
  - user profiles use training-split reviews only
  - item profiles use all reviews (hotel metadata, not a target)

"""

import os
import time

import numpy as np
import pandas as pd


TEXT_EMB_DIR = "data/processed/text_emb"


def load_text_embeddings(emb_dir: str = TEXT_EMB_DIR):
    """Load pre-computed user and item text embeddings.

    Returns (user_emb, item_emb) as numpy arrays.
    """
    user_path = os.path.join(emb_dir, "user_text_emb.npy")
    item_path = os.path.join(emb_dir, "item_text_emb.npy")

    if not os.path.exists(user_path) or not os.path.exists(item_path):
        raise FileNotFoundError(
            f"Text embeddings not found at {emb_dir}/. "
            f"Run: python scripts/encode_text.py --kcore 20"
        )

    user_emb = np.load(user_path)
    item_emb = np.load(item_path)
    print(f"Loaded text embeddings: users={user_emb.shape}, items={item_emb.shape}")
    return user_emb, item_emb


def _read_df(kcore_dir, name):
    """Try parquet first, fall back to CSV."""
    pq = os.path.join(kcore_dir, f"{name}.parquet")
    csv = os.path.join(kcore_dir, f"{name}.csv")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    if os.path.exists(csv):
        return pd.read_csv(csv)
    raise FileNotFoundError(f"no {name} file at {pq} or {csv}")


def clean_texts(series):
    """Fill NaN, convert to str, replace blanks with a placeholder."""
    texts = series.fillna("").astype(str).tolist()
    for i in range(len(texts)):
        if not texts[i].strip():
            texts[i] = "no review"
    return texts


def average_by_id(ids, embeddings, num_ids, dim):
    """Average embeddings grouped by ID. Returns (num_ids, dim) array."""
    sums = np.zeros((num_ids, dim), dtype=np.float64)
    counts = np.zeros(num_ids, dtype=np.int64)

    for i in range(len(ids)):
        idx = ids[i]
        sums[idx] += embeddings[i]
        counts[idx] += 1

    result = np.zeros((num_ids, dim), dtype=np.float32)
    mask = counts > 0
    result[mask] = (sums[mask] / counts[mask, np.newaxis]).astype(np.float32)

    n_missing = int(np.sum(~mask))
    if n_missing > 0:
        print(f"  {n_missing} IDs have no reviews, using zero vector")

    return result


def encode_reviews(
    kcore_dir: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 512,
    device: str = "cuda",
    output_dir: str = TEXT_EMB_DIR,
):
    """Encode all reviews and compute per-user/per-item averages.

    Saves user_text_emb.npy and item_text_emb.npy to output_dir.
    """
    from sentence_transformers import SentenceTransformer

    os.makedirs(output_dir, exist_ok=True)

    user_path = os.path.join(output_dir, "user_text_emb.npy")
    item_path = os.path.join(output_dir, "item_text_emb.npy")

    if os.path.exists(user_path) and os.path.exists(item_path):
        print(f"Embeddings already exist at {output_dir}, skipping")
        return

    print(f"Loading data from {kcore_dir}...")
    all_df = _read_df(kcore_dir, "interactions")
    train_df = _read_df(kcore_dir, "train")

    num_users = int(all_df["user_id"].max() + 1)
    num_items = int(all_df["item_id"].max() + 1)
    print(f"  {len(all_df):,} interactions, {num_users:,} users, {num_items:,} items")

    print(f"\nLoading {model_name}...")
    encoder = SentenceTransformer(model_name, device=device)
    dim = encoder.get_sentence_embedding_dimension()
    print(f"  loaded on {device}, dim={dim}")

    # encode all reviews
    print(f"\nEncoding {len(all_df):,} reviews...")
    t0 = time.time()
    texts = clean_texts(all_df["text"])

    n = len(texts)
    all_embs = np.zeros((n, dim), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        emb = encoder.encode(texts[start:end], show_progress_bar=False,
                             convert_to_numpy=True)
        all_embs[start:end] = emb
        if (start // batch_size) % 20 == 0:
            print(f"  {end:,}/{n:,} ({100*end/n:.1f}%)", flush=True)

    print(f"  done in {time.time() - t0:.0f}s")

    # item profiles: average over all reviews
    print("\nAggregating item profiles (all splits)...")
    item_emb = average_by_id(all_df["item_id"].values, all_embs, num_items, dim)

    # user profiles: average over training reviews only
    print("Aggregating user profiles (train split only)...")
    t0 = time.time()
    train_pairs = set(zip(train_df["user_id"].values, train_df["item_id"].values))
    train_mask = np.array([
        (u, i) in train_pairs
        for u, i in zip(all_df["user_id"].values, all_df["item_id"].values)
    ], dtype=bool)

    train_user_ids = all_df["user_id"].values[train_mask]
    train_embs = all_embs[train_mask]
    user_emb = average_by_id(train_user_ids, train_embs, num_users, dim)
    print(f"  done in {time.time() - t0:.0f}s")

    np.save(user_path, user_emb)
    np.save(item_path, item_emb)
    print(f"\nSaved {user_path} ({os.path.getsize(user_path)/1e6:.1f} MB)")
    print(f"Saved {item_path} ({os.path.getsize(item_path)/1e6:.1f} MB)")
