"""
Preprocess raw HotelRec data into a clean parquet dataset.

The HotelRec dataset (Antognini & Faltings, LREC 2020) is distributed as a
single large JSON file (~10GB, 50M reviews) from SWITCHdrive. Loading it all
at once would need 50+ GB RAM, so we stream-parse with ijson and write in
chunks to parquet.

Each review has:
  - hotel_url, author, date, rating, title, text, property_dict (sub-ratings)

Output: parquet with columns [user_id, item_id, rating, text, date],
        plus JSON files for user/item ID mappings.

Usage:
    python -m src.data.preprocess --kcore 20 --config configs/data.yaml
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# sub-rating keys as they appear in the actual HotelRec property_dict
SUB_RATING_KEYS = [
    "service", "cleanliness", "location", "value",
    "rooms", "sleep quality", "check in / front desk", "business service",
]

SUB_RATING_COLS = [
    "Service", "Cleanliness", "Location", "Value",
    "Rooms", "Sleep Quality", "Check-In", "Business Service",
]


def parse_review(r: dict) -> dict:
    """Parse a single review dict from the HotelRec JSON format."""
    rec = {
        "user_url": r.get("author", r.get("user_url", "")),
        "hotel_url": r.get("hotel_url", ""),
        "rating": float(r.get("rating", r.get("overall_rating", 0))),
        "text": r.get("text", ""),
        "date": r.get("date", ""),
    }
    props = r.get("property_dict", {}) or {}
    for raw_key, col_name in zip(SUB_RATING_KEYS, SUB_RATING_COLS):
        val = props.get(raw_key, r.get(col_name, r.get(raw_key, None)))
        rec[col_name] = float(val) if val is not None else None
    return rec


def stream_reviews(fpath: str):
    """
    Stream-parse a large JSON file of reviews without loading it all into RAM.

    Uses ijson if available (handles 10GB+ files in ~2GB RAM).
    Falls back to chunked reading for smaller files.
    """
    try:
        import ijson
        print(f"  Using ijson for streaming parse (memory-safe)")
        with open(fpath, "rb") as f:
            # ijson.items yields each element of the top-level JSON array
            for review in ijson.items(f, "item"):
                yield parse_review(review)
    except ImportError:
        # Fallback: load entire file. Only works if you have enough RAM.
        print(f"  WARNING: ijson not installed. Loading entire file into memory.")
        print(f"  For large files, install ijson: pip install ijson")
        with open(fpath) as f:
            data = json.load(f)
        if isinstance(data, list):
            for r in data:
                yield parse_review(r)
        else:
            yield parse_review(data)


def load_raw_data(raw_dir: str, chunk_size: int = 500_000) -> pd.DataFrame:
    """
    Load HotelRec data from raw_dir. Handles both single-file (HotelRec.json)
    and multi-file layouts. Streams large files in chunks to limit memory.
    """
    raw_path = Path(raw_dir)
    json_files = sorted(raw_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {raw_dir}")

    print(f"Found {len(json_files)} JSON file(s) in {raw_dir}")

    all_chunks = []
    total = 0
    chunk_buf = []

    for fpath in json_files:
        fsize = fpath.stat().st_size
        print(f"  Parsing {fpath.name} ({fsize / 1e9:.1f} GB)...")

        if fsize > 100_000_000:  # > 100MB → stream
            for rec in stream_reviews(str(fpath)):
                chunk_buf.append(rec)
                if len(chunk_buf) >= chunk_size:
                    df_chunk = pd.DataFrame(chunk_buf)
                    all_chunks.append(df_chunk)
                    total += len(chunk_buf)
                    print(f"    {total:,} reviews processed...")
                    chunk_buf = []
        else:
            # Small file — load directly
            with open(fpath) as f:
                data = json.load(f)
            if isinstance(data, list):
                for r in data:
                    chunk_buf.append(parse_review(r))
            elif isinstance(data, dict):
                chunk_buf.append(parse_review(data))

    # flush remaining
    if chunk_buf:
        all_chunks.append(pd.DataFrame(chunk_buf))
        total += len(chunk_buf)

    df = pd.concat(all_chunks, ignore_index=True)
    print(f"Loaded {len(df):,} raw reviews from {len(json_files)} file(s)")
    return df


def build_id_maps(df: pd.DataFrame) -> tuple:
    """Map user_url and hotel_url to contiguous integer IDs."""
    users = sorted(df["user_url"].unique())
    items = sorted(df["hotel_url"].unique())

    user2id = {u: i for i, u in enumerate(users)}
    item2id = {it: i for i, it in enumerate(items)}

    df = df.copy()
    df["user_id"] = df["user_url"].map(user2id)
    df["item_id"] = df["hotel_url"].map(item2id)

    return user2id, item2id, df


def kcore_filter(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Iteratively remove users and items with fewer than k interactions
    until convergence.
    """
    prev_len = -1
    iteration = 0
    while len(df) != prev_len:
        prev_len = len(df)
        iteration += 1

        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df = df[df["user_id"].isin(valid_users)]

        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df = df[df["item_id"].isin(valid_items)]

        print(f"  k-core iter {iteration}: {len(df):,} interactions, "
              f"{df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")

    return df.reset_index(drop=True)


def remap_ids(df: pd.DataFrame) -> tuple:
    """Re-map IDs to contiguous range after k-core filtering."""
    users = sorted(df["user_url"].unique())
    items = sorted(df["hotel_url"].unique())

    user2id = {u: i for i, u in enumerate(users)}
    item2id = {it: i for i, it in enumerate(items)}

    df = df.copy()
    df["user_id"] = df["user_url"].map(user2id)
    df["item_id"] = df["hotel_url"].map(item2id)

    return user2id, item2id, df


def print_stats(df: pd.DataFrame, label: str = ""):
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    n_inter = len(df)
    sparsity = 1.0 - n_inter / (n_users * n_items)
    print(f"\n--- {label} Statistics ---")
    print(f"Users:        {n_users:,}")
    print(f"Items:        {n_items:,}")
    print(f"Interactions: {n_inter:,}")
    print(f"Sparsity:     {sparsity:.6f}")
    print(f"Avg reviews/user: {n_inter / n_users:.1f}")
    print(f"Avg reviews/item: {n_inter / n_items:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess HotelRec data")
    parser.add_argument("--kcore", type=int, default=None,
                        help="k-core value (overrides config default)")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    k = args.kcore or cfg["kcore"]["default"]
    raw_dir = cfg["dataset"]["raw_dir"]
    out_dir = cfg["dataset"]["processed_dir"]

    print(f"Config: {args.config}")
    print(f"k-core: {k}")
    print(f"Raw dir: {raw_dir}")

    # load (streaming for large files)
    df = load_raw_data(raw_dir)

    # drop reviews with missing rating
    df = df[df["rating"] > 0].reset_index(drop=True)

    # build initial ID maps
    _, _, df = build_id_maps(df)

    # k-core filtering
    print(f"\nApplying {k}-core filtering...")
    df = kcore_filter(df, k)

    # remap IDs to contiguous range after filtering
    user2id, item2id, df = remap_ids(df)

    print_stats(df, f"{k}-core")

    # prepare output columns
    sub_ratings = cfg.get("sub_ratings", [])
    keep_cols = ["user_id", "item_id", "rating", "text", "date"] + sub_ratings
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # save
    kcore_dir = os.path.join(out_dir, f"{k}core")
    os.makedirs(kcore_dir, exist_ok=True)

    parquet_path = os.path.join(kcore_dir, "interactions.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved interactions to {parquet_path}")

    with open(os.path.join(kcore_dir, "user2id.json"), "w") as f:
        json.dump(user2id, f)
    with open(os.path.join(kcore_dir, "item2id.json"), "w") as f:
        json.dump(item2id, f)
    print(f"Saved ID mappings to {kcore_dir}/")


if __name__ == "__main__":
    main()
