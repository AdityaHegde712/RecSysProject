"""
Preprocess raw HotelRec JSON files into a clean parquet dataset.

Raw format: one JSON file per hotel, each containing an array of review objects.
Output: single parquet with columns [user_id, item_id, rating, text, date, <sub_ratings>],
        plus JSON files for user/item ID mappings.

Usage:
    python -m src.data.preprocess --kcore 20 --config configs/data.yaml
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_hotel_json(fpath: str) -> list[dict]:
    """Parse a single hotel JSON file and return flat review records."""
    with open(fpath) as f:
        reviews = json.load(f)

    records = []
    for r in reviews:
        rec = {
            "user_url": r.get("user_url", ""),
            "hotel_url": r.get("hotel_url", ""),
            "rating": float(r.get("overall_rating", 0)),
            "text": r.get("text", ""),
            "date": r.get("date", ""),
        }
        # sub-ratings are optional
        for sr in [
            "Service", "Cleanliness", "Location", "Value",
            "Rooms", "Sleep Quality", "Check-In", "Business Service",
        ]:
            val = r.get(sr, None)
            rec[sr] = float(val) if val is not None else None
        records.append(rec)
    return records


def load_raw_data(raw_dir: str) -> pd.DataFrame:
    """Load all hotel JSON files from raw_dir into a single DataFrame."""
    json_files = sorted(Path(raw_dir).glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {raw_dir}")

    all_records = []
    for fpath in tqdm(json_files, desc="Loading JSON files"):
        all_records.extend(parse_hotel_json(str(fpath)))

    df = pd.DataFrame(all_records)
    print(f"Loaded {len(df):,} raw reviews from {len(json_files)} files")
    return df


def build_id_maps(df: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
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
    until convergence. Standard k-core filtering.
    """
    prev_len = -1
    iteration = 0
    while len(df) != prev_len:
        prev_len = len(df)
        iteration += 1

        # filter users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df = df[df["user_id"].isin(valid_users)]

        # filter items
        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df = df[df["item_id"].isin(valid_items)]

        print(f"  k-core iter {iteration}: {len(df):,} interactions, "
              f"{df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")

    return df.reset_index(drop=True)


def remap_ids(df: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
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

    # load
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
    # only keep sub_rating cols that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # save
    kcore_dir = os.path.join(out_dir, f"{k}core")
    os.makedirs(kcore_dir, exist_ok=True)

    parquet_path = os.path.join(kcore_dir, "interactions.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved interactions to {parquet_path}")

    # save ID mappings
    with open(os.path.join(kcore_dir, "user2id.json"), "w") as f:
        json.dump(user2id, f)
    with open(os.path.join(kcore_dir, "item2id.json"), "w") as f:
        json.dump(item2id, f)
    print(f"Saved ID mappings to {kcore_dir}/")


if __name__ == "__main__":
    main()
