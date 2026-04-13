"""
Preprocess raw HotelRec data into a clean parquet dataset.

Uses a two-pass approach to handle 50M reviews without OOM:
  Pass 1: Count user/item frequencies, compute k-core membership (no text in memory)
  Pass 2: Re-read file, only keep rows that survive k-core filter

Usage:
    python -m src.data.preprocess --kcore 20 --config configs/data.yaml
"""

import argparse
import json
import os
from collections import Counter

import pandas as pd
import yaml
from tqdm import tqdm


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_kcore_sets(raw_path: str, k: int) -> tuple[set, set, dict]:
    """
    Pass 1: read only user/item/rating fields, compute k-core membership.
    Returns (valid_users, valid_items, full_stats_dict).
    """
    print(f"Pass 1: counting user/item frequencies...")
    user_counts = Counter()
    item_counts = Counter()
    total = 0
    n_valid = 0

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Pass 1 (counting)", total=50_264_531):
            total += 1
            r = json.loads(line.strip())
            rating = float(r.get("rating", 0))
            user = r.get("author", "")
            item = r.get("hotel_url", "")
            if rating > 0 and user and item:
                user_counts[user] += 1
                item_counts[item] += 1
                n_valid += 1

    full_stats = {
        "total_raw": total,
        "total_valid": n_valid,
        "n_users_full": len(user_counts),
        "n_items_full": len(item_counts),
    }
    print(f"  Total raw: {total:,}, valid: {n_valid:,}")
    print(f"  Users: {len(user_counts):,}, Items: {len(item_counts):,}")

    print(f"\nComputing {k}-core membership...")
    user_items = {}
    item_users = {}

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Pass 1b (building graph)", total=50_264_531):
            r = json.loads(line.strip())
            rating = float(r.get("rating", 0))
            user = r.get("author", "")
            item = r.get("hotel_url", "")
            if rating > 0 and user and item:
                if user not in user_items:
                    user_items[user] = set()
                user_items[user].add(item)
                if item not in item_users:
                    item_users[item] = set()
                item_users[item].add(user)

    iteration = 0
    changed = True
    while changed:
        changed = False
        iteration += 1

        users_to_remove = [u for u, items in user_items.items() if len(items) < k]
        if users_to_remove:
            changed = True
            for u in users_to_remove:
                for item in user_items[u]:
                    item_users[item].discard(u)
                del user_items[u]

        items_to_remove = [i for i, users in item_users.items() if len(users) < k]
        if items_to_remove:
            changed = True
            for i in items_to_remove:
                for user in item_users[i]:
                    user_items[user].discard(i)
                del item_users[i]

        n_users = len(user_items)
        n_items = len(item_users)
        n_inter = sum(len(v) for v in user_items.values())
        print(f"  k-core iter {iteration}: {n_inter:,} interactions, "
              f"{n_users:,} users, {n_items:,} items")

    valid_users = set(user_items.keys())
    valid_items = set(item_users.keys())

    n_inter = sum(len(v) for v in user_items.values())
    sparsity = 1.0 - n_inter / (len(valid_users) * len(valid_items)) if valid_users and valid_items else 1.0

    full_stats.update({
        "n_users_kcore": len(valid_users),
        "n_items_kcore": len(valid_items),
        "n_interactions_kcore": n_inter,
        "sparsity_kcore": sparsity,
    })

    print(f"\n--- {k}-core Statistics ---")
    print(f"Users:        {len(valid_users):,}")
    print(f"Items:        {len(valid_items):,}")
    print(f"Interactions: {n_inter:,}")
    print(f"Sparsity:     {sparsity:.6f}")

    return valid_users, valid_items, full_stats


def load_filtered_data(raw_path: str, valid_users: set, valid_items: set) -> pd.DataFrame:
    """Pass 2: re-read file, only keeping rows in the k-core."""
    records = []

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Pass 2 (loading k-core)", total=50_264_531):
            r = json.loads(line.strip())
            rating = float(r.get("rating", 0))
            user = r.get("author", "")
            item = r.get("hotel_url", "")

            if rating <= 0 or not user or not item:
                continue
            if user not in valid_users or item not in valid_items:
                continue

            prop = r.get("property_dict", {})
            records.append({
                "user_url": user,
                "hotel_url": item,
                "rating": rating,
                "text": r.get("text", ""),
                "date": r.get("date", ""),
                "title": r.get("title", ""),
                "service": prop.get("service"),
                "cleanliness": prop.get("cleanliness"),
                "location": prop.get("location"),
                "value": prop.get("value"),
                "rooms": prop.get("rooms"),
                "sleep_quality": prop.get("sleep quality"),
                "check_in": prop.get("check in", prop.get("check-in")),
                "business_service": prop.get("business service"),
            })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df):,} k-core reviews")
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


def main():
    parser = argparse.ArgumentParser(description="Preprocess HotelRec data")
    parser.add_argument("--kcore", type=int, default=None)
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--raw-file", type=str, default=None,
                        help="Path to HotelRec.txt (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    k = args.kcore or cfg["kcore"]["default"]
    raw_path = args.raw_file or cfg["dataset"]["raw_file"]
    out_dir = cfg["dataset"]["processed_dir"]

    print(f"Config: {args.config}")
    print(f"k-core: {k}")
    print(f"Raw file: {raw_path}")

    valid_users, valid_items, stats = compute_kcore_sets(raw_path, k)

    stats_path = os.path.join(out_dir, "full_stats.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved full-dataset stats to {stats_path}")

    df = load_filtered_data(raw_path, valid_users, valid_items)
    user2id, item2id, df = build_id_maps(df)

    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    n_inter = len(df)
    sparsity = 1.0 - n_inter / (n_users * n_items)
    print(f"\n--- Final {k}-core Statistics ---")
    print(f"Users:        {n_users:,}")
    print(f"Items:        {n_items:,}")
    print(f"Interactions: {n_inter:,}")
    print(f"Sparsity:     {sparsity:.6f}")

    sub_ratings = ["service", "cleanliness", "location", "value",
                   "rooms", "sleep_quality", "check_in", "business_service"]
    keep_cols = ["user_id", "item_id", "rating", "text", "date", "title"] + sub_ratings
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

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
