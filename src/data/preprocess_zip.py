"""
Preprocess HotelRec data directly from the zip archive.

Streams reviews from the ~50GB zip without extracting to disk.
The zip contains ~365K per-hotel JSON files, each with an array of reviews.
We flatten them into a single stream, apply k-core filtering, and save
the same parquet files that the rest of the pipeline expects.

This replaces the two-step preprocess.py + split.py workflow when you
want to avoid creating a 50GB intermediate HotelRec.txt file.

Usage:
    python -m src.data.preprocess_zip --kcore 20
    python -m src.data.preprocess_zip --kcore 20 --max-reviews 500000  # sample
"""

import argparse
import json
import os
import zipfile
from collections import Counter

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _iter_reviews_from_zip(zip_path: str, max_reviews: int = 0):
    """Yield one review dict at a time from the HotelRec zip.

    Each entry in the zip is a per-hotel JSON file containing an array
    of review objects. We open each file inside the zip, parse the JSON
    array, and yield individual reviews.

    Args:
        zip_path: path to HotelRec.zip
        max_reviews: stop after this many reviews (0 = no limit)
    """
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        for name in names:
            with zf.open(name) as f:
                try:
                    data = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                reviews = data if isinstance(data, list) else [data]
                for review in reviews:
                    yield review
                    count += 1
                    if max_reviews > 0 and count >= max_reviews:
                        return


def compute_kcore_sets(zip_path: str, k: int, max_reviews: int = 0):
    """Pass 1: stream reviews from zip, compute k-core membership.

    Returns (valid_users, valid_items, stats_dict).
    """
    print(f"Pass 1: counting user/item frequencies from zip...")
    user_counts = Counter()
    item_counts = Counter()
    total = 0
    n_valid = 0

    for r in tqdm(_iter_reviews_from_zip(zip_path, max_reviews),
                  desc="Pass 1 (counting)"):
        total += 1
        rating = float(r.get("rating", 0))
        user = r.get("author", "")
        item = r.get("hotel_url", "")
        if rating > 0 and user and item:
            user_counts[user] += 1
            item_counts[item] += 1
            n_valid += 1

    print(f"  Total: {total:,}, valid: {n_valid:,}")
    print(f"  Users: {len(user_counts):,}, Items: {len(item_counts):,}")

    # build bipartite graph for k-core
    print(f"\nComputing {k}-core membership...")
    user_items = {}
    item_users = {}

    for r in tqdm(_iter_reviews_from_zip(zip_path, max_reviews),
                  desc="Pass 1b (graph)"):
        rating = float(r.get("rating", 0))
        user = r.get("author", "")
        item = r.get("hotel_url", "")
        if rating > 0 and user and item:
            user_items.setdefault(user, set()).add(item)
            item_users.setdefault(item, set()).add(user)

    # iterative k-core pruning
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

        n_u = len(user_items)
        n_i = len(item_users)
        n_e = sum(len(v) for v in user_items.values())
        print(f"  iter {iteration}: {n_u:,} users, {n_i:,} items, {n_e:,} edges")

    valid_users = set(user_items.keys())
    valid_items = set(item_users.keys())

    stats = {
        "total_raw": total,
        "total_valid": n_valid,
        "n_users_full": len(user_counts),
        "n_items_full": len(item_counts),
        "kcore": k,
        "n_users_kcore": len(valid_users),
        "n_items_kcore": len(valid_items),
    }

    print(f"\n{k}-core: {len(valid_users):,} users, {len(valid_items):,} items")
    return valid_users, valid_items, stats


def load_filtered_data(zip_path: str, valid_users: set, valid_items: set,
                       max_reviews: int = 0) -> pd.DataFrame:
    """Pass 2: stream from zip again, keep only k-core rows."""
    records = []

    for r in tqdm(_iter_reviews_from_zip(zip_path, max_reviews),
                  desc="Pass 2 (loading k-core)"):
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


def build_id_maps(df: pd.DataFrame):
    """Map user_url and hotel_url to contiguous integer IDs."""
    users = sorted(df["user_url"].unique())
    items = sorted(df["hotel_url"].unique())

    user2id = {u: i for i, u in enumerate(users)}
    item2id = {it: i for i, it in enumerate(items)}

    df = df.copy()
    df["user_id"] = df["user_url"].map(user2id)
    df["item_id"] = df["hotel_url"].map(item2id)

    return user2id, item2id, df


def split_data(df: pd.DataFrame, cfg: dict):
    """Random 80/10/10 split with fixed seed."""
    seed = cfg["split"]["seed"]
    train_ratio = cfg["split"]["train"]
    val_ratio = cfg["split"]["val"]
    test_ratio = cfg["split"]["test"]

    holdout_ratio = val_ratio + test_ratio
    train_df, holdout_df = train_test_split(
        df, test_size=holdout_ratio, random_state=seed)

    val_relative = val_ratio / holdout_ratio
    val_df, test_df = train_test_split(
        holdout_df, test_size=(1 - val_relative), random_state=seed)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess HotelRec directly from zip (no extraction)")
    parser.add_argument("--kcore", type=int, default=None)
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--zip-file", type=str, default=None,
                        help="Path to HotelRec.zip (auto-detected in data/raw/)")
    parser.add_argument("--max-reviews", type=int, default=0,
                        help="Limit reviews for testing (0 = all)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    k = args.kcore or cfg["kcore"]["default"]
    out_dir = cfg["dataset"]["processed_dir"]

    # find zip file
    zip_path = args.zip_file
    if not zip_path:
        raw_dir = cfg["dataset"].get("raw_dir", "data/raw")
        candidates = [f for f in os.listdir(raw_dir)
                      if f.endswith(".zip")] if os.path.isdir(raw_dir) else []
        if candidates:
            zip_path = os.path.join(raw_dir, candidates[0])
        else:
            # fall back to checking for HotelRec.txt (already extracted)
            txt_path = cfg["dataset"].get("raw_file", "data/raw/HotelRec.txt")
            if os.path.exists(txt_path):
                print(f"No zip found, but {txt_path} exists.")
                print(f"Use the original preprocess.py instead:")
                print(f"  python -m src.data.preprocess --kcore {k}")
                return
            print(f"ERROR: No zip file found in {raw_dir}/")
            print(f"Download first: bash scripts/download_data.sh full")
            return

    print(f"Config: {args.config}")
    print(f"k-core: {k}")
    print(f"Zip: {zip_path}")
    if args.max_reviews > 0:
        print(f"Max reviews: {args.max_reviews:,} (sample mode)")

    # check if already processed
    kcore_dir = os.path.join(out_dir, f"{k}core")
    train_path = os.path.join(kcore_dir, "train.parquet")
    if os.path.exists(train_path):
        n = len(pd.read_parquet(train_path))
        if n > 0:
            print(f"\n[OK] Already processed ({n:,} training rows at {kcore_dir}/)")
            print(f"Delete {kcore_dir}/ to re-process.")
            return

    # pass 1: k-core
    valid_users, valid_items, stats = compute_kcore_sets(
        zip_path, k, args.max_reviews)

    os.makedirs(out_dir, exist_ok=True)
    stats_path = os.path.join(out_dir, "full_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # pass 2: load filtered data
    df = load_filtered_data(zip_path, valid_users, valid_items, args.max_reviews)
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

    # select columns
    sub_ratings = ["service", "cleanliness", "location", "value",
                   "rooms", "sleep_quality", "check_in", "business_service"]
    keep_cols = ["user_id", "item_id", "rating", "text", "date", "title"] + sub_ratings
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # save interactions
    os.makedirs(kcore_dir, exist_ok=True)
    parquet_path = os.path.join(kcore_dir, "interactions.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved interactions to {parquet_path}")

    # save ID mappings
    with open(os.path.join(kcore_dir, "user2id.json"), "w") as f:
        json.dump(user2id, f)
    with open(os.path.join(kcore_dir, "item2id.json"), "w") as f:
        json.dump(item2id, f)

    # split into train/val/test
    print("\nSplitting into train/val/test...")
    train_df, val_df, test_df = split_data(df, cfg)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = os.path.join(kcore_dir, f"{name}.parquet")
        split_df.to_parquet(out_path, index=False)
        print(f"  {name}: {len(split_df):,} interactions -> {out_path}")

    print(f"\nDone. Output: {kcore_dir}/")
    print(f"No HotelRec.txt needed — read directly from {zip_path}")


if __name__ == "__main__":
    main()
