"""
Preprocess raw HotelRec data into a clean parquet dataset.

The HotelRec dataset (Antognini & Faltings, LREC 2020) is distributed as a
compressed archive (~10GB) containing ~365K JSON files (one per hotel), totaling
~50GB uncompressed. We stream directly from the archive to avoid extracting
everything to disk.

Each review has:
  - hotel_url, author, date, rating, title, text, property_dict (sub-ratings)

Output: parquet with columns [user_id, item_id, rating, text, date],
        plus JSON files for user/item ID mappings.

Usage:
    python -m src.data.preprocess --kcore 20 --config configs/data.yaml
"""

import argparse
import io
import json
import os
import tarfile
import zipfile
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


def stream_from_tar(archive_path: str):
    """Stream reviews from a tar.gz or tar.bz2 archive without full extraction."""
    print(f"  Streaming from tar archive: {archive_path}")
    n_files = 0
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if not (member.name.endswith(".json") or member.name.endswith(".txt")):
                continue
            n_files += 1
            f = tar.extractfile(member)
            if f is None:
                continue
            try:
                if member.size > 100_000_000:  # > 100MB → stream line-by-line (JSONL)
                    print(f"  Reading {member.name} ({member.size / 1e9:.1f} GB uncompressed)")
                    text_stream = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                    for line in text_stream:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield parse_review(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                else:
                    content = f.read().decode("utf-8", errors="replace")
                    data = json.loads(content)
                    if isinstance(data, list):
                        for r in data:
                            yield parse_review(r)
                    elif isinstance(data, dict):
                        yield parse_review(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            finally:
                f.close()
            if n_files % 10000 == 0:
                print(f"    {n_files} hotel files processed...")
    print(f"  Done: {n_files} hotel files from archive")


def stream_from_zip(archive_path: str):
    """Stream reviews from a zip archive. Handles both multi-file and single-file layouts."""
    print(f"  Streaming from zip: {archive_path}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            # accept .json and .txt files
            if not (name.endswith(".json") or name.endswith(".txt")):
                continue

            print(f"  Reading {name} ({info.file_size / 1e9:.1f} GB uncompressed)")

            if info.file_size > 100_000_000:  # > 100MB → stream line-by-line (JSONL)
                with zf.open(name) as f:
                    text_stream = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                    n = 0
                    for line in text_stream:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield parse_review(json.loads(line))
                            n += 1
                            if n % 1_000_000 == 0:
                                print(f"    {n:,} reviews streamed...")
                        except json.JSONDecodeError:
                            continue
                    print(f"  Done: {n:,} reviews from {name}")
                continue

            # Small file — load directly
            try:
                content = zf.read(name).decode("utf-8", errors="replace")
                data = json.loads(content)
                if isinstance(data, list):
                    for r in data:
                        yield parse_review(r)
                elif isinstance(data, dict):
                    yield parse_review(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue


def stream_from_json(fpath: str):
    """Stream reviews from a single JSON or JSONL file."""
    fsize = os.path.getsize(fpath)

    if fsize > 100_000_000:  # > 100MB → stream line-by-line (JSONL)
        print(f"  Streaming {fpath} line-by-line ({fsize / 1e9:.1f} GB)")
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield parse_review(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return

    # Small file — try JSON array first, fall back to JSONL
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    content_stripped = content.lstrip()
    if content_stripped.startswith("["):
        try:
            data = json.loads(content)
            for r in (data if isinstance(data, list) else [data]):
                yield parse_review(r)
            return
        except json.JSONDecodeError:
            pass

    # JSONL fallback
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield parse_review(json.loads(line))
        except json.JSONDecodeError:
            continue


def find_data_source(raw_dir: str):
    """
    Find the data source in raw_dir. Could be:
    1. An archive file (.tar.gz, .tar.bz2, .zip)
    2. A single large JSON file
    3. Multiple JSON files (one per hotel)
    Returns (source_type, path_or_paths)
    """
    raw_path = Path(raw_dir)

    # check for archives first
    for ext in ["*.tar.gz", "*.tar.bz2", "*.tgz", "*.zip"]:
        archives = list(raw_path.glob(ext))
        if archives:
            return ("archive", archives[0])

    # check for JSON / TXT files
    data_extensions = {".json", ".txt"}
    data_files = sorted(
        p for p in raw_path.iterdir()
        if p.is_file() and p.suffix.lower() in data_extensions
    )
    if data_files:
        if len(data_files) == 1:
            return ("single_json", data_files[0])
        return ("multi_json", data_files)

    # check subdirectories (archive might extract into a subfolder)
    for subdir in raw_path.iterdir():
        if subdir.is_dir():
            sub_files = sorted(
                p for p in subdir.iterdir()
                if p.is_file() and p.suffix.lower() in data_extensions
            )
            if sub_files:
                return ("multi_json", sub_files)

    raise FileNotFoundError(
        f"No data found in {raw_dir}. Expected .tar.gz/.zip archive or .json/.txt files.\n"
        f"Download with: bash scripts/download_data.sh full"
    )


def load_raw_data(raw_dir: str, chunk_size: int = 500_000) -> pd.DataFrame:
    """Load HotelRec data, streaming from archive or JSON files."""
    source_type, source = find_data_source(raw_dir)
    print(f"Data source: {source_type} → {source}")

    all_chunks = []
    total = 0
    chunk_buf = []

    if source_type == "archive":
        archive_path = str(source)
        if archive_path.endswith(".zip"):
            gen = stream_from_zip(archive_path)
        else:
            gen = stream_from_tar(archive_path)
    elif source_type == "single_json":
        gen = stream_from_json(str(source))
    else:
        # multi_json: iterate over files
        def multi_gen():
            for fpath in source:
                yield from stream_from_json(str(fpath))
        gen = multi_gen()

    for rec in gen:
        chunk_buf.append(rec)
        if len(chunk_buf) >= chunk_size:
            all_chunks.append(pd.DataFrame(chunk_buf))
            total += len(chunk_buf)
            print(f"  {total:,} reviews loaded...")
            chunk_buf = []

    if chunk_buf:
        all_chunks.append(pd.DataFrame(chunk_buf))
        total += len(chunk_buf)

    df = pd.concat(all_chunks, ignore_index=True)
    print(f"Loaded {len(df):,} total reviews")
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
    """Iteratively remove users/items with fewer than k interactions."""
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

    # load (streaming from archive or JSON)
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
