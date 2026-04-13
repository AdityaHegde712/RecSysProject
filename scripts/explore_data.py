#!/usr/bin/env python3
"""
Explore the raw HotelRec dataset and print statistics for the Day 6 checkpoint.

Reads JSONL files from the raw data directory, computes counts, distributions,
and coverage stats, then prints a formatted summary you can paste straight
into the checkpoint document.

Usage:
    python scripts/explore_data.py --data_dir data/raw --sample_size 0

    --sample_size 0   process ALL files (default)
    --sample_size 50  only process 50 files (quick sanity check)
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

SUB_RATING_KEYS = [
    "service", "cleanliness", "location", "value",
    "rooms", "sleep quality", "check-in", "business service",
]

# Display names for pretty printing
SUB_RATING_DISPLAY = {
    "service": "Service",
    "cleanliness": "Cleanliness",
    "location": "Location",
    "value": "Value",
    "rooms": "Rooms",
    "sleep quality": "Sleep Quality",
    "check-in": "Check-In",
    "business service": "Business Service",
}


def parse_jsonl_file(fpath: str) -> list[dict]:
    """Parse a single JSONL file (one JSON object per line)."""
    records = []
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue
    return records


def parse_json_file(fpath: str) -> list[dict]:
    """Parse a single JSON file (array of review objects)."""
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]
        except json.JSONDecodeError:
            return []


def load_file(fpath: str) -> list[dict]:
    """
    Try JSONL first (one object per line), fall back to plain JSON array.
    The HotelRec dataset can come in either format depending on how it was
    downloaded / extracted.
    """
    # Peek at the first non-empty line to decide format
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        first_line = ""
        for raw in f:
            first_line = raw.strip()
            if first_line:
                break

    if not first_line:
        return []

    # If the file starts with '[', it's probably a JSON array
    if first_line.startswith("["):
        return parse_json_file(fpath)
    else:
        return parse_jsonl_file(fpath)


# ── main logic ───────────────────────────────────────────────────────────────

def explore(data_dir: str, sample_size: int = 0):
    data_path = Path(data_dir)

    # Find all data files (json, jsonl, txt — HotelRec uses .txt sometimes)
    extensions = {".json", ".jsonl", ".txt"}
    all_files = sorted(
        p for p in data_path.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not all_files:
        print(f"ERROR: No JSON/JSONL/TXT files found in {data_dir}")
        print("Make sure you've downloaded and extracted the HotelRec dataset.")
        sys.exit(1)

    total_files = len(all_files)
    if sample_size > 0:
        all_files = all_files[:sample_size]
    n_files = len(all_files)

    print(f"Found {total_files} data files in {data_dir}")
    if sample_size > 0:
        print(f"Processing first {n_files} files (--sample_size {sample_size})")
    else:
        print(f"Processing all {n_files} files")
    print()

    # Accumulators
    total_reviews = 0
    users = set()
    items = set()
    ratings = []
    review_lengths = []          # word count per review
    dates = []                   # raw date strings
    user_review_counts = Counter()
    item_review_counts = Counter()
    sub_rating_counts = {k: 0 for k in SUB_RATING_KEYS}
    sub_rating_values = {k: [] for k in SUB_RATING_KEYS}

    for idx, fpath in enumerate(all_files):
        if (idx + 1) % 500 == 0 or idx == 0:
            print(f"  [{idx + 1}/{n_files}] Processing {fpath.name}...")

        records = load_file(str(fpath))

        for rec in records:
            total_reviews += 1

            # User / item identifiers
            user = rec.get("author") or rec.get("user_url") or rec.get("user_id", "")
            item = rec.get("hotel_url") or rec.get("item_id", "")
            if user:
                users.add(user)
                user_review_counts[user] += 1
            if item:
                items.add(item)
                item_review_counts[item] += 1

            # Overall rating
            rating = rec.get("rating") or rec.get("overall_rating")
            if rating is not None:
                try:
                    ratings.append(float(rating))
                except (ValueError, TypeError):
                    pass

            # Review text length
            text = rec.get("text", "")
            if text:
                word_count = len(text.split())
                review_lengths.append(word_count)

            # Date
            date_val = rec.get("date", "")
            if date_val:
                dates.append(str(date_val))

            # Sub-ratings — could be in property_dict or top-level
            prop = rec.get("property_dict", {}) or {}
            for key in SUB_RATING_KEYS:
                # Check property_dict first, then top-level (with underscore variant)
                val = prop.get(key) or prop.get(key.replace("-", " "))
                if val is None:
                    val = rec.get(key) or rec.get(key.replace(" ", "_"))
                if val is not None:
                    sub_rating_counts[key] += 1
                    try:
                        sub_rating_values[key].append(float(val))
                    except (ValueError, TypeError):
                        pass

    # ── compute stats ────────────────────────────────────────────────────

    n_users = len(users)
    n_items = len(items)
    n_interactions = total_reviews

    if n_users > 0 and n_items > 0:
        density = n_interactions / (n_users * n_items) * 100
        sparsity = 100.0 - density
    else:
        density = 0.0
        sparsity = 100.0

    avg_per_user = n_interactions / n_users if n_users else 0
    avg_per_item = n_interactions / n_items if n_items else 0

    ratings_arr = np.array(ratings) if ratings else np.array([0.0])
    lengths_arr = np.array(review_lengths) if review_lengths else np.array([0])

    # User activity distribution
    user_counts_list = list(user_review_counts.values())
    user_counts_arr = np.array(user_counts_list) if user_counts_list else np.array([0])
    users_1_review = sum(1 for c in user_counts_list if c == 1)
    users_lt5 = sum(1 for c in user_counts_list if c < 5)

    # Item popularity distribution
    item_counts_list = list(item_review_counts.values())
    item_counts_arr = np.array(item_counts_list) if item_counts_list else np.array([0])

    # Temporal distribution
    year_counts = Counter()
    for d in dates:
        # Try to extract year from various date formats
        d_str = str(d).strip()
        for fmt_try in [d_str[:4], d_str[-4:]]:
            try:
                year = int(fmt_try)
                if 1990 <= year <= 2030:
                    year_counts[year] += 1
                    break
            except ValueError:
                continue

    # ── print results ────────────────────────────────────────────────────

    sep = "=" * 70
    print(f"\n{sep}")
    print("  HOTELREC DATASET — EXPLORATION SUMMARY")
    if sample_size > 0:
        print(f"  (based on {n_files}/{total_files} files)")
    print(sep)

    print(f"\n{'─' * 40}")
    print("  1. BASIC STATISTICS")
    print(f"{'─' * 40}")
    print(f"  Files processed:       {n_files:>15,}")
    print(f"  Total reviews:         {n_interactions:>15,}")
    print(f"  Unique users:          {n_users:>15,}")
    print(f"  Unique items (hotels): {n_items:>15,}")
    print(f"  Density:               {density:>14.5f}%")
    print(f"  Sparsity:              {sparsity:>14.5f}%")
    print(f"  Avg reviews/user:      {avg_per_user:>15.2f}")
    print(f"  Median reviews/user:   {np.median(user_counts_arr):>15.1f}")
    print(f"  Avg reviews/item:      {avg_per_item:>15.2f}")
    print(f"  Median reviews/item:   {np.median(item_counts_arr):>15.1f}")

    print(f"\n{'─' * 40}")
    print("  2. RATING DISTRIBUTION")
    print(f"{'─' * 40}")
    print(f"  Mean rating:           {np.mean(ratings_arr):>15.2f}")
    print(f"  Median rating:         {np.median(ratings_arr):>15.1f}")
    print(f"  Std rating:            {np.std(ratings_arr):>15.2f}")
    print(f"  Min rating:            {np.min(ratings_arr):>15.1f}")
    print(f"  Max rating:            {np.max(ratings_arr):>15.1f}")
    print()
    print("  Rating histogram:")
    if len(ratings) > 0:
        for star in [1, 2, 3, 4, 5]:
            count = sum(1 for r in ratings if round(r) == star)
            pct = count / len(ratings) * 100
            bar = "█" * int(pct / 2)
            print(f"    {star}★: {count:>10,} ({pct:5.1f}%) {bar}")

    print(f"\n{'─' * 40}")
    print("  3. USER ACTIVITY DISTRIBUTION")
    print(f"{'─' * 40}")
    print(f"  Users with exactly 1 review:  {users_1_review:>10,} ({users_1_review / n_users * 100:.2f}%)" if n_users else "  N/A")
    print(f"  Users with < 5 reviews:       {users_lt5:>10,} ({users_lt5 / n_users * 100:.2f}%)" if n_users else "  N/A")
    print(f"  Max reviews by one user:      {np.max(user_counts_arr):>10,}")
    print()
    print("  Activity buckets:")
    buckets = [(1, 1), (2, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, None)]
    for lo, hi in buckets:
        if hi is None:
            count = sum(1 for c in user_counts_list if c >= lo)
            label = f"  {lo}+"
        else:
            count = sum(1 for c in user_counts_list if lo <= c <= hi)
            label = f"  {lo}-{hi}"
        pct = count / n_users * 100 if n_users else 0
        print(f"    {label:>8s} reviews: {count:>10,} users ({pct:5.1f}%)")

    print(f"\n{'─' * 40}")
    print("  4. SUB-RATING COVERAGE")
    print(f"{'─' * 40}")
    for key in SUB_RATING_KEYS:
        cnt = sub_rating_counts[key]
        pct = cnt / n_interactions * 100 if n_interactions else 0
        vals = sub_rating_values[key]
        avg_str = f"avg={np.mean(vals):.2f}" if vals else "avg=N/A"
        display = SUB_RATING_DISPLAY[key]
        print(f"    {display:<20s}: {cnt:>10,} ({pct:5.2f}%)  {avg_str}")

    print(f"\n{'─' * 40}")
    print("  5. REVIEW TEXT LENGTH (words)")
    print(f"{'─' * 40}")
    if len(review_lengths) > 0:
        print(f"  Reviews with text:     {len(review_lengths):>15,} ({len(review_lengths) / n_interactions * 100:.1f}%)")
        print(f"  Mean length:           {np.mean(lengths_arr):>15.2f}")
        print(f"  Median length:         {np.median(lengths_arr):>15.1f}")
        print(f"  Std length:            {np.std(lengths_arr):>15.2f}")
        print(f"  Min length:            {np.min(lengths_arr):>15}")
        print(f"  Max length:            {np.max(lengths_arr):>15}")
        print()
        print("  Length buckets:")
        len_buckets = [(1, 10), (11, 50), (51, 100), (101, 200), (201, 500), (501, None)]
        for lo, hi in len_buckets:
            if hi is None:
                count = sum(1 for l in review_lengths if l >= lo)
                label = f"{lo}+"
            else:
                count = sum(1 for l in review_lengths if lo <= l <= hi)
                label = f"{lo}-{hi}"
            pct = count / len(review_lengths) * 100
            print(f"    {label:>8s} words: {count:>10,} ({pct:5.1f}%)")
    else:
        print("  No review text found.")

    print(f"\n{'─' * 40}")
    print("  6. TEMPORAL DISTRIBUTION")
    print(f"{'─' * 40}")
    if year_counts:
        for year in sorted(year_counts.keys()):
            cnt = year_counts[year]
            pct = cnt / n_interactions * 100
            bar = "█" * max(1, int(pct))
            print(f"    {year}: {cnt:>10,} ({pct:5.1f}%) {bar}")
    else:
        print("  Could not parse dates.")

    # ── markdown table for easy copy-paste ───────────────────────────────

    print(f"\n{sep}")
    print("  COPY-PASTE TABLE FOR CHECKPOINT (Section 1a)")
    print(sep)
    print()
    print("| Stat | Value |")
    print("|------|-------|")
    print(f"| Number of users | {n_users:,} |")
    print(f"| Number of items | {n_items:,} |")
    print(f"| Number of interactions | {n_interactions:,} |")
    print(f"| Density | {density:.5f}% |")
    print(f"| Sparsity | {sparsity:.5f}% |")
    print(f"| Avg interactions per user | {avg_per_user:.2f} |")
    print(f"| Median interactions per user | {np.median(user_counts_arr):.1f} |")
    print(f"| Avg interactions per item | {avg_per_item:.2f} |")
    print(f"| Median interactions per item | {np.median(item_counts_arr):.1f} |")
    print(f"| Mean rating | {np.mean(ratings_arr):.2f} ± {np.std(ratings_arr):.2f} |")
    print(f"| Median rating | {np.median(ratings_arr):.1f} |")
    print(f"| Mean review length (words) | {np.mean(lengths_arr):.2f} |")

    sub_parts = []
    for key in SUB_RATING_KEYS:
        pct = sub_rating_counts[key] / n_interactions * 100 if n_interactions else 0
        sub_parts.append(f"{SUB_RATING_DISPLAY[key]} ({pct:.1f}%)")
    print(f"| Sub-rating coverage | {', '.join(sub_parts)} |")
    print()

    print(f"{sep}")
    print("  Done.")
    print(sep)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Explore the raw HotelRec dataset and print statistics "
                    "for the Day 6 checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files
  python scripts/explore_data.py --data_dir data/raw --sample_size 0

  # Quick test with 10 files
  python scripts/explore_data.py --data_dir data/raw --sample_size 10
        """,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Path to directory containing raw HotelRec JSON/JSONL files "
             "(default: data/raw)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Number of files to process. 0 = all files (default: 0)",
    )
    args = parser.parse_args()

    explore(args.data_dir, args.sample_size)


if __name__ == "__main__":
    main()
